# BiRefNet high-resolution matting checkpoint

A local snapshot of [`ZhengPeng7/BiRefNet_HR-matting`](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting):
a background-removal / matting network that produces one soft alpha per image
separating the subject from the scene. Unlike the stock BiRefNet checkpoint it is
trained at `2048x2048` on matting data (P3M-10k, TR-humans, AM-2k and related
sets) rather than general salient-object data.

- Upstream: <https://github.com/ZhengPeng7/BiRefNet>
- License: MIT
- Size: 444,473,596 bytes (`model.safetensors`)
- Architecture code: shipped by the repo itself as `birefnet.py` /
  `BiRefNet_config.py` and loaded via `trust_remote_code=True`, so lorakit
  carries no BiRefNet model code of its own.
- Loader: [src/lorakit/subject_masks.py](../../src/lorakit/subject_masks.py)

## Purpose in lorakit

The alpha owns the **person / scene split**. Training anchors the background to
the frozen base model using `1 - alpha`, so any pixel the alpha claims can never
be anchored, and with `face_parse.background_source: subject_mask` the parse map
only grades weights *inside* whatever the alpha called the subject.

That makes an over-eager alpha expensive: scenery it wrongly claims is excluded
from the anchor, and if the parser then gives it a zero training weight the
region is neither trained nor anchored.

## Why not the stock `ZhengPeng7/BiRefNet`

The general checkpoint solves salient-object detection, and on indoor portraits
it tends to treat furniture the subject leans on as part of the salient object —
often with a hard alpha (confident, not merely blurry). Those scene pixels can
therefore never be anchored.

This HR-matting checkpoint is tighter on those failure cases; images that
already had clean alphas barely move. The `-portrait` and 1024 `-matting`
variants both help less than the 2048 matting weights here.

## Input / output contract

| Item | Specification |
|---|---|
| Input | `float32` tensor `[B, 3, 2048, 2048]`, RGB |
| Preprocessing | resize to 2048x2048 bilinear, scale to `[0, 1]`, ImageNet normalization (mean `0.485/0.456/0.406`, std `0.229/0.224/0.225`) |
| Output | list of progressively refined logit maps; the **last** is the final one |
| Use | `sigmoid(logits[-1])` -> alpha in `[0, 1]`, resized back to the source size |

`input_size` must be 2048 to get the benefit — feeding this checkpoint 1024 gives
up most of its advantage over the 1024 `-matting` variant.

```yaml
subject_mask:
  model_id: models/birefnet-hr-matting
  input_size: 2048
```

`config.json` sets `bb_pretrained: false`, so the swin backbone is built empty
and never downloads ImageNet weights.

## Limits

- Runs in fp32: the swin backbone is numerically fragile in fp16.
- 2048x2048 is 4x the pixels of the default 1024 path, so mask building is
  correspondingly slower and needs more VRAM. It runs once and is cached in
  `masks.json`.
- Not bit-reproducible. Repeated runs on the same GPU differ by up to ~8/255 at
  soft edges from cuDNN algorithm selection, which is why masks are generated
  once and committed to the dataset folder rather than rebuilt per run.
- Still a single foreground/background alpha: it carries no notion of clothing,
  hair or identity. The parse map supplies those.
