# BiSeNet face-parsing checkpoint (ResNet-34)

`bisenet_resnet34.pt` is a 19-class face-parsing segmentation network trained on
**CelebAMask-HQ**. It labels every pixel of a portrait as skin, hair, neck,
clothing, background, or one of the individual facial features.

- Upstream: <https://github.com/yakhyo/face-parsing>
- License: MIT, Copyright (c) 2024 Yakhyokhuja Valikhujaev
- Size: 95,869,966 bytes
- Architecture code: [src/lorakit/face_parsing_net.py](../../src/lorakit/face_parsing_net.py)

## Purpose in lorakit

The parse map is the highest-fidelity mask source available to training. Unlike a
face box or a background-removal silhouette, it separates **clothing** from the
subject, which lets training

1. exclude clothing entirely so the LoRA never memorizes the outfit,
2. down-weight face **skin** relative to hair/neck/features, and
3. use zoom-out augmentation, because padded canvas can be given zero weight.

See [docs/face-parsing.md](../../docs/face-parsing.md).

## Class indices

Index 0 is background; 1-18 follow the CelebAMask-HQ attribute order.

| Index | Name | Meaning | lorakit group |
|---|---|---|---|
| 0 | - | background | `background` |
| 1 | `skin` | face skin | `skin` |
| 2 | `l_brow` | left eyebrow | `features` |
| 3 | `r_brow` | right eyebrow | `features` |
| 4 | `l_eye` | left eye | `features` |
| 5 | `r_eye` | right eye | `features` |
| 6 | `eye_g` | eyeglasses | `accessories` |
| 7 | `l_ear` | left ear | `features` |
| 8 | `r_ear` | right ear | `features` |
| 9 | `ear_r` | earring | `accessories` |
| 10 | `nose` | nose | `features` |
| 11 | `mouth` | mouth interior | `features` |
| 12 | `u_lip` | upper lip | `features` |
| 13 | `l_lip` | lower lip | `features` |
| 14 | `neck` | neck | `neck` |
| 15 | `neck_l` | necklace | `accessories` |
| 16 | `cloth` | clothing | `cloth` |
| 17 | `hair` | hair | `hair` |
| 18 | `hat` | hat | `accessories` |

## Input / output contract

| Item | Specification |
|---|---|
| Input | `float32` tensor `[B, 3, 512, 512]`, RGB |
| Preprocessing | resize to 512x512 bilinear, scale to `[0, 1]`, ImageNet normalization (mean `0.485/0.456/0.406`, std `0.229/0.224/0.225`) |
| Output | 3-tuple of logits, each `[B, 19, 512, 512]` (main head plus two auxiliary heads) |
| Use | `logits[0].argmax(dim=1)` -> `[B, 512, 512]` class indices |

The auxiliary heads exist only for deep supervision during training; inference
uses the first output.

## Checkpoint format

A plain `OrderedDict` state dict with 289 keys, not a wrapped checkpoint:

```python
from lorakit.face_parsing_net import load_bisenet

model = load_bisenet(
    "models/face-parsing-bisenet-resnet34/bisenet_resnet34.pt",
    backbone_name="resnet34",
    device="cuda",
)
```

Keys are prefixed by module path: `fpn.backbone.*` (the ResNet-34 trunk, including
an unused `fc`), `fpn.arm16/arm32/conv_head*/conv_avg.*`, `ffm.*`, and the three
`conv_out*` heads. The names match torchvision's ResNet exactly, so the backbone
is constructed with `weights=None` and never downloads ImageNet weights.

## Limits

- Trained on near-frontal, roughly aligned portraits; profile or heavily occluded
  faces degrade, most often by leaking hair into background.
- `neck` and `cloth` share a soft boundary at high collars; expect a few pixels of
  bleed either way.
- Sunglasses and thick frames are labeled `eye_g`, which occludes `l_eye`/`r_eye`
  rather than being layered over them.
- Segmentation only - it carries no identity information.

## Dependencies

- Python 3.12+
- PyTorch 2.2+
- torchvision 0.20+

No extra packages are required, unlike the BiRefNet and InsightFace mask sources.
