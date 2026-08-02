# lorakit

**Fine-tune Stable Diffusion XL (SDXL) with LoRA from a single YAML file.**

lorakit is a small, fast, config-driven toolkit for SDXL DreamBooth / LoRA training. Point it at a folder of images, edit one YAML file, and run a single command. It's built on the DreamBooth training code from Hugging Face's [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) and takes its configuration style from [ai-toolkit](https://github.com/ostris/ai-toolkit).

> Used in production by [FaceHarmony.ai](https://faceharmony.ai).

---

## Highlights

- **One command, one config file** — no code required.
- **Fast** — latent caching, optional `torch.compile`, and bf16 training reach **~5.9x faster steps** than a naive 4-bit setup on a 24 GB RTX 4090 (see [Performance](#performance)).
- **Fits your GPU** — train on 24 GB in bf16, or drop to **4-bit QLoRA** to fit 16 GB cards.
- **Live previews** — generates sample images during training so you can watch progress.
- **Resumable** — checkpoints let you stop and continue.
- Flexible optimizers (AdamW, AdamW8bit, AdamWScheduleFree, AnchoredAdamW), LR schedulers, and LoRA targets for both the UNet and text encoders.
- Optional latent face background preservation, which keeps LoRA changes focused on the subject rather than memorizing the training scene — including a 19-class [face parser](docs/face-parsing.md) that keeps clothing out of the LoRA entirely.

---

## Requirements

- An NVIDIA GPU (16 GB+ recommended; 24 GB for the fastest bf16 path).
- [uv](https://docs.astral.sh/uv/) for dependency management.
- PyTorch wheels are resolved from the [CUDA 13.2 index](https://download.pytorch.org/whl/cu132) automatically.

## Installation

```bash
git clone https://github.com/TensorHarmony/lorakit.git
cd lorakit
uv sync
```

That's it — `bitsandbytes` (quantization) and Triton (for `torch.compile`) are installed by default, so quantization and compilation work out of the box.

Run everything with `uv run lorakit ...`, or activate the venv first:

```bash
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\activate
```

---

## Quickstart

**1. Gather your images.** Put 10–30 images of your subject in a folder:

```
D:/datasets/my_subject/
├── img01.jpg
├── img02.png
└── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`. Images are auto-resized and cropped to the training `resolution` (default 1024). No captions or subfolders needed.

**2. Copy and edit a config.** Pick one of the [example configs](#choosing-a-config), copy it out of `config/examples/`, and change at least:

```yaml
name: "my_subject"                          # names your output folder + weights
instant_prompt: "SKS"                       # the trigger token for your subject
class_prompt: "man"                         # the subject's class
config:
  train:
    dataset_folder: "D:/datasets/my_subject" # absolute path to your images
```

```bash
cp config/examples/train_lora_sdxl_24gb_4090_bf16_metal_1.0.yaml config/my_run.yaml
# edit config/my_run.yaml
```

> Anything you put directly in `config/` (other than `config/examples/`) is gitignored, so your personal run configs with local paths won't be committed.

**3. Train.**

```bash
uv run lorakit config/my_run.yaml
```

When it finishes, your LoRA is at `output/<name>_<version>/pytorch_lora_weights.safetensors`.

---

## Choosing a config

All examples live in `config/examples/`. Copy one into `config/` and edit it.

| Example config | GPU | Speed | Notes |
|---|---|---|---|
| `train_lora_sdxl_24gb_4090_bf16_metal_1.0.yaml` | 24 GB | **Fastest** | bf16 base + `torch.compile` + latent caching. Recommended on a 4090. |
| `train_lora_sdxl_24gb_4090_4bit_1.0.yaml` | 24 GB | Fast | 4-bit QLoRA base. Lower VRAM, slightly slower than bf16. |
| `train_lora_sdxl_24gb_4090_1.0.yaml` | 24 GB | — | Fully-commented reference of every option. |
| `train_lora_sdxl_16gb_t4_1.0.yaml` | 16 GB | — | Low-VRAM (e.g. T4) using 4-bit quantization. |

Not sure? On a 24 GB card use the **bf16 metal** config. On 16 GB, use the **T4** config.

---

## Focusing training on the subject

By default the LoRA learns everything in your images, including the room and the
outfit. `face_mask_source` picks a per-image mask that tells training which pixels
matter:

| Source | Region | CLI | Docs |
|---|---|---|---|
| `face_parse` | 19-class segmentation: clothing excluded, face skin down-weighted | `lorakit-parse` | [face parsing](docs/face-parsing.md) |
| `subject_mask` | BiRefNet subject silhouette | `lorakit-masks` | — |
| `manifest` | InsightFace face box | `lorakit-faces` | [latent face objectives](docs/background-preservation.md) |

`face_parse` is the most precise: it is the only source that separates clothing
from the person, and it also unlocks zoom-out augmentation.

```yaml
config:
  train:
    face_mask_source: face_parse
    face_focus:
      enabled: true
    background_preservation:
      enabled: true
      weight: 0.5
```

Masks are built automatically on the first run; the CLIs just let you inspect
them beforehand.

---

## Using your trained LoRA

The output is a standard diffusers LoRA, so you can load it into any SDXL pipeline:

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/my_subject_1.0/pytorch_lora_weights.safetensors")

image = pipe("portrait photo of SKS man, natural light, 50mm", num_inference_steps=25).images[0]
image.save("result.jpg")
```

(Use your own `instant_prompt` trigger token in the prompt.)

### What lands in `output/`

```
output/<name>_<version>/
├── pytorch_lora_weights.safetensors          # final LoRA
├── config.yaml                               # the exact config used
├── samples/                                  # preview images during training
├── logs/                                     # training logs
└── checkpoint_<name>_<version>_<step>/       # resumable checkpoints
```

---

## Performance

lorakit caches VAE latents once at startup (skipping the per-step VAE pass) and lets you toggle a few high-impact knobs. Measured on an RTX 4090 at 1024px, batch size 1:

| Setup | ms / step | Speedup |
|---|--:|--:|
| 4-bit baseline (gradient checkpointing on, no caching) | ~1334 | 1.0x |
| 4-bit + latent caching + gradient checkpointing off | ~423 | 3.2x |
| bf16 base (no quantization) | ~288 | 4.6x |
| **bf16 + `torch.compile`** | **~227** | **5.9x** |

Tuning knobs (under `train:`):

```yaml
gradient_checkpointing: false  # off = ~2x faster backward; on = lower VRAM
cache_latents: true            # encode images once, not every step
torch_compile: true            # TorchInductor-compiled UNet (bf16 base)
torch_compile_mode: "default"  # "default" | "reduce-overhead" | "max-autotune"
```

Tips:
- **On a 24 GB card, prefer a bf16 base over 4-bit** — 4-bit saves memory but adds per-step dequantization overhead. Use 4-bit only when you're VRAM-limited.
- `torch.compile` pays a one-time compilation cost on the first step (~1–3 min), then runs faster for the rest of training. Avoid combining it with 4-bit quantization.
- If you hit out-of-memory, set `gradient_checkpointing: true`.

---

## Quantization (QLoRA)

lorakit can load the base SDXL model in 8-bit or 4-bit precision via [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) while training the LoRA adapters in higher precision (QLoRA-style). This is what makes 16 GB training possible. Enable it in the `model` section:

```yaml
model:
  name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
  quantization:
    bits: 4                        # 4 (recommended for training) or 8
    quantize_text_encoder: false   # also quantize the two CLIP text encoders
    bnb_4bit_quant_type: "nf4"     # "nf4" or "fp4" (4-bit only)
    bnb_4bit_use_double_quant: true
```

- **Use `bits: 4` for training.** 4-bit NF4 (true QLoRA) computes matmuls in your fp16/bf16 `dtype`, preserving the gradient signal to the LoRA adapters so they learn at the normal learning rate — and it uses *less* memory than 8-bit.
- `bits: 8` uses LLM.int8(), which is tuned for **inference**; its backward pass attenuates gradients, so LoRA learns more slowly. Prefer it only when 4-bit isn't an option.
- The VAE is always kept in fp32 (never quantized) to avoid NaN losses.

---

## Resuming training

Point `resume_from_checkpoint` at a saved checkpoint folder (or `"latest"`):

```yaml
config:
  train:
    resume_from_checkpoint: "latest"
```

---

## Profiling (optional)

To see a per-section `ms/step` breakdown of the training loop (data loading, VAE, UNet forward, backward, optimizer), enable the built-in profiler:

```yaml
config:
  train:
    profile:
      enabled: true
      warmup: 5        # steps excluded from the averages
      report_every: 0  # 0 = print summary only at the end
```

Ready-made profiling configs live in `config/examples/profile_4bit.yaml` and `config/examples/profile_bf16.yaml`.

---

## Troubleshooting

- **`No training images found`** — `dataset_folder` must point directly at a folder containing images (not subfolders), using an absolute path.
- **Out of memory** — set `gradient_checkpointing: true`, switch to a 4-bit config, or lower `resolution`.
- **`Cannot find a working triton installation`** — run `uv sync` (Triton is a declared dependency); avoid running with a stale/hand-modified environment. `torch.compile` requires it.
- **Quantization import errors** — run `uv sync` to (re)install `bitsandbytes`.

---

## Roadmap

- [ ] Prior preservation option
- [x] EMA (Exponential Moving Average) support
- [ ] FLUX.1 integration

## Contributing

Contributions are welcome — please open an issue or pull request.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

Built on the DreamBooth branch of [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) from Hugging Face. Special thanks to Abhishek Thakur for his work on AutoTrain Advanced.
