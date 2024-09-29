# lorakit

A simple SDXL fine-tuning toolkit based on the DreamBooth branch of AutoTrain Advanced from 🤗, inspired by the way ai-toolkit approaches configuration.

## Overview

lorakit is a flexible toolkit for fine-tuning Stable Diffusion XL (SDXL) models using the DreamBooth technique. It leverages the capabilities of AutoTrain Advanced and provides an easy-to-use configuration-based approach for customizing your training process. Additionally, lorakit supports quick experimentation for research purposes, allowing users to rapidly iterate on ideas and test different configurations with minimal setup.

## Features

- SDXL fine-tuning using DreamBooth
- LoRA (Low-Rank Adaptation) support for unet and text encoder
- Configurable training parameters
- Support for various optimizers (AdamW, AdamW8bit, AdamWScheduleFree, Prodigy)
- Customizable learning rate schedulers
- Gradient checkpointing and accumulation
- Mixed precision training (fp16, bf16, fp32)
- Resumable training from checkpoints
- Sample generation during training

## ToDo

- [ ] Prior preservation option
- [ ] EMA (Exponential Moving Average) support
- [ ] Quantization
- [ ] Getting rid of "Loading pipeline components..."
- [ ] Integrating FLUX.1

## Installation

```
git clone https://github.com/faceharmonyai/lorakit.git lorakit
cd lorakit
python -m venv .venv
source .venv/bin/activate # or .venv/Scripts/activate on Windows
pip install .
```

## Usage

1. Prepare your dataset and configuration file.
2. Run the training process using the `lorakit` command-line tool:

```
lorakit examples/train_lora_sdxl_24gb_1.0.yaml
```

## Configuration

lorakit uses YAML configuration files for easy customization of the training process. Find an example configuration file in the `examples` directory.

## Real-World Applications

lorakit has been successfully used in production environments, including the [FaceHarmony.ai](https://faceharmony.ai) app, demonstrating its reliability and effectiveness in real-world AI applications.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is based on the DreamBooth branch of AutoTrain Advanced from Hugging Face 🤗. We appreciate their contributions to the open-source community and special thank to Abhishek Thakur for his amazing work on AutoTrain Advanced.

## Keywords

LoRA, SDXL fine-tuning, DreamBooth, AI art generation, text-to-image models, Stable Diffusion XL, machine learning, deep learning, neural networks, transfer learning, low-rank adaptation, diffusion models, generative AI, PyTorch, Hugging Face, AI research, automation toolkit, fine-tuning techniques, AutoTrain Advanced
