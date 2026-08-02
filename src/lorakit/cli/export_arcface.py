"""Export an official ArcFace PyTorch backbone checkpoint to TorchScript."""

import argparse
from pathlib import Path

import torch

from lorakit.arcface import ArcFaceR50Encoder, iresnet50


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ms1mv3 ArcFace R50 backbone.pth to a TorchScript encoder."
    )
    parser.add_argument("checkpoint", type=Path, help="Official ms1mv3 ArcFace R50 backbone.pth")
    parser.add_argument("output", type=Path, help="Destination .torchscript file")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}. Use --overwrite to replace it.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("ArcFace checkpoint must be a state-dict mapping")
    backbone = iresnet50().float().eval()
    incompatible = backbone.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "ArcFace R50 checkpoint is incompatible with the expected IResNet architecture"
        )
    model = ArcFaceR50Encoder(backbone).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model).save(str(args.output))
    print(f"Exported frozen ArcFace R50 encoder to {args.output}")


if __name__ == "__main__":
    main()
