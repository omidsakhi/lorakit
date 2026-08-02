"""Command-line utility for generating offline subject alpha masks."""

import argparse
from pathlib import Path

from lorakit.subject_masks import DEFAULT_INPUT_SIZE, DEFAULT_MODEL_ID, ensure_subject_masks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BiRefNet subject masks used for background-preserving "
            "loss weighting during training."
        )
    )
    parser.add_argument("image_folder", type=Path, help="Folder containing training images")
    parser.add_argument(
        "mask_folder",
        type=Path,
        nargs="?",
        default=None,
        help="Folder to write masks and masks.json (default: <image_folder>/masks)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="BiRefNet model id")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.01,
        help="Reject masks covering less than this fraction of the image",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when an image has no usable subject mask",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks even when the existing ones are up to date",
    )
    args = parser.parse_args()
    if args.input_size <= 0:
        parser.error("--input-size must be positive")
    if not 0.0 <= args.min_coverage < 1.0:
        parser.error("--min-coverage must be in [0, 1)")

    ensure_subject_masks(
        args.image_folder,
        args.mask_folder,
        device=args.device,
        model_id=args.model,
        input_size=args.input_size,
        min_coverage=args.min_coverage,
        allow_missing=not args.strict,
        force=args.overwrite,
    )


if __name__ == "__main__":
    main()
