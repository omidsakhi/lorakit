"""Command-line utility for generating offline face-parsing label maps."""

import argparse
from pathlib import Path

from lorakit.face_parsing import (
    DEFAULT_BACKBONE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL_PATH,
    ensure_face_parses,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BiSeNet face-parsing label maps used to exclude clothing and "
            "down-weight face skin during training."
        )
    )
    parser.add_argument("image_folder", type=Path, help="Folder containing training images")
    parser.add_argument(
        "parse_folder",
        type=Path,
        nargs="?",
        default=None,
        help="Folder to write parse maps and parse.json (default: <image_folder>/face_parse)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH), help="BiSeNet checkpoint"
    )
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default=DEFAULT_BACKBONE)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.01,
        help="Reject parses whose subject region covers less than this fraction",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also write color-coded overlays to <parse_folder>/previews",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when an image has no usable face parse",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate parse maps even when the existing ones are up to date",
    )
    args = parser.parse_args()
    if args.input_size <= 0:
        parser.error("--input-size must be positive")
    if not 0.0 <= args.min_coverage < 1.0:
        parser.error("--min-coverage must be in [0, 1)")

    ensure_face_parses(
        args.image_folder,
        args.parse_folder,
        device=args.device,
        model_path=args.model_path,
        backbone=args.backbone,
        input_size=args.input_size,
        min_coverage=args.min_coverage,
        allow_missing=not args.strict,
        force=args.overwrite,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
