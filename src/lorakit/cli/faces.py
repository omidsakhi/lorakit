"""Command-line utility for generating offline training face manifests."""

import argparse
from pathlib import Path

from lorakit.faces_prep import ensure_face_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate the offline face-box manifest used by face focus / identity loss.")
    )
    parser.add_argument(
        "image_folder", type=Path, help="Folder containing subject/reference images"
    )
    parser.add_argument("output", type=Path, help="Path to write faces.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection",
        choices=["largest", "highest-confidence"],
        default="largest",
        help="How to choose a face when an image contains several",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a partial manifest instead of failing when an image has no detected face",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    if args.det_size <= 0 or args.det_threshold <= 0 or args.det_threshold > 1:
        parser.error("--det-size must be positive and --det-threshold must be in (0, 1]")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}. Use --overwrite to replace it.")

    ensure_face_manifest(
        args.image_folder,
        args.output,
        device=args.device,
        model_name=args.model,
        det_size=args.det_size,
        det_threshold=args.det_threshold,
        selection=args.selection,
        allow_missing=args.allow_missing,
        force=True,
    )


if __name__ == "__main__":
    main()
