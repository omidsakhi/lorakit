"""Prepare offline ``faces.json`` manifests for training (InsightFace).

Training imports this when a manifest must be created and, for the sample
ArcFace metric, partway through the run -- so InsightFace is an ordinary
dependency rather than an extra.
"""

from __future__ import annotations

import json
from pathlib import Path

from lorakit.face_manifest import generate_face_manifest, image_paths


def build_insightface_detector(
    *,
    model_name: str = "buffalo_l",
    device: str = "cuda",
    det_size: int = 640,
    det_threshold: float = 0.5,
):
    """Construct an InsightFace ``app.get`` callable for one-shot detection."""
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
    except ImportError as error:
        raise RuntimeError(
            f"Face detection needs a package this environment is missing: {error} "
            "Run `uv sync` to install lorakit's dependencies."
        ) from error
    if device == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(
        ctx_id=0 if device == "cuda" else -1,
        det_size=(det_size, det_size),
        det_thresh=det_threshold,
    )
    active_providers = app.models["detection"].session.get_providers()
    if device == "cuda" and "CUDAExecutionProvider" not in active_providers:
        print("WARNING: InsightFace CUDA provider is unavailable; using CPU detection.")
    return app.get


def face_manifest_needs_build(image_folder: str | Path, output: str | Path) -> bool:
    """True when ``output`` is missing or any training image is newer than it."""
    image_folder = Path(image_folder)
    output = Path(output)
    if not output.is_file():
        return True
    try:
        data = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(data, dict):
        return True
    paths = image_paths(image_folder)
    output_mtime = output.stat().st_mtime
    return any(path.stat().st_mtime > output_mtime for path in paths)


def ensure_face_manifest(
    image_folder: str | Path,
    output: str | Path | None = None,
    *,
    device: str = "cuda",
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_threshold: float = 0.5,
    selection: str = "largest",
    allow_missing: bool = True,
    force: bool = False,
) -> Path:
    """Create ``faces.json`` if missing/stale; return the manifest path."""
    image_folder = Path(image_folder)
    output = Path(output) if output is not None else image_folder / "faces.json"
    if not force and not face_manifest_needs_build(image_folder, output):
        return output

    print(f"Generating face manifest for {image_folder} -> {output}")
    paths = image_paths(image_folder)
    detector = build_insightface_detector(
        model_name=model_name,
        device=device,
        det_size=det_size,
        det_threshold=det_threshold,
    )
    manifest, missing = generate_face_manifest(paths, detector, selection=selection)
    if missing and not allow_missing:
        names = ", ".join(path.name for path in missing)
        raise RuntimeError(
            f"No valid face box for {len(missing)} image(s): {names}. "
            "Fix the images/detector settings or set allow_missing."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} face box(es) to {output}")
    if missing:
        print(f"WARNING: skipped {len(missing)} image(s) with no valid detected face.")
    return output
