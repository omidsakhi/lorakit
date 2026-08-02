import json
from collections import OrderedDict
from pathlib import Path

import yaml


def preprocess_config(config: OrderedDict, name: str = None):
    if "job" not in config:
        raise ValueError("config file must have a job section")
    if "config" not in config:
        raise ValueError("config file must have a config section")
    config_string = json.dumps(config)
    config_string = config_string.replace("[name]", name)
    config = json.loads(config_string, object_pairs_hook=OrderedDict)
    return config


def resolve_config_path(config_file_path: str | Path) -> Path:
    """Locate a config file when cwd may be ``external/lorakit`` rather than the repo root."""
    path = Path(config_file_path)
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        raise ValueError(f"config file {path} does not exist")

    if path.is_file():
        return path.resolve()

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate

    raise ValueError(f"config file {path} does not exist (cwd={cwd})")


def _search_bases(config_path: Path | None) -> list[Path]:
    """Search roots for relative paths: cwd first, then ancestors and config dirs."""
    seen: set[Path] = set()
    bases: list[Path] = []

    def add(directory: Path) -> None:
        directory = directory.resolve()
        if directory not in seen:
            seen.add(directory)
            bases.append(directory)

    cwd = Path.cwd()
    add(cwd)
    for parent in cwd.parents:
        add(parent)
    if config_path is not None:
        config_dir = config_path.parent.resolve()
        add(config_dir)
        for parent in config_dir.parents:
            add(parent)
    return bases


def _looks_like_project_root(base: Path) -> bool:
    if (base / "data").is_dir() or (base / "configs").is_dir():
        return True
    return (base / "pyproject.toml").is_file() and (base / "src").is_dir()


def resolve_user_path(
    path: str | Path,
    *,
    config_path: Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve paths from config values relative to cwd and repo root.

    Nested configs (e.g. ``configs/ablation-study/runs/foo.yaml``) must not
    anchor ``results/`` or ``data/`` paths under the config directory.
    """
    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        return resolved

    bases = _search_bases(config_path)

    if must_exist:
        for base in bases:
            candidate = (base / p).resolve()
            if candidate.exists():
                return candidate
        searched = ", ".join(str(b) for b in bases)
        raise FileNotFoundError(f"Path not found: {p} (searched relative to {searched})")

    if p.parts:
        anchor = p.parts[0]
        for base in bases:
            if (base / anchor).exists():
                return (base / p).resolve()

    for base in bases:
        if _looks_like_project_root(base):
            return (base / p).resolve()

    return (Path.cwd() / p).resolve()


def get_config(config_file_path_or_dict: str | dict | OrderedDict | Path):
    if isinstance(config_file_path_or_dict, dict) or isinstance(
        config_file_path_or_dict, OrderedDict
    ):
        return preprocess_config(config_file_path_or_dict), None

    path = resolve_config_path(config_file_path_or_dict)

    if path.suffix == ".json" or path.suffix == ".jsonc":
        with open(path) as f:
            config = json.load(f, object_pairs_hook=OrderedDict)
    elif path.suffix == ".yaml" or path.suffix == ".yml":
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"config file {path} has an invalid extension")

    return preprocess_config(config, path.stem), path
