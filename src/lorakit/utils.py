from collections import OrderedDict
from pathlib import Path

import yaml

from lorakit.config import get_config


def save_config(config, output_path):
    # Convert OrderedDict to regular dict
    def ordered_dict_to_dict(od):
        if isinstance(od, dict):
            return dict((k, ordered_dict_to_dict(v)) for k, v in od.items())
        elif isinstance(od, list):
            return [ordered_dict_to_dict(item) for item in od]
        else:
            return od

    config_dict = ordered_dict_to_dict(config)

    # Use Path for file operations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as YAML with improved formatting
    with output_path.open("w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def resolve_output_folder(output_folder: str | Path) -> Path:
    """Create an output root relative to the current process directory."""
    output_path = Path(output_folder).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def ensure_fresh_experiment_folder(
    experiment_folder: str | Path, *, resume_from_checkpoint=None
) -> None:
    """Refuse to start a fresh train that would overwrite an existing run.

    Resume is allowed to reuse the folder so checkpoints / logs can continue.
    """
    path = Path(experiment_folder)
    if path.exists() and resume_from_checkpoint is None:
        raise FileExistsError(
            f"Experiment folder already exists: {path}. "
            "Bump `version` for a new run, or set train.resume_from_checkpoint "
            "to resume without overwriting previous logs and metrics."
        )


def get_job(config_path: str | dict | OrderedDict):
    config, config_file = get_config(config_path)
    if not config["job"]:
        raise ValueError("config file is invalid. Missing 'job' key")

    job = config["job"]
    if job == "train":
        from lorakit.train import TrainJob

        if "config" not in config:
            raise ValueError("config file is invalid. Missing 'config' key")
        version = config.get("version", None)
        if not version:
            raise ValueError("config file is invalid. Missing 'version' key")
        name = config.get("name", None)
        if not name:
            raise ValueError("config file is invalid. Missing 'name' key")
        output_folder = config.get("output_folder", None)
        if not output_folder:
            raise ValueError("config file is invalid. Missing 'output_folder' key")
        output_folder = resolve_output_folder(output_folder)
        experiment_folder = Path(output_folder) / f"{name}_{version}"
        resume_from_checkpoint = (config["config"].get("train") or {}).get("resume_from_checkpoint")
        ensure_fresh_experiment_folder(
            experiment_folder, resume_from_checkpoint=resume_from_checkpoint
        )
        train_job = TrainJob(
            config["config"], version, name, str(output_folder), config_path=config_file
        )

        # save config as yaml file in the experiment folder
        save_config(config, train_job._experiment_folder / "config.yaml")
        return train_job
    if job == "sample":
        from lorakit.sample import SampleJob

        if "config" not in config:
            raise ValueError("config file is invalid. Missing 'config' key")
        version = config.get("version", None)
        if not version:
            raise ValueError("config file is invalid. Missing 'version' key")
        name = config.get("name", None)
        if not name:
            raise ValueError("config file is invalid. Missing 'name' key")
        output_folder = config.get("output_folder", None)
        if not output_folder:
            raise ValueError("config file is invalid. Missing 'output_folder' key")
        output_folder = resolve_output_folder(output_folder)
        sample_job = SampleJob(
            config["config"], version, name, str(output_folder), config_path=config_file
        )
        save_config(config, sample_job._experiment_folder / "config.yaml")
        return sample_job
    else:
        raise ValueError(f"job {job} is not supported")
