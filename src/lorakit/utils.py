from pathlib import Path
from typing import OrderedDict, Union
from lorakit.config import get_config
import yaml

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
    with output_path.open('w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
def get_job(
        config_path: Union[str, dict, OrderedDict]
):
    config = get_config(config_path)
    if not config['job']:
        raise ValueError(f"config file is invalid. Missing 'job' key")

    job = config['job']
    if job == 'train':
        from lorakit.train import TrainJob
        if 'config' not in config:
            raise ValueError(f"config file is invalid. Missing 'config' key")
        version = config.get('version', None)
        if not version:
            raise ValueError(f"config file is invalid. Missing 'version' key")
        name = config.get('name', None)
        if not name:
            raise ValueError(f"config file is invalid. Missing 'name' key")
        output_folder = config.get('output_folder', None)
        if not output_folder:
            raise ValueError(f"config file is invalid. Missing 'output_folder' key")
        train_job = TrainJob(config['config'], version, name, output_folder)

        # save config as yaml file in the experiment folder
        save_config(config, train_job._experiment_folder / "config.yaml")
        return train_job
    else:
        raise ValueError(f"job {job} is not supported")

