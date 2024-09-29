from pathlib import Path
import json
from typing import OrderedDict, Union
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

def get_config(config_file_path_or_dict: Union[str, dict, OrderedDict, Path]):

    if isinstance(config_file_path_or_dict, dict) or isinstance(config_file_path_or_dict, OrderedDict):
        return preprocess_config(config_file_path_or_dict)
    
    path = Path(config_file_path_or_dict)
    if not path.exists() or not path.is_file():
        raise ValueError(f"config file {path} does not exist")

    #check if the file is a json or yaml file
    if path.suffix == '.json' or path.suffix == '.jsonc':
        with open(path, 'r') as f:
            config = json.load(f, object_pairs_hook=OrderedDict)
    elif path.suffix == '.yaml' or path.suffix == '.yml':
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"config file {path} has an invalid extension")

    return preprocess_config(config, path.stem)
