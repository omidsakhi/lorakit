from pathlib import Path

class BaseJob:
    def __init__(self, version, name, root_folder):
        self._version = version
        self._name = name
        self._root_folder = root_folder
        self._experiment_folder = Path(self._root_folder, f"{self._name}_{self._version}")
    
    def get_experiment_folder(self):
        return self._experiment_folder

    def run(self):
        pass

