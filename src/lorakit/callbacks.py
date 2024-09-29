from typing import Callable

class BaseCallback:
    def __init__(self):
        pass

class ProgressCallback(BaseCallback):
    def __init__(self, progress_function: Callable[[float], None]):
        self._progress_function = progress_function

    def set_progress(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("Progress value must be a float")
        normalized_value = max(0.0, min(1.0, value))
        self._progress_function(normalized_value)

class LossCallback(BaseCallback):
    def __init__(self, loss_function: Callable[[float], float]):
        self._loss_function = loss_function

    def set_loss(self, loss: float) -> None:
        if not isinstance(loss, float):
            raise ValueError("Loss value must be a float")
        self._loss_function(loss)
