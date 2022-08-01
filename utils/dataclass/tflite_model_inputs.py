from dataclasses import dataclass

from utils.enums import TFOptimize, TFLiteQuant


@dataclass(frozen=True)
class TFLiteModelInputs(object):
    root_dir: str
    dataset_name: str
    method: TFLiteQuant
    optimizer: TFOptimize

    def update_method(self, method: TFLiteQuant) -> None:
        object.__setattr__(self, "method", method)
