from dataclasses import dataclass

from utils.enums import TFOptimize, TFLiteMethods


@dataclass(frozen=True)
class TFLiteModelInputs(object):
    root_dir: str
    dataset_name: str
    optimizer: TFOptimize
    method: TFLiteMethods

    def update_method(self, method: TFLiteMethods) -> None:
        object.__setattr__(self, "method", method)
