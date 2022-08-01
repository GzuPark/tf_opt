from dataclasses import dataclass
from typing import Union

from utils.enums import TFOptimize


@dataclass(frozen=True)
class KerasModelInputs(object):
    root_dir: str
    batch_size: int
    epochs: int
    valid_split: float
    model_filename: str
    base_model_filename: Union[str, None]
    method: str
    optimizer: Union[TFOptimize, None]
    verbose: bool
