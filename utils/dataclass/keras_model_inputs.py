from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class KerasModelInputs(object):
    root_dir: str
    batch_size: int
    epochs: int
    valid_split: float
    model_filename: str
    base_model_filename: Union[str, None]
    method: Union[str, None]
    verbose: bool
