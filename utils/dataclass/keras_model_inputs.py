from dataclasses import dataclass, asdict
from typing import Any, Dict, Union


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
