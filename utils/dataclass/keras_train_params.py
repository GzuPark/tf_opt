from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class TrainParams(object):
    batch_size: int
    epochs: int
    validation_split: float
    verbose: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
