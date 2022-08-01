from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class Result(object):
    method: str
    optimizer: str
    accuracy: float
    total_time: float
    model_file_size: int

    def __post_init__(self) -> None:
        self.accuracy *= 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
