from dataclasses import dataclass


@dataclass(frozen=True)
class TFLiteModelInputs(object):
    root_dir: str
    dataset_name: str
    optimizer: str
    method: str

    def update_method(self, method: str) -> None:
        object.__setattr__(self, "method", method)
