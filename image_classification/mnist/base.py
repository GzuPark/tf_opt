import os

from typing import Any, Dict


class BaseModel(object):

    def __init__(
            self,
            root_dir: str,
            dataset: Dict[str, Any],
            valid_split: float = 0.0,
            batch_size: int = 128,
            epochs: int = 5,
            verbose: bool = False,
    ) -> None:
        self.valid_split = valid_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = 1 if verbose else 0

        self.x_train = dataset["x_train"]
        self.y_train = dataset["y_train"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]

        self.ckpt_dir = os.path.join(root_dir, "ckpt", "mnist")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def _compile(self) -> None:
        raise NotImplementedError

    def create_model(self) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError
