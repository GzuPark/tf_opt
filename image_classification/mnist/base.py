import os

from typing import Any, Dict

from utils.dataclass import KerasModelInputs, Result


class BaseModel(object):

    def __init__(self, inputs: KerasModelInputs, dataset: Dict[str, Any]) -> None:
        self.valid_split = inputs.valid_split
        self.batch_size = inputs.batch_size
        self.epochs = inputs.epochs
        self.verbose = 1 if inputs.verbose else 0

        self.x_train = dataset["x_train"]
        self.y_train = dataset["y_train"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]

        self.ckpt_dir = os.path.join(inputs.root_dir, "ckpt", "mnist")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def _compile(self) -> None:
        raise NotImplementedError

    def create_model(self) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> Result:
        raise NotImplementedError
