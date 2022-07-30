import logging
import os

from time import perf_counter
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from image_classification.mnist import BaseModel


class PruningModel(BaseModel):

    def __init__(
            self,
            root_dir: str,
            base_model_name: str,
            dataset: Dict[str, Any],
            valid_split: float,
            batch_size: int,
            epochs: int,
            logger: logging.Logger,
            verbose: bool = False,
    ) -> None:
        super().__init__(root_dir, dataset, valid_split, batch_size, epochs, verbose)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, "mnist_prune_keras.h5")
        self._base_model = None

        base_model_path = os.path.join(self.ckpt_dir, base_model_name)
        if os.path.exists(base_model_path):
            self._base_model = tf.keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Do not exist {base_model_path} file.\nTry train the BasicModel.")

        self._logger = logger
        self._logger.info("Run weight pruning")

    def create_model(self) -> None:
        pruning_params = dict()

        _num_data = self.x_train.shape[0] * (1 - self.valid_split)
        pruning_params["pruning_schedule"] = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.5,
            final_sparsity=0.8,
            begin_step=0,
            end_step=np.ceil(_num_data / self.batch_size).astype(np.int32) * self.epochs,
        )

        self.model = tfmot.sparsity.keras.prune_low_magnitude(self._base_model, **pruning_params)

        if self.verbose:
            self.model.summary()

    def _compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self) -> None:
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            return

        train_kwargs = dict()
        train_kwargs["batch_size"] = self.batch_size
        train_kwargs["epochs"] = self.epochs
        train_kwargs["callbacks"] = [tfmot.sparsity.keras.UpdatePruningStep()]
        train_kwargs["validation_split"] = self.valid_split
        train_kwargs["verbose"] = self.verbose

        self._compile()
        self.model.fit(self.x_train, self.y_train, **train_kwargs)

        model_for_export = tfmot.sparsity.keras.strip_pruning(self.model)
        tf.keras.models.save_model(model_for_export, self.model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        self._compile()

        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = "prune"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self.model_path)

        self._logger.info(result)

        return result
