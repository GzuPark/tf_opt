import logging
import os

from time import perf_counter
from typing import Any, Dict

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from image_classification.mnist import BaseModel


class CQATModel(BaseModel):
    """Cluster preserving quantization aware training"""

    def __init__(
            self,
            root_dir: str,
            base_model_name: str,
            dataset: Dict[str, Any],
            valid_split: float,
            batch_size: int,
            epochs: int,
            logger: logging.Logger,
            method: str = "CQAT",
            verbose: bool = False,
    ) -> None:
        super().__init__(root_dir, dataset, valid_split, batch_size, epochs)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, f"mnist_cluster_{method}_keras.h5")
        self.verbose = 1 if verbose else 0
        self._method = method.lower() if method.lower() in {"qat", "cqat"} else "cqat"
        self._base_model = None

        base_model_path = os.path.join(self.ckpt_dir, base_model_name)
        if os.path.exists(base_model_path):
            self._base_model = tf.keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Do not exist {base_model_path} file.\nTry train the ClusteringModel.")

        self._logger = logger
        self._logger.info(f"Run clustering {method.upper()}")

    def _get_qat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_model(self._base_model)

    def _get_cqat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_apply(
            tfmot.quantization.keras.quantize_annotate_model(self._base_model),
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(),
        )

    def create_model(self) -> None:
        models = dict()
        models["qat"] = self._get_qat_model
        models["cqat"] = self._get_cqat_model

        target_model = models.get(self._method, self._get_cqat_model)
        self.model = target_model()

        if self.verbose:
            self.model.summary()

    def _compile(self):
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self) -> None:
        # TODO: CQAT can save, but load model has issues
        if os.path.exists(self.model_path) and self._method != "cqat":
            with tfmot.quantization.keras.quantize_scope():
                self.model = tf.keras.models.load_model(self.model_path)
            return

        train_kwargs = dict()
        train_kwargs["batch_size"] = self.batch_size
        train_kwargs["epochs"] = self.epochs
        train_kwargs["validation_split"] = self.valid_split
        train_kwargs["verbose"] = self.verbose

        self._compile()
        self.model.fit(self.x_train, self.y_train, **train_kwargs)

        tf.keras.models.save_model(self.model, self.model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = f"cluster_{self._method}"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self.model_path)

        self._logger.info(result)

        return result