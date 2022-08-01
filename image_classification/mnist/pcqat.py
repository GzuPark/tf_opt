import logging
import os

from time import perf_counter
from typing import Any, Dict

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from image_classification.mnist import BaseModel
from utils.dataclass import KerasModelInputs, Result
from utils.enums import TFOptimize


class PCQATModel(BaseModel):
    """Sparsity and cluster preserving quantization aware training"""

    def __init__(self, inputs: KerasModelInputs, dataset: Dict[str, Any], logger: logging.Logger) -> None:
        super().__init__(inputs, dataset)

        self.model_path = os.path.join(self.ckpt_dir, inputs.model_filename)
        self.model = None

        self._base_model_path = os.path.join(self.ckpt_dir, inputs.base_model_filename)
        self._base_model = None

        self._method = inputs.method
        self._optimizer = inputs.optimizer

        self._logger = logger
        self._logger.info(f"Run {self._optimizer}")

    def _get_qat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_model(self._base_model)

    def _get_pcqat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_apply(
            tfmot.quantization.keras.quantize_annotate_model(self._base_model),
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True),
        )

    def _load_base_model(self) -> None:
        if os.path.exists(self._base_model_path):
            self._base_model = tf.keras.models.load_model(self._base_model_path)
        else:
            raise ValueError(f"Do not exist {self._base_model_path} file.\nTry train the PruneClusterModel.")

    def create_model(self) -> None:
        self._load_base_model()

        models = dict()
        models[TFOptimize.PruningClusteringQAT] = self._get_qat_model
        models[TFOptimize.PruningClusteringPCQAT] = self._get_pcqat_model

        target_model = models.get(self._optimizer, self._get_pcqat_model)
        self.model = target_model()

        if self.verbose:
            self.model.summary()

    def _compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def _load_model(self) -> bool:
        success = False

        try:
            if os.path.exists(self.model_path):
                with tfmot.quantization.keras.quantize_scope():
                    self.model = tf.keras.models.load_model(self.model_path)
                    success = True
        except ValueError as e:
            self._logger.error(f"PCQAT model cannot load {self.model_path}. {e}")

        return success

    def train(self) -> None:
        if self._load_model():
            return

        self._compile()
        self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.valid_split,
            verbose=self.verbose,
        )

        tf.keras.models.save_model(self.model, self.model_path, include_optimizer=True)

    def evaluate(self) -> Result:
        if (self.model is None) and (not self._load_model()):
            self._logger.warning(f"PCQAT model train start.")
            self.create_model()
            self.train()

        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        end_time = perf_counter()

        result = Result(
            method=str(self._method),
            optimizer=str(self._optimizer),
            accuracy=accuracy,
            total_time=end_time - start_time,
            model_file_size=os.path.getsize(self.model_path),
        )

        self._logger.info(result.to_dict())

        return result
