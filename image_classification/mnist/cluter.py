import logging
import os

from time import perf_counter
from typing import Any, Dict

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from image_classification.mnist import BaseModel
from utils.dataclass import KerasModelInputs, Result


class ClusteringModel(BaseModel):

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

    def _load_base_model(self) -> None:
        if os.path.exists(self._base_model_path):
            self._base_model = tf.keras.models.load_model(self._base_model_path)
        else:
            raise ValueError(f"Do not exist {self._base_model_path} file.\nTry train the BasicModel.")

    def create_model(self) -> None:
        self._load_base_model()

        clustered_params = dict()
        # clustered_params["number_of_clusters"] = 16
        # clustered_params["cluster_centroids_init"] = tfmot.clustering.keras.CentroidInitialization.LINEAR
        clustered_params["number_of_clusters"] = 8
        clustered_params["cluster_centroids_init"] = tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
        clustered_params["cluster_per_channel"] = True

        self.model = tfmot.clustering.keras.cluster_weights(self._base_model, **clustered_params)

        if self.verbose:
            self.model.summary()

    def _compile(self):
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def _load_model(self) -> bool:
        success = False

        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self._compile()
            success = True

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

        model_for_export = tfmot.clustering.keras.strip_clustering(self.model)
        tf.keras.models.save_model(model_for_export, self.model_path, include_optimizer=True)

    def evaluate(self) -> Result:
        if (self.model is None) and (not self._load_model()):
            self._logger.error(f"Cannot load {self.model_path}.")
            raise ValueError(f"Cannot load {self.model_path}.")

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
