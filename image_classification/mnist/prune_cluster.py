import logging
import os

from time import perf_counter
from typing import Any, Dict

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tensorflow_model_optimization.python.core.clustering.keras.experimental.cluster as pc_cluster

from image_classification.mnist import BaseModel


class PruneClusterModel(BaseModel):
    """Sparsity preserving clustering"""

    def __init__(
            self,
            root_dir: str,
            model_filename: str,
            base_model_filename: str,
            dataset: Dict[str, Any],
            valid_split: float,
            batch_size: int,
            epochs: int,
            logger: logging.Logger,
            verbose: bool = False,
    ) -> None:
        super().__init__(root_dir, dataset, valid_split, batch_size, epochs, verbose)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, model_filename)
        self._base_model_path = os.path.join(self.ckpt_dir, base_model_filename)
        self._base_model = None

        self._logger = logger
        self._logger.info("Run pruning-clustering")

    def _load_base_model(self) -> None:
        if os.path.exists(self._base_model_path):
            self._base_model = tf.keras.models.load_model(self._base_model_path)
        else:
            raise ValueError(f"Do not exist {self._base_model_path} file.\nTry train the BasicModel.")

    def create_model(self) -> None:
        self._load_base_model()

        clustering_params = dict()
        clustering_params["number_of_clusters"] = 8
        clustering_params["cluster_centroids_init"] = tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
        clustering_params["preserve_sparsity"] = True

        self.model = pc_cluster.cluster_weights(self._base_model, **clustering_params)

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

        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self._compile()
            success = True

        return success

    def train(self) -> None:
        if self._load_model():
            return

        train_kwargs = dict()
        train_kwargs["batch_size"] = self.batch_size
        train_kwargs["epochs"] = self.epochs
        train_kwargs["validation_split"] = self.valid_split
        train_kwargs["verbose"] = self.verbose

        self._compile()
        self.model.fit(self.x_train, self.y_train, **train_kwargs)

        model_for_export = tfmot.clustering.keras.strip_clustering(self.model)
        tf.keras.models.save_model(model_for_export, self.model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        if (self.model is None) and (not self._load_model()):
            self._logger.error(f"Cannot load {self.model_path}.")
            raise ValueError(f"Cannot load {self.model_path}.")

        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = f"prune_cluster"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self.model_path)

        self._logger.info(result)

        return result
