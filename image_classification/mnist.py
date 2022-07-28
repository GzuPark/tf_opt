import os

from time import perf_counter
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


class _BaseModel(object):

    def __init__(self, root_path: str, validation_split: float = 0.0) -> None:
        self.ckpt_dir = os.path.join(root_path, "ckpt")
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self._data_dir = os.path.join(root_path, "data", "mnist_data")
        self._validation_split = validation_split
        self._batch_size = 128
        self._epochs = 5

        self._check_dirs()
        self._download_dataset()

    def _check_dirs(self) -> None:
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

    def _download_dataset(self) -> None:
        data_path = os.path.join(self._data_dir, "mnist.npz")
        train_dataset, test_dateset = tf.keras.datasets.mnist.load_data(path=data_path)

        self.x_train, self.y_train = train_dataset
        self.x_test, self.y_test = test_dateset

        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0

    def create_model(self, summary: bool = False) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError


class BasicModel(_BaseModel):

    def __init__(self, root_path: str, validation_split: float = 0.0) -> None:
        super().__init__(root_path, validation_split)
        self._model_path = os.path.join(self.ckpt_dir, "mnist_none_keras.h5")
        self._validation_split = validation_split
        self._batch_size = 128
        self._epochs = 5

        print("Run without optimizing")

    def create_model(self, summary: bool = False) -> None:
        input_layer = tf.keras.Input(shape=(28, 28))
        x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_layer)
        x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10)(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)

        if summary:
            self.model.summary()

    def _compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self) -> None:
        if os.path.exists(self._model_path):
            self.model = tf.keras.models.load_model(self._model_path)
            return

        kwargs = dict()
        kwargs["batch_size"] = self._batch_size
        kwargs["epochs"] = self._epochs

        if self._validation_split > 0:
            kwargs["validation_split"] = self._validation_split

        self._compile()
        self.model.fit(self.x_train, self.y_train, **kwargs)

        tf.keras.models.save_model(self.model, self._model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = "None"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self._model_path)

        return result


class PruningModel(_BaseModel):

    def __init__(self, root_path: str, base_model: tf.keras.Model, validation_split: float = 0.0) -> None:
        super().__init__(root_path, validation_split)
        self._model_path = os.path.join(self.ckpt_dir, "mnist_prune_keras.h5")
        self._base_model = base_model
        self._validation_split = validation_split
        self._batch_size = 128
        self._epochs = 5

        print("Run weight pruning")

    def create_model(self, summary: bool = False) -> None:
        pruning_params = dict()

        _num_data = self.x_train.shape[0] * (1 - self._validation_split)
        pruning_params["pruning_schedule"] = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.5,
            final_sparsity=0.8,
            begin_step=0,
            end_step=np.ceil(_num_data / self._batch_size).astype(np.int32) * self._epochs,
        )

        self.model = tfmot.sparsity.keras.prune_low_magnitude(self._base_model, **pruning_params)

        if summary:
            self.model.summary()

    def _compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self) -> None:
        if os.path.exists(self._model_path):
            self.model = tf.keras.models.load_model(self._model_path)
            return

        kwargs = dict()
        kwargs["batch_size"] = self._batch_size
        kwargs["epochs"] = self._epochs
        kwargs["callbacks"] = [tfmot.sparsity.keras.UpdatePruningStep()]

        if self._validation_split > 0:
            kwargs["validation_split"] = self._validation_split

        self._compile()
        self.model.fit(self.x_train, self.y_train, **kwargs)

        model_for_export = tfmot.sparsity.keras.strip_pruning(self.model)
        tf.keras.models.save_model(model_for_export, self._model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        self._compile()

        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = "prune"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self._model_path)

        return result


class QuantizationModel(_BaseModel):

    def __init__(self, root_path: str, base_model: tf.keras.Model, validation_split: float = 0.0) -> None:
        super().__init__(root_path, validation_split)
        self._model_path = os.path.join(self.ckpt_dir, "mnist_quantize_keras.h5")
        self._base_model = base_model
        self._validation_split = validation_split
        self._batch_size = 128
        self._epochs = 5

        print("Run quantization")

    def create_model(self, summary: bool = False) -> None:
        self.model = tfmot.quantization.keras.quantize_model(self._base_model)

        if summary:
            self.model.summary()

    def _compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self) -> None:
        if os.path.exists(self._model_path):
            with tfmot.quantization.keras.quantize_scope():
                self.model = tf.keras.models.load_model(self._model_path)
            return

        kwargs = dict()
        kwargs["batch_size"] = self._batch_size
        kwargs["epochs"] = self._epochs

        if self._validation_split > 0:
            kwargs["validation_split"] = self._validation_split

        self._compile()
        self.model.fit(self.x_train, self.y_train, **kwargs)

        tf.keras.models.save_model(self.model, self._model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = "quantize"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self._model_path)

        return result


class ClusteringModel(_BaseModel):

    def __init__(self, root_path: str, base_model: tf.keras.Model, validation_split: float = 0.0) -> None:
        super().__init__(root_path, validation_split)
        self._model_path = os.path.join(self.ckpt_dir, f"mnist_cluster_keras.h5")
        self._base_model = base_model
        self._validation_split = validation_split
        self._batch_size = 128
        self._epochs = 5

        print("Run weight clustering")

    def create_model(self, summary: bool = False) -> None:
        clustered_params = dict()
        # clustered_params["number_of_clusters"] = 16
        # clustered_params["cluster_centroids_init"] = tfmot.clustering.keras.CentroidInitialization.LINEAR
        clustered_params["number_of_clusters"] = 8
        clustered_params["cluster_centroids_init"] = tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
        clustered_params["cluster_per_channel"] = True

        self.model = tfmot.clustering.keras.cluster_weights(self._base_model, **clustered_params)

        if summary:
            self.model.summary()

    def _compile(self):
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self) -> None:
        if os.path.exists(self._model_path):
            self.model = tf.keras.models.load_model(self._model_path)
            return

        kwargs = dict()
        kwargs["batch_size"] = self._batch_size
        kwargs["epochs"] = self._epochs

        if self._validation_split > 0:
            kwargs["validation_split"] = self._validation_split

        self._compile()
        self.model.fit(self.x_train, self.y_train, **kwargs)

        model_for_export = tfmot.clustering.keras.strip_clustering(self.model)
        tf.keras.models.save_model(model_for_export, self._model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        self._compile()

        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = "cluster"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self._model_path)

        return result


class CQATModel(_BaseModel):
    """Cluster preserving quantization aware training"""

    def __init__(self,
                 root_path: str,
                 base_model: tf.keras.Model,
                 method: str = "CQAT",
                 validation_split: float = 0.0,
                 ) -> None:
        super().__init__(root_path, validation_split)
        self._model_path = os.path.join(self.ckpt_dir, f"mnist_cluster_{method}_keras.h5")
        self._base_model = base_model
        self._method = method.lower() if method.lower() in {"qat", "cqat"} else "cqat"
        self._validation_split = validation_split
        self._batch_size = 128
        self._epochs = 5

        print(f"Run {method.upper()}")

    def _get_qat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_model(self._base_model)

    def _get_cqat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_apply(
            tfmot.quantization.keras.quantize_annotate_model(self._base_model),
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(),
        )

    def create_model(self, summary: bool = False) -> None:
        models = dict()
        models["qat"] = self._get_qat_model
        models["cqat"] = self._get_cqat_model

        target_model = models.get(self._method, self._get_cqat_model)
        self.model = target_model()

        if summary:
            self.model.summary()

    def _compile(self):
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self) -> None:
        # TODO: CQAT can save, but load model has issues
        if os.path.exists(self._model_path) and self._method != "cqat":
            with tfmot.quantization.keras.quantize_scope():
                self.model = tf.keras.models.load_model(self._model_path)
            return

        kwargs = dict()
        kwargs["batch_size"] = self._batch_size
        kwargs["epochs"] = self._epochs

        if self._validation_split > 0:
            kwargs["validation_split"] = self._validation_split

        self._compile()
        self.model.fit(self.x_train, self.y_train, **kwargs)

        tf.keras.models.save_model(self.model, self._model_path, include_optimizer=True)

    def evaluate(self) -> Dict[str, Any]:
        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = f"cluster_{self._method}"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self._model_path)

        return result
