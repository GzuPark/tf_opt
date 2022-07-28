import os

from time import perf_counter
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def load_dataset(root_dir: str) -> Dict[str, np.ndarray]:
    data_dir = os.path.join(root_dir, "data", "mnist")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_path = os.path.join(data_dir, "mnist.npz")
    train_dataset, test_dateset = tf.keras.datasets.mnist.load_data(path=data_path)

    x_train, y_train = train_dataset
    x_test, y_test = test_dateset

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    result = dict()
    result["x_train"] = x_train
    result["y_train"] = y_train
    result["x_test"] = x_test
    result["y_test"] = y_test

    return result


class _BaseModel(object):

    def __init__(self, root_dir: str, dataset: Dict[str, Any], valid_split: float = 0.0) -> None:
        self.valid_split = valid_split
        self.batch_size = 128
        self.epochs = 5

        self.x_train = dataset["x_train"]
        self.y_train = dataset["y_train"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]

        self.ckpt_dir = os.path.join(root_dir, "ckpt")
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


class BasicModel(_BaseModel):

    def __init__(
            self,
            root_dir: str,
            dataset: Dict[str, Any],
            valid_split: float,
            verbose: bool = False
    ) -> None:
        super().__init__(root_dir, dataset, valid_split)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, "mnist_none_keras.h5")
        self.verbose = 1 if verbose else 0

        print("Run without optimizing")

    def create_model(self) -> None:
        input_layer = tf.keras.Input(shape=(28, 28))
        x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_layer)
        x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10)(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)

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
        result["opt"] = "None"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self.model_path)

        return result


class PruningModel(_BaseModel):

    def __init__(
            self,
            root_dir: str,
            base_model_name: str,
            dataset: Dict[str, Any],
            valid_split: float,
            verbose: bool = False
    ) -> None:
        super().__init__(root_dir, dataset, valid_split)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, "mnist_prune_keras.h5")
        self.verbose = 1 if verbose else 0
        self._base_model = None

        base_model_path = os.path.join(self.ckpt_dir, base_model_name)
        if os.path.exists(base_model_path):
            self._base_model = tf.keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Do not exist {base_model_path} file.\nTry train the BasicModel.")

        print("Run weight pruning")

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

        return result


class QuantizationModel(_BaseModel):

    def __init__(
            self,
            root_dir: str,
            base_model_name: str,
            dataset: Dict[str, Any],
            valid_split: float,
            verbose: bool = False
    ) -> None:
        super().__init__(root_dir, dataset, valid_split)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, "mnist_quantize_keras.h5")
        self.verbose = 1 if verbose else 0
        self._base_model = None

        base_model_path = os.path.join(self.ckpt_dir, base_model_name)
        if os.path.exists(base_model_path):
            self._base_model = tf.keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Do not exist {base_model_path} file.\nTry train the BasicModel.")

        print("Run quantization")

    def create_model(self) -> None:
        self.model = tfmot.quantization.keras.quantize_model(self._base_model)

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
        result["opt"] = "quantize"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self.model_path)

        return result


class ClusteringModel(_BaseModel):

    def __init__(
            self,
            root_dir: str,
            base_model_name: str,
            dataset: Dict[str, Any],
            valid_split: float,
            verbose: bool = False
    ) -> None:
        super().__init__(root_dir, dataset, valid_split)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, "mnist_cluster_keras.h5")
        self.verbose = 1 if verbose else 0
        self._base_model = None

        base_model_path = os.path.join(self.ckpt_dir, base_model_name)
        if os.path.exists(base_model_path):
            self._base_model = tf.keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Do not exist {base_model_path} file.\nTry train the BasicModel.")

        print("Run weight clustering")

    def create_model(self) -> None:
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

    def train(self) -> None:
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
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
        self._compile()

        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        end_time = perf_counter()

        result = dict()
        result["method"] = "keras"
        result["opt"] = "cluster"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self.model_path)

        return result


class CQATModel(_BaseModel):
    """Cluster preserving quantization aware training"""

    def __init__(
            self,
            root_dir: str,
            base_model_name: str,
            dataset: Dict[str, Any],
            valid_split: float,
            method: str = "CQAT",
            verbose: bool = False
    ) -> None:
        super().__init__(root_dir, dataset, valid_split)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, f"mnist_cluster_{method}_keras.h5")
        self.verbose = 1 if verbose else 0
        self._method = method.lower() if method.lower() in {"qat", "cqat"} else "cqat"
        self._base_model = None

        base_model_path = os.path.join(self.ckpt_dir, base_model_name)
        if os.path.exists(base_model_path):
            self._base_model = tf.keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Do not exist {base_model_path} file.\nTry train the BasicModel.")

        print(f"Run clustering {method.upper()}")

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

        return result


class PQATModel(_BaseModel):
    """Pruning preserving quantization aware training"""

    def __init__(
            self,
            root_dir: str,
            base_model_name: str,
            dataset: Dict[str, Any],
            valid_split: float,
            method: str = "PQAT",
            verbose: bool = False
    ) -> None:
        super().__init__(root_dir, dataset, valid_split)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, f"mnist_prune_{method}_keras.h5")
        self.verbose = 1 if verbose else 0
        self._method = method.lower() if method.lower() in {"qat", "pqat"} else "pqat"
        self._base_model = None

        base_model_path = os.path.join(self.ckpt_dir, base_model_name)
        if os.path.exists(base_model_path):
            self._base_model = tf.keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Do not exist {base_model_path} file.\nTry train the BasicModel.")

        print(f"Run pruning {method.upper()}")

    def _get_qat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_model(self._base_model)

    def _get_pqat_model(self) -> Any:
        return tfmot.quantization.keras.quantize_apply(
            tfmot.quantization.keras.quantize_annotate_model(self._base_model),
            tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme(),
        )

    def create_model(self) -> None:
        models = dict()
        models["qat"] = self._get_qat_model
        models["pqat"] = self._get_pqat_model

        target_model = models.get(self._method, self._get_pqat_model)
        self.model = target_model()

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
        result["opt"] = f"prune_{self._method}"
        result["accuracy"] = accuracy
        result["total_time"] = end_time - start_time
        result["model_file_size"] = os.path.getsize(self.model_path)

        return result
