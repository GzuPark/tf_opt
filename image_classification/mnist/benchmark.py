import gc
import logging
import os

from time import sleep
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

import image_classification.mnist as mnist

from image_classification.benchmark_interface import BenchmarkInterface
from image_classification.tflite_converter import ImageClassificationConverter


class Benchmark(BenchmarkInterface):

    def __init__(
            self,
            path: str,
            batch_size: int = 128,
            epochs: int = 5,
            valid_split: float = 0.1,
            verbose: bool = False,
    ) -> None:
        self.root_dir = path
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_split = valid_split
        self.verbose = verbose
        self.dataset = self.load_dataset(path)

        self.keras_kwargs = None
        self.tflite_kwargs = None

        self.tflite_methods_group = dict()
        self.tflite_methods_group["default"] = ["fp32", "fp16", "dynamic", "uint8"]
        self.tflite_methods_group["all"] = ["fp32", "fp16", "dynamic", "uint8", "int16x8"]

        self.tflite_methods = None

    @staticmethod
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

    def get_keras_kwargs(self) -> Dict[str, Any]:
        result = dict()

        result["root_dir"] = self.root_dir
        result["batch_size"] = self.batch_size
        result["epochs"] = self.epochs
        result["valid_split"] = self.valid_split
        result["verbose"] = self.verbose
        result["dataset"] = self.dataset

        return result

    def get_tflite_kwargs(self) -> Dict[str, Any]:
        result = dict()

        result["root_dir"] = self.root_dir
        result["dataset_name"] = "mnist"
        result["dataset"] = self.dataset

        return result

    def get_optimize_module(self, optimize: str) -> Any:
        result = dict()

        result["none"] = self._get_optimize_none
        result["prune"] = self._get_optimize_prune
        result["quant"] = self._get_optimize_quant
        result["cluster"] = self._get_optimize_cluster
        result["cluster_qat"] = self._get_optimize_cluster_qat
        result["cluster_cqat"] = self._get_optimize_cluster_cqat
        result["prune_qat"] = self._get_optimize_prune_qat
        result["prune_pqat"] = self._get_optimize_prune_pqat
        result["prune_cluster"] = self._get_optimize_prune_cluster
        result["prune_cluster_qat"] = self._get_optimize_prune_cluster_qat
        result["prune_cluster_pcqat"] = self._get_optimize_prune_cluster_pcqat

        return result.get(optimize, self._get_optimize_none)

    def _get_optimize_none(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "none"

        self.tflite_methods = self.tflite_methods_group.get("all")

        return mnist.BasicModel

    def _get_optimize_prune(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_none_keras.h5"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "prune"

        self.tflite_methods = self.tflite_methods_group.get("all")

        return mnist.PruningModel

    def _get_optimize_quant(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_none_keras.h5"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "quant"

        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.QuantizationModel

    def _get_optimize_cluster(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_none_keras.h5"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "cluster"

        self.tflite_methods = self.tflite_methods_group.get("all")

        return mnist.ClusteringModel

    def _get_optimize_cluster_qat(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_cluster_keras.h5"
        self.keras_kwargs["method"] = "qat"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "cluster_qat"

        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.CQATModel

    def _get_optimize_cluster_cqat(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_cluster_keras.h5"
        self.keras_kwargs["method"] = "cqat"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "cluster_cqat"

        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.CQATModel

    def _get_optimize_prune_qat(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_prune_keras.h5"
        self.keras_kwargs["method"] = "qat"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "prune_qat"
        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.PQATModel

    def _get_optimize_prune_pqat(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_prune_keras.h5"
        self.keras_kwargs["method"] = "pqat"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "prune_pqat"

        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.PQATModel

    def _get_optimize_prune_cluster(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_prune_keras.h5"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "prune_cluster"

        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.PruneClusterModel

    def _get_optimize_prune_cluster_qat(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_prune_cluster_keras.h5"
        self.keras_kwargs["method"] = "qat"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "prune_cluster_qat"

        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.PCQATModel

    def _get_optimize_prune_cluster_pcqat(self) -> Any:
        self.keras_kwargs = self.get_keras_kwargs()
        self.keras_kwargs["base_model_name"] = "mnist_prune_cluster_keras.h5"
        self.keras_kwargs["method"] = "pcqat"

        self.tflite_kwargs = self.get_tflite_kwargs()
        self.tflite_kwargs["optimizer"] = "prune_cluster_pcqat"

        self.tflite_methods = self.tflite_methods_group.get("default")

        return mnist.PCQATModel

    def run_modules(self, module: Any, logger: logging.Logger) -> List[Dict[str, Any]]:
        tf.keras.backend.clear_session()
        gc.collect()
        sleep(3)

        result = list()

        model = module()(logger=logger, **self.keras_kwargs)
        model.create_model()
        model.train()
        result.append(model.evaluate())

        for method in self.tflite_methods:
            converter = ImageClassificationConverter(method=method, logger=logger, **self.tflite_kwargs)
            converter.convert(model=model.model)
            result.append(converter.evaluate())

            del converter

        del model
        self.keras_kwargs = None
        self.tflite_kwargs = None

        return result
