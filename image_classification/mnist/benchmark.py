import gc
import logging
import os

from time import sleep
from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf

import image_classification.mnist as mnist

from image_classification.benchmark_interface import BenchmarkInterface
from image_classification.tflite_converter import ImageClassificationConverter
from utils.dataclass import KerasModelInputs, Result


class Benchmark(BenchmarkInterface):

    def __init__(
            self,
            path: str,
            tflite_methods: List[str],
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

        self.tflite_methods = tflite_methods
        self._keras_inputs = None
        self.tflite_kwargs = None

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

    def _get_keras_inputs(
            self,
            model_filename: str,
            base_model_filename: Union[str, None] = None,
            method: Union[str, None] = None,
    ) -> KerasModelInputs:
        return KerasModelInputs(
            root_dir=self.root_dir,
            batch_size=self.batch_size,
            epochs=self.epochs,
            valid_split=self.valid_split,
            model_filename=f"mnist_{model_filename}_keras.h5",
            base_model_filename=None if base_model_filename is None else f"mnist_{base_model_filename}_keras.h5",
            method=method,
            verbose=self.verbose,
        )

    def get_tflite_kwargs(self, optimizer: str) -> Dict[str, Any]:
        result = dict()

        result["root_dir"] = self.root_dir
        result["dataset_name"] = "mnist"
        result["dataset"] = self.dataset
        result["optimizer"] = optimizer

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
        self._keras_inputs = self._get_keras_inputs("none")
        self.tflite_kwargs = self.get_tflite_kwargs("none")
        return mnist.BasicModel

    def _get_optimize_prune(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("prune", "none")
        self.tflite_kwargs = self.get_tflite_kwargs("prune")
        return mnist.PruningModel

    def _get_optimize_quant(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("quant", "none")
        self.tflite_kwargs = self.get_tflite_kwargs("quant")
        return mnist.QuantizationModel

    def _get_optimize_cluster(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("cluster", "none")
        self.tflite_kwargs = self.get_tflite_kwargs("cluster")
        return mnist.ClusteringModel

    def _get_optimize_cluster_qat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("cluster_qat", "cluster", "qat")
        self.tflite_kwargs = self.get_tflite_kwargs("cluster_qat")
        return mnist.CQATModel

    def _get_optimize_cluster_cqat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("cluster_cqat", "cluster", "cqat")
        self.tflite_kwargs = self.get_tflite_kwargs("cluster_cqat")
        return mnist.CQATModel

    def _get_optimize_prune_qat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("prune_qat", "prune", "qat")
        self.tflite_kwargs = self.get_tflite_kwargs("prune_qat")
        return mnist.PQATModel

    def _get_optimize_prune_pqat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("prune_pqat", "prune", "pqat")
        self.tflite_kwargs = self.get_tflite_kwargs("prune_pqat")
        return mnist.PQATModel

    def _get_optimize_prune_cluster(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("prune_cluster", "prune")
        self.tflite_kwargs = self.get_tflite_kwargs("prune_cluster")
        return mnist.PruneClusterModel

    def _get_optimize_prune_cluster_qat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("prune_cluster_qat", "prune_cluster", "qat")
        self.tflite_kwargs = self.get_tflite_kwargs("prune_cluster_qat")
        return mnist.PCQATModel

    def _get_optimize_prune_cluster_pcqat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs("prune_cluster_pcqat", "prune_cluster", "pcqat")
        self.tflite_kwargs = self.get_tflite_kwargs("prune_cluster_pcqat")
        return mnist.PCQATModel

    def run_modules(
            self, module: Any,
            logger: logging.Logger,
            only_infer: bool = False,
    ) -> List[Result]:
        tf.keras.backend.clear_session()
        gc.collect()
        sleep(3)

        result = list()

        # model = module()(logger=logger, **self.keras_kwargs)
        model = module()(self._keras_inputs, self.dataset, logger)
        if not only_infer:
            model.create_model()
            model.train()
        result.append(model.evaluate())

        for method in self.tflite_methods:
            try:
                converter = ImageClassificationConverter(method=method, logger=logger, **self.tflite_kwargs)
                converter.convert(model=model.model)
                result.append(converter.evaluate())
                del converter
            except RuntimeError as e:
                _e = str(e).replace("\n", "")
                logger.error(f"TFLite cannot convert '{self.tflite_kwargs.get('optimizer')}' '{method}'. {_e}")

        del model
        self._keras_inputs = None
        self.tflite_kwargs = None

        return result
