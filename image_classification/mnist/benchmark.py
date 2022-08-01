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
from utils.dataclass import KerasModelInputs, TFLiteModelInputs, Result
from utils.enums import TFOptimize, TFLiteQuant


class Benchmark(BenchmarkInterface):

    def __init__(
            self,
            path: str,
            tflite_methods: List[TFLiteQuant],
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
        self._keras_inputs: KerasModelInputs
        self._tflite_inputs: TFLiteModelInputs

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
            method: TFOptimize,
            base_model_filename: Union[TFOptimize, None] = None,
    ) -> KerasModelInputs:
        return KerasModelInputs(
            root_dir=self.root_dir,
            batch_size=self.batch_size,
            epochs=self.epochs,
            valid_split=self.valid_split,
            model_filename=f"mnist_{str(method)}_keras.h5",
            base_model_filename=None if base_model_filename is None else f"mnist_{str(base_model_filename)}_keras.h5",
            method=method,
            verbose=self.verbose,
        )

    def _get_tflite_inputs(
            self,
            optimizer: TFOptimize,
            method: TFLiteQuant = TFLiteQuant.Dynamic,
    ) -> TFLiteModelInputs:
        return TFLiteModelInputs(
            root_dir=self.root_dir,
            dataset_name="mnist",
            optimizer=optimizer,
            method=method,
        )

    def get_optimize_module(self, optimize: TFOptimize) -> Any:
        result = dict()

        result[TFOptimize.NONE] = self._get_optimize_none
        result[TFOptimize.Pruning] = self._get_optimize_prune
        result[TFOptimize.Quantization] = self._get_optimize_quant
        result[TFOptimize.Clustering] = self._get_optimize_cluster
        result[TFOptimize.ClusteringQAT] = self._get_optimize_cluster_qat
        result[TFOptimize.ClusteringCQAT] = self._get_optimize_cluster_cqat
        result[TFOptimize.PruningQAT] = self._get_optimize_prune_qat
        result[TFOptimize.PruningPQAT] = self._get_optimize_prune_pqat
        result[TFOptimize.PruningClustering] = self._get_optimize_prune_cluster
        result[TFOptimize.PruningClusteringQAT] = self._get_optimize_prune_cluster_qat
        result[TFOptimize.PruningClusteringPCQAT] = self._get_optimize_prune_cluster_pcqat

        return result.get(optimize, self._get_optimize_none)

    def _get_optimize_none(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.NONE)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.NONE)
        return mnist.BasicModel

    def _get_optimize_prune(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.Pruning, TFOptimize.NONE)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.Pruning)
        return mnist.PruningModel

    def _get_optimize_quant(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.Quantization, TFOptimize.NONE)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.Quantization)
        return mnist.QuantizationModel

    def _get_optimize_cluster(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.Clustering, TFOptimize.Clustering)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.Clustering)
        return mnist.ClusteringModel

    def _get_optimize_cluster_qat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.ClusteringQAT, TFOptimize.Clustering)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.ClusteringQAT)
        return mnist.CQATModel

    def _get_optimize_cluster_cqat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.ClusteringCQAT, TFOptimize.Clustering)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.ClusteringCQAT)
        return mnist.CQATModel

    def _get_optimize_prune_qat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.PruningQAT, TFOptimize.Pruning)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.PruningQAT)
        return mnist.PQATModel

    def _get_optimize_prune_pqat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.PruningPQAT, TFOptimize.Pruning)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.PruningPQAT)
        return mnist.PQATModel

    def _get_optimize_prune_cluster(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.PruningClustering, TFOptimize.Pruning)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.PruningClustering)
        return mnist.PruneClusterModel

    def _get_optimize_prune_cluster_qat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.PruningClusteringQAT, TFOptimize.PruningClustering)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.PruningClusteringQAT)
        return mnist.PCQATModel

    def _get_optimize_prune_cluster_pcqat(self) -> Any:
        self._keras_inputs = self._get_keras_inputs(TFOptimize.PruningClusteringPCQAT, TFOptimize.PruningClustering)
        self._tflite_inputs = self._get_tflite_inputs(TFOptimize.PruningClusteringPCQAT)
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

        model = module()(self._keras_inputs, self.dataset, logger)
        if not only_infer:
            model.create_model()
            model.train()
        result.append(model.evaluate())

        for method in self.tflite_methods:
            try:
                self._tflite_inputs.update_method(method)
                converter = ImageClassificationConverter(self._tflite_inputs, self.dataset, logger)
                converter.convert(model=model.model)
                result.append(converter.evaluate())
                del converter
            except RuntimeError as e:
                _e = str(e).replace("\n", "")
                logger.error(f"TFLite cannot convert '{self._tflite_inputs.optimizer}' '{method}'. {_e}")

        del model
        self._keras_inputs: KerasModelInputs
        self._tflite_inputs: TFLiteModelInputs

        return result
