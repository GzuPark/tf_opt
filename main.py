import os.path

import utils

from image_classification import mnist
from utils.enums import TFOptimize, TFLiteQuant


def run_mnist(path: str) -> None:
    tf_model_optimize_methods = [
        TFOptimize.NONE,
        TFOptimize.Pruning,
        TFOptimize.Quantization,
        TFOptimize.Clustering,
        TFOptimize.ClusteringQAT,
        TFOptimize.ClusteringCQAT,
        TFOptimize.PruningQAT,
        TFOptimize.PruningPQAT,
        TFOptimize.PruningClustering,
        TFOptimize.PruningClusteringQAT,
        TFOptimize.PruningClusteringPCQAT,
    ]

    tflite_quantize_methods = [
        TFLiteQuant.FP32,
        TFLiteQuant.FP16,
        TFLiteQuant.Dynamic,
        TFLiteQuant.UINT8,
        TFLiteQuant.INT16x8,
    ]

    logger = utils.get_logger(path, "mnist")
    benchmark = mnist.Benchmark(path, tflite_quantize_methods)
    result = list()

    for tf_model_optimize_method in tf_model_optimize_methods:
        module = benchmark.get_optimize_module(tf_model_optimize_method)
        result.extend(benchmark.run_modules(module, logger, only_infer=False))

    utils.print_table(result, "ms", "KB")


def main() -> None:
    utils.set_seed()

    root_path = os.path.dirname(__file__)
    run_mnist(root_path)


if __name__ == '__main__':
    main()
