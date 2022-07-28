import gc
import os.path

from time import sleep
from typing import Any, Dict, List

import tensorflow as tf

from image_classification import mnist, ImageClassificationConverter
from utils import set_seed


def run_modules(
        module: Any,
        keras_kwargs: Dict[str, Any],
        tflite_kwargs: Dict[str, Any],
        tflite_methods: List[str],
) -> List[Dict[str, Any]]:
    tf.keras.backend.clear_session()
    gc.collect()
    sleep(2)

    result = list()

    model = module(**keras_kwargs)
    model.create_model()
    model.train()
    result.append(model.evaluate())

    for method in tflite_methods:
        converter = ImageClassificationConverter(method=method, **tflite_kwargs)
        converter.convert(model=model.model)
        result.append(converter.evaluate())

    return result


def run_mnist(path: str) -> None:
    result = list()

    # RuntimeError: Quantization to 16x8-bit not yet supported for op: 'DEQUANTIZE'
    tflite_methods_1 = ["fp32", "fp16", "dynamic", "uint8"]
    tflite_methods_2 = tflite_methods_1 + ["int16x8"]

    dataset = mnist.load_dataset(path)

    keras_kwargs = dict()
    keras_kwargs["root_dir"] = path
    keras_kwargs["dataset"] = dataset
    keras_kwargs["valid_split"] = 0.1
    keras_kwargs["verbose"] = False

    tflite_kwargs = dict()
    tflite_kwargs["root_dir"] = path
    tflite_kwargs["dataset_name"] = "mnist"
    tflite_kwargs["dataset"] = dataset

    # No optimized
    tflite_kwargs["optimizer"] = "none"
    result.extend(run_modules(mnist.BasicModel, keras_kwargs, tflite_kwargs, tflite_methods_2))

    # Pruning
    keras_kwargs["base_model_name"] = "mnist_none_keras.h5"
    tflite_kwargs["optimizer"] = "prune"
    result.extend(run_modules(mnist.PruningModel, keras_kwargs, tflite_kwargs, tflite_methods_2))

    # Quantization
    keras_kwargs["base_model_name"] = "mnist_none_keras.h5"
    tflite_kwargs["optimizer"] = "quantize"
    result.extend(run_modules(mnist.QuantizationModel, keras_kwargs, tflite_kwargs, tflite_methods_1))

    # Clustering
    keras_kwargs["base_model_name"] = "mnist_none_keras.h5"
    tflite_kwargs["optimizer"] = "cluster"
    result.extend(run_modules(mnist.ClusteringModel, keras_kwargs, tflite_kwargs, tflite_methods_2))

    # Clustering - QAT
    keras_kwargs["base_model_name"] = "mnist_cluster_keras.h5"
    keras_kwargs["method"] = "qat"
    tflite_kwargs["optimizer"] = "cluster_qat"
    result.extend(run_modules(mnist.CQATModel, keras_kwargs, tflite_kwargs, tflite_methods_1))

    # Clustering - CQAT
    keras_kwargs["base_model_name"] = "mnist_cluster_keras.h5"
    keras_kwargs["method"] = "cqat"
    tflite_kwargs["optimizer"] = "cluster_cqat"
    result.extend(run_modules(mnist.CQATModel, keras_kwargs, tflite_kwargs, tflite_methods_1))

    # Print out
    print_results(result)


def print_results(result: List[Dict[str, Any]]) -> None:
    print(f"| {'Method':>10} | {'Model optimize':>15} | {'Accuracy':>12} | {'Total time':>15} | {'File size':>15} |")
    print(f"|{'-' * 11}:|{'-' * 16}:|{'-' * 13}:|{'-' * 16}:|{'-' * 16}:|")

    for res in result:
        _method = res['method']
        _opt = res['opt']
        _acc = res['accuracy'] * 100
        _time = res['total_time'] * 1000
        _size = res['model_file_size'] / 1024
        print(f"| {_method:>10} | {_opt:>15} | {_acc:>10.2f} % | {_time:>12.1f} ms | {_size:>12.2f} KB |")


def main() -> None:
    set_seed()

    root_path = os.path.dirname(__file__)
    run_mnist(root_path)


if __name__ == '__main__':
    main()
