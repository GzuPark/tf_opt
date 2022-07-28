import os.path

from typing import Any, Dict, List

from image_classification import mnist
from utils import set_seed


def run_mnist(path: str, tflite_methods: Dict[str, Dict[str, Any]]) -> None:
    optimizes = [
        "none",
        "prune",
        "quantize",
        "cluster",
        "cluster_qat",
        "cluster_cqat",
        "prune_qat",
        "prune_pqat",
        "prune_cluster_qat",
        "prune_cluster_pcqat",
    ]

    benchmark = mnist.Benchmark(path)
    result = list()

    for optimize in optimizes:
        module = benchmark.get_optimize_module(optimize)
        method = None

        if optimize in tflite_methods["default"]["possible_opts"]:
            method = tflite_methods["default"]["methods"]
        elif optimize in tflite_methods["all"]["possible_opts"]:
            method = tflite_methods["all"]["methods"]

        res = benchmark.run_modules(module, method)
        result.extend(res)

    # Print out
    print_results(result)


def print_results(result: List[Dict[str, Any]]) -> None:
    print(f"| {'Method':>10} | {'Model optimize':>20} | {'Accuracy':>12} | {'Total time':>15} | {'File size':>15} |")
    print(f"|{'-' * 11}:|{'-' * 21}:|{'-' * 13}:|{'-' * 16}:|{'-' * 16}:|")

    for res in result:
        _method = res['method']
        _opt = res['opt']
        _acc = res['accuracy'] * 100
        _time = res['total_time'] * 1000
        _size = res['model_file_size'] / 1024
        print(f"| {_method:>10} | {_opt:>20} | {_acc:>10.2f} % | {_time:>12.1f} ms | {_size:>12.2f} KB |")


def main() -> None:
    set_seed()

    root_path = os.path.dirname(__file__)

    tflite_methods = dict()
    tflite_methods["all"] = {
        "possible_opts": {
            "none",
            "prune",
            "cluster",
        },
        "methods": ["fp32", "fp16", "dynamic", "uint8", "int16x8"],
    }
    tflite_methods["default"] = {
        "possible_opts": {
            "quantize",
            "cluster",
            "cluster_qat",
            "cluster_cqat",
            "prune_qat",
            "prune_pqat",
            "prune_cluster_qat",
            "prune_cluster_pcqat",
        },
        # RuntimeError: Quantization to 16x8-bit not yet supported for op: 'DEQUANTIZE'
        "methods": ["fp32", "fp16", "dynamic", "uint8"],
    }

    run_mnist(root_path, tflite_methods)


if __name__ == '__main__':
    main()
