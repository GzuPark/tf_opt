import os.path

from image_classification import mnist
from utils import print_outputs, set_seed


def run_mnist(path: str) -> None:
    optimizes = [
        "none",
        "prune",
        "quant",
        "cluster",
        "cluster_qat",
        "cluster_cqat",
        "prune_qat",
        "prune_pqat",
        "prune_cluster",
        "prune_cluster_qat",
        "prune_cluster_pcqat",
    ]

    benchmark = mnist.Benchmark(path)
    result = list()

    for optimize in optimizes:
        module = benchmark.get_optimize_module(optimize)
        result.extend(benchmark.run_modules(module))

    print_outputs(result, "ms", "KB")


def main() -> None:
    set_seed()

    root_path = os.path.dirname(__file__)
    run_mnist(root_path)


if __name__ == '__main__':
    main()
