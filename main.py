import os.path

import utils

from image_classification import mnist


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

    logger = utils.get_logger(path, "mnist")
    benchmark = mnist.Benchmark(path)
    result = list()

    for optimize in optimizes:
        module = benchmark.get_optimize_module(optimize)
        result.extend(benchmark.run_modules(module, logger))

    utils.print_table(result, "ms", "KB")


def main() -> None:
    utils.set_seed()

    root_path = os.path.dirname(__file__)
    run_mnist(root_path)


if __name__ == '__main__':
    main()
