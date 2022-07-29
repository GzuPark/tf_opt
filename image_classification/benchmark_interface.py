import logging

from abc import ABC, abstractmethod
from typing import Any


class BenchmarkInterface(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load_dataset(root_dir: str) -> Any:
        pass

    @abstractmethod
    def get_optimize_module(self, optimize: str) -> Any:
        pass

    @abstractmethod
    def _get_optimize_none(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_prune(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_quant(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_cluster(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_cluster_qat(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_cluster_cqat(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_prune_qat(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_prune_pqat(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_prune_cluster(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_prune_cluster_qat(self) -> Any:
        pass

    @abstractmethod
    def _get_optimize_prune_cluster_pcqat(self) -> Any:
        pass

    @abstractmethod
    def run_modules(self, module: Any, logger: logging.Logger) -> Any:
        pass
