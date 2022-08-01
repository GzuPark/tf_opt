from enum import Enum


class TFOptimize(Enum):
    NONE = "none"
    Pruning = "prune"
    Quantization = "quant"
    Clustering = "cluster"
    ClusteringQAT = "cluster_qat"
    ClusteringCQAT = "cluster_cqat"
    PruningQAT = "prune_qat"
    PruningPQAT = "prune_pqat"
    PruningClustering = "prune_cluster"
    PruningClusteringQAT = "prune_cluster_qat"
    PruningClusteringPCQAT = "prune_cluster_pcqat"

    def __str__(self) -> str:
        return str(self.value)
