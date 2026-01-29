"""
Density analysis module using DBSCAN + k-NN hybrid approach.

Responsibilities:
  - Cluster enemies using DBSCAN (eps=800m, min_samples=1)
  - Compute per-cluster density metrics
  - Recommend pair allocations per cluster
  - Compute dynamic net spacing based on cluster density
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from ml.constants import (
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    NET_SPACING_BASE,
    NET_SPACING_MIN,
    NET_SPACING_MAX,
    MAX_PAIRS_PER_CLUSTER,
    MIN_PAIRS_PER_CLUSTER,
    TOTAL_FRIENDLY_PAIRS,
    DENSITY_HIGH_THRESHOLD,
    DENSITY_LOW_THRESHOLD,
)


@dataclass
class ClusterMetrics:
    """Metrics for a single enemy cluster."""
    cluster_id: int
    center: Tuple[float, float]
    size: int                   # number of enemies in cluster
    mean_nn_distance: float     # mean nearest-neighbor distance (meters)
    mean_pairwise_distance: float
    recommended_pairs: int      # how many friendly pairs to assign
    net_spacing: float          # recommended net spacing (meters)
    enemy_indices: List[int]    # indices into the original positions array


class DensityAnalyzer:
    """DBSCAN-based density analysis for enemy formations."""

    def __init__(
        self,
        eps: float = DBSCAN_EPS,
        min_samples: int = DBSCAN_MIN_SAMPLES,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def cluster_enemies(self, positions: np.ndarray) -> np.ndarray:
        """
        Cluster enemy positions using DBSCAN.

        Args:
            positions: (N, 2) array of enemy (x, y) positions

        Returns:
            (N,) array of cluster labels (>= 0). With min_samples=1,
            there should be no noise points (-1).
        """
        if len(positions) == 0:
            return np.array([], dtype=int)
        labels = self.dbscan.fit_predict(positions)
        return labels

    def compute_cluster_metrics(
        self, positions: np.ndarray, labels: np.ndarray
    ) -> List[ClusterMetrics]:
        """
        Compute detailed metrics for each cluster.

        Args:
            positions: (N, 2) enemy positions
            labels: (N,) cluster labels from DBSCAN

        Returns:
            List of ClusterMetrics for each cluster
        """
        unique_labels = sorted(set(labels))
        # Remove noise label if present
        unique_labels = [l for l in unique_labels if l >= 0]

        metrics = []
        for cid in unique_labels:
            mask = labels == cid
            cluster_pos = positions[mask]
            indices = list(np.where(mask)[0])
            n = len(cluster_pos)

            # Center
            center = (float(np.mean(cluster_pos[:, 0])),
                       float(np.mean(cluster_pos[:, 1])))

            # Density metrics
            if n == 1:
                mean_nn = 0.0
                mean_pw = 0.0
            elif n == 2:
                d = float(np.linalg.norm(cluster_pos[0] - cluster_pos[1]))
                mean_nn = d
                mean_pw = d
            else:
                pw_dist = cdist(cluster_pos, cluster_pos)
                np.fill_diagonal(pw_dist, np.inf)
                nn_distances = np.min(pw_dist, axis=1)
                mean_nn = float(np.mean(nn_distances))

                np.fill_diagonal(pw_dist, 0.0)
                triu = pw_dist[np.triu_indices(n, k=1)]
                mean_pw = float(np.mean(triu))

            # Pair allocation
            recommended = self._recommend_pairs(n, mean_nn)

            # Net spacing
            net_spacing = self._compute_net_size(mean_nn)

            metrics.append(ClusterMetrics(
                cluster_id=cid,
                center=center,
                size=n,
                mean_nn_distance=mean_nn,
                mean_pairwise_distance=mean_pw,
                recommended_pairs=recommended,
                net_spacing=net_spacing,
                enemy_indices=indices,
            ))

        return metrics

    def _recommend_pairs(self, cluster_size: int, mean_nn: float) -> int:
        """
        Recommend number of friendly pairs for a cluster based on
        its size and density.
        """
        if cluster_size <= 1:
            return MIN_PAIRS_PER_CLUSTER

        # Density-based allocation
        if mean_nn < DENSITY_HIGH_THRESHOLD:
            pairs = 3
        elif mean_nn < DENSITY_LOW_THRESHOLD:
            pairs = 2
        else:
            pairs = 1

        # Also scale by cluster size
        if cluster_size >= 5:
            pairs = max(pairs, 2)
        if cluster_size >= 8:
            pairs = max(pairs, 3)

        return min(pairs, MAX_PAIRS_PER_CLUSTER)

    def _compute_net_size(self, mean_nn: float) -> float:
        """
        Compute recommended net spacing based on cluster density.
        Dense clusters → smaller nets. Sparse clusters → larger nets.
        """
        if mean_nn <= 0.0:
            return NET_SPACING_BASE

        # Linear interpolation between min and max
        # At mean_nn = DENSITY_HIGH_THRESHOLD → NET_SPACING_MIN
        # At mean_nn = DENSITY_LOW_THRESHOLD → NET_SPACING_MAX
        if mean_nn <= DENSITY_HIGH_THRESHOLD:
            return NET_SPACING_MIN
        elif mean_nn >= DENSITY_LOW_THRESHOLD:
            return NET_SPACING_MAX
        else:
            t = (mean_nn - DENSITY_HIGH_THRESHOLD) / (DENSITY_LOW_THRESHOLD - DENSITY_HIGH_THRESHOLD)
            return NET_SPACING_MIN + t * (NET_SPACING_MAX - NET_SPACING_MIN)

    def analyze(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, List[ClusterMetrics]]:
        """
        Full density analysis pipeline.

        Args:
            positions: (N, 2) enemy positions

        Returns:
            (labels, cluster_metrics) tuple
        """
        labels = self.cluster_enemies(positions)
        metrics = self.compute_cluster_metrics(positions, labels)
        return labels, metrics

    def recommend_pairs_allocation(
        self,
        cluster_metrics: List[ClusterMetrics],
        total_pairs: int = TOTAL_FRIENDLY_PAIRS,
    ) -> Dict[int, int]:
        """
        Allocate friendly pairs across clusters, respecting total budget.

        Returns dict: cluster_id -> number_of_pairs_assigned
        """
        if not cluster_metrics:
            return {}

        # Raw recommendations
        raw = {m.cluster_id: m.recommended_pairs for m in cluster_metrics}
        total_requested = sum(raw.values())

        if total_requested <= total_pairs:
            # Have enough pairs
            return raw

        # Need to scale down proportionally
        allocation = {}
        remaining = total_pairs
        # Sort by size descending (prioritize bigger clusters)
        sorted_clusters = sorted(cluster_metrics, key=lambda m: m.size, reverse=True)

        for m in sorted_clusters:
            if remaining <= 0:
                allocation[m.cluster_id] = 0
            else:
                scaled = max(1, round(raw[m.cluster_id] * total_pairs / total_requested))
                scaled = min(scaled, remaining)
                allocation[m.cluster_id] = scaled
                remaining -= scaled

        # Distribute any leftover
        if remaining > 0:
            for m in sorted_clusters:
                if remaining <= 0:
                    break
                if allocation.get(m.cluster_id, 0) < MAX_PAIRS_PER_CLUSTER:
                    allocation[m.cluster_id] = allocation.get(m.cluster_id, 0) + 1
                    remaining -= 1

        return allocation
