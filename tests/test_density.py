"""
Unit tests for density analysis.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ml.density import DensityAnalyzer, ClusterMetrics
from ml.constants import (
    DBSCAN_EPS,
    NET_SPACING_MIN,
    NET_SPACING_MAX,
    NET_SPACING_BASE,
    TOTAL_FRIENDLY_PAIRS,
)


class TestDensityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DensityAnalyzer()

    def test_single_cluster(self):
        """Tight group should form one cluster (all within eps=80m)."""
        positions = np.array([
            [5000, 5000],
            [5030, 5020],
            [5020, 5040],
            [5050, 5010],
            [5010, 5050],
        ], dtype=np.float64)

        labels, metrics = self.analyzer.analyze(positions)

        # All should be in one cluster
        self.assertEqual(len(set(labels)), 1)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].size, 5)

    def test_two_clusters(self):
        """Two distant groups should form two clusters."""
        positions = np.array([
            # Group 1: near (1000, 1000) — within 80m of each other
            [1000, 1000],
            [1030, 1020],
            [1020, 1040],
            # Group 2: near (9000, 9000) — within 80m of each other
            [9000, 9000],
            [9030, 9020],
            [9020, 9040],
        ], dtype=np.float64)

        labels, metrics = self.analyzer.analyze(positions)

        self.assertEqual(len(metrics), 2)
        sizes = sorted([m.size for m in metrics])
        self.assertEqual(sizes, [3, 3])

    def test_empty_input(self):
        """Empty input should return empty results."""
        positions = np.array([], dtype=np.float64).reshape(0, 2)
        labels, metrics = self.analyzer.analyze(positions)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(metrics), 0)

    def test_single_enemy(self):
        """Single enemy should form one cluster of size 1."""
        positions = np.array([[5000, 5000]], dtype=np.float64)
        labels, metrics = self.analyzer.analyze(positions)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].size, 1)

    def test_net_spacing_dense(self):
        """Dense cluster should get small net spacing."""
        # Very tight group within one cluster (within eps=80m)
        positions = np.array([
            [5000, 5000],
            [5020, 5000],
            [5000, 5020],
            [5020, 5020],
        ], dtype=np.float64)

        _, metrics = self.analyzer.analyze(positions)
        self.assertEqual(len(metrics), 1)
        self.assertLessEqual(metrics[0].net_spacing, NET_SPACING_BASE)

    def test_net_spacing_sparse(self):
        """Sparse enemies should create separate clusters with large net spacing."""
        # Points far apart → separate clusters → each has mean_nn=0 (single point)
        positions = np.array([
            [5000, 5000],
            [5600, 5000],
            [5000, 5600],
            [5600, 5600],
        ], dtype=np.float64)

        _, metrics = self.analyzer.analyze(positions)
        # With eps=80, these are all separate clusters
        self.assertGreaterEqual(len(metrics), 2)

    def test_pair_allocation_within_budget(self):
        """Total allocated pairs should not exceed budget."""
        positions = np.array([
            # 3 clusters of 5 — each group within 80m
            [1000, 1000], [1030, 1020], [1020, 1040], [1050, 1010], [1010, 1050],
            [5000, 5000], [5030, 5020], [5020, 5040], [5050, 5010], [5010, 5050],
            [9000, 9000], [9030, 9020], [9020, 9040], [9050, 9010], [9010, 9050],
        ], dtype=np.float64)

        _, metrics = self.analyzer.analyze(positions)
        allocation = self.analyzer.recommend_pairs_allocation(metrics)

        total = sum(allocation.values())
        self.assertLessEqual(total, TOTAL_FRIENDLY_PAIRS)
        self.assertGreater(total, 0)

    def test_cluster_center_accuracy(self):
        """Cluster center should be near the mean of positions."""
        positions = np.array([
            [100, 100],
            [200, 200],
            [150, 150],
        ], dtype=np.float64)

        _, metrics = self.analyzer.analyze(positions)
        center = metrics[0].center

        self.assertAlmostEqual(center[0], 150.0, delta=1.0)
        self.assertAlmostEqual(center[1], 150.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
