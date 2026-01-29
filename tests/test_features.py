"""
Unit tests for feature extraction.
"""

import sys
import os
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ml.features import FeatureExtractor
from ml.constants import NUM_FEATURES, DEFENSE_CENTER


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(DEFENSE_CENTER)

    def test_output_shape(self):
        """Feature vector should be 19-dimensional."""
        positions = np.array([
            [6000, 6000],
            [6100, 6100],
            [6200, 6050],
        ], dtype=np.float64)
        velocities = np.array([
            [-50, -50],
            [-50, -50],
            [-50, -50],
        ], dtype=np.float64)

        features = self.extractor.extract(positions, velocities)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (NUM_FEATURES,))

    def test_single_enemy_returns_features(self):
        """Even a single enemy should produce features (with 0 for pairwise)."""
        positions = np.array([[7000, 5500]], dtype=np.float64)
        velocities = np.array([[-50, 0]], dtype=np.float64)

        features = self.extractor.extract(positions, velocities)
        self.assertIsNotNone(features)
        self.assertEqual(features[18], 1.0)  # num_active_enemies

    def test_empty_returns_none(self):
        """Empty input should return None."""
        positions = np.array([], dtype=np.float64).reshape(0, 2)
        velocities = np.array([], dtype=np.float64).reshape(0, 2)

        features = self.extractor.extract(positions, velocities)
        self.assertIsNone(features)

    def test_circular_std_narrow(self):
        """Narrow angle spread should produce small circular std."""
        angles = np.array([0.1, 0.12, 0.08, 0.11, 0.09])
        std = FeatureExtractor.circular_std(angles)
        self.assertLess(std, 5.0)  # should be very small (degrees)

    def test_circular_std_wide(self):
        """Wide angle spread should produce large circular std."""
        angles = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        std = FeatureExtractor.circular_std(angles)
        self.assertGreater(std, 50.0)  # should be large (degrees)

    def test_angular_range_narrow(self):
        """Narrow cluster should have small angular range."""
        angles = np.array([0.1, 0.15, 0.12, 0.08, 0.11])
        rng = FeatureExtractor.angular_range(angles)
        self.assertLess(rng, 10.0)  # degrees

    def test_angular_range_full_circle(self):
        """Points spread around full circle should have ~360° range."""
        angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        rng = FeatureExtractor.angular_range(angles)
        self.assertGreater(rng, 300.0)

    def test_quadrant_count(self):
        """Test quadrant counting."""
        # All in first quadrant
        angles = np.array([0.1, 0.2, 0.3])
        self.assertEqual(FeatureExtractor.count_quadrants(angles), 1)

        # Two quadrants
        angles = np.array([0.1, 0.2, math.pi / 2 + 0.1])
        self.assertEqual(FeatureExtractor.count_quadrants(angles), 2)

        # All four
        angles = np.array([0.1, math.pi / 2 + 0.1, -math.pi + 0.1, -0.1])
        self.assertEqual(FeatureExtractor.count_quadrants(angles), 4)

    def test_spearhead_features(self):
        """SPEARHEAD pattern should have small angular std."""
        # All enemies from the east
        n = 10
        base_angle = 0.0
        spread = 0.1  # very narrow
        positions = []
        velocities = []
        for i in range(n):
            angle = base_angle + np.random.uniform(-spread, spread)
            dist = 5500 + np.random.uniform(0, 500)
            x = DEFENSE_CENTER[0] + dist * math.cos(angle)
            y = DEFENSE_CENTER[1] + dist * math.sin(angle)
            dx = DEFENSE_CENTER[0] - x
            dy = DEFENSE_CENTER[1] - y
            d = math.sqrt(dx**2 + dy**2)
            positions.append([x, y])
            velocities.append([50 * dx / d, 50 * dy / d])

        features = self.extractor.extract(
            np.array(positions), np.array(velocities)
        )
        # f2 = angular std should be small
        self.assertLess(features[1], 30.0)

    def test_encirclement_features(self):
        """ENCIRCLEMENT pattern should have large angular std and low ETA std."""
        n = 10
        dist = 5500
        positions = []
        velocities = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = DEFENSE_CENTER[0] + dist * math.cos(angle)
            y = DEFENSE_CENTER[1] + dist * math.sin(angle)
            dx = DEFENSE_CENTER[0] - x
            dy = DEFENSE_CENTER[1] - y
            d = math.sqrt(dx**2 + dy**2)
            positions.append([x, y])
            velocities.append([50 * dx / d, 50 * dy / d])

        features = self.extractor.extract(
            np.array(positions), np.array(velocities)
        )
        # f2 = angular std should be large
        self.assertGreater(features[1], 60.0)
        # f14 = ETA std should be small (similar distances, similar speeds)
        self.assertLess(features[13], 15.0)


if __name__ == "__main__":
    unittest.main()
