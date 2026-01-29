"""
Unit tests for tactical mapper.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ml.labels import FormationClass, auto_label_snapshot
from ml.tactical_mapper import TacticalMapper, TacticalDecision
from ml.constants import DEFENSE_CENTER


class TestAutoLabel(unittest.TestCase):
    def test_concentrated_label(self):
        """Narrow angle group should be labeled CONCENTRATED (집중)."""
        import math
        n = 10
        positions = []
        velocities = []
        base_angle = 0.0
        for i in range(n):
            angle = base_angle + np.random.uniform(-0.1, 0.1)
            dist = 5500
            x = DEFENSE_CENTER[0] + dist * math.cos(angle)
            y = DEFENSE_CENTER[1] + dist * math.sin(angle)
            dx = DEFENSE_CENTER[0] - x
            dy = DEFENSE_CENTER[1] - y
            d = math.sqrt(dx**2 + dy**2)
            positions.append([x, y])
            velocities.append([50 * dx / d, 50 * dy / d])

        label = auto_label_snapshot(
            np.array(positions), np.array(velocities)
        )
        self.assertEqual(label, FormationClass.CONCENTRATED)

    def test_wave_label(self):
        """Same direction, layered distances should be WAVE (파상)."""
        import math
        # 3 waves of 3 ships, same approach angle, separated by ~800m
        base_angle = 0.5
        positions = []
        velocities = []
        for wave_idx in range(3):
            wave_dist = 5500 + wave_idx * 800  # 5500, 6300, 7100
            for i in range(3):
                lateral = (i - 1) * 0.05  # small lateral spread
                angle = base_angle + lateral
                x = DEFENSE_CENTER[0] + wave_dist * math.cos(angle)
                y = DEFENSE_CENTER[1] + wave_dist * math.sin(angle)
                dx = DEFENSE_CENTER[0] - x
                dy = DEFENSE_CENTER[1] - y
                d = math.sqrt(dx**2 + dy**2)
                positions.append([x, y])
                velocities.append([50 * dx / d, 50 * dy / d])

        label = auto_label_snapshot(
            np.array(positions), np.array(velocities)
        )
        self.assertEqual(label, FormationClass.WAVE)

    def test_diversionary_label(self):
        """360° spread should be DIVERSIONARY (양동)."""
        import math
        n = 10
        positions = []
        velocities = []
        dist = 5500
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = DEFENSE_CENTER[0] + dist * math.cos(angle)
            y = DEFENSE_CENTER[1] + dist * math.sin(angle)
            dx = DEFENSE_CENTER[0] - x
            dy = DEFENSE_CENTER[1] - y
            d = math.sqrt(dx**2 + dy**2)
            positions.append([x, y])
            velocities.append([50 * dx / d, 50 * dy / d])

        label = auto_label_snapshot(
            np.array(positions), np.array(velocities)
        )
        self.assertEqual(label, FormationClass.DIVERSIONARY)

    def test_too_few_enemies(self):
        """< 3 enemies should return None."""
        positions = np.array([[6000, 6000], [6100, 6100]], dtype=np.float64)
        velocities = np.array([[-50, -50], [-50, -50]], dtype=np.float64)
        label = auto_label_snapshot(positions, velocities)
        self.assertIsNone(label)


class TestTacticalMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = TacticalMapper()

    def test_decision_output_structure(self):
        """compute_decision should return TacticalDecision."""
        import math
        n = 8
        positions = []
        velocities = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            dist = 5500
            x = DEFENSE_CENTER[0] + dist * math.cos(angle)
            y = DEFENSE_CENTER[1] + dist * math.sin(angle)
            dx = DEFENSE_CENTER[0] - x
            dy = DEFENSE_CENTER[1] - y
            d = math.sqrt(dx**2 + dy**2)
            positions.append([x, y])
            velocities.append([50 * dx / d, 50 * dy / d])

        enemy_pos = np.array(positions)
        enemy_vel = np.array(velocities)

        # 5 pair positions
        pair_pos = np.array([
            [3500, 5500],
            [5500, 3500],
            [7500, 5500],
            [5500, 7500],
            [5500, 5500],
        ], dtype=np.float64)

        decision = self.mapper.compute_decision(enemy_pos, enemy_vel, pair_pos)

        self.assertIsInstance(decision, TacticalDecision)
        self.assertIsInstance(decision.formation_class, FormationClass)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertGreater(len(decision.cluster_metrics), 0)

    def test_all_pairs_assigned(self):
        """With enough enemies, all pairs should get assignments."""
        import math
        n = 10
        positions = []
        velocities = []
        for i in range(n):
            angle = np.random.uniform(0, 0.5)
            dist = 5500
            x = DEFENSE_CENTER[0] + dist * math.cos(angle)
            y = DEFENSE_CENTER[1] + dist * math.sin(angle)
            dx = DEFENSE_CENTER[0] - x
            dy = DEFENSE_CENTER[1] - y
            d = math.sqrt(dx**2 + dy**2)
            positions.append([x, y])
            velocities.append([50 * dx / d, 50 * dy / d])

        enemy_pos = np.array(positions)
        enemy_vel = np.array(velocities)

        pair_pos = np.array([
            [7000, 5500],
            [7200, 5600],
            [7100, 5400],
            [7300, 5500],
            [7000, 5600],
        ], dtype=np.float64)

        decision = self.mapper.compute_decision(enemy_pos, enemy_vel, pair_pos)

        # At least some pairs should be assigned targets
        self.assertGreater(len(decision.pair_target_enemy), 0)


if __name__ == "__main__":
    unittest.main()
