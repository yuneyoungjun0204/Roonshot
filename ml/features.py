"""
Feature extraction for enemy formation classification.

Produces a 19-dimensional feature vector from a snapshot of enemy positions
and velocities, relative to the defense center.

Feature groups:
  [f1-f4]   Angular distribution
  [f5-f8]   Distance distribution
  [f9-f12]  Speed / heading consistency
  [f13-f15] ETA (estimated time of arrival) consistency
  [f16-f18] Pairwise density metrics
  [f19]     Number of active enemies
"""

import math
import numpy as np
from typing import Tuple, Optional

from ml.constants import DEFENSE_CENTER, NUM_FEATURES


class FeatureExtractor:
    """Extracts a 19-dimensional feature vector from enemy snapshot data."""

    def __init__(self, center: Tuple[float, float] = DEFENSE_CENTER):
        self.center = np.array(center, dtype=np.float64)

    # -----------------------------------------------------------------
    # Circular statistics helpers
    # -----------------------------------------------------------------
    @staticmethod
    def circular_mean(angles_rad: np.ndarray) -> float:
        """Compute circular mean of angles (radians). Returns radians."""
        s = np.mean(np.sin(angles_rad))
        c = np.mean(np.cos(angles_rad))
        return float(np.arctan2(s, c))

    @staticmethod
    def circular_std(angles_rad: np.ndarray) -> float:
        """
        Circular standard deviation in degrees.
        Uses the formula: sqrt(-2 * ln(R)) where R = resultant length.
        """
        if len(angles_rad) < 2:
            return 0.0
        s = np.mean(np.sin(angles_rad))
        c = np.mean(np.cos(angles_rad))
        R = np.sqrt(s ** 2 + c ** 2)
        R = np.clip(R, 1e-10, 1.0)
        return float(np.degrees(np.sqrt(-2.0 * np.log(R))))

    @staticmethod
    def angular_range(angles_rad: np.ndarray) -> float:
        """
        Compute the smallest arc (in degrees) that contains all bearing angles.
        Handles wraparound correctly.
        """
        if len(angles_rad) < 2:
            return 0.0
        sorted_angles = np.sort(angles_rad % (2 * np.pi))
        # Gaps between consecutive sorted angles
        gaps = np.diff(sorted_angles)
        # Include the wraparound gap
        wrap_gap = (2 * np.pi) - sorted_angles[-1] + sorted_angles[0]
        all_gaps = np.append(gaps, wrap_gap)
        max_gap = np.max(all_gaps)
        arc = 2 * np.pi - max_gap
        return float(np.degrees(arc))

    # -----------------------------------------------------------------
    # Per-enemy feature computation
    # -----------------------------------------------------------------
    def _compute_per_enemy(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> dict:
        """
        Compute per-enemy features.

        Args:
            positions: (N, 2) enemy positions
            velocities: (N, 2) enemy velocities

        Returns dict with arrays for bearings, distances, speeds, etas, etc.
        """
        dx = positions[:, 0] - self.center[0]
        dy = positions[:, 1] - self.center[1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        bearings = np.arctan2(dy, dx)

        # Approach speed (radial component toward center)
        unit_to_center_x = np.where(distances > 0.1, -dx / distances, 0.0)
        unit_to_center_y = np.where(distances > 0.1, -dy / distances, 0.0)
        approach_speed = (
            velocities[:, 0] * unit_to_center_x
            + velocities[:, 1] * unit_to_center_y
        )

        # Speed magnitude
        speed_mag = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)

        # Heading alignment: how well does the velocity direction align
        # with the vector pointing toward center?
        # cos(angle between velocity and to-center vector)
        heading_alignment = np.where(
            speed_mag > 0.1,
            approach_speed / speed_mag,
            0.0,
        )

        # ETA
        safe_approach = np.maximum(approach_speed, 1.0)
        etas = distances / safe_approach

        return {
            "bearings": bearings,
            "distances": distances,
            "approach_speed": approach_speed,
            "speed_mag": speed_mag,
            "heading_alignment": heading_alignment,
            "etas": etas,
        }

    def _compute_pairwise(self, positions: np.ndarray) -> dict:
        """
        Compute pairwise distance metrics.
        """
        n = len(positions)
        if n < 2:
            return {
                "mean_pairwise": 0.0,
                "std_pairwise": 0.0,
                "mean_nn": 0.0,
            }

        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        pw_dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # Extract upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(n, k=1)
        pw_values = pw_dist[triu_idx]

        # Nearest neighbor distances
        np.fill_diagonal(pw_dist, np.inf)
        nn_distances = np.min(pw_dist, axis=1)

        return {
            "mean_pairwise": float(np.mean(pw_values)),
            "std_pairwise": float(np.std(pw_values)),
            "mean_nn": float(np.mean(nn_distances)),
        }

    # -----------------------------------------------------------------
    # Quadrant counting
    # -----------------------------------------------------------------
    @staticmethod
    def count_quadrants(bearings_rad: np.ndarray) -> int:
        """Count how many of 4 quadrants are occupied by at least one enemy."""
        # Quadrants: [0, pi/2), [pi/2, pi), [-pi, -pi/2), [-pi/2, 0)
        quadrants = set()
        for b in bearings_rad:
            if 0 <= b < math.pi / 2:
                quadrants.add(0)
            elif math.pi / 2 <= b < math.pi:
                quadrants.add(1)
            elif -math.pi <= b < -math.pi / 2:
                quadrants.add(2)
            else:
                quadrants.add(3)
        return len(quadrants)

    # -----------------------------------------------------------------
    # Main extraction
    # -----------------------------------------------------------------
    def extract(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract 19-dimensional feature vector from enemy snapshot.

        Args:
            positions: (N, 2) array of enemy (x, y) positions
            velocities: (N, 2) array of enemy (vx, vy) velocities

        Returns:
            (19,) numpy array or None if too few enemies
        """
        n = len(positions)
        if n < 1:
            return None

        per = self._compute_per_enemy(positions, velocities)
        pw = self._compute_pairwise(positions)

        features = np.zeros(NUM_FEATURES, dtype=np.float64)

        # --- Angular distribution (f1-f4) ---
        features[0] = self.circular_mean(per["bearings"])           # f1: mean bearing (rad)
        features[1] = self.circular_std(per["bearings"])            # f2: angular std (deg)
        features[2] = self.angular_range(per["bearings"])           # f3: angular range (deg)
        features[3] = self.count_quadrants(per["bearings"])         # f4: num quadrants

        # --- Distance distribution (f5-f8) ---
        features[4] = np.mean(per["distances"])                     # f5
        features[5] = np.std(per["distances"])                      # f6
        features[6] = np.min(per["distances"])                      # f7
        features[7] = np.max(per["distances"])                      # f8

        # --- Speed consistency (f9-f12) ---
        features[8] = np.mean(per["approach_speed"])                # f9
        features[9] = np.std(per["approach_speed"])                 # f10
        features[10] = np.mean(per["heading_alignment"])            # f11
        features[11] = np.std(per["heading_alignment"])             # f12

        # --- Time consistency (f13-f15) ---
        features[12] = np.mean(per["etas"])                         # f13
        features[13] = np.std(per["etas"])                          # f14
        features[14] = np.min(per["etas"])                          # f15

        # --- Density (f16-f18) ---
        features[15] = pw["mean_pairwise"]                          # f16
        features[16] = pw["std_pairwise"]                           # f17
        features[17] = pw["mean_nn"]                                # f18

        # --- Other (f19) ---
        features[18] = float(n)                                     # f19

        return features
