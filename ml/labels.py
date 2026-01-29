"""
Formation classification labels and auto-labeling logic.

Three formation classes matching the original attack patterns:
  0 - CONCENTRATED (집중): Narrow angle (<30°), dense approach
  1 - WAVE (파상):         Wide angle (>60°), synchronized arrival (ETA std <15s)
  2 - DIVERSIONARY (양동):  Wide angle, irregular arrival times
"""

import math
import numpy as np
from enum import IntEnum
from typing import List, Tuple, Optional

from ml.constants import (
    DEFENSE_CENTER,
    SPEARHEAD_ANGLE_THRESHOLD,
    WAVE_DISTANCE_STD_THRESHOLD,
    MIN_ENEMIES_FOR_ML,
)


class FormationClass(IntEnum):
    CONCENTRATED = 0   # 집중
    WAVE = 1           # 파상
    DIVERSIONARY = 2   # 양동


# Display names for UI
FORMATION_NAMES = {
    FormationClass.CONCENTRATED: "CONCENTRATED",
    FormationClass.WAVE: "WAVE",
    FormationClass.DIVERSIONARY: "DIVERSIONARY",
}

# Recommended defense response
FORMATION_DEFENSE = {
    FormationClass.CONCENTRATED: "해당 구역에 2~3개 페어 집중 배치",
    FormationClass.WAVE: "5개 페어 전체 분산 배치",
    FormationClass.DIVERSIONARY: "순차적 대응, 페어 재활용",
}


def _circular_std(angles_rad: np.ndarray) -> float:
    """
    Compute circular standard deviation for angular data.
    Handles 2*pi wraparound correctly.

    Returns value in degrees.
    """
    if len(angles_rad) < 2:
        return 0.0
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    R = min(R, 1.0)  # clamp for numerical safety
    circular_var = 1.0 - R
    # circular std in radians, then convert to degrees
    circ_std_rad = np.sqrt(-2.0 * np.log(max(R, 1e-10)))
    return np.degrees(circ_std_rad)


def _compute_bearings(
    positions: np.ndarray, center: Tuple[float, float]
) -> np.ndarray:
    """Compute bearing angles (radians) from center to each position."""
    dx = positions[:, 0] - center[0]
    dy = positions[:, 1] - center[1]
    return np.arctan2(dy, dx)


def _compute_etas(
    positions: np.ndarray,
    velocities: np.ndarray,
    center: Tuple[float, float],
) -> np.ndarray:
    """
    Estimate time-of-arrival for each enemy to the defense center.
    Uses radial approach speed (component of velocity toward center).
    """
    dx = center[0] - positions[:, 0]
    dy = center[1] - positions[:, 1]
    distances = np.sqrt(dx ** 2 + dy ** 2)

    # Unit vectors toward center
    unit_dx = np.where(distances > 0.1, dx / distances, 0.0)
    unit_dy = np.where(distances > 0.1, dy / distances, 0.0)

    # Radial speed (positive = approaching)
    radial_speed = velocities[:, 0] * unit_dx + velocities[:, 1] * unit_dy
    radial_speed = np.maximum(radial_speed, 1.0)  # avoid division by zero

    return distances / radial_speed


def auto_label_snapshot(
    positions: np.ndarray,
    velocities: np.ndarray,
    center: Tuple[float, float] = DEFENSE_CENTER,
) -> Optional[FormationClass]:
    """
    Automatically label a snapshot of enemy positions/velocities.

    Classification rules:
      - angular_std < 30° AND distance_std < 400m → CONCENTRATED (집중)
        Tight cluster from one direction
      - angular_std < 30° AND distance_std >= 400m → WAVE (파상)
        Layered waves from same direction (sequential rows separated by distance)
      - angular_std >= 30° → DIVERSIONARY (양동)
        Enemies from multiple directions

    Args:
        positions: (N, 2) array of enemy (x, y) positions
        velocities: (N, 2) array of enemy (vx, vy) velocities
        center: defense center coordinates

    Returns:
        FormationClass or None if too few enemies for classification
    """
    n = len(positions)
    if n < MIN_ENEMIES_FOR_ML:
        return None

    # 1. Compute angular spread
    bearings = _compute_bearings(positions, center)
    ang_std = _circular_std(bearings)

    # 2. Compute distance distribution
    dx = positions[:, 0] - center[0]
    dy = positions[:, 1] - center[1]
    distances = np.sqrt(dx ** 2 + dy ** 2)
    dist_std = np.std(distances)

    # 3. Classification rules
    if ang_std < SPEARHEAD_ANGLE_THRESHOLD:
        # Narrow approach: distinguish CONCENTRATED vs WAVE by distance layering
        if dist_std >= WAVE_DISTANCE_STD_THRESHOLD:
            return FormationClass.WAVE          # Layered waves, same direction
        else:
            return FormationClass.CONCENTRATED  # Tight cluster, same direction
    else:
        return FormationClass.DIVERSIONARY      # Wide angular spread
