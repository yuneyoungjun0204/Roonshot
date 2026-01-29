"""
Extended attack pattern generators for training data.

Generates enemy positions and velocities for each of the 3 formation classes
with controlled randomization to produce diverse training samples.

Patterns:
  - CONCENTRATED (집중): All enemies from one direction, tight cluster
  - WAVE (파상): Same direction, layered waves (1진→2진→3진) with distance gaps
  - DIVERSIONARY (양동): Enemies from random directions, irregular timing
"""

import math
import random
import numpy as np
from typing import Tuple, List

from ml.constants import DEFENSE_CENTER, ENEMY_SPEED, SPAWN_RADIUS_BASE, WAVE_SPAWN_RADIUS, WORLD_SIZE


def _clamp(val: float, lo: float = 0.0, hi: float = WORLD_SIZE) -> float:
    return max(lo, min(hi, val))


def generate_spearhead(
    num_enemies: int = 10,
    center: Tuple[float, float] = DEFENSE_CENTER,
    spawn_radius: float = SPAWN_RADIUS_BASE,
    speed: float = ENEMY_SPEED,
    angle_spread_deg: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate CONCENTRATED formation: narrow angle, dense approach.
    All enemies from one direction in a tight cluster.

    Returns:
        positions: (N, 2) array
        velocities: (N, 2) array
    """
    if angle_spread_deg is None:
        angle_spread_deg = random.uniform(5.0, 25.0)  # Keep under 30 deg

    base_angle = random.uniform(0, 2 * math.pi)
    spread_rad = math.radians(angle_spread_deg) / 2

    positions = []
    velocities = []

    for _ in range(num_enemies):
        angle = base_angle + random.uniform(-spread_rad, spread_rad)
        dist = spawn_radius + random.uniform(-300, 500)

        x = _clamp(center[0] + dist * math.cos(angle))
        y = _clamp(center[1] + dist * math.sin(angle))

        # Velocity toward center with small perturbation
        dx = center[0] - x
        dy = center[1] - y
        d = math.sqrt(dx * dx + dy * dy)
        if d > 0.1:
            s = speed * random.uniform(0.9, 1.1)
            vx = (dx / d) * s
            vy = (dy / d) * s
        else:
            vx, vy = 0.0, 0.0

        positions.append([x, y])
        velocities.append([vx, vy])

    return np.array(positions), np.array(velocities)


def generate_wave(
    num_enemies: int = 9,
    center: Tuple[float, float] = DEFENSE_CENTER,
    spawn_radius: float = WAVE_SPAWN_RADIUS,
    speed: float = ENEMY_SPEED,
    num_waves: int = None,
    wave_gap: float = None,
    formation_width_deg: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate WAVE formation: layered waves from the SAME direction.

    Visual shape: Sequential rows approaching like ocean waves.
      - 1st wave (closest to center) --- 3 ships side by side
      - 2nd wave (behind 1st)        --- 3 ships side by side
      - 3rd wave (furthest)          --- 3 ships side by side

    Key characteristics:
      - All waves share the same approach angle (narrow angular spread)
      - Waves are separated by distance (simulating time delay)
      - Within each wave, ships form a lateral line (perpendicular to approach)

    Args:
        num_enemies: total number of enemies
        center: defense center
        spawn_radius: base spawn distance for the 1st (closest) wave
        speed: approach speed
        num_waves: number of wave layers (default: 2~4)
        wave_gap: distance between waves in meters (default: 600~1200)
        formation_width_deg: lateral spread within each wave in degrees (default: 8~20)
    """
    if num_waves is None:
        num_waves = random.choice([2, 3, 3, 4])
    if wave_gap is None:
        wave_gap = random.uniform(600.0, 1200.0)
    if formation_width_deg is None:
        formation_width_deg = random.uniform(8.0, 20.0)

    # All waves approach from the same direction
    approach_angle = random.uniform(0, 2 * math.pi)
    lateral_spread_rad = math.radians(formation_width_deg) / 2

    enemies_per_wave = num_enemies // num_waves
    remainder = num_enemies % num_waves

    positions = []
    velocities = []

    for wave_idx in range(num_waves):
        # Each successive wave is further from center (time delay effect)
        wave_dist = spawn_radius + wave_idx * wave_gap
        wave_dist += random.uniform(-100, 100)  # small jitter

        # Number of enemies in this wave
        n_in_wave = enemies_per_wave + (1 if wave_idx < remainder else 0)

        for i in range(n_in_wave):
            # Lateral spread: distribute ships side by side within wave
            if n_in_wave > 1:
                frac = i / (n_in_wave - 1) - 0.5  # -0.5 to +0.5
                lateral_offset = frac * 2 * lateral_spread_rad
            else:
                lateral_offset = 0.0

            angle = approach_angle + lateral_offset
            angle += random.uniform(-0.02, 0.02)  # tiny jitter

            dist = wave_dist + random.uniform(-50, 50)

            x = _clamp(center[0] + dist * math.cos(angle))
            y = _clamp(center[1] + dist * math.sin(angle))

            # Velocity straight toward center
            dx = center[0] - x
            dy = center[1] - y
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0.1:
                s = speed * random.uniform(0.95, 1.05)
                vx = (dx / d) * s
                vy = (dy / d) * s
            else:
                vx, vy = 0.0, 0.0

            positions.append([x, y])
            velocities.append([vx, vy])

    return np.array(positions), np.array(velocities)


def generate_scattered(
    num_enemies: int = 10,
    center: Tuple[float, float] = DEFENSE_CENTER,
    spawn_radius: float = SPAWN_RADIUS_BASE,
    speed: float = ENEMY_SPEED,
    num_directions: int = 3,
    direction_spread_deg: float = None,
    cluster_spread_deg: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate DIVERSIONARY formation: 3 clustered groups from 3 directions (~120° apart).

    Each direction has a small cluster of 3-4 ships approaching together.
    This simulates a coordinated multi-directional attack.

    Visual shape:
      - Direction 1 (0°)   : 3-4 ships clustered together
      - Direction 2 (120°) : 3-4 ships clustered together
      - Direction 3 (240°) : 3-4 ships clustered together

    Args:
        num_enemies: total number of enemies (distributed across directions)
        center: defense center
        spawn_radius: base spawn distance
        speed: approach speed
        num_directions: number of attack directions (default: 3)
        direction_spread_deg: randomization in direction angles (default: 10-30°)
        cluster_spread_deg: spread within each cluster (default: 8-15°)
    """
    if direction_spread_deg is None:
        direction_spread_deg = random.uniform(10.0, 30.0)
    if cluster_spread_deg is None:
        cluster_spread_deg = random.uniform(8.0, 15.0)

    # Base angles: 120° apart (360° / 3 = 120°)
    base_angle_offset = random.uniform(0, 2 * math.pi)  # Random rotation
    direction_angles = []
    for i in range(num_directions):
        base_angle = base_angle_offset + (2 * math.pi * i / num_directions)
        # Add some randomization to direction (±direction_spread_deg/2)
        angle_jitter = math.radians(random.uniform(-direction_spread_deg/2, direction_spread_deg/2))
        direction_angles.append(base_angle + angle_jitter)

    # Distribute enemies across directions (3-4 per direction typically)
    enemies_per_direction = num_enemies // num_directions
    remainder = num_enemies % num_directions

    positions = []
    velocities = []

    cluster_spread_rad = math.radians(cluster_spread_deg) / 2

    for dir_idx, dir_angle in enumerate(direction_angles):
        # Number of enemies in this direction
        n_in_cluster = enemies_per_direction + (1 if dir_idx < remainder else 0)

        # Distance variation for this cluster (slight layering within cluster)
        cluster_base_dist = spawn_radius + random.uniform(-500, 500)

        for i in range(n_in_cluster):
            # Spread ships within the cluster (lateral spread)
            if n_in_cluster > 1:
                frac = i / (n_in_cluster - 1) - 0.5  # -0.5 to +0.5
                lateral_offset = frac * 2 * cluster_spread_rad
            else:
                lateral_offset = 0.0

            angle = dir_angle + lateral_offset
            angle += random.uniform(-0.03, 0.03)  # tiny jitter

            # Slight distance variation within cluster
            dist = cluster_base_dist + random.uniform(-200, 300)

            x = _clamp(center[0] + dist * math.cos(angle))
            y = _clamp(center[1] + dist * math.sin(angle))

            # Velocity toward center
            dx = center[0] - x
            dy = center[1] - y
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0.1:
                s = speed * random.uniform(0.9, 1.1)
                vx = (dx / d) * s
                vy = (dy / d) * s
            else:
                vx, vy = 0.0, 0.0

            positions.append([x, y])
            velocities.append([vx, vy])

    return np.array(positions), np.array(velocities)


# For backward compatibility
generate_encirclement = generate_wave


# -----------------------------------------------------------------------
# Variant generators for more diversity in training data
# -----------------------------------------------------------------------

def generate_spearhead_variants(
    num_samples: int = 100,
    enemies_range: Tuple[int, int] = (5, 15),
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate multiple CONCENTRATED samples with varying parameters."""
    samples = []
    for _ in range(num_samples):
        n = random.randint(*enemies_range)
        spread = random.uniform(5, 28)
        radius = SPAWN_RADIUS_BASE + random.uniform(-500, 1000)
        pos, vel = generate_spearhead(n, angle_spread_deg=spread, spawn_radius=radius)
        samples.append((pos, vel, 0))  # label = CONCENTRATED
    return samples


def generate_wave_variants(
    num_samples: int = 100,
    enemies_range: Tuple[int, int] = (6, 15),
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate multiple WAVE samples with varying parameters."""
    samples = []
    for _ in range(num_samples):
        n = random.randint(*enemies_range)
        num_waves = random.choice([2, 3, 4])
        wave_gap = random.uniform(500, 1500)
        radius = WAVE_SPAWN_RADIUS + random.uniform(-200, 500)
        pos, vel = generate_wave(n, num_waves=num_waves, wave_gap=wave_gap, spawn_radius=radius)
        samples.append((pos, vel, 1))  # label = WAVE
    return samples


def generate_scattered_variants(
    num_samples: int = 100,
    enemies_range: Tuple[int, int] = (5, 15),
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate multiple DIVERSIONARY samples."""
    samples = []
    for _ in range(num_samples):
        n = random.randint(*enemies_range)
        radius = SPAWN_RADIUS_BASE + random.uniform(-500, 1500)
        pos, vel = generate_scattered(n, spawn_radius=radius)
        samples.append((pos, vel, 2))  # label = DIVERSIONARY
    return samples
