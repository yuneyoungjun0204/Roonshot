"""
Data generation pipeline for formation classification training.

Generates snapshots at the moment enemies first cross the 5km boundary,
matching the inference-time behavior (classify once at 5km entry).

Output formats:
  - CSV: data/training_data/formation_dataset.csv
  - PyTorch tensors: data/training_data/formation_tensors.pt
"""

import os
import sys
import math
import random
import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.constants import (
    DEFENSE_CENTER,
    ENEMY_SPEED,
    SPAWN_RADIUS_BASE,
    DT,
    NUM_FEATURES,
    FEATURE_NAMES,
    FORMATION_LOCK_RADIUS,
    AUGMENT_TEMPORAL_STEPS,
    AUGMENT_TEMPORAL_RANGE,
)
from ml.labels import FormationClass, auto_label_snapshot
from ml.features import FeatureExtractor
from data.extended_patterns import (
    generate_spearhead,
    generate_encirclement,
    generate_scattered,
)


class HeadlessSimulation:
    """
    Minimal physics simulation without Pygame.
    Runs enemy movement toward the defense center.
    """

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        center: Tuple[float, float] = DEFENSE_CENTER,
        dt: float = DT,
    ):
        self.positions = positions.copy().astype(np.float64)
        self.velocities = velocities.copy().astype(np.float64)
        self.center = np.array(center, dtype=np.float64)
        self.dt = dt
        self.time = 0.0

    def step(self, num_steps: int = 1):
        """Advance simulation by num_steps time steps."""
        for _ in range(num_steps):
            self.positions += self.velocities * self.dt
            self.time += self.dt

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current (positions, velocities) copy."""
        return self.positions.copy(), self.velocities.copy()

    def get_active_mask(self, min_distance: float = 100.0) -> np.ndarray:
        """Return mask of enemies that haven't reached the center yet."""
        dx = self.positions[:, 0] - self.center[0]
        dy = self.positions[:, 1] - self.center[1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        return distances > min_distance

    def get_distances(self) -> np.ndarray:
        """Return distances from each enemy to the center."""
        dx = self.positions[:, 0] - self.center[0]
        dy = self.positions[:, 1] - self.center[1]
        return np.sqrt(dx ** 2 + dy ** 2)

    def any_within_radius(self, radius: float) -> bool:
        """Check if any enemy is within the given radius."""
        return bool(np.any(self.get_distances() <= radius))


def generate_5km_entry_samples(
    generator_fn,
    label: int,
    num_scenarios: int = 200,
    enemies_range: Tuple[int, int] = (5, 15),
    lock_radius: float = FORMATION_LOCK_RADIUS,
    max_sim_steps: int = 5000,
    temporal_augment: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """
    Generate labeled feature vectors at the 5km boundary crossing moment.

    For each scenario:
      1. Spawn enemies outside the lock radius
      2. Simulate until the first enemy crosses 5km
      3. Take multiple snapshots around that moment (temporal augmentation)
      4. Extract features and assign label

    Args:
        generator_fn: function(num_enemies) -> (positions, velocities)
        label: integer class label
        num_scenarios: number of scenarios to generate
        enemies_range: (min, max) enemies per scenario
        lock_radius: radius at which to capture snapshot (5000m)
        max_sim_steps: safety limit to prevent infinite loops
        temporal_augment: if True, capture multiple snapshots around lock moment

    Returns:
        List of (feature_vector_19d, label) tuples
    """
    extractor = FeatureExtractor()
    samples = []

    for _ in range(num_scenarios):
        n_enemies = random.randint(*enemies_range)
        pos, vel = generator_fn(n_enemies)

        sim = HeadlessSimulation(pos, vel)

        # Fast-forward until first enemy enters 5km radius
        crossed = False
        steps_to_lock = 0
        for step in range(max_sim_steps):
            if sim.any_within_radius(lock_radius):
                crossed = True
                steps_to_lock = step
                break
            sim.step(1)

        if not crossed:
            continue

        # === TEMPORAL AUGMENTATION ===
        # Capture snapshots at different time points around the lock moment
        if temporal_augment:
            # Go back to capture "before lock" snapshots
            time_offsets = []
            for i in range(AUGMENT_TEMPORAL_STEPS):
                # Spread snapshots: -RANGE, 0, +RANGE (approximately)
                offset = int((i - AUGMENT_TEMPORAL_STEPS // 2) * (AUGMENT_TEMPORAL_RANGE // max(1, AUGMENT_TEMPORAL_STEPS - 1)))
                time_offsets.append(offset)

            for offset in time_offsets:
                # Re-simulate to the target time point
                sim_copy = HeadlessSimulation(pos.copy(), vel.copy())
                target_step = max(0, steps_to_lock + offset)

                # Fast forward to target step
                if target_step > 0:
                    sim_copy.step(target_step)

                # Take snapshot
                snap_pos, snap_vel = sim_copy.snapshot()

                # Only use active enemies (not yet at center)
                mask = sim_copy.get_active_mask()
                if np.sum(mask) < 3:
                    continue

                active_pos = snap_pos[mask]
                active_vel = snap_vel[mask]

                features = extractor.extract(active_pos, active_vel)
                if features is not None:
                    samples.append((features, label))
        else:
            # Original behavior: single snapshot at lock moment
            snap_pos, snap_vel = sim.snapshot()

            mask = sim.get_active_mask()
            if np.sum(mask) < 3:
                continue

            active_pos = snap_pos[mask]
            active_vel = snap_vel[mask]

            features = extractor.extract(active_pos, active_vel)
            if features is not None:
                samples.append((features, label))

    return samples


def generate_full_dataset(
    samples_per_class: int = 1000,
    output_dir: str = "data/training_data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced dataset with all 3 formation classes.
    Each sample is captured at the 5km boundary crossing moment.

    Args:
        samples_per_class: number of samples per class
        output_dir: directory for saving output files

    Returns:
        (X, y) where X is (N, 19) features and y is (N,) labels
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1 snapshot per scenario now, so scenarios = samples_per_class
    scenarios = samples_per_class

    print(f"Generating {scenarios} scenarios per class (1 snapshot each at 5km entry)...")

    all_samples = []

    # CONCENTRATED (집중)
    print("  Generating CONCENTRATED (집중) samples...")
    concentrated = generate_5km_entry_samples(
        generate_spearhead, FormationClass.CONCENTRATED, scenarios
    )
    all_samples.extend(concentrated)
    print(f"    -> {len(concentrated)} samples")

    # WAVE (파상)
    print("  Generating WAVE (파상) samples...")
    wave = generate_5km_entry_samples(
        generate_encirclement, FormationClass.WAVE, scenarios
    )
    all_samples.extend(wave)
    print(f"    -> {len(wave)} samples")

    # DIVERSIONARY (양동)
    print("  Generating DIVERSIONARY (양동) samples...")
    diversionary = generate_5km_entry_samples(
        generate_scattered, FormationClass.DIVERSIONARY, scenarios
    )
    all_samples.extend(diversionary)
    print(f"    -> {len(diversionary)} samples")

    # Shuffle
    random.shuffle(all_samples)

    # Convert to arrays
    X = np.array([s[0] for s in all_samples], dtype=np.float32)
    y = np.array([s[1] for s in all_samples], dtype=np.int64)

    print(f"\nTotal dataset: {len(X)} samples")
    for c in range(3):
        print(f"  Class {c} ({FormationClass(c).name}): {np.sum(y == c)}")

    # Save CSV
    csv_path = os.path.join(output_dir, "formation_dataset.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = FEATURE_NAMES + ["label"]
        writer.writerow(header)
        for i in range(len(X)):
            row = list(X[i]) + [int(y[i])]
            writer.writerow(row)
    print(f"Saved CSV: {csv_path}")

    # Save PyTorch tensors
    try:
        import torch
        tensor_path = os.path.join(output_dir, "formation_tensors.pt")
        torch.save({
            "X": torch.tensor(X),
            "y": torch.tensor(y),
            "feature_names": FEATURE_NAMES,
            "class_names": [fc.name for fc in FormationClass],
        }, tensor_path)
        print(f"Saved tensors: {tensor_path}")
    except ImportError:
        print("PyTorch not available, skipping tensor output")

    return X, y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate formation training data")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Samples per class (default: 1000)")
    parser.add_argument("--output", type=str, default="data/training_data",
                        help="Output directory")
    args = parser.parse_args()

    X, y = generate_full_dataset(
        samples_per_class=args.samples,
        output_dir=args.output,
    )
    print(f"\nDone! Generated {len(X)} total samples.")
