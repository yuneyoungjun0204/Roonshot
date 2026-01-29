"""
DefenseMLSystem: top-level integration class for the ML defense pipeline.

Formation is classified ONCE when the first enemy enters the 5km radius,
then locked for the entire engagement.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from ml.constants import (
    DEFENSE_CENTER,
    TOTAL_FRIENDLY_PAIRS,
    MIN_ENEMIES_FOR_ML,
    NET_SPACING_BASE,
    FORMATION_LOCK_RADIUS,
)
from ml.features import FeatureExtractor
from ml.labels import FormationClass, auto_label_snapshot
from ml.density import DensityAnalyzer
from ml.classifier import FormationClassifier, load_classifier
from ml.tactical_mapper import TacticalMapper, TacticalDecision


class DefenseMLSystem:
    """
    Complete ML defense system that replaces FriendlyAI.assign_friendly_to_enemy().

    Formation classification is performed ONCE at the moment the first enemy
    enters the 5km radius from the defense center. The result is locked and
    used for the entire engagement.

    Density analysis (DBSCAN clustering, pair allocation) continues to update
    every frame since enemy positions change.

    Usage:
        system = DefenseMLSystem()
        system.load_model("models/formation_classifier.pt")

        # Each frame:
        decision = system.update(enemy_positions, enemy_velocities, pair_positions)
    """

    def __init__(
        self,
        center: Tuple[float, float] = DEFENSE_CENTER,
        total_pairs: int = TOTAL_FRIENDLY_PAIRS,
        lock_radius: float = FORMATION_LOCK_RADIUS,
    ):
        self.center = np.array(center, dtype=np.float64)
        self.total_pairs = total_pairs
        self.lock_radius = lock_radius
        self.mapper = TacticalMapper(center, total_pairs)
        self.last_decision: Optional[TacticalDecision] = None
        self._model_loaded = False

        # Formation lock state
        self._formation_locked = False
        self._locked_formation: Optional[FormationClass] = None
        self._locked_confidence: float = 0.0

    def load_model(
        self,
        path: str = "models/formation_classifier.pt",
        scaler_path: str = "models/feature_scaler.npz",
    ) -> bool:
        """
        Load trained classifier and its feature scaler. Returns True on success.
        If loading fails, the system uses rule-based classification.
        """
        model = load_classifier(path, scaler_path)
        if model is not None:
            self.mapper.set_classifier(model)
            self._model_loaded = True
            return True
        self._model_loaded = False
        return False

    @property
    def has_model(self) -> bool:
        return self._model_loaded

    @property
    def formation_locked(self) -> bool:
        return self._formation_locked

    def _any_enemy_within_radius(self, enemy_positions: np.ndarray) -> bool:
        """Check if any enemy is within the lock radius."""
        if len(enemy_positions) == 0:
            return False
        dx = enemy_positions[:, 0] - self.center[0]
        dy = enemy_positions[:, 1] - self.center[1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        return bool(np.any(distances <= self.lock_radius))

    def _try_lock_formation(
        self,
        enemy_positions: np.ndarray,
        enemy_velocities: np.ndarray,
    ):
        """
        If not yet locked and an enemy is within 5km, classify and lock.
        """
        if self._formation_locked:
            return

        if not self._any_enemy_within_radius(enemy_positions):
            return

        # Lock now: classify using all current enemies
        fc, confidence = self.mapper.classify_formation(
            enemy_positions, enemy_velocities
        )
        self._locked_formation = fc
        self._locked_confidence = confidence
        self._formation_locked = True

    def update(
        self,
        enemy_positions: np.ndarray,
        enemy_velocities: np.ndarray,
        pair_positions: np.ndarray,
    ) -> TacticalDecision:
        """
        Compute tactical decision for the current frame.

        Formation: classified once at 5km entry, then locked.
        Density/clustering: updated every frame.
        """
        # Try to lock formation if not yet locked
        self._try_lock_formation(enemy_positions, enemy_velocities)

        # Compute decision with locked formation override
        decision = self.mapper.compute_decision(
            enemy_positions,
            enemy_velocities,
            pair_positions,
            locked_formation=self._locked_formation,
            locked_confidence=self._locked_confidence,
        )
        self.last_decision = decision
        return decision

    def get_enemy_assignment(
        self, pair_index: int
    ) -> Optional[int]:
        """Get the enemy index assigned to a specific pair."""
        if self.last_decision is None:
            return None
        return self.last_decision.pair_target_enemy.get(pair_index)

    def get_net_spacing(self, pair_index: int) -> float:
        """Get recommended net spacing for a pair."""
        if self.last_decision is None:
            return NET_SPACING_BASE
        return self.last_decision.pair_net_spacing.get(pair_index, NET_SPACING_BASE)

    def get_formation_info(self) -> Tuple[str, float]:
        """Get current formation classification and confidence."""
        if self.last_decision is None:
            return "UNKNOWN", 0.0
        return self.last_decision.formation_class.name, self.last_decision.confidence

    def reset(self):
        """Reset formation lock for a new engagement."""
        self._formation_locked = False
        self._locked_formation = None
        self._locked_confidence = 0.0
        self.last_decision = None
