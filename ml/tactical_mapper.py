"""
Tactical mapper: combines classification + density analysis to produce
pair-to-cluster assignments using the Hungarian algorithm.

Pipeline:
  1. Classify formation → determines strategic posture
  2. Analyze density → cluster enemies, compute metrics
  3. Allocate pairs to clusters → based on formation + density
  4. Within each cluster, assign pairs to specific enemies
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ml.constants import (
    DEFENSE_CENTER,
    TOTAL_FRIENDLY_PAIRS,
    CONFIDENCE_THRESHOLD,
    MIN_ENEMIES_FOR_ML,
)
from ml.labels import FormationClass, auto_label_snapshot
from ml.features import FeatureExtractor
from ml.density import DensityAnalyzer, ClusterMetrics


from dataclasses import dataclass


@dataclass
class TacticalDecision:
    """Output of the tactical mapping process."""
    formation_class: FormationClass
    confidence: float
    cluster_metrics: List[ClusterMetrics]
    # Maps cluster_id → list of pair indices (0-based into friendly_pairs list)
    pair_assignments: Dict[int, List[int]]
    # Maps pair_index → target enemy index (into active enemies array)
    pair_target_enemy: Dict[int, int]
    # Per-pair recommended net spacing
    pair_net_spacing: Dict[int, float]


class TacticalMapper:
    """
    Combines formation classification and density analysis to produce
    tactical pair assignments.
    """

    def __init__(
        self,
        center: Tuple[float, float] = DEFENSE_CENTER,
        total_pairs: int = TOTAL_FRIENDLY_PAIRS,
    ):
        self.center = np.array(center, dtype=np.float64)
        self.total_pairs = total_pairs
        self.feature_extractor = FeatureExtractor(center)
        self.density_analyzer = DensityAnalyzer()
        self.classifier = None  # set via set_classifier()

    def set_classifier(self, classifier):
        """Set the trained PyTorch classifier (FormationClassifier)."""
        self.classifier = classifier

    def classify_formation(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Tuple[FormationClass, float]:
        """
        Classify the enemy formation.
        Uses trained classifier if available; falls back to rule-based.
        """
        n = len(positions)
        if n < MIN_ENEMIES_FOR_ML:
            return FormationClass.DIVERSIONARY, 0.0

        # Try ML classifier
        if self.classifier is not None:
            features = self.feature_extractor.extract(positions, velocities)
            if features is not None:
                fc, conf = self.classifier.predict(features)
                if conf >= CONFIDENCE_THRESHOLD:
                    return fc, conf

        # Rule-based fallback
        label = auto_label_snapshot(positions, velocities, tuple(self.center))
        if label is not None:
            return label, 1.0
        return FormationClass.DIVERSIONARY, 0.0

    def _assign_pairs_to_clusters(
        self,
        cluster_metrics: List[ClusterMetrics],
        pair_positions: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Assign friendly pairs to clusters using the Hungarian algorithm.

        Args:
            cluster_metrics: list of ClusterMetrics
            pair_positions: (P, 2) array of pair center positions

        Returns:
            Dict mapping cluster_id → list of assigned pair indices
        """
        if not cluster_metrics or len(pair_positions) == 0:
            return {}

        # Get pair allocation per cluster
        allocation = self.density_analyzer.recommend_pairs_allocation(
            cluster_metrics, self.total_pairs
        )

        # Build cost matrix: rows = pair slots, cols = pairs
        # Expand clusters into slots based on allocation
        slots = []  # (cluster_id, slot_index)
        for m in cluster_metrics:
            n_pairs = allocation.get(m.cluster_id, 0)
            for s in range(n_pairs):
                slots.append(m.cluster_id)

        n_slots = len(slots)
        n_pairs = len(pair_positions)

        if n_slots == 0:
            return {}

        # Build cost matrix: distance from each pair to each cluster center
        cluster_centers = np.array([
            m.center for m in cluster_metrics
            for _ in range(allocation.get(m.cluster_id, 0))
        ])

        if len(cluster_centers) == 0:
            return {}

        # Pad to square matrix if needed
        size = max(n_slots, n_pairs)
        cost_matrix = np.full((size, size), 1e9)

        if n_slots > 0 and n_pairs > 0:
            real_costs = cdist(cluster_centers[:n_slots], pair_positions[:n_pairs])
            cost_matrix[:n_slots, :n_pairs] = real_costs

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build assignment map
        result: Dict[int, List[int]] = {m.cluster_id: [] for m in cluster_metrics}
        for r, c in zip(row_ind, col_ind):
            if r < n_slots and c < n_pairs:
                cid = slots[r]
                result[cid].append(int(c))

        return result

    def _assign_pairs_to_enemies(
        self,
        cluster_metrics: List[ClusterMetrics],
        pair_cluster_map: Dict[int, List[int]],
        enemy_positions: np.ndarray,
        pair_positions: np.ndarray,
    ) -> Dict[int, int]:
        """
        Within each cluster, assign each pair to a specific enemy.
        Uses Hungarian within each cluster.

        Returns:
            Dict mapping pair_index → enemy_index (in original array)
        """
        pair_targets: Dict[int, int] = {}

        for m in cluster_metrics:
            assigned_pairs = pair_cluster_map.get(m.cluster_id, [])
            if not assigned_pairs or not m.enemy_indices:
                continue

            # Cluster enemy positions
            e_idx = np.array(m.enemy_indices)
            e_pos = enemy_positions[e_idx]

            p_idx = np.array(assigned_pairs)
            p_pos = pair_positions[p_idx]

            # Cost matrix: pair → enemy distance
            n_e = len(e_idx)
            n_p = len(p_idx)
            size = max(n_e, n_p)
            cost = np.full((size, size), 1e9)
            real_costs = cdist(p_pos[:n_p], e_pos[:n_e])
            cost[:n_p, :n_e] = real_costs

            r_ind, c_ind = linear_sum_assignment(cost)

            for r, c in zip(r_ind, c_ind):
                if r < n_p and c < n_e:
                    pair_targets[int(p_idx[r])] = int(e_idx[c])

        return pair_targets

    def compute_decision(
        self,
        enemy_positions: np.ndarray,
        enemy_velocities: np.ndarray,
        pair_positions: np.ndarray,
        locked_formation: Optional[FormationClass] = None,
        locked_confidence: float = 0.0,
    ) -> TacticalDecision:
        """
        Full tactical decision pipeline.

        Args:
            enemy_positions: (E, 2) active enemy positions
            enemy_velocities: (E, 2) active enemy velocities
            pair_positions: (P, 2) friendly pair center positions
            locked_formation: if set, use this instead of re-classifying
            locked_confidence: confidence of the locked formation

        Returns:
            TacticalDecision with all assignments
        """
        # 1. Classify (use locked formation if available)
        if locked_formation is not None:
            fc, confidence = locked_formation, locked_confidence
        else:
            fc, confidence = self.classify_formation(enemy_positions, enemy_velocities)

        # 2. Density analysis
        labels, metrics = self.density_analyzer.analyze(enemy_positions)

        # 3. Pair → cluster assignment
        pair_cluster = self._assign_pairs_to_clusters(metrics, pair_positions)

        # 4. Pair → enemy assignment
        pair_enemy = self._assign_pairs_to_enemies(
            metrics, pair_cluster, enemy_positions, pair_positions
        )

        # 5. Net spacing per pair
        pair_spacing: Dict[int, float] = {}
        for m in metrics:
            for pidx in pair_cluster.get(m.cluster_id, []):
                pair_spacing[pidx] = m.net_spacing

        return TacticalDecision(
            formation_class=fc,
            confidence=confidence,
            cluster_metrics=metrics,
            pair_assignments=pair_cluster,
            pair_target_enemy=pair_enemy,
            pair_net_spacing=pair_spacing,
        )
