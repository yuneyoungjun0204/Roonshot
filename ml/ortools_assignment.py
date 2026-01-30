"""
OR-Tools Based Optimal Assignment Solver
========================================
Implements Linear Sum Assignment (Hungarian Algorithm) using Google OR-Tools
for optimal 1:1 pair-to-cluster mapping in USV defense scenarios.

Key Features:
  - Guaranteed 1:1 unique mapping (no duplicates)
  - Deterministic results (no LLM variability)
  - Uses effective distance formula: distance + |relative_angle| × 1.5
  - Supports ETA-based priority weighting
  - Fast computation for real-time tactical decisions

Reference: https://github.com/google/or-tools
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ml.constants import DEFENSE_CENTER
from ml.labels import FormationClass


@dataclass
class AssignmentResult:
    """Result of OR-Tools assignment optimization."""
    pair_assignments: Dict[int, int]  # pair_id -> cluster_id
    total_cost: float  # Total effective distance cost
    cost_matrix: np.ndarray  # Full cost matrix for debugging
    reasoning: str  # Human-readable explanation


def calculate_effective_distance(
    pair_pos: Tuple[float, float],
    pair_heading_deg: float,
    cluster_center: Tuple[float, float],
    angle_weight: float = 1.5,
) -> float:
    """
    Calculate effective distance using naval doctrine v2 formula.

    Formula: effective_distance = distance + |relative_angle| × angle_weight

    Args:
        pair_pos: (x, y) position of friendly pair center
        pair_heading_deg: Heading angle of pair in degrees
        cluster_center: (x, y) position of enemy cluster center
        angle_weight: Multiplier for angle penalty (default: 1.5 from doctrine)

    Returns:
        Effective distance (lower = better match)
    """
    # Direct distance
    dx = cluster_center[0] - pair_pos[0]
    dy = cluster_center[1] - pair_pos[1]
    distance = math.sqrt(dx * dx + dy * dy)

    # Bearing to cluster (Pygame Y-axis inverted)
    bearing_rad = math.atan2(-dy, dx)  # Negate dy for Pygame coords
    bearing_deg = math.degrees(bearing_rad)

    # Relative angle (difference between heading and bearing)
    relative_angle = bearing_deg - pair_heading_deg
    # Normalize to [-180, 180]
    while relative_angle > 180:
        relative_angle -= 360
    while relative_angle < -180:
        relative_angle += 360

    # Effective distance formula from naval doctrine v2
    # 10° angle difference = 15m distance equivalent (1.5 × 10 = 15)
    effective_dist = distance + abs(relative_angle) * angle_weight

    return effective_dist


def calculate_eta_factor(
    cluster_center: Tuple[float, float],
    cluster_velocity: Tuple[float, float],
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
    max_eta: float = 200.0,
) -> float:
    """
    Calculate ETA-based priority factor.

    Returns:
        Factor in [0, 1] where lower ETA = higher value (closer to 1)
    """
    vx, vy = cluster_velocity
    speed = math.sqrt(vx * vx + vy * vy)

    if speed < 1.0:
        return 0.5  # Stationary cluster, neutral priority

    # Distance to defense center
    dx = defense_center[0] - cluster_center[0]
    dy = defense_center[1] - cluster_center[1]
    dist = math.sqrt(dx * dx + dy * dy)

    # Unit vector to defense center
    ux, uy = dx / dist, dy / dist

    # Approach speed (velocity component toward defense center)
    approach_speed = vx * ux + vy * uy

    if approach_speed <= 0:
        return 0.0  # Moving away, lowest priority

    eta = dist / approach_speed
    # Normalize: lower ETA = higher factor (closer to 1)
    eta_factor = max(0.0, 1.0 - eta / max_eta)

    return eta_factor


def build_cost_matrix(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
    angle_weight: float = 1.5,
    eta_weight: float = 0.2,
) -> np.ndarray:
    """
    Build cost matrix for Linear Sum Assignment.

    Cost = effective_distance + eta_penalty

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        clusters: List of cluster dicts with 'cluster_id', 'center', 'velocity'
        formation: Formation type (affects ETA weighting)
        defense_center: Defense center position
        angle_weight: Weight for angle penalty in effective distance
        eta_weight: Weight for ETA factor (0-1)

    Returns:
        Cost matrix of shape (n_agents, n_clusters)
    """
    n_agents = len(agents)
    n_clusters = len(clusters)

    # Adjust ETA weight based on formation type
    if formation == FormationClass.CONCENTRATED:
        eta_weight = 0.2  # Focus on direction, not timing
    elif formation == FormationClass.WAVE:
        eta_weight = 0.4  # Timing more important for wave attacks
    else:  # DIVERSIONARY
        eta_weight = 0.2  # Global optimization

    cost_matrix = np.zeros((n_agents, n_clusters))

    for i, agent in enumerate(agents):
        pos = tuple(agent['pos'])
        heading = agent['angle']

        for j, cluster in enumerate(clusters):
            center = tuple(cluster['center'])
            velocity = tuple(cluster.get('velocity', (0, 0)))

            # Effective distance (primary cost)
            eff_dist = calculate_effective_distance(
                pos, heading, center, angle_weight
            )

            # ETA factor (secondary cost)
            eta_factor = calculate_eta_factor(
                center, velocity, defense_center
            )
            # Convert to penalty: lower ETA = lower penalty
            eta_penalty = (1.0 - eta_factor) * 500  # Scale to distance units

            # Combined cost
            cost_matrix[i, j] = eff_dist + eta_weight * eta_penalty

    return cost_matrix


def ortools_optimal_assignment(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass = FormationClass.DIVERSIONARY,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
) -> AssignmentResult:
    """
    Compute optimal 1:1 pair-to-cluster assignment using OR-Tools.

    This uses Google OR-Tools Linear Sum Assignment solver (Hungarian algorithm)
    to find the optimal assignment that minimizes total effective distance.

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        clusters: List of cluster dicts with 'cluster_id', 'center', 'velocity'
        formation: Formation type for ETA weighting
        defense_center: Defense center position

    Returns:
        AssignmentResult with optimal assignments
    """
    if not agents or not clusters:
        return AssignmentResult(
            pair_assignments={},
            total_cost=0.0,
            cost_matrix=np.array([]),
            reasoning="No agents or clusters to assign",
        )

    n_agents = len(agents)
    n_clusters = len(clusters)

    # Build cost matrix
    cost_matrix = build_cost_matrix(
        agents, clusters, formation, defense_center
    )

    # Pad to square matrix if needed
    size = max(n_agents, n_clusters)
    padded_cost = np.full((size, size), 1e9)
    padded_cost[:n_agents, :n_clusters] = cost_matrix

    try:
        # Try OR-Tools first
        from ortools.linear_solver import pywraplp

        # Create solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if solver is None:
            raise ImportError("SCIP solver not available")

        # Decision variables: x[i,j] = 1 if agent i assigned to cluster j
        x = {}
        for i in range(size):
            for j in range(size):
                x[i, j] = solver.BoolVar(f'x[{i},{j}]')

        # Constraint 1: Each agent assigned to exactly one cluster
        for i in range(size):
            solver.Add(solver.Sum([x[i, j] for j in range(size)]) == 1)

        # Constraint 2: Each cluster assigned to exactly one agent
        for j in range(size):
            solver.Add(solver.Sum([x[i, j] for i in range(size)]) == 1)

        # Objective: Minimize total cost
        objective = solver.Objective()
        for i in range(size):
            for j in range(size):
                objective.SetCoefficient(x[i, j], padded_cost[i, j])
        objective.SetMinimization()

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            # Extract assignments
            assignments = {}
            total_cost = 0.0

            for i in range(n_agents):
                for j in range(n_clusters):
                    if x[i, j].solution_value() > 0.5:
                        pair_id = agents[i]['id']
                        cluster_id = clusters[j]['cluster_id']
                        assignments[pair_id] = cluster_id
                        total_cost += cost_matrix[i, j]
                        break

            reasoning = (
                f"OR-Tools optimal assignment: {n_agents} pairs -> {n_clusters} clusters. "
                f"Total effective distance: {total_cost:.1f}"
            )

            return AssignmentResult(
                pair_assignments=assignments,
                total_cost=total_cost,
                cost_matrix=cost_matrix,
                reasoning=reasoning,
            )
        else:
            raise Exception(f"Solver status: {status}")

    except ImportError:
        # Fallback to scipy if OR-Tools not available
        print("[ORTools] OR-Tools not available, using scipy fallback")
        return scipy_optimal_assignment(
            agents, clusters, formation, defense_center, cost_matrix
        )
    except Exception as e:
        print(f"[ORTools] OR-Tools solver failed: {e}, using scipy fallback")
        return scipy_optimal_assignment(
            agents, clusters, formation, defense_center, cost_matrix
        )


def scipy_optimal_assignment(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass = FormationClass.DIVERSIONARY,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
    cost_matrix: Optional[np.ndarray] = None,
) -> AssignmentResult:
    """
    Fallback assignment using scipy's linear_sum_assignment (Hungarian algorithm).

    This is used when OR-Tools is not available.
    """
    from scipy.optimize import linear_sum_assignment

    if not agents or not clusters:
        return AssignmentResult(
            pair_assignments={},
            total_cost=0.0,
            cost_matrix=np.array([]),
            reasoning="No agents or clusters to assign",
        )

    n_agents = len(agents)
    n_clusters = len(clusters)

    # Build cost matrix if not provided
    if cost_matrix is None:
        cost_matrix = build_cost_matrix(
            agents, clusters, formation, defense_center
        )

    # Pad to square matrix if needed
    size = max(n_agents, n_clusters)
    padded_cost = np.full((size, size), 1e9)
    padded_cost[:n_agents, :n_clusters] = cost_matrix

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(padded_cost)

    # Build assignment map
    assignments = {}
    total_cost = 0.0

    for r, c in zip(row_ind, col_ind):
        if r < n_agents and c < n_clusters:
            pair_id = agents[r]['id']
            cluster_id = clusters[c]['cluster_id']
            assignments[pair_id] = cluster_id
            total_cost += cost_matrix[r, c]

    reasoning = (
        f"Scipy Hungarian assignment: {n_agents} pairs -> {n_clusters} clusters. "
        f"Total effective distance: {total_cost:.1f}"
    )

    return AssignmentResult(
        pair_assignments=assignments,
        total_cost=total_cost,
        cost_matrix=cost_matrix,
        reasoning=reasoning,
    )


def get_optimal_assignment(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass = FormationClass.DIVERSIONARY,
    confidence: float = 0.8,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
) -> Tuple[Dict[int, int], str]:
    """
    High-level function to get optimal pair-to-cluster assignment.

    Tries OR-Tools first, falls back to scipy if unavailable.

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        clusters: List of cluster dicts with 'cluster_id', 'center', 'velocity'
        formation: Formation type for ETA weighting
        confidence: Formation classification confidence (for logging)
        defense_center: Defense center position

    Returns:
        Tuple of (assignments dict, reasoning string)
    """
    result = ortools_optimal_assignment(
        agents, clusters, formation, defense_center
    )

    # Log assignment details
    print("\n" + "=" * 50)
    print("【OR-Tools Optimal Assignment】")
    print("=" * 50)
    print(f"Formation: {formation.name} (Confidence: {confidence:.1%})")
    print(f"Agents: {len(agents)}, Clusters: {len(clusters)}")
    print(f"Total Cost: {result.total_cost:.1f}")
    print("-" * 40)
    print("Assignments:")
    for pair_id, cluster_id in sorted(result.pair_assignments.items()):
        agent = next((a for a in agents if a['id'] == pair_id), None)
        cluster = next((c for c in clusters if c['cluster_id'] == cluster_id), None)
        if agent and cluster:
            eff_dist = result.cost_matrix[
                agents.index(agent), clusters.index(cluster)
            ]
            print(f"  Pair {pair_id} -> Cluster {cluster_id} (cost: {eff_dist:.1f})")
    print("=" * 50 + "\n")

    return result.pair_assignments, result.reasoning


# =============================================================================
# INTEGRATION WITH EXISTING SYSTEM
# =============================================================================

class AssignmentMode:
    """Assignment mode enumeration."""
    ORTOOLS = "ortools"              # Static: Lock assignment at 5km, never change
    ORTOOLS_DYNAMIC = "ortools_dynamic"  # Dynamic: Recalculate optimal assignment every frame
    LLM = "llm"
    SCIPY = "scipy"  # Fallback only


def get_tactical_assignment(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass,
    confidence: float = 0.8,
    mode: str = AssignmentMode.ORTOOLS,
    model_name: str = "qwen2.5:7b-instruct",
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
) -> Tuple[Dict[int, int], str]:
    """
    Unified tactical assignment function supporting multiple modes.

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        clusters: List of cluster dicts with 'cluster_id', 'center', 'velocity'
        formation: Formation type
        confidence: Formation classification confidence
        mode: Assignment mode ('ortools', 'llm', 'scipy')
        model_name: LLM model name (only used in LLM mode)
        defense_center: Defense center position

    Returns:
        Tuple of (assignments dict, reasoning string)
    """
    if mode == AssignmentMode.ORTOOLS:
        return get_optimal_assignment(
            agents, clusters, formation, confidence, defense_center
        )
    elif mode == AssignmentMode.LLM:
        # Import LLM commander only when needed
        from ml.llm_commander import get_tactical_command
        return get_tactical_command(
            agents, clusters, formation, confidence,
            use_llm=True, model_name=model_name
        )
    elif mode == AssignmentMode.SCIPY:
        result = scipy_optimal_assignment(
            agents, clusters, formation, defense_center
        )
        return result.pair_assignments, result.reasoning
    else:
        print(f"[Warning] Unknown mode '{mode}', using OR-Tools")
        return get_optimal_assignment(
            agents, clusters, formation, confidence, defense_center
        )


# =============================================================================
# DYNAMIC REASSIGNMENT (ORTOOLS_DYNAMIC MODE)
# =============================================================================

def compute_dynamic_assignment(
    agents: List[Dict],
    active_enemies: List[Dict],
    formation: FormationClass = FormationClass.DIVERSIONARY,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
    previous_assignments: Optional[Dict[int, int]] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, int], Dict[int, int], str]:
    """
    Compute optimal pair-to-enemy assignment dynamically.

    This is called every frame (or periodically) to recalculate the best
    assignment based on current positions of pairs and enemies.

    Key differences from static mode:
      - Assigns pairs directly to INDIVIDUAL ENEMIES (not clusters)
      - Recalculates every frame based on current positions
      - Returns both pair->enemy mapping and changes from previous frame

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        active_enemies: List of active enemy dicts with 'id', 'pos', 'velocity'
        formation: Formation type for ETA weighting
        defense_center: Defense center position
        previous_assignments: Previous frame's assignments for change detection
        verbose: Whether to print debug info

    Returns:
        Tuple of:
          - assignments: Dict[pair_id, enemy_id] - current optimal assignment
          - changes: Dict[pair_id, enemy_id] - pairs that changed assignment
          - reasoning: Human-readable explanation
    """
    if not agents or not active_enemies:
        return {}, {}, "No agents or enemies to assign"

    n_agents = len(agents)
    n_enemies = len(active_enemies)

    # Build cost matrix: pairs (rows) vs enemies (columns)
    cost_matrix = np.zeros((n_agents, n_enemies))

    # ETA weight based on formation
    if formation == FormationClass.CONCENTRATED:
        eta_weight = 0.2
    elif formation == FormationClass.WAVE:
        eta_weight = 0.4
    else:
        eta_weight = 0.2

    for i, agent in enumerate(agents):
        pos = tuple(agent['pos'])
        heading = agent['angle']

        for j, enemy in enumerate(active_enemies):
            enemy_pos = tuple(enemy['pos'])
            enemy_vel = tuple(enemy.get('velocity', (0, 0)))

            # Effective distance (primary cost)
            eff_dist = calculate_effective_distance(
                pos, heading, enemy_pos, angle_weight=1.5
            )

            # ETA factor (secondary cost)
            eta_factor = calculate_eta_factor(
                enemy_pos, enemy_vel, defense_center
            )
            eta_penalty = (1.0 - eta_factor) * 500

            cost_matrix[i, j] = eff_dist + eta_weight * eta_penalty

    # Pad to square matrix
    size = max(n_agents, n_enemies)
    padded_cost = np.full((size, size), 1e9)
    padded_cost[:n_agents, :n_enemies] = cost_matrix

    # Solve using scipy (faster for frequent calls)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(padded_cost)

    # Build assignment map: pair_id -> enemy_id
    assignments = {}
    total_cost = 0.0

    for r, c in zip(row_ind, col_ind):
        if r < n_agents and c < n_enemies:
            pair_id = agents[r]['id']
            enemy_id = active_enemies[c]['id']
            assignments[pair_id] = enemy_id
            total_cost += cost_matrix[r, c]

    # Detect changes from previous assignment
    changes = {}
    if previous_assignments is not None:
        for pair_id, enemy_id in assignments.items():
            prev_enemy = previous_assignments.get(pair_id)
            if prev_enemy != enemy_id:
                changes[pair_id] = enemy_id
                if verbose:
                    print(f"[DYNAMIC] Pair {pair_id}: Enemy {prev_enemy} -> Enemy {enemy_id}")

    # Build reasoning
    n_changes = len(changes)
    if n_changes > 0:
        reasoning = f"Dynamic reassignment: {n_changes} pairs changed targets. Total cost: {total_cost:.1f}"
    else:
        reasoning = f"Dynamic assignment stable. Total cost: {total_cost:.1f}"

    return assignments, changes, reasoning


def get_dynamic_assignment(
    agents: List[Dict],
    active_enemies: List['USV'],  # Actual USV objects
    formation: FormationClass = FormationClass.DIVERSIONARY,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
    previous_assignments: Optional[Dict[int, int]] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, int], Dict[int, int], str]:
    """
    High-level wrapper for dynamic assignment from USV objects.

    Converts USV objects to dict format and calls compute_dynamic_assignment.

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        active_enemies: List of active USV enemy objects
        formation: Formation type
        defense_center: Defense center position
        previous_assignments: Previous frame's pair->enemy mapping
        verbose: Whether to print debug info

    Returns:
        Tuple of (assignments, changes, reasoning)
    """
    # Convert USV objects to dict format
    enemy_dicts = []
    for e in active_enemies:
        enemy_dicts.append({
            'id': e.id,
            'pos': (e.x, e.y),
            'velocity': (e.vx, e.vy),
        })

    return compute_dynamic_assignment(
        agents, enemy_dicts, formation, defense_center,
        previous_assignments, verbose
    )
