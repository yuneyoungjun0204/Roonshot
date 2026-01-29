"""
ML-Integrated USV Simulator with LLM Tactical Command
======================================================
Extends the original usv_simulator.py with:
  - ML-driven formation classification
  - LLM-based tactical pair-to-cluster mapping (RAG + naval doctrine)
  - 4-direction spawn around mothership, stationary until 5km lock
  - Dynamic per-pair net_spacing from DensityAnalyzer

Flow:
  1. Friendly USVs spawn at 4 cardinal directions, facing outward, STATIONARY
  2. Enemies approach the defense center
  3. At 5km lock: ML classifies formation, LLM decides pair-cluster mapping
  4. Each pair receives assignment and begins moving toward target cluster

Requirements: pygame, numpy, pandas, torch, scikit-learn, scipy, ollama (optional)
Run: python usv_simulator_ml.py
"""

import pygame
import numpy as np
import math
import random
from typing import List, Tuple, Optional, Dict

# Import original components
from usv_simulator import (
    CONFIG,
    AttackPattern,
    USV,
    CaptureEvent,
    SimulationLog,
    PhysicsEngine,
    AttackPatternGenerator,
    FriendlyAI,
    Renderer,
)

# Import ML system
from ml.inference import DefenseMLSystem
from ml.labels import FormationClass, FORMATION_NAMES
from ml.constants import (
    DEFENSE_CENTER, NET_SPACING_BASE, MISSED_THRESHOLD, ESCAPE_DISTANCE_THRESHOLD,
    RETURNING_HOME_CLUSTER_ID, HOME_ARRIVAL_THRESHOLD,
    POST_CAPTURE_DISTANCE_THRESHOLD, POST_CAPTURE_ANGLE_THRESHOLD,
)

# Import LLM Commander
from ml.llm_commander import LLMCommander, rule_based_mapping, get_tactical_command

# Import OR-Tools Assignment (default mode)
from ml.ortools_assignment import (
    AssignmentMode,
    get_tactical_assignment,
    get_optimal_assignment,
)


class MLRenderer(Renderer):
    """Extended renderer with ML visualization overlays."""

    def draw_cluster_overlay(
        self,
        enemy_positions: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_colors: List[Tuple[int, int, int]],
    ):
        """Draw colored circles around enemies and convex hull boundaries per cluster."""
        if len(enemy_positions) == 0:
            return

        # Per-enemy circles
        for i, (pos, label) in enumerate(zip(enemy_positions, cluster_labels)):
            if label < 0:
                continue
            color = cluster_colors[label % len(cluster_colors)]
            screen_pos = self.world_to_screen(float(pos[0]), float(pos[1]))
            radius = 18
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, 80), (radius, radius), radius)
            pygame.draw.circle(surf, (*color, 150), (radius, radius), radius, 2)
            self.screen.blit(surf, (screen_pos[0] - radius, screen_pos[1] - radius))

        # Cluster boundary polygons (convex hull)
        self._draw_cluster_boundaries(enemy_positions, cluster_labels, cluster_colors)

    def _draw_cluster_boundaries(
        self,
        enemy_positions: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_colors: List[Tuple[int, int, int]],
    ):
        """Draw convex hull boundaries and filled regions for each cluster."""
        from scipy.spatial import ConvexHull

        unique_labels = set(cluster_labels)
        padding = 120.0  # world-unit padding around hull

        for cid in unique_labels:
            if cid < 0:
                continue
            mask = cluster_labels == cid
            cluster_pos = enemy_positions[mask]
            color = cluster_colors[cid % len(cluster_colors)]
            n = len(cluster_pos)

            if n == 1:
                # Single point: draw a padded circle
                sp = self.world_to_screen(float(cluster_pos[0][0]), float(cluster_pos[0][1]))
                r = self.scale_distance(padding)
                r = max(r, 22)
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*color, 25), (r, r), r)
                pygame.draw.circle(surf, (*color, 120), (r, r), r, 2)
                self.screen.blit(surf, (sp[0] - r, sp[1] - r))

            elif n == 2:
                # Two points: draw a rounded rectangle (capsule) via thick line + end circles
                p1 = cluster_pos[0]
                p2 = cluster_pos[1]
                sp1 = self.world_to_screen(float(p1[0]), float(p1[1]))
                sp2 = self.world_to_screen(float(p2[0]), float(p2[1]))
                r = self.scale_distance(padding)
                r = max(r, 18)

                # Filled capsule via transparent surface
                dx = sp2[0] - sp1[0]
                dy = sp2[1] - sp1[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1:
                    length = 1
                nx = -dy / length * r
                ny = dx / length * r

                points = [
                    (sp1[0] + nx, sp1[1] + ny),
                    (sp2[0] + nx, sp2[1] + ny),
                    (sp2[0] - nx, sp2[1] - ny),
                    (sp1[0] - nx, sp1[1] - ny),
                ]

                # Draw on full-screen transparent surface
                overlay = pygame.Surface(
                    (self.config["screen_width"], self.config["screen_height"]),
                    pygame.SRCALPHA,
                )
                pygame.draw.polygon(overlay, (*color, 25), points)
                pygame.draw.polygon(overlay, (*color, 120), points, 2)
                pygame.draw.circle(overlay, (*color, 25), sp1, r)
                pygame.draw.circle(overlay, (*color, 120), sp1, r, 2)
                pygame.draw.circle(overlay, (*color, 25), sp2, r)
                pygame.draw.circle(overlay, (*color, 120), sp2, r, 2)
                self.screen.blit(overlay, (0, 0))

            else:
                # 3+ points: convex hull with padding
                try:
                    hull = ConvexHull(cluster_pos)
                except Exception:
                    continue

                hull_pts = cluster_pos[hull.vertices]
                center = np.mean(hull_pts, axis=0)

                # Expand hull outward by padding
                expanded = []
                for pt in hull_pts:
                    direction = pt - center
                    dist = np.linalg.norm(direction)
                    if dist > 0:
                        direction = direction / dist
                    expanded_pt = pt + direction * padding
                    sp = self.world_to_screen(float(expanded_pt[0]), float(expanded_pt[1]))
                    expanded.append(sp)

                if len(expanded) >= 3:
                    overlay = pygame.Surface(
                        (self.config["screen_width"], self.config["screen_height"]),
                        pygame.SRCALPHA,
                    )
                    pygame.draw.polygon(overlay, (*color, 25), expanded)
                    pygame.draw.polygon(overlay, (*color, 120), expanded, 2)
                    self.screen.blit(overlay, (0, 0))

    def draw_ml_panel(
        self,
        formation_name: str,
        confidence: float,
        num_clusters: int,
        has_model: bool,
        formation_locked: bool = False,
    ):
        """Draw ML status panel on the left side."""
        panel_width = 280
        panel_height = 165
        panel_x = 20
        panel_y = 20

        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surf.fill(self.config["ui_bg_color"])
        border_color = (60, 150, 60) if formation_locked else (60, 90, 60)
        pygame.draw.rect(panel_surf, border_color, (0, 0, panel_width, panel_height), 2)
        self.screen.blit(panel_surf, (panel_x, panel_y))

        # Title
        title = self.font_medium.render("ML Defense System", True, (100, 255, 150))
        self.screen.blit(title, (panel_x + 15, panel_y + 10))

        y_off = 50
        line_h = 24

        # Model status
        model_status = "Neural Net" if has_model else "Rule-Based"
        model_color = (100, 255, 100) if has_model else (255, 255, 100)
        text = self.font_small.render(f"Mode: {model_status}", True, model_color)
        self.screen.blit(text, (panel_x + 15, panel_y + y_off))

        # Formation class + lock status
        if formation_locked:
            lock_label = f"Formation: {formation_name} [LOCKED]"
            lock_color = (100, 255, 200)
        else:
            lock_label = f"Formation: {formation_name}"
            lock_color = (255, 200, 100)
        text = self.font_small.render(lock_label, True, lock_color)
        self.screen.blit(text, (panel_x + 15, panel_y + y_off + line_h))

        # Confidence
        conf_color = (100, 255, 100) if confidence > 0.8 else (255, 255, 100) if confidence > 0.6 else (255, 150, 100)
        text = self.font_small.render(f"Confidence: {confidence:.1%}", True, conf_color)
        self.screen.blit(text, (panel_x + 15, panel_y + y_off + line_h * 2))

        # Clusters
        text = self.font_small.render(f"Clusters: {num_clusters}", True, self.config["text_color"])
        self.screen.blit(text, (panel_x + 15, panel_y + y_off + line_h * 3))

        # Lock radius info
        if not formation_locked:
            text = self.font_small.render("Waiting for 5km entry...", True, (180, 180, 100))
            self.screen.blit(text, (panel_x + 15, panel_y + y_off + line_h * 4))

    def draw_assignment_lines(
        self,
        friendly_pairs: List[Tuple],  # List of (f1, f2) USV pairs
        pair_assignments: Dict[int, int],  # pair_idx -> cluster_id
        cluster_centers: Dict[int, Tuple[float, float]],  # cluster_id -> (x, y)
        cluster_colors: List[Tuple[int, int, int]],
    ):
        """
        Draw lines connecting friendly pairs to their assigned cluster centers.

        Args:
            friendly_pairs: List of (USV, USV) tuples for each pair
            pair_assignments: Mapping from pair index to cluster ID
            cluster_centers: Mapping from cluster ID to (x, y) center position
            cluster_colors: List of colors for each cluster
        """
        if not pair_assignments or not cluster_centers:
            return

        for pair_idx, (f1, f2) in enumerate(friendly_pairs):
            assigned_cluster_id = pair_assignments.get(pair_idx)
            if assigned_cluster_id is None:
                continue

            cluster_center = cluster_centers.get(int(assigned_cluster_id))
            if cluster_center is None:
                continue

            # Get pair center
            pair_center_x = (f1.x + f2.x) / 2
            pair_center_y = (f1.y + f2.y) / 2

            # Convert to screen coordinates
            pair_screen = self.world_to_screen(pair_center_x, pair_center_y)
            cluster_screen = self.world_to_screen(cluster_center[0], cluster_center[1])

            # Get color for this cluster
            color = cluster_colors[int(assigned_cluster_id) % len(cluster_colors)]

            # Draw dashed line from pair center to cluster center
            self._draw_dashed_line(
                pair_screen, cluster_screen, color, dash_length=10, gap_length=6
            )

            # Draw pair index label at pair center
            label = self.font_small.render(f"P{pair_idx}", True, color)
            self.screen.blit(label, (pair_screen[0] - 10, pair_screen[1] - 25))

            # Draw cluster ID label at cluster center
            label = self.font_small.render(f"C{assigned_cluster_id}", True, color)
            self.screen.blit(label, (cluster_screen[0] - 10, cluster_screen[1] - 25))

    def draw_target_lines(
        self,
        friendly_pairs: List[Tuple],  # List of (f1, f2) USV pairs
        pair_targets: Dict[int, 'USV'],  # pair_idx -> current target enemy
        pair_assignments: Dict[int, int],  # pair_idx -> cluster_id (for colors)
        cluster_colors: List[Tuple[int, int, int]],
    ):
        """
        Draw lines connecting friendly pairs to their CURRENT target enemies.
        Lines update every frame as targets move or change.

        Args:
            friendly_pairs: List of (USV, USV) tuples for each pair
            pair_targets: Mapping from pair index to current target USV (updated each frame)
            pair_assignments: Mapping from pair index to cluster ID (for color selection)
            cluster_colors: List of colors for each cluster
        """
        if not pair_targets:
            return

        for pair_idx, (f1, f2) in enumerate(friendly_pairs):
            target_enemy = pair_targets.get(pair_idx)
            if target_enemy is None or not target_enemy.is_active:
                continue

            # Get assigned cluster for color
            assigned_cluster_id = pair_assignments.get(pair_idx, 0)

            # Get pair center
            pair_center_x = (f1.x + f2.x) / 2
            pair_center_y = (f1.y + f2.y) / 2

            # Target is the current enemy position (updates every frame!)
            target_x = target_enemy.x
            target_y = target_enemy.y

            # Convert to screen coordinates
            pair_screen = self.world_to_screen(pair_center_x, pair_center_y)
            target_screen = self.world_to_screen(target_x, target_y)

            # Get color for this cluster
            color = cluster_colors[int(assigned_cluster_id) % len(cluster_colors)]

            # Draw dashed line from pair center to target enemy
            self._draw_dashed_line(
                pair_screen, target_screen, color, dash_length=10, gap_length=6
            )

            # Draw pair index label at pair center
            label = self.font_small.render(f"P{pair_idx}", True, color)
            self.screen.blit(label, (pair_screen[0] - 10, pair_screen[1] - 25))

    def _draw_dashed_line(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        dash_length: int = 10,
        gap_length: int = 5,
    ):
        """Draw a dashed line between two points."""
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1:
            return

        # Normalize direction
        dx /= distance
        dy /= distance

        # Draw dashes
        current_dist = 0
        drawing = True

        while current_dist < distance:
            segment_length = dash_length if drawing else gap_length
            next_dist = min(current_dist + segment_length, distance)

            if drawing:
                start_x = int(x1 + dx * current_dist)
                start_y = int(y1 + dy * current_dist)
                end_x = int(x1 + dx * next_dist)
                end_y = int(y1 + dy * next_dist)
                pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 2)

            current_dist = next_dist
            drawing = not drawing


# Cluster colors for visualization
CLUSTER_COLORS = [
    (255, 100, 100),   # Red
    (100, 255, 100),   # Green
    (100, 100, 255),   # Blue
    (255, 255, 100),   # Yellow
    (255, 100, 255),   # Magenta
    (100, 255, 255),   # Cyan
    (255, 180, 100),   # Orange
    (180, 100, 255),   # Purple
]


class MLSimulation:
    """Main simulation controller with ML-driven defense and tactical command."""

    def __init__(
        self,
        attack_pattern: AttackPattern = AttackPattern.CONCENTRATED,
        num_enemies: int = 15,
        num_friendly_pairs: int = 5,
        model_path: str = "models/formation_classifier.pt",
        assignment_mode: str = AssignmentMode.ORTOOLS,  # Default: OR-Tools optimal assignment
    ):
        pygame.init()
        pygame.display.set_caption("USV Defense Simulator - ML Enhanced")

        self.screen = pygame.display.set_mode(
            (CONFIG["screen_width"], CONFIG["screen_height"])
        )
        self.clock = pygame.time.Clock()
        self.renderer = MLRenderer(self.screen, CONFIG)

        self.attack_pattern = attack_pattern
        self.num_enemies = num_enemies
        self.num_friendly_pairs = num_friendly_pairs

        self.sim_time = 0.0
        self.running = True
        self.paused = False

        self.friendlies: List[USV] = []
        self.enemies: List[USV] = []
        self.enemy_spawn_times: dict = {}

        self.log = SimulationLog()
        self.neutralized_count = 0
        self.total_enemies = num_enemies
        self._counted_enemy_ids: set = set()  # Track which enemies have been counted (debug)

        # ML System
        self.ml_system = DefenseMLSystem(
            center=CONFIG["defense_center"],
            total_pairs=num_friendly_pairs,
        )
        self.ml_system.load_model(model_path)

        # Tactical Command state
        self._command_issued = False  # True after tactical mapping decision
        self._pair_assignments: Dict[int, int] = {}  # pair_id -> cluster_id
        self._assignment_reasoning = ""
        self._assignment_mode = assignment_mode  # 'ortools', 'llm', or 'scipy'
        self._cluster_centers: Dict[int, Tuple[float, float]] = {}  # cluster_id -> (x, y)
        self._cluster_enemy_ids: Dict[int, set] = {}  # cluster_id -> set of enemy IDs (from DBSCAN)
        self._pair_targets: Dict[int, USV] = {}  # pair_idx -> FIXED target enemy (only changes when target dies)
        self._targeted_enemies: set = set()  # Set of enemy IDs already targeted (for unique targeting)

        # Missed enemy tracking and reserve reassignment
        self._missed_enemies: set = set()  # Enemy IDs that have "passed" their assigned pair
        self._reserve_pairs: set = set()  # Pair indices that are in reserve (no cluster assigned)

        # Home positions for return after capture
        self._pair_home_positions: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        # pair_idx -> ((f1_x, f1_y), (f2_x, f2_y))

        # ML state for rendering
        self._cluster_labels = np.array([])
        self._formation_name = "UNKNOWN"
        self._confidence = 0.0
        self._num_clusters = 0

        self._initialize_simulation()

    def _initialize_simulation(self):
        """
        Initialize USVs with 4-direction spawn around mothership.

        Spawn Pattern:
          - Pairs distributed across 4 cardinal directions (N, E, S, W) in round-robin
          - Example: 8 pairs → 2 at N, 2 at E, 2 at S, 2 at W
          - Multiple pairs at same direction have different patrol radii (no overlap)
          - Each pair faces OUTWARD (away from mothership)
          - All pairs start STATIONARY (vx=vy=0) until LLM command
        """
        defense_center = CONFIG["defense_center"]
        world_size = CONFIG["world_size"]
        spawn_radius = CONFIG["safe_zone_radii"][-1] + 500

        mothership_pos = CONFIG["mothership_position"]

        # 4 cardinal directions only (N, E, S, W)
        # Angles: N=90°, E=0°, S=270°, W=180° (in standard math coords)
        cardinal_angles = [
            math.radians(90),    # North
            math.radians(0),     # East
            math.radians(270),   # South
            math.radians(180),   # West
        ]

        base_patrol_radius = 100  # 100m from mothership (close protection)
        radius_step = 150  # Distance between pairs at same direction
        net_spacing = 80.0
        usv_id = 0

        # Track how many pairs are placed at each direction
        direction_counts = [0, 0, 0, 0]  # N, E, S, W

        for pair_idx in range(self.num_friendly_pairs):
            # Round-robin direction selection: 0,1,2,3,0,1,2,3,...
            direction_idx = pair_idx % 4
            angle = cardinal_angles[direction_idx]

            # Calculate patrol radius (offset for multiple pairs at same direction)
            layer = direction_counts[direction_idx]
            patrol_radius = base_patrol_radius + layer * radius_step
            direction_counts[direction_idx] += 1

            center_x = mothership_pos[0] + patrol_radius * math.cos(angle)
            center_y = mothership_pos[1] + patrol_radius * math.sin(angle)

            # Perpendicular offset for pair members
            perp = angle + math.pi / 2
            offset = net_spacing / 2

            # Face OUTWARD (away from mothership, toward potential threats)
            heading_outward = angle

            f1 = USV(
                id=usv_id,
                x=center_x + offset * math.cos(perp),
                y=center_y + offset * math.sin(perp),
                heading=heading_outward,
                is_friendly=True,
                pair_id=usv_id + 1,
                max_speed=CONFIG["friendly_speed"],
            )
            f2 = USV(
                id=usv_id + 1,
                x=center_x - offset * math.cos(perp),
                y=center_y - offset * math.sin(perp),
                heading=heading_outward,
                is_friendly=True,
                pair_id=usv_id,
                max_speed=CONFIG["friendly_speed"],
            )

            # Start STATIONARY - no velocity until LLM command
            f1.vx, f1.vy = 0.0, 0.0
            f2.vx, f2.vy = 0.0, 0.0

            self.friendlies.extend([f1, f2])

            # Save home positions for return after capture
            self._pair_home_positions[pair_idx] = ((f1.x, f1.y), (f2.x, f2.y))

            usv_id += 2

        # Initialize enemies (same as original)
        if self.attack_pattern == AttackPattern.CONCENTRATED:
            self.enemies = AttackPatternGenerator.generate_concentrated(
                self.num_enemies, spawn_radius, defense_center, world_size
            )
        elif self.attack_pattern == AttackPattern.WAVE:
            wave_spawn_radius = 7000  # 7km — further out so formation is visible at 5km lock
            enemies_with_timing = AttackPatternGenerator.generate_wave(
                self.num_enemies, 3, wave_spawn_radius, defense_center, world_size
            )
            print(f"[INIT] Wave pattern: generate_wave returned {len(enemies_with_timing)} enemies")
            for enemy, spawn_time in enemies_with_timing:
                self.enemies.append(enemy)
                self.enemy_spawn_times[enemy.id] = spawn_time
            print(f"[INIT] After loop: self.enemies has {len(self.enemies)} enemies, total_enemies={self.total_enemies}")
            # Debug: Print all enemy IDs and their initial states
            print(f"[INIT] Enemy details:")
            for e in self.enemies:
                spawn_t = self.enemy_spawn_times.get(e.id, "N/A")
                print(f"  Enemy {e.id}: is_active={e.is_active}, spawn_time={spawn_t}, pos=({e.x:.0f}, {e.y:.0f})")
        elif self.attack_pattern == AttackPattern.DIVERSIONARY:
            self.enemies = AttackPatternGenerator.generate_diversionary(
                self.num_enemies, spawn_radius, defense_center, world_size
            )

    def _get_pair_list(self) -> List[Tuple[USV, USV]]:
        """Get list of active friendly pairs."""
        pairs = []
        seen = set()
        for f in self.friendlies:
            if f.is_active and f.pair_id is not None and f.id not in seen:
                partner = next(
                    (p for p in self.friendlies if p.id == f.pair_id and p.is_active),
                    None,
                )
                if partner:
                    pairs.append((f, partner))
                    seen.add(f.id)
                    seen.add(partner.id)
        return pairs

    def _update_pair_direct_chase(
        self,
        f1: USV,
        f2: USV,
        target_enemy: USV,
        net_spacing: float = 80.0,
    ):
        """
        Update pair to directly chase the target enemy (no interception prediction).
        This ensures friendlies always move TOWARDS the enemy, never backing up.
        """
        # Target position is the enemy's current position
        target_x = target_enemy.x
        target_y = target_enemy.y

        # Calculate perpendicular positions for net formation
        # Use enemy's velocity direction, or fallback to pair-to-enemy direction
        enemy_speed = math.sqrt(target_enemy.vx**2 + target_enemy.vy**2)
        if enemy_speed > 1.0:
            # Enemy is moving - align net perpendicular to enemy's heading
            enemy_heading = math.atan2(target_enemy.vy, target_enemy.vx)
        else:
            # Enemy is stationary - align net perpendicular to approach direction
            pair_center_x = (f1.x + f2.x) / 2
            pair_center_y = (f1.y + f2.y) / 2
            enemy_heading = math.atan2(
                target_y - pair_center_y,
                target_x - pair_center_x
            )

        perp_angle = enemy_heading + math.pi / 2
        offset = net_spacing / 2

        # Target positions for each friendly in the pair
        target1 = (
            target_x + offset * math.cos(perp_angle),
            target_y + offset * math.sin(perp_angle)
        )
        target2 = (
            target_x - offset * math.cos(perp_angle),
            target_y - offset * math.sin(perp_angle)
        )

        # Set velocities directly towards targets (full speed chase)
        f1.set_velocity_towards(target1[0], target1[1], CONFIG["friendly_speed"])
        f2.set_velocity_towards(target2[0], target2[1], CONFIG["friendly_speed"])

    def _check_and_reassign_missed_enemies(
        self,
        pairs: List[Tuple[USV, USV]],
    ):
        """
        Check if any assigned enemies have "passed" their friendly pairs.

        An enemy is considered "missed" if:
          - enemy_dist_to_mothership < pair_dist_to_mothership

        When an enemy is missed:
          1. Mark the enemy as missed
          2. Release the original pair (becomes available)
          3. Find the optimal reserve pair to intercept
          4. Assign the missed enemy to the reserve pair
        """
        mothership_pos = CONFIG["mothership_position"]

        # Collect missed enemies and their original pairs
        newly_missed = []

        for pair_idx, (f1, f2) in enumerate(pairs):
            target_enemy = self._pair_targets.get(pair_idx)
            if target_enemy is None or not target_enemy.is_active:
                continue

            # Skip if already marked as missed
            if target_enemy.id in self._missed_enemies:
                continue

            # Calculate distances to mothership
            pair_center_x = (f1.x + f2.x) / 2
            pair_center_y = (f1.y + f2.y) / 2
            pair_dist = math.sqrt(
                (pair_center_x - mothership_pos[0])**2 +
                (pair_center_y - mothership_pos[1])**2
            )

            enemy_dist = math.sqrt(
                (target_enemy.x - mothership_pos[0])**2 +
                (target_enemy.y - mothership_pos[1])**2
            )

            # Enemy is closer to mothership than the pair by THRESHOLD = MISSED
            # This means the enemy has "passed" the pair and is now between the pair and mothership
            if enemy_dist < pair_dist - MISSED_THRESHOLD:
                newly_missed.append((pair_idx, target_enemy))
                self._missed_enemies.add(target_enemy.id)
                print(f"[MISSED] Enemy {target_enemy.id} passed Pair {pair_idx}!")
                print(f"         enemy_dist={enemy_dist:.0f}m < pair_dist={pair_dist:.0f}m - {MISSED_THRESHOLD}m threshold")
                print(f"         Enemy pos: ({target_enemy.x:.0f}, {target_enemy.y:.0f})")
                print(f"         Pair center: ({pair_center_x:.0f}, {pair_center_y:.0f})")

        if not newly_missed:
            return

        # Collect pair indices that just missed their targets (exclude from reserve candidates)
        pairs_that_missed = {orig_pair_idx for orig_pair_idx, _ in newly_missed}

        # Find TRUE reserve pairs (pairs that were NEVER assigned to a cluster)
        reserve_pairs = []
        for pair_idx, (f1, f2) in enumerate(pairs):
            # Skip pairs that just missed - they shouldn't be reassigned to the same enemy
            if pair_idx in pairs_that_missed:
                continue

            assigned_cluster = self._pair_assignments.get(pair_idx)

            # TRUE RESERVE: Only pairs with NO cluster assignment at all
            # Pairs with cluster assignment but no target are NOT reserve - they're waiting for new enemies in their cluster
            if assigned_cluster is None:
                reserve_pairs.append(pair_idx)

        if not reserve_pairs:
            print(f"[MISSED] No reserve pairs available for {len(newly_missed)} missed enemies!")
            print(f"[MISSED] All pair assignments: {self._pair_assignments}")
            return

        print(f"[REASSIGN] {len(newly_missed)} missed enemies, {len(reserve_pairs)} TRUE reserve pairs: {reserve_pairs}")

        # Optimal reassignment using effective distance
        for orig_pair_idx, missed_enemy in newly_missed:
            if not reserve_pairs:
                print(f"[REASSIGN] No more reserve pairs for Enemy {missed_enemy.id}")
                break

            # Find the best reserve pair for this missed enemy
            best_pair_idx = None
            best_cost = float('inf')

            for reserve_idx in reserve_pairs:
                f1, f2 = pairs[reserve_idx]
                pair_center = ((f1.x + f2.x) / 2, (f1.y + f2.y) / 2)
                pair_heading = math.degrees(f1.heading)

                # Calculate effective distance
                from ml.ortools_assignment import calculate_effective_distance
                eff_dist = calculate_effective_distance(
                    pair_center, pair_heading,
                    (missed_enemy.x, missed_enemy.y),
                    angle_weight=1.5
                )

                if eff_dist < best_cost:
                    best_cost = eff_dist
                    best_pair_idx = reserve_idx

            if best_pair_idx is not None:
                # Release old pair's target
                if orig_pair_idx in self._pair_targets:
                    old_target = self._pair_targets[orig_pair_idx]
                    if old_target and old_target.id in self._targeted_enemies:
                        self._targeted_enemies.discard(old_target.id)
                    self._pair_targets[orig_pair_idx] = None

                # Assign to new reserve pair
                self._pair_targets[best_pair_idx] = missed_enemy
                self._targeted_enemies.add(missed_enemy.id)
                reserve_pairs.remove(best_pair_idx)

                # Create a virtual cluster assignment for this pair
                # Use negative cluster ID to indicate "missed enemy reassignment"
                virtual_cluster_id = -1 * missed_enemy.id
                self._pair_assignments[best_pair_idx] = virtual_cluster_id

                f1_res, f2_res = pairs[best_pair_idx]
                res_center = ((f1_res.x + f2_res.x) / 2, (f1_res.y + f2_res.y) / 2)
                print(f"[REASSIGN] Reserve Pair {best_pair_idx} DEPLOYED -> Enemy {missed_enemy.id}")
                print(f"           Pair {best_pair_idx} pos: ({res_center[0]:.0f}, {res_center[1]:.0f})")
                print(f"           Enemy {missed_enemy.id} pos: ({missed_enemy.x:.0f}, {missed_enemy.y:.0f})")
                print(f"           Effective distance cost: {best_cost:.1f}")

    def _remove_enemy_from_cluster(self, enemy_id: int) -> int:
        """
        Remove an enemy from its cluster membership.

        Called when:
          1. Enemy is captured (neutralized)
          2. Enemy has escaped from its cluster

        Args:
            enemy_id: ID of the enemy to remove

        Returns:
            cluster_id the enemy was removed from, or -1 if not found
        """
        for cluster_id, enemy_ids in self._cluster_enemy_ids.items():
            if enemy_id in enemy_ids:
                enemy_ids.discard(enemy_id)
                print(f"[CLUSTER] Enemy {enemy_id} removed from Cluster {cluster_id}")
                print(f"          Cluster {cluster_id} remaining members: {sorted(enemy_ids)}")
                return cluster_id
        return -1

    def _check_escaped_enemies(self, pairs: List[Tuple[USV, USV]]):
        """
        Check if any enemies have escaped from their assigned cluster.

        An enemy is considered "escaped" if:
          - Distance from cluster center > ESCAPE_DISTANCE_THRESHOLD
          - Enemy is still active

        Escaped enemies are:
          1. Removed from cluster membership
          2. Reassigned to the closest available pair (reserve or pair that finished its target)
        """
        if not self._cluster_centers or not self._cluster_enemy_ids:
            return

        escaped_enemies = []

        for cluster_id, enemy_ids in list(self._cluster_enemy_ids.items()):
            if cluster_id not in self._cluster_centers:
                continue

            cluster_center = self._cluster_centers[cluster_id]

            for enemy_id in list(enemy_ids):  # Copy to avoid modification during iteration
                # Find the enemy object
                enemy = next((e for e in self.enemies if e.id == enemy_id), None)
                if enemy is None or not enemy.is_active:
                    continue

                # Calculate distance from cluster center
                dist_to_center = math.sqrt(
                    (enemy.x - cluster_center[0])**2 +
                    (enemy.y - cluster_center[1])**2
                )

                # Check if escaped
                if dist_to_center > ESCAPE_DISTANCE_THRESHOLD:
                    escaped_enemies.append((enemy, cluster_id))
                    print(f"[ESCAPED] Enemy {enemy.id} escaped from Cluster {cluster_id}!")
                    print(f"          Distance to cluster center: {dist_to_center:.0f}m > {ESCAPE_DISTANCE_THRESHOLD}m threshold")
                    print(f"          Enemy pos: ({enemy.x:.0f}, {enemy.y:.0f})")
                    print(f"          Cluster center: ({cluster_center[0]:.0f}, {cluster_center[1]:.0f})")

        if not escaped_enemies:
            return

        # Remove escaped enemies from their clusters
        for enemy, old_cluster_id in escaped_enemies:
            self._remove_enemy_from_cluster(enemy.id)

        # Find available pairs to chase escaped enemies
        # Available = reserve (no assignment) OR finished their target (target dead/captured)
        available_pairs = []
        for pair_idx, (f1, f2) in enumerate(pairs):
            assigned_cluster = self._pair_assignments.get(pair_idx)
            current_target = self._pair_targets.get(pair_idx)

            # Reserve pair (never assigned)
            if assigned_cluster is None:
                available_pairs.append(pair_idx)
            # Pair that finished its target (target is dead/captured)
            elif current_target is None or not current_target.is_active:
                # Check if the pair is NOT already targeting an escaped enemy (negative cluster_id)
                if assigned_cluster >= 0:  # Only regular clusters, not virtual ones
                    available_pairs.append(pair_idx)

        if not available_pairs:
            print(f"[ESCAPED] No available pairs for {len(escaped_enemies)} escaped enemies!")
            return

        print(f"[ESCAPED] {len(escaped_enemies)} escaped enemies, {len(available_pairs)} available pairs: {available_pairs}")

        # Assign escaped enemies to available pairs using effective distance
        for enemy, old_cluster_id in escaped_enemies:
            if not available_pairs:
                print(f"[ESCAPED] No more pairs for Enemy {enemy.id}")
                break

            # Skip if enemy is already targeted
            if enemy.id in self._targeted_enemies:
                continue

            # Find the best pair for this escaped enemy
            best_pair_idx = None
            best_cost = float('inf')

            for pair_idx in available_pairs:
                f1, f2 = pairs[pair_idx]
                pair_center = ((f1.x + f2.x) / 2, (f1.y + f2.y) / 2)
                pair_heading = math.degrees(f1.heading)

                from ml.ortools_assignment import calculate_effective_distance
                eff_dist = calculate_effective_distance(
                    pair_center, pair_heading,
                    (enemy.x, enemy.y),
                    angle_weight=1.5
                )

                if eff_dist < best_cost:
                    best_cost = eff_dist
                    best_pair_idx = pair_idx

            if best_pair_idx is not None:
                # Clear old target assignment if exists
                old_target = self._pair_targets.get(best_pair_idx)
                if old_target and old_target.id in self._targeted_enemies:
                    self._targeted_enemies.discard(old_target.id)

                # Assign escaped enemy to this pair
                self._pair_targets[best_pair_idx] = enemy
                self._targeted_enemies.add(enemy.id)
                available_pairs.remove(best_pair_idx)

                # Use negative cluster ID to indicate "escaped enemy assignment"
                virtual_cluster_id = -2 * enemy.id  # -2x to distinguish from missed (-1x)
                self._pair_assignments[best_pair_idx] = virtual_cluster_id

                f1, f2 = pairs[best_pair_idx]
                pair_center = ((f1.x + f2.x) / 2, (f1.y + f2.y) / 2)
                print(f"[ESCAPED] Pair {best_pair_idx} ASSIGNED -> Escaped Enemy {enemy.id}")
                print(f"          Pair pos: ({pair_center[0]:.0f}, {pair_center[1]:.0f})")
                print(f"          Enemy pos: ({enemy.x:.0f}, {enemy.y:.0f})")
                print(f"          Effective distance: {best_cost:.1f}")

    def _check_post_capture_reassignment(
        self,
        capturing_pair_idx: int,
        f1: USV,
        f2: USV,
        pairs: List[Tuple[USV, USV]],
    ) -> bool:
        """
        After capturing an enemy, check if there's a nearby enemy to chase instead of returning home.

        Criteria for reassignment:
          1. Enemy must be within POST_CAPTURE_DISTANCE_THRESHOLD (150m) of the pair
          2. Enemy must be within POST_CAPTURE_ANGLE_THRESHOLD (30°) of pair's heading
          3. If enemy is already assigned to another pair, compare effectiveness

        Args:
            capturing_pair_idx: Index of the pair that just captured an enemy
            f1, f2: The two USVs in the pair
            pairs: List of all pairs for comparison

        Returns:
            True if reassigned to a new target, False if should return home
        """
        from ml.ortools_assignment import calculate_effective_distance

        pair_center = ((f1.x + f2.x) / 2, (f1.y + f2.y) / 2)
        pair_heading = math.degrees(f1.heading)

        # Find candidate enemies
        candidates = []

        for enemy in self.enemies:
            if not enemy.is_active:
                continue

            # Skip if this enemy is the one we just captured (should already be inactive)
            if enemy.id in self._counted_enemy_ids:
                continue

            # Calculate distance
            dist = math.sqrt(
                (enemy.x - pair_center[0])**2 +
                (enemy.y - pair_center[1])**2
            )

            # Check distance threshold
            if dist > POST_CAPTURE_DISTANCE_THRESHOLD:
                continue

            # Calculate bearing to enemy
            dx = enemy.x - pair_center[0]
            dy = enemy.y - pair_center[1]
            bearing_rad = math.atan2(-dy, dx)  # Pygame Y-axis inverted
            bearing_deg = math.degrees(bearing_rad)

            # Calculate relative angle
            relative_angle = bearing_deg - pair_heading
            # Normalize to [-180, 180]
            while relative_angle > 180:
                relative_angle -= 360
            while relative_angle < -180:
                relative_angle += 360

            # Check angle threshold
            if abs(relative_angle) > POST_CAPTURE_ANGLE_THRESHOLD:
                continue

            # Calculate effective distance for this pair
            eff_dist = calculate_effective_distance(
                pair_center, pair_heading,
                (enemy.x, enemy.y),
                angle_weight=1.5
            )

            candidates.append({
                'enemy': enemy,
                'dist': dist,
                'angle': relative_angle,
                'eff_dist': eff_dist,
            })

        if not candidates:
            return False

        # Sort by effective distance (best first)
        candidates.sort(key=lambda c: c['eff_dist'])

        print(f"[POST-CAPTURE] Pair {capturing_pair_idx} found {len(candidates)} nearby candidates")

        for candidate in candidates:
            enemy = candidate['enemy']
            my_eff_dist = candidate['eff_dist']

            # Check if this enemy is already targeted by another pair
            assigned_pair_idx = None
            for pidx, target in self._pair_targets.items():
                if target is not None and target.id == enemy.id:
                    assigned_pair_idx = pidx
                    break

            if assigned_pair_idx is not None and assigned_pair_idx != capturing_pair_idx:
                # Enemy is already assigned - compare effectiveness
                other_f1, other_f2 = pairs[assigned_pair_idx]
                other_center = ((other_f1.x + other_f2.x) / 2, (other_f1.y + other_f2.y) / 2)
                other_heading = math.degrees(other_f1.heading)

                other_eff_dist = calculate_effective_distance(
                    other_center, other_heading,
                    (enemy.x, enemy.y),
                    angle_weight=1.5
                )

                print(f"[POST-CAPTURE] Comparing for Enemy {enemy.id}:")
                print(f"              Pair {capturing_pair_idx} eff_dist={my_eff_dist:.1f}")
                print(f"              Pair {assigned_pair_idx} eff_dist={other_eff_dist:.1f}")

                # If I'm more effective (lower cost), steal the target
                if my_eff_dist < other_eff_dist:
                    # Release from other pair
                    self._pair_targets[assigned_pair_idx] = None
                    self._targeted_enemies.discard(enemy.id)

                    # The other pair should find a new target from its cluster
                    # (it will be handled in the next update cycle)
                    print(f"[POST-CAPTURE] Pair {capturing_pair_idx} STEALS Enemy {enemy.id} from Pair {assigned_pair_idx}")
                    print(f"              (my cost {my_eff_dist:.1f} < their cost {other_eff_dist:.1f})")
                else:
                    # Other pair is more effective, skip this candidate
                    print(f"[POST-CAPTURE] Pair {assigned_pair_idx} keeps Enemy {enemy.id} (better positioned)")
                    continue

            # Assign this enemy to the capturing pair
            self._pair_targets[capturing_pair_idx] = enemy
            self._targeted_enemies.add(enemy.id)

            # Use virtual cluster ID for post-capture assignment (-3x)
            virtual_cluster_id = -3 * enemy.id
            self._pair_assignments[capturing_pair_idx] = virtual_cluster_id

            print(f"[POST-CAPTURE] Pair {capturing_pair_idx} -> Enemy {enemy.id}")
            print(f"              dist={candidate['dist']:.0f}m, angle={candidate['angle']:.1f}°, eff_dist={my_eff_dist:.1f}")

            return True  # Successfully reassigned

        return False  # No suitable candidate found

    def _issue_tactical_command(
        self,
        pairs: List[Tuple[USV, USV]],
        clusters_data: List[Dict],
        formation: FormationClass,
        confidence: float,
    ):
        """
        Issue tactical command for pair-to-cluster mapping.
        Uses OR-Tools (default), LLM, or scipy based on assignment_mode.
        Called ONCE when formation lock is triggered.
        """
        if self._command_issued:
            return

        # Prepare agent data
        agents_data = []
        for pair_idx, (f1, f2) in enumerate(pairs):
            center_x = (f1.x + f2.x) / 2
            center_y = (f1.y + f2.y) / 2
            heading = math.degrees(f1.heading)
            agents_data.append({
                'id': pair_idx,
                'pos': (center_x, center_y),
                'angle': heading,
            })

        # Store cluster centers and enemy IDs for target selection
        self._cluster_centers = {}
        self._cluster_enemy_ids = {}  # cluster_id -> set of enemy IDs
        for cluster in clusters_data:
            cid = cluster['cluster_id']
            center = cluster['center']
            self._cluster_centers[cid] = (float(center[0]), float(center[1]))
            # Store enemy IDs (not indices!) that belong to this cluster
            enemy_ids = cluster.get('enemy_ids', [])
            self._cluster_enemy_ids[cid] = set(enemy_ids)

        # Get tactical assignment based on mode
        try:
            assignments, reasoning = get_tactical_assignment(
                agents=agents_data,
                clusters=clusters_data,
                formation=formation,
                confidence=confidence,
                mode=self._assignment_mode,
                model_name="qwen2.5:7b-instruct",
            )
            self._pair_assignments = assignments
            self._assignment_reasoning = reasoning
            self._command_issued = True

            mode_label = {
                AssignmentMode.ORTOOLS: "OR-Tools Optimal",
                AssignmentMode.LLM: "LLM Tactical",
                AssignmentMode.SCIPY: "Scipy Hungarian",
            }.get(self._assignment_mode, self._assignment_mode)

            print("\n" + "=" * 60)
            print(f"【{mode_label} ASSIGNMENT ISSUED - LOCKED】")
            print("=" * 60)
            print(f"Mode: {self._assignment_mode.upper()}")
            print(f"Formation: {formation.name} (Confidence: {confidence:.1%})")
            print(f"Assignments (FIXED): {assignments}")
            print(f"Cluster Centers (FIXED): {self._cluster_centers}")
            print(f"Reasoning: {reasoning}")
            print("=" * 60)
            print("Cluster Enemy Membership:")
            for cid, enemy_ids in sorted(self._cluster_enemy_ids.items()):
                print(f"  Cluster {cid}: Enemies {sorted(enemy_ids)}")
            print("=" * 60)

            # Identify reserve pairs (not assigned to any cluster)
            all_pair_ids = set(range(len(agents_data)))
            assigned_pair_ids = set(assignments.keys())
            reserve_pair_ids = all_pair_ids - assigned_pair_ids
            if reserve_pair_ids:
                print(f"RESERVE PAIRS (stay stationary): {sorted(reserve_pair_ids)}")
            print("NOTE: Assignments are now LOCKED and will NOT change!")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"[Tactical Command] Error: {e}, using scipy fallback")
            from ml.ortools_assignment import scipy_optimal_assignment
            result = scipy_optimal_assignment(agents_data, clusters_data, formation)
            self._pair_assignments = result.pair_assignments
            self._assignment_reasoning = f"Scipy fallback: {e}"
            self._command_issued = True

    def update(self):
        """
        Update simulation state with ML classification and LLM tactical command.

        Flow:
          1. Before 5km lock: Enemies move, friendlies stay STATIONARY
          2. At 5km lock: ML classifies, LLM issues pair-cluster mapping
          3. After command: Friendlies move toward assigned clusters only
        """
        if self.paused:
            return

        dt = CONFIG["dt"]
        defense_center = CONFIG["defense_center"]

        # Debug: Print state at start of frame (only every 60 frames = 1 second)
        if int(self.sim_time * 60) % 60 == 0 and self.sim_time > 0:
            active_count = len([e for e in self.enemies if e.is_active])
            print(f"[TICK t={self.sim_time:.1f}] neutralized={self.neutralized_count}, active={active_count}, counted_ids={len(self._counted_enemy_ids)}")

        # Activate wave enemies
        for enemy in self.enemies:
            if enemy.id in self.enemy_spawn_times:
                if self.sim_time >= self.enemy_spawn_times[enemy.id] and not enemy.is_active:
                    enemy.is_active = True
                    print(f"[WAVE-ACTIVATE] Enemy {enemy.id} activated at t={self.sim_time:.1f}")

        # Update enemy positions
        for enemy in self.enemies:
            if enemy.is_active:
                if self.attack_pattern == AttackPattern.DIVERSIONARY:
                    if random.random() < 0.02:
                        enemy.set_velocity_towards(
                            defense_center[0] + random.uniform(-300, 300),
                            defense_center[1] + random.uniform(-300, 300),
                            enemy.max_speed,
                        )
                enemy.update_position(dt)

        # --- ML-DRIVEN ASSIGNMENT ---
        active_enemies = [e for e in self.enemies if e.is_active]
        pairs = self._get_pair_list()

        if active_enemies and pairs:
            # Build arrays for ML system
            enemy_positions = np.array([[e.x, e.y] for e in active_enemies])
            enemy_velocities = np.array([[e.vx, e.vy] for e in active_enemies])
            pair_positions = np.array([
                [(f1.x + f2.x) / 2, (f1.y + f2.y) / 2]
                for f1, f2 in pairs
            ])

            # Get ML decision (formation classification + density analysis)
            decision = self.ml_system.update(
                enemy_positions, enemy_velocities, pair_positions
            )

            # Update rendering state (only show classification after lock)
            if self.ml_system.formation_locked:
                self._formation_name = FORMATION_NAMES.get(
                    decision.formation_class, decision.formation_class.name
                )
                self._confidence = decision.confidence

                # --- TACTICAL COMMAND AT LOCK MOMENT ---
                if not self._command_issued:
                    # Prepare cluster data with enemy IDs (not indices!)
                    clusters_data = []
                    for m in decision.cluster_metrics:
                        avg_vx = np.mean([active_enemies[i].vx for i in m.enemy_indices]) if m.enemy_indices else 0
                        avg_vy = np.mean([active_enemies[i].vy for i in m.enemy_indices]) if m.enemy_indices else 0
                        # Convert indices to actual enemy IDs for stable targeting
                        enemy_ids = [active_enemies[i].id for i in m.enemy_indices if i < len(active_enemies)]
                        clusters_data.append({
                            'cluster_id': m.cluster_id,
                            'center': m.center,
                            'count': m.size,
                            'velocity': (avg_vx, avg_vy),
                            'enemy_indices': m.enemy_indices,
                            'enemy_ids': enemy_ids,  # Actual enemy IDs for targeting
                        })

                    self._issue_tactical_command(
                        pairs, clusters_data,
                        decision.formation_class, decision.confidence
                    )
            else:
                self._formation_name = "ANALYZING..."
                self._confidence = 0.0

            self._num_clusters = len(decision.cluster_metrics)

            # Get cluster labels for active enemies
            if len(decision.cluster_metrics) > 0:
                from ml.density import DensityAnalyzer
                analyzer = DensityAnalyzer()
                self._cluster_labels = analyzer.cluster_enemies(enemy_positions)
            else:
                self._cluster_labels = np.zeros(len(active_enemies), dtype=int)

            # --- FRIENDLY MOVEMENT (only after tactical command) ---
            if self._command_issued:
                # Clean up targeted enemies set (remove dead enemies)
                self._targeted_enemies = {
                    eid for eid in self._targeted_enemies
                    if any(e.id == eid and e.is_active for e in self.enemies)
                }

                # Move friendlies based on pair-cluster assignments
                # Pairs without assignment stay as RESERVE (stationary)
                for pair_idx, (f1, f2) in enumerate(pairs):
                    net_spacing = NET_SPACING_BASE
                    pair_center = np.array([(f1.x + f2.x) / 2, (f1.y + f2.y) / 2])

                    # Check if this pair has a cluster assignment
                    assigned_cluster_id = self._pair_assignments.get(pair_idx)

                    # === NO ASSIGNMENT = RESERVE UNIT (stay stationary) ===
                    if assigned_cluster_id is None:
                        # No cluster assigned - this pair is reserve, stay put
                        f1.vx = f1.vy = 0
                        f2.vx = f2.vy = 0
                        self._pair_targets[pair_idx] = None
                        continue

                    # === RETURNING TO HOME POSITION ===
                    if assigned_cluster_id == RETURNING_HOME_CLUSTER_ID:
                        home_pos = self._pair_home_positions.get(pair_idx)
                        if home_pos is not None:
                            (home_f1_x, home_f1_y), (home_f2_x, home_f2_y) = home_pos

                            # Check if arrived at home
                            dist_f1 = math.sqrt((f1.x - home_f1_x)**2 + (f1.y - home_f1_y)**2)
                            dist_f2 = math.sqrt((f2.x - home_f2_x)**2 + (f2.y - home_f2_y)**2)

                            if dist_f1 <= HOME_ARRIVAL_THRESHOLD and dist_f2 <= HOME_ARRIVAL_THRESHOLD:
                                # Arrived at home - become reserve
                                print(f"[HOME] Pair {pair_idx} arrived at home position, now RESERVE")
                                self._pair_assignments[pair_idx] = None
                                self._pair_targets[pair_idx] = None
                                f1.vx = f1.vy = 0
                                f2.vx = f2.vy = 0
                                # Snap to exact home position
                                f1.x, f1.y = home_f1_x, home_f1_y
                                f2.x, f2.y = home_f2_x, home_f2_y
                            else:
                                # Still returning - move toward home
                                f1.set_velocity_towards(home_f1_x, home_f1_y, CONFIG["friendly_speed"])
                                f2.set_velocity_towards(home_f2_x, home_f2_y, CONFIG["friendly_speed"])
                                # Log occasionally
                                if int(self.sim_time * 10) % 50 == 0:
                                    print(f"[RETURN] Pair {pair_idx} returning home, dist=({dist_f1:.0f}m, {dist_f2:.0f}m)")
                        continue

                    # === REASSIGNED PAIR (chasing missed/escaped/post-capture enemy) ===
                    # Negative cluster_id indicates direct enemy assignment:
                    #   -1x = missed enemy, -2x = escaped enemy, -3x = post-capture reassignment
                    if assigned_cluster_id < 0:
                        # This pair was reassigned to chase a specific missed/escaped enemy
                        # The target is already set in _pair_targets by _check_and_reassign_missed_enemies
                        target_enemy = self._pair_targets.get(pair_idx)
                        if target_enemy is not None and target_enemy.is_active:
                            # Log chase status occasionally
                            if int(self.sim_time * 10) % 50 == 0:  # Every 5 seconds
                                pair_center = ((f1.x + f2.x) / 2, (f1.y + f2.y) / 2)
                                dist = math.sqrt((target_enemy.x - pair_center[0])**2 + (target_enemy.y - pair_center[1])**2)
                                print(f"[CHASE] Reassigned Pair {pair_idx} -> Enemy {target_enemy.id}, dist={dist:.0f}m")
                            self._update_pair_direct_chase(f1, f2, target_enemy, net_spacing)
                        else:
                            # Target was neutralized - return to home position
                            print(f"[RETURN] Pair {pair_idx} mission complete, returning to HOME")
                            self._pair_assignments[pair_idx] = RETURNING_HOME_CLUSTER_ID
                            self._pair_targets[pair_idx] = None
                        continue

                    # === FIXED TARGET LOGIC with UNIQUE TARGETING ===
                    # Check if current target is still alive
                    current_target = self._pair_targets.get(pair_idx)
                    if current_target is not None and current_target.is_active:
                        # Keep chasing the same target (NO CHANGE)
                        target_enemy = current_target
                    else:
                        # Remove old target from targeted set if it died
                        if current_target is not None:
                            self._targeted_enemies.discard(current_target.id)

                        # Need new target: either first assignment or target died
                        target_enemy = None

                        if len(active_enemies) > 0:
                            # Get enemy IDs that belong to this cluster (from DBSCAN at lock time)
                            cluster_enemy_ids = self._cluster_enemy_ids.get(int(assigned_cluster_id), set())

                            # Find closest UNTARGETED enemy in the assigned cluster ONLY
                            min_dist = float('inf')
                            for e in active_enemies:
                                # Only consider enemies that belong to this cluster
                                if e.id not in cluster_enemy_ids:
                                    continue
                                # Skip already targeted enemies (unique 1:1 targeting)
                                if e.id in self._targeted_enemies:
                                    continue
                                if e.is_active:
                                    dist_to_pair = np.linalg.norm([
                                        e.x - pair_center[0],
                                        e.y - pair_center[1]
                                    ])
                                    if dist_to_pair < min_dist:
                                        min_dist = dist_to_pair
                                        target_enemy = e

                        # NO FALLBACK: If cluster exhausted, stay stationary (don't chase other clusters)
                        # This pair completed its mission or is waiting for reassignment

                        # Log new target assignment and mark as targeted
                        if target_enemy is not None:
                            old_id = current_target.id if current_target else "None"
                            self._targeted_enemies.add(target_enemy.id)
                            print(f"[Target] Pair {pair_idx}: {old_id} -> Enemy {target_enemy.id} (Cluster {assigned_cluster_id})")

                    # Store target (only changes when target dies)
                    self._pair_targets[pair_idx] = target_enemy

                    # Update friendly pair movement - DIRECT CHASE
                    if target_enemy is not None:
                        self._update_pair_direct_chase(
                            f1, f2, target_enemy, net_spacing
                        )
                    else:
                        # Cluster exhausted (all enemies dead/targeted) - stay stationary
                        # This pair completed its mission, waiting as reserve
                        f1.vx = f1.vy = 0
                        f2.vx = f2.vy = 0

                # === CHECK FOR MISSED ENEMIES AND REASSIGN TO RESERVES ===
                # An enemy is "missed" if it's closer to mothership than its assigned pair
                self._check_and_reassign_missed_enemies(pairs)

                # === CHECK FOR ESCAPED ENEMIES ===
                # An enemy is "escaped" if it moves beyond ESCAPE_DISTANCE_THRESHOLD from cluster center
                self._check_escaped_enemies(pairs)

            # else: Friendlies stay stationary (vx=vy=0) until tactical command
        else:
            self._cluster_labels = np.array([])

        # Update friendly positions (only after tactical command issued)
        if self._command_issued:
            for f in self.friendlies:
                if f.is_active:
                    f.update_position(dt)
        # else: Friendlies remain stationary at initial positions

        # Check captures
        for f in self.friendlies:
            if f.is_active and f.pair_id is not None:
                partner = next(
                    (p for p in self.friendlies if p.id == f.pair_id and p.is_active),
                    None,
                )
                if partner and f.id < partner.id:
                    for enemy in self.enemies:
                        if enemy.is_active:
                            if PhysicsEngine.check_net_capture(
                                enemy, f, partner,
                                CONFIG["capture_distance"],
                                CONFIG["net_max_length"],
                            ):
                                print(f"[CAPTURE-PRE] enemy.id={enemy.id}, is_active={enemy.is_active}, neutralized_count={self.neutralized_count}")
                                if enemy.id in self._counted_enemy_ids:
                                    print(f"[BUG!] Enemy {enemy.id} already counted! Skipping double-count.")
                                    continue
                                self._counted_enemy_ids.add(enemy.id)
                                enemy.is_active = False
                                enemy.vx = 0.0
                                enemy.vy = 0.0
                                self.neutralized_count += 1

                                # Remove captured enemy from cluster membership
                                self._remove_enemy_from_cluster(enemy.id)
                                # Also remove from targeted set
                                self._targeted_enemies.discard(enemy.id)

                                # === POST-CAPTURE REASSIGNMENT or RETURN TO HOME ===
                                # Find which pair captured this enemy
                                pairs = self._get_pair_list()
                                for pair_idx, (p1, p2) in enumerate(pairs):
                                    if (p1.id == f.id and p2.id == partner.id) or (p1.id == partner.id and p2.id == f.id):
                                        # Clear old target first
                                        self._pair_targets[pair_idx] = None

                                        # Try to find a nearby enemy to chase
                                        reassigned = self._check_post_capture_reassignment(
                                            pair_idx, p1, p2, pairs
                                        )

                                        if not reassigned:
                                            # No nearby target - return to home position
                                            self._pair_assignments[pair_idx] = RETURNING_HOME_CLUSTER_ID
                                            print(f"[RETURN] Pair {pair_idx} captured Enemy {enemy.id}, returning to HOME position")
                                        break

                                print(f"[CAPTURE-POST] enemy.id={enemy.id}, neutralized_count={self.neutralized_count}, total={self.total_enemies}, remaining={self.total_enemies - self.neutralized_count}")
                                self.renderer.add_explosion(enemy.x, enemy.y, self.sim_time)

                                dist_to_center = PhysicsEngine.distance_to_point(
                                    enemy.x, enemy.y, defense_center[0], defense_center[1]
                                )
                                self.log.capture_events.append(CaptureEvent(
                                    time=self.sim_time,
                                    enemy_id=enemy.id,
                                    friendly_pair=(f.id, partner.id),
                                    position=(enemy.x, enemy.y),
                                    distance_to_center=dist_to_center,
                                ))

        # Check mothership collisions
        mothership_pos = CONFIG["mothership_position"]
        collision_distance = CONFIG["mothership_collision_distance"]
        for enemy in self.enemies:
            if enemy.is_active:
                dist = PhysicsEngine.distance_to_point(
                    enemy.x, enemy.y, mothership_pos[0], mothership_pos[1]
                )
                if dist <= collision_distance:
                    print(f"[MOTHERSHIP-PRE] enemy.id={enemy.id}, is_active={enemy.is_active}, dist={dist:.1f}, neutralized_count={self.neutralized_count}")
                    if enemy.id in self._counted_enemy_ids:
                        print(f"[BUG!] Enemy {enemy.id} already counted! Skipping double-count.")
                        enemy.is_active = False  # Still deactivate to prevent further checks
                        enemy.vx = 0.0
                        enemy.vy = 0.0
                        continue
                    self._counted_enemy_ids.add(enemy.id)
                    enemy.is_active = False
                    enemy.vx = 0.0
                    enemy.vy = 0.0
                    self.neutralized_count += 1
                    print(f"[MOTHERSHIP-POST] enemy.id={enemy.id}, neutralized_count={self.neutralized_count}, total={self.total_enemies}, remaining={self.total_enemies - self.neutralized_count}")
                    self.renderer.add_explosion(
                        enemy.x, enemy.y, self.sim_time, 1.0, True
                    )
                    dist_to_center = PhysicsEngine.distance_to_point(
                        enemy.x, enemy.y, defense_center[0], defense_center[1]
                    )
                    self.log.capture_events.append(CaptureEvent(
                        time=self.sim_time,
                        enemy_id=enemy.id,
                        friendly_pair=(-1, -1),
                        position=(enemy.x, enemy.y),
                        distance_to_center=dist_to_center,
                    ))

        # Log
        self.log.timestamps.append(self.sim_time)
        self.log.neutralized_count.append(self.neutralized_count)
        remaining = self.total_enemies - self.neutralized_count
        self.log.remaining_enemies.append(remaining)

        self.sim_time += dt

        # End conditions: all neutralized, or time up
        if remaining <= 0 or self.sim_time >= CONFIG["max_time"]:
            active_count = len([e for e in self.enemies if e.is_active])
            print(f"\n{'='*60}")
            print(f"[END] SIMULATION ENDING")
            print(f"{'='*60}")
            print(f"[END] total_enemies={self.total_enemies}")
            print(f"[END] neutralized_count={self.neutralized_count}")
            print(f"[END] remaining (calculated)={remaining}")
            print(f"[END] active enemies (actual)={active_count}")
            print(f"[END] counted_enemy_ids={sorted(self._counted_enemy_ids)}")
            print(f"[END] len(self.enemies)={len(self.enemies)}")
            print(f"[END] time={self.sim_time:.1f}")
            if active_count != remaining:
                print(f"[END] WARNING: Mismatch! active_count ({active_count}) != remaining ({remaining})")
                print(f"[END] This indicates a bug in neutralized_count tracking!")
                # List all enemy states for debugging
                for e in self.enemies:
                    print(f"  Enemy {e.id}: is_active={e.is_active}, was_counted={e.id in self._counted_enemy_ids}")
            print(f"{'='*60}\n")
            self.running = False

    def render(self):
        """Render with ML overlays."""
        self.renderer.draw_background()
        self.renderer.draw_safe_zones(CONFIG["defense_center"])
        self.renderer.draw_mothership(CONFIG["mothership_position"])

        # Draw cluster overlay for active enemies
        active_enemies = [e for e in self.enemies if e.is_active]
        if len(self._cluster_labels) == len(active_enemies) and len(active_enemies) > 0:
            enemy_pos = np.array([[e.x, e.y] for e in active_enemies])
            self.renderer.draw_cluster_overlay(
                enemy_pos, self._cluster_labels, CLUSTER_COLORS
            )

        # Draw assignment lines (pair -> current target enemy visualization)
        if self._command_issued and self._pair_targets:
            pairs = self._get_pair_list()
            self.renderer.draw_target_lines(
                pairs,
                self._pair_targets,
                self._pair_assignments,
                CLUSTER_COLORS,
            )

        # Draw nets
        drawn_pairs = set()
        for f in self.friendlies:
            if f.is_active and f.pair_id is not None and f.id not in drawn_pairs:
                partner = next(
                    (p for p in self.friendlies if p.id == f.pair_id), None
                )
                if partner and partner.is_active:
                    self.renderer.draw_net(f, partner)
                    drawn_pairs.add(f.id)
                    drawn_pairs.add(partner.id)

        # Draw USVs
        for f in self.friendlies:
            self.renderer.draw_usv(f, CONFIG["friendly_color"])
        for e in self.enemies:
            if e.is_active:
                self.renderer.draw_usv(e, CONFIG["enemy_color"])

        # Effects
        self.renderer.draw_explosions(self.sim_time)

        # Standard UI
        remaining = self.total_enemies - self.neutralized_count
        num_friendlies = len([f for f in self.friendlies if f.is_active])
        self.renderer.draw_ui(
            self.sim_time,
            self.neutralized_count,
            self.total_enemies,
            remaining,
            self.attack_pattern.value,
            num_friendlies,
        )

        # ML panel
        self.renderer.draw_ml_panel(
            self._formation_name,
            self._confidence,
            self._num_clusters,
            self.ml_system.has_model,
            self.ml_system.formation_locked,
        )

        self.renderer.draw_legend()
        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.__init__(
                        self.attack_pattern,
                        self.num_enemies,
                        self.num_friendly_pairs,
                        assignment_mode=self._assignment_mode,
                    )

    def save_log(self, filename: str = "usv_ml_log.csv"):
        """Save simulation log."""
        import pandas as pd

        data = {
            "timestamp": [],
            "event_type": [],
            "enemy_id": [],
            "friendly_pair": [],
            "position_x": [],
            "position_y": [],
            "distance_to_center": [],
            "neutralized_count": [],
            "remaining_enemies": [],
        }

        for event in self.log.capture_events:
            data["timestamp"].append(event.time)
            data["event_type"].append("capture")
            data["enemy_id"].append(event.enemy_id)
            data["friendly_pair"].append(f"{event.friendly_pair[0]}-{event.friendly_pair[1]}")
            data["position_x"].append(event.position[0])
            data["position_y"].append(event.position[1])
            data["distance_to_center"].append(event.distance_to_center)

            idx = min(
                range(len(self.log.timestamps)),
                key=lambda i: abs(self.log.timestamps[i] - event.time),
            )
            data["neutralized_count"].append(self.log.neutralized_count[idx])
            data["remaining_enemies"].append(self.log.remaining_enemies[idx])

        data["timestamp"].append(self.sim_time)
        data["event_type"].append("simulation_end")
        data["enemy_id"].append(None)
        data["friendly_pair"].append(None)
        data["position_x"].append(None)
        data["position_y"].append(None)
        data["distance_to_center"].append(None)
        data["neutralized_count"].append(self.neutralized_count)
        data["remaining_enemies"].append(self.total_enemies - self.neutralized_count)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Log saved to {filename}")

    def run(self):
        """Main simulation loop."""
        print("=" * 60)
        print("USV Defense Simulator - ML Enhanced")
        print("=" * 60)
        print(f"Attack Pattern: {self.attack_pattern.value}")
        print(f"Enemies: {self.num_enemies} (total_enemies={self.total_enemies})")
        print(f"Enemy objects created: {len(self.enemies)}")
        print(f"Initial neutralized_count: {self.neutralized_count}")
        print(f"Initial active enemies: {len([e for e in self.enemies if e.is_active])}")
        print(f"Friendly Pairs: {self.num_friendly_pairs}")
        print(f"ML Model: {'Loaded' if self.ml_system.has_model else 'Rule-based fallback'}")
        print("-" * 60)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R     - Reset simulation")
        print("  ESC   - Exit")
        print("=" * 60)

        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)

        remaining = self.total_enemies - self.neutralized_count
        rate = (self.neutralized_count / self.total_enemies * 100) if self.total_enemies > 0 else 0

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE (ML Enhanced)")
        print("=" * 60)
        print(f"Duration: {self.sim_time:.2f} seconds")
        print(f"Enemies Neutralized: {self.neutralized_count}/{self.total_enemies}")
        print(f"Neutralization Rate: {rate:.1f}%")
        print(f"Enemies Remaining: {remaining}")
        locked_str = " [LOCKED]" if self.ml_system.formation_locked else ""
        print(f"Formation: {self._formation_name} ({self._confidence:.1%}){locked_str}")
        print("=" * 60)

        self.save_log()
        pygame.quit()


def main():
    """Entry point for ML simulator with OR-Tools/LLM tactical command."""
    print("\n" + "=" * 50)
    print("USV Defense Simulator - ML + Tactical Command")
    print("=" * 50)
    print("\n【Attack Pattern Selection】")
    print("1. Concentrated Attack (집중형)")
    print("2. Wave Attack (파상형)")
    print("3. Diversionary Attack (양동형)")
    print("-" * 40)

    try:
        choice = input("Select attack pattern (1-3) [default: 1]: ").strip()
        if choice == "" or choice == "1":
            pattern = AttackPattern.CONCENTRATED
        elif choice == "2":
            pattern = AttackPattern.WAVE
        elif choice == "3":
            pattern = AttackPattern.DIVERSIONARY
        else:
            print("Invalid choice, using Concentrated")
            pattern = AttackPattern.CONCENTRATED

        enemies_input = input("Number of enemies (5-15) [default: 10]: ").strip()
        num_enemies = int(enemies_input) if enemies_input else 10
        num_enemies = max(5, min(15, num_enemies))

        pairs_input = input("Number of friendly pairs (2-12) [default: 12]: ").strip()
        num_pairs = int(pairs_input) if pairs_input else 12
        num_pairs = max(2, min(12, num_pairs))

        print("\n【Assignment Mode Selection】")
        print("1. OR-Tools Optimal (default) - Fast, deterministic, guaranteed 1:1 mapping")
        print("2. LLM Tactical - Uses Ollama LLM for tactical reasoning")
        print("3. Scipy Hungarian - Fallback mode using scipy")
        print("-" * 40)

        mode_input = input("Select assignment mode (1-3) [default: 1]: ").strip()
        if mode_input == "" or mode_input == "1":
            assignment_mode = AssignmentMode.ORTOOLS
        elif mode_input == "2":
            assignment_mode = AssignmentMode.LLM
        elif mode_input == "3":
            assignment_mode = AssignmentMode.SCIPY
        else:
            print("Invalid choice, using OR-Tools")
            assignment_mode = AssignmentMode.ORTOOLS

        if assignment_mode == AssignmentMode.ORTOOLS:
            print("\n【OR-Tools Mode (Default)】")
            print("  - Linear Sum Assignment for optimal 1:1 mapping")
            print("  - Uses effective distance: distance + |angle| × 1.5")
            print("  - Deterministic results, no variability")
        elif assignment_mode == AssignmentMode.LLM:
            print("\n【LLM Mode】")
            print("  - Recommended model: qwen2.5:7b-instruct (via Ollama)")
            print("  - Ensure Ollama is running: ollama serve")
            print("  - Pull model if needed: ollama pull qwen2.5:7b-instruct")
        else:
            print("\n【Scipy Mode】")
            print("  - Hungarian algorithm via scipy.optimize.linear_sum_assignment")

    except (ValueError, KeyboardInterrupt):
        print("\nUsing default settings...")
        pattern = AttackPattern.CONCENTRATED
        num_enemies = 10
        num_pairs = 12
        assignment_mode = AssignmentMode.ORTOOLS

    mode_label = {
        AssignmentMode.ORTOOLS: "OR-Tools Optimal",
        AssignmentMode.LLM: "LLM Tactical",
        AssignmentMode.SCIPY: "Scipy Hungarian",
    }.get(assignment_mode, assignment_mode)

    print("\n" + "-" * 40)
    print("【Simulation Configuration】")
    print(f"  Pattern: {pattern.value}")
    print(f"  Enemies: {num_enemies}")
    print(f"  Friendly Pairs: {num_pairs}")
    print(f"  Assignment Mode: {mode_label}")
    print("-" * 40 + "\n")

    sim = MLSimulation(
        attack_pattern=pattern,
        num_enemies=num_enemies,
        num_friendly_pairs=num_pairs,
        assignment_mode=assignment_mode,
    )
    sim.run()


if __name__ == "__main__":
    main()
