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
from ml.constants import DEFENSE_CENTER, NET_SPACING_BASE

# Import LLM Commander
from ml.llm_commander import LLMCommander, rule_based_mapping, get_tactical_command


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
    """Main simulation controller with ML-driven defense and LLM tactical command."""

    def __init__(
        self,
        attack_pattern: AttackPattern = AttackPattern.CONCENTRATED,
        num_enemies: int = 15,
        num_friendly_pairs: int = 5,
        model_path: str = "models/formation_classifier.pt",
        use_llm: bool = False,  # Set True to enable LLM tactical command
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

        # ML System
        self.ml_system = DefenseMLSystem(
            center=CONFIG["defense_center"],
            total_pairs=num_friendly_pairs,
        )
        self.ml_system.load_model(model_path)

        # LLM Commander state
        self._llm_command_issued = False  # True after LLM mapping decision
        self._llm_pair_assignments: Dict[int, int] = {}  # pair_id -> cluster_id
        self._llm_reasoning = ""
        self._use_llm = use_llm  # Whether to use LLM or rule-based fallback

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
          - 5 pairs at 4 cardinal directions (N, E, S, W) + 1 extra at NE
          - Each pair faces OUTWARD (away from mothership)
          - All pairs start STATIONARY (vx=vy=0) until LLM command
        """
        defense_center = CONFIG["defense_center"]
        world_size = CONFIG["world_size"]
        spawn_radius = CONFIG["safe_zone_radii"][-1] + 500

        mothership_pos = CONFIG["mothership_position"]

        # 4 cardinal directions + 1 extra (for 5 pairs)
        # Angles: N=90°, E=0°, S=270°, W=180°, NE=45° (in standard math coords)
        # Note: Pygame has Y inverted, so N=-90°, S=90° on screen
        cardinal_angles = [
            math.radians(90),    # North (pair 0)
            math.radians(0),     # East (pair 1)
            math.radians(270),   # South (pair 2)
            math.radians(180),   # West (pair 3)
            math.radians(45),    # North-East (pair 4, extra)
        ]

        patrol_radius = CONFIG["safe_zone_radii"][1] - 200  # ~1800m from mothership
        net_spacing = 80.0
        usv_id = 0

        for pair_idx in range(min(self.num_friendly_pairs, len(cardinal_angles))):
            angle = cardinal_angles[pair_idx]
            center_x = mothership_pos[0] + patrol_radius * math.cos(angle)
            center_y = mothership_pos[1] + patrol_radius * math.sin(angle)

            # Perpendicular offset for pair members
            perp = angle + math.pi / 2
            offset = net_spacing / 2

            # Face OUTWARD (away from mothership, toward potential threats)
            heading_outward = angle  # Facing outward

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
            usv_id += 2

        # Handle extra pairs beyond 5 (distribute evenly)
        for pair_idx in range(len(cardinal_angles), self.num_friendly_pairs):
            angle = 2 * math.pi * pair_idx / self.num_friendly_pairs
            center_x = mothership_pos[0] + patrol_radius * math.cos(angle)
            center_y = mothership_pos[1] + patrol_radius * math.sin(angle)
            perp = angle + math.pi / 2
            offset = net_spacing / 2

            f1 = USV(
                id=usv_id,
                x=center_x + offset * math.cos(perp),
                y=center_y + offset * math.sin(perp),
                heading=angle,
                is_friendly=True,
                pair_id=usv_id + 1,
                max_speed=CONFIG["friendly_speed"],
            )
            f2 = USV(
                id=usv_id + 1,
                x=center_x - offset * math.cos(perp),
                y=center_y - offset * math.sin(perp),
                heading=angle,
                is_friendly=True,
                pair_id=usv_id,
                max_speed=CONFIG["friendly_speed"],
            )
            f1.vx, f1.vy = 0.0, 0.0
            f2.vx, f2.vy = 0.0, 0.0
            self.friendlies.extend([f1, f2])
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
            for enemy, spawn_time in enemies_with_timing:
                self.enemies.append(enemy)
                self.enemy_spawn_times[enemy.id] = spawn_time
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

    def _issue_llm_command(
        self,
        pairs: List[Tuple[USV, USV]],
        clusters_data: List[Dict],
        formation: FormationClass,
        confidence: float,
    ):
        """
        Issue LLM tactical command for pair-to-cluster mapping.
        Called ONCE when formation lock is triggered.
        """
        if self._llm_command_issued:
            return

        # Prepare agent data for LLM
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

        # Get tactical command (LLM or rule-based fallback)
        try:
            assignments, reasoning = get_tactical_command(
                agents=agents_data,
                clusters=clusters_data,
                formation=formation,
                confidence=confidence,
                use_llm=self._use_llm,
                model_name="qwen2.5:7b-instruct",
            )
            self._llm_pair_assignments = assignments
            self._llm_reasoning = reasoning
            self._llm_command_issued = True

            print("\n" + "=" * 60)
            print("【LLM TACTICAL COMMAND ISSUED】")
            print("=" * 60)
            print(f"Formation: {formation.name} (Confidence: {confidence:.1%})")
            print(f"Assignments: {assignments}")
            print(f"Reasoning: {reasoning}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"[LLM Command] Error: {e}, using rule-based fallback")
            self._llm_pair_assignments = rule_based_mapping(
                agents_data, clusters_data, formation
            )
            self._llm_reasoning = "Rule-based fallback (LLM error)"
            self._llm_command_issued = True

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

        # Activate wave enemies
        for enemy in self.enemies:
            if enemy.id in self.enemy_spawn_times:
                if self.sim_time >= self.enemy_spawn_times[enemy.id] and not enemy.is_active:
                    enemy.is_active = True

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

                # --- LLM COMMAND AT LOCK MOMENT ---
                if not self._llm_command_issued:
                    # Prepare cluster data for LLM
                    clusters_data = []
                    for m in decision.cluster_metrics:
                        avg_vx = np.mean([active_enemies[i].vx for i in m.enemy_indices]) if m.enemy_indices else 0
                        avg_vy = np.mean([active_enemies[i].vy for i in m.enemy_indices]) if m.enemy_indices else 0
                        clusters_data.append({
                            'cluster_id': m.cluster_id,
                            'center': m.center,
                            'count': m.size,  # ClusterMetrics uses 'size' not 'count'
                            'velocity': (avg_vx, avg_vy),
                            'enemy_indices': m.enemy_indices,
                        })

                    self._issue_llm_command(
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

            # --- FRIENDLY MOVEMENT (only after LLM command) ---
            if self._llm_command_issued:
                # Move friendlies based on LLM pair-cluster assignments
                for pair_idx, (f1, f2) in enumerate(pairs):
                    # Get assigned cluster from LLM
                    assigned_cluster_id = self._llm_pair_assignments.get(pair_idx)

                    # Find target enemy within assigned cluster
                    target_enemy = None
                    net_spacing = NET_SPACING_BASE

                    if assigned_cluster_id is not None:
                        # Find the cluster metrics for this cluster
                        cluster_metrics = None
                        for m in decision.cluster_metrics:
                            if m.cluster_id == assigned_cluster_id:
                                cluster_metrics = m
                                break

                        if cluster_metrics and cluster_metrics.enemy_indices:
                            # Find closest enemy in the assigned cluster
                            pair_center = np.array([(f1.x + f2.x) / 2, (f1.y + f2.y) / 2])
                            min_dist = float('inf')
                            for enemy_idx in cluster_metrics.enemy_indices:
                                if enemy_idx < len(active_enemies):
                                    e = active_enemies[enemy_idx]
                                    dist = np.linalg.norm([e.x - pair_center[0], e.y - pair_center[1]])
                                    if dist < min_dist:
                                        min_dist = dist
                                        target_enemy = e
                            net_spacing = cluster_metrics.net_spacing

                    # Update friendly pair movement
                    if target_enemy is not None:
                        FriendlyAI.update_friendly_pair(
                            f1, f2, target_enemy, defense_center, net_spacing
                        )
                    else:
                        # No assignment or cluster depleted - patrol around mothership
                        FriendlyAI.update_friendly_pair(
                            f1, f2, None, defense_center, NET_SPACING_BASE
                        )
            # else: Friendlies stay stationary (vx=vy=0) until command
        else:
            self._cluster_labels = np.array([])

        # Update friendly positions (only after LLM command issued)
        if self._llm_command_issued:
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
                                enemy.is_active = False
                                enemy.vx = 0.0
                                enemy.vy = 0.0
                                self.neutralized_count += 1
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
                    enemy.is_active = False
                    enemy.vx = 0.0
                    enemy.vy = 0.0
                    self.neutralized_count += 1
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
        print(f"Enemies: {self.num_enemies}")
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
    """Entry point for ML simulator with LLM tactical command."""
    print("\n" + "=" * 50)
    print("USV Defense Simulator - ML + LLM Tactical Command")
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

        pairs_input = input("Number of friendly pairs (2-8) [default: 5]: ").strip()
        num_pairs = int(pairs_input) if pairs_input else 5
        num_pairs = max(2, min(8, num_pairs))

        llm_input = input("Enable LLM tactical command? (y/n) [default: n]: ").strip().lower()
        use_llm = llm_input == 'y' or llm_input == 'yes'

        if use_llm:
            print("\n【LLM Mode Enabled】")
            print("  - Recommended model: qwen2.5:7b-instruct (via Ollama)")
            print("  - Ensure Ollama is running: ollama serve")
            print("  - Pull model if needed: ollama pull qwen2.5:7b-instruct")
        else:
            print("\n【Rule-based Mode】")
            print("  - Using Hungarian algorithm for pair-cluster assignment")

    except (ValueError, KeyboardInterrupt):
        print("\nUsing default settings...")
        pattern = AttackPattern.CONCENTRATED
        num_enemies = 10
        num_pairs = 5
        use_llm = False

    print("\n" + "-" * 40)
    print("【Simulation Configuration】")
    print(f"  Pattern: {pattern.value}")
    print(f"  Enemies: {num_enemies}")
    print(f"  Friendly Pairs: {num_pairs}")
    print(f"  LLM Command: {'Enabled' if use_llm else 'Disabled (Rule-based)'}")
    print("-" * 40 + "\n")

    sim = MLSimulation(
        attack_pattern=pattern,
        num_enemies=num_enemies,
        num_friendly_pairs=num_pairs,
        use_llm=use_llm,
    )
    sim.run()


if __name__ == "__main__":
    main()
