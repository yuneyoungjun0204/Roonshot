"""
Full-Scale High-Fidelity USV Simulator for Research
====================================================
A comprehensive simulation of Unmanned Surface Vehicles (USV) with net capture mechanics.

Requirements: pygame, numpy, pandas
Run: python usv_simulator.py
"""

import pygame
import numpy as np
import pandas as pd
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Simulation
    "dt": 0.05,  # Time step (seconds)
    "max_time": 300,  # Max simulation time (seconds)

    # Display
    "screen_width": 1600,
    "screen_height": 1200,
    "world_size": 18000,  # 18km x 18km simulation world
    "view_radius": 9000,  # visible radius from defense center (9km viewport)

    # Colors (Dark Theme)
    "bg_color": (15, 15, 25),
    "grid_color": (30, 30, 50),
    "friendly_color": (0, 200, 255),
    "enemy_color": (255, 80, 80),
    "net_color": (100, 255, 150),
    "captured_color": (255, 200, 0),
    "safe_zone_1km": (50, 50, 80, 80),
    "safe_zone_2km": (40, 40, 70, 60),
    "text_color": (220, 220, 240),
    "ui_bg_color": (25, 25, 40, 200),

    # USV Parameters
    "friendly_speed": 75.0,  # m/s (3x increased)
    "enemy_speed": 150.0,  # m/s (high-speed threat, 3x increased)
    "capture_distance": 15.0,  # meters - net capture threshold
    "net_max_length": 200.0,  # meters - maximum net extension

    # Defense Zone
    "defense_center": (9000, 9000),  # Center of defense (meters)
    "safe_zone_radii": [1000, 2000, 3000, 4000, 5000, 6000, 7000],  # 1km ~ 7km radii

    # Mothership
    "mothership_size": 200.0,  # meters (square side length)
    "mothership_color": (150, 150, 200),
    "mothership_position": (9000, 9000),  # Position at defense center
    "mothership_collision_distance": 100.0,  # 0.4km = 400m collision detection radius
}


class AttackPattern(Enum):
    CONCENTRATED = "concentrated"
    WAVE = "wave"
    DIVERSIONARY = "diversionary"


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class USV:
    """Unmanned Surface Vehicle"""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0  # radians
    is_friendly: bool = True
    is_active: bool = True
    target_id: Optional[int] = None
    pair_id: Optional[int] = None  # For friendly USV pairs
    max_speed: float = 25.0

    def update_heading(self):
        """Update heading based on velocity vector"""
        if abs(self.vx) > 0.01 or abs(self.vy) > 0.01:
            self.heading = math.atan2(self.vy, self.vx)

    def update_position(self, dt: float):
        """Update position based on velocity"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.update_heading()

    def set_velocity_towards(self, target_x: float, target_y: float, speed: float):
        """Set velocity vector towards a target position"""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0.1:
            self.vx = (dx / dist) * speed
            self.vy = (dy / dist) * speed
        else:
            self.vx = 0
            self.vy = 0


@dataclass
class CaptureEvent:
    """Records a capture event for logging"""
    time: float
    enemy_id: int
    friendly_pair: Tuple[int, int]
    position: Tuple[float, float]
    distance_to_center: float


@dataclass
class SimulationLog:
    """Stores simulation data for CSV export"""
    timestamps: List[float] = field(default_factory=list)
    friendly_positions: List[dict] = field(default_factory=list)
    enemy_positions: List[dict] = field(default_factory=list)
    capture_events: List[CaptureEvent] = field(default_factory=list)
    neutralized_count: List[int] = field(default_factory=list)
    remaining_enemies: List[int] = field(default_factory=list)


# =============================================================================
# PHYSICS ENGINE
# =============================================================================
class PhysicsEngine:
    """Handles kinematics and collision detection"""

    @staticmethod
    def point_to_line_segment_distance(px: float, py: float,
                                        x1: float, y1: float,
                                        x2: float, y2: float) -> float:
        """
        Calculate minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2)
        Uses the proper line-segment algorithm to handle endpoints correctly.
        """
        # Vector from line start to point
        dx = x2 - x1
        dy = y2 - y1

        # Line segment length squared
        line_len_sq = dx*dx + dy*dy

        if line_len_sq < 0.0001:
            # Line segment is essentially a point
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        # Parameter t for projection onto line
        t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / line_len_sq))

        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance to closest point
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    @staticmethod
    def point_to_infinite_line_distance(px: float, py: float,
                                         x1: float, y1: float,
                                         x2: float, y2: float) -> float:
        """
        Calculate distance from point to infinite line using the formula:
        d = |((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)| / sqrt((y2-y1)^2 + (x2-x1)^2)
        """
        numerator = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if denominator < 0.0001:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        return numerator / denominator

    @staticmethod
    def check_net_capture(enemy: USV, friendly1: USV, friendly2: USV,
                          capture_distance: float, max_net_length: float) -> bool:
        """
        Check if enemy USV is captured by the net between two friendly USVs.
        Considers:
        1. Distance from enemy to net line segment
        2. Net length constraint
        3. Enemy must be active
        """
        if not enemy.is_active or not friendly1.is_active or not friendly2.is_active:
            return False

        # Check net length
        net_length = math.sqrt((friendly2.x - friendly1.x)**2 +
                               (friendly2.y - friendly1.y)**2)
        if net_length > max_net_length:
            return False

        # Calculate distance to net
        distance = PhysicsEngine.point_to_line_segment_distance(
            enemy.x, enemy.y,
            friendly1.x, friendly1.y,
            friendly2.x, friendly2.y
        )

        return distance <= capture_distance

    @staticmethod
    def distance_to_point(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def check_point_in_square(px: float, py: float,
                              square_center_x: float, square_center_y: float,
                              square_size: float) -> bool:
        """
        Check if a point is inside a square.
        Square is centered at (square_center_x, square_center_y) with side length square_size.
        """
        half_size = square_size / 2
        return (px >= square_center_x - half_size and
                px <= square_center_x + half_size and
                py >= square_center_y - half_size and
                py <= square_center_y + half_size)


# =============================================================================
# ATTACK PATTERN GENERATORS
# =============================================================================
class AttackPatternGenerator:
    """Generates different enemy attack patterns"""

    @staticmethod
    def generate_concentrated(num_enemies: int, spawn_radius: float,
                              target: Tuple[float, float],
                              world_size: float) -> List[USV]:
        """
        Concentrated Attack: All enemies spawn from a single direction
        and converge on the target simultaneously.
        """
        enemies = []

        # Random spawn angle (single direction)
        base_angle = random.uniform(0, 2 * math.pi)
        angle_spread = math.pi / 6  # 30 degree spread

        for i in range(num_enemies):
            angle = base_angle + random.uniform(-angle_spread, angle_spread)
            distance = spawn_radius + random.uniform(0, 500)

            x = target[0] + distance * math.cos(angle)
            y = target[1] + distance * math.sin(angle)

            # Clamp to world bounds
            x = max(0, min(world_size, x))
            y = max(0, min(world_size, y))

            enemy = USV(
                id=100 + i,
                x=x, y=y,
                is_friendly=False,
                max_speed=CONFIG["enemy_speed"]
            )
            enemy.set_velocity_towards(target[0], target[1], CONFIG["enemy_speed"])
            enemies.append(enemy)

        return enemies

    @staticmethod
    def generate_wave(num_enemies: int, num_waves: int,
                      spawn_radius: float, target: Tuple[float, float],
                      world_size: float, wave_delay: float = 5.0,
                      wave_gap: float = 800.0) -> List[Tuple[USV, float]]:
        """
        Wave Attack: Sequential waves from the SAME direction.

        All waves share one approach angle. Each wave is a lateral line
        of ships (side by side), separated by distance gaps.
          - 1st wave (closest): spawns at spawn_radius, activated at t=0
          - 2nd wave (behind):  spawns at spawn_radius + wave_gap, activated at t=5s
          - 3rd wave (furthest): spawns at spawn_radius + 2*wave_gap, activated at t=10s

        Returns list of (USV, spawn_time) tuples.
        """
        enemies_with_timing = []
        enemies_per_wave = num_enemies // num_waves
        remainder = num_enemies % num_waves

        # Single approach direction for ALL waves
        base_angle = random.uniform(0, 2 * math.pi)
        lateral_spread = math.radians(random.uniform(8.0, 20.0)) / 2

        enemy_idx = 0
        for wave in range(num_waves):
            wave_dist = spawn_radius + wave * wave_gap + random.uniform(-100, 100)
            spawn_time = wave * wave_delay

            n_in_wave = enemies_per_wave + (1 if wave < remainder else 0)

            for i in range(n_in_wave):
                # Lateral spread: ships side by side within each wave
                if n_in_wave > 1:
                    frac = i / (n_in_wave - 1) - 0.5
                    lateral_offset = frac * 2 * lateral_spread
                else:
                    lateral_offset = 0.0

                angle = base_angle + lateral_offset + random.uniform(-0.02, 0.02)
                distance = wave_dist + random.uniform(-50, 50)

                x = target[0] + distance * math.cos(angle)
                y = target[1] + distance * math.sin(angle)

                x = max(0, min(world_size, x))
                y = max(0, min(world_size, y))

                enemy = USV(
                    id=100 + enemy_idx,
                    x=x, y=y,
                    is_friendly=False,
                    is_active=False,
                    max_speed=CONFIG["enemy_speed"]
                )
                enemy.set_velocity_towards(target[0], target[1], CONFIG["enemy_speed"])
                enemies_with_timing.append((enemy, spawn_time))
                enemy_idx += 1

        return enemies_with_timing

    @staticmethod
    def generate_diversionary(num_enemies: int, spawn_radius: float,
                               target: Tuple[float, float],
                               world_size: float) -> List[USV]:
        """
        Diversionary Attack: 3 clustered groups from 3 directions (~120° apart).

        Each direction has a small cluster of 3-4 ships approaching together.
        This simulates a coordinated multi-directional attack.

        Visual shape:
          - Direction 1 (0°)   : 3-4 ships clustered
          - Direction 2 (120°) : 3-4 ships clustered
          - Direction 3 (240°) : 3-4 ships clustered
        """
        enemies = []
        num_directions = 3
        cluster_spread_deg = random.uniform(8.0, 15.0)  # Spread within each cluster
        cluster_spread_rad = math.radians(cluster_spread_deg) / 2

        # Base angles: 120° apart with random rotation
        base_angle_offset = random.uniform(0, 2 * math.pi)
        direction_angles = []
        for i in range(num_directions):
            base_angle = base_angle_offset + (2 * math.pi * i / num_directions)
            # Add some randomization to direction (±15°)
            angle_jitter = random.uniform(-math.pi/12, math.pi/12)
            direction_angles.append(base_angle + angle_jitter)

        # Distribute enemies across directions
        enemies_per_direction = num_enemies // num_directions
        remainder = num_enemies % num_directions

        enemy_id = 100

        for dir_idx, dir_angle in enumerate(direction_angles):
            # Number of enemies in this direction
            n_in_cluster = enemies_per_direction + (1 if dir_idx < remainder else 0)

            # Distance variation for this cluster
            cluster_base_dist = spawn_radius + random.uniform(-300, 300)

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
                distance = cluster_base_dist + random.uniform(-150, 200)

                x = target[0] + distance * math.cos(angle)
                y = target[1] + distance * math.sin(angle)
                x = max(0, min(world_size, x))
                y = max(0, min(world_size, y))

                enemy = USV(
                    id=enemy_id,
                    x=x, y=y,
                    is_friendly=False,
                    max_speed=CONFIG["enemy_speed"]
                )
                enemy.set_velocity_towards(target[0], target[1], CONFIG["enemy_speed"])
                enemies.append(enemy)
                enemy_id += 1

        return enemies


# =============================================================================
# FRIENDLY AI CONTROLLER
# =============================================================================
class FriendlyAI:
    """AI controller for friendly USV pairs"""

    @staticmethod
    def assign_friendly_to_enemy(friendlies: List[USV], enemies: List[USV],
                                  defense_center: Tuple[float, float]) -> dict:
        """
        Assigns friendly USV pairs to intercept enemies.

        Strategy:
        1. Prioritize enemies closest to defense center
        2. Assign pairs based on optimal interception angles
        3. Unassigned pairs patrol the perimeter

        Returns: Dictionary mapping enemy_id -> (friendly1_id, friendly2_id)
        """
        assignments = {}

        # Get active enemies sorted by distance to defense center
        active_enemies = [e for e in enemies if e.is_active]
        active_enemies.sort(key=lambda e: PhysicsEngine.distance_to_point(
            e.x, e.y, defense_center[0], defense_center[1]
        ))

        # Get friendly pairs
        friendly_pairs = []
        paired_ids = set()
        for f in friendlies:
            if f.is_active and f.pair_id is not None and f.id not in paired_ids:
                partner = next((p for p in friendlies if p.id == f.pair_id and p.is_active), None)
                if partner:
                    friendly_pairs.append((f, partner))
                    paired_ids.add(f.id)
                    paired_ids.add(partner.id)

        # Assign pairs to enemies
        assigned_pairs = set()
        for enemy in active_enemies:
            if len(assigned_pairs) >= len(friendly_pairs):
                break

            # Find best pair for this enemy
            best_pair = None
            best_score = float('inf')

            for idx, (f1, f2) in enumerate(friendly_pairs):
                if idx in assigned_pairs:
                    continue

                # Score based on distance and interception potential
                pair_center = ((f1.x + f2.x) / 2, (f1.y + f2.y) / 2)
                dist_to_enemy = PhysicsEngine.distance_to_point(
                    pair_center[0], pair_center[1], enemy.x, enemy.y
                )

                # Predict enemy position
                pred_time = dist_to_enemy / CONFIG["friendly_speed"]
                pred_x = enemy.x + enemy.vx * pred_time
                pred_y = enemy.y + enemy.vy * pred_time

                interception_dist = PhysicsEngine.distance_to_point(
                    pair_center[0], pair_center[1], pred_x, pred_y
                )

                score = interception_dist

                if score < best_score:
                    best_score = score
                    best_pair = idx

            if best_pair is not None:
                f1, f2 = friendly_pairs[best_pair]
                assignments[enemy.id] = (f1.id, f2.id)
                assigned_pairs.add(best_pair)

        return assignments

    @staticmethod
    def calculate_interception_point(friendly: USV, enemy: USV,
                                      defense_center: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate optimal interception point for a friendly USV.
        Uses predictive interception based on enemy trajectory.
        """
        # Predict where enemy will be
        enemy_speed = math.sqrt(enemy.vx**2 + enemy.vy**2)
        if enemy_speed < 0.1:
            return (enemy.x, enemy.y)

        # Distance from enemy to defense center
        dist_to_center = PhysicsEngine.distance_to_point(
            enemy.x, enemy.y, defense_center[0], defense_center[1]
        )

        # Time for enemy to reach center
        time_to_center = dist_to_center / enemy_speed if enemy_speed > 0 else 999

        # Interception point (aim for halfway to center)
        interception_time = min(time_to_center * 0.5,
                                dist_to_center / CONFIG["friendly_speed"])

        pred_x = enemy.x + enemy.vx * interception_time
        pred_y = enemy.y + enemy.vy * interception_time

        return (pred_x, pred_y)

    @staticmethod
    def update_friendly_pair(f1: USV, f2: USV, target_enemy: Optional[USV],
                             defense_center: Tuple[float, float],
                             net_spacing: float = 80.0):
        """
        Update velocities for a friendly USV pair to intercept target.
        Maintains net formation while pursuing.
        """
        if target_enemy is None or not target_enemy.is_active:
            # Patrol mode - circle around defense center
            angle1 = math.atan2(f1.y - defense_center[1], f1.x - defense_center[0])
            angle2 = math.atan2(f2.y - defense_center[1], f2.x - defense_center[0])

            patrol_radius = CONFIG["safe_zone_radii"][1] - 200  # 2km - 200m

            # Move along circular path
            target1 = (
                defense_center[0] + patrol_radius * math.cos(angle1 + 0.02),
                defense_center[1] + patrol_radius * math.sin(angle1 + 0.02)
            )
            target2 = (
                defense_center[0] + patrol_radius * math.cos(angle2 + 0.02),
                defense_center[1] + patrol_radius * math.sin(angle2 + 0.02)
            )

            f1.set_velocity_towards(target1[0], target1[1], CONFIG["friendly_speed"] * 0.5)
            f2.set_velocity_towards(target2[0], target2[1], CONFIG["friendly_speed"] * 0.5)
            return

        # Interception mode
        interception = FriendlyAI.calculate_interception_point(f1, target_enemy, defense_center)

        # Calculate perpendicular positions for net formation
        enemy_heading = math.atan2(target_enemy.vy, target_enemy.vx)
        perp_angle = enemy_heading + math.pi / 2

        offset = net_spacing / 2

        target1 = (
            interception[0] + offset * math.cos(perp_angle),
            interception[1] + offset * math.sin(perp_angle)
        )
        target2 = (
            interception[0] - offset * math.cos(perp_angle),
            interception[1] - offset * math.sin(perp_angle)
        )

        f1.set_velocity_towards(target1[0], target1[1], CONFIG["friendly_speed"])
        f2.set_velocity_towards(target2[0], target2[1], CONFIG["friendly_speed"])


# =============================================================================
# RENDERER
# =============================================================================
class Renderer:
    """Handles all visualization"""

    def __init__(self, screen: pygame.Surface, config: dict):
        self.screen = screen
        self.config = config
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Explosion effects
        self.explosions = []  # [(x, y, start_time, duration, is_mothership_collision)]

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates (viewport-based)."""
        cx, cy = self.config["defense_center"]
        vr = self.config["view_radius"]
        view_size = 2 * vr

        scale_x = self.config["screen_width"] / view_size
        scale_y = self.config["screen_height"] / view_size

        screen_x = int((x - (cx - vr)) * scale_x)
        screen_y = int(self.config["screen_height"] - (y - (cy - vr)) * scale_y)

        return (screen_x, screen_y)

    def scale_distance(self, distance: float) -> int:
        """Scale a world distance to screen pixels (viewport-based)."""
        vr = self.config["view_radius"]
        scale = self.config["screen_width"] / (2 * vr)
        return int(distance * scale)

    def draw_background(self):
        """Draw background with grid"""
        self.screen.fill(self.config["bg_color"])

        # Draw grid
        grid_spacing = 1000  # 1km grid
        vr = self.config["view_radius"]
        scale = self.config["screen_width"] / (2 * vr)
        pixel_spacing = int(grid_spacing * scale)

        for x in range(0, self.config["screen_width"], pixel_spacing):
            pygame.draw.line(self.screen, self.config["grid_color"],
                           (x, 0), (x, self.config["screen_height"]), 1)
        for y in range(0, self.config["screen_height"], pixel_spacing):
            pygame.draw.line(self.screen, self.config["grid_color"],
                           (0, y), (self.config["screen_width"], y), 1)

    def draw_safe_zones(self, center: Tuple[float, float]):
        """Draw semi-transparent safe zone circles at 1km through 5km"""
        screen_center = self.world_to_screen(center[0], center[1])

        radii = self.config["safe_zone_radii"]  # [1000, 2000, 3000, 4000, 5000]

        for i, radius_m in enumerate(radii):
            radius_px = self.scale_distance(radius_m)
            if radius_px < 2:
                continue

            # Color gets dimmer for outer rings
            brightness = max(30, 70 - i * 8)
            alpha = max(40, 80 - i * 8)
            ring_color = (brightness, brightness, brightness + 30, alpha)

            surf = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, ring_color, (radius_px, radius_px), radius_px, 2)
            self.screen.blit(surf,
                            (screen_center[0] - radius_px, screen_center[1] - radius_px))

            # Label
            km_label = f"{radius_m // 1000}km"
            label_color = (brightness + 20, brightness + 20, brightness + 50)
            label = self.font_small.render(km_label, True, label_color)
            self.screen.blit(label, (screen_center[0] + radius_px + 5, screen_center[1]))

        # Defense center marker
        pygame.draw.circle(self.screen, (100, 100, 150), screen_center, 8, 2)
        pygame.draw.line(self.screen, (100, 100, 150),
                        (screen_center[0] - 12, screen_center[1]),
                        (screen_center[0] + 12, screen_center[1]), 2)
        pygame.draw.line(self.screen, (100, 100, 150),
                        (screen_center[0], screen_center[1] - 12),
                        (screen_center[0], screen_center[1] + 12), 2)

    def draw_mothership(self, position: Tuple[float, float]):
        """Draw mothership as a large square"""
        ship_size = self.config["mothership_size"]
        screen_pos = self.world_to_screen(position[0], position[1])
        screen_size = self.scale_distance(ship_size)
        
        # Draw square centered at position
        half_size = screen_size // 2
        rect = pygame.Rect(
            screen_pos[0] - half_size,
            screen_pos[1] - half_size,
            screen_size,
            screen_size
        )
        
        # Fill and outline
        pygame.draw.rect(self.screen, self.config["mothership_color"], rect)
        pygame.draw.rect(self.screen, (200, 200, 255), rect, 3)
        
        # Draw center marker
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 5)

    def draw_usv(self, usv: USV, color: Tuple[int, int, int], size: int = 12):
        """Draw a USV as an arrow/triangle showing heading"""
        if not usv.is_active:
            return

        pos = self.world_to_screen(usv.x, usv.y)
        heading = -usv.heading  # Flip for screen coordinates

        # Triangle points
        front = (
            pos[0] + size * math.cos(heading),
            pos[1] + size * math.sin(heading)
        )
        back_left = (
            pos[0] + size * 0.6 * math.cos(heading + 2.5),
            pos[1] + size * 0.6 * math.sin(heading + 2.5)
        )
        back_right = (
            pos[0] + size * 0.6 * math.cos(heading - 2.5),
            pos[1] + size * 0.6 * math.sin(heading - 2.5)
        )

        pygame.draw.polygon(self.screen, color, [front, back_left, back_right])
        pygame.draw.polygon(self.screen, (255, 255, 255), [front, back_left, back_right], 1)

    def draw_net(self, f1: USV, f2: USV, captured: bool = False):
        """Draw net connection between two friendly USVs"""
        if not f1.is_active or not f2.is_active:
            return

        pos1 = self.world_to_screen(f1.x, f1.y)
        pos2 = self.world_to_screen(f2.x, f2.y)

        # Check net length
        net_length = math.sqrt((f2.x - f1.x)**2 + (f2.y - f1.y)**2)

        if net_length <= self.config["net_max_length"]:
            color = self.config["captured_color"] if captured else self.config["net_color"]
            width = 3 if captured else 2

            # Draw dashed line for net
            dash_length = 10
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0:
                num_dashes = int(dist / dash_length)
                for i in range(0, num_dashes, 2):
                    start_t = i / num_dashes
                    end_t = min((i + 1) / num_dashes, 1.0)
                    start = (int(pos1[0] + dx * start_t), int(pos1[1] + dy * start_t))
                    end = (int(pos1[0] + dx * end_t), int(pos1[1] + dy * end_t))
                    pygame.draw.line(self.screen, color, start, end, width)

    def add_explosion(self, x: float, y: float, current_time: float, duration: float = 0.5, is_mothership_collision: bool = False):
        """Add an explosion effect"""
        self.explosions.append((x, y, current_time, duration, is_mothership_collision))

    def draw_explosions(self, current_time: float):
        """Draw and update explosion effects"""
        active_explosions = []

        for x, y, start_time, duration, is_mothership_collision in self.explosions:
            elapsed = current_time - start_time
            if elapsed < duration:
                active_explosions.append((x, y, start_time, duration, is_mothership_collision))

                # Explosion animation
                progress = elapsed / duration
                
                # Different explosion size and color for mothership collision
                if is_mothership_collision:
                    # Large, intense explosion for mothership collision
                    radius = int(40 + 80 * progress)  # Much larger
                    alpha = int(255 * (1 - progress * 0.7))  # Lasts longer
                    # Red/orange intense color
                    color = (255, int(100 * (1 - progress)), 0, alpha)
                    inner_color = (255, 255, int(150 * (1 - progress)), min(255, alpha + 80))
                else:
                    # Normal capture explosion
                    radius = int(20 + 30 * progress)
                    alpha = int(255 * (1 - progress))
                    # Yellow/orange color
                    color = (255, int(200 * (1 - progress)), 0, alpha)
                    inner_color = (255, 255, int(200 * (1 - progress)), min(255, alpha + 50))

                pos = self.world_to_screen(x, y)

                # Create explosion surface
                surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (radius, radius), radius)

                # Inner flash
                inner_radius = int(radius * 0.5)
                pygame.draw.circle(surf, inner_color, (radius, radius), inner_radius)
                
                # Additional outer ring for mothership collision
                if is_mothership_collision:
                    outer_radius = int(radius * 0.8)
                    outer_color = (255, int(150 * (1 - progress)), int(50 * (1 - progress)), int(alpha * 0.5))
                    pygame.draw.circle(surf, outer_color, (radius, radius), outer_radius, 3)

                self.screen.blit(surf, (pos[0] - radius, pos[1] - radius))

        self.explosions = active_explosions

    def draw_ui(self, sim_time: float, neutralized: int, total_enemies: int,
                remaining: int, attack_pattern: str, num_friendlies: int = 0):
        """Draw UI dashboard"""
        # UI background panel
        panel_width = 280
        panel_height = 220  # Increased height for more info
        panel_x = self.config["screen_width"] - panel_width - 20
        panel_y = 20

        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surf.fill(self.config["ui_bg_color"])
        pygame.draw.rect(panel_surf, (60, 60, 90), (0, 0, panel_width, panel_height), 2)
        self.screen.blit(panel_surf, (panel_x, panel_y))

        # Title
        title = self.font_medium.render("USV Defense Status", True, self.config["text_color"])
        self.screen.blit(title, (panel_x + 15, panel_y + 10))

        # Stats
        y_offset = 55
        line_height = 26

        # Initial settings
        settings_text = self.font_small.render(f"Enemies: {total_enemies} | Friendlies: {num_friendlies}", True, (180, 180, 200))
        self.screen.blit(settings_text, (panel_x + 15, panel_y + y_offset))

        # Time
        time_text = self.font_small.render(f"Time: {sim_time:.1f}s", True, self.config["text_color"])
        self.screen.blit(time_text, (panel_x + 15, panel_y + y_offset + line_height))

        # Pattern
        pattern_text = self.font_small.render(f"Pattern: {attack_pattern}", True, self.config["text_color"])
        self.screen.blit(pattern_text, (panel_x + 15, panel_y + y_offset + line_height * 2))

        # Neutralized rate
        rate = (neutralized / total_enemies * 100) if total_enemies > 0 else 0
        rate_color = (100, 255, 100) if rate > 70 else (255, 255, 100) if rate > 40 else (255, 100, 100)
        rate_text = self.font_small.render(f"Neutralized: {rate:.1f}%", True, rate_color)
        self.screen.blit(rate_text, (panel_x + 15, panel_y + y_offset + line_height * 3))

        # Remaining enemies
        remaining_color = (100, 255, 100) if remaining < 5 else (255, 255, 100) if remaining < 10 else (255, 100, 100)
        remaining_text = self.font_small.render(f"Remaining: {remaining}", True, remaining_color)
        self.screen.blit(remaining_text, (panel_x + 15, panel_y + y_offset + line_height * 4))

        # Progress bar
        bar_x = panel_x + 15
        bar_y = panel_y + y_offset + line_height * 5 + 5
        bar_width = panel_width - 30
        bar_height = 12

        pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, bar_y, bar_width, bar_height))
        fill_width = int(bar_width * (rate / 100))
        pygame.draw.rect(self.screen, rate_color, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, (100, 100, 130), (bar_x, bar_y, bar_width, bar_height), 1)

    def draw_legend(self):
        """Draw legend in bottom left"""
        legend_x = 20
        legend_y = self.config["screen_height"] - 100

        # Friendly
        pygame.draw.polygon(self.screen, self.config["friendly_color"],
                           [(legend_x + 15, legend_y), (legend_x + 5, legend_y + 10),
                            (legend_x + 5, legend_y - 10)])
        text = self.font_small.render("Friendly USV", True, self.config["text_color"])
        self.screen.blit(text, (legend_x + 25, legend_y - 8))

        # Enemy
        pygame.draw.polygon(self.screen, self.config["enemy_color"],
                           [(legend_x + 15, legend_y + 30), (legend_x + 5, legend_y + 40),
                            (legend_x + 5, legend_y + 20)])
        text = self.font_small.render("Enemy USV", True, self.config["text_color"])
        self.screen.blit(text, (legend_x + 25, legend_y + 22))

        # Net
        pygame.draw.line(self.screen, self.config["net_color"],
                        (legend_x + 5, legend_y + 55), (legend_x + 20, legend_y + 55), 2)
        text = self.font_small.render("Capture Net", True, self.config["text_color"])
        self.screen.blit(text, (legend_x + 25, legend_y + 47))


# =============================================================================
# MAIN SIMULATION
# =============================================================================
class Simulation:
    """Main simulation controller"""

    def __init__(self, attack_pattern: AttackPattern = AttackPattern.CONCENTRATED,
                 num_enemies: int = 15, num_friendly_pairs: int = 5):
        pygame.init()
        pygame.display.set_caption("USV Defense Simulator")

        self.screen = pygame.display.set_mode(
            (CONFIG["screen_width"], CONFIG["screen_height"])
        )
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen, CONFIG)

        self.attack_pattern = attack_pattern
        self.num_enemies = num_enemies
        self.num_friendly_pairs = num_friendly_pairs

        self.sim_time = 0.0
        self.running = True
        self.paused = False

        self.friendlies: List[USV] = []
        self.enemies: List[USV] = []
        self.enemy_spawn_times: dict = {}  # For wave attacks

        self.log = SimulationLog()
        self.neutralized_count = 0
        self.total_enemies = num_enemies

        self._initialize_simulation()

    def _initialize_simulation(self):
        """Initialize USVs based on attack pattern"""
        defense_center = CONFIG["defense_center"]
        world_size = CONFIG["world_size"]
        spawn_radius = CONFIG["safe_zone_radii"][-1] + 500  # 5km + 500m

        # Mothership position
        mothership_pos = CONFIG["mothership_position"]
        mothership_size = CONFIG["mothership_size"]
        half_size = mothership_size / 2

        # Initialize friendly USVs at mothership corners
        # Corners: top-left, top-right, bottom-right, bottom-left
        # Headings: 0° (right), 90° (up), 180° (left), 270° (down)
        corner_positions = [
            (mothership_pos[0] - half_size, mothership_pos[1] - half_size),  # Top-left
            (mothership_pos[0] + half_size, mothership_pos[1] - half_size),  # Top-right
            (mothership_pos[0] + half_size, mothership_pos[1] + half_size),  # Bottom-right
            (mothership_pos[0] - half_size, mothership_pos[1] + half_size),  # Bottom-left
        ]
        corner_headings = [
            0.0,           # 0° (right/east)
            math.pi / 2,  # 90° (up/north)
            math.pi,      # 180° (left/west)
            3 * math.pi / 2,  # 270° (down/south)
        ]

        # Create friendly USVs at each corner (2 per corner if enough pairs requested)
        # Calculate how many USVs per corner (max 2 per corner)
        total_usvs_needed = self.num_friendly_pairs * 2
        usvs_per_corner = min(2, max(1, total_usvs_needed // 4))  # Distribute across 4 corners
        corner_offset = 30.0  # Offset distance from corner for second USV
        
        usv_id = 0
        for corner_idx in range(4):
            if usv_id >= total_usvs_needed:
                break
                
            corner_x, corner_y = corner_positions[corner_idx]
            heading = corner_headings[corner_idx]
            
            # Determine how many USVs to create at this corner
            usvs_at_this_corner = min(usvs_per_corner, total_usvs_needed - usv_id)
            
            # Create USVs at this corner
            corner_usv_ids = []
            for usv_in_corner in range(usvs_at_this_corner):
                # Offset position for second USV at same corner
                if usv_in_corner == 0:
                    usv_x, usv_y = corner_x, corner_y
                else:
                    # Offset perpendicular to heading
                    perp_angle = heading + math.pi / 2
                    usv_x = corner_x + corner_offset * math.cos(perp_angle)
                    usv_y = corner_y + corner_offset * math.sin(perp_angle)
                
                # Determine pair_id: same corner USVs pair together
                pair_id = None
                if usvs_at_this_corner == 2:
                    if usv_in_corner == 0:
                        pair_id = usv_id + 1  # Pair with second USV at same corner
                    else:
                        pair_id = usv_id - 1  # Pair with first USV at same corner
                
                # Create USV
                usv = USV(
                    id=usv_id,
                    x=usv_x,
                    y=usv_y,
                    heading=heading,
                    is_friendly=True,
                    pair_id=pair_id,
                    max_speed=CONFIG["friendly_speed"]
                )
                # Set initial velocity based on heading
                usv.vx = CONFIG["friendly_speed"] * math.cos(heading)
                usv.vy = CONFIG["friendly_speed"] * math.sin(heading)
                
                self.friendlies.append(usv)
                usv_id += 1

        # Initialize enemies based on attack pattern
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

    def update(self):
        """Update simulation state"""
        if self.paused:
            return

        dt = CONFIG["dt"]
        defense_center = CONFIG["defense_center"]

        # Activate enemies for wave attacks
        for enemy in self.enemies:
            if enemy.id in self.enemy_spawn_times:
                if self.sim_time >= self.enemy_spawn_times[enemy.id] and not enemy.is_active:
                    enemy.is_active = True

        # Update enemy positions and behaviors
        for enemy in self.enemies:
            if enemy.is_active:
                # Diversionary enemies: update heading periodically
                if self.attack_pattern == AttackPattern.DIVERSIONARY:
                    if random.random() < 0.02:  # 2% chance per frame to adjust course
                        enemy.set_velocity_towards(
                            defense_center[0] + random.uniform(-300, 300),
                            defense_center[1] + random.uniform(-300, 300),
                            enemy.max_speed
                        )

                enemy.update_position(dt)

        # Assign friendly pairs to enemies
        assignments = FriendlyAI.assign_friendly_to_enemy(
            self.friendlies, self.enemies, defense_center
        )

        # Update friendly USVs
        for f in self.friendlies:
            if f.is_active and f.pair_id is not None:
                partner = next((p for p in self.friendlies if p.id == f.pair_id), None)
                if partner and f.id < partner.id:  # Update only once per pair
                    # Find assigned enemy
                    target_enemy = None
                    for enemy_id, (f1_id, f2_id) in assignments.items():
                        if f.id == f1_id or f.id == f2_id:
                            target_enemy = next((e for e in self.enemies if e.id == enemy_id), None)
                            break

                    FriendlyAI.update_friendly_pair(f, partner, target_enemy, defense_center)

        # Update friendly positions
        for f in self.friendlies:
            if f.is_active:
                f.update_position(dt)

        # Check for captures
        for f in self.friendlies:
            if f.is_active and f.pair_id is not None:
                partner = next((p for p in self.friendlies if p.id == f.pair_id and p.is_active), None)
                if partner and f.id < partner.id:
                    for enemy in self.enemies:
                        if enemy.is_active:
                            if PhysicsEngine.check_net_capture(
                                enemy, f, partner,
                                CONFIG["capture_distance"],
                                CONFIG["net_max_length"]
                            ):
                                # Capture! - Destroy enemy immediately
                                enemy.is_active = False
                                enemy.vx = 0.0  # Stop movement
                                enemy.vy = 0.0
                                self.neutralized_count += 1

                                # Add explosion effect
                                self.renderer.add_explosion(enemy.x, enemy.y, self.sim_time)

                                # Log capture event
                                dist_to_center = PhysicsEngine.distance_to_point(
                                    enemy.x, enemy.y, defense_center[0], defense_center[1]
                                )
                                self.log.capture_events.append(CaptureEvent(
                                    time=self.sim_time,
                                    enemy_id=enemy.id,
                                    friendly_pair=(f.id, partner.id),
                                    position=(enemy.x, enemy.y),
                                    distance_to_center=dist_to_center
                                ))

        # Check for enemy collisions with mothership
        # If enemy enters within mothership_collision_distance, it explodes and is destroyed
        mothership_pos = CONFIG["mothership_position"]
        collision_distance = CONFIG["mothership_collision_distance"]
        for enemy in self.enemies:
            if enemy.is_active:
                # Check distance from enemy to mothership center
                dist_to_mothership = PhysicsEngine.distance_to_point(
                    enemy.x, enemy.y,
                    mothership_pos[0], mothership_pos[1]
                )
                if dist_to_mothership <= collision_distance:
                    # Enemy entered collision distance - explode and destroy it immediately
                    enemy.is_active = False
                    enemy.vx = 0.0  # Stop movement immediately
                    enemy.vy = 0.0
                    self.neutralized_count += 1
                    
                    # Add large explosion effect for mothership collision
                    self.renderer.add_explosion(enemy.x, enemy.y, self.sim_time, duration=1.0, is_mothership_collision=True)
                    
                    # Log collision event
                    dist_to_center = PhysicsEngine.distance_to_point(
                        enemy.x, enemy.y, defense_center[0], defense_center[1]
                    )
                    self.log.capture_events.append(CaptureEvent(
                        time=self.sim_time,
                        enemy_id=enemy.id,
                        friendly_pair=(-1, -1),  # No friendly pair for mothership collision
                        position=(enemy.x, enemy.y),
                        distance_to_center=dist_to_center
                    ))

        # Log current state
        self.log.timestamps.append(self.sim_time)
        self.log.neutralized_count.append(self.neutralized_count)
        remaining = self.total_enemies - self.neutralized_count
        self.log.remaining_enemies.append(remaining)

        # Update simulation time
        self.sim_time += dt

        # Check end conditions
        if remaining <= 0 or self.sim_time >= CONFIG["max_time"]:
            self.running = False

    def render(self):
        """Render current state"""
        self.renderer.draw_background()
        self.renderer.draw_safe_zones(CONFIG["defense_center"])
        
        # Draw mothership
        self.renderer.draw_mothership(CONFIG["mothership_position"])

        # Draw nets between friendly pairs
        drawn_pairs = set()
        for f in self.friendlies:
            if f.is_active and f.pair_id is not None and f.id not in drawn_pairs:
                partner = next((p for p in self.friendlies if p.id == f.pair_id), None)
                if partner and partner.is_active:
                    self.renderer.draw_net(f, partner)
                    drawn_pairs.add(f.id)
                    drawn_pairs.add(partner.id)

        # Draw friendly USVs
        for f in self.friendlies:
            self.renderer.draw_usv(f, CONFIG["friendly_color"])

        # Draw enemy USVs (only active ones)
        for e in self.enemies:
            if e.is_active:  # Only draw active enemies
                self.renderer.draw_usv(e, CONFIG["enemy_color"])

        # Draw explosions
        self.renderer.draw_explosions(self.sim_time)

        # Draw UI
        remaining = self.total_enemies - self.neutralized_count
        num_friendlies = len([f for f in self.friendlies if f.is_active])
        self.renderer.draw_ui(
            self.sim_time,
            self.neutralized_count,
            self.total_enemies,
            remaining,
            self.attack_pattern.value,
            num_friendlies
        )

        self.renderer.draw_legend()

        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.__init__(self.attack_pattern, self.num_enemies, self.num_friendly_pairs)

    def save_log(self, filename: str = "usv_log.csv"):
        """Save simulation log to CSV"""
        # Prepare data
        data = {
            'timestamp': [],
            'event_type': [],
            'enemy_id': [],
            'friendly_pair': [],
            'position_x': [],
            'position_y': [],
            'distance_to_center': [],
            'neutralized_count': [],
            'remaining_enemies': []
        }

        # Add capture events
        for event in self.log.capture_events:
            data['timestamp'].append(event.time)
            data['event_type'].append('capture')
            data['enemy_id'].append(event.enemy_id)
            data['friendly_pair'].append(f"{event.friendly_pair[0]}-{event.friendly_pair[1]}")
            data['position_x'].append(event.position[0])
            data['position_y'].append(event.position[1])
            data['distance_to_center'].append(event.distance_to_center)

            # Find corresponding neutralized count
            idx = min(
                range(len(self.log.timestamps)),
                key=lambda i: abs(self.log.timestamps[i] - event.time)
            )
            data['neutralized_count'].append(self.log.neutralized_count[idx])
            data['remaining_enemies'].append(self.log.remaining_enemies[idx])

        # Add final state
        data['timestamp'].append(self.sim_time)
        data['event_type'].append('simulation_end')
        data['enemy_id'].append(None)
        data['friendly_pair'].append(None)
        data['position_x'].append(None)
        data['position_y'].append(None)
        data['distance_to_center'].append(None)
        data['neutralized_count'].append(self.neutralized_count)
        data['remaining_enemies'].append(len([e for e in self.enemies if e.is_active]))

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Log saved to {filename}")

    def run(self):
        """Main simulation loop"""
        print("=" * 60)
        print("USV Defense Simulator")
        print("=" * 60)
        print(f"Attack Pattern: {self.attack_pattern.value}")
        print(f"Enemies: {self.num_enemies}")
        print(f"Friendly Pairs: {self.num_friendly_pairs}")
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
            self.clock.tick(60)  # 60 FPS

        # Final stats
        remaining = self.total_enemies - self.neutralized_count
        rate = (self.neutralized_count / self.total_enemies * 100) if self.total_enemies > 0 else 0

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Duration: {self.sim_time:.2f} seconds")
        print(f"Enemies Neutralized: {self.neutralized_count}/{self.total_enemies}")
        print(f"Neutralization Rate: {rate:.1f}%")
        print(f"Enemies Remaining: {remaining}")

        # Breach analysis
        breached = [e for e in self.enemies if e.is_active and
                   PhysicsEngine.distance_to_point(e.x, e.y,
                       CONFIG["defense_center"][0], CONFIG["defense_center"][1]) < CONFIG["safe_zone_radii"][0]]
        if breached:
            print(f"ALERT: {len(breached)} enemies breached 1km safe zone!")

        print("=" * 60)

        # Save log
        self.save_log()

        pygame.quit()


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    """Main entry point with pattern selection"""
    print("\nUSV Defense Simulator - Pattern Selection")
    print("-" * 40)
    print("1. Concentrated Attack")
    print("2. Wave Attack")
    print("3. Diversionary Attack")
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

        pairs_input = input("Number of friendly pairs (2-8) [default: 8]: ").strip()
        num_pairs = int(pairs_input) if pairs_input else 8
        num_pairs = max(2, min(8, num_pairs))

    except (ValueError, KeyboardInterrupt):
        print("\nUsing default settings...")
        pattern = AttackPattern.CONCENTRATED
        num_enemies = 15
        num_pairs = 5

    sim = Simulation(
        attack_pattern=pattern,
        num_enemies=num_enemies,
        num_friendly_pairs=num_pairs
    )
    sim.run()


if __name__ == "__main__":
    main()
