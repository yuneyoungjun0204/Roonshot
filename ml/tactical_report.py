"""
Tactical Report Generator for LLM-based Command System

Generates natural language tactical situation reports from:
  - Friendly agent positions and headings
  - Enemy cluster data from DBSCAN
  - Formation classification results

The report uses formal military language (격식체) suitable for
command decision-making.

Note: Pygame coordinate system has Y-axis inverted (positive Y = down).
Angles are calculated accordingly.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ml.constants import DEFENSE_CENTER, ENEMY_SPEED
from ml.labels import FormationClass


@dataclass
class AgentState:
    """State of a friendly USV pair."""
    id: int
    position: Tuple[float, float]  # (x, y)
    heading_deg: float  # degrees, 0=East, 90=North (Pygame: 90=South)


@dataclass
class ClusterState:
    """State of an enemy cluster from DBSCAN."""
    cluster_id: int
    center: Tuple[float, float]  # (x, y)
    count: int  # number of enemies
    velocity: Tuple[float, float]  # (vx, vy) average velocity
    enemy_indices: List[int]  # indices of enemies in this cluster


def _normalize_angle(angle_deg: float) -> float:
    """Normalize angle to [-180, 180] range."""
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


def _calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _calculate_bearing(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
    """
    Calculate bearing angle from one point to another.
    Returns degrees: 0=East, 90=North (in standard math coords).
    For Pygame (Y inverted): 0=East, -90=North on screen.
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    # Note: Pygame has Y increasing downward, so we negate dy for standard angles
    angle_rad = math.atan2(-dy, dx)  # Negate dy for Pygame coords
    return math.degrees(angle_rad)


def _calculate_relative_angle(heading_deg: float, bearing_deg: float) -> float:
    """
    Calculate the relative angle between agent's heading and target bearing.
    Returns: angle in degrees [-180, 180]. 0 = target directly ahead.
    """
    diff = bearing_deg - heading_deg
    return _normalize_angle(diff)


def _calculate_eta(distance: float, velocity: Tuple[float, float],
                   target: Tuple[float, float], source: Tuple[float, float]) -> float:
    """
    Calculate estimated time of arrival to the defense center.
    Returns seconds, or float('inf') if not approaching.
    """
    speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    if speed < 1.0:
        return float('inf')

    # Vector from source to target (defense center)
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    dist_to_target = math.sqrt(dx ** 2 + dy ** 2)

    if dist_to_target < 1.0:
        return 0.0

    # Unit vector to target
    ux, uy = dx / dist_to_target, dy / dist_to_target

    # Approach speed (velocity component toward target)
    approach_speed = velocity[0] * ux + velocity[1] * uy

    if approach_speed <= 0:
        return float('inf')  # Moving away

    return dist_to_target / approach_speed


def _get_direction_name(angle_deg: float) -> str:
    """Convert angle to Korean cardinal direction name."""
    # Normalize to [0, 360)
    angle = angle_deg % 360

    if 337.5 <= angle or angle < 22.5:
        return "동쪽(E)"
    elif 22.5 <= angle < 67.5:
        return "북동쪽(NE)"
    elif 67.5 <= angle < 112.5:
        return "북쪽(N)"
    elif 112.5 <= angle < 157.5:
        return "북서쪽(NW)"
    elif 157.5 <= angle < 202.5:
        return "서쪽(W)"
    elif 202.5 <= angle < 247.5:
        return "남서쪽(SW)"
    elif 247.5 <= angle < 292.5:
        return "남쪽(S)"
    else:
        return "남동쪽(SE)"


def _get_angle_assessment(relative_angle: float) -> str:
    """Get assessment of relative angle for engagement."""
    abs_angle = abs(relative_angle)
    if abs_angle <= 45:
        return "즉시 기동 가능"
    elif abs_angle <= 90:
        return "소폭 선회 필요"
    elif abs_angle <= 135:
        return "대폭 선회 필요"
    else:
        return "완전 반전 필요 (비효율)"


def _format_distance(distance: float) -> str:
    """Format distance in meters or kilometers."""
    if distance >= 1000:
        return f"{distance / 1000:.1f}km"
    return f"{distance:.0f}m"


def _format_eta(eta: float) -> str:
    """Format ETA in seconds or minutes."""
    if eta == float('inf'):
        return "접근 중 아님"
    if eta >= 60:
        minutes = int(eta // 60)
        seconds = int(eta % 60)
        return f"{minutes}분 {seconds}초"
    return f"{eta:.0f}초"


class TacticalReportGenerator:
    """
    Generates natural language tactical situation reports for LLM consumption.
    """

    def __init__(self, defense_center: Tuple[float, float] = DEFENSE_CENTER):
        self.defense_center = defense_center

    def generate_report(
        self,
        agents: List[AgentState],
        clusters: List[ClusterState],
        formation: FormationClass,
        confidence: float,
    ) -> str:
        """
        Generate a comprehensive tactical situation report.

        Args:
            agents: List of friendly agent states
            clusters: List of enemy cluster states
            formation: Classified formation type
            confidence: Classification confidence (0-1)

        Returns:
            Formatted tactical report string
        """
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("【전술 상황 보고서 (TACTICAL SITUATION REPORT)】")
        lines.append("=" * 60)
        lines.append("")

        # 1. Formation Assessment
        lines.append("■ 적 포메이션 판정")
        lines.append("-" * 40)
        formation_kr = {
            FormationClass.CONCENTRATED: "집중형 (CONCENTRATED)",
            FormationClass.WAVE: "파상형 (WAVE)",
            FormationClass.DIVERSIONARY: "양동형 (DIVERSIONARY)",
        }.get(formation, "미상")
        lines.append(f"  포메이션: {formation_kr}")
        lines.append(f"  신뢰도: {confidence * 100:.1f}%")
        lines.append("")

        # 2. Enemy Cluster Summary
        lines.append("■ 적 군집 현황")
        lines.append("-" * 40)
        lines.append(f"  탐지 군집 수: {len(clusters)}개")

        total_enemies = sum(c.count for c in clusters)
        lines.append(f"  총 적 USV 수: {total_enemies}척")
        lines.append("")

        # 3. Detailed Cluster Analysis
        for cluster in sorted(clusters, key=lambda c: c.cluster_id):
            lines.append(f"  【군집 {cluster.cluster_id}】")

            # Distance and direction from defense center
            dist = _calculate_distance(cluster.center, self.defense_center)
            bearing = _calculate_bearing(self.defense_center, cluster.center)
            direction = _get_direction_name(bearing)

            lines.append(f"    위치: {direction} 방향, 거리 {_format_distance(dist)}")
            lines.append(f"    적 수: {cluster.count}척")

            # Velocity and ETA
            speed = math.sqrt(cluster.velocity[0] ** 2 + cluster.velocity[1] ** 2)
            eta = _calculate_eta(dist, cluster.velocity, self.defense_center, cluster.center)
            lines.append(f"    속도: {speed:.1f} m/s")
            lines.append(f"    예상 도착 시간(ETA): {_format_eta(eta)}")
            lines.append("")

        # 4. Friendly Force Status
        lines.append("■ 아군 전력 현황")
        lines.append("-" * 40)
        lines.append(f"  가용 USV 쌍: {len(agents)}개")
        lines.append("")

        # 5. Pair-Cluster Analysis Matrix
        lines.append("■ 아군-적 군집 분석")
        lines.append("-" * 40)

        for agent in sorted(agents, key=lambda a: a.id):
            lines.append(f"  【아군 쌍 {agent.id}】")
            lines.append(f"    현재 위치: ({agent.position[0]:.0f}, {agent.position[1]:.0f})")
            lines.append(f"    헤딩: {agent.heading_deg:.0f}°")
            lines.append("")

            for cluster in sorted(clusters, key=lambda c: c.cluster_id):
                dist = _calculate_distance(agent.position, cluster.center)
                bearing = _calculate_bearing(agent.position, cluster.center)
                rel_angle = _calculate_relative_angle(agent.heading_deg, bearing)
                angle_assessment = _get_angle_assessment(rel_angle)

                lines.append(f"      → 군집 {cluster.cluster_id}:")
                lines.append(f"         거리: {_format_distance(dist)}")
                lines.append(f"         방위각: {bearing:.0f}° ({_get_direction_name(bearing)})")
                lines.append(f"         상대각도: {rel_angle:.0f}° - {angle_assessment}")
                lines.append("")

        # 6. Tactical Recommendations (summary for LLM)
        lines.append("■ 전술 권고사항")
        lines.append("-" * 40)

        if formation == FormationClass.CONCENTRATED:
            lines.append("  - 집중형 공격 감지: 해당 방향에 전력 집중 필요")
            lines.append("  - 각도 차이가 작은 쌍 우선 배정 권장")
            lines.append("  - 예비 쌍 1개는 모선 근접 경계 유지")
        elif formation == FormationClass.WAVE:
            lines.append("  - 파상형 공격 감지: ETA 순 순차 요격 필요")
            lines.append("  - 1차 파에 주력, 2차 파는 순차 전환")
            lines.append("  - 파 간 시간 간격 활용하여 대응")
        elif formation == FormationClass.DIVERSIONARY:
            lines.append("  - 양동형 공격 감지: 균등 분산 배치 필요")
            lines.append("  - 각 군집에 최소 1쌍 배정")
            lines.append("  - 가장 위협적 방향에 예비 전력 배치")

        lines.append("")
        lines.append("=" * 60)
        lines.append("【보고 종료】")
        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_mapping_context(
        self,
        agents: List[AgentState],
        clusters: List[ClusterState],
        formation: FormationClass,
    ) -> Dict:
        """
        Generate structured context data for LLM mapping decision.

        Returns a dictionary with all relevant tactical metrics.
        """
        context = {
            "formation": formation.name,
            "num_agents": len(agents),
            "num_clusters": len(clusters),
            "agents": [],
            "clusters": [],
            "pair_cluster_matrix": [],
        }

        # Agent details
        for agent in agents:
            context["agents"].append({
                "id": agent.id,
                "position": agent.position,
                "heading_deg": agent.heading_deg,
            })

        # Cluster details with ETAs
        for cluster in clusters:
            dist = _calculate_distance(cluster.center, self.defense_center)
            eta = _calculate_eta(dist, cluster.velocity, self.defense_center, cluster.center)
            bearing = _calculate_bearing(self.defense_center, cluster.center)

            context["clusters"].append({
                "cluster_id": cluster.cluster_id,
                "center": cluster.center,
                "count": cluster.count,
                "distance": dist,
                "bearing_deg": bearing,
                "eta_seconds": eta if eta != float('inf') else -1,
                "direction": _get_direction_name(bearing),
            })

        # Pair-cluster analysis matrix
        for agent in agents:
            agent_row = {"agent_id": agent.id, "cluster_scores": []}
            for cluster in clusters:
                dist = _calculate_distance(agent.position, cluster.center)
                bearing = _calculate_bearing(agent.position, cluster.center)
                rel_angle = _calculate_relative_angle(agent.heading_deg, bearing)

                # Calculate engagement score (higher = better match)
                dist_score = 1.0 - min(dist / 10000.0, 1.0)  # Normalize to 10km max
                angle_score = math.cos(math.radians(rel_angle))  # -1 to 1
                angle_score = (angle_score + 1) / 2  # Normalize to 0-1

                total_score = 0.4 * dist_score + 0.6 * angle_score

                agent_row["cluster_scores"].append({
                    "cluster_id": cluster.cluster_id,
                    "distance": dist,
                    "relative_angle": rel_angle,
                    "engagement_score": total_score,
                    "assessment": _get_angle_assessment(rel_angle),
                })

            context["pair_cluster_matrix"].append(agent_row)

        return context


def prepare_llm_input(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass = FormationClass.DIVERSIONARY,
    confidence: float = 0.8,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
) -> Tuple[str, Dict]:
    """
    Main entry point for preparing LLM input from Pygame simulation data.

    Args:
        agents: List of dicts with keys: 'id', 'pos' (x,y), 'angle' (degrees)
        clusters: List of dicts with keys: 'cluster_id', 'center' (x,y),
                  'count', 'velocity' (vx,vy), 'enemy_indices' (optional)
        formation: Classified formation type
        confidence: Classification confidence
        defense_center: Defense center coordinates

    Returns:
        Tuple of (report_text, structured_context)
    """
    # Convert input dicts to dataclasses
    agent_states = [
        AgentState(
            id=a['id'],
            position=tuple(a['pos']),
            heading_deg=a['angle'],
        )
        for a in agents
    ]

    cluster_states = [
        ClusterState(
            cluster_id=c['cluster_id'],
            center=tuple(c['center']),
            count=c['count'],
            velocity=tuple(c['velocity']),
            enemy_indices=c.get('enemy_indices', []),
        )
        for c in clusters
    ]

    # Generate report and context
    generator = TacticalReportGenerator(defense_center)
    report = generator.generate_report(agent_states, cluster_states, formation, confidence)
    context = generator.generate_mapping_context(agent_states, cluster_states, formation)

    return report, context
