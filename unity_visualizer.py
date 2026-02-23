"""
Unity Visualizer - pygame으로 Unity 전장 데이터 실시간 시각화
===========================================================
tactical_server.py의 --visualize 옵션으로 사용됩니다.
Unity 전장의 아군/적군/모선 위치, 클러스터, 배정 라인을 표시합니다.
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

from unity_bridge import BattlefieldState
from PARAM import CLUSTER_COLORS


class UnityVisualizer:
    """Unity 전장 데이터를 pygame으로 실시간 렌더링"""

    def __init__(self, width: int = 1200, height: int = 900):
        pygame.init()
        pygame.display.set_caption("Unity Tactical Visualizer")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # 폰트
        self.font_large = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_medium = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 13)

        # 뷰포트 설정 (Unity 좌표 → 화면 좌표)
        self.view_center_x = 0.0
        self.view_center_z = 0.0
        self.view_radius = 500.0  # 초기 뷰포트 반경 (Unity 미터)
        self._auto_fit = True

        # 색상
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_FRIENDLY = (0, 200, 255)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_MOTHERSHIP = (150, 150, 200)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_NET = (100, 255, 150)

    def world_to_screen(self, wx: float, wz: float) -> Tuple[int, int]:
        """Unity 월드 좌표 → 화면 좌표"""
        # 마진 확보
        margin = 60
        view_size = self.view_radius * 2
        if view_size < 1:
            view_size = 1

        sx = margin + (wx - self.view_center_x + self.view_radius) / view_size * (self.width - margin * 2)
        sy = margin + (wz - self.view_center_z + self.view_radius) / view_size * (self.height - margin * 2)
        # z축 반전 (Unity z+ = 화면 위)
        sy = self.height - sy
        return (int(sx), int(sy))

    def _auto_fit_view(self, state: BattlefieldState):
        """모든 오브젝트가 보이도록 뷰포트 자동 조정"""
        if not self._auto_fit or state is None:
            return

        all_x = []
        all_z = []

        # 모선
        all_x.append(state.mothership_x)
        all_z.append(state.mothership_z)

        # 아군
        for f in state.friendlies:
            all_x.append(f.x)
            all_z.append(f.z)

        # 적군
        for e in state.enemies:
            if e.is_active:
                all_x.append(e.x)
                all_z.append(e.z)

        if len(all_x) < 2:
            return

        min_x, max_x = min(all_x), max(all_x)
        min_z, max_z = min(all_z), max(all_z)

        self.view_center_x = (min_x + max_x) / 2
        self.view_center_z = (min_z + max_z) / 2

        range_x = max_x - min_x
        range_z = max_z - min_z
        self.view_radius = max(range_x, range_z) / 2 + 100  # 여유 마진

    def update(
        self,
        state: Optional[BattlefieldState],
        formation_name: str = "UNKNOWN",
        confidence: float = 0.0,
        num_clusters: int = 0,
        cluster_labels: np.ndarray = np.array([]),
        cluster_centers: Dict[int, Tuple[float, float]] = None,
        pair_assignments: Dict[int, int] = None,
        has_model: bool = False,
        formation_locked: bool = False,
    ) -> bool:
        """
        한 프레임 렌더링. quit 이벤트 발생 시 True 반환.
        """
        if cluster_centers is None:
            cluster_centers = {}
        if pair_assignments is None:
            pair_assignments = {}

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                if event.key == pygame.K_f:
                    self._auto_fit = not self._auto_fit

        # 뷰포트: 첫 프레임에서만 자동 조정 후 고정
        if state and self._auto_fit:
            self._auto_fit_view(state)
            self._auto_fit = False  # 첫 프레임 이후 고정

        # 배경
        self.screen.fill(self.COLOR_BG)

        # 그리드
        self._draw_grid()

        if state is not None:
            # 클러스터 오버레이
            if len(cluster_labels) > 0:
                self._draw_clusters(state, cluster_labels, cluster_centers)

            # 배정 라인
            if pair_assignments:
                self._draw_assignment_lines(state, pair_assignments, cluster_labels)

            # 모선
            self._draw_mothership(state.mothership_x, state.mothership_z)

            # 적군
            for i, e in enumerate(state.enemies):
                if e.is_active:
                    color = self.COLOR_ENEMY
                    # 클러스터 색상
                    if i < len(cluster_labels) and cluster_labels[i] >= 0:
                        cid = cluster_labels[i]
                        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                    self._draw_ship(e.x, e.z, e.heading, color, e.id)

            # 아군
            for f in state.friendlies:
                self._draw_ship(f.x, f.z, f.heading, self.COLOR_FRIENDLY, f.id)

            # 아군 그물 라인
            if len(state.friendlies) >= 2:
                for i in range(0, len(state.friendlies) - 1, 2):
                    f1 = state.friendlies[i]
                    f2 = state.friendlies[i + 1]
                    p1 = self.world_to_screen(f1.x, f1.z)
                    p2 = self.world_to_screen(f2.x, f2.z)
                    pygame.draw.line(self.screen, self.COLOR_NET, p1, p2, 2)

        # UI 패널
        self._draw_info_panel(state, formation_name, confidence, num_clusters,
                              has_model, formation_locked, pair_assignments)

        pygame.display.flip()
        self.clock.tick(30)
        return False

    def _draw_grid(self):
        """배경 그리드"""
        # 뷰 범위에 따른 그리드 간격 계산
        grid_spacing = 100
        if self.view_radius > 500:
            grid_spacing = 200
        if self.view_radius > 2000:
            grid_spacing = 500

        start_x = int((self.view_center_x - self.view_radius) // grid_spacing) * grid_spacing
        end_x = int((self.view_center_x + self.view_radius) // grid_spacing + 1) * grid_spacing
        start_z = int((self.view_center_z - self.view_radius) // grid_spacing) * grid_spacing
        end_z = int((self.view_center_z + self.view_radius) // grid_spacing + 1) * grid_spacing

        x = start_x
        while x <= end_x:
            p1 = self.world_to_screen(x, start_z)
            p2 = self.world_to_screen(x, end_z)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
            x += grid_spacing

        z = start_z
        while z <= end_z:
            p1 = self.world_to_screen(start_x, z)
            p2 = self.world_to_screen(end_x, z)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
            z += grid_spacing

    def _draw_mothership(self, x: float, z: float):
        """모선 그리기"""
        pos = self.world_to_screen(x, z)
        pygame.draw.rect(self.screen, self.COLOR_MOTHERSHIP,
                         (pos[0] - 12, pos[1] - 12, 24, 24), 2)
        label = self.font_small.render("MS", True, self.COLOR_MOTHERSHIP)
        self.screen.blit(label, (pos[0] - 8, pos[1] - 25))

    def _draw_ship(self, x: float, z: float, heading: float, color, ship_id: str = ""):
        """선박 삼각형 그리기"""
        pos = self.world_to_screen(x, z)
        size = 8

        # heading (degrees) → 삼각형 방향
        rad = math.radians(-heading + 90)  # Unity Y축 회전 → 화면 좌표
        cos_h = math.cos(rad)
        sin_h = math.sin(rad)

        # 삼각형 꼭짓점
        tip = (pos[0] + cos_h * size * 1.5, pos[1] - sin_h * size * 1.5)
        left = (pos[0] - cos_h * size + sin_h * size * 0.7,
                pos[1] + sin_h * size + cos_h * size * 0.7)
        right = (pos[0] - cos_h * size - sin_h * size * 0.7,
                 pos[1] + sin_h * size - cos_h * size * 0.7)

        pygame.draw.polygon(self.screen, color, [tip, left, right])

        # ID 라벨
        if ship_id:
            label = self.font_small.render(ship_id, True, color)
            self.screen.blit(label, (pos[0] + 10, pos[1] - 5))

    def _draw_clusters(self, state, cluster_labels, cluster_centers):
        """클러스터 영역 표시"""
        active_enemies = [e for e in state.enemies if e.is_active]

        for cid, center in cluster_centers.items():
            color = CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]
            screen_pos = self.world_to_screen(center[0], center[1])

            # 클러스터 중심 마커
            pygame.draw.circle(self.screen, color, screen_pos, 6, 2)

            # 클러스터 ID 라벨
            label = self.font_small.render(f"C{cid}", True, color)
            self.screen.blit(label, (screen_pos[0] - 8, screen_pos[1] - 20))

    def _draw_assignment_lines(self, state, pair_assignments, cluster_labels):
        """배정 라인 (아군 쌍 → 타겟 적군)"""
        active_enemies = [e for e in state.enemies if e.is_active]

        for pair_idx, enemy_idx in pair_assignments.items():
            # 아군 쌍 중심
            f_start = pair_idx * 2
            if f_start + 1 >= len(state.friendlies):
                continue
            f1 = state.friendlies[f_start]
            f2 = state.friendlies[f_start + 1]
            pair_center = self.world_to_screen((f1.x + f2.x) / 2, (f1.z + f2.z) / 2)

            # 타겟 적군
            if 0 <= enemy_idx < len(active_enemies):
                enemy = active_enemies[enemy_idx]
                enemy_pos = self.world_to_screen(enemy.x, enemy.z)

                # 클러스터 색상
                color = (255, 255, 100)
                if enemy_idx < len(cluster_labels) and cluster_labels[enemy_idx] >= 0:
                    cid = cluster_labels[enemy_idx]
                    color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]

                # 점선 그리기
                self._draw_dashed_line(pair_center, enemy_pos, color)

                # 쌍 라벨
                label = self.font_small.render(f"P{pair_idx}", True, color)
                self.screen.blit(label, (pair_center[0] - 10, pair_center[1] - 20))

    def _draw_dashed_line(self, start, end, color, dash=8, gap=5):
        """점선 그리기"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1:
            return
        dx /= dist
        dy /= dist
        pos = 0
        drawing = True
        while pos < dist:
            seg = dash if drawing else gap
            next_pos = min(pos + seg, dist)
            if drawing:
                sx = int(start[0] + dx * pos)
                sy = int(start[1] + dy * pos)
                ex = int(start[0] + dx * next_pos)
                ey = int(start[1] + dy * next_pos)
                pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), 2)
            pos = next_pos
            drawing = not drawing

    def _draw_info_panel(self, state, formation_name, confidence, num_clusters,
                         has_model, formation_locked, pair_assignments):
        """정보 패널"""
        panel_w = 300
        panel_h = 200
        panel_x = 10
        panel_y = 10

        # 반투명 배경
        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((25, 25, 40, 200))
        border = (60, 150, 60) if formation_locked else (60, 90, 60)
        pygame.draw.rect(surf, border, (0, 0, panel_w, panel_h), 2)
        self.screen.blit(surf, (panel_x, panel_y))

        y = panel_y + 10
        line_h = 24

        title = self.font_medium.render("Unity Tactical Viewer", True, (100, 255, 150))
        self.screen.blit(title, (panel_x + 15, y))
        y += line_h + 5

        # 연결 상태
        connected = state is not None
        status_color = (100, 255, 100) if connected else (255, 100, 100)
        status_text = "Connected" if connected else "Waiting..."
        text = self.font_small.render(f"Unity: {status_text}", True, status_color)
        self.screen.blit(text, (panel_x + 15, y))
        y += line_h

        # 모델 상태
        model_text = "Neural Net" if has_model else "Rule-Based"
        model_color = (100, 255, 100) if has_model else (255, 255, 100)
        text = self.font_small.render(f"Mode: {model_text}", True, model_color)
        self.screen.blit(text, (panel_x + 15, y))
        y += line_h

        # 포메이션
        lock_str = " [LOCKED]" if formation_locked else ""
        text = self.font_small.render(f"Formation: {formation_name}{lock_str}", True, (255, 200, 100))
        self.screen.blit(text, (panel_x + 15, y))
        y += line_h

        # 신뢰도
        conf_color = (100, 255, 100) if confidence > 0.8 else (255, 255, 100)
        text = self.font_small.render(f"Confidence: {confidence:.1%}", True, conf_color)
        self.screen.blit(text, (panel_x + 15, y))
        y += line_h

        # 클러스터/배정
        text = self.font_small.render(f"Clusters: {num_clusters} | Pairs: {len(pair_assignments)}", True, self.COLOR_TEXT)
        self.screen.blit(text, (panel_x + 15, y))
        y += line_h

        # 전장 정보
        if state:
            active_enemies = sum(1 for e in state.enemies if e.is_active)
            text = self.font_small.render(
                f"Friendly: {len(state.friendlies)} | Enemy: {active_enemies}",
                True, self.COLOR_TEXT
            )
            self.screen.blit(text, (panel_x + 15, y))

    def quit(self):
        """pygame 종료"""
        pygame.quit()
