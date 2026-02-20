"""
Tactical Server - Unity 전장 데이터로 전술 배정 수행
====================================================
Unity에서 전장 데이터를 수신 → ML 포메이션 분류 → DBSCAN 클러스터링 → 배정 결과 전송

Usage:
  python tactical_server.py                     # 서버만 (headless)
  python tactical_server.py --visualize         # pygame 시각화 포함
  python tactical_server.py --port 9877         # 포트 지정
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Dict, Tuple, Optional

# Roonshot 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unity_bridge import UnityBridgeServer, BattlefieldState, AssignmentData
from ml.inference import DefenseMLSystem
from ml.labels import FormationClass, FORMATION_NAMES
from ml.ortools_assignment import (
    AssignmentMode,
    get_tactical_assignment,
    get_optimal_assignment,
    get_dynamic_assignment,
)
from ml.density import DensityAnalyzer

from PARAM import (
    DEFENSE_CENTER,
    FORMATION_LOCK_RADIUS,
    CLUSTER_COLORS,
    DYNAMIC_UPDATE_INTERVAL,
    ANGLE_WEIGHT,
)


class TacticalServer:
    """Unity 전장 데이터 기반 전술 배정 서버"""

    def __init__(
        self,
        port: int = 9877,
        model_path: str = "models/formation_classifier.pt",
        assignment_mode: str = AssignmentMode.ORTOOLS,
        visualize: bool = False,
        num_friendly_pairs: int = 1,
    ):
        self.port = port
        self.assignment_mode = assignment_mode
        self.visualize = visualize
        self.num_friendly_pairs = num_friendly_pairs

        # Unity 브릿지
        self.bridge = UnityBridgeServer(port=port)

        # ML 시스템 (모선 위치는 Unity에서 동적 수신)
        self._defense_center = DEFENSE_CENTER
        self.ml_system = DefenseMLSystem(
            center=self._defense_center,
            total_pairs=num_friendly_pairs,
        )
        self.ml_system.load_model(model_path)

        # 상태 추적
        self._last_assignment_time = 0.0
        self._assignment_interval = 1.0  # 초 (배정 업데이트 주기)
        self._last_state: Optional[BattlefieldState] = None
        self._cluster_labels = np.array([])
        self._formation_name = "UNKNOWN"
        self._confidence = 0.0
        self._num_clusters = 0
        self._cluster_centers: Dict[int, Tuple[float, float]] = {}
        self._pair_assignments: Dict[int, int] = {}

        # 시각화
        self._visualizer = None

    def run(self):
        """메인 서버 루프"""
        print("=" * 60)
        print("  Tactical Server for Unity BoatAttack")
        print(f"  Port: {self.port}")
        print(f"  Assignment Mode: {self.assignment_mode}")
        print(f"  Visualize: {self.visualize}")
        print(f"  ML Model: {'Loaded' if self.ml_system.has_model else 'Rule-Based'}")
        print("=" * 60)

        self.bridge.start()

        # 시각화 초기화
        if self.visualize:
            try:
                from unity_visualizer import UnityVisualizer
                self._visualizer = UnityVisualizer()
                print("[TacticalServer] Visualizer initialized")
            except ImportError as e:
                print(f"[TacticalServer] Visualizer unavailable: {e}")
                self.visualize = False

        try:
            while True:
                state = self.bridge.get_latest_state()

                if state is not None and state is not self._last_state:
                    self._last_state = state
                    self._process_state(state)

                # 시각화 업데이트
                if self.visualize and self._visualizer:
                    should_quit = self._visualizer.update(
                        state=state,
                        formation_name=self._formation_name,
                        confidence=self._confidence,
                        num_clusters=self._num_clusters,
                        cluster_labels=self._cluster_labels,
                        cluster_centers=self._cluster_centers,
                        pair_assignments=self._pair_assignments,
                        has_model=self.ml_system.has_model,
                        formation_locked=self.ml_system.formation_locked,
                    )
                    if should_quit:
                        break
                else:
                    time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[TacticalServer] Stopped by user")
        finally:
            self.bridge.stop()
            if self._visualizer:
                self._visualizer.quit()

    def _process_state(self, state: BattlefieldState):
        """전장 상태 처리 → ML 분류 → 배정"""
        enemy_positions = state.get_enemy_positions()
        enemy_velocities = state.get_enemy_velocities()
        pair_positions = state.get_friendly_pair_positions()

        if len(enemy_positions) == 0:
            return

        # 모선 위치 업데이트 (Unity 데이터 기반)
        ms_pos = state.get_mothership_pos()
        if ms_pos != (0.0, 0.0):
            self._defense_center = ms_pos
            self.ml_system.center = np.array(ms_pos, dtype=np.float64)

        # ML 시스템 업데이트 (포메이션 분류 + 클러스터링)
        decision = self.ml_system.update(
            enemy_positions, enemy_velocities, pair_positions
        )

        if decision is None:
            return

        # 결과 추출
        self._formation_name = decision.formation_class.name
        self._confidence = decision.confidence
        self._num_clusters = len(decision.cluster_metrics)

        # 클러스터 레이블/중심 추출
        if decision.cluster_metrics:
            labels = np.full(len(enemy_positions), -1, dtype=int)
            centers = {}
            for cm in decision.cluster_metrics:
                for idx in cm.member_indices:
                    if idx < len(labels):
                        labels[idx] = cm.cluster_id
                centers[cm.cluster_id] = (cm.center[0], cm.center[1])
            self._cluster_labels = labels
            self._cluster_centers = centers

        # 배정 결과 추출
        self._pair_assignments = {}
        if decision.pair_target_enemy:
            self._pair_assignments = dict(decision.pair_target_enemy)

        # 배정 결과를 Unity로 전송
        current_time = time.time()
        if current_time - self._last_assignment_time >= self._assignment_interval:
            self._last_assignment_time = current_time
            self._send_assignment(state, decision)

    def _send_assignment(self, state, decision):
        """배정 결과를 Unity Bridge를 통해 전송"""
        assignment = AssignmentData()
        assignment.formation = self._formation_name
        assignment.confidence = self._confidence
        assignment.num_clusters = self._num_clusters
        assignment.cluster_centers = self._cluster_centers

        # 배정 매핑 생성
        assignments = []
        active_enemies = [e for e in state.enemies if e.is_active]

        for pair_idx, enemy_idx in self._pair_assignments.items():
            pair_ids = []
            # 아군 쌍 ID 매핑
            f_start = pair_idx * 2
            if f_start < len(state.friendlies):
                pair_ids.append(state.friendlies[f_start].id)
            if f_start + 1 < len(state.friendlies):
                pair_ids.append(state.friendlies[f_start + 1].id)

            # 적군 ID 매핑
            target_id = ""
            if 0 <= enemy_idx < len(active_enemies):
                target_id = active_enemies[enemy_idx].id

            # 클러스터 ID
            cluster_id = -1
            if enemy_idx < len(self._cluster_labels):
                cluster_id = int(self._cluster_labels[enemy_idx])

            assignments.append({
                "pair": pair_ids,
                "target_enemy_id": target_id,
                "cluster_id": cluster_id,
            })

        assignment.assignments = assignments
        self.bridge.send_assignment(assignment)

        if assignments:
            print(f"[Tactical] {self._formation_name} (conf={self._confidence:.1%}) "
                  f"| {len(assignments)} assignments sent")


def main():
    parser = argparse.ArgumentParser(description="Tactical Server for Unity BoatAttack")
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument("--visualize", action="store_true",
                        help="Enable pygame visualization")
    parser.add_argument("--model", type=str, default="models/formation_classifier.pt",
                        help="Path to ML model")
    parser.add_argument("--mode", type=str, default="ortools",
                        choices=["ortools", "ortools_dynamic", "scipy"],
                        help="Assignment algorithm")
    parser.add_argument("--pairs", type=int, default=1,
                        help="Number of friendly pairs")
    args = parser.parse_args()

    server = TacticalServer(
        port=args.port,
        model_path=args.model,
        assignment_mode=args.mode,
        visualize=args.visualize,
        num_friendly_pairs=args.pairs,
    )
    server.run()


if __name__ == "__main__":
    main()
