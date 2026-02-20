"""
Unity Bridge - TCP 양방향 통신 서버
===================================
Unity 전장 데이터를 수신하고, 전술 배정 결과를 전송합니다.

Protocol:
  Unity → Bridge: JSON lines (전장 데이터, 10Hz)
  Bridge → Unity: JSON lines (배정 결과, 변경 시에만)
"""

import socket
import json
import threading
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class UnityShipData:
    """Unity에서 수신한 선박 데이터"""
    id: str
    x: float  # Unity x
    z: float  # Unity z (→ Roonshot y)
    vx: float = 0.0
    vz: float = 0.0
    heading: float = 0.0  # degrees
    speed: float = 0.0
    is_active: bool = True
    tag: str = ""  # "Friendly", "attack_boat", "MotherShip"
    pair_id: str = ""


@dataclass
class BattlefieldState:
    """Unity 전장 상태 (파싱 완료)"""
    friendlies: List[UnityShipData] = field(default_factory=list)
    enemies: List[UnityShipData] = field(default_factory=list)
    mothership_x: float = 0.0
    mothership_z: float = 0.0
    time: float = 0.0
    frame: int = 0

    def get_enemy_positions(self) -> np.ndarray:
        """활성 적군 위치 배열 (N, 2) - Roonshot 좌표계"""
        active = [e for e in self.enemies if e.is_active]
        if not active:
            return np.zeros((0, 2))
        return np.array([[e.x, e.z] for e in active])

    def get_enemy_velocities(self) -> np.ndarray:
        """활성 적군 속도 배열 (N, 2)"""
        active = [e for e in self.enemies if e.is_active]
        if not active:
            return np.zeros((0, 2))
        return np.array([[e.vx, e.vz] for e in active])

    def get_friendly_pair_positions(self) -> np.ndarray:
        """아군 쌍 중심 위치 배열 (P, 2)"""
        # 2대씩 묶어서 쌍 중심 계산
        if len(self.friendlies) < 2:
            return np.zeros((0, 2))
        pairs = []
        for i in range(0, len(self.friendlies) - 1, 2):
            f1 = self.friendlies[i]
            f2 = self.friendlies[i + 1]
            cx = (f1.x + f2.x) / 2
            cz = (f1.z + f2.z) / 2
            pairs.append([cx, cz])
        return np.array(pairs) if pairs else np.zeros((0, 2))

    def get_mothership_pos(self) -> Tuple[float, float]:
        return (self.mothership_x, self.mothership_z)


@dataclass
class AssignmentData:
    """전술 배정 결과"""
    assignments: List[Dict] = field(default_factory=list)
    # [{"pair": ["F0", "F1"], "target_enemy_id": "E0", "cluster_id": 0}, ...]
    formation: str = "UNKNOWN"
    confidence: float = 0.0
    num_clusters: int = 0
    cluster_centers: Dict[int, Tuple[float, float]] = field(default_factory=dict)


class UnityBridgeServer:
    """Unity ↔ Roonshot TCP 브릿지 서버"""

    def __init__(self, port: int = 9877, on_data_received: Optional[Callable] = None):
        self.port = port
        self.on_data_received = on_data_received
        self._server_socket = None
        self._client_socket = None
        self._running = False
        self._thread = None
        self._recv_buffer = ""

        # 최신 전장 상태 (스레드 안전)
        self._lock = threading.Lock()
        self._latest_state: Optional[BattlefieldState] = None
        self._pending_assignment: Optional[str] = None  # 전송 대기 JSON

    def start(self):
        """서버 시작 (백그라운드 스레드)"""
        self._running = True
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()
        print(f"[UnityBridge] Server started on port {self.port}")

    def stop(self):
        """서버 종료"""
        self._running = False
        if self._client_socket:
            try:
                self._client_socket.close()
            except Exception:
                pass
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        print("[UnityBridge] Server stopped")

    def get_latest_state(self) -> Optional[BattlefieldState]:
        """최신 전장 상태 반환 (스레드 안전)"""
        with self._lock:
            return self._latest_state

    def send_assignment(self, assignment: AssignmentData):
        """배정 결과를 Unity로 전송"""
        data = {
            "type": "assignment",
            "assignments": assignment.assignments,
            "formation": assignment.formation,
            "confidence": assignment.confidence,
            "num_clusters": assignment.num_clusters,
            "cluster_centers": {
                str(k): list(v) for k, v in assignment.cluster_centers.items()
            },
        }
        msg = json.dumps(data) + "\n"
        with self._lock:
            self._pending_assignment = msg

    def _server_loop(self):
        """TCP 서버 메인 루프"""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(("0.0.0.0", self.port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(1.0)

        while self._running:
            try:
                print(f"[UnityBridge] Waiting for Unity connection on port {self.port}...")
                client, addr = self._server_socket.accept()
                self._client_socket = client
                client.settimeout(0.1)
                print(f"[UnityBridge] Unity connected from {addr}")
                self._handle_client(client)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[UnityBridge] Accept error: {e}")
                    time.sleep(1)

    def _handle_client(self, client: socket.socket):
        """클라이언트 연결 처리"""
        self._recv_buffer = ""
        while self._running:
            # 수신
            try:
                data = client.recv(8192).decode("utf-8")
                if not data:
                    print("[UnityBridge] Unity disconnected")
                    break
                self._recv_buffer += data
                self._process_buffer()
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[UnityBridge] Recv error: {e}")
                break

            # 대기 중인 배정 결과 전송
            with self._lock:
                pending = self._pending_assignment
                self._pending_assignment = None
            if pending:
                try:
                    client.sendall(pending.encode("utf-8"))
                except Exception as e:
                    print(f"[UnityBridge] Send error: {e}")
                    break

        self._client_socket = None

    def _process_buffer(self):
        """수신 버퍼에서 완전한 JSON 라인 파싱"""
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                state = self._parse_battlefield(data)
                with self._lock:
                    self._latest_state = state
                if self.on_data_received:
                    self.on_data_received(state)
            except json.JSONDecodeError:
                pass

    def _parse_battlefield(self, data: dict) -> BattlefieldState:
        """JSON → BattlefieldState 변환"""
        state = BattlefieldState()
        state.time = data.get("time", 0.0)
        state.frame = data.get("frame", 0)

        # 모선
        ms = data.get("mothership", {})
        state.mothership_x = ms.get("x", 0.0)
        state.mothership_z = ms.get("z", 0.0)

        # 아군
        for ship in data.get("friendlies", []):
            state.friendlies.append(UnityShipData(
                id=ship.get("id", ""),
                x=ship.get("x", 0.0),
                z=ship.get("z", 0.0),
                vx=ship.get("vx", 0.0),
                vz=ship.get("vz", 0.0),
                heading=ship.get("heading", 0.0),
                speed=ship.get("speed", 0.0),
                is_active=ship.get("active", True),
                tag="Friendly",
                pair_id=ship.get("pair_id", ""),
            ))

        # 적군
        for ship in data.get("enemies", []):
            state.enemies.append(UnityShipData(
                id=ship.get("id", ""),
                x=ship.get("x", 0.0),
                z=ship.get("z", 0.0),
                vx=ship.get("vx", 0.0),
                vz=ship.get("vz", 0.0),
                heading=ship.get("heading", 0.0),
                speed=ship.get("speed", 0.0),
                is_active=ship.get("active", True),
                tag="attack_boat",
            ))

        return state
