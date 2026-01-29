"""
Naval Tactical Doctrine for USV Defense Operations (해상 USV 방어 작전 교범)

This module contains the tactical doctrine knowledge base for the LLM-RAG system.
The doctrine provides principles for:
  1. Formation-specific engagement strategies
  2. Pair-to-cluster optimal mapping rules
  3. Distance, angle, and ETA-based prioritization
  4. Net capture interception tactics

Based on the USV Defense Simulator project's three formation types:
  - CONCENTRATED (집중형): Dense single-direction attack
  - WAVE (파상형): Layered sequential waves from same direction
  - DIVERSIONARY (양동형): Scattered multi-directional approach
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DoctrineChunk:
    """A single chunk of doctrine text with metadata."""
    id: str
    title: str
    content: str
    category: str  # "general", "concentrated", "wave", "diversionary", "mapping"
    priority: int  # 1=highest, 5=lowest


# =============================================================================
# NAVAL TACTICAL DOCTRINE KNOWLEDGE BASE
# =============================================================================

DOCTRINE_CHUNKS: List[DoctrineChunk] = [
    # -------------------------------------------------------------------------
    # GENERAL PRINCIPLES (일반 원칙)
    # -------------------------------------------------------------------------
    DoctrineChunk(
        id="GEN-001",
        title="기본 방어 원칙",
        content="""
【기본 방어 원칙】
1. 모선(Mothership) 보호가 최우선 임무이다.
2. 적 USV의 5km 진입 시점에 포메이션을 확정하고 요격 명령을 하달한다.
3. 아군 USV 쌍(Pair)은 그물망(Net) 포획 방식으로 적을 무력화한다.
4. 각 아군 쌍은 하나의 적 군집(Cluster)에 배정되어 해당 군집 내 적만 추적한다.
5. 요격 우선순위: ETA(예상 도착 시간)가 짧은 위협부터 처리한다.
""",
        category="general",
        priority=1
    ),

    DoctrineChunk(
        id="GEN-002",
        title="아군 쌍 배치 원칙",
        content="""
【아군 쌍(Pair) 배치 원칙】
1. 초기 배치: 모선 주변 4방향(N, E, S, W)에 균등 배치, 외측을 향해 대기
2. 명령 하달 전까지 정지 상태 유지 (불필요한 기동 금지)
3. 5km 진입 감지 시 LLM 지휘관의 매핑 명령에 따라 기동 개시
4. 각 쌍은 배정된 군집 방향으로만 이동하며, 타 군집 적은 무시한다.
5. 쌍 간 협력: 동일 군집 배정 시, 가까운 적부터 순차 요격
""",
        category="general",
        priority=1
    ),

    DoctrineChunk(
        id="GEN-003",
        title="거리 및 각도 고려사항",
        content="""
【거리 및 각도 기반 매핑 원칙】
1. 거리(Distance): 아군-적 군집 간 직선 거리가 가까울수록 유리
2. 상대 각도(Relative Angle): 아군 헤딩과 적 방향 차이
   - 0°~45°: 즉시 기동 가능 (최적)
   - 45°~90°: 소폭 선회 필요 (양호)
   - 90°~135°: 대폭 선회 필요 (주의)
   - 135°~180°: 완전 반전 필요 (회피 권장)
3. 회전 시간 고려: 각도가 클수록 요격 지연 발생
4. 매핑 점수 = 거리_점수 × (1 - 각도_패널티)
""",
        category="mapping",
        priority=2
    ),

    DoctrineChunk(
        id="GEN-004",
        title="ETA 기반 우선순위",
        content="""
【ETA(예상 도착 시간) 기반 우선순위】
1. ETA = 적_거리 / 적_속도 (적의 모선 도달 예상 시간)
2. ETA가 가장 짧은 군집 = 가장 긴급한 위협
3. 긴급 위협 군집에 더 많은 아군 쌍 배정
4. ETA 기반 배정 공식:
   - ETA < 60초: 최우선 (쌍 2~3개 배정)
   - 60초 ≤ ETA < 120초: 우선 (쌍 1~2개 배정)
   - ETA ≥ 120초: 일반 (쌍 1개 배정)
5. 복수 군집 동시 접근 시, ETA 순으로 순차 대응
""",
        category="mapping",
        priority=2
    ),

    # -------------------------------------------------------------------------
    # CONCENTRATED FORMATION (집중형)
    # -------------------------------------------------------------------------
    DoctrineChunk(
        id="CONC-001",
        title="집중형 포메이션 특성",
        content="""
【집중형(CONCENTRATED) 포메이션 특성】
1. 정의: 적이 좁은 각도(30° 이내)에서 밀집하여 단일 방향으로 돌진
2. 특징:
   - 단일 군집 형성 (DBSCAN 결과 클러스터 1개)
   - 높은 밀도, 짧은 거리 편차
   - ETA 편차 작음 (동시 도착 경향)
3. 위협도: 상(HIGH) - 집중 화력으로 방어선 돌파 시도
4. 식별 지표:
   - angular_std < 30°
   - distance_std < 400m
   - 단일 접근 벡터
""",
        category="concentrated",
        priority=1
    ),

    DoctrineChunk(
        id="CONC-002",
        title="집중형 대응 전술",
        content="""
【집중형(CONCENTRATED) 대응 전술】
1. 집중 방어: 해당 방향에 아군 쌍 2~3개 집중 배치
2. 다층 요격: 전방에 1쌍, 후방에 1~2쌍 배치하여 누수 방지
3. 매핑 원칙:
   - 적 접근 방향과 가장 가까운 쌍을 1차 요격조로 지정
   - 각도 차이가 작은 쌍 우선 배정 (선회 시간 최소화)
   - 예비 쌍 1개는 모선 근접 방어에 배치
4. 그물망 간격: 밀집 대형이므로 좁은 간격(40~60m) 유지
5. 주의: 한 방향에 전력 집중 시 측면 공백 발생 - 최소 1쌍은 반대편 경계
""",
        category="concentrated",
        priority=1
    ),

    # -------------------------------------------------------------------------
    # WAVE FORMATION (파상형)
    # -------------------------------------------------------------------------
    DoctrineChunk(
        id="WAVE-001",
        title="파상형 포메이션 특성",
        content="""
【파상형(WAVE) 포메이션 특성】
1. 정의: 같은 방향에서 시간차를 두고 연속적인 파도처럼 접근
2. 특징:
   - 복수 군집 형성 (거리 기준 2~3개 레이어)
   - 좁은 각도(30° 이내), 넓은 거리 편차(400m+)
   - ETA 편차 큼 (순차 도착)
3. 위협도: 중상(MEDIUM-HIGH) - 방어선 피로 유발 전술
4. 식별 지표:
   - angular_std < 30°
   - distance_std ≥ 400m
   - 동일 방향, 상이한 거리 레이어
""",
        category="wave",
        priority=1
    ),

    DoctrineChunk(
        id="WAVE-002",
        title="파상형 대응 전술",
        content="""
【파상형(WAVE) 대응 전술】
1. 순차 요격: 가장 가까운 파(1차 파)부터 순서대로 요격
2. 매핑 원칙:
   - 1차 파(가장 가까움): 쌍 2개 배정 - 신속 무력화
   - 2차 파: 쌍 1~2개 배정 - 1차 요격 완료 후 전환 가능
   - 3차 파: 쌍 1개 배정 또는 예비
3. 동적 재배치: 1차 파 요격 완료 후, 해당 쌍은 2차 파로 전환
4. 그물망 간격: 중간 간격(60~80m) - 파 간 이동 시간 확보
5. 핵심: 각 파 사이의 시간 간격을 활용하여 순차적 대응
   - 1차 파 ETA 기준 요격 → 2차 파 도착 전 완료 목표
""",
        category="wave",
        priority=1
    ),

    # -------------------------------------------------------------------------
    # DIVERSIONARY FORMATION (양동형)
    # -------------------------------------------------------------------------
    DoctrineChunk(
        id="DIV-001",
        title="양동형 포메이션 특성",
        content="""
【양동형(DIVERSIONARY) 포메이션 특성】
1. 정의: 적이 여러 방향에서 분산 접근하여 방어력 분산 유도
2. 특징:
   - 복수 군집 형성 (방향별 2~4개 클러스터)
   - 넓은 각도 분포(30°+)
   - 거리/속도 편차 큼
3. 위협도: 중(MEDIUM) - 단일 방향 화력은 약하나 전방위 위협
4. 식별 지표:
   - angular_std ≥ 30°
   - 다방향 접근 벡터
   - 클러스터 수 2개 이상
""",
        category="diversionary",
        priority=1
    ),

    DoctrineChunk(
        id="DIV-002",
        title="양동형 대응 전술",
        content="""
【양동형(DIVERSIONARY) 대응 전술】
1. 분산 방어: 각 군집에 최소 1쌍씩 균등 배치
2. 매핑 원칙:
   - 각 쌍은 가장 가까운 군집에 배정 (거리 우선)
   - 각도 패널티 적용: 반대 방향 군집 회피
   - ETA가 빠른 군집에 추가 1쌍 배정 고려
3. 우선순위 판단:
   - 실제 위협(Real Threat): 직선 경로로 모선 접근
   - 양동(Feint): 우회 경로 또는 느린 속도
   - 양동으로 판단 시 최소 병력만 배치
4. 그물망 간격: 넓은 간격(80~100m) - 넓은 수색 범위 확보
5. 예비 전력: 가장 위협적 방향에 예비 1쌍 배치
""",
        category="diversionary",
        priority=1
    ),

    # -------------------------------------------------------------------------
    # MAPPING DECISION RULES (매핑 결정 규칙)
    # -------------------------------------------------------------------------
    DoctrineChunk(
        id="MAP-001",
        title="최적 매핑 알고리즘",
        content="""
【최적 매핑 결정 알고리즘】
1. 입력 데이터:
   - 아군 쌍: {id, 위치(x,y), 헤딩각도}
   - 적 군집: {cluster_id, 중심좌표, 적 수, 평균속도, ETA}

2. 매핑 점수 계산 (각 쌍-군집 조합):
   점수 = w1×거리점수 + w2×각도점수 + w3×ETA점수

   - 거리점수 = 1 - (거리 / 최대거리)
   - 각도점수 = cos(상대각도) [1=정면, 0=측면, -1=후면]
   - ETA점수 = 1 - (ETA / 최대ETA)

   가중치 기본값: w1=0.4, w2=0.3, w3=0.3

3. 헝가리안 알고리즘으로 최적 매핑 결정

4. 제약 조건:
   - 각 쌍은 하나의 군집에만 배정
   - 군집당 최소 1쌍, 최대 3쌍
   - 각도 차이 135° 초과 시 해당 조합 제외
""",
        category="mapping",
        priority=1
    ),

    DoctrineChunk(
        id="MAP-002",
        title="포메이션별 가중치 조정",
        content="""
【포메이션별 매핑 가중치 조정】

1. 집중형(CONCENTRATED):
   - 거리 가중치: 0.3 (↓ 감소)
   - 각도 가중치: 0.5 (↑ 증가) - 신속 대응 중요
   - ETA 가중치: 0.2
   - 특수: 단일 군집이므로 쌍 집중 배치

2. 파상형(WAVE):
   - 거리 가중치: 0.3
   - 각도 가중치: 0.3
   - ETA 가중치: 0.4 (↑ 증가) - 파 순서 중요
   - 특수: ETA 순으로 군집 우선순위 결정

3. 양동형(DIVERSIONARY):
   - 거리 가중치: 0.5 (↑ 증가) - 근접 위협 우선
   - 각도 가중치: 0.3
   - ETA 가중치: 0.2
   - 특수: 실제 위협/양동 구분 후 배정
""",
        category="mapping",
        priority=2
    ),

    DoctrineChunk(
        id="MAP-003",
        title="예외 상황 처리",
        content="""
【예외 상황 처리 지침】

1. 쌍 수 부족 (군집 수 > 쌍 수):
   - ETA가 가장 긴 군집에는 쌍 미배정
   - 해당 군집은 1차 요격 후 잔여 쌍이 처리

2. 쌍 수 과잉 (군집 수 < 쌍 수):
   - 가장 위협적 군집에 추가 쌍 배정
   - 예비 쌍 1개는 모선 근접 경계

3. 적 전멸 후:
   - 모든 쌍 모선 귀환
   - 초기 4방향 대기 위치로 복귀

4. 요격 실패 (적 5km 내 돌파):
   - 해당 적 추적 우선
   - 인근 쌍 지원 투입
""",
        category="mapping",
        priority=3
    ),
]


def get_doctrine_by_category(category: str) -> List[DoctrineChunk]:
    """Get all doctrine chunks for a specific category."""
    return [d for d in DOCTRINE_CHUNKS if d.category == category]


def get_doctrine_by_formation(formation: str) -> List[DoctrineChunk]:
    """Get doctrine relevant to a specific formation type."""
    formation_lower = formation.lower()
    relevant = []

    # Always include general principles
    relevant.extend(get_doctrine_by_category("general"))
    relevant.extend(get_doctrine_by_category("mapping"))

    # Add formation-specific doctrine
    if "concentrated" in formation_lower or "집중" in formation_lower:
        relevant.extend(get_doctrine_by_category("concentrated"))
    elif "wave" in formation_lower or "파상" in formation_lower:
        relevant.extend(get_doctrine_by_category("wave"))
    elif "diversionary" in formation_lower or "양동" in formation_lower:
        relevant.extend(get_doctrine_by_category("diversionary"))
    else:
        # Unknown formation - include all
        relevant.extend(get_doctrine_by_category("concentrated"))
        relevant.extend(get_doctrine_by_category("wave"))
        relevant.extend(get_doctrine_by_category("diversionary"))

    # Sort by priority
    relevant.sort(key=lambda x: x.priority)
    return relevant


def get_all_doctrine_texts() -> List[str]:
    """Get all doctrine content as a list of text strings for vectorization."""
    return [f"[{d.id}] {d.title}\n{d.content}" for d in DOCTRINE_CHUNKS]


def get_doctrine_metadata() -> List[Dict]:
    """Get metadata for each doctrine chunk (for vector store)."""
    return [
        {
            "id": d.id,
            "title": d.title,
            "category": d.category,
            "priority": d.priority,
        }
        for d in DOCTRINE_CHUNKS
    ]
