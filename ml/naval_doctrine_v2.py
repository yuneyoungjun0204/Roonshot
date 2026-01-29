"""
Naval Tactical Doctrine v2.0 for USV Defense Operations
========================================================
해상 USV 방어 작전 교범 (RAG + LLM 최적화 버전)

This doctrine is specifically designed for:
  - RAG (Retrieval-Augmented Generation) vector search
  - LLM structured output parsing
  - Real-time tactical decision making

Document Structure:
  - Chapter 1: Core Mapping Principles (핵심 매핑 원칙)
  - Chapter 2: Formation-Specific Tactics (포메이션별 전술)
  - Chapter 3: Decision Algorithm (의사결정 알고리즘)
  - Chapter 4: Output Format Specification (출력 형식 규격)
  - Chapter 5: Example Scenarios (예시 시나리오)
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DoctrineChunk:
    """A single chunk of doctrine text with metadata for RAG."""
    id: str
    title: str
    content: str
    category: str
    tags: List[str]  # For enhanced retrieval
    priority: int    # 1=highest


# =============================================================================
# NAVAL TACTICAL DOCTRINE v2.0
# =============================================================================

DOCTRINE_CHUNKS_V2: List[DoctrineChunk] = [

    # =========================================================================
    # CHAPTER 1: CORE MAPPING PRINCIPLES (핵심 매핑 원칙)
    # =========================================================================

    DoctrineChunk(
        id="CORE-001",
        title="제1조: 최우선 임무",
        content="""
【제1조: 최우선 임무】

1. 모선(Mothership) 보호가 모든 작전의 최우선 목표이다.
2. 아군 USV의 손실보다 모선 방어를 우선한다.
3. 모든 매핑 결정은 "모선까지의 적 도달 시간(ETA)" 최소화를 기준으로 한다.

[핵심 원칙]
- 모선 방어 > 적 섬멸 > 아군 보존
- ETA가 가장 짧은 적 군집이 최우선 위협이다
""",
        category="core",
        tags=["mission", "priority", "mothership", "ETA"],
        priority=1
    ),

    DoctrineChunk(
        id="CORE-002",
        title="제2조: 실질 도달 시간 계산법",
        content="""
【제2조: 실질 도달 시간(Effective Arrival Time) 계산법】

단순 거리가 아닌 '실질 도달 시간'을 기준으로 매핑한다.

[계산 공식]
실질_도달_시간 = 직선_이동_시간 + 선회_시간

여기서:
- 직선_이동_시간 = 거리(m) / 아군_속도(m/s)
- 선회_시간 = |상대_각도| / 선회_속도(deg/s)

[기본 파라미터]
- 아군 USV 속도: 25 m/s
- 아군 USV 선회 속도: 약 30 deg/s (추정)

[교환 비율 - CRITICAL]
헤딩 각도가 10° 더 정렬된 경우 = 직선거리 15m 추가와 동등
→ 10° 각도 이점 ≈ 15m 거리 이점
→ 1° 각도 이점 ≈ 1.5m 거리 이점

[예시]
- 아군A: 거리 500m, 상대각도 30°
- 아군B: 거리 450m, 상대각도 60°

아군A 실질거리 = 500 + (30 × 1.5) = 545m (환산)
아군B 실질거리 = 450 + (60 × 1.5) = 540m (환산)
→ 아군B가 약간 유리하나, 거의 동등. 다른 요소 고려 필요.
""",
        category="core",
        tags=["EAT", "calculation", "heading", "distance", "trade-off"],
        priority=1
    ),

    DoctrineChunk(
        id="CORE-003",
        title="제3조: 전역 최적화 원칙",
        content="""
【제3조: 전역 최적화 원칙 (Global Optimization)】

개별 쌍의 최적 매핑보다 **전체 아군의 총 이동 비용 합산**이 최소화되는
전역 최적 매핑을 수행한다.

[원칙]
1. 개별 최적 ≠ 전역 최적
2. 한 쌍의 최단 경로가 다른 쌍의 경로를 방해하면 안 된다
3. 경로 교차(crossing)는 최소화한다

[전역 비용 함수]
Total_Cost = Σ (실질_도달_시간[i]) + Σ (경로_교차_패널티)

[경로 교차 패널티]
- 두 아군 쌍의 경로가 교차하면 +50m 패널티 (환산 거리)
- 이는 충돌 회피 기동으로 인한 지연을 반영

[예외]
- 긴급 위협(ETA < 30초)에는 전역 최적화보다 해당 위협 즉시 대응 우선
""",
        category="core",
        tags=["global", "optimization", "total_cost", "path_crossing"],
        priority=1
    ),

    DoctrineChunk(
        id="CORE-004",
        title="제4조: 군집당 병력 배분",
        content="""
【제4조: 군집당 병력 배분 기준】

적 군집의 규모와 위협도에 따라 아군 쌍을 배분한다.

[기본 배분 공식]
배정_쌍_수 = ceil(적_수 / 3)
단, 최소 1쌍, 최대 4쌍

[위협도 가중치]
- ETA < 60초: 기본 배분 × 1.5 (반올림)
- 60초 ≤ ETA < 120초: 기본 배분 × 1.0
- ETA ≥ 120초: 기본 배분 × 0.8 (반올림, 최소 1)

[예시]
- 군집A: 적 6척, ETA 50초 → ceil(6/3) × 1.5 = 3쌍
- 군집B: 적 4척, ETA 90초 → ceil(4/3) × 1.0 = 2쌍
- 군집C: 적 2척, ETA 150초 → ceil(2/3) × 0.8 = 1쌍

[쌍 부족 시 우선순위]
ETA가 짧은 군집부터 배정, 나머지 군집은 순차 대응
""",
        category="core",
        tags=["allocation", "pair_count", "threat_level", "ETA"],
        priority=2
    ),

    # =========================================================================
    # CHAPTER 2: FORMATION-SPECIFIC TACTICS (포메이션별 전술)
    # =========================================================================

    DoctrineChunk(
        id="FORM-CONC-001",
        title="집중형(CONCENTRATED) 공격 대응",
        content="""
【집중형(CONCENTRATED) 공격 대응 전술】

[식별 특징]
- 적이 좁은 각도(30° 이내)에서 밀집 접근
- 단일 군집 또는 매우 인접한 2개 군집
- 높은 동시 도달 위협

[대응 전략: 집중 방어]
1. 해당 방향에 가용 쌍의 60-70% 집중 배치
2. 나머지 30-40%는 측면 경계 유지

[매핑 우선순위]
1순위: 해당 방향을 바라보는 쌍 (상대각도 < 45°)
2순위: 가장 가까운 쌍 (거리 기준)
3순위: 나머지 쌍 (백업/측면 경계)

[그물망 간격]
- 권장: 40-60m (밀집 대형 대응)
- 적 간 평균 거리의 80% 이하로 설정

[주의사항]
- 전 병력 집중 금지: 최소 1쌍은 반대 방향 경계
- 후속 파도 가능성 항상 고려
""",
        category="concentrated",
        tags=["concentrated", "dense", "single_direction", "focus_defense"],
        priority=1
    ),

    DoctrineChunk(
        id="FORM-WAVE-001",
        title="파상형(WAVE) 공격 대응",
        content="""
【파상형(WAVE) 공격 대응 전술】

[식별 특징]
- 같은 방향에서 시간차 접근 (2-3개 파도)
- 거리 편차 큼 (400m+), 각도 편차 작음 (30° 이내)
- ETA 분산: 1진, 2진, 3진 순차 도달

[대응 전략: 동시 다발 출격 - CRITICAL]
⚠️ 순차 대응 금지! 모든 파도에 동시 매핑!

1. 1진(가장 가까운 파): 쌍 2-3개 즉시 배정
2. 2진(중간 파): 쌍 2개 동시 배정
3. 3진(후방 파): 쌍 1-2개 동시 배정

[핵심 원리]
- 1진 요격 완료 대기 후 2진 대응 → 비효율
- 모든 파도 동시 출격 → 각 파도 도달 전 차단 가능

[매핑 예시 - 9척, 3파도, 쌍 5개]
- 1진 (3척, ETA 60초): 쌍 0, 쌍 1 배정
- 2진 (3척, ETA 90초): 쌍 2, 쌍 3 배정
- 3진 (3척, ETA 120초): 쌍 4 배정

출력: [0=>0, 1=>0, 2=>1, 3=>1, 4=>2]

[그물망 간격]
- 권장: 60-80m (파도 간 이동 여유 확보)
""",
        category="wave",
        tags=["wave", "sequential", "simultaneous_deployment", "multi_wave"],
        priority=1
    ),

    DoctrineChunk(
        id="FORM-DIV-001",
        title="양동형(DIVERSIONARY) 공격 대응",
        content="""
【양동형(DIVERSIONARY) 공격 대응 전술】

[식별 특징]
- 적이 여러 방향에서 분산 접근 (각도 편차 30°+)
- 복수 군집 (2-4개)
- 일부는 양동, 일부는 실제 위협

[대응 전략: 전역 최적 분산]
1. 각 군집에 최소 1쌍 균등 배정
2. 실제 위협 군집에 추가 병력 집중

[실제 위협 vs 양동 판별]
실제 위협 징후:
- 모선을 향한 직선 경로
- 높은 접근 속도
- ETA가 짧음

양동 징후:
- 우회 경로
- 느린 속도 또는 불규칙 기동
- ETA가 김

[매핑 우선순위]
1. 전역 비용 최소화 (총 이동 거리 합 최소)
2. 실제 위협 군집에 가까운 쌍 우선 배정
3. 양동 군집에는 가장 가까운 쌍만 최소 배정

[예시 - 군집 3개, 쌍 5개]
- 군집0 (실제 위협, 5척): 쌍 2개
- 군집1 (양동 의심, 3척): 쌍 1개
- 군집2 (실제 위협, 4척): 쌍 2개

출력: [0=>0, 1=>0, 2=>1, 3=>2, 4=>2]

[그물망 간격]
- 권장: 80-100m (넓은 탐색 범위)
""",
        category="diversionary",
        tags=["diversionary", "scattered", "decoy", "real_threat", "global_optimal"],
        priority=1
    ),

    # =========================================================================
    # CHAPTER 3: DECISION ALGORITHM (의사결정 알고리즘)
    # =========================================================================

    DoctrineChunk(
        id="ALGO-001",
        title="매핑 의사결정 플로우차트",
        content="""
【매핑 의사결정 플로우차트】

[STEP 1: 포메이션 확인]
├─ CONCENTRATED → STEP 2A
├─ WAVE → STEP 2B
└─ DIVERSIONARY → STEP 2C

[STEP 2A: 집중형 처리]
1. 적 방향 식별
2. 해당 방향 쌍 우선 선별 (상대각도 < 45°)
3. 선별된 쌍 중 실질거리 짧은 순 배정
4. 60-70% 집중, 30-40% 측면 경계
→ 출력 생성

[STEP 2B: 파상형 처리]
1. 각 파도(군집) ETA 순 정렬
2. 모든 파도에 동시 병력 배분
3. 각 파도에 대해 실질거리 짧은 쌍 배정
4. 쌍-군집 매핑 동시 확정
→ 출력 생성

[STEP 2C: 양동형 처리]
1. 각 군집 위협도 평가 (ETA, 경로, 속도)
2. 전역 비용 매트릭스 계산
3. 총 비용 최소화 매핑 수행
4. 실제 위협에 추가 병력 조정
→ 출력 생성

[STEP 3: 출력 검증]
- 모든 쌍이 배정되었는가?
- 미배정 군집이 있는가? (쌍 부족 시 ETA 순 우선)
- 경로 교차가 과도하지 않은가?
→ 최종 출력
""",
        category="algorithm",
        tags=["flowchart", "decision", "step_by_step"],
        priority=1
    ),

    DoctrineChunk(
        id="ALGO-002",
        title="실질거리 계산 상세",
        content="""
【실질거리(Effective Distance) 계산 상세】

[입력]
- 아군 위치: (ax, ay)
- 아군 헤딩: heading_deg (0°=동, 90°=북)
- 적 군집 중심: (ex, ey)

[계산 절차]

1. 직선 거리 계산
   distance = sqrt((ex - ax)² + (ey - ay)²)

2. 방위각 계산 (아군→적)
   bearing = atan2(-(ey - ay), ex - ax) × (180/π)
   (Y축 반전 주의: Pygame 좌표계)

3. 상대 각도 계산
   relative_angle = bearing - heading_deg
   (범위 정규화: -180° ~ +180°)

4. 실질거리 계산
   effective_distance = distance + |relative_angle| × 1.5

[예시]
아군 위치: (9000, 9100), 헤딩: 90° (북쪽)
적 위치: (9000, 10500)

distance = 1400m
bearing = 90° (정북)
relative_angle = 90° - 90° = 0°
effective_distance = 1400 + 0 = 1400m

→ 정면 타겟이므로 실질거리 = 직선거리
""",
        category="algorithm",
        tags=["calculation", "effective_distance", "bearing", "relative_angle"],
        priority=2
    ),

    # =========================================================================
    # CHAPTER 4: OUTPUT FORMAT SPECIFICATION (출력 형식 규격)
    # =========================================================================

    DoctrineChunk(
        id="OUTPUT-001",
        title="출력 형식 규격 - CRITICAL",
        content="""
【출력 형식 규격 - 엄격 준수 필수】

[형식]
[쌍ID=>군집ID, 쌍ID=>군집ID, ...]

[규칙]
1. 대괄호 [] 로 시작하고 끝난다
2. 각 매핑은 "쌍ID=>군집ID" 형식
3. 매핑 간 구분자는 ", " (쉼표+공백)
4. 쌍ID와 군집ID는 정수
5. 추가 텍스트, 설명, 줄바꿈 금지

[올바른 예시]
[0=>0, 1=>0, 2=>1, 3=>2, 4=>2]
[0=>1, 1=>0, 2=>0]
[0=>0, 1=>1, 2=>2, 3=>3, 4=>0, 5=>1, 6=>2, 7=>3]

[잘못된 예시]
쌍0은 군집0에 배정합니다. (X - 텍스트 포함)
{0: 0, 1: 1} (X - 잘못된 형식)
[0->0, 1->1] (X - 잘못된 구분자)
[0=>0, 1=>1,] (X - 끝에 쉼표)

[빈 응답 불가]
최소 1개 이상의 매핑 필수. 쌍이 있으면 반드시 배정.
""",
        category="output",
        tags=["format", "output", "parsing", "strict"],
        priority=1
    ),

    # =========================================================================
    # CHAPTER 5: EXAMPLE SCENARIOS (예시 시나리오)
    # =========================================================================

    DoctrineChunk(
        id="EXAMPLE-001",
        title="예시: 집중형 공격 시나리오",
        content="""
【예시 시나리오 1: 집중형 공격】

[상황]
- 포메이션: CONCENTRATED
- 적 군집: 1개 (군집ID: 0)
  - 위치: 북쪽 방향
  - 적 수: 8척
  - ETA: 70초
- 아군 쌍: 5개
  - 쌍0: 북쪽, 상대각도 10°, 거리 800m
  - 쌍1: 동쪽, 상대각도 80°, 거리 600m
  - 쌍2: 남쪽, 상대각도 170°, 거리 750m
  - 쌍3: 서쪽, 상대각도 90°, 거리 650m
  - 쌍4: 북동쪽, 상대각도 35°, 거리 850m

[분석]
실질거리 계산:
- 쌍0: 800 + 10×1.5 = 815m ← 최우선
- 쌍1: 600 + 80×1.5 = 720m
- 쌍2: 750 + 170×1.5 = 1005m
- 쌍3: 650 + 90×1.5 = 785m
- 쌍4: 850 + 35×1.5 = 902.5m

배분: ceil(8/3) × 1.0 = 3쌍 → 군집0에 3쌍
측면 경계: 2쌍

[결정]
군집0에: 쌍0 (815m), 쌍1 (720m), 쌍3 (785m)
측면 경계: 쌍2, 쌍4 (가장 먼 실질거리)

실제로는 측면 경계 쌍도 군집0에 배정 (단일 군집이므로)

[출력]
[0=>0, 1=>0, 2=>0, 3=>0, 4=>0]
""",
        category="example",
        tags=["example", "concentrated", "scenario"],
        priority=3
    ),

    DoctrineChunk(
        id="EXAMPLE-002",
        title="예시: 파상형 공격 시나리오",
        content="""
【예시 시나리오 2: 파상형 공격】

[상황]
- 포메이션: WAVE
- 적 군집: 3개 (같은 방향, 거리 차이)
  - 군집0 (1진): 적 3척, ETA 50초, 북쪽 5000m
  - 군집1 (2진): 적 3척, ETA 80초, 북쪽 5800m
  - 군집2 (3진): 적 3척, ETA 110초, 북쪽 6600m
- 아군 쌍: 5개 (모두 모선 근처)
  - 쌍0: 북쪽, 상대각도 5°
  - 쌍1: 동쪽, 상대각도 85°
  - 쌍2: 남쪽, 상대각도 175°
  - 쌍3: 서쪽, 상대각도 95°
  - 쌍4: 북동쪽, 상대각도 40°

[분석]
파상형 → 동시 다발 출격!

병력 배분:
- 군집0 (1진, 긴급): 2쌍
- 군집1 (2진): 2쌍
- 군집2 (3진): 1쌍

북쪽 향하는 쌍 우선:
- 쌍0: 상대각도 5° → 군집0 (1진)
- 쌍4: 상대각도 40° → 군집0 (1진)
- 쌍1: 상대각도 85° → 군집1 (2진)
- 쌍3: 상대각도 95° → 군집1 (2진)
- 쌍2: 상대각도 175° → 군집2 (3진)

[출력]
[0=>0, 1=>1, 2=>2, 3=>1, 4=>0]
""",
        category="example",
        tags=["example", "wave", "scenario", "simultaneous"],
        priority=3
    ),

    DoctrineChunk(
        id="EXAMPLE-003",
        title="예시: 양동형 공격 시나리오",
        content="""
【예시 시나리오 3: 양동형 공격】

[상황]
- 포메이션: DIVERSIONARY
- 적 군집: 3개 (다방향)
  - 군집0: 북쪽, 적 4척, ETA 60초, 직선 경로 (실제 위협)
  - 군집1: 동쪽, 적 2척, ETA 90초, 우회 경로 (양동 의심)
  - 군집2: 서쪽, 적 3척, ETA 75초, 직선 경로 (실제 위협)
- 아군 쌍: 5개
  - 쌍0: 북쪽
  - 쌍1: 동쪽
  - 쌍2: 남쪽
  - 쌍3: 서쪽
  - 쌍4: 북동쪽

[분석]
전역 최적화 필요

위협도 평가:
- 군집0: 실제 위협 (직선, 빠른 ETA) → 2쌍
- 군집1: 양동 의심 (우회) → 1쌍
- 군집2: 실제 위협 (직선) → 2쌍

최적 배정 (상대각도 기준):
- 군집0 (북): 쌍0 (북), 쌍4 (북동) 배정
- 군집1 (동): 쌍1 (동) 배정
- 군집2 (서): 쌍3 (서), 쌍2 (남→서 근접) 배정

[출력]
[0=>0, 1=>1, 2=>2, 3=>2, 4=>0]
""",
        category="example",
        tags=["example", "diversionary", "scenario", "decoy"],
        priority=3
    ),

    # =========================================================================
    # CHAPTER 6: SPECIAL SITUATIONS (특수 상황)
    # =========================================================================

    DoctrineChunk(
        id="SPECIAL-001",
        title="특수 상황: 병력 부족",
        content="""
【특수 상황 1: 아군 쌍 수 < 적 군집 수】

[상황]
적 군집이 아군 쌍보다 많을 때

[대응]
1. ETA 기준 군집 정렬 (짧은 순)
2. ETA 상위 군집에 우선 배정
3. 후순위 군집은 1차 요격 완료 후 순차 대응

[예시]
군집 4개, 쌍 3개
- 군집0: ETA 50초 → 쌍0, 쌍1 배정
- 군집1: ETA 70초 → 쌍2 배정
- 군집2: ETA 90초 → 미배정 (순차 대응)
- 군집3: ETA 120초 → 미배정

출력: [0=>0, 1=>0, 2=>1]
(군집2, 3은 출력에 포함 안 함 - 배정된 쌍만 출력)
""",
        category="special",
        tags=["shortage", "priority", "sequential"],
        priority=2
    ),

    DoctrineChunk(
        id="SPECIAL-002",
        title="특수 상황: 병력 과잉",
        content="""
【특수 상황 2: 아군 쌍 수 > 적 군집 수】

[상황]
아군 쌍이 적 군집보다 많을 때

[대응]
1. 가장 위협적인 군집에 추가 쌍 배정
2. 예비 쌍 1-2개는 모선 근접 경계 유지
3. 예비 쌍도 군집에 배정하되, 가장 가까운 군집으로

[예시]
군집 2개, 쌍 5개
- 군집0: 적 5척, ETA 60초 → 쌍 3개
- 군집1: 적 3척, ETA 80초 → 쌍 2개

출력: [0=>0, 1=>0, 2=>0, 3=>1, 4=>1]
""",
        category="special",
        tags=["surplus", "reserve", "additional"],
        priority=2
    ),

    DoctrineChunk(
        id="SPECIAL-003",
        title="특수 상황: 긴급 위협",
        content="""
【특수 상황 3: 긴급 위협 (ETA < 30초)】

[상황]
적이 이미 5km 내 깊숙이 침투하여 ETA가 30초 미만

[대응]
1. 전역 최적화 무시
2. 해당 군집에 가용 쌍 50% 이상 즉시 투입
3. 상대각도 무시하고 가장 가까운 쌍 우선

[원칙]
긴급 상황에서는 회전 시간보다 절대 거리가 중요
→ 1.5m/deg 계수 → 0.5m/deg로 감소

[예시]
군집0: ETA 25초 (긴급!)
쌍 5개 중 3개 즉시 투입 (가장 가까운 순)

출력에서 해당 군집에 집중 배정
""",
        category="special",
        tags=["emergency", "urgent", "critical", "ETA"],
        priority=1
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_doctrine_chunks() -> List[DoctrineChunk]:
    """Get all doctrine chunks."""
    return DOCTRINE_CHUNKS_V2


def get_doctrine_by_category(category: str) -> List[DoctrineChunk]:
    """Get doctrine chunks by category."""
    return [d for d in DOCTRINE_CHUNKS_V2 if d.category == category]


def get_doctrine_by_tags(tags: List[str]) -> List[DoctrineChunk]:
    """Get doctrine chunks that match any of the given tags."""
    result = []
    for d in DOCTRINE_CHUNKS_V2:
        if any(tag in d.tags for tag in tags):
            result.append(d)
    return result


def get_doctrine_for_formation(formation: str) -> List[DoctrineChunk]:
    """Get relevant doctrine for a specific formation type."""
    formation_lower = formation.lower()
    relevant = []

    # Always include core principles
    relevant.extend(get_doctrine_by_category("core"))
    relevant.extend(get_doctrine_by_category("algorithm"))
    relevant.extend(get_doctrine_by_category("output"))

    # Formation-specific
    if "concentrated" in formation_lower or "집중" in formation_lower:
        relevant.extend(get_doctrine_by_category("concentrated"))
        relevant.extend([d for d in DOCTRINE_CHUNKS_V2 if "EXAMPLE-001" in d.id])
    elif "wave" in formation_lower or "파상" in formation_lower:
        relevant.extend(get_doctrine_by_category("wave"))
        relevant.extend([d for d in DOCTRINE_CHUNKS_V2 if "EXAMPLE-002" in d.id])
    elif "diversionary" in formation_lower or "양동" in formation_lower:
        relevant.extend(get_doctrine_by_category("diversionary"))
        relevant.extend([d for d in DOCTRINE_CHUNKS_V2 if "EXAMPLE-003" in d.id])

    # Special situations
    relevant.extend(get_doctrine_by_category("special"))

    # Sort by priority
    relevant.sort(key=lambda x: x.priority)
    return relevant


def get_all_doctrine_texts() -> List[str]:
    """Get all doctrine content as text strings for vectorization."""
    return [f"[{d.id}] {d.title}\n{d.content}" for d in DOCTRINE_CHUNKS_V2]


def get_doctrine_metadata() -> List[Dict]:
    """Get metadata for each doctrine chunk."""
    return [
        {
            "id": d.id,
            "title": d.title,
            "category": d.category,
            "tags": d.tags,
            "priority": d.priority,
        }
        for d in DOCTRINE_CHUNKS_V2
    ]


# =============================================================================
# QUICK REFERENCE (빠른 참조용)
# =============================================================================

QUICK_REFERENCE = """
【빠른 참조 - 매핑 결정 요약】

1. 실질거리 = 거리 + |상대각도| × 1.5

2. 포메이션별 핵심:
   - CONCENTRATED: 해당 방향 집중, 60-70% 배치
   - WAVE: 모든 파도 동시 출격! 순차 대응 금지
   - DIVERSIONARY: 전역 최적화, 총 거리 합 최소화

3. 출력 형식: [쌍ID=>군집ID, 쌍ID=>군집ID, ...]

4. 배분 공식: ceil(적수/3), ETA<60초면 ×1.5
"""
