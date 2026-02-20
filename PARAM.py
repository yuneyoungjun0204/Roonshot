"""
Centralized Parameter Configuration for USV Defense Simulator
=============================================================
모든 시뮬레이션/ML/전술 파라미터를 한 곳에서 관리합니다.

Sections:
  1. SIMULATION CORE        - 시뮬레이션 기본 설정
  2. DISPLAY & RENDERING    - 화면 및 렌더링
  3. USV PARAMETERS          - 선박 속도/포획 관련
  4. DEFENSE GEOMETRY        - 방어 구역 좌표/반경
  5. MOTHERSHIP              - 모선 관련
  6. FORMATION CLASSIFICATION - 포메이션 분류 임계값
  7. FEATURE EXTRACTION      - 특징 추출 설정
  8. DBSCAN / DENSITY        - 클러스터링 및 밀도 분석
  9. CLASSIFIER MODEL        - 분류 모델 구조 및 학습
  10. TACTICAL ASSIGNMENT    - 전술 배정 파라미터
  11. PAIR BEHAVIOR          - 아군 쌍 행동 파라미터
  12. VRP SOLVER             - VRP 솔버 설정
  13. DATA GENERATION        - 데이터 생성 (학습용)
  14. DATA AUGMENTATION      - 데이터 증강 설정
"""

# =============================================================================
# 1. SIMULATION CORE (시뮬레이션 기본 설정)
# =============================================================================
SIM_DT = 0.05              # 시뮬레이션 타임스텝 (seconds)
SIM_MAX_TIME = 300          # 최대 시뮬레이션 시간 (seconds)

# =============================================================================
# 2. DISPLAY & RENDERING (화면 및 렌더링)
# =============================================================================
SCREEN_WIDTH = 1600         # 화면 너비 (pixels)
SCREEN_HEIGHT = 1200        # 화면 높이 (pixels)
WORLD_SIZE = 18000.0        # 시뮬레이션 월드 크기 (18km x 18km)
VIEW_RADIUS = 9000.0        # 뷰포트 반경 (9km)

# Colors (Dark Theme)
COLOR_BG = (15, 15, 25)
COLOR_GRID = (30, 30, 50)
COLOR_FRIENDLY = (0, 200, 255)
COLOR_ENEMY = (255, 80, 80)
COLOR_NET = (100, 255, 150)
COLOR_CAPTURED = (255, 200, 0)
COLOR_SAFE_ZONE_1KM = (50, 50, 80, 80)
COLOR_SAFE_ZONE_2KM = (40, 40, 70, 60)
COLOR_TEXT = (220, 220, 240)
COLOR_UI_BG = (25, 25, 40, 200)
COLOR_MOTHERSHIP = (150, 150, 200)

# Cluster visualization colors
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

# Cluster boundary padding (world units)
CLUSTER_BOUNDARY_PADDING = 120.0

# =============================================================================
# 3. USV PARAMETERS (선박 속도/포획 관련)
# =============================================================================
FRIENDLY_SPEED = 75.0       # 아군 속도 (m/s)
ENEMY_SPEED = 150.0         # 적군 속도 (m/s) - 고속 위협
CAPTURE_DISTANCE = 15.0     # 그물 포획 거리 (meters)
NET_MAX_LENGTH = 200.0      # 최대 그물 길이 (meters)

# =============================================================================
# 4. DEFENSE GEOMETRY (방어 구역 좌표/반경)
# =============================================================================
DEFENSE_CENTER = (9000.0, 9000.0)       # 방어 중심 좌표 (meters)
SAFE_ZONE_RADII = [1000, 2000, 3000, 4000, 5000, 6000, 7000]  # 안전 구역 반경 (1km~7km)

# =============================================================================
# 5. MOTHERSHIP (모선 관련)
# =============================================================================
MOTHERSHIP_SIZE = 200.0                 # 모선 크기 (meters, 정사각형 변 길이)
MOTHERSHIP_POSITION = (9000.0, 9000.0)  # 모선 위치 (= 방어 중심)
MOTHERSHIP_COLLISION_DISTANCE = 100.0   # 모선 충돌 감지 반경 (meters)

# =============================================================================
# 6. FORMATION CLASSIFICATION (포메이션 분류 임계값)
# =============================================================================
# Angular spread thresholds (degrees)
SPEARHEAD_ANGLE_THRESHOLD = 30.0        # < 30° → 좁은 접근 (CONCENTRATED or WAVE)

# Distance std threshold (meters) — CONCENTRATED vs WAVE 구분
WAVE_DISTANCE_STD_THRESHOLD = 400.0     # distance_std >= 400m → WAVE

# Legacy constants
ENCIRCLEMENT_ANGLE_THRESHOLD = 60.0
ENCIRCLEMENT_ETA_STD_THRESHOLD = 15.0

# Minimum enemies for ML classification
MIN_ENEMIES_FOR_ML = 3                  # 이 미만이면 rule-based fallback

# Formation lock radius
FORMATION_LOCK_RADIUS = 5000.0          # meters - 적이 이 반경 진입 시 포메이션 확정

# Confidence threshold for rule-based fallback
CONFIDENCE_THRESHOLD = 0.6

# =============================================================================
# 7. FEATURE EXTRACTION (특징 추출 설정)
# =============================================================================
NUM_FEATURES = 19
FEATURE_NAMES = [
    # Angular distribution (f1-f4)
    "mean_bearing",
    "angular_std",
    "angular_range",
    "num_quadrants",
    # Distance distribution (f5-f8)
    "mean_distance",
    "distance_std",
    "min_distance",
    "max_distance",
    # Speed consistency (f9-f12)
    "mean_approach_speed",
    "speed_std",
    "heading_alignment",
    "alignment_std",
    # Time consistency (f13-f15)
    "mean_eta",
    "eta_std",
    "min_eta",
    # Density (f16-f18)
    "mean_pairwise_distance",
    "pairwise_distance_std",
    "mean_nn_distance",
    # Other (f19)
    "num_active_enemies",
]

# =============================================================================
# 8. DBSCAN / DENSITY (클러스터링 및 밀도 분석)
# =============================================================================
DBSCAN_EPS = 150.0              # meters - DBSCAN 이웃 반경
DBSCAN_MIN_SAMPLES = 1          # 최소 클러스터 크기

# Net spacing parameters (meters)
NET_SPACING_BASE = 80.0         # 기본 그물 간격
NET_SPACING_MIN = 40.0          # 최소 그물 간격 (밀집 클러스터)
NET_SPACING_MAX = 100.0         # 최대 그물 간격 (희소 클러스터)

# Pair allocation
MAX_PAIRS_PER_CLUSTER = 3
MIN_PAIRS_PER_CLUSTER = 1
TOTAL_FRIENDLY_PAIRS = 5

# Density thresholds for pair allocation
DENSITY_HIGH_THRESHOLD = 1000.0     # mean NN < 이 값 → high density → 3 pairs
DENSITY_LOW_THRESHOLD = 2500.0      # mean NN > 이 값 → low density → 1 pair

# =============================================================================
# 9. CLASSIFIER MODEL (분류 모델 구조 및 학습)
# =============================================================================
CLASSIFIER_INPUT_DIM = 19
CLASSIFIER_HIDDEN1 = 64
CLASSIFIER_HIDDEN2 = 32
CLASSIFIER_OUTPUT_DIM = 3       # CONCENTRATED, WAVE, DIVERSIONARY

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 40

# Dropout rates
CLASSIFIER_DROPOUT1 = 0.2
CLASSIFIER_DROPOUT2 = 0.1

# =============================================================================
# 10. TACTICAL ASSIGNMENT (전술 배정 파라미터)
# =============================================================================
# Effective distance formula: distance + |relative_angle| × ANGLE_WEIGHT
ANGLE_WEIGHT = 1.5              # 각도 가중치 (10° = 15m 등가)

# ETA-based weighting
ETA_WEIGHT_DEFAULT = 0.2        # 기본 ETA 가중치
ETA_WEIGHT_WAVE = 0.4           # WAVE 포메이션 ETA 가중치
ETA_MAX = 200.0                 # 최대 ETA 정규화 기준 (seconds)
ETA_PENALTY_SCALE = 500.0       # ETA 패널티 스케일링

# Cost matrix padding value
COST_MATRIX_BIG = 1e9           # 할당 불가 조합의 비용

# =============================================================================
# 11. PAIR BEHAVIOR (아군 쌍 행동 파라미터)
# =============================================================================
# Missed enemy detection
MISSED_THRESHOLD = 100.0        # meters - 적이 쌍보다 이만큼 더 모선에 가까우면 "놓침"

# Escaped enemy detection
ESCAPE_DISTANCE_THRESHOLD = 800.0   # meters - 클러스터 중심에서 이 거리 초과 시 "탈출"

# Pair return to home
RETURNING_HOME_CLUSTER_ID = -99999  # 귀환 상태 마커
HOME_ARRIVAL_THRESHOLD = 50.0       # meters - 홈 도착 판정 거리

# Post-capture reassignment
POST_CAPTURE_DISTANCE_THRESHOLD = 150.0     # meters - 포획 후 근접 적 탐색 반경
POST_CAPTURE_ANGLE_THRESHOLD = 30.0         # degrees - 포획 후 근접 적 각도 범위

# Spawn layout
PAIR_BASE_PATROL_RADIUS = 100.0     # meters - 모선으로부터 초기 배치 거리
PAIR_RADIUS_STEP = 150.0            # meters - 같은 방향 쌍 간 거리 간격

# Dynamic mode
DYNAMIC_UPDATE_INTERVAL = 5         # seconds - 동적 재배정 간격

# =============================================================================
# 12. VRP SOLVER (VRP 솔버 설정)
# =============================================================================
VRP_TIME_LIMIT_SECONDS = 5          # VRP 솔버 시간 제한
VRP_BIG_DISTANCE = 10**7            # VRP 거리 행렬 무한대 대체값

# =============================================================================
# 13. DATA GENERATION (데이터 생성 - 학습용)
# =============================================================================
# NOTE: These speeds differ from simulation speeds (FRIENDLY_SPEED, ENEMY_SPEED)
# because training data uses different speed profiles
DATA_GEN_ENEMY_SPEED = 100.0        # m/s (학습 데이터용 적 속도)
DATA_GEN_FRIENDLY_SPEED = 50.0      # m/s (학습 데이터용 아군 속도)
DATA_GEN_SPAWN_RADIUS_BASE = 7500.0 # meters - CONCENTRATED/DIVERSIONARY 스폰 반경
DATA_GEN_WAVE_SPAWN_RADIUS = 7000.0 # meters - WAVE 1st wave 스폰 반경
DATA_GEN_DT = 0.05                  # 학습 데이터 시뮬레이션 타임스텝

# Wave pattern parameters
WAVE_SPAWN_RADIUS = 7000            # meters - Wave 공격 스폰 반경 (시뮬레이터)
WAVE_NUM_WAVES = 3                  # 기본 wave 수
WAVE_DELAY = 5.0                    # seconds - wave 간 시간 간격
WAVE_GAP = 800.0                    # meters - wave 간 거리 간격

# Attack pattern spawn
CONCENTRATED_ANGLE_SPREAD = 30.0    # degrees - 집중형 각도 분산 (pi/6)
CONCENTRATED_SPAWN_JITTER = 500.0   # meters - 집중형 스폰 거리 변동
DIVERSIONARY_NUM_DIRECTIONS = 3     # 양동형 접근 방향 수

# =============================================================================
# 14. DATA AUGMENTATION (데이터 증강 설정)
# =============================================================================
AUGMENT_NOISE_STD = 0.05            # Gaussian noise std (feature std 비율)
AUGMENT_TEMPORAL_STEPS = 3          # lock point 주변 temporal snapshot 수
AUGMENT_TEMPORAL_RANGE = 20         # ±20 시뮬레이션 스텝
AUGMENT_ROTATION_ENABLED = True     # 랜덤 회전 증강 활성화
AUGMENT_SCALE_RANGE = (0.9, 1.1)   # 위치 스케일링 범위

# =============================================================================
# LLM COMMANDER (LLM 전술 지휘관)
# =============================================================================
LLM_DEFAULT_MODEL = "qwen2.5:7b-instruct"
LLM_OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.1

# =============================================================================
# CONFIG DICT (usv_simulator.py 호환용)
# =============================================================================
# usv_simulator.py의 CONFIG dict를 PARAM 값으로 구성
CONFIG = {
    # Simulation
    "dt": SIM_DT,
    "max_time": SIM_MAX_TIME,

    # Display
    "screen_width": SCREEN_WIDTH,
    "screen_height": SCREEN_HEIGHT,
    "world_size": WORLD_SIZE,
    "view_radius": VIEW_RADIUS,

    # Colors
    "bg_color": COLOR_BG,
    "grid_color": COLOR_GRID,
    "friendly_color": COLOR_FRIENDLY,
    "enemy_color": COLOR_ENEMY,
    "net_color": COLOR_NET,
    "captured_color": COLOR_CAPTURED,
    "safe_zone_1km": COLOR_SAFE_ZONE_1KM,
    "safe_zone_2km": COLOR_SAFE_ZONE_2KM,
    "text_color": COLOR_TEXT,
    "ui_bg_color": COLOR_UI_BG,

    # USV Parameters
    "friendly_speed": FRIENDLY_SPEED,
    "enemy_speed": ENEMY_SPEED,
    "capture_distance": CAPTURE_DISTANCE,
    "net_max_length": NET_MAX_LENGTH,

    # Defense Zone
    "defense_center": DEFENSE_CENTER,
    "safe_zone_radii": SAFE_ZONE_RADII,

    # Mothership
    "mothership_size": MOTHERSHIP_SIZE,
    "mothership_color": COLOR_MOTHERSHIP,
    "mothership_position": MOTHERSHIP_POSITION,
    "mothership_collision_distance": MOTHERSHIP_COLLISION_DISTANCE,
}
