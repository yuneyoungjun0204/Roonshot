"""
Constants and thresholds for the ML formation classification system.
"""

# =============================================================================
# DEFENSE GEOMETRY
# =============================================================================
DEFENSE_CENTER = (9000.0, 9000.0)
WORLD_SIZE = 18000.0

# =============================================================================
# FORMATION CLASSIFICATION THRESHOLDS
# =============================================================================
# Angular spread thresholds (in degrees)
SPEARHEAD_ANGLE_THRESHOLD = 30.0    # < 30° → narrow approach (CONCENTRATED or WAVE)

# Distance std threshold (meters) — separates CONCENTRATED from WAVE
# Both have narrow angle, but WAVE has layered distances (sequential waves)
WAVE_DISTANCE_STD_THRESHOLD = 400.0  # distance_std >= 400m within narrow angle → WAVE

# Legacy constants (kept for reference)
ENCIRCLEMENT_ANGLE_THRESHOLD = 60.0
ENCIRCLEMENT_ETA_STD_THRESHOLD = 15.0

# Minimum active enemies for ML classification
MIN_ENEMIES_FOR_ML = 3  # Below this, fall back to rule-based

# Formation lock: classify once when first enemy enters this radius
FORMATION_LOCK_RADIUS = 5000.0  # meters from defense center

# =============================================================================
# FEATURE EXTRACTION
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
# DBSCAN DENSITY ANALYSIS
# =============================================================================
DBSCAN_EPS = 150.0          # meters - neighborhood radius
DBSCAN_MIN_SAMPLES = 1      # minimum cluster size

# Net spacing parameters
NET_SPACING_BASE = 80.0      # meters - base net spacing
NET_SPACING_MIN = 40.0       # meters - minimum net spacing (dense cluster)
NET_SPACING_MAX = 100.0      # meters - maximum net spacing (sparse cluster)

# Pair allocation
MAX_PAIRS_PER_CLUSTER = 3
MIN_PAIRS_PER_CLUSTER = 1
TOTAL_FRIENDLY_PAIRS = 5

# Density thresholds for pair allocation
DENSITY_HIGH_THRESHOLD = 1000.0    # mean NN < 400m → high density → 3 pairs
DENSITY_LOW_THRESHOLD = 2500.0    # mean NN > 1200m → low density → 1 pair

# =============================================================================
# CLASSIFIER MODEL
# =============================================================================
CLASSIFIER_INPUT_DIM = 19
CLASSIFIER_HIDDEN1 = 64
CLASSIFIER_HIDDEN2 = 32
CLASSIFIER_OUTPUT_DIM = 3  # CONCENTRATED (집중), WAVE (파상), DIVERSIONARY (양동)

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15

# Confidence threshold for rule-based fallback
CONFIDENCE_THRESHOLD = 0.6

# =============================================================================
# DATA GENERATION
# =============================================================================
ENEMY_SPEED = 50.0           # m/s
FRIENDLY_SPEED = 25.0        # m/s
SPAWN_RADIUS_BASE = 7500.0   # meters from center (CONCENTRATED / DIVERSIONARY) — matches simulator safe_zone_radii[-1]+500
WAVE_SPAWN_RADIUS = 7000.0   # meters from center (WAVE 1st wave) — matches simulator hardcoded value
DT = 0.05                    # simulation timestep
