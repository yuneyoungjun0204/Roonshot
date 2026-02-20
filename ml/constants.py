"""
Constants and thresholds for the ML formation classification system.

All values are imported from the centralized PARAM.py file.
This module re-exports them for backward compatibility with existing ml/* imports.
"""

import sys
import os

# Ensure project root is on the path so we can import PARAM
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PARAM import (
    # Defense Geometry
    DEFENSE_CENTER,
    WORLD_SIZE,

    # Formation Classification Thresholds
    SPEARHEAD_ANGLE_THRESHOLD,
    WAVE_DISTANCE_STD_THRESHOLD,
    ENCIRCLEMENT_ANGLE_THRESHOLD,
    ENCIRCLEMENT_ETA_STD_THRESHOLD,
    MIN_ENEMIES_FOR_ML,
    FORMATION_LOCK_RADIUS,

    # Feature Extraction
    NUM_FEATURES,
    FEATURE_NAMES,

    # DBSCAN Density Analysis
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    NET_SPACING_BASE,
    NET_SPACING_MIN,
    NET_SPACING_MAX,
    MAX_PAIRS_PER_CLUSTER,
    MIN_PAIRS_PER_CLUSTER,
    TOTAL_FRIENDLY_PAIRS,
    DENSITY_HIGH_THRESHOLD,
    DENSITY_LOW_THRESHOLD,

    # Classifier Model
    CLASSIFIER_INPUT_DIM,
    CLASSIFIER_HIDDEN1,
    CLASSIFIER_HIDDEN2,
    CLASSIFIER_OUTPUT_DIM,
    LEARNING_RATE,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    CONFIDENCE_THRESHOLD,

    # Missed Enemy Detection
    MISSED_THRESHOLD,

    # Escaped Enemy Detection
    ESCAPE_DISTANCE_THRESHOLD,

    # Pair Return to Home
    RETURNING_HOME_CLUSTER_ID,
    HOME_ARRIVAL_THRESHOLD,

    # Post-Capture Reassignment
    POST_CAPTURE_DISTANCE_THRESHOLD,
    POST_CAPTURE_ANGLE_THRESHOLD,

    # Tactical Assignment
    ANGLE_WEIGHT,
    ETA_WEIGHT_DEFAULT,
    ETA_WEIGHT_WAVE,
    ETA_MAX,
    ETA_PENALTY_SCALE,
    COST_MATRIX_BIG,

    # VRP Solver
    VRP_TIME_LIMIT_SECONDS,
    VRP_BIG_DISTANCE,

    # Classifier Dropout
    CLASSIFIER_DROPOUT1,
    CLASSIFIER_DROPOUT2,

    # LLM Commander
    LLM_DEFAULT_MODEL,
    LLM_OLLAMA_BASE_URL,
    LLM_TEMPERATURE,

    # Data Augmentation
    AUGMENT_NOISE_STD,
    AUGMENT_TEMPORAL_STEPS,
    AUGMENT_TEMPORAL_RANGE,
    AUGMENT_ROTATION_ENABLED,
    AUGMENT_SCALE_RANGE,
)

# Data Generation - re-export with original names for backward compatibility
from PARAM import (
    DATA_GEN_ENEMY_SPEED as ENEMY_SPEED,
    DATA_GEN_FRIENDLY_SPEED as FRIENDLY_SPEED,
    DATA_GEN_SPAWN_RADIUS_BASE as SPAWN_RADIUS_BASE,
    DATA_GEN_WAVE_SPAWN_RADIUS as WAVE_SPAWN_RADIUS,
    DATA_GEN_DT as DT,
)
