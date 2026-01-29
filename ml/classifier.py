"""
PyTorch formation classifier model.

Architecture: MLP 19 → 64 → 32 → 3

IMPORTANT: The model is trained on StandardScaler-normalized features.
The scaler parameters (mean, scale) must be loaded and applied at inference time.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path

from ml.constants import (
    CLASSIFIER_INPUT_DIM,
    CLASSIFIER_HIDDEN1,
    CLASSIFIER_HIDDEN2,
    CLASSIFIER_OUTPUT_DIM,
    CONFIDENCE_THRESHOLD,
)
from ml.labels import FormationClass


class FormationClassifier(nn.Module):
    """MLP classifier for enemy formation type."""

    def __init__(
        self,
        input_dim: int = CLASSIFIER_INPUT_DIM,
        hidden1: int = CLASSIFIER_HIDDEN1,
        hidden2: int = CLASSIFIER_HIDDEN2,
        output_dim: int = CLASSIFIER_OUTPUT_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(0.1),
            nn.Linear(hidden2, output_dim),
        )
        # StandardScaler parameters (set after training or loading)
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_scale: Optional[np.ndarray] = None

    def set_scaler(self, mean: np.ndarray, scale: np.ndarray):
        """Set StandardScaler parameters for feature normalization at inference."""
        self._scaler_mean = mean.astype(np.float32)
        self._scaler_scale = scale.astype(np.float32)

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Apply StandardScaler normalization if scaler is available."""
        if self._scaler_mean is not None and self._scaler_scale is not None:
            return (features - self._scaler_mean) / self._scaler_scale
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities from a numpy feature vector.
        Applies StandardScaler normalization before inference.

        Args:
            features: (19,) or (B, 19) numpy array (raw features)

        Returns:
            (3,) or (B, 3) probability array
        """
        self.eval()
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features = self._scale_features(features.astype(np.float32))

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()

    def predict(
        self, features: np.ndarray
    ) -> Tuple[FormationClass, float]:
        """
        Predict formation class with confidence.
        Applies StandardScaler normalization before inference.

        Args:
            features: (19,) numpy feature vector (raw features)

        Returns:
            (FormationClass, confidence) tuple
        """
        probs = self.predict_proba(features)
        if probs.ndim == 2:
            probs = probs[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        return FormationClass(class_idx), confidence


def load_classifier(
    model_path: str = "models/formation_classifier.pt",
    scaler_path: str = "models/feature_scaler.npz",
) -> Optional[FormationClassifier]:
    """
    Load a trained classifier and its feature scaler from disk.

    Returns None if the model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        return None
    model = FormationClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    # Load scaler parameters
    scaler_file = Path(scaler_path)
    if scaler_file.exists():
        scaler_data = np.load(scaler_file)
        model.set_scaler(scaler_data["mean"], scaler_data["scale"])
    else:
        print(f"[WARNING] Feature scaler not found at {scaler_path}. "
              "Model predictions may be inaccurate without normalization.")

    return model


def save_classifier(
    model: FormationClassifier,
    model_path: str = "models/formation_classifier.pt",
):
    """Save a trained classifier to disk."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
