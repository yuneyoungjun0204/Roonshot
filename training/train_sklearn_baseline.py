"""
Scikit-learn baseline for formation classification.
Uses RandomForest to validate features before training a neural net.

Target: >85% accuracy on validation set.
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from ml.constants import FEATURE_NAMES
from ml.labels import FormationClass


def load_csv_dataset(path: str = "data/training_data/formation_dataset.csv"):
    """Load dataset from CSV."""
    import csv
    X_list, y_list = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            X_list.append([float(v) for v in row[:-1]])
            y_list.append(int(row[-1]))
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def train_baseline(data_path: str = "data/training_data/formation_dataset.csv"):
    """Train and evaluate RandomForest baseline."""
    print("=" * 60)
    print("Scikit-learn Baseline: RandomForest")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    X, y = load_csv_dataset(data_path)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    for c in range(3):
        print(f"  Class {c} ({FormationClass(c).name}): {np.sum(y == c)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train RandomForest
    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    # Evaluate
    train_acc = rf.score(X_train_s, y_train)
    test_acc = rf.score(X_test_s, y_test)
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # Cross-validation
    print("\n5-fold cross-validation...")
    cv_scores = cross_val_score(rf, scaler.transform(X), y, cv=5, n_jobs=-1)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Detailed report
    y_pred = rf.predict(X_test_s)
    class_names = [FormationClass(c).name for c in range(3)]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(FEATURE_NAMES))):
        idx = indices[i]
        print(f"  {i+1}. {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")

    return rf, scaler, test_acc


if __name__ == "__main__":
    train_baseline()
