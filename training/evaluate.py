"""
Evaluation and visualization script for the formation classifier.
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import classification_report, confusion_matrix

from ml.labels import FormationClass
from ml.classifier import FormationClassifier, load_classifier


def evaluate_model(
    model_path: str = "models/formation_classifier.pt",
    data_path: str = "data/training_data/formation_dataset.csv",
    scaler_path: str = "models/feature_scaler.npz",
):
    """Evaluate trained model on full dataset."""
    print("=" * 60)
    print("Formation Classifier Evaluation")
    print("=" * 60)

    # Load model
    model = load_classifier(model_path)
    if model is None:
        print(f"Model not found at {model_path}")
        return

    # Load data
    import csv
    X_list, y_list = [], []
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X_list.append([float(v) for v in row[:-1]])
            y_list.append(int(row[-1]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Apply scaler if available
    if os.path.exists(scaler_path):
        scaler_data = np.load(scaler_path)
        X = (X - scaler_data["mean"]) / scaler_data["scale"]

    # Predict
    probs = model.predict_proba(X)
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Report
    class_names = [FormationClass(c).name for c in range(3)]
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    # Confidence analysis
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {confidences.mean():.4f}")
    print(f"  Min:  {confidences.min():.4f}")
    print(f"  Max:  {confidences.max():.4f}")

    correct_mask = y_pred == y
    if np.any(correct_mask):
        print(f"  Correct predictions mean confidence: {confidences[correct_mask].mean():.4f}")
    if np.any(~correct_mask):
        print(f"  Wrong predictions mean confidence:   {confidences[~correct_mask].mean():.4f}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for c in range(3):
        mask = y == c
        if np.any(mask):
            acc = np.mean(y_pred[mask] == c)
            print(f"  {FormationClass(c).name}: {acc:.4f} ({np.sum(mask)} samples)")

    # Try to plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix heatmap
        ax = axes[0]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)

        # Confidence distribution
        ax = axes[1]
        for c in range(3):
            mask = y == c
            ax.hist(confidences[mask], bins=30, alpha=0.5, label=class_names[c])
        ax.set_title("Confidence Distribution by Class")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend()

        plt.tight_layout()
        plot_path = "models/evaluation_plots.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlots saved to {plot_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available, skipping plots")


if __name__ == "__main__":
    evaluate_model()
