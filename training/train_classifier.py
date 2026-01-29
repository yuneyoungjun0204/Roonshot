"""
PyTorch training script for the FormationClassifier.

Uses Adam optimizer + CrossEntropyLoss with early stopping.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.constants import (
    LEARNING_RATE,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    NUM_FEATURES,
    AUGMENT_NOISE_STD,
)
from ml.labels import FormationClass
from ml.classifier import FormationClassifier, save_classifier


def augment_features(X: np.ndarray, y: np.ndarray, noise_std: float = AUGMENT_NOISE_STD) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to training features.

    Augmentation techniques:
    1. Gaussian noise injection
    2. Feature-wise scaling perturbation

    Args:
        X: (N, 19) feature array
        y: (N,) label array
        noise_std: Standard deviation for Gaussian noise (as fraction of feature std)

    Returns:
        Augmented (X_aug, y_aug) with ~2x original size
    """
    n_samples, n_features = X.shape

    # Original data
    X_list = [X]
    y_list = [y]

    # 1. Gaussian noise augmentation
    feature_stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero
    noise = np.random.randn(n_samples, n_features) * feature_stds * noise_std
    X_noisy = X + noise
    X_list.append(X_noisy)
    y_list.append(y.copy())

    # 2. Small scaling perturbation (simulate measurement uncertainty)
    scale_factors = np.random.uniform(0.95, 1.05, size=(n_samples, n_features))
    X_scaled = X * scale_factors
    X_list.append(X_scaled)
    y_list.append(y.copy())

    # Combine
    X_aug = np.vstack(X_list).astype(np.float32)
    y_aug = np.concatenate(y_list).astype(np.int64)

    return X_aug, y_aug


def load_dataset(path: str = "data/training_data/formation_tensors.pt"):
    """Load dataset from PyTorch tensor file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["X"].numpy(), data["y"].numpy()


def load_csv_dataset(path: str = "data/training_data/formation_dataset.csv"):
    """Fallback: load from CSV."""
    import csv
    X_list, y_list = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            X_list.append([float(v) for v in row[:-1]])
            y_list.append(int(row[-1]))
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def train_classifier(
    data_path: str = "data/training_data",
    model_save_path: str = "models/formation_classifier.pt",
    scaler_save_path: str = "models/feature_scaler.npz",
):
    """Train the PyTorch FormationClassifier."""
    print("=" * 60)
    print("PyTorch FormationClassifier Training")
    print("=" * 60)

    # Load data
    tensor_path = os.path.join(data_path, "formation_tensors.pt")
    csv_path = os.path.join(data_path, "formation_dataset.csv")

    if os.path.exists(tensor_path):
        print(f"Loading from {tensor_path}...")
        X, y = load_dataset(tensor_path)
    elif os.path.exists(csv_path):
        print(f"Loading from {csv_path}...")
        X, y = load_csv_dataset(csv_path)
    else:
        print("No data found! Run data/generator.py first.")
        return None

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Data augmentation (only on training set)
    print(f"Before augmentation: {X_train.shape[0]} training samples")
    X_train, y_train = augment_features(X_train, y_train)
    print(f"After augmentation: {X_train.shape[0]} training samples (3x)")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # Save scaler parameters for inference
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    np.savez(scaler_save_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"Scaler saved to {scaler_save_path}")

    # Create DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FormationClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )

    # Training loop with early stopping
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(X_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

        # Learning rate scheduler step
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            save_classifier(model, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Model saved to {model_save_path}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FormationClassifier")
    parser.add_argument("--data", type=str, default="data/training_data")
    parser.add_argument("--model", type=str, default="models/formation_classifier.pt")
    args = parser.parse_args()

    train_classifier(data_path=args.data, model_save_path=args.model)
