"""
MNIST-Training mit unserem selbstgebauten Neuronalen Netz
=========================================================
Demonstriert: Daten laden, Training-Loop, Evaluation.
"""

import numpy as np
from nn_core import NeuralNetwork, Dense, ReLU, SGD


def load_mnist():
    """Lädt MNIST aus dem lokalen Cache oder lädt es herunter."""
    import gzip
    import os
    from urllib import request

    cache_dir = os.path.join(os.path.dirname(__file__), ".mnist_cache")
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    for name, fname in files.items():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            print(f"  Lade {fname} herunter...")
            request.urlretrieve(base_url + fname, path)

    def load_images(path):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def load_labels(path):
        with gzip.open(path, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    X_train = load_images(os.path.join(cache_dir, files["train_images"]))
    y_train = load_labels(os.path.join(cache_dir, files["train_labels"]))
    X_test = load_images(os.path.join(cache_dir, files["test_images"]))
    y_test = load_labels(os.path.join(cache_dir, files["test_labels"]))

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 60)
    print("  Neuronales Netz von Grund auf — MNIST-Training")
    print("=" * 60)

    # ── Daten laden ──────────────────────────────────────────
    print("\n📦 Lade MNIST-Daten...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"   Train: {X_train.shape[0]:,} Bilder, Test: {X_test.shape[0]:,} Bilder")

    # ── Netzwerk bauen ──────────────────────────────────────
    print("\n🧠 Baue Netzwerk: 784 → 128 → 64 → 10")
    net = NeuralNetwork([
        Dense(784, 128), ReLU(),
        Dense(128, 64),  ReLU(),
        Dense(64, 10),
    ])

    # Parameter zählen
    total_params = sum(
        l.W.size + l.b.size
        for l in net.layers
        if isinstance(l, Dense)
    )
    print(f"   Parameter: {total_params:,}")

    # ── Training ─────────────────────────────────────────────
    print("\n🏋️ Training (10 Epochen)...")
    optimizer = SGD(lr=0.1, momentum=0.9)
    batch_size = 64
    epochs = 10

    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        total_loss = 0
        total_acc = 0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            loss, acc = net.train_step(x_batch, y_batch, optimizer)
            total_loss += loss
            total_acc += acc
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        # Test-Accuracy
        test_preds = net.predict(X_test)
        test_acc = np.mean(test_preds == y_test)

        print(f"   Epoche {epoch+1:2d}: "
              f"Loss={avg_loss:.4f}  "
              f"Train-Acc={avg_acc:.3f}  "
              f"Test-Acc={test_acc:.3f}")

    # ── Finale Evaluation ────────────────────────────────────
    print("\n📊 Finale Evaluation:")
    test_preds = net.predict(X_test)
    test_acc = np.mean(test_preds == y_test)
    print(f"   Test-Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Confusion-Matrix (vereinfacht)
    from collections import Counter
    errors = [(true, pred) for true, pred in zip(y_test, test_preds) if true != pred]
    print(f"   Fehlklassifikationen: {len(errors)}/{len(y_test)}")

    print("\n✅ Training abgeschlossen!")


if __name__ == "__main__":
    main()
