"""
Neuronales Netz von Grund auf — nur mit NumPy
=============================================
Implementiert: Dense Layer, ReLU/Sigmoid/Softmax, Cross-Entropy Loss,
Forward Pass, Backpropagation, SGD-Optimizer.

Kein TensorFlow, kein PyTorch — nur Mathematik.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Aktivierungsfunktionen
# ═══════════════════════════════════════════════════════════════

class ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)"""
    def __init__(self):
        self.cache = None  # Speichert Input für Backward-Pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.cache > 0)


class Sigmoid:
    """Sigmoid: f(x) = 1 / (1 + e^(-x))"""
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.cache = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.cache * (1 - self.cache)


class Softmax:
    """Softmax mit numerischer Stabilität (subtract max)"""
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        out = exp / np.sum(exp, axis=1, keepdims=True)
        self.cache = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Softmax + Cross-Entropy wird kombiniert (siehe Loss.backward)
        return dout


# ═══════════════════════════════════════════════════════════════
# Loss-Funktionen
# ═══════════════════════════════════════════════════════════════

class CrossEntropyLoss:
    """Cross-Entropy Loss für Klassifikation.
    Kombiniert Softmax + Cross-Entropy für numerische Stabilität."""

    def __init__(self):
        self.cache = None
        self.y_true = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Args:
            logits: (N, C) — Rohwerte vor Softmax
            y_true: (N,)   — Integer-Labels 0..C-1
        """
        self.y_true = y_true
        N = logits.shape[0]

        # Numerisch stabil: log(softmax) = logits - log(sum(exp(logits)))
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
        correct_logits = shifted[np.arange(N), y_true]

        loss = np.mean(log_sum_exp - correct_logits)
        self.cache = shifted
        return loss

    def backward(self) -> np.ndarray:
        """Gradient der kombinierten Softmax + Cross-Entropy"""
        N = self.y_true.shape[0]
        shifted = self.cache
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        # Gradient: (probs - one_hot) / N
        grad = probs.copy()
        grad[np.arange(N), self.y_true] -= 1
        return grad / N


# ═══════════════════════════════════════════════════════════════
# Layer
# ═══════════════════════════════════════════════════════════════

class Dense:
    """Vollständig verbundener Layer: y = x @ W + b"""

    def __init__(self, input_dim: int, output_dim: int):
        # He-Initialisierung (gut für ReLU)
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.cache = None  # (x, W) für Backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = (x, self.W)
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, W = self.cache
        # Gradienten für Input, W, b
        self.dW = x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ W.T  # Gradient für vorherigen Layer


# ═══════════════════════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════════════════════

class SGD:
    """Stochastic Gradient Descent mit Momentum"""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}  # id(layer) -> {W: v_W, b: v_b}

    def step(self, layers: list[Dense]):
        for layer in layers:
            lid = id(layer)
            if lid not in self.velocities:
                self.velocities[lid] = {"W": 0, "b": 0}

            # Momentum-Update
            self.velocities[lid]["W"] = (
                self.momentum * self.velocities[lid]["W"] - self.lr * layer.dW
            )
            self.velocities[lid]["b"] = (
                self.momentum * self.velocities[lid]["b"] - self.lr * layer.db
            )

            layer.W += self.velocities[lid]["W"]
            layer.b += self.velocities[lid]["b"]


# ═══════════════════════════════════════════════════════════════
# Das komplette Netzwerk
# ═══════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    Einfaches Feedforward-Netzwerk mit beliebig vielen Layern.

    Beispiel:
        net = NeuralNetwork([
            Dense(784, 128), ReLU(),
            Dense(128, 64),  ReLU(),
            Dense(64, 10),
        ])
    """

    def __init__(self, architecture: list):
        self.layers = architecture
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward-Pass durch alle Layer. Gibt Logits zurück."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray):
        """Backward-Pass (rückwärts durch alle Layer)"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_step(self, x: np.ndarray, y: np.ndarray,
                   optimizer: SGD) -> tuple[float, float]:
        """
        Ein Trainingsschritt: Forward → Loss → Backward → Update.

        Returns:
            (loss, accuracy)
        """
        # Forward
        logits = self.forward(x)

        # Loss
        loss = self.loss_fn.forward(logits, y)

        # Accuracy
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)

        # Backward
        grad = self.loss_fn.backward()
        self.backward(grad)

        # Update
        dense_layers = [layer for layer in self.layers if isinstance(layer, Dense)]
        optimizer.step(dense_layers)

        return loss, acc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Vorhersage: gibt Klassen-Labels zurück"""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Vorhersage: gibt Wahrscheinlichkeiten zurück"""
        logits = self.forward(x)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)
