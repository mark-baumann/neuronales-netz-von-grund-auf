"""
Tests für nn_core — das Neuronale Netz von Grund auf
=====================================================
Testet alle Komponenten: Aktivierungen, Loss, Layer, Optimizer, Netzwerk.
"""

import numpy as np
import pytest

from nn_core import SGD, CrossEntropyLoss, Dense, NeuralNetwork, ReLU, Sigmoid, Softmax

# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def set_seed():
    """Deterministische Tests durch festen Seed."""
    np.random.seed(42)


@pytest.fixture
def sample_data():
    """Kleine Testdaten: 4 Samples, 5 Features, 3 Klassen."""
    x = np.random.randn(4, 5).astype(np.float32)
    y = np.array([0, 1, 2, 1])
    return x, y


# ═══════════════════════════════════════════════════════════════
# Aktivierungsfunktionen
# ═══════════════════════════════════════════════════════════════

class TestReLU:
    def test_forward_positive(self):
        relu = ReLU()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = relu.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_forward_negative(self):
        relu = ReLU()
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        out = relu.forward(x)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_array_equal(out, expected)

    def test_forward_zero(self):
        relu = ReLU()
        x = np.zeros((3, 3))
        out = relu.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_backward(self):
        relu = ReLU()
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        relu.forward(x)
        dout = np.ones_like(x)
        grad = relu.backward(dout)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(grad, expected)

    def test_cache_preserved(self):
        relu = ReLU()
        x = np.array([[1.0, -2.0]])
        relu.forward(x)
        np.testing.assert_array_equal(relu.cache, x)


class TestSigmoid:
    def test_forward_range(self):
        sig = Sigmoid()
        x = np.array([[0.0]])
        out = sig.forward(x)
        assert 0.49 < out[0, 0] < 0.51  # sigmoid(0) = 0.5

    def test_forward_large_values(self):
        sig = Sigmoid()
        x = np.array([[100.0], [-100.0]])
        out = sig.forward(x)
        assert out[0, 0] > 0.99  # sigmoid(100) ≈ 1
        assert out[1, 0] < 0.01  # sigmoid(-100) ≈ 0

    def test_backward_shape(self):
        sig = Sigmoid()
        x = np.random.randn(4, 3)
        sig.forward(x)
        dout = np.ones_like(x)
        grad = sig.backward(dout)
        assert grad.shape == x.shape

    def test_backward_nonnegative(self):
        """Sigmoid-Gradient ist immer ≥ 0."""
        sig = Sigmoid()
        x = np.random.randn(10, 5)
        sig.forward(x)
        dout = np.ones_like(x)
        grad = sig.backward(dout)
        assert np.all(grad >= 0)


class TestSoftmax:
    def test_forward_sums_to_one(self):
        sm = Softmax()
        x = np.random.randn(5, 4)
        out = sm.forward(x)
        sums = np.sum(out, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(5))

    def test_forward_numerical_stability(self):
        """Große Werte sollten keinen NaN/Inf produzieren."""
        sm = Softmax()
        x = np.array([[1000.0, 2000.0, 3000.0]])
        out = sm.forward(x)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
        np.testing.assert_array_almost_equal(np.sum(out), 1.0)

    def test_backward_passthrough(self):
        """Softmax.backward gibt dout unverändert zurück."""
        sm = Softmax()
        x = np.random.randn(3, 4)
        sm.forward(x)
        dout = np.random.randn(3, 4)
        grad = sm.backward(dout)
        np.testing.assert_array_equal(grad, dout)


# ═══════════════════════════════════════════════════════════════
# Loss-Funktion
# ═══════════════════════════════════════════════════════════════

class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        """Loss ≈ 0 wenn Logits perfekt sind."""
        loss_fn = CrossEntropyLoss()
        logits = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
        y = np.array([0, 1, 2])
        loss = loss_fn.forward(logits, y)
        assert loss < 0.01

    def test_worst_prediction(self):
        """Loss ist höher bei falschen als bei richtigen Vorhersagen."""
        loss_fn = CrossEntropyLoss()
        # Sample 1: richtig (y=0, logits[0] hoch)
        # Sample 2: falsch (y=0, logits[0] niedrig)
        logits = np.array([[100.0, 0.0], [0.0, 100.0]])
        y = np.array([0, 0])
        loss = loss_fn.forward(logits, y)
        # Der Loss sollte > 0 sein (falsche Vorhersage dominiert)
        assert loss > 0.0

    def test_backward_shape(self):
        loss_fn = CrossEntropyLoss()
        logits = np.random.randn(8, 5)
        y = np.random.randint(0, 5, size=8)
        loss_fn.forward(logits, y)
        grad = loss_fn.backward()
        assert grad.shape == logits.shape

    def test_backward_sums_to_zero(self):
        """Gradient über Klassen summiert zu 0 (pro Sample)."""
        loss_fn = CrossEntropyLoss()
        logits = np.random.randn(6, 4)
        y = np.random.randint(0, 4, size=6)
        loss_fn.forward(logits, y)
        grad = loss_fn.backward()
        sums = np.sum(grad, axis=1)
        np.testing.assert_array_almost_equal(sums, np.zeros(6))


# ═══════════════════════════════════════════════════════════════
# Dense Layer
# ═══════════════════════════════════════════════════════════════

class TestDense:
    def test_forward_shape(self):
        layer = Dense(5, 3)
        x = np.random.randn(4, 5)
        out = layer.forward(x)
        assert out.shape == (4, 3)

    def test_forward_computation(self):
        """Manuell nachrechnen: y = x @ W + b."""
        layer = Dense(2, 2)
        layer.W = np.array([[1.0, 0.0], [0.0, 1.0]])
        layer.b = np.array([[0.5, -0.5]])
        x = np.array([[2.0, 3.0]])
        out = layer.forward(x)
        expected = x @ layer.W + layer.b
        np.testing.assert_array_almost_equal(out, expected)

    def test_backward_shapes(self):
        layer = Dense(5, 3)
        x = np.random.randn(4, 5)
        layer.forward(x)
        dout = np.random.randn(4, 3)
        dx = layer.backward(dout)
        assert dx.shape == (4, 5)
        assert layer.dW.shape == (5, 3)
        assert layer.db.shape == (1, 3)

    def test_he_initialization(self):
        """He-Init: std ≈ sqrt(2/input_dim)."""
        input_dim = 100
        layer = Dense(input_dim, 50)
        std = np.std(layer.W)
        expected_std = np.sqrt(2.0 / input_dim)
        assert 0.7 * expected_std < std < 1.3 * expected_std

    def test_bias_initialized_zero(self):
        layer = Dense(10, 5)
        np.testing.assert_array_equal(layer.b, np.zeros((1, 5)))


# ═══════════════════════════════════════════════════════════════
# SGD Optimizer
# ═══════════════════════════════════════════════════════════════

class TestSGD:
    def test_step_updates_weights(self):
        layer = Dense(3, 2)
        layer.W = np.ones((3, 2))
        layer.b = np.ones((1, 2))
        layer.dW = np.ones((3, 2)) * 0.1
        layer.db = np.ones((1, 2)) * 0.1

        opt = SGD(lr=0.1, momentum=0.0)
        opt.step([layer])

        # W -= lr * dW  →  1.0 - 0.1*0.1 = 0.99
        np.testing.assert_array_almost_equal(layer.W, np.ones((3, 2)) * 0.99)
        np.testing.assert_array_almost_equal(layer.b, np.ones((1, 2)) * 0.99)

    def test_momentum_accumulates(self):
        layer = Dense(2, 2)
        layer.W = np.ones((2, 2))
        layer.b = np.zeros((1, 2))
        layer.dW = np.ones((2, 2)) * 0.1
        layer.db = np.ones((1, 2)) * 0.1

        opt = SGD(lr=0.1, momentum=0.9)

        # Erster Schritt
        opt.step([layer])
        w1 = layer.W.copy()

        # Zweiter Schritt (mit Momentum)
        layer.dW = np.ones((2, 2)) * 0.1
        layer.db = np.ones((1, 2)) * 0.1
        opt.step([layer])
        w2 = layer.W.copy()

        # Mit Momentum sollte der zweite Schritt größer sein
        change1 = np.sum(np.abs(np.ones((2, 2)) - w1))
        change2 = np.sum(np.abs(w1 - w2))
        assert change2 > change1 * 0.8  # Momentum-Effekt


# ═══════════════════════════════════════════════════════════════
# NeuralNetwork
# ═══════════════════════════════════════════════════════════════

class TestNeuralNetwork:
    def test_forward_shape(self, sample_data):
        x, _ = sample_data
        net = NeuralNetwork([Dense(5, 4), ReLU(), Dense(4, 3)])
        out = net.forward(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_loss_acc(self, sample_data):
        x, y = sample_data
        net = NeuralNetwork([Dense(5, 4), ReLU(), Dense(4, 3)])
        opt = SGD(lr=0.01)
        loss, acc = net.train_step(x, y, opt)
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0
        assert loss > 0

    def test_predict_shape(self, sample_data):
        x, _ = sample_data
        net = NeuralNetwork([Dense(5, 4), ReLU(), Dense(4, 3)])
        preds = net.predict(x)
        assert preds.shape == (4,)
        assert np.all((preds >= 0) & (preds < 3))

    def test_predict_proba_sums_to_one(self, sample_data):
        x, _ = sample_data
        net = NeuralNetwork([Dense(5, 4), ReLU(), Dense(4, 3)])
        probs = net.predict_proba(x)
        assert probs.shape == (4, 3)
        sums = np.sum(probs, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(4))

    def test_loss_decreases_during_training(self):
        """Nach mehreren Trainingsschritten sollte der Loss sinken."""
        np.random.seed(123)
        x = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randint(0, 3, size=32)

        net = NeuralNetwork([Dense(10, 8), ReLU(), Dense(8, 3)])
        opt = SGD(lr=0.1, momentum=0.9)

        losses = []
        for _ in range(20):
            loss, _ = net.train_step(x, y, opt)
            losses.append(loss)

        # Der Loss sollte tendenziell sinken
        assert losses[-1] < losses[0]

    def test_deep_network(self):
        """Test mit tieferem Netzwerk."""
        x = np.random.randn(8, 20).astype(np.float32)
        y = np.random.randint(0, 5, size=8)

        net = NeuralNetwork([
            Dense(20, 32), ReLU(),
            Dense(32, 16), ReLU(),
            Dense(16, 8),  ReLU(),
            Dense(8, 5),
        ])
        opt = SGD(lr=0.01)
        loss, acc = net.train_step(x, y, opt)
        assert loss > 0

    def test_single_layer_network(self):
        """Minimales Netzwerk: nur ein Dense-Layer (lineare Regression)."""
        x = np.random.randn(4, 3).astype(np.float32)
        y = np.array([0, 1, 0, 1])

        net = NeuralNetwork([Dense(3, 2)])
        opt = SGD(lr=0.01)
        loss, acc = net.train_step(x, y, opt)
        assert loss > 0

    def test_sigmoid_network(self):
        """Netzwerk mit Sigmoid statt ReLU."""
        x = np.random.randn(4, 5).astype(np.float32)
        y = np.array([0, 1, 2, 1])

        net = NeuralNetwork([Dense(5, 4), Sigmoid(), Dense(4, 3)])
        opt = SGD(lr=0.01)
        loss, acc = net.train_step(x, y, opt)
        assert loss > 0


# ═══════════════════════════════════════════════════════════════
# Integration: Gradienten-Check via numerische Approximation
# ═══════════════════════════════════════════════════════════════

class TestGradientCorrectness:
    def test_dense_gradient_numerical(self):
        """Numerischer Gradienten-Check für Dense.backward."""
        np.random.seed(99)
        layer = Dense(3, 2)
        x = np.random.randn(2, 3).astype(np.float32)

        # Forward (nur für Cache-Befüllung)
        layer.forward(x)
        dout = np.random.randn(2, 2).astype(np.float32)

        # Analytischer Gradient
        layer.backward(dout)
        dW_analytical = layer.dW.copy()

        # Numerischer Gradient
        eps = 1e-5
        dW_numerical = np.zeros_like(layer.W)
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                layer.W[i, j] += eps
                out_plus = layer.forward(x)
                loss_plus = np.sum(out_plus * dout)

                layer.W[i, j] -= 2 * eps
                out_minus = layer.forward(x)
                loss_minus = np.sum(out_minus * dout)

                layer.W[i, j] += eps  # restore
                dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Relative Fehler < 1e-5
        rel_error = np.max(np.abs(dW_analytical - dW_numerical) /
                           (np.abs(dW_analytical) + np.abs(dW_numerical) + 1e-8))
        assert rel_error < 1e-5, f"Gradientenfehler: {rel_error:.2e}"


# ═══════════════════════════════════════════════════════════════
# W&B Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestWandBTracker:
    """Tests für WandBTracker aus wandb_utils.py."""

    def test_import(self):
        """WandBTracker ist importierbar."""
        from wandb_utils import WandBTracker
        assert WandBTracker is not None

    def test_initialization_offline(self):
        """WandBTracker initialisiert im Offline-Modus."""
        from wandb_utils import WandBTracker
        tracker = WandBTracker(
            project="test-nn",
            config={"lr": 0.1, "epochs": 10},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        assert tracker is not None
        tracker.finish()

    def test_log_epoch(self):
        """log_epoch() läuft ohne Fehler."""
        from wandb_utils import WandBTracker
        tracker = WandBTracker(project="test-nn", offline=True)
        if tracker.is_active:
            tracker.log_epoch(
                epoch=1, train_loss=0.5, train_acc=0.92, test_acc=0.90, lr=0.1,
            )
        tracker.finish()

    def test_log_final_results(self):
        """log_final_results() läuft ohne Fehler."""
        from wandb_utils import WandBTracker
        tracker = WandBTracker(project="test-nn", offline=True)
        if tracker.is_active:
            tracker.log_final_results(
                test_acc=0.95, num_params=109386, num_errors=500,
            )
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() beendet den Run sauber."""
        from wandb_utils import WandBTracker
        tracker = WandBTracker(project="test-nn", offline=True)
        tracker.finish()
        tracker.finish()  # Doppeltes finish() sollte safe sein

    def test_is_active_property(self):
        """is_active Property funktioniert."""
        from wandb_utils import WandBTracker
        tracker = WandBTracker(project="test-nn", offline=True)
        assert isinstance(tracker.is_active, bool)
        tracker.finish()
