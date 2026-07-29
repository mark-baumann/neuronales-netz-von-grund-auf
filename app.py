"""
Streamlit-App: Neuronales Netz von Grund auf
=============================================
NN-Architektur visualisieren, Forward-Pass animieren, Backpropagation erklären, MNIST-Training.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Modul-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent))

from nn_core import Dense, ReLU, Sigmoid, Softmax, CrossEntropyLoss, SGD, NeuralNetwork

st.set_page_config(page_title="NN von Grund auf", page_icon="🧠", layout="wide")
st.title("🧠 Neuronales Netz von Grund auf")
st.markdown("Nur mit NumPy — kein TensorFlow, kein PyTorch. Reine Mathematik!")

page = st.sidebar.radio(
    "Bereich wählen",
    ["Architektur", "Forward-Pass", "Backpropagation", "MNIST-Training", "Gewichte visualisieren"]
)

# ═══════════════════════════════════════════════════════════════════════════
# ARCHITEKTUR
# ═══════════════════════════════════════════════════════════════════════════
if page == "Architektur":
    st.header("🏗️ Netzwerk-Architektur")

    st.markdown("""
    ### 784 → 128 → 64 → 10
    
    **Layer für Layer:**
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Input-Layer")
        st.markdown("""
        **784 Neuronen**
        - 28×28 Pixel = 784
        - Jedes Pixel ein Input
        - Werte: [0, 1] (normalisiert)
        """)
        st.metric("Dimension", "784")

    with col2:
        st.subheader("Hidden-Layer 1")
        st.markdown("""
        **128 Neuronen + ReLU**
        - Dense(784, 128)
        - He-Initialisierung
        - ReLU-Aktivierung
        """)
        st.metric("Parameter", f"{784*128 + 128:,}")

    with col3:
        st.subheader("Hidden-Layer 2")
        st.markdown("""
        **64 Neuronen + ReLU**
        - Dense(128, 64)
        - He-Initialisierung
        - ReLU-Aktivierung
        """)
        st.metric("Parameter", f"{128*64 + 64:,}")

    with col4:
        st.subheader("Output-Layer")
        st.markdown("""
        **10 Neuronen (Logits)**
        - Dense(64, 10)
        - Keine Aktivierung
        - Softmax + CrossEntropy
        """)
        st.metric("Parameter", f"{64*10 + 10:,}")

    total = 784*128 + 128 + 128*64 + 64 + 64*10 + 10
    st.metric("Gesamt-Parameter", f"{total:,}")

    st.subheader("Aktivierungsfunktionen")
    func = st.selectbox("Funktion wählen", ["ReLU", "Sigmoid", "Softmax"])

    x = np.linspace(-5, 5, 200)
    fig, ax = plt.subplots(figsize=(8, 4))

    if func == "ReLU":
        y = np.maximum(0, x)
        ax.set_title("ReLU: f(x) = max(0, x)")
    elif func == "Sigmoid":
        y = 1 / (1 + np.exp(-x))
        ax.set_title("Sigmoid: f(x) = 1/(1+e^(-x))")
    else:
        exp_x = np.exp(x - np.max(x))
        y = exp_x / np.sum(exp_x)
        ax.set_title("Softmax (Beispiel mit 200 Werten)")

    ax.plot(x, y, linewidth=2, color="#FF6B6B")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# FORWARD-PASS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Forward-Pass":
    st.header("➡️ Forward-Pass — Schritt für Schritt")

    st.markdown("""
    Der Forward-Pass berechnet die Ausgabe des Netzwerks für einen Input.
    
    **Ablauf:**
    1. Input → Dense-Layer 1: `h1 = x @ W1 + b1`
    2. Aktivierung: `a1 = ReLU(h1)`
    3. Dense-Layer 2: `h2 = a1 @ W2 + b2`
    4. Aktivierung: `a2 = ReLU(h2)`
    5. Dense-Layer 3: `logits = a2 @ W3 + b3`
    6. Softmax: `probs = softmax(logits)`
    """)

    if st.button("Forward-Pass mit Zufallsdaten simulieren"):
        # Kleines Netzwerk bauen
        net = NeuralNetwork([
            Dense(784, 128), ReLU(),
            Dense(128, 64), ReLU(),
            Dense(64, 10),
        ])

        # Zufälligen Input
        x = np.random.randn(1, 784).astype(np.float32) * 0.1

        # Forward-Pass mit Zwischenergebnissen
        st.subheader("Zwischenergebnisse")

        out = x
        st.write(f"**Input:** Shape {out.shape}, Wertebereich [{out.min():.3f}, {out.max():.3f}]")

        for i, layer in enumerate(net.layers):
            out = layer.forward(out)
            name = type(layer).__name__
            if isinstance(layer, Dense):
                st.write(f"**Layer {i}: {name}** → Shape {out.shape}, Wertebereich [{out.min():.3f}, {out.max():.3f}]")
            else:
                st.write(f"**Layer {i}: {name}** → Shape {out.shape}, Wertebereich [{out.min():.3f}, {out.max():.3f}]")

        # Softmax-Probabilities
        shifted = out - np.max(out, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        st.subheader("Softmax-Wahrscheinlichkeiten")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(10), probs[0], color="#4ECDC4", edgecolor="white")
        ax.set_xlabel("Ziffer")
        ax.set_ylabel("Wahrscheinlichkeit")
        ax.set_title("Vorhergesagte Wahrscheinlichkeiten (vor Training)")
        ax.set_xticks(range(10))
        st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# BACKPROPAGATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Backpropagation":
    st.header("⬅️ Backpropagation — Gradienten fließen rückwärts")

    st.markdown("""
    ### Die Kettenregel in Aktion
    
    Backpropagation berechnet die Gradienten der Loss-Funktion bezüglich aller Parameter.
    
    **Ablauf (rückwärts):**
    1. **Loss-Gradient:** `∂L/∂logits` (Softmax + CrossEntropy kombiniert)
    2. **Layer 3:** `∂L/∂W3 = a2.T @ ∂L/∂logits`, `∂L/∂b3 = sum(∂L/∂logits)`
    3. **ReLU 2:** `∂L/∂a2 = ∂L/∂h2 * (h2 > 0)` (Gradient nur wo Input > 0)
    4. **Layer 2:** `∂L/∂W2 = a1.T @ ∂L/∂h2`
    5. **ReLU 1:** Gradient durchreichen
    6. **Layer 1:** `∂L/∂W1 = x.T @ ∂L/∂h1`
    """)

    st.subheader("ReLU-Gradient visualisieren")
    st.markdown("ReLU leitet den Gradienten nur durch, wenn der Input > 0 war. Sonst: 0.")

    x = np.linspace(-5, 5, 200)
    relu = np.maximum(0, x)
    relu_grad = (x > 0).astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, relu, linewidth=2, color="#FF6B6B")
    ax1.set_title("ReLU: f(x) = max(0, x)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, relu_grad, linewidth=2, color="#4ECDC4")
    ax2.set_title("ReLU-Gradient: f'(x) = 1 wenn x>0, sonst 0")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.subheader("SGD mit Momentum")
    st.markdown("""
    **Update-Regel:**
    ```
    v = momentum * v - lr * gradient
    weight = weight + v
    ```
    
    Momentum (0.9) beschleunigt die Konvergenz und glättet Oszillationen.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# MNIST-TRAINING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "MNIST-Training":
    st.header("🏋️ MNIST-Training live")

    st.markdown("Trainiere das selbstgebaute NN auf MNIST — direkt in der App!")

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Epochen", 1, 20, 5)
    with col2:
        lr = st.selectbox("Learning-Rate", [0.01, 0.05, 0.1, 0.2, 0.5], index=2)
    with col3:
        batch_size = st.selectbox("Batch-Size", [32, 64, 128], index=1)

    if st.button("Training starten", type="primary"):
        from train_mnist import load_mnist

        with st.spinner("Lade MNIST-Daten..."):
            X_train, y_train, X_test, y_test = load_mnist()
        st.success(f"✅ Daten geladen: {X_train.shape[0]:,} Train, {X_test.shape[0]:,} Test")

        # Netzwerk bauen
        net = NeuralNetwork([
            Dense(784, 128), ReLU(),
            Dense(128, 64), ReLU(),
            Dense(64, 10),
        ])
        optimizer = SGD(lr=lr, momentum=0.9)

        # Training
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        acc_chart = st.empty()

        train_losses = []
        test_accs = []

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

            total_loss = 0
            total_acc = 0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                loss, acc = net.train_step(x_batch, y_batch, optimizer)
                total_loss += loss
                total_acc += acc
                n_batches += 1

            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            test_preds = net.predict(X_test)
            test_acc = np.mean(test_preds == y_test)

            train_losses.append(avg_loss)
            test_accs.append(test_acc)

            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoche {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train-Acc={avg_acc:.3f}, Test-Acc={test_acc:.3f}")

        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(range(1, epochs+1), train_losses, "b-o", linewidth=2)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoche")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(range(1, epochs+1), test_accs, "g-o", linewidth=2)
        ax2.set_title("Test-Accuracy")
        ax2.set_xlabel("Epoche")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

        st.success(f"✅ Training abgeschlossen! Finale Test-Accuracy: {test_accs[-1]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# GEWICHTE VISUALISIEREN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Gewichte visualisieren":
    st.header("🔍 Gewichte visualisieren")

    st.markdown("""
    Die Gewichte des ersten Layers (784×128) können als 28×28-Bilder visualisiert werden.
    Jedes der 128 Neuronen lernt, auf bestimmte Muster zu reagieren.
    """)

    if st.button("Zufällige Gewichte anzeigen"):
        # Zufällige Gewichte generieren (He-Initialisierung)
        W = np.random.randn(784, 128) * np.sqrt(2.0 / 784)

        n_cols = 8
        n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
        axes = axes.flatten()

        for i in range(n_rows * n_cols):
            neuron_weights = W[:, i].reshape(28, 28)
            axes[i].imshow(neuron_weights, cmap="RdBu", vmin=-0.5, vmax=0.5)
            axes[i].set_title(f"Neuron {i+1}", fontsize=8)
            axes[i].axis("off")

        fig.suptitle("Gewichte des ersten Hidden-Layers (784→128)", fontsize=14)
        st.pyplot(fig)

        st.info("💡 Nach dem Training zeigen die Gewichte erkennbare Muster (Striche, Kurven, Kreise).")

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **NN von Grund auf** — Nur NumPy!")
st.sidebar.markdown("[GitHub Repository](https://github.com/mark-baumann/neuronales-netz-von-grund-auf)")
