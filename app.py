"""
Streamlit-App: Neuronales Netz von Grund auf
============================================
Interaktive Visualisierung: Architektur, Forward-Pass, Backpropagation,
MNIST-Training live mit NumPy.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Repo-Module importieren
sys.path.insert(0, os.path.dirname(__file__))
from nn_core import Dense, ReLU, Sigmoid, Softmax, CrossEntropyLoss, SGD, NeuralNetwork

st.set_page_config(page_title="Neuronales Netz von Grund auf", layout="wide")
st.title("🧠 Neuronales Netz von Grund auf")
st.markdown("### Nur mit NumPy — kein TensorFlow, kein PyTorch")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏗️ Architektur", "⚡ Forward-Pass", "↩️ Backpropagation", "🏋️ MNIST-Training"
])

# ═══════════════════════════════════════════════════════════════
# Tab 1: Architektur
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("🏗️ Netzwerk-Architektur")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Layer konfigurieren")
        n_hidden = st.slider("Anzahl Hidden-Layer", 1, 5, 2)
        layers_config = []
        input_dim = 784
        for i in range(n_hidden):
            default = max(256 // (2**i), 10)
            neurons = st.number_input(
                f"Neuronen in Layer {i+1}",
                min_value=2, max_value=512, value=default, step=2,
                key=f"neurons_{i}"
            )
            layers_config.append(neurons)

        output_dim = st.number_input("Output-Neuronen", 2, 100, 10, key="output_neurons")

    with col2:
        st.subheader("Visualisierung")

        # Baue Architektur-Liste
        dims = [input_dim] + layers_config + [output_dim]
        layer_names = []
        for i in range(len(dims) - 1):
            layer_names.append(f"Dense({dims[i]}, {dims[i+1]})")
            if i < len(dims) - 2:
                layer_names.append("ReLU")

        # Zeichne Architektur-Diagramm
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-1, len(dims) * 2)
        ax.set_ylim(-max(dims) / 2 - 20, max(dims) / 2 + 20)
        ax.axis("off")

        for i, dim in enumerate(dims):
            x = i * 2
            # Zeichne Neuronen als Kreise
            max_show = min(dim, 20)
            spacing = max(1, dim / max_show)
            for j in range(max_show):
                y = (j - max_show / 2) * spacing * 2
                ax.add_patch(plt.Circle((x, y), 0.3, fill=True,
                                        color="#4A90D9" if i == 0 else
                                        "#50C878" if i == len(dims) - 1 else
                                        "#FF6B6B", alpha=0.8))
            # Label
            label = f"Input\n{dim}" if i == 0 else f"Output\n{dim}" if i == len(dims) - 1 else f"Hidden {i}\n{dim}"
            ax.text(x, -max(dims) / 2 - 15, label, ha="center", fontsize=9, fontweight="bold")

            # Verbindungen zum nächsten Layer
            if i < len(dims) - 1:
                next_max = min(dims[i+1], 20)
                next_spacing = max(1, dims[i+1] / next_max)
                for j in range(min(max_show, 5)):
                    y1 = (j - max_show / 2) * spacing * 2
                    for k in range(min(next_max, 5)):
                        y2 = (k - next_max / 2) * next_spacing * 2
                        ax.plot([x + 0.3, x + 1.7], [y1, y2], 'gray', alpha=0.15, linewidth=0.5)

        ax.set_title("Netzwerk-Architektur", fontsize=14, fontweight="bold")
        st.pyplot(fig)

        # Parameter zählen
        total = sum(dims[i] * dims[i+1] + dims[i+1] for i in range(len(dims) - 1))
        st.metric("Gesamt-Parameter", f"{total:,}")
        st.caption(f"Gewichte: {sum(dims[i] * dims[i+1] for i in range(len(dims) - 1)):,} | "
                   f"Biases: {sum(dims[i+1] for i in range(len(dims) - 1)):,}")

    st.subheader("📖 Layer-Erklärung")
    with st.expander("Dense Layer (Vollständig verbunden)", expanded=False):
        st.markdown(r"""
        **Formel:** $y = x \cdot W + b$

        - **x**: Eingabevektor (z.B. 784 Pixel)
        - **W**: Gewichtsmatrix — wird beim Training gelernt
        - **b**: Bias — Verschiebung
        - **Initialisierung**: He-Initialisierung $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$

        Jedes Neuron ist mit **jedem** Neuron des vorherigen Layers verbunden.
        """)

    with st.expander("ReLU (Rectified Linear Unit)", expanded=False):
        st.markdown(r"""
        **Formel:** $f(x) = \max(0, x)$

        - Negative Werte → 0
        - Positive Werte → unverändert
        - **Vorteil**: Kein "Vanishing Gradient" wie bei Sigmoid
        - **Nachteil**: "Dying ReLU" — Neuronen, die immer 0 ausgeben
        """)
        x_relu = np.linspace(-5, 5, 100)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(x_relu, np.maximum(0, x_relu), 'b-', linewidth=2)
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.axvline(0, color='gray', linestyle='--')
        ax2.set_title("ReLU: f(x) = max(0, x)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with st.expander("Softmax", expanded=False):
        st.markdown(r"""
        **Formel:** $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

        - Wandelt **Logits** (Rohwerte) in **Wahrscheinlichkeiten** um
        - Summe aller Ausgaben = 1.0
        - Die höchste Wahrscheinlichkeit "gewinnt" → Klassenvorhersage
        """)

# ═══════════════════════════════════════════════════════════════
# Tab 2: Forward-Pass
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("⚡ Forward-Pass Schritt für Schritt")

    st.markdown("""
    Der **Forward-Pass** ist die Vorhersage-Phase: Daten fließen von links nach rechts
    durch das Netzwerk, Layer für Layer.
    """)

    if st.button("🔄 Forward-Pass simulieren", type="primary"):
        # Kleines Demo-Netzwerk
        demo_net = NeuralNetwork([
            Dense(4, 3), ReLU(),
            Dense(3, 2),
        ])

        # Beispiel-Input
        x_demo = np.array([[1.0, 0.5, -0.3, 2.0]])

        st.subheader("1. Input")
        st.code(f"x = {x_demo.tolist()}  (4 Features)")

        # Schritt für Schritt
        out = x_demo
        step = 1
        for layer in demo_net.layers:
            out = layer.forward(out)
            name = type(layer).__name__

            if isinstance(layer, Dense):
                st.subheader(f"{step}. {name}")
                st.markdown(f"**Gewichtsmatrix W** ({layer.W.shape[0]}×{layer.W.shape[1]}):")
                st.dataframe(np.round(layer.W, 3))
                st.markdown(f"**Bias b** (1×{layer.b.shape[1]}):")
                st.dataframe(np.round(layer.b, 3))
                st.markdown(f"**Berechnung:** `x @ W + b`")
                st.code(f"Output = {np.round(out, 4).tolist()}")
            elif isinstance(layer, ReLU):
                st.subheader(f"{step}. {name}")
                st.markdown(f"**Berechnung:** `max(0, x)` — negative Werte werden zu 0")
                st.code(f"Output = {np.round(out, 4).tolist()}")

            step += 1

        st.subheader(f"{step}. Softmax (implizit in Loss)")
        shifted = out - np.max(out, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        st.markdown("**Wandelt Logits in Wahrscheinlichkeiten:**")
        st.code(f"Wahrscheinlichkeiten = {np.round(probs, 4).tolist()}")
        st.markdown(f"**Vorhersage:** Klasse **{np.argmax(probs)}** "
                    f"(Konfidenz: {probs[0, np.argmax(probs)]:.2%})")

        st.success("✅ Forward-Pass abgeschlossen!")

# ═══════════════════════════════════════════════════════════════
# Tab 3: Backpropagation
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("↩️ Backpropagation Schritt für Schritt")

    st.markdown("""
    **Backpropagation** berechnet die Gradienten rückwärts durch das Netzwerk.
    Das ist der "Lern"-Schritt — hier wird berechnet, wie stark jeder Parameter
    zum Fehler beigetragen hat.

    **Kettenregel:** dL/dW = dL/dy · dy/dW
    """)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📐 Gradienten-Fluss")
        st.markdown("""
        | Schritt | Berechnung |
        |---------|-----------|
        | 1. Loss | dL/d(logits) |
        | 2. Dense (rückwärts) | dL/dx = dL/dy · Wᵀ |
        | 3. ReLU (rückwärts) | dL/dx = dL/dy · 𝟙[x > 0] |
        | 4. Dense (rückwärts) | dL/dW = xᵀ · dL/dy |
        """)

    with col_b:
        st.subheader("🔢 Beispiel-Rechnung")
        if st.button("📊 Backpropagation demonstrieren", type="primary"):
            # Mini-Netzwerk für Demo
            bp_net = NeuralNetwork([
                Dense(2, 3), ReLU(),
                Dense(3, 2),
            ])

            x_bp = np.array([[0.5, -0.2]])
            y_true = np.array([1])  # Zielklasse

            # Forward
            logits = bp_net.forward(x_bp)
            loss = bp_net.loss_fn.forward(logits, y_true)

            st.markdown(f"**Input:** {x_bp.tolist()}")
            st.markdown(f"**Logits:** {np.round(logits, 4).tolist()}")
            st.markdown(f"**Zielklasse:** {y_true[0]}")
            st.markdown(f"**Loss:** {loss:.4f}")

            # Backward
            grad = bp_net.loss_fn.backward()
            st.markdown(f"**Loss-Gradient (dL/dlogits):**")
            st.code(f"{np.round(grad, 4).tolist()}")

            for layer in reversed(bp_net.layers):
                grad = layer.backward(grad)
                if isinstance(layer, Dense):
                    st.markdown(f"**dW (Gradient der Gewichte):**")
                    st.dataframe(np.round(layer.dW, 4))
                    st.markdown(f"**db (Gradient des Bias):**")
                    st.dataframe(np.round(layer.db, 4))

            st.success("✅ Backpropagation demonstriert!")

    st.subheader("📖 Die Intuition")
    st.info("""
    **Merkregel:** Backpropagation = "Wer war schuld am Fehler?"

    1. Der Loss sagt: "Die Vorhersage war um X daneben"
    2. Der Gradient fließt rückwärts und verteilt die "Schuld" auf jeden Parameter
    3. Jeder Parameter wird proportional zu seinem Gradienten angepasst
    4. Nächste Vorhersage ist (hoffentlich) besser!
    """)

# ═══════════════════════════════════════════════════════════════
# Tab 4: MNIST-Training
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("🏋️ MNIST-Training live")

    st.markdown("""
    Trainiere ein echtes Neuronales Netz auf dem MNIST-Datensatz (handgeschriebene Ziffern).
    Alles live — du siehst, wie Loss und Accuracy sich verbessern!
    """)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        lr = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5], value=0.1)
    with col_cfg2:
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
    with col_cfg3:
        n_epochs = st.slider("Epochen", 1, 20, 5)

    if st.button("🚀 Training starten", type="primary"):
        # MNIST laden
        with st.spinner("📦 Lade MNIST-Daten..."):
            from train_mnist import load_mnist
            X_train, y_train, X_test, y_test = load_mnist()

        st.success(f"✅ Daten geladen: {X_train.shape[0]:,} Train / {X_test.shape[0]:,} Test")

        # Netzwerk bauen
        net = NeuralNetwork([
            Dense(784, 128), ReLU(),
            Dense(128, 64), ReLU(),
            Dense(64, 10),
        ])
        optimizer = SGD(lr=lr, momentum=0.9)

        # Live-Chart
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        progress_bar = st.progress(0)

        loss_history = []
        acc_history = []
        test_acc_history = []

        for epoch in range(n_epochs):
            # Shuffle
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

            total_loss = 0
            total_acc = 0
            n_batches = 0

            for i in range(0, min(len(X_train), 5000), batch_size):
                x_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                loss, acc = net.train_step(x_batch, y_batch, optimizer)
                total_loss += loss
                total_acc += acc
                n_batches += 1

            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            test_preds = net.predict(X_test[:2000])
            test_acc = np.mean(test_preds == y_test[:2000])

            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            test_acc_history.append(test_acc)

            # Update Charts
            fig_chart, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(range(1, len(loss_history) + 1), loss_history, 'r-o', markersize=6)
            ax1.set_title("Loss über Epochen", fontweight="bold")
            ax1.set_xlabel("Epoche")
            ax1.set_ylabel("Loss")
            ax1.grid(True, alpha=0.3)

            ax2.plot(range(1, len(acc_history) + 1), acc_history, 'b-o', label="Train", markersize=6)
            ax2.plot(range(1, len(test_acc_history) + 1), test_acc_history, 'g-s', label="Test", markersize=6)
            ax2.set_title("Accuracy über Epochen", fontweight="bold")
            ax2.set_xlabel("Epoche")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)

            plt.tight_layout()
            chart_placeholder.pyplot(fig_chart)
            plt.close()

            metrics_placeholder.metric(
                f"Epoche {epoch + 1}/{n_epochs}",
                f"Test-Acc: {test_acc:.2%}",
                f"Loss: {avg_loss:.4f} | Train-Acc: {avg_acc:.2%}"
            )
            progress_bar.progress((epoch + 1) / n_epochs)

        # Finale Evaluation
        st.balloons()
        final_preds = net.predict(X_test)
        final_acc = np.mean(final_preds == y_test)
        st.success(f"### 🎉 Training abgeschlossen! Finale Test-Accuracy: **{final_acc:.2%}**")

        # Zeige einige Vorhersagen
        st.subheader("Beispiel-Vorhersagen")
        n_show = 10
        idx_show = np.random.choice(len(X_test), n_show, replace=False)
        fig_samples, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            img_idx = idx_show[i]
            ax.imshow(X_test[img_idx].reshape(28, 28), cmap='gray')
            pred = final_preds[img_idx]
            true = y_test[img_idx]
            color = 'green' if pred == true else 'red'
            ax.set_title(f"Pred: {pred} | True: {true}", color=color, fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig_samples)

st.sidebar.markdown("""
### 📚 Über diese App

Diese App demonstriert ein **Neuronales Netz von Grund auf** —
implementiert nur mit **NumPy**, ohne Deep-Learning-Frameworks.

**Was du lernst:**
- 🏗️ Wie ein NN aufgebaut ist
- ⚡ Wie der Forward-Pass funktioniert
- ↩️ Wie Backpropagation Gradienten berechnet
- 🏋️ Wie Training in der Praxis aussieht

**Code:** `nn_core.py` im Repo
""")
