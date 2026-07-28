# Neuronales Netz von Grund auf 🧠

**Nur NumPy. Kein TensorFlow, kein PyTorch. Reine Mathematik.**

Dieses Repository implementiert ein vollständiges Feedforward-Neuronales Netzwerk von Grund auf — ideal zum Lernen und Verstehen der fundamentalen Konzepte.

## 📦 Enthaltene Komponenten

| Komponente | Beschreibung |
|---|---|
| `Dense` | Vollständig verbundener Layer: `y = x @ W + b` |
| `ReLU` | Rectified Linear Unit: `f(x) = max(0, x)` |
| `Sigmoid` | Sigmoid-Aktivierung: `f(x) = 1/(1+e^(-x))` |
| `Softmax` | Softmax mit numerischer Stabilität |
| `CrossEntropyLoss` | Cross-Entropy Loss (kombiniert mit Softmax) |
| `SGD` | Stochastic Gradient Descent mit Momentum |
| `NeuralNetwork` | Komplettes Netzwerk mit Forward/Backward/Train |

## 🚀 Quickstart

```bash
# Abhängigkeiten installieren
uv pip install numpy

# MNIST trainieren
python train_mnist.py
```

## 🧪 Erwartetes Ergebnis

Nach 10 Epochen auf MNIST:
- **Test-Accuracy: ~97%**
- Trainingszeit: ~30 Sekunden (CPU)
- ~110.000 Parameter

## 📖 Lernpfad

1. **`nn_core.py`** — Die Bausteine verstehen:
   - Wie funktioniert ein Dense-Layer? (Matrix-Multiplikation!)
   - Was macht ReLU? (Nichtlinearität!)
   - Wie funktioniert Backpropagation? (Kettenregel!)

2. **`train_mnist.py`** — Das große Ganze:
   - Daten laden und vorbereiten
   - Training-Loop: Forward → Loss → Backward → Update
   - Evaluation und Metriken

3. **Experimente:**
   - Andere Architekturen ausprobieren (mehr/weniger Layer)
   - Learning-Rate und Momentum variieren
   - Sigmoid statt ReLU testen

## 🔬 Die Mathematik dahinter

### Forward-Pass
```
h₁ = ReLU(x @ W₁ + b₁)
h₂ = ReLU(h₁ @ W₂ + b₂)
ŷ  = h₂ @ W₃ + b₃       (Logits)
```

### Backward-Pass (Kettenregel)
```
∂L/∂ŷ  → Loss.backward()
∂L/∂h₂ ← ∂L/∂ŷ @ W₃ᵀ   (Dense.backward)
∂L/∂h₁ ← ∂L/∂h₂ ⊙ ReLU'(h₂)  (ReLU.backward)
...
```

### Parameter-Update (SGD mit Momentum)
```
v = β·v - η·∂L/∂W
W = W + v
```

## 📚 Weiterführende Ressourcen

- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Karpathy: micrograd](https://github.com/karpathy/micrograd)
- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/)
