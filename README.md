# 🧠 Neuronales Netz von Grund auf — Nur mit NumPy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

Ein vollständig von Hand implementiertes neuronales Netz — **ohne TensorFlow, ohne PyTorch, nur mit NumPy**. Dieses Projekt zeigt, was wirklich unter der Haube passiert: Forward-Pass, Backpropagation, Gradient Descent und Training auf MNIST — alles Schritt für Schritt nachvollziehbar und interaktiv visualisiert.

## ✨ Features

- **🏗️ Architektur-Visualisierung** — Netzwerk-Topologie mit Layer-Dimensionen und Parameteranzahl
- **⚡ Forward-Pass** — Live-Demonstration: Eingabe → Gewichtete Summe → Aktivierung → Ausgabe
- **↩️ Backpropagation** — Gradientenfluss rückwärts durch das Netz, Schritt für Schritt erklärt
- **🏋️ MNIST-Training** — Training auf echten MNIST-Daten mit Live-Verlustkurve
- **🧩 Modulare Bausteine** — `Dense`, `ReLU`, `SGD` als wiederverwendbare Komponenten
- **📊 W&B-Integration** — Experiment-Tracking mit Weights & Biases
- **✅ Vollständig getestet** — Unit-Tests für alle Kernkomponenten

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/neuronales-netz-von-grund-auf.git
cd neuronales-netz-von-grund-auf

# Virtuelle Umgebung erstellen
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Abhängigkeiten installieren
uv pip install -e ".[dev]"
```

## 🎯 Nutzung

```bash
# Streamlit-App starten
streamlit run app.py
```

Die App öffnet sich im Browser unter `http://localhost:8501`. Erkunde die vier Tabs: Architektur, Forward-Pass, Backpropagation und MNIST-Training.

## 🧪 Tests ausführen

```bash
pytest -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **NumPy** | Alle mathematischen Operationen (Matrix-Multiplikation, Aktivierungsfunktionen) |
| **Streamlit** | Interaktive Web-App |
| **Matplotlib** | Visualisierung von Trainingsverläufen und Architektur |
| **Weights & Biases** | Experiment-Tracking |
| **Pytest** | Test-Framework |
| **Ruff** | Linting & Code-Qualität |

## 📁 Projektstruktur

```
neuronales-netz-von-grund-auf/
├── app.py                  # Streamlit-Hauptapp
├── pyproject.toml          # Projekt-Konfiguration
├── nn_core.py              # Kern-Implementierung (Dense, ReLU, NeuralNetwork, SGD)
├── train_mnist.py          # MNIST-Trainingsskript
├── wandb_utils.py          # W&B-Integration
├── test_nn_core.py         # Unit-Tests für nn_core
└── tests/
    └── test_wandb_utils.py # Tests für W&B-Utilities
```

## 🧮 Was du lernst

- **Forward-Pass**: Wie aus Eingabedaten eine Vorhersage wird
- **Backpropagation**: Wie der Fehler durch das Netz zurückfließt
- **Gradient Descent**: Wie Gewichte optimiert werden
- **Aktivierungsfunktionen**: ReLU und ihre Ableitung
- **Loss-Funktionen**: Cross-Entropy für Klassifikation

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Der beste Weg, neuronale Netze zu verstehen, ist, sie selbst zu bauen. Dieses Projekt verzichtet bewusst auf High-Level-Frameworks und zeigt jeden Rechenschritt.*
