"""
W&B Experiment Tracking für Neuronales Netz von Grund auf
=========================================================
Integriert Weights & Biases in das Training des neuronalen Netzes.
Loggt Epochen-Metriken, Modell-Architektur und Hyperparameter.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="neuronales-netz", config={...})
    tracker.log_epoch(epoch=1, train_loss=0.5, train_acc=0.92, test_acc=0.90, lr=0.01)
    tracker.finish()
"""

import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für das Neuronale Netz von Grund auf.

    Features:
    - Epochen-Metriken (Loss, Accuracy)
    - Modell-Architektur-Logging
    - Finale Ergebnisse tracken
    """

    def __init__(self, project: str = "neuronales-netz",
                 config: dict = None, tags: list = None,
                 group: str = None, job_type: str = "train",
                 notes: str = None, offline: bool = False):
        self.project = project
        self.run = None

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["nn", "from-scratch"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  test_acc: float, lr: float):
        """Loggt eine Trainings-Epoche."""
        self.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "test/accuracy": test_acc,
            "learning_rate": lr,
        })

    def log_final_results(self, test_acc: float, num_params: int,
                          num_errors: int):
        """Loggt finale Ergebnisse nach dem Training."""
        self.log({
            "final/test_accuracy": test_acc,
            "final/num_parameters": num_params,
            "final/num_errors": num_errors,
        })

    def finish(self):
        """Beendet den W&B-Run. Sicher bei mehrfachem Aufruf."""
        if self.run:
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
