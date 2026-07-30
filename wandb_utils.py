"""
W&B Experiment Tracking für Neuronales Netz von Grund auf
==========================================================
Integriert Weights & Biases in das selbstgebaute Neuronale Netz.
Loggt Trainingsmetriken, Modell-Architektur und Hyperparameter.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="neuronales-netz", config={...})
    tracker.log_epoch(epoch=1, train_loss=0.5, train_acc=0.92, test_acc=0.90)
    tracker.finish()
"""

import os
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für das selbstgebaute Neuronale Netz.

    Features:
    - Epochen-Metriken (Loss, Train-Acc, Test-Acc)
    - Modell-Architektur-Logging
    - Hyperparameter-Tracking
    - Trainingszeit-Messung
    """

    def __init__(
        self,
        project: str = "neuronales-netz-von-grund-auf",
        config: dict | None = None,
        tags: list | None = None,
        group: str | None = None,
        job_type: str = "train",
        notes: str | None = None,
        offline: bool = False,
    ):
        self.project = project
        self.run = None
        self._start_time = time.time()

        if WANDB_AVAILABLE:
            try:
                no_key = not os.environ.get("WANDB_API_KEY")
                mode = "offline" if offline or no_key else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["neural-network", "numpy", "from-scratch"],
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
                            stderr=subprocess.DEVNULL,
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int | None = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_acc: float,
        lr: float | None = None,
    ):
        """Loggt eine Trainings-Epoche."""
        metrics = {
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "test/accuracy": test_acc,
            "epoch": epoch,
        }
        if lr is not None:
            metrics["train/learning_rate"] = lr
        self.log(metrics, step=epoch)

    def log_final_results(self, test_acc: float, num_params: int, num_errors: int):
        """Loggt finale Evaluations-Ergebnisse."""
        self.log({
            "final/test_accuracy": test_acc,
            "final/num_parameters": num_params,
            "final/num_errors": num_errors,
        })

    def finish(self):
        """Beendet den W&B-Run."""
        elapsed = time.time() - self._start_time
        if self.run:
            self.log({"total_time_seconds": elapsed})
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
