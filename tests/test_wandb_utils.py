"""
Tests für wandb_utils.py — W&B Experiment Tracking für Neuronales Netz von Grund auf.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import WandBTracker, WANDB_AVAILABLE


class TestWandBTracker:
    """Tests für WandBTracker (Neuronales Netz von Grund auf)."""

    def test_initialization_offline(self):
        """Tracker sollte im Offline-Modus initialisieren."""
        tracker = WandBTracker(
            project="test-neuronales-netz",
            config={"hidden_size": 128, "learning_rate": 0.01, "epochs": 10},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert tracker.is_active
            assert tracker.run is not None
        else:
            assert not tracker.is_active
        tracker.finish()

    def test_log_epoch(self):
        """Epochen-Metriken sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-neuronales-netz", offline=True)
        if tracker.is_active:
            tracker.log_epoch(epoch=1, train_loss=0.5, train_acc=0.92, test_acc=0.90, lr=0.01)
        tracker.finish()

    def test_log_final_results(self):
        """Finale Ergebnisse sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-neuronales-netz", offline=True)
        if tracker.is_active:
            tracker.log_final_results(test_acc=0.95, num_params=10000, num_errors=50)
        tracker.finish()

    def test_log_metrics(self):
        """Allgemeine Metriken sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-neuronales-netz", offline=True)
        if tracker.is_active:
            tracker.log({"custom_metric": 0.95})
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() sollte den Run beenden und doppeltes finish() sollte safe sein."""
        tracker = WandBTracker(project="test-neuronales-netz", offline=True)
        tracker.finish()
        tracker.finish()
        assert not tracker.is_active

    def test_multiple_epochs(self):
        """Mehrere Epochen sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-neuronales-netz", offline=True)
        if tracker.is_active:
            for epoch in range(10):
                tracker.log_epoch(epoch=epoch, train_loss=1.0 - epoch * 0.08,
                                 train_acc=0.5 + epoch * 0.05,
                                 test_acc=0.5 + epoch * 0.04,
                                 lr=0.01 * (0.95 ** epoch))
        tracker.finish()
