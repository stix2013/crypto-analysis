"""Continuous learning pipeline for model management."""

import pickle
import random
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_analysis.online.generator import OnlineSignalGenerator


class ContinuousLearningPipeline:
    """Manages continuous training and deployment of models.

    Handles data streaming, model versioning, and A/B testing
    for continuous model improvement in production.

    Attributes:
        checkpoint_dir: Directory for model checkpoints
        active_model: Currently deployed model
        candidate_model: Model being tested for promotion
        data_buffer: Buffer for incoming streaming data
        model_performance: Performance tracking for A/B testing
        ab_test_active: Whether A/B test is running
        ab_split_ratio: Traffic fraction for candidate model
    """

    def __init__(self, checkpoint_dir: str = "./models") -> None:
        """Initialize continuous learning pipeline.

        Args:
            checkpoint_dir: Directory for saving model checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.active_model: OnlineSignalGenerator | None = None
        self.candidate_model: OnlineSignalGenerator | None = None
        self.data_buffer: deque[pd.DataFrame] = deque(maxlen=10000)

        self.model_performance: dict[str, dict[str, list[float]]] = {}
        self.ab_test_active = False
        self.ab_split_ratio = 0.1

    def stream_data(self, new_data: pd.DataFrame) -> None:
        """Add new market data to buffer.

        Args:
            new_data: New market data to process
        """
        self.data_buffer.append(new_data)

        if len(self.data_buffer) >= 1000 and len(self.data_buffer) % 500 == 0:
            self._trigger_retraining()

    def _trigger_retraining(self) -> None:
        """Start retraining process in background."""
        print("[Pipeline] Triggering model retraining...")

        training_data = pd.concat(list(self.data_buffer), ignore_index=True)

        self.candidate_model = OnlineSignalGenerator(name="Candidate")
        self.candidate_model.fit(training_data)

        self.ab_test_active = True
        print("[Pipeline] A/B test started")

    def get_prediction(self, data: pd.DataFrame) -> tuple[list[Any], str]:
        """Get prediction from appropriate model.

        Args:
            data: Market data for prediction

        Returns:
            Tuple of (signals, model_source)
        """
        if not self.ab_test_active or self.active_model is None:
            if self.active_model is None:
                self.active_model = OnlineSignalGenerator()
                if len(self.data_buffer) > 0:
                    combined = pd.concat(list(self.data_buffer))
                    if len(combined) >= 500:
                        self.active_model.fit(combined)

            if self.active_model is None or not self.active_model.is_fitted:
                return [], "none"

            signals = self.active_model.generate(data)
            return signals, "active"

        if random.random() < self.ab_split_ratio:
            signals = self.candidate_model.generate(data)
            return signals, "candidate"
        else:
            signals = self.active_model.generate(data)
            return signals, "active"

    def update_performance(self, model_type: str, prediction: float, actual_return: float) -> None:
        """Track model performance for comparison.

        Args:
            model_type: "active" or "candidate"
            prediction: Model prediction
            actual_return: Actual return observed
        """
        if model_type not in self.model_performance:
            self.model_performance[model_type] = {"predictions": [], "returns": []}

        self.model_performance[model_type]["predictions"].append(prediction)
        self.model_performance[model_type]["returns"].append(actual_return)

        if self.ab_test_active and len(self.model_performance["candidate"]["returns"]) > 100:
            self._evaluate_ab_test()

    def _evaluate_ab_test(self) -> None:
        """Determine if candidate model should be promoted."""
        active_returns = np.array(self.model_performance["active"]["returns"][-100:])
        candidate_returns = np.array(self.model_performance["candidate"]["returns"][-100:])

        active_sharpe = active_returns.mean() / (active_returns.std() + 1e-10)
        candidate_sharpe = candidate_returns.mean() / (candidate_returns.std() + 1e-10)

        if candidate_sharpe > active_sharpe * 1.1:
            print(
                f"[Pipeline] Promoting candidate model! "
                f"Sharpe: {active_sharpe:.2f} -> {candidate_sharpe:.2f}"
            )
            self._promote_candidate()

    def _promote_candidate(self) -> None:
        """Promote candidate to active model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.active_model is not None:
            self._save_model(self.active_model, f"model_{timestamp}_retired.pkl")

        self.active_model = self.candidate_model
        self.candidate_model = None
        self.ab_test_active = False

        self.model_performance.clear()

        print("[Pipeline] Model promotion complete")

    def _save_model(self, model: OnlineSignalGenerator, filename: str) -> None:
        """Save model to disk.

        Args:
            model: Model to save
            filename: Output filename
        """
        path = self.checkpoint_dir / filename
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"[Pipeline] Model saved to {path}")

    def load_model(self, filename: str) -> None:
        """Load model from disk.

        Args:
            filename: Model file to load
        """
        path = self.checkpoint_dir / filename
        with open(path, "rb") as f:
            self.active_model = pickle.load(f)
        print(f"[Pipeline] Model loaded from {path}")
