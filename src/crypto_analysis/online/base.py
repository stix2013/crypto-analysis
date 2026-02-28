"""Base classes for online learning components."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class MarketRegime:
    """Represents current market state.

    Attributes:
        regime_id: Numeric identifier for the regime
        name: Human-readable regime name
        features: Extracted features used for classification
        start_time: When this regime was first detected
        confidence: Confidence score (0.0 to 1.0)
        statistics: Additional regime statistics
    """

    regime_id: int
    name: str
    features: np.ndarray
    start_time: datetime
    confidence: float = 0.0
    statistics: dict[str, Any] = field(default_factory=dict)


class OnlineModel(ABC):
    """Base class for online learning models.

    Provides common interface for models that can be updated
    incrementally with new data samples.
    """

    def __init__(self, name: str, learning_rate: float = 0.01) -> None:
        """Initialize online model.

        Args:
            name: Model identifier
            learning_rate: Learning rate for updates
        """
        self.name = name
        self.lr = learning_rate
        self.is_initialized = False
        self.update_count = 0
        self.performance_history: deque[dict[str, Any]] = deque(maxlen=1000)

    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Update model with single or batch of samples.

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Training loss or None
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)
        """

    def update_performance(self, prediction: np.ndarray, actual: np.ndarray) -> None:
        """Track prediction accuracy for adaptive learning.

        Args:
            prediction: Model predictions
            actual: Actual target values
        """
        accuracy = np.mean((np.sign(prediction) == np.sign(actual)).astype(float))
        self.performance_history.append(
            {
                "timestamp": datetime.now(),
                "accuracy": accuracy,
                "prediction": float(prediction.mean()),
                "actual": float(actual.mean()),
            }
        )
