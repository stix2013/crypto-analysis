"""Tests for online base classes."""

import numpy as np
import pytest
from datetime import datetime

from crypto_analysis.online.base import MarketRegime, OnlineModel


class DummyOnlineModel(OnlineModel):
    """Dummy implementation for testing."""

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.update_count += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


class TestMarketRegime:
    """Tests for MarketRegime dataclass."""

    def test_creation(self):
        """Test MarketRegime creation."""
        features = np.array([0.1, 0.2, 0.3])
        regime = MarketRegime(
            regime_id=1,
            name="trending_up",
            features=features,
            start_time=datetime.now(),
            confidence=0.8,
        )

        assert regime.regime_id == 1
        assert regime.name == "trending_up"
        assert regime.confidence == 0.8
        np.testing.assert_array_equal(regime.features, features)

    def test_default_values(self):
        """Test default values."""
        regime = MarketRegime(
            regime_id=0,
            name="volatile",
            features=np.zeros(5),
            start_time=datetime.now(),
        )

        assert regime.confidence == 0.0
        assert regime.statistics == {}


class TestOnlineModel:
    """Tests for OnlineModel base class."""

    def test_initialization(self):
        """Test OnlineModel initialization."""
        model = DummyOnlineModel(name="TestModel", learning_rate=0.01)

        assert model.name == "TestModel"
        assert model.lr == 0.01
        assert model.is_initialized is False
        assert model.update_count == 0

    def test_update_performance(self):
        """Test performance tracking."""
        model = DummyOnlineModel(name="Test")

        predictions = np.array([1, -1, 1])
        actuals = np.array([1, 1, -1])

        model.update_performance(predictions, actuals)

        assert len(model.performance_history) == 1
        record = model.performance_history[0]
        assert "timestamp" in record
        assert "accuracy" in record
        assert record["accuracy"] == pytest.approx(2 / 3)

    def test_performance_history_limit(self):
        """Test performance history max length."""
        model = DummyOnlineModel(name="Test")

        for _ in range(1500):
            model.update_performance(np.array([1]), np.array([1]))

        assert len(model.performance_history) == 1000
