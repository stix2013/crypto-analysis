"""Tests for Online Signal Generator."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from crypto_analysis.online.generator import OnlineSignalGenerator


def create_test_data(n_points: int = 300) -> pd.DataFrame:
    """Create test OHLCV data."""
    np.random.seed(42)

    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
    volumes = np.random.uniform(1000, 5000, n_points)

    return pd.DataFrame(
        {
            "open": prices + np.random.randn(n_points) * 0.2,
            "high": prices + np.abs(np.random.randn(n_points)),
            "low": prices - np.abs(np.random.randn(n_points)),
            "close": prices,
            "volume": volumes,
        },
        index=pd.date_range("2023-01-01", periods=n_points, freq="1h"),
    )


class TestOnlineSignalGenerator:
    """Tests for OnlineSignalGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = OnlineSignalGenerator(
            name="TestGenerator", sequence_length=30, update_frequency=5
        )

        assert generator.name == "TestGenerator"
        assert generator.sequence_length == 30
        assert generator.update_frequency == 5
        assert generator.is_fitted is False

    def test_get_features(self):
        """Test feature extraction."""
        generator = OnlineSignalGenerator(sequence_length=30)
        data = create_test_data(500)

        features = generator.get_features(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_generate_before_fit(self):
        """Test generate returns empty before fitting."""
        generator = OnlineSignalGenerator(sequence_length=30)
        data = create_test_data(200)

        signals = generator.generate(data)

        assert signals == []

    def test_generate_insufficient_data(self):
        """Test generate with insufficient data."""
        generator = OnlineSignalGenerator(sequence_length=60)
        data = create_test_data(50)

        signals = generator.generate(data)

        assert signals == []

    @patch("crypto_analysis.online.generator.TF_AVAILABLE", False)
    @patch("crypto_analysis.online.generator.TORCH_AVAILABLE", False)
    def test_generate_without_ml_libs(self):
        """Test generate works without TensorFlow/PyTorch."""
        generator = OnlineSignalGenerator(sequence_length=30)
        data = create_test_data(500)

        generator.fit(data)

        assert generator.is_fitted
        assert generator.lstm is None
        assert generator.nn is None

    def test_regime_threshold_mapping(self):
        """Test regime-specific thresholds."""
        from crypto_analysis.online.base import MarketRegime

        generator = OnlineSignalGenerator()

        thresholds = {
            "trending_up": generator._get_regime_threshold(
                MarketRegime(0, "trending_up", np.zeros(10), pd.Timestamp.now())
            ),
            "trending_down": generator._get_regime_threshold(
                MarketRegime(0, "trending_down", np.zeros(10), pd.Timestamp.now())
            ),
            "ranging": generator._get_regime_threshold(
                MarketRegime(0, "ranging", np.zeros(10), pd.Timestamp.now())
            ),
            "volatile": generator._get_regime_threshold(
                MarketRegime(0, "volatile", np.zeros(10), pd.Timestamp.now())
            ),
            "crash": generator._get_regime_threshold(
                MarketRegime(0, "crash", np.zeros(10), pd.Timestamp.now())
            ),
        }

        assert thresholds["trending_up"] < thresholds["ranging"]
        assert thresholds["volatile"] < thresholds["crash"]
        assert thresholds["crash"] == 0.4

    def test_model_weights_initialization(self):
        """Test initial model weights."""
        generator = OnlineSignalGenerator()

        assert generator.model_weights == {
            "lstm": 0.25,
            "nn": 0.25,
            "rf": 0.25,
            "pa": 0.25,
        }

    def test_error_buffer_tracking(self):
        """Test error buffer is used for weight updates."""
        generator = OnlineSignalGenerator()

        for _ in range(25):
            generator.error_buffer.append(0.3)

        generator._update_model_weights()

        assert generator.model_weights == {
            "lstm": 0.25,
            "nn": 0.25,
            "rf": 0.25,
            "pa": 0.25,
        }

    def test_error_buffer_preserves_weights(self):
        """Test weights preserved with good performance."""
        generator = OnlineSignalGenerator()

        for _ in range(25):
            generator.error_buffer.append(0.2)

        original_weights = generator.model_weights.copy()
        generator._update_model_weights()

        assert generator.model_weights == original_weights


class TestOnlineSignalGeneratorIntegration:
    """Integration tests for OnlineSignalGenerator."""

    @pytest.fixture
    def trained_generator(self):
        """Create a trained generator."""
        generator = OnlineSignalGenerator(sequence_length=20)
        data = create_test_data(300)
        generator.fit(data)
        return generator

    def test_fit_sets_initialized(self, trained_generator):
        """Test fit sets is_initialized."""
        assert trained_generator.is_fitted is True

    def test_fit_creates_scaler(self, trained_generator):
        """Test fit creates scaler."""
        assert trained_generator.scaler is not None
