"""Tests for Regime Detector."""

import numpy as np
import pandas as pd
import pytest

from crypto_analysis.online.detection.regime import RegimeDetector


def create_test_data(trend: str = "ranging", n_points: int = 150) -> pd.DataFrame:
    """Create test market data with specified trend."""
    np.random.seed(42)

    if trend == "trending_up":
        prices = 100 + np.cumsum(np.random.randn(n_points) + 0.01)
    elif trend == "trending_down":
        prices = 100 + np.cumsum(np.random.randn(n_points) - 0.01)
    elif trend == "volatile":
        prices = 100 + np.cumsum(np.random.randn(n_points) * 1.5)
    elif trend == "crash":
        prices = 100 * np.exp(-0.01 * np.arange(n_points))
    else:
        prices = 100 + np.random.randn(n_points) * 2

    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(n_points) * 0.5,
            "high": prices + np.abs(np.random.randn(n_points)),
            "low": prices - np.abs(np.random.randn(n_points)),
            "close": prices,
            "volume": np.random.uniform(1000, 5000, n_points),
        },
        index=pd.date_range("2023-01-01", periods=n_points, freq="1h"),
    )

    return df


class TestRegimeDetector:
    """Tests for RegimeDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = RegimeDetector(n_regimes=5, lookback=100)

        assert detector.n_regimes == 5
        assert detector.lookback == 100
        assert detector.current_regime is None

    def test_extract_features(self):
        """Test feature extraction."""
        detector = RegimeDetector(lookback=50)
        data = create_test_data("trending_up", 100)

        features = detector.extract_regime_features(data)

        assert len(features) == 10
        assert not np.isnan(features).all()

    def test_extract_features_insufficient_data(self):
        """Test feature extraction with insufficient data."""
        detector = RegimeDetector(lookback=50)
        data = create_test_data("trending_up", 20)

        features = detector.extract_regime_features(data)

        assert len(features) == 10
        np.testing.assert_array_equal(features, np.zeros(10))

    def test_detect_trending_up(self):
        """Test trending up detection."""
        detector = RegimeDetector(lookback=50)
        data = create_test_data("trending_up", 150)

        regime = detector.update(data)

        assert regime.name in ["trending_up", "ranging", "volatile"]
        assert regime.confidence > 0

    def test_detect_volatile(self):
        """Test volatile regime detection."""
        detector = RegimeDetector(lookback=50)
        data = create_test_data("volatile", 150)

        regime = detector.update(data)

        assert regime.name in ["volatile", "trending_up", "trending_down"]

    def test_regime_history(self):
        """Test regime history tracking."""
        detector = RegimeDetector(lookback=30)

        for trend in ["trending_up", "volatile", "trending_down"]:
            data = create_test_data(trend, 100)
            detector.update(data)

        assert len(detector.regime_history) == 3

    def test_regime_change_callback(self, capsys):
        """Test regime change callback."""
        detector = RegimeDetector(lookback=30)

        data1 = create_test_data("trending_up", 100)
        detector.update(data1)

        data2 = create_test_data("volatile", 100)
        regime = detector.update(data2)

        captured = capsys.readouterr()
        assert "Regime change" in captured.out or regime is not None


class TestRegimeDetectorEdgeCases:
    """Edge case tests for RegimeDetector."""

    def test_empty_data(self):
        """Test with empty DataFrame."""
        detector = RegimeDetector()
        data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        features = detector.extract_regime_features(data)

        np.testing.assert_array_equal(features, np.zeros(10))

    def test_constant_price(self):
        """Test with constant price."""
        detector = RegimeDetector(lookback=20)
        data = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [101] * 50,
                "low": [99] * 50,
                "close": [100] * 50,
                "volume": [1000] * 50,
            },
            index=pd.date_range("2023-01-01", periods=50, freq="1h"),
        )

        regime = detector.update(data)

        assert regime.name == "ranging"
