"""Tests for base signal classes."""

import pandas as pd
import pytest
from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType


class TestSignalType:
    """Test SignalType enum."""

    def test_signal_type_values(self):
        """Test signal type enum values."""
        assert SignalType.ENTRY_LONG.value == "entry_long"
        assert SignalType.ENTRY_SHORT.value == "entry_short"
        assert SignalType.EXIT_LONG.value == "exit_long"
        assert SignalType.EXIT_SHORT.value == "exit_short"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.RISK_OFF.value == "risk_off"


class TestSignal:
    """Test Signal dataclass."""

    def test_signal_creation(self):
        """Test creating a signal."""
        timestamp = pd.Timestamp("2023-01-01")
        signal = Signal(
            symbol="BTC",
            signal_type=SignalType.ENTRY_LONG,
            confidence=0.8,
            timestamp=timestamp,
            source="Test",
        )

        assert signal.symbol == "BTC"
        assert signal.signal_type == SignalType.ENTRY_LONG
        assert signal.confidence == 0.8
        assert signal.timestamp == timestamp
        assert signal.source == "Test"
        assert signal.metadata == {}

    def test_signal_with_metadata(self):
        """Test creating signal with metadata."""
        signal = Signal(
            symbol="ETH",
            signal_type=SignalType.EXIT_LONG,
            confidence=0.9,
            timestamp=pd.Timestamp("2023-01-01"),
            source="Test",
            metadata={"reason": "profit_target", "pnl": 0.05},
        )

        assert signal.metadata["reason"] == "profit_target"
        assert signal.metadata["pnl"] == 0.05

    def test_signal_confidence_validation(self):
        """Test confidence validation."""
        with pytest.raises(ValueError):
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=1.5,  # > 1.0
                timestamp=pd.Timestamp("2023-01-01"),
            )

        with pytest.raises(ValueError):
            Signal(
                symbol="BTC",
                signal_type=SignalType.ENTRY_LONG,
                confidence=-0.1,  # < 0.0
                timestamp=pd.Timestamp("2023-01-01"),
            )


class TestSignalGenerator:
    """Test SignalGenerator abstract base class."""

    def test_generator_initialization(self):
        """Test generator initialization."""

        class ConcreteGenerator(SignalGenerator):
            def fit(self, data):
                pass

            def generate(self, data, current_position=None):
                return []

            def get_features(self, data):
                return data

        gen = ConcreteGenerator(name="TestGen", lookback_period=50)

        assert gen.name == "TestGen"
        assert gen.lookback_period == 50
        assert gen.is_fitted is False
        assert gen.feature_importance == {}
