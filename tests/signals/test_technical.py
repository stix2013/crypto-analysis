"""Tests for technical indicators and signal generators."""

import numpy as np
import pandas as pd
import pytest
from crypto_analysis.signals.technical import TechnicalPatternGenerator


class TestTechnicalPatternGenerator:
    """Tests for TechnicalPatternGenerator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with some patterns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="1h")
        price = 30000 + np.cumsum(np.random.randn(200) * 100)

        df = pd.DataFrame(
            {
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": np.random.uniform(100, 1000, 200) * price,
            },
            index=dates,
        )

        return df

    def test_generator_initialization(self):
        """Test generator initializes with default patterns."""
        gen = TechnicalPatternGenerator()
        assert gen.name == "Technical_Patterns"
        assert gen.lookback_period == 100

    def test_generator_custom_name(self):
        """Test generator with custom name."""
        gen = TechnicalPatternGenerator(name="Custom")
        assert gen.name == "Custom"

    def test_generate_with_insufficient_data(self, sample_data):
        """Test generate returns empty with insufficient data."""
        gen = TechnicalPatternGenerator()
        small_data = sample_data.iloc[:10]
        signals = gen.generate(small_data)
        assert signals == []

    def test_generate_with_sufficient_data(self, sample_data):
        """Test generate with sufficient data."""
        gen = TechnicalPatternGenerator()
        signals = gen.generate(sample_data)
        assert isinstance(signals, list)

    def test_generate_with_current_position(self, sample_data):
        """Test generate respects current position."""
        gen = TechnicalPatternGenerator()
        signals = gen.generate(sample_data, current_position=1.0)
        assert isinstance(signals, list)

    def test_generate_with_negative_position(self, sample_data):
        """Test generate with short position."""
        gen = TechnicalPatternGenerator()
        signals = gen.generate(sample_data, current_position=-1.0)
        assert isinstance(signals, list)

    def test_position_compatible_entry_long_with_long(self):
        """Test position compatibility check."""
        gen = TechnicalPatternGenerator()
        from crypto_analysis.signals.base import SignalType

        assert not gen._position_compatible(SignalType.ENTRY_LONG, 1.0)
        assert gen._position_compatible(SignalType.ENTRY_LONG, 0.0)
        assert gen._position_compatible(SignalType.ENTRY_LONG, None)

    def test_position_compatible_entry_short_with_short(self):
        """Test position compatibility for short entry."""
        gen = TechnicalPatternGenerator()
        from crypto_analysis.signals.base import SignalType

        assert not gen._position_compatible(SignalType.ENTRY_SHORT, -1.0)
        assert gen._position_compatible(SignalType.ENTRY_SHORT, 0.0)
        assert gen._position_compatible(SignalType.ENTRY_SHORT, None)

    def test_position_compatible_exit_long_without_position(self):
        """Test exit signal without position."""
        gen = TechnicalPatternGenerator()
        from crypto_analysis.signals.base import SignalType

        assert not gen._position_compatible(SignalType.EXIT_LONG, 0.0)
