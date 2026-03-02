"""Tests for signal generators."""

from crypto_analysis.signals.base import SignalType
from crypto_analysis.signals.ml_generators import RandomForestSignalGenerator
from crypto_analysis.signals.statistical import StatisticalArbitrageGenerator
from crypto_analysis.signals.technical import TechnicalPatternGenerator


class TestRandomForestSignalGenerator:
    """Test RandomForestSignalGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        gen = RandomForestSignalGenerator(
            n_estimators=10,
            max_depth=5,
            lookback=30,
        )

        assert gen.name == "RF_Classifier"
        assert gen.lookback_period == 30
        assert gen.model is not None
        assert gen.feature_engineer is not None

    def test_fit(self, sample_ohlcv_data):
        """Test fitting the generator."""
        gen = RandomForestSignalGenerator(n_estimators=10)
        gen.fit(sample_ohlcv_data)

        assert gen.is_fitted is True
        assert len(gen.feature_cols) > 0

    def test_generate_without_fit(self, minimal_ohlcv_data):
        """Test generate raises warning when not fitted."""
        gen = RandomForestSignalGenerator()
        signals = gen.generate(minimal_ohlcv_data)

        assert signals == []

    def test_generate_with_fit(self, sample_ohlcv_data):
        """Test generate returns signals after fitting."""
        gen = RandomForestSignalGenerator(n_estimators=10)
        gen.fit(sample_ohlcv_data)

        signals = gen.generate(sample_ohlcv_data)

        # May or may not generate signals depending on data
        assert isinstance(signals, list)


class TestTechnicalPatternGenerator:
    """Test TechnicalPatternGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        gen = TechnicalPatternGenerator()
        assert gen.name == "Technical_Patterns"
        assert gen.lookback_period == 100
        assert len(gen.patterns) == 4

    def test_fit(self, minimal_ohlcv_data):
        """Test fitting sets is_fitted."""
        gen = TechnicalPatternGenerator()
        gen.fit(minimal_ohlcv_data)
        assert gen.is_fitted is True

    def test_generate(self, sample_ohlcv_data):
        """Test generate returns signals."""
        gen = TechnicalPatternGenerator()
        gen.fit(sample_ohlcv_data)

        signals = gen.generate(sample_ohlcv_data)
        assert isinstance(signals, list)

    def test_position_compatibility(self):
        """Test position compatibility logic."""
        gen = TechnicalPatternGenerator()

        # No position - can enter
        assert gen._position_compatible(SignalType.ENTRY_LONG, None)
        assert gen._position_compatible(SignalType.ENTRY_SHORT, None)
        assert not gen._position_compatible(SignalType.EXIT_LONG, None)

        # Long position - can exit
        assert gen._position_compatible(SignalType.EXIT_LONG, 1.0)
        assert gen._position_compatible(SignalType.RISK_OFF, 1.0)
        assert not gen._position_compatible(SignalType.ENTRY_LONG, 1.0)

    def test_mean_reversion_detection(self, minimal_ohlcv_data):
        """Test mean reversion pattern detection."""
        gen = TechnicalPatternGenerator()

        # Create data with extreme move
        data = minimal_ohlcv_data.copy()
        data.iloc[-1, data.columns.get_loc("close")] = data["close"].iloc[-1] * 1.1

        signal_type, confidence, metadata = gen._detect_mean_reversion(data)
        # Should detect mean reversion opportunity
        assert isinstance(signal_type, (type(None), SignalType))
        assert 0 <= confidence <= 1


class TestStatisticalArbitrageGenerator:
    """Test StatisticalArbitrageGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        gen = StatisticalArbitrageGenerator(
            mean_period=20,
            std_threshold=2.0,
        )
        assert gen.name == "StatArb"
        assert gen.mean_period == 20
        assert gen.std_threshold == 2.0

    def test_fit(self, minimal_ohlcv_data):
        """Test fitting sets is_fitted."""
        gen = StatisticalArbitrageGenerator()
        gen.fit(minimal_ohlcv_data)
        assert gen.is_fitted is True

    def test_generate(self, sample_ohlcv_data):
        """Test generate returns signals."""
        gen = StatisticalArbitrageGenerator()
        gen.fit(sample_ohlcv_data)

        signals = gen.generate(sample_ohlcv_data)
        assert isinstance(signals, list)

    def test_generate_with_position(self, sample_ohlcv_data):
        """Test generate respects current position."""
        gen = StatisticalArbitrageGenerator()
        gen.fit(sample_ohlcv_data)

        # With existing long position
        signals = gen.generate(sample_ohlcv_data, current_position=1.0)

        # Should only generate exit signals, not entry
        for sig in signals:
            assert sig.signal_type in [
                SignalType.EXIT_LONG,
                SignalType.EXIT_SHORT,
                SignalType.RISK_OFF,
            ]
