"""Extended tests for machine learning signal generators."""

import numpy as np
import pandas as pd
import pytest
from crypto_analysis.signals.base import SignalType
from crypto_analysis.signals.ml_generators import (
    LSTMSignalGenerator,
    RandomForestSignalGenerator,
    TF_AVAILABLE,
)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestLSTMSignalGenerator:
    """Test LSTMSignalGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        gen = LSTMSignalGenerator(
            sequence_length=10,
            n_features=5,
            lstm_units=[32, 16],
            dropout=0.1
        )

        assert gen.name == "LSTM_Predictor"
        assert gen.sequence_length == 10
        assert gen.n_features == 5
        assert gen.lstm_units == [32, 16]
        assert gen.dropout == 0.1
        assert gen.model is None

    def test_build_model(self):
        """Test building the Keras model."""
        gen = LSTMSignalGenerator(sequence_length=10, n_features=5, lstm_units=[16])
        gen.build_model()

        assert gen.model is not None
        assert len(gen.model.inputs) == 1
        assert len(gen.model.outputs) == 2  # direction and volatility
        assert gen.model.input_shape == (None, 10, 5)

    def test_fit_insufficient_data(self, minimal_ohlcv_data):
        """Test fit with insufficient data raises error."""
        gen = LSTMSignalGenerator(sequence_length=120)  # minimal_ohlcv_data has 100 rows
        with pytest.raises(ValueError, match="Not enough data for sequences"):
            gen.fit(minimal_ohlcv_data)

    def test_fit_and_generate(self, sample_ohlcv_data):
        """Test full fit and generate cycle."""
        # Using small parameters for faster test
        gen = LSTMSignalGenerator(
            sequence_length=10,
            lstm_units=[8],
            dropout=0.0
        )

        # Fit with very few epochs for speed
        gen.fit(sample_ohlcv_data, epochs=1, batch_size=32)

        assert gen.is_fitted is True
        assert gen.vol_median_ > 0

        # Generate signals
        signals = gen.generate(sample_ohlcv_data)
        assert isinstance(signals, list)

        # Test generate with positions
        signals_long = gen.generate(sample_ohlcv_data, current_position=1.0)
        for sig in signals_long:
            # Should not generate long entries if already long
            assert sig.signal_type != SignalType.ENTRY_LONG

        signals_short = gen.generate(sample_ohlcv_data, current_position=-1.0)
        for sig in signals_short:
            # Should not generate short entries if already short
            assert sig.signal_type != SignalType.ENTRY_SHORT

    def test_generate_unfitted(self, sample_ohlcv_data):
        """Test generating without fitting."""
        gen = LSTMSignalGenerator()
        with pytest.warns(UserWarning, match="Model not fitted yet"):
            signals = gen.generate(sample_ohlcv_data)
        assert signals == []


class TestRandomForestSignalGeneratorExtended:
    """Extended tests for RandomForestSignalGenerator."""

    def test_fit_with_val_split(self, sample_ohlcv_data):
        """Test fitting with validation split explicitly."""
        gen = RandomForestSignalGenerator(n_estimators=5)
        gen.fit(sample_ohlcv_data, val_ratio=0.3)

        assert gen.is_fitted is True
        assert hasattr(gen, 'feature_importance')
        assert len(gen.feature_importance) == len(gen.feature_cols)

    def test_get_features(self, sample_ohlcv_data):
        """Test get_features method."""
        gen = RandomForestSignalGenerator(n_estimators=5)
        gen.fit(sample_ohlcv_data)

        features = gen.get_features(sample_ohlcv_data)
        assert isinstance(features, pd.DataFrame)
        assert all(col in features.columns for col in gen.feature_cols)

    def test_generate_edge_cases(self, sample_ohlcv_data):
        """Test generation with different position states."""
        gen = RandomForestSignalGenerator(n_estimators=10)
        gen.fit(sample_ohlcv_data)

        # We can't guarantee a signal will be generated with random data,
        # but we can verify the logic if we mock the model output.

        # Mock predict_proba to return high confidence "UP"
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.1, 0.9]]) # 10% down, 90% up

        original_model = gen.model
        gen.model = MockModel()

        # Case 1: No position -> Should get ENTRY_LONG
        signals = gen.generate(sample_ohlcv_data, current_position=None)
        assert any(s.signal_type == SignalType.ENTRY_LONG for s in signals)

        # Case 2: Already long -> Should NOT get ENTRY_LONG
        signals = gen.generate(sample_ohlcv_data, current_position=1.0)
        assert not any(s.signal_type == SignalType.ENTRY_LONG for s in signals)

        # Mock predict_proba to return high confidence "DOWN"
        class MockModelDown:
            def predict_proba(self, X):
                return np.array([[0.9, 0.1]]) # 90% down, 10% up

        gen.model = MockModelDown()

        # Case 3: No position -> Should get ENTRY_SHORT
        signals = gen.generate(sample_ohlcv_data, current_position=0.0)
        assert any(s.signal_type == SignalType.ENTRY_SHORT for s in signals)

        # Case 4: Already short -> Should NOT get ENTRY_SHORT
        signals = gen.generate(sample_ohlcv_data, current_position=-1.0)
        assert not any(s.signal_type == SignalType.ENTRY_SHORT for s in signals)

        # Restore model
        gen.model = original_model
