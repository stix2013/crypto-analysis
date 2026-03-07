"""Tests for Online LSTM model."""

import numpy as np
import pytest

try:
    from crypto_analysis.online.models.online_lstm import OnlineLSTM

    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCH_AVAILABLE = False
    OnlineLSTM = None  # type: ignore[assignment,misc]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOnlineLSTM:
    """Tests for OnlineLSTM class."""

    def test_initialization(self):
        """Test OnlineLSTM initialization."""
        model = OnlineLSTM(
            sequence_length=10, n_features=5, units=[32, 16], learning_rate=0.01
        )

        assert model.name == "OnlineLSTM"
        assert model.sequence_length == 10
        assert model.n_features == 5
        assert model.units == [32, 16]
        assert model.lr == 0.01
        assert model.device.type == "cpu"
        assert model.model is not None
        assert model.optimizer is not None
        assert model.criterion is not None
        assert len(model.sequence_buffer) == 0
        assert model.sequence_buffer.maxlen == 100

    def test_partial_fit(self):
        """Test partial_fit updates weights and returns loss."""
        model = OnlineLSTM(sequence_length=10, n_features=5)

        X = np.random.randn(1, 10, 5)
        y = np.random.randn(1, 1)

        loss = model.partial_fit(X, y)

        assert isinstance(loss, float)
        assert model.update_count == 1
        assert len(model.sequence_buffer) == 1
        assert model.hidden_state is not None

    def test_partial_fit_multiple_batches(self):
        """Test partial_fit with multiple batches (different sizes)."""
        model = OnlineLSTM(sequence_length=10, n_features=5)

        # Batch size 1
        X1 = np.random.randn(1, 10, 5)
        y1 = np.random.randn(1, 1)
        model.partial_fit(X1, y1)
        assert model.hidden_state[0].size(1) == 1

        # Batch size 2
        X2 = np.random.randn(2, 10, 5)
        y2 = np.random.randn(2, 1)
        model.partial_fit(X2, y2)
        assert model.hidden_state[0].size(1) == 2

    def test_predict(self):
        """Test predict returns correct shape."""
        model = OnlineLSTM(sequence_length=10, n_features=5)

        X = np.random.randn(3, 10, 5)
        predictions = model.predict(X)

        assert predictions.shape == (3, 1)
        assert isinstance(predictions, np.ndarray)

    def test_set_learning_rate(self):
        """Test updating learning rate."""
        model = OnlineLSTM(sequence_length=10, n_features=5, learning_rate=0.01)
        model.set_learning_rate(0.05)

        assert model.lr == 0.05
        for param_group in model.optimizer.param_groups:
            assert param_group["lr"] == 0.05

    def test_reset_states(self):
        """Test resetting hidden states."""
        model = OnlineLSTM(sequence_length=10, n_features=5)

        X = np.random.randn(1, 10, 5)
        y = np.random.randn(1, 1)
        model.partial_fit(X, y)

        assert model.hidden_state is not None
        model.reset_states()
        assert model.hidden_state is None

    def test_buffer_reset_states(self):
        """Test reset_states is called after 100 updates."""
        model = OnlineLSTM(sequence_length=10, n_features=5)

        X = np.random.randn(1, 10, 5)
        y = np.random.randn(1, 1)

        for _ in range(100):
            model.partial_fit(X, y)

        # After 100th fit, reset_states() is called
        assert model.hidden_state is None
        assert model.update_count == 100
