"""Tests for Online Neural Network with EWC."""

import numpy as np
import pytest

try:
    import torch
    from crypto_analysis.online.models.online_nn import OnlineNeuralNetwork

    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCH_AVAILABLE = False
    OnlineNeuralNetwork = None  # type: ignore[assignment,misc]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOnlineNeuralNetwork:
    """Tests for OnlineNeuralNetwork class."""

    def test_initialization(self):
        """Test initialization with default and custom params."""
        model = OnlineNeuralNetwork(
            input_dim=10, hidden_dims=[32, 16], lr=0.01, ewc_lambda=50
        )

        assert model.name == "OnlineNeuralNetwork"
        assert model.input_dim == 10
        assert model.hidden_dims == [32, 16]
        assert model.lr == 0.01
        assert model.ewc_lambda == 50
        assert model.device.type == "cpu"
        assert model.model is not None
        assert model.optimizer is not None
        assert model.criterion is not None
        assert model.task_count == 0
        assert model.fisher_matrix == {}
        assert model.optimal_params == {}

    def test_partial_fit(self):
        """Test partial_fit updates weights and returns loss."""
        model = OnlineNeuralNetwork(input_dim=10)

        X = np.random.randn(5, 10)
        y = np.random.randn(5)

        loss = model.partial_fit(X, y)

        assert isinstance(loss, float)
        assert model.update_count == 1
        assert model.task_count == 0

    def test_partial_fit_with_fisher(self):
        """Test partial_fit updates Fisher matrix."""
        model = OnlineNeuralNetwork(input_dim=10)

        X = np.random.randn(5, 10)
        y = np.random.randn(5)

        model.partial_fit(X, y, compute_fisher=True)

        assert model.task_count == 1
        assert len(model.fisher_matrix) > 0
        assert len(model.optimal_params) > 0

    def test_ewc_loss_calculation(self):
        """Test EWC loss is applied after first task."""
        model = OnlineNeuralNetwork(input_dim=10, ewc_lambda=100.0)

        X = np.random.randn(5, 10)
        y = np.random.randn(5)

        # First task: No EWC loss
        model.partial_fit(X, y, compute_fisher=True)

        # Second task: EWC loss should be applied
        # We expect loss to potentially increase due to regularization
        loss2 = model.partial_fit(X, y)

        assert model.task_count == 1
        assert isinstance(loss2, float)

    def test_predict(self):
        """Test predict returns correct shape."""
        model = OnlineNeuralNetwork(input_dim=10)

        X = np.random.randn(3, 10)
        predictions = model.predict(X)

        assert predictions.shape == (3,)
        assert isinstance(predictions, np.ndarray)

    def test_set_learning_rate(self):
        """Test updating learning rate."""
        model = OnlineNeuralNetwork(input_dim=10, lr=0.01)
        model.set_learning_rate(0.05)

        assert model.lr == 0.05
        for param_group in model.optimizer.param_groups:
            assert param_group["lr"] == 0.05

    def test_build_network(self):
        """Test internal network builder architecture."""
        model = OnlineNeuralNetwork(input_dim=5, hidden_dims=[10])
        # [Linear(5,10), ReLU, Dropout, Linear(10,1)]
        # Total modules in Sequential: 4
        assert len(model.model) == 4
        assert isinstance(model.model[0], torch.nn.Linear)
        assert model.model[0].in_features == 5
        assert model.model[0].out_features == 10
        assert isinstance(model.model[3], torch.nn.Linear)
        assert model.model[3].in_features == 10
        assert model.model[3].out_features == 1
