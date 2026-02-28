"""Online Neural Network with Elastic Weight Consolidation."""


import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore

from crypto_analysis.online.base import OnlineModel


class OnlineNeuralNetwork(OnlineModel):
    """PyTorch-based online neural network with EWC.

    Uses Elastic Weight Consolidation to prevent catastrophic
    forgetting when learning new patterns (market regimes).

    Note:
        Requires PyTorch to be installed. Import will fail gracefully
        if PyTorch is not available.

    Attributes:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        lr: Learning rate
        ewc_lambda: EWC regularization strength
        device: Computation device (cuda/cpu)
        model: PyTorch neural network
        fisher_matrix: Fisher information for EWC
        optimal_params: Stored optimal parameters
        task_count: Number of learning tasks completed
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        lr: float = 0.001,
        ewc_lambda: float = 100,
    ) -> None:
        """Initialize Online Neural Network.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer sizes (default: [128, 64])
            lr: Learning rate
            ewc_lambda: Elastic Weight Consolidation strength
        """
        super().__init__(name="OnlineNeuralNetwork", learning_rate=lr)

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for OnlineNeuralNetwork. Install with: pip install torch"
            )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_network()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.fisher_matrix: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}
        self.task_count = 0

    def _build_network(self) -> nn.Sequential:
        """Build the neural network architecture."""
        layers: list[nn.Module] = []
        prev_dim = self.input_dim

        for h_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers).to(self.device)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, compute_fisher: bool = False) -> float:
        """Train with EWC regularization.

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            compute_fisher: Whether to update Fisher information

        Returns:
            Training loss value
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device).reshape(-1, 1)

        self.optimizer.zero_grad()
        output = self.model(X_tensor)
        loss = self.criterion(output, y_tensor)

        if self.task_count > 0 and self.fisher_matrix:
            ewc_loss = self._compute_ewc_loss()
            loss += self.ewc_lambda * ewc_loss

        loss.backward()
        self.optimizer.step()

        if compute_fisher:
            self._update_fisher(X_tensor, y_tensor)

        self.update_count += 1
        return loss.item()

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute elastic weight consolidation penalty."""
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        return loss

    def _update_fisher(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Update Fisher information matrix for EWC.

        Args:
            X: Input features
            y: Target values
        """
        self.model.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.fisher_matrix:
                    self.fisher_matrix[name] = torch.zeros_like(param)
                self.fisher_matrix[name] += param.grad.data**2

        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

        self.task_count += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(X_tensor)
        return output.cpu().numpy().flatten()
