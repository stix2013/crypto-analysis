"""Online LSTM with truncated backpropagation through time."""

from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from crypto_analysis.online.base import OnlineModel

if TORCH_AVAILABLE:

    class LSTMPyTorch(nn.Module):
        """PyTorch LSTM for online learning."""

        def __init__(
            self, input_size: int, hidden_size: int, num_layers: int = 2
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out

    class OnlineLSTM(OnlineModel):
        """LSTM with online learning using truncated backpropagation.

        Maintains hidden state and updates on recent sequences. Uses
        stateful LSTM for efficient mini-batch updates.

        Note:
            Requires PyTorch to be installed. Import will fail gracefully
            if PyTorch is not available.

        Attributes:
            sequence_length: Number of time steps in input sequences
            n_features: Number of input features
            units: List of LSTM layer sizes
            lr: Learning rate
            model: PyTorch LSTM model
            sequence_buffer: Buffer for recent sequences
        """

        def __init__(
            self,
            sequence_length: int = 60,
            n_features: int = 50,
            units: list[int] | None = None,
            learning_rate: float = 0.001,
        ) -> None:
            """Initialize Online LSTM.

            Args:
                sequence_length: Number of time steps in sequences
                n_features: Number of input features
                units: LSTM layer sizes (default: [64, 32])
                learning_rate: Learning rate for optimization
            """
            super().__init__(name="OnlineLSTM", learning_rate=learning_rate)

            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is required for OnlineLSTM. Install with: pip install torch"
                )

            self.sequence_length = sequence_length
            self.n_features = n_features
            self.units = units or [64, 32]
            self.lr = learning_rate
            self.device = torch.device("cpu")

            self.model = LSTMPyTorch(
                n_features, self.units[-1], num_layers=len(self.units)
            )
            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()

            self.sequence_buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(
                maxlen=sequence_length * 10
            )

            self.hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

        def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float | None:
            """Update model with new sequence using TBPTT.

            Args:
                X: Input sequence of shape (1, sequence_length, n_features)
                y: Target of shape (1, 1)

            Returns:
                Training loss
            """
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device).reshape(-1, 1)

            self.sequence_buffer.append((X, y))

            self.model.train()
            self.optimizer.zero_grad()

            # Reset or detach hidden state
            # If batch size changes (e.g., last batch in fit()), we must reset
            if self.hidden_state is not None:
                if self.hidden_state[0].size(1) != X_tensor.size(0):
                    self.hidden_state = None
                else:
                    self.hidden_state = tuple(h.detach() for h in self.hidden_state)

            lstm_out, self.hidden_state = self.model.lstm(
                X_tensor, self.hidden_state
            )

            output = self.model.fc(lstm_out[:, -1, :])
            loss = self.criterion(output, y_tensor)

            loss.backward()
            self.optimizer.step()

            if len(self.sequence_buffer) % 100 == 0:
                self.reset_states()

            self.update_count += 1
            return float(loss.item())

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions on input sequences.

            Args:
                X: Input sequences of shape (n_samples, sequence_length, n_features)

            Returns:
                Predictions of shape (n_samples, 1)
            """
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)

                lstm_out, _ = self.model.lstm(X_tensor)
                output = self.model.fc(lstm_out[:, -1, :])

            return np.asarray(output.cpu().numpy())

        def set_learning_rate(self, lr: float) -> None:
            """Update the optimizer learning rate.

            Args:
                lr: New learning rate value
            """
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.lr = lr

        def reset_states(self) -> None:
            """Reset LSTM hidden states."""
            self.hidden_state = None

else:
    OnlineLSTM = None  # type: ignore[assignment,misc]
