"""Online LSTM with truncated backpropagation through time."""

from collections import deque

import numpy as np

try:
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from crypto_analysis.online.base import OnlineModel


class OnlineLSTM(OnlineModel):
    """LSTM with online learning using truncated backpropagation.

    Maintains hidden state and updates on recent sequences. Uses
    stateful LSTM for efficient mini-batch updates.

    Note:
        Requires TensorFlow to be installed. Import will fail gracefully
        if TensorFlow is not available.

    Attributes:
        sequence_length: Number of time steps in input sequences
        n_features: Number of input features
        units: List of LSTM layer sizes
        lr: Learning rate
        model: Compiled Keras model
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

        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for OnlineLSTM. Install with: pip install tensorflow"
            )

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units or [64, 32]
        self.lr = learning_rate
        self.model: Model | None = None
        self.sequence_buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=sequence_length * 10
        )
        self.build_model()

    def build_model(self) -> None:
        """Build LSTM with stateful capabilities."""
        inputs = Input(batch_shape=(1, self.sequence_length, self.n_features))
        x = inputs

        for units in self.units:
            x = LSTM(units, stateful=True, return_sequences=True)(x)
            x = Dropout(0.2)(x)

        x = LSTM(self.units[-1], stateful=True)(x)
        outputs = Dense(1, activation="tanh")(x)

        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer="adam",  # Will be replaced with custom lr
            loss="mse",
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float | None:
        """Update model with new sequence using TBPTT.

        Args:
            X: Input sequence of shape (1, sequence_length, n_features)
            y: Target of shape (1, 1)

        Returns:
            Training loss
        """
        if self.model is None:
            return None

        self.sequence_buffer.append((X, y))

        if len(self.sequence_buffer) % 100 == 0:
            self.model.reset_states()

        loss = self.model.train_on_batch(X, y)

        if len(self.sequence_buffer) >= 32:
            recent_batch = list(self.sequence_buffer)[-32:]
            X_batch = np.concatenate([x for x, _ in recent_batch])
            y_batch = np.concatenate([y for _, y in recent_batch])
            self.model.train_on_batch(X_batch, y_batch)

        self.update_count += 1
        return float(loss) if isinstance(loss, (float, int)) else None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input sequences.

        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)

        Returns:
            Predictions of shape (n_samples, 1)
        """
        if self.model is None:
            return np.zeros((len(X), 1))
        return self.model.predict(X, verbose=0)

    def set_learning_rate(self, lr: float) -> None:
        """Update the optimizer learning rate.

        Args:
            lr: New learning rate value
        """
        if self.model is not None:
            import tensorflow.keras.backend as K

            K.set_value(self.model.optimizer.learning_rate, lr)
        self.lr = lr

    def reset_states(self) -> None:
        """Reset LSTM hidden states."""
        if self.model is not None:
            self.model.reset_states()
