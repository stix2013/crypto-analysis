"""Online learning models."""

from crypto_analysis.online.models.online_nn import OnlineNeuralNetwork
from crypto_analysis.online.models.online_rf import OnlineRandomForest

LSTM_AVAILABLE = False
try:
    from crypto_analysis.online.models.online_lstm import OnlineLSTM  # noqa: F401

    LSTM_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "OnlineRandomForest",
    "OnlineNeuralNetwork",
]

if LSTM_AVAILABLE:
    __all__.append("OnlineLSTM")
