"""Online signal generator with real-time adaptation."""

from collections import deque
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler

from crypto_analysis.online.base import MarketRegime
from crypto_analysis.online.detection.adaptive_lr import AdaptiveLearningRate
from crypto_analysis.online.detection.regime import RegimeDetector
from crypto_analysis.online.models.online_lstm import OnlineLSTM
from crypto_analysis.online.models.online_nn import OnlineNeuralNetwork
from crypto_analysis.online.models.online_rf import OnlineRandomForest
from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType
from crypto_analysis.signals.features import FeatureEngineer

try:
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OnlineSignalGenerator(SignalGenerator):
    """Signal generator with full online learning capabilities.

    Adapts to market changes in real-time using an ensemble of
    online learning models with regime-aware signal generation.

    Attributes:
        sequence_length: LSTM sequence length
        update_frequency: How often to update models
        lstm: Online LSTM model (optional)
        nn: Online Neural Network with EWC (optional)
        rf: Online Random Forest
        pa_classifier: Passive-Aggressive classifier
        regime_detector: Market regime detection
        lr_scheduler: Adaptive learning rate
        feature_engineer: Feature engineering
        scaler: Feature scaling
        model_weights: Adaptive ensemble weights
    """

    def __init__(
        self,
        name: str = "Online_Adaptive",
        sequence_length: int = 60,
        update_frequency: int = 1,
    ) -> None:
        """Initialize Online Signal Generator.

        Args:
            name: Generator name
            sequence_length: LSTM sequence length
            update_frequency: Model update frequency
        """
        super().__init__(name, lookback_period=sequence_length + 100)

        self.sequence_length = sequence_length
        self.update_frequency = update_frequency
        self.samples_since_update = 0

        self.rf = OnlineRandomForest(n_trees=10)
        self.pa_classifier = PassiveAggressiveClassifier(C=0.1)

        self.lstm: Optional[OnlineLSTM] = None
        if TF_AVAILABLE:
            try:
                self.lstm = OnlineLSTM(sequence_length=sequence_length)
            except ImportError:
                pass

        self.nn: Optional[OnlineNeuralNetwork] = None
        if TORCH_AVAILABLE:
            try:
                self.nn = OnlineNeuralNetwork(input_dim=50)
            except ImportError:
                pass

        self.regime_detector = RegimeDetector()
        self.lr_scheduler = AdaptiveLearningRate()
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()

        self.is_fitted = False
        self.feature_cols: list[str] = []
        self.prediction_buffer: deque[dict[str, Any]] = deque(maxlen=100)
        self.error_buffer: deque[float] = deque(maxlen=50)

        self.model_weights: dict[str, float] = {
            "lstm": 0.25,
            "nn": 0.25,
            "rf": 0.25,
            "pa": 0.25,
        }

    def fit(self, data: pd.DataFrame, warm_start: bool = True) -> None:
        """Initial training on historical data.

        Args:
            data: Historical market data
            warm_start: Whether to use warm starting
        """
        print(f"[{self.name}] Initial training...")

        features_df = self.feature_engineer.create_features(data, include_targets=True)
        self.feature_cols = self.feature_engineer.get_feature_columns(
            features_df, exclude_targets=True
        )

        feature_data = features_df[self.feature_cols].values
        self.scaler.fit(feature_data)
        scaled = self.scaler.transform(feature_data)

        X_lstm: list[np.ndarray] = []
        y_lstm: list[np.ndarray] = []
        X_other: list[np.ndarray] = []
        y_other: list[int] = []

        for i in range(self.sequence_length, len(scaled) - 6):
            X_lstm.append(scaled[i - self.sequence_length : i])
            X_other.append(scaled[i])

            future_return = features_df["target_return_6"].iloc[i]
            y_lstm.append(np.sign(future_return))
            y_other.append(1 if future_return > 0 else 0)

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        X_other = np.array(X_other)
        y_other = np.array(y_other)

        if self.lstm is not None and len(X_lstm) > 0:
            print(f"[{self.name}] Training LSTM...")
            for i in range(0, len(X_lstm), 32):
                batch_X = X_lstm[i : i + 32]
                batch_y = y_lstm[i : i + 32].reshape(-1, 1)
                if len(batch_X) > 0:
                    self.lstm.partial_fit(batch_X, batch_y)

        if self.nn is not None and len(X_other) > 0:
            print(f"[{self.name}] Training Neural Network...")
            for i in range(0, len(X_other), 64):
                self.nn.partial_fit(X_other[i : i + 64], y_other[i : i + 64].reshape(-1, 1))

        if len(X_other) > 0:
            print(f"[{self.name}] Training Random Forest...")
            self.rf.partial_fit(X_other, y_other)

            print(f"[{self.name}] Training PA Classifier...")
            batch_size = 1000
            for i in range(0, len(X_other), batch_size):
                end_idx = min(i + batch_size, len(X_other))
                self.pa_classifier.partial_fit(
                    X_other[i:end_idx],
                    y_other[i:end_idx],
                    classes=[0, 1],
                )

        self.is_fitted = True
        print(f"[{self.name}] Initial training complete!")

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get features for data.

        Args:
            data: Market data

        Returns:
            Feature DataFrame
        """
        features_df = self.feature_engineer.create_features(data)
        return features_df[self.feature_cols]

    def generate(self, data: pd.DataFrame, current_position: float | None = None) -> list[Signal]:
        """Generate signals with online adaptation.

        Args:
            data: Market data
            current_position: Current position (optional)

        Returns:
            List of trading signals
        """
        if not self.is_fitted or len(data) < self.lookback_period:
            return []

        regime = self.regime_detector.update(data)

        features_df = self.feature_engineer.create_features(data)
        if len(features_df) == 0:
            return []

        feature_data = features_df[self.feature_cols].values
        if len(feature_data) == 0:
            return []

        scaled = self.scaler.transform(feature_data)

        if len(scaled) < self.sequence_length:
            return []

        current_sequence = scaled[-self.sequence_length :].reshape(1, self.sequence_length, -1)
        current_point = scaled[-1:].reshape(1, -1)

        predictions: dict[str, float] = {"lstm": 0.0, "nn": 0.0, "rf": 0.0, "pa": 0.0}

        if self.lstm is not None:
            try:
                pred_lstm = self.lstm.predict(current_sequence)[0][0]
                predictions["lstm"] = pred_lstm
            except Exception:
                predictions["lstm"] = 0.0

        if self.nn is not None:
            try:
                pred_nn = self.nn.predict(current_point)[0]
                predictions["nn"] = pred_nn
            except Exception:
                predictions["nn"] = 0.0

        try:
            pred_rf = self.rf.predict(current_point)[0]
            predictions["rf"] = pred_rf
        except Exception:
            predictions["rf"] = 0.0

        try:
            pred_pa = self.pa_classifier.predict(current_point)[0]
            predictions["pa"] = pred_pa * 2 - 1
        except Exception:
            predictions["pa"] = 0.0

        self._update_model_weights()

        total_weight = sum(self.model_weights.values())
        ensemble_pred = sum(
            predictions[model] * (weight / total_weight)
            for model, weight in self.model_weights.items()
        )

        pred_variance = np.var(list(predictions.values()))
        confidence = 1 / (1 + pred_variance)

        self.samples_since_update += 1
        if self.samples_since_update >= self.update_frequency:
            self._online_update(data, scaled, ensemble_pred)
            self.samples_since_update = 0

        timestamp = data.index[-1]
        symbol = "BTC"
        signals = []

        threshold = self._get_regime_threshold(regime)

        if ensemble_pred > threshold and (current_position is None or current_position <= 0):
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    confidence=float(confidence),
                    timestamp=timestamp,
                    source=self.name,
                    metadata={
                        "ensemble_prediction": float(ensemble_pred),
                        "individual_predictions": predictions,
                        "regime": regime.name,
                        "model_weights": self.model_weights.copy(),
                    },
                )
            )
        elif ensemble_pred < -threshold and (current_position is None or current_position >= 0):
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_SHORT,
                    confidence=float(confidence),
                    timestamp=timestamp,
                    source=self.name,
                    metadata={
                        "ensemble_prediction": float(ensemble_pred),
                        "individual_predictions": predictions,
                        "regime": regime.name,
                        "model_weights": self.model_weights.copy(),
                    },
                )
            )

        self.prediction_buffer.append(
            {
                "timestamp": timestamp,
                "prediction": ensemble_pred,
                "features": current_sequence if abs(ensemble_pred) > 0.3 else None,
            }
        )

        return signals

    def _online_update(
        self, data: pd.DataFrame, scaled_features: np.ndarray, last_prediction: float
    ) -> None:
        """Update models with most recent outcome.

        Args:
            data: Market data
            scaled_features: Scaled feature array
            last_prediction: Last prediction made
        """
        if len(self.prediction_buffer) < 6:
            return

        old_pred = self.prediction_buffer[-6]

        if old_pred["features"] is None:
            return

        current_price = data["close"].iloc[-1]
        old_price = data["close"].iloc[-6]
        actual_return = (current_price - old_price) / old_price
        actual_direction = np.sign(actual_return)

        if self.lstm is not None:
            try:
                self.lstm.partial_fit(old_pred["features"], np.array([[actual_direction]]))
            except Exception:
                pass

        recent_X = scaled_features[-10:]
        future_returns = data["close"].pct_change().shift(-6).iloc[-10:].values
        recent_y = (future_returns > 0).astype(int)
        recent_y = recent_y[~np.isnan(future_returns)]
        recent_X = recent_X[: len(recent_y)]

        if len(recent_X) == len(recent_y) and len(recent_y) > 0:
            try:
                if self.nn is not None:
                    self.nn.partial_fit(recent_X, recent_y.reshape(-1, 1))
                self.rf.partial_fit(recent_X, recent_y)
                self.pa_classifier.partial_fit(recent_X, recent_y, classes=[0, 1])
            except Exception:
                pass

        prediction_error = abs(last_prediction - actual_direction)
        self.error_buffer.append(prediction_error)

        recent_vol = data["close"].pct_change().iloc[-20:].std()
        self.lr_scheduler.update(prediction_error, recent_vol)

    def _update_model_weights(self) -> None:
        """Dynamically adjust model weights based on recent performance."""
        if len(self.error_buffer) < 20:
            return

        recent_error = np.mean(list(self.error_buffer)[-20:])

        if recent_error > 0.4:
            self.model_weights = dict.fromkeys(self.model_weights, 0.25)

    def _get_regime_threshold(self, regime: MarketRegime) -> float:
        """Adjust signal threshold based on market regime.

        Args:
            regime: Current market regime

        Returns:
            Adjusted threshold value
        """
        base_threshold = 0.1

        regime_adjustments = {
            "trending_up": 0.05,
            "trending_down": 0.05,
            "ranging": 0.2,
            "volatile": 0.15,
            "crash": 0.3,
        }

        return base_threshold + regime_adjustments.get(regime.name, 0.1)
