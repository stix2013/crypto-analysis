"""Machine learning-based signal generators."""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType
from crypto_analysis.signals.features import FeatureEngineer

try:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        Input,
    )
    from tensorflow.keras.models import Model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    Model = None  # type: ignore
    EarlyStopping = None  # type: ignore
    ReduceLROnPlateau = None  # type: ignore
    LSTM = None  # type: ignore
    Dense = None  # type: ignore
    Input = None  # type: ignore


class LSTMSignalGenerator(SignalGenerator):
    """LSTM-based signal generator for sequence prediction.

    Predicts price direction and volatility regime using LSTM neural networks.

    Attributes:
        sequence_length: Number of time steps in input sequences
        n_features: Number of input features
        lstm_units: List of LSTM layer sizes
        dropout: Dropout rate for regularization
        model: Compiled Keras model
        feature_engineer: FeatureEngineer instance
        scaler: StandardScaler for feature normalization
        feature_cols: List of feature column names

    """

    def __init__(
        self,
        name: str = "LSTM_Predictor",
        sequence_length: int = 60,
        n_features: int = 50,
        lstm_units: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        """Initialize LSTM signal generator.

        Args:
            name: Generator name
            sequence_length: Number of time steps
            n_features: Number of features
            lstm_units: LSTM layer sizes (default: [128, 64])
            dropout: Dropout rate

        """
        if lstm_units is None:
            lstm_units = [128, 64]
        super().__init__(name, lookback_period=sequence_length + 50)

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model: Model | None = None
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def build_model(self) -> None:
        """Build LSTM architecture with attention mechanism."""
        inputs = Input(shape=(self.sequence_length, self.n_features))

        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_seq = i < len(self.lstm_units) - 1
            x = LSTM(
                units,
                return_sequences=return_seq,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
            )(x)

        # Multiple outputs: direction probability and expected return
        direction = Dense(3, activation="softmax", name="direction")(x)
        volatility = Dense(1, activation="sigmoid", name="volatility")(x)

        self.model = Model(inputs=inputs, outputs=[direction, volatility])
        self.model.compile(
            optimizer="adam",
            loss={"direction": "categorical_crossentropy", "volatility": "mse"},
            loss_weights={"direction": 1.0, "volatility": 0.5},
            metrics={"direction": "accuracy"},
        )

    def fit(
        self,
        data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        val_ratio: float = 0.2,
    ) -> None:
        """Train LSTM on historical data with proper chronological validation.

        Uses a chronological train/val split. The scaler and volatility
        threshold are fitted on training data only to prevent leakage.

        Args:
            data: OHLCV DataFrame
            epochs: Number of training epochs
            batch_size: Training batch size
            val_ratio: Fraction of data reserved for validation

        """
        print(f"[{self.name}] Preparing data...")

        # Feature engineering
        features_df = self.feature_engineer.create_features(data, include_targets=True)
        self.feature_cols = self.feature_engineer.get_feature_columns(features_df)

        feature_data = features_df[self.feature_cols].values
        n_total = len(feature_data)
        n_sequences = n_total - self.sequence_length - 6

        if n_sequences <= 0:
            raise ValueError(
                f"Not enough data for sequences: {n_total} rows, "
                f"need at least {self.sequence_length + 7}"
            )

        # Chronological split point (based on sequence count)
        n_train_seq = int((1 - val_ratio) * n_sequences)
        # Index in feature_data where training sequences end
        train_end_idx = n_train_seq + self.sequence_length

        # Fit scaler on TRAIN data only to prevent leakage
        self.scaler.fit(feature_data[:train_end_idx])
        scaled_features = self.scaler.transform(feature_data)

        # Compute volatility median on TRAIN targets only
        vol_values = features_df["target_vol_6"].values
        train_vol = vol_values[
            self.sequence_length : self.sequence_length + n_train_seq
        ]
        self.vol_median_ = float(np.nanmedian(train_vol))

        # Create sequences
        X, y_dir, y_vol = [], [], []
        for i in range(n_sequences):
            X.append(scaled_features[i : i + self.sequence_length])

            # Target: direction in 6 periods
            future_return = features_df["target_return_6"].iloc[
                i + self.sequence_length
            ]
            if future_return > 0.01:
                y_dir.append([0, 0, 1])  # Up
            elif future_return < -0.01:
                y_dir.append([1, 0, 0])  # Down
            else:
                y_dir.append([0, 1, 0])  # Neutral

            # Target: binarize vol using train-only median
            future_vol = vol_values[i + self.sequence_length]
            y_vol.append(1 if future_vol > self.vol_median_ else 0)

        X = np.array(X)
        y_dir = np.array(y_dir)
        y_vol = np.array(y_vol)

        # Chronological train/val split
        X_train, X_val = X[:n_train_seq], X[n_train_seq:]
        y_dir_train, y_dir_val = y_dir[:n_train_seq], y_dir[n_train_seq:]
        y_vol_train, y_vol_val = y_vol[:n_train_seq], y_vol[n_train_seq:]

        print(
            f"[{self.name}] Training on {len(X_train)} sequences, "
            f"validating on {len(X_val)} sequences"
        )

        if self.model is None:
            self.n_features = X.shape[2]
            self.build_model()

        self.model.fit(
            X_train,
            {"direction": y_dir_train, "volatility": y_vol_train},
            validation_data=(X_val, {"direction": y_dir_val, "volatility": y_vol_val}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=3),
            ],
            verbose=1,
        )

        self.is_fitted = True
        print(f"[{self.name}] Training complete!")

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for this generator.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with feature columns

        """
        features_df = self.feature_engineer.create_features(data)
        return features_df[self.feature_cols]

    def generate(
        self,
        data: pd.DataFrame,
        current_position: float | None = None,
    ) -> list[Signal]:
        """Generate signals based on LSTM predictions.

        Args:
            data: OHLCV DataFrame
            current_position: Current position size

        Returns:
            List of generated signals

        """
        if not self.is_fitted:
            warnings.warn(f"[{self.name}] Model not fitted yet!", stacklevel=2)
            return []

        if len(data) < self.lookback_period:
            return []

        # Prepare features
        features_df = self.feature_engineer.create_features(data)
        feature_data = features_df[self.feature_cols].values
        scaled = self.scaler.transform(feature_data)

        # Get last sequence
        sequence = scaled[-self.sequence_length :].reshape(1, self.sequence_length, -1)

        # Predict
        direction_pred, vol_pred = self.model.predict(sequence, verbose=0)

        # Parse predictions
        down_prob, neutral_prob, up_prob = direction_pred[0]
        high_vol_prob = vol_pred[0][0]

        signals = []
        timestamp = data.index[-1]
        symbol = "BTC"  # Should be parameterized

        # Generate signal based on prediction confidence
        max_prob = max(down_prob, neutral_prob, up_prob)

        if max_prob > 0.6:  # Confidence threshold
            if up_prob == max_prob and (
                current_position is None or current_position <= 0
            ):
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,
                        confidence=float(up_prob),
                        timestamp=timestamp,
                        source=self.name,
                        metadata={
                            "predicted_direction": "up",
                            "down_prob": float(down_prob),
                            "neutral_prob": float(neutral_prob),
                            "up_prob": float(up_prob),
                            "high_volatility_expected": bool(high_vol_prob > 0.5),
                        },
                    )
                )
            elif down_prob == max_prob and (
                current_position is None or current_position >= 0
            ):
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,
                        confidence=float(down_prob),
                        timestamp=timestamp,
                        source=self.name,
                        metadata={
                            "predicted_direction": "down",
                            "high_volatility_expected": bool(high_vol_prob > 0.5),
                        },
                    )
                )

        # Risk-off signal for high volatility
        if high_vol_prob > 0.7 and current_position is not None:
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.RISK_OFF,
                    confidence=float(high_vol_prob),
                    timestamp=timestamp,
                    source=self.name,
                    metadata={"reason": "high_volatility_expected"},
                )
            )

        return signals


class RandomForestSignalGenerator(SignalGenerator):
    """Random Forest-based signal generator.

    Good for feature importance analysis and non-linear pattern detection.

    Attributes:
        model: RandomForestClassifier instance
        feature_engineer: FeatureEngineer instance
        scaler: StandardScaler for feature normalization
        feature_cols: List of feature column names

    """

    def __init__(
        self,
        name: str = "RF_Classifier",
        n_estimators: int = 200,
        max_depth: int = 10,
        lookback: int = 50,
    ) -> None:
        """Initialize Random Forest signal generator.

        Args:
            name: Generator name
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
            lookback: Minimum data points needed

        """
        super().__init__(name, lookback_period=lookback)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=50,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        )
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def fit(self, data: pd.DataFrame, val_ratio: float = 0.2) -> None:
        """Train Random Forest classifier with chronological train/val split.

        Uses a chronological split so the model is evaluated on unseen
        future data. The scaler is fitted on training data only.

        Args:
            data: OHLCV DataFrame
            val_ratio: Fraction of data reserved for validation

        """
        print(f"[{self.name}] Training Random Forest...")

        # Prepare features
        features_df = self.feature_engineer.create_features(data, include_targets=True)
        self.feature_cols = self.feature_engineer.get_feature_columns(features_df)

        # Create binary classification target (up vs down, excluding small moves)
        features_df["target_binary"] = 0
        features_df.loc[features_df["target_return_3"] > 0.005, "target_binary"] = 1
        features_df.loc[features_df["target_return_3"] < -0.005, "target_binary"] = -1

        # Chronological train/val split BEFORE filtering neutrals
        split_idx = int((1 - val_ratio) * len(features_df))
        train_all = features_df.iloc[:split_idx]
        val_all = features_df.iloc[split_idx:]

        # Remove neutral cases (only from the data we train/evaluate on)
        train_data = train_all[train_all["target_binary"] != 0].copy()
        val_data = val_all[val_all["target_binary"] != 0].copy()

        X_train = train_data[self.feature_cols].values
        y_train = (train_data["target_binary"] > 0).astype(int).values

        # Fit scaler on TRAIN data only
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # Train on train set only
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        if len(val_data) > 0:
            X_val = val_data[self.feature_cols].values
            y_val = (val_data["target_binary"] > 0).astype(int).values
            X_val_scaled = self.scaler.transform(X_val)
            val_accuracy = self.model.score(X_val_scaled, y_val)
            print(
                f"[{self.name}] Validation accuracy: {val_accuracy:.4f} "
                f"(train={len(train_data)}, val={len(val_data)} samples)"
            )

        # Store feature importance
        self.feature_importance = dict(
            zip(self.feature_cols, self.model.feature_importances_)
        )

        self.is_fitted = True
        print(f"[{self.name}] Training complete!")
        top_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]
        print(f"Top 5 features: {top_features}")

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for this generator.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with feature columns

        """
        features_df = self.feature_engineer.create_features(data)
        return features_df[self.feature_cols]

    def generate(
        self,
        data: pd.DataFrame,
        current_position: float | None = None,
    ) -> list[Signal]:
        """Generate signals based on RF predictions.

        Args:
            data: OHLCV DataFrame
            current_position: Current position size

        Returns:
            List of generated signals

        """
        if not self.is_fitted:
            return []

        if len(data) < self.lookback_period:
            return []

        # Prepare features
        features_df = self.feature_engineer.create_features(data)
        if len(features_df) == 0:
            return []
        X = features_df[self.feature_cols].values[-1:].reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict probability
        proba = self.model.predict_proba(X_scaled)[0]
        up_prob = proba[1]
        down_prob = proba[0]

        timestamp = data.index[-1]
        symbol = "BTC"
        signals = []

        # Generate signal if confidence high
        if max(up_prob, down_prob) > 0.65:
            if up_prob > down_prob and (
                current_position is None or current_position <= 0
            ):
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,
                        confidence=float(up_prob),
                        timestamp=timestamp,
                        source=self.name,
                        metadata={"model_probability": float(up_prob)},
                    )
                )
            elif down_prob > up_prob and (
                current_position is None or current_position >= 0
            ):
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,
                        confidence=float(down_prob),
                        timestamp=timestamp,
                        source=self.name,
                        metadata={"model_probability": float(down_prob)},
                    )
                )

        return signals
