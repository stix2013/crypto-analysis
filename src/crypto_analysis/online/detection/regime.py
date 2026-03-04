"""Market regime detection using unsupervised learning."""

import typing
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from crypto_analysis.online.base import MarketRegime


class RegimeDetector:
    """Detects market regimes using rule-based classification.

    Identifies 5 market regimes:
    - trending_up: Strong upward price movement
    - trending_down: Strong downward price movement
    - ranging: Sideways market with low trend
    - volatile: High volatility without clear direction
    - crash: Sharp downward movement with high volatility

    Attributes:
        n_regimes: Number of regime categories
        lookback: Number of periods for feature extraction
        current_regime: Currently detected regime
        regime_history: Historical regime detections
    """

    def __init__(self, n_regimes: int = 5, lookback: int = 100) -> None:
        """Initialize Regime Detector.

        Args:
            n_regimes: Number of regime categories
            lookback: Periods for feature extraction
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.regime_models: dict[int, Any] = {}
        self.current_regime: MarketRegime | None = None
        self.regime_history: deque[MarketRegime] = deque(maxlen=1000)
        self.feature_buffer: deque[np.ndarray] = deque(maxlen=lookback * 2)

        self._regime_stats_data: dict = {"returns": [], "volatility": [], "volume": []}

    @property
    def regime_stats(self) -> dict:
        return self._regime_stats_data

    @regime_stats.setter
    def regime_stats(self, value: dict) -> None:
        self._regime_stats_data = value

    def _get_regime_stat(self, regime_id: int) -> dict:
        if regime_id not in self._regime_stats_data:
            self._regime_stats_data[regime_id] = {
                "returns": [],
                "volatility": [],
                "volume": [],
            }
        return typing.cast(dict[Any, Any], self._regime_stats_data[regime_id])

    def extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification.

        Args:
            data: Market data with OHLCV columns

        Returns:
            Feature vector of shape (10,)
        """
        if len(data) < self.lookback:
            return np.zeros(10)

        recent = data.tail(self.lookback)
        returns = recent["close"].pct_change().dropna()

        features = np.zeros(10)

        # 0: Momentum
        if recent["close"].iloc[0] > 0:
            features[0] = recent["close"].iloc[-1] / recent["close"].iloc[0] - 1

        # 1: Trend
        if len(recent) > 1:
            close_mean = recent["close"].mean()
            if close_mean > 0:
                trend_coef = np.polyfit(range(len(recent)), recent["close"], 1)[0]
                features[1] = trend_coef / close_mean

        # 2: Volatility
        features[2] = returns.std() * np.sqrt(365 * 24)

        # 3: Volatility Acceleration
        if len(returns) > 1:
            prev_vol = returns.shift(1).std()
            if prev_vol > 0:
                features[3] = returns.std() / prev_vol - 1

        # 4: Relative Volume
        vol_rolling = recent["volume"].rolling(50).mean()
        if not vol_rolling.empty and vol_rolling.iloc[-1] > 0:
            features[4] = recent["volume"].mean() / vol_rolling.iloc[-1]

        # 5: Deviation from Moving Average (Z-Score like)
        close_std = recent["close"].std()
        if close_std > 0:
            ma20 = recent["close"].rolling(20).mean()
            if not ma20.empty:
                features[5] = (recent["close"].iloc[-1] - ma20.iloc[-1]) / close_std

        # 6: Range
        close_mean = recent["close"].mean()
        if close_mean > 0:
            features[6] = (recent["high"].max() - recent["low"].min()) / close_mean

        # 7, 8: Distribution shape
        if len(returns) > 10:
            features[7] = stats.skew(returns)
            features[8] = stats.kurtosis(returns)

        # 9: Efficiency Ratio
        if "true_range" in recent.columns:
            tr_sum = recent["true_range"].sum()
            if tr_sum > 0:
                features[9] = (
                    abs(recent["close"].iloc[-1] - recent["close"].iloc[0]) / tr_sum
                )

        return features

    def update(self, data: pd.DataFrame) -> MarketRegime:
        """Update regime detection and return current regime.

        Args:
            data: Market data with OHLCV columns

        Returns:
            Current detected market regime
        """
        features = self.extract_regime_features(data)
        self.feature_buffer.append(features)

        returns = features[0]
        volatility = features[2]
        trend_strength = features[1]

        if volatility > 0.8:
            regime_name = "volatile"
            regime_id = 0
        elif returns < -0.15:
            regime_name = "crash"
            regime_id = 1
        elif trend_strength > 0.001:
            regime_name = "trending_up"
            regime_id = 2
        elif trend_strength < -0.001:
            regime_name = "trending_down"
            regime_id = 3
        else:
            regime_name = "ranging"
            regime_id = 4

        stat = self._get_regime_stat(regime_id)
        stat["returns"].append(returns)
        stat["volatility"].append(volatility)
        stat["volume"].append(features[4])

        regime = MarketRegime(
            regime_id=regime_id,
            name=regime_name,
            features=features,
            start_time=data.index[-1],
            confidence=self._calculate_regime_confidence(regime_id, features),
        )

        if self.current_regime is None or self.current_regime.regime_id != regime_id:
            self._on_regime_change(self.current_regime, regime)

        self.current_regime = regime
        self.regime_history.append(regime)

        return regime

    def _calculate_regime_confidence(
        self, regime_id: int, features: np.ndarray
    ) -> float:
        """Calculate confidence in regime classification.

        Args:
            regime_id: Detected regime ID
            features: Current feature vector

        Returns:
            Confidence score between 0 and 1
        """
        stat = self._get_regime_stat(regime_id)
        if len(stat["returns"]) < 10:
            return 0.5

        hist_returns = np.array(stat["returns"][-100:])
        current_return = features[0]

        mean_ret = hist_returns.mean()
        std_ret = hist_returns.std() + 1e-10

        z_score = abs(current_return - mean_ret) / std_ret
        confidence = 1 / (1 + z_score)

        return float(min(confidence, 0.99))

    def _on_regime_change(
        self, old_regime: MarketRegime | None, new_regime: MarketRegime
    ) -> None:
        """Handle regime transition.

        Args:
            old_regime: Previous regime (can be None)
            new_regime: Newly detected regime
        """
        print(
            f"Regime change: {old_regime.name if old_regime else 'None'} -> {new_regime.name}"
        )

    def get_regime_specific_model(self, base_models: dict[str, Any]) -> Any:
        """Select appropriate model for current regime.

        Args:
            base_models: Dictionary of available models

        Returns:
            Model appropriate for current regime
        """
        if self.current_regime is None:
            return base_models.get("default")

        regime_model_map = {
            "trending_up": "trend_following",
            "trending_down": "trend_following",
            "ranging": "mean_reversion",
            "volatile": "momentum",
            "crash": "risk_off",
        }

        model_type = regime_model_map.get(self.current_regime.name, "default")
        return base_models.get(model_type, base_models.get("default"))
