"""Feature engineering for cryptocurrency trading signals."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Comprehensive feature engineering for crypto trading.

    Creates technical, statistical, and market microstructure features
    from OHLCV (Open, High, Low, Close, Volume) data.

    Attributes:
        scaler: StandardScaler for feature normalization
        feature_names: List of generated feature column names

    """

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    def create_features(
        self,
        df: pd.DataFrame,
        include_targets: bool = False,
    ) -> pd.DataFrame:
        """Create comprehensive feature set from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
            include_targets: Whether to generate target variables

        Returns:
            DataFrame with all features (and targets if requested)

        """
        data = df.copy()

        # Price-based features
        data = self._add_price_features(data)

        # Volume features
        data = self._add_volume_features(data)

        # Volatility features
        data = self._add_volatility_features(data)

        # Trend features
        data = self._add_trend_features(data)

        # Momentum features
        data = self._add_momentum_features(data)

        # Market microstructure
        data = self._add_microstructure_features(data)

        # Time features
        data = self._add_time_features(data)

        if include_targets:
            data = self._add_targets(data)
            feature_cols = self._get_base_feature_columns(data)
            data = data.dropna(subset=feature_cols)
        else:
            data = data.dropna()

        return data

    def _get_base_feature_columns(self, data: pd.DataFrame) -> list[str]:
        """Get list of base feature column names (non-target)."""
        exclude = ["open", "high", "low", "close", "volume"]
        exclude += [col for col in data.columns if col.startswith("target_")]
        return [col for col in data.columns if col not in exclude]

    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price action features.

        Features added:
            - returns: Percentage change in close price
            - log_returns: Log returns
            - close_position: Position within bar (0=low, 1=high)
            - body_size: Relative candle body size
            - upper_shadow: Upper shadow size
            - lower_shadow: Lower shadow size
            - gap: Opening gap from previous close

        """
        # Returns
        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"] / data["close"].shift(1))

        # Price position within bar
        data["close_position"] = (data["close"] - data["low"]) / (
            data["high"] - data["low"] + 1e-10
        )

        # Body size
        data["body_size"] = abs(data["close"] - data["open"]) / data["open"]
        data["upper_shadow"] = (data["high"] - data[["close", "open"]].max(axis=1)) / data["open"]
        data["lower_shadow"] = (data[["close", "open"]].min(axis=1) - data["low"]) / data["open"]

        # Gap analysis
        data["gap"] = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)

        return data

    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features.

        Features added:
            - volume_ma_10: 10-period volume moving average
            - volume_ma_30: 30-period volume moving average
            - volume_ratio: Current volume vs 30-period average
            - volume_trend: Linear trend in volume
            - volume_price_trend: Volume-price trend indicator
            - obv: On-balance volume
            - vwap: Volume-weighted average price
            - vwap_distance: Distance from VWAP

        """
        # Volume moving averages
        data["volume_ma_10"] = data["volume"].rolling(10).mean()
        data["volume_ma_30"] = data["volume"].rolling(30).mean()
        data["volume_ratio"] = data["volume"] / data["volume_ma_30"]

        # Volume trend (linear regression slope)
        data["volume_trend"] = (
            data["volume"]
            .rolling(10)
            .apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0],
                raw=True,
            )
        )

        # Price-volume relationship
        data["volume_price_trend"] = data["volume"] * data["returns"]
        data["obv"] = (np.sign(data["close"].diff()) * data["volume"]).cumsum()

        # VWAP
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        data["vwap"] = (typical_price * data["volume"]).rolling(20).sum() / data["volume"].rolling(
            20
        ).sum()
        data["vwap_distance"] = (data["close"] - data["vwap"]) / data["vwap"]

        return data

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features.

        Features added:
            - volatility_{5,10,20,50}: Annualized volatility
            - true_range: True range
            - atr_14: 14-period Average True Range
            - atr_ratio: Current TR vs ATR
            - bb_**: Bollinger Bands (middle, upper, lower, position, width)
            - kc_**: Keltner Channels

        """
        # Standard volatility
        for window in [5, 10, 20, 50]:
            data[f"volatility_{window}"] = data["returns"].rolling(window).std() * np.sqrt(365 * 24)

        # True Range and ATR
        data["tr1"] = data["high"] - data["low"]
        data["tr2"] = abs(data["high"] - data["close"].shift(1))
        data["tr3"] = abs(data["low"] - data["close"].shift(1))
        data["true_range"] = data[["tr1", "tr2", "tr3"]].max(axis=1)
        data["atr_14"] = data["true_range"].rolling(14).mean()
        data["atr_ratio"] = data["true_range"] / data["atr_14"]

        # Bollinger Bands
        data["bb_middle"] = data["close"].rolling(20).mean()
        data["bb_std"] = data["close"].rolling(20).std()
        data["bb_upper"] = data["bb_middle"] + 2 * data["bb_std"]
        data["bb_lower"] = data["bb_middle"] - 2 * data["bb_std"]
        data["bb_position"] = (data["close"] - data["bb_lower"]) / (
            data["bb_upper"] - data["bb_lower"] + 1e-10
        )
        data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]

        # Keltner Channels
        data["kc_middle"] = data["close"].rolling(20).mean()
        data["kc_upper"] = data["kc_middle"] + 2 * data["atr_14"]
        data["kc_lower"] = data["kc_middle"] - 2 * data["atr_14"]

        return data

    def _add_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend identification features.

        Features added:
            - ma_{7,14,21,50,200}: Moving averages
            - ma_**_slope: Slope of moving average
            - close_ma_**_ratio: Close price relative to MA
            - ma_7_21_cross: MA crossover signal
            - ma_50_200_cross: Golden/death cross
            - adx: Average Directional Index
            - trend_slope_**: Linear trend slope
            - trend_r2_**: Trend strength (R²)

        """
        # Moving averages
        for ma in [7, 14, 21, 50, 200]:
            data[f"ma_{ma}"] = data["close"].rolling(ma).mean()
            data[f"ma_{ma}_slope"] = data[f"ma_{ma}"].diff(5) / data[f"ma_{ma}"].shift(5)
            data[f"close_ma_{ma}_ratio"] = data["close"] / data[f"ma_{ma}"]

        # Moving average crossovers
        data["ma_7_21_cross"] = (data["ma_7"] > data["ma_21"]).astype(int).diff()
        data["ma_50_200_cross"] = (data["ma_50"] > data["ma_200"]).astype(int).diff()

        # Trend strength
        data["adx"] = self._calculate_adx(data)

        # Linear regression trend
        for window in [14, 30]:
            data[f"trend_slope_{window}"] = (
                data["close"]
                .rolling(window)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean(),
                    raw=True,
                )
            )
            data[f"trend_r2_{window}"] = (
                data["close"]
                .rolling(window)
                .apply(
                    lambda x: np.corrcoef(range(len(x)), x)[0, 1] ** 2,
                    raw=True,
                )
            )

        return data

    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators.

        Features added:
            - rsi_14: Relative Strength Index
            - rsi_slope: RSI momentum
            - macd: MACD line
            - macd_signal: Signal line
            - macd_histogram: Histogram
            - stoch_k: Stochastic %K
            - stoch_d: Stochastic %D
            - williams_r: Williams %R
            - roc_{5,10,20}: Rate of Change

        """
        # RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["rsi_14"] = 100 - (100 / (1 + rs))
        data["rsi_slope"] = data["rsi_14"].diff(3)

        # MACD
        exp1 = data["close"].ewm(span=12, adjust=False).mean()
        exp2 = data["close"].ewm(span=26, adjust=False).mean()
        data["macd"] = exp1 - exp2
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_histogram"] = data["macd"] - data["macd_signal"]

        # Stochastic
        low_min = data["low"].rolling(window=14).min()
        high_max = data["high"].rolling(window=14).max()
        data["stoch_k"] = 100 * (data["close"] - low_min) / (high_max - low_min + 1e-10)
        data["stoch_d"] = data["stoch_k"].rolling(window=3).mean()

        # Williams %R
        data["williams_r"] = -100 * (high_max - data["close"]) / (high_max - low_min + 1e-10)

        # Rate of Change
        for period in [5, 10, 20]:
            data[f"roc_{period}"] = (data["close"] - data["close"].shift(period)) / data[
                "close"
            ].shift(period)

        return data

    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features.

        Features added:
            - spread_est: Bid-ask spread estimation
            - intraday_range: Price range relative to open
            - efficiency_ratio: Trend efficiency
            - fractal_dim: Fractal dimension approximation

        """
        # Bid-ask spread estimation (if no order book data)
        data["spread_est"] = 2 * (data["high"] - data["low"]) / (data["high"] + data["low"])

        # Intraday volatility patterns
        data["intraday_range"] = (data["high"] - data["low"]) / data["open"]

        # Efficiency ratio
        data["efficiency_ratio"] = abs(data["close"] - data["close"].shift(10)) / (
            data["true_range"].rolling(10).sum()
        )

        # Fractal dimension approximation
        data["fractal_dim"] = (
            np.log(data["true_range"].rolling(10).sum())
            / np.log(10)
            / np.log(data[["high", "low"]].std(axis=1))
        )

        return data

    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features.

        Features added:
            - hour: Hour of day
            - day_of_week: Day of week (0=Monday)
            - month: Month
            - hour_sin/cos: Cyclical hour encoding
            - dow_sin/cos: Cyclical day encoding
            - is_weekend: Weekend indicator

        """
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["month"] = data.index.month

        # Cyclical encoding
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
        data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # Weekend effect (for crypto, weekends often different)
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

        return data

    def _add_targets(
        self,
        data: pd.DataFrame,
        forward_periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """Create target variables for supervised learning.

        Args:
            data: DataFrame with features
            forward_periods: List of periods for forward-looking targets

        Returns:
            DataFrame with added target columns

        """
        if forward_periods is None:
            forward_periods = [1, 3, 6, 12]

        for period in forward_periods:
            # Future returns
            data[f"target_return_{period}"] = data["close"].shift(-period) / data["close"] - 1

            # Binary direction
            data[f"target_direction_{period}"] = (data[f"target_return_{period}"] > 0).astype(int)

            # Volatility regime
            future_vol = data["returns"].shift(-period).rolling(period).std()
            data[f"target_vol_{period}"] = future_vol

            # Target for classification (strong up, up, neutral, down, strong down)
            returns = data[f"target_return_{period}"]
            thresholds = [
                returns.quantile(0.2),
                returns.quantile(0.4),
                returns.quantile(0.6),
                returns.quantile(0.8),
            ]

            def classify_return(r: float) -> int:
                """Classify return into 5 categories."""
                if r < thresholds[0]:
                    return 0  # Strong down
                elif r < thresholds[1]:
                    return 1  # Down
                elif r < thresholds[2]:
                    return 2  # Neutral
                elif r < thresholds[3]:
                    return 3  # Up
                else:
                    return 4  # Strong up

            data[f"target_class_{period}"] = returns.apply(classify_return)

        return data

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index.

        Args:
            data: DataFrame with high, low, close
            period: ADX calculation period

        Returns:
            Series with ADX values

        """
        plus_dm = data["high"].diff()
        minus_dm = data["low"].diff().abs()

        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)

        tr = data["true_range"]

        plus_di = 100 * plus_dm.rolling(period).mean() / tr.rolling(period).mean()
        minus_di = 100 * minus_dm.rolling(period).mean() / tr.rolling(period).mean()

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx

    def get_feature_columns(
        self,
        data: pd.DataFrame,
        exclude_targets: bool = True,
    ) -> list[str]:
        """Get list of feature column names.

        Args:
            data: DataFrame with all columns
            exclude_targets: Whether to exclude target columns

        Returns:
            List of feature column names

        """
        exclude = ["open", "high", "low", "close", "volume"]
        if exclude_targets:
            exclude += [col for col in data.columns if "target_" in col]

        return [col for col in data.columns if col not in exclude]
