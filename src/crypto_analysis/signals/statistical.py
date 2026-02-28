"""Statistical arbitrage signal generators."""


import pandas as pd

from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType


class StatisticalArbitrageGenerator(SignalGenerator):
    """Mean reversion signals based on statistical analysis.

    Implements Bollinger Bands mean reversion and RSI extreme signals.

    Attributes:
        mean_period: Period for mean calculation
        std_threshold: Z-score threshold for signals

    """

    def __init__(
        self,
        name: str = "StatArb",
        lookback: int = 100,
        mean_period: int = 20,
        std_threshold: float = 2.0,
    ) -> None:
        """Initialize statistical arbitrage generator.

        Args:
            name: Generator name
            lookback: Minimum data points needed
            mean_period: Period for moving average
            std_threshold: Z-score threshold for signals

        """
        super().__init__(name, lookback_period=lookback)
        self.mean_period = mean_period
        self.std_threshold = std_threshold

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the generator (no-op for statistical arbitrage).

        Args:
            data: OHLCV DataFrame

        """
        self.is_fitted = True

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return raw data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Same DataFrame

        """
        return data

    def generate(
        self,
        data: pd.DataFrame,
        current_position: float | None = None,
    ) -> list[Signal]:
        """Generate mean reversion signals.

        Args:
            data: OHLCV DataFrame
            current_position: Current position size

        Returns:
            List of generated signals

        """
        if len(data) < self.lookback_period:
            return []

        signals = []
        timestamp = data.index[-1]
        symbol = "BTC"

        # Bollinger Band mean reversion
        sma = data["close"].rolling(self.mean_period).mean().iloc[-1]
        std = data["close"].rolling(self.mean_period).std().iloc[-1]
        current = data["close"].iloc[-1]

        z_score = (current - sma) / (std + 1e-10)

        # Check for extreme deviation
        if z_score < -self.std_threshold and (current_position is None or current_position <= 0):
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    confidence=min(abs(z_score) / 4, 0.95),
                    timestamp=timestamp,
                    source=self.name,
                    metadata={
                        "z_score": float(z_score),
                        "sma": float(sma),
                        "strategy": "bb_mean_reversion",
                    },
                )
            )
        elif z_score > self.std_threshold and (current_position is None or current_position >= 0):
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_SHORT,
                    confidence=min(abs(z_score) / 4, 0.95),
                    timestamp=timestamp,
                    source=self.name,
                    metadata={
                        "z_score": float(z_score),
                        "sma": float(sma),
                        "strategy": "bb_mean_reversion",
                    },
                )
            )

        # RSI extreme
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        if rsi < 20 and (current_position is None or current_position <= 0):
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    confidence=(30 - rsi) / 30,
                    timestamp=timestamp,
                    source=f"{self.name}_RSI",
                    metadata={"rsi": float(rsi)},
                )
            )
        elif rsi > 80 and (current_position is None or current_position >= 0):
            signals.append(
                Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_SHORT,
                    confidence=(rsi - 70) / 30,
                    timestamp=timestamp,
                    source=f"{self.name}_RSI",
                    metadata={"rsi": float(rsi)},
                )
            )

        return signals
