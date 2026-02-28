"""Technical pattern-based signal generators."""

import pandas as pd
from scipy.signal import find_peaks

from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType


class TechnicalPatternGenerator(SignalGenerator):
    """Pattern-based signal generation using classical technical analysis.

    Detects patterns like double bottoms, double tops, breakouts,
    and mean reversion opportunities.

    Attributes:
        patterns: Dictionary of pattern name to detection function

    """

    def __init__(self, name: str = "Technical_Patterns") -> None:
        """Initialize technical pattern generator.

        Args:
            name: Generator name

        """
        super().__init__(name, lookback_period=100)
        self.patterns: dict[
            str,
            callable,
        ] = {
            "double_bottom": self._detect_double_bottom,
            "double_top": self._detect_double_top,
            "breakout": self._detect_breakout,
            "mean_reversion": self._detect_mean_reversion,
        }

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the generator (no-op for technical patterns).

        Args:
            data: OHLCV DataFrame

        """
        # Technical patterns don't require fitting
        self.is_fitted = True

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return raw data (no feature transformation).

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
        """Generate signals based on pattern detection.

        Args:
            data: OHLCV DataFrame
            current_position: Current position size

        Returns:
            List of detected pattern signals

        """
        if len(data) < self.lookback_period:
            return []

        signals = []
        timestamp = data.index[-1]
        symbol = "BTC"

        # Check each pattern
        for pattern_name, pattern_func in self.patterns.items():
            signal_type, confidence, metadata = pattern_func(data)

            if signal_type and self._position_compatible(signal_type, current_position):
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=confidence,
                        timestamp=timestamp,
                        source=f"{self.name}_{pattern_name}",
                        metadata=metadata,
                    )
                )

        return signals

    def _position_compatible(
        self,
        signal_type: SignalType,
        current_position: float | None,
    ) -> bool:
        """Check if signal is compatible with current position.

        Args:
            signal_type: Type of signal to generate
            current_position: Current position size

        Returns:
            True if compatible, False otherwise

        """
        if current_position is None or abs(current_position) < 1e-10:
            return signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]
        elif current_position > 0:
            return signal_type in [SignalType.EXIT_LONG, SignalType.RISK_OFF]
        else:
            return signal_type in [SignalType.EXIT_SHORT, SignalType.RISK_OFF]

    def _detect_double_bottom(
        self,
        data: pd.DataFrame,
        lookback: int = 30,
    ) -> tuple[SignalType | None, float, dict]:
        """Detect W-shaped double bottom pattern.

        Args:
            data: OHLCV DataFrame
            lookback: Number of bars to look back

        Returns:
            Tuple of (signal_type, confidence, metadata)

        """
        recent = data.tail(lookback)
        lows = recent["low"].values

        # Find local minima
        minima_idx, _ = find_peaks(-lows, distance=5, prominence=lows.std() * 0.5)

        if len(minima_idx) >= 2:
            # Check if two bottoms at similar level
            bottom1 = lows[minima_idx[-2]]
            bottom2 = lows[minima_idx[-1]]

            if abs(bottom1 - bottom2) / bottom1 < 0.02:  # Within 2%
                # Check for breakout above neckline
                neckline = recent["high"].iloc[minima_idx[-2] : minima_idx[-1]].max()
                current = data["close"].iloc[-1]

                if current > neckline * 0.99:  # Near or above neckline
                    return (
                        SignalType.ENTRY_LONG,
                        0.7,
                        {
                            "pattern": "double_bottom",
                            "neckline": neckline,
                            "bottom_levels": [bottom1, bottom2],
                        },
                    )

        return None, 0, {}

    def _detect_double_top(
        self,
        data: pd.DataFrame,
        lookback: int = 30,
    ) -> tuple[SignalType | None, float, dict]:
        """Detect M-shaped double top pattern.

        Args:
            data: OHLCV DataFrame
            lookback: Number of bars to look back

        Returns:
            Tuple of (signal_type, confidence, metadata)

        """
        recent = data.tail(lookback)
        highs = recent["high"].values

        maxima_idx, _ = find_peaks(highs, distance=5, prominence=highs.std() * 0.5)

        if len(maxima_idx) >= 2:
            top1 = highs[maxima_idx[-2]]
            top2 = highs[maxima_idx[-1]]

            if abs(top1 - top2) / top1 < 0.02:
                neckline = recent["low"].iloc[maxima_idx[-2] : maxima_idx[-1]].min()
                current = data["close"].iloc[-1]

                if current < neckline * 1.01:
                    return (
                        SignalType.ENTRY_SHORT,
                        0.7,
                        {
                            "pattern": "double_top",
                            "neckline": neckline,
                        },
                    )

        return None, 0, {}

    def _detect_breakout(
        self,
        data: pd.DataFrame,
        lookback: int = 20,
    ) -> tuple[SignalType | None, float, dict]:
        """Detect volatility breakout from consolidation.

        Args:
            data: OHLCV DataFrame
            lookback: Number of bars for consolidation detection

        Returns:
            Tuple of (signal_type, confidence, metadata)

        """
        recent = data.tail(lookback)

        # Check for consolidation (low volatility)
        consolidation_range = (recent["high"].max() - recent["low"].min()) / recent["close"].mean()

        if consolidation_range < 0.05:  # Less than 5% range
            # Check for breakout
            current = data["close"].iloc[-1]
            upper = recent["high"].max()
            lower = recent["low"].min()

            if current > upper * 0.998:
                return (
                    SignalType.ENTRY_LONG,
                    0.6,
                    {
                        "pattern": "breakout_long",
                        "consolidation_range": consolidation_range,
                    },
                )
            elif current < lower * 1.002:
                return (
                    SignalType.ENTRY_SHORT,
                    0.6,
                    {
                        "pattern": "breakout_short",
                        "consolidation_range": consolidation_range,
                    },
                )

        return None, 0, {}

    def _detect_mean_reversion(
        self,
        data: pd.DataFrame,
    ) -> tuple[SignalType | None, float, dict]:
        """Detect extreme moves likely to revert.

        Args:
            data: OHLCV DataFrame

        Returns:
            Tuple of (signal_type, confidence, metadata)

        """
        # Calculate z-score of returns
        returns = data["close"].pct_change().tail(50)
        z_score = (returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-10)

        if abs(z_score) > 2.5:  # More than 2.5 std devs
            if z_score < -2.5:
                return (
                    SignalType.ENTRY_LONG,
                    min(abs(z_score) / 4, 0.9),
                    {
                        "z_score": float(z_score),
                        "pattern": "mean_reversion_up",
                    },
                )
            else:
                return (
                    SignalType.ENTRY_SHORT,
                    min(abs(z_score) / 4, 0.9),
                    {
                        "z_score": float(z_score),
                        "pattern": "mean_reversion_down",
                    },
                )

        return None, 0, {}
