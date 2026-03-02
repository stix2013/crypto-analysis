"""Base classes and types for signal generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class SignalType(Enum):
    """Types of trading signals."""

    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    RISK_OFF = "risk_off"  # Emergency exit


@dataclass
class Signal:
    """A trading signal with metadata.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTC", "ETH")
        signal_type: Type of signal (entry, exit, etc.)
        confidence: Confidence score between 0.0 and 1.0
        timestamp: When the signal was generated
        metadata: Additional information (predicted price, stop levels, etc.)
        source: Which generator created this signal

    """

    symbol: str
    signal_type: SignalType
    confidence: float
    timestamp: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        """Validate confidence is within bounds."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


class SignalGenerator(ABC):
    """Abstract base class for all signal generators.

    All signal generators must implement fit(), generate(), and get_features().

    Attributes:
        name: Unique name for this generator
        lookback_period: Minimum data points needed
        is_fitted: Whether the generator has been trained
        feature_importance: Dictionary of feature names to importance scores

    """

    def __init__(self, name: str, lookback_period: int = 100) -> None:
        """Initialize the signal generator.

        Args:
            name: Unique name for this generator
            lookback_period: Minimum data points needed for generation

        """
        self.name = name
        self.lookback_period = lookback_period
        self.is_fitted = False
        self.feature_importance: dict[str, float] = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Train the generator on historical data.

        Args:
            data: OHLCV data with required columns

        """
        pass

    @abstractmethod
    def generate(
        self,
        data: pd.DataFrame,
        current_position: float | None = None,
    ) -> list[Signal]:
        """Generate signals based on current data.

        Args:
            data: OHLCV data with required columns
            current_position: Current position size (positive=long, negative=short)

        Returns:
            List of generated signals (may be empty)

        """
        pass

    @abstractmethod
    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features used by this generator.

        Args:
            data: OHLCV data

        Returns:
            DataFrame with feature columns

        """
        pass
