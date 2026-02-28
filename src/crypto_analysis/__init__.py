"""Crypto Analysis - Signal generation for cryptocurrency trading."""

__version__ = "0.1.0"

from crypto_analysis.signals.aggregator import SignalAggregator
from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType
from crypto_analysis.signals.features import FeatureEngineer

__all__ = [
    "Signal",
    "SignalGenerator",
    "SignalType",
    "SignalAggregator",
    "FeatureEngineer",
]
