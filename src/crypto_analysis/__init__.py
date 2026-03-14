"""Crypto Analysis - Signal generation for cryptocurrency trading."""

__version__ = "0.1.0"

from crypto_analysis.online import (
    AdaptiveLearningRate,
    ContinuousLearningPipeline,
    MarketRegime,
    OnlineModel,
    OnlineSignalGenerator,
    RegimeDetector,
)
from crypto_analysis.settings import Settings, get_settings
from crypto_analysis.signals.aggregator import SignalAggregator
from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType
from crypto_analysis.signals.features import FeatureEngineer

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # Signals
    "Signal",
    "SignalGenerator",
    "SignalType",
    "SignalAggregator",
    "FeatureEngineer",
    # Online Learning
    "MarketRegime",
    "OnlineModel",
    "RegimeDetector",
    "AdaptiveLearningRate",
    "OnlineSignalGenerator",
    "ContinuousLearningPipeline",
]
