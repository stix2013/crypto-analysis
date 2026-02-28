"""Online learning module for real-time market adaptation.

This module provides components for adapting to changing market conditions:
- Online learning models (Random Forest, LSTM, Neural Network)
- Market regime detection
- Adaptive learning rate scheduling
- Continuous learning pipeline
"""

from crypto_analysis.online.base import MarketRegime, OnlineModel
from crypto_analysis.online.detection.adaptive_lr import AdaptiveLearningRate
from crypto_analysis.online.detection.regime import RegimeDetector
from crypto_analysis.online.generator import OnlineSignalGenerator
from crypto_analysis.online.pipeline import ContinuousLearningPipeline

__all__ = [
    "MarketRegime",
    "OnlineModel",
    "RegimeDetector",
    "AdaptiveLearningRate",
    "OnlineSignalGenerator",
    "ContinuousLearningPipeline",
]
