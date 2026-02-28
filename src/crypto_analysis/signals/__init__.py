"""Signal generation modules."""

from crypto_analysis.signals.aggregator import SignalAggregator
from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType
from crypto_analysis.signals.features import FeatureEngineer
from crypto_analysis.signals.ml_generators import (
    LSTMSignalGenerator,
    RandomForestSignalGenerator,
)
from crypto_analysis.signals.statistical import StatisticalArbitrageGenerator
from crypto_analysis.signals.strategy import (
    DataHandler,
    MLStrategy,
    Order,
    OrderType,
    PortfolioManager,
    Position,
    Side,
    Strategy,
)
from crypto_analysis.signals.technical import TechnicalPatternGenerator

__all__ = [
    "Signal",
    "SignalGenerator",
    "SignalType",
    "FeatureEngineer",
    "LSTMSignalGenerator",
    "RandomForestSignalGenerator",
    "TechnicalPatternGenerator",
    "StatisticalArbitrageGenerator",
    "SignalAggregator",
    "MLStrategy",
    "Strategy",
    "DataHandler",
    "PortfolioManager",
    "Order",
    "Position",
    "Side",
    "OrderType",
]
