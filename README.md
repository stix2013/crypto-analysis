# Crypto Analysis

Signal generation system integrating machine learning, technical analysis, and statistical methods for cryptocurrency trading. Uses PyTorch (CPU-only) for adaptive online learning.

## Features

- **Adaptive Online Learning**: Continuous model updates with real-time market data, regime detection, and adaptive learning rates.
- **ML-based Generators**: PyTorch-based LSTM and Random Forest models for price prediction and classification.
- **Technical Analysis**: Pattern recognition (double top/bottom, breakouts, mean reversion) and comprehensive feature engineering (RSI, MFI, Choppiness Index).
- **Signal Aggregation**: Combine multiple generators with weighted confidence, majority vote, or best confidence methods.
- **Risk Management**: Integrated Stop-Loss (SL) and Take-Profit (TP) triggers, and dynamic volatility-adjusted position sizing.
- **Backtesting & Optimization**: Robust event-driven engine with realistic execution, performance metrics (Sharpe/Sortino), and parameter grid search.
- **Analytics & Visualization**: Equity curve plotting and drawdown analysis.

## Installation

```bash
# Clone the repository
cd crypto-analysis

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### 1. Training & Running Online Models
```bash
# Use the helper script to train an online model on ETHUSDT
./run_training.sh ETHUSDT 15m 5000
```

### 2. Manual Signal Generation
```python
import pandas as pd
from crypto_analysis.signals import (
    RandomForestSignalGenerator,
    TechnicalPatternGenerator,
    SignalAggregator,
)

# ... (see examples/basic_usage.py for details)
```

## Project Structure

```
crypto-analysis/
├── src/crypto_analysis/
│   ├── data/           # Binance API client
│   ├── online/         # Adaptive learning & regime detection
│   ├── signals/        # Core logic & Backtester
│   └── utils/          # Analytics & Optimization
├── tests/              # Comprehensive test suites
├── scripts/            # Training and Prediction CLIs
├── signals/            # Generated signals (CSV)
├── models/             # Trained model checkpoints (joblib)
├── run_training.sh     # Training runner
├── pyproject.toml
└── README.md
```

## Signal Types

- `ENTRY_LONG`: Buy signal
- `ENTRY_SHORT`: Short sell signal
- `EXIT_LONG`: Close long position
- `EXIT_SHORT`: Close short position
- `HOLD`: No action
- `RISK_OFF`: Emergency exit


## Generators

### ML Generators

- **LSTMSignalGenerator**: PyTorch LSTM-based sequence prediction with attention
- **RandomForestSignalGenerator**: Random Forest classifier with feature importance

### Technical Generators

- **TechnicalPatternGenerator**: Double top/bottom, breakouts, mean reversion

### Statistical Generators

- **StatisticalArbitrageGenerator**: Bollinger Bands and RSI extremes

## License

MIT
