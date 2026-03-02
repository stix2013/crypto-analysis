# Crypto Analysis

Signal generation system integrating machine learning, technical analysis, and statistical methods for cryptocurrency trading.

## Features

- **ML-based Generators**: LSTM and Random Forest models for price prediction
- **Technical Analysis**: Pattern recognition (double top/bottom, breakouts, mean reversion)
- **Statistical Arbitrage**: Bollinger Bands and RSI-based signals
- **Signal Aggregation**: Combine multiple generators with weighted confidence, majority vote, or best confidence methods
- **Feature Engineering**: Comprehensive technical indicators and market microstructure features
- **Backtesting Framework**: Robust event-driven engine with realistic execution (slippage/fees) and performance metrics

## Installation

```bash
# Clone the repository
cd crypto-analysis

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from crypto_analysis.signals import (
    RandomForestSignalGenerator,
    TechnicalPatternGenerator,
    StatisticalArbitrageGenerator,
    SignalAggregator,
)

# Create sample data (or load your own)
from examples.basic_usage import create_sample_data
df = create_sample_data()

train_df = df[:int(len(df) * 0.7)]
test_df = df[int(len(df) * 0.7):]

# Create generators
rf_gen = RandomForestSignalGenerator(n_estimators=100)
rf_gen.fit(train_df)

tech_gen = TechnicalPatternGenerator()
tech_gen.fit(train_df)

stat_gen = StatisticalArbitrageGenerator()
stat_gen.fit(train_df)

# Create aggregator
aggregator = SignalAggregator(method="weighted_confidence")
aggregator.add_generator(rf_gen, weight=1.0)
aggregator.add_generator(tech_gen, weight=0.8)
aggregator.add_generator(stat_gen, weight=1.0)

# Generate signals
for i in range(100, len(test_df)):
    window = test_df.iloc[:i]
    all_signals = [g.generate(window) for g in aggregator.generators]
    final = aggregator.aggregate(all_signals)
    if final:
        print(f"{final.signal_type.value}: {final.confidence:.2f}")
```

## Project Structure

```
crypto-analysis/
├── src/crypto_analysis/
│   ├── signals/
│   │   ├── base.py              # SignalType, Signal, SignalGenerator
│   │   ├── features.py          # FeatureEngineer
│   │   ├── ml_generators.py    # LSTM & RandomForest generators
│   │   ├── technical.py        # TechnicalPatternGenerator
│   │   ├── statistical.py      # StatisticalArbitrageGenerator
│   │   ├── aggregator.py       # SignalAggregator
│   │   └── strategy.py         # Strategy, Portfolio, Orders
│   └── utils/
├── tests/
│   └── signals/
│       ├── test_base.py
│       ├── test_features.py
│       ├── test_generators.py
│       └── test_aggregator.py
├── examples/
│   └── basic_usage.py
├── pyproject.toml
└── README.md
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

### Linting

```bash
# Check with ruff
ruff check src/

# Auto-fix
ruff check --fix src/

# Format
ruff format src/

# Type checking
mypy src/
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

- **LSTMSignalGenerator**: LSTM-based sequence prediction with attention
- **RandomForestSignalGenerator**: Random Forest classifier with feature importance

### Technical Generators

- **TechnicalPatternGenerator**: Double top/bottom, breakouts, mean reversion

### Statistical Generators

- **StatisticalArbitrageGenerator**: Bollinger Bands and RSI extremes

## License

MIT
