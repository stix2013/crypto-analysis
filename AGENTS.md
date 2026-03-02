# AGENTS.md - Crypto Analysis Project

## Project Overview
Signal generation system integrating machine learning, technical analysis, and statistical methods for cryptocurrency trading. Features adaptive online learning, event-driven backtesting, and advanced risk management (SL/TP, volatility-adjusted sizing).

## Build/Lint/Test Commands

### Training & Simulation
```bash
# Run online learning training pipeline
./run_training.sh BTCUSDT 1h 5000

# Run prediction/inference with trained model
python scripts/predict.py model_btcusdt.joblib --symbol BTCUSDT --interval 1h
```

### Testing
```bash
# Run all tests
pytest

# Run backtester tests
pytest tests/signals/test_backtest.py

# Run online learning tests
pytest tests/online/

# Run tests with coverage
pytest --cov=src --cov-report=term-missing
```

### Linting & Formatting
```bash
# Run ruff linter
ruff check src/

# Auto-fix linting issues
ruff check --fix src/

# Format code with ruff
ruff format src/

# Run mypy type checking
mypy src/

# Full lint check (ruff + mypy)
ruff check src/ && mypy src/
```

### Development
```bash
# Install in editable mode
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit run --all-files
```

## Code Style Guidelines

### General
- Python 3.11+ required
- Use `ruff` for linting and formatting
- Type hints required for all function signatures
- Run `ruff check --fix` and `ruff format` before commits

### Imports
```python
# Standard library first
import os
import json
from typing import Optional, Union

# Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Local application
from crypto_analysis.indicators.base import Indicator
from crypto_analysis.signals.registry import SignalRegistry
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `RSIIndicator`, `SignalGenerator`)
- **Functions/variables**: `snake_case` (e.g., `calculate_signal`, `price_data`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_WINDOW_SIZE`, `MAX_LOOKBACK`)
- **Private methods**: prefix with `_` (e.g., `_compute_rolling_mean`)
- **Files**: `snake_case.py` (e.g., `signal_registry.py`, `test_indicators.py`)

### Type Hints
```python
# Use specific types, avoid Any
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series: ...

def get_signal(
    data: pd.DataFrame,
    config: dict[str, float]
) -> int:  # 1=buy, -1=sell, 0=hold

# Use Optional for nullable returns
def get_cached_indicator(name: str) -> Optional[Indicator]: ...
```

### Error Handling
```python
# Use custom exceptions for domain errors
class SignalError(Exception):
    """Base exception for signal generation errors."""
    pass

class InsufficientDataError(SignalError):
    """Raised when not enough data for calculation."""
    pass

# Handle gracefully with specific exceptions
def calculate_indicator(data: pd.Series) -> pd.Series:
    if len(data) < MIN_PERIODS:
        raise InsufficientDataError(
            f"Need at least {MIN_PERIODS} data points, got {len(data)}"
        )
    return compute_indicator(data)
```

### Project Structure
```
crypto-analysis/
├── src/
│   └── crypto_analysis/
│       ├── __init__.py
│       ├── data/              # Binance API client
│       ├── online/            # Adaptive online learning models & pipelines
│       │   ├── models/        # OnlineNN, OnlineLSTM, OnlineRF
│       │   └── detection/     # Regime & Adaptive Learning Rate
│       ├── signals/           # Core signal generation & aggregation
│       │   ├── features.py    # Feature engineering (RSI, MFI, etc.)
│       │   ├── ml_generators.py # LSTM and RF generators
│       │   ├── strategy.py    # Portfolio, Order, and MLStrategy
│       │   └── backtest.py    # Event-driven Backtester
│       └── utils/             # Performance analytics & optimization
│           ├── analytics.py   # Sharpe, Sortino, Equity curve plotting
│           └── optimization.py # Parameter grid search
├── tests/
│   ├── online/                # Tests for continuous learning pipeline
│   ├── signals/               # Tests for backtesting & signal logic
│   └── data/                  # Tests for Binance API fetching
├── scripts/
│   ├── train_online.py        # Core training CLI
│   └── predict.py             # Model inference CLI
├── run_training.sh            # Training wrapper script
├── pyproject.toml
└── AGENTS.md
```

## Agent Skills

### Crypto Trader Skill
The `crypto-trader` skill is available in this workspace. It provides specialized knowledge and workflows for:
- **Market Analysis**: Fetching and processing data from Binance.
- **Indicator Implementation**: Creating new technical indicators following the project's patterns.
- **Strategy Development**: Building and aggregating signal generators.
- **Backtesting**: Running simulations using the `PortfolioManager`.
- **Online Learning**: Utilizing adaptive models for real-time updates.

To enable this skill in your session, run:
```bash
/skills reload
```

### Signal Generator Implementation Pattern
```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

from crypto_analysis.signals.base import Signal, SignalGenerator, SignalType

class TechnicalIndicatorGenerator(SignalGenerator):
    """Example implementation of a technical signal generator."""
    
    def __init__(self, name: str, rsi_period: int = 14):
        super().__init__(name, lookback_period=rsi_period + 50)
        self.rsi_period = rsi_period
    
    def fit(self, data: pd.DataFrame) -> None:
        # Technical generators often don't need fitting
        self.is_fitted = True
    
    def generate(self, data: pd.DataFrame, current_position: Optional[float] = None) -> list[Signal]:
        if len(data) < self.lookback_period:
            return []
            
        # Calculation logic using vectorized pandas/numpy
        # ...
        
        return [Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.ENTRY_LONG,
            confidence=0.8,
            timestamp=data.index[-1],
            source=self.name
        )]

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # Return features used for generation
        return pd.DataFrame(...)
```

### Testing Guidelines
- Use `pytest` with `pytest-mock` for mocking
- Test file: `tests/signals/test_rsi.py`
- Test class: `TestRSIIndicator`
- Test function: `test_rsi_values_correct`
- Use `pandas.testing.assert_series_equal` for Series comparisons
- Include edge case tests (empty data, single value, NaN handling)

### Data Handling
- Use `pd.Series` and `pd.DataFrame` for all data
- Validate inputs at function boundaries
- Return clean Series with proper index
- Handle NaN values explicitly (drop, fill, or propagate)
- Use `pd.Timestamp` for dates, not raw strings

### Performance Considerations
- Use vectorized operations over loops
- Use `numba` for hot paths if needed
- Cache computed indicators
- Use `np.ndarray` for numerical arrays
- Lazy evaluation for expensive computations
