# AGENTS.md - Crypto Analysis Project

## Project Overview
Signal generation system integrating machine learning, technical analysis, and statistical methods for cryptocurrency trading.

## Build/Lint/Test Commands

### Testing
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_indicators.py

# Run single test function
pytest tests/test_indicators.py::test_rsi_calculation -v

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run tests matching pattern
pytest -k "test_rsi"
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
from src.indicators.base import Indicator
from src.signals.registry import SignalRegistry
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
│       ├── indicators/        # Technical indicators (RSI, MACD, etc.)
│       │   ├── __init__.py
│       │   ├── base.py        # Base class for indicators
│       │   └── rsi.py
│       ├── signals/           # Signal generation
│       │   ├── __init__.py
│       │   ├── registry.py    # Signal registry
│       │   └── combinator.py  # Combine multiple signals
│       ├── ml/                # ML models
│       ├── statistical/       # Statistical methods
│       └── utils/             # Utilities
├── tests/
│   ├── __init__.py
│   ├── indicators/
│   └── signals/
├── pyproject.toml
├── ruff.toml
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

### Indicator Implementation Pattern
```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class Indicator(ABC):
    """Base class for all technical indicators."""
    
    name: str
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate indicator values."""
        pass
    
    def validate(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns."""
        required = getattr(self, 'required_columns', [])
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

class RSIIndicator(Indicator):
    name = "rsi"
    required_columns = ["close"]
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate(data)
        close = data["close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

### Testing Guidelines
- Use `pytest` with `pytest-mock` for mocking
- Test file: `tests/indicators/test_rsi.py`
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
