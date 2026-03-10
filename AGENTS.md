# AGENTS.md - Crypto Analysis Project

## Project Overview
Signal generation system integrating machine learning, technical analysis, and statistical methods for cryptocurrency trading. Features adaptive online learning (PyTorch CPU-only), event-driven backtesting, and advanced risk management (SL/TP, volatility-adjusted sizing).

## Build/Lint/Test Commands

### Training & Simulation
```bash
# Run online learning training pipeline
# Outputs saved to ./signals/ and ./models/
./run_training.sh BTCUSDT 1h 5000

# Run prediction/inference with trained model
# Using resolved path: models/model_btcusdt_1h.joblib
python scripts/predict.py BTCUSDT --interval 1h
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
│   ├── data/                  # Tests for Binance API fetching
│   └── test_performance.py    # Performance & scaling benchmarks
├── scripts/
│   ├── train_online.py        # Core training CLI
│   └── predict.py             # Model inference CLI
├── signals/                   # Generated trading signals (CSV)
├── models/                    # Trained model checkpoints (joblib)
├── docker-compose.data.yml # Redis broker infrastructure
├── docker-compose.worker.yml # Celery worker infrastructure
├── docker-manage.sh    # Management script for Data/Worker
├── run_training.sh     # Training wrapper script
├── pyproject.toml
├── worker/
│   ├── celery_app.py          # Celery app configuration
│   ├── tasks.py               # Shared Celery tasks (using @shared_task)
│   ├── requirements.txt       # Worker-specific dependencies (PyTorch CPU)
│   └── Dockerfile             # Multi-stage build for Celery worker
└── AGENTS.md

```

## Celery Worker Architecture

### Overview
The system uses Celery for asynchronous task processing, including data fetching, model training, and backtesting. The worker runs in a Docker container using a multi-stage build optimized for PyTorch (CPU-only).

### Key Components
- **Broker/Backend**: Redis (`redis://redis:6379/0`)
- **App Instance**: Defined in `worker/celery_app.py`.
- **Tasks**: Defined in `worker/tasks.py` using `@shared_task` to avoid circular dependencies with the app instance.
- **Environment**: Configured via `worker/.env` and `docker-compose.worker.yml`.

### Best Practices for Tasks
- **Decoupling**: Always use `@shared_task` instead of `@app.task` to prevent circular imports between the app configuration and task definitions.
- **Type Safety**: Explicitly cast numeric arguments (e.g., `int(bars)`) at the start of the task. CLI/External triggers often pass strings.
- **Orchestration**: To run logic from one task within another synchronously, refactor the core logic into a separate Python function (e.g., `_task_logic_core`) and call that function from both tasks. **Never use `.get()` or `.apply().get()` inside a task**, as it triggers Celery's blocking safety checks.
- **Module Imports**: Import library code from `crypto_analysis.*` directly. The Docker environment sets `PYTHONPATH=/app` to enable this.
- **ML Compatibility**: ML models include `TORCH_AVAILABLE` guards. Workers will skip ML updates if PyTorch is not fully initialized.
- **Pathing**: Use `/app/signals` and `/app/models` for persistent storage, mapped to Docker volumes.

### Running the Services
The project uses a management script to orchestrate both Data (Redis) and Worker (Celery) infrastructure.

```bash
# Start all services (Redis + 1 Worker)
./docker-manage.sh up

# Start with multiple worker instances
./docker-manage.sh up --workers 3

# Check status of all services
./docker-manage.sh status

# Stop all services
./docker-manage.sh down
```

### Manual Docker Management (Advanced)
If needed, you can manage the infrastructure components separately:
- **Data (Redis)**: `docker compose -f docker-compose.data.yml`
- **Worker (Celery)**: `docker compose -f docker-compose.worker.yml`

Access Flower (Monitoring) at [http://localhost:5555](http://localhost:5555) when the worker is running.

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
- **Vectorized Operations**: Use NumPy and Pandas vectorized operations over loops. Closed-form solutions (e.g., for linear regression) are preferred over iterative `apply()` methods.
- **Complexity Management**: Ensure training loops scale linearly $O(N)$. Pre-calculate features once for the entire dataset before entering iterative simulation loops.
- **Benchmarking**: Use `tests/test_performance.py` to verify that feature calculation and training logic maintain acceptable performance as data size increases.
- **Lazy Evaluation**: Use lazy evaluation for expensive computations and cache computed indicators where possible.
- **Resource Usage**: ML models use PyTorch CPU-only to minimize infrastructure requirements while maintaining high inference speed.
