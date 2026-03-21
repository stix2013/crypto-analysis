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

### Environment Variables
All environment variables in the project should be accessed through the centralized settings system in `src/crypto_analysis/settings.py`. Direct access to environment variables via `os.getenv()` or `os.environ.get()` should be avoided.

For example, variables like `ENABLE_BACKTEST_PLOTS` and `BACKTEST_INITIAL_CAPITAL` are accessed through the settings system (`settings.backtest.enable_plots` and `settings.backtest.initial_capital`) rather than directly from the environment.

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

### Debugging Model Issues

When prediction signals are degenerate (all same prediction or repeating values):

1. **Check RF buffer diversity**: Load model and verify each tree's sample buffer has different label distributions
   ```python
   import joblib
   model = joblib.load('models/model_xxx.joblib')
   for i, buf in enumerate(model.rf.sample_buffers):
       labels = [s[1] for s in buf]
       print(f'Buffer {i}: {sum(labels)} ones out of {len(labels)}')
   ```

2. **Verify tree predictions vary**: Test individual tree predictions on different inputs
   ```python
   for tree in model.rf.trees:
       print(tree.predict(test_point))
   ```

3. **Common bug**: `OnlineRandomForest.partial_fit()` must use Poisson sampling (random bagging), not add all samples to all trees. See `src/crypto_analysis/online/models/online_rf.py`.

4. **Data leakage in online updates**: The `_online_update()` method computes `actual_direction` from PAST returns but the model predicts FUTURE returns. This mismatch causes degenerate behavior. Fixed by adding `enable_online_update` parameter to disable online updates during training/inference.

5. **Retrain after fixing**: Any models trained with buggy code must be retrained.

### Online Learning Configuration

The `OnlineSignalGenerator` class (in `src/crypto_analysis/online/generator.py`) supports:

- **`enable_online_update`** (default `True`): When `False`, skips online model updates. Should be disabled during training and prediction to prevent data leakage.
- **`random_seed`** (default 42): Ensures reproducible results across runs.
- **`actual_direction`** (optional): Pre-computed actual direction for online updates, bypassing the leaky computation.

## Workflow Orchestration

1. **Document Plans**: All plans must be written down in the `tasks` folder.
### 2. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 3. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to "tasks/todo.md" with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md"
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
The following Python libraries are available:
...
