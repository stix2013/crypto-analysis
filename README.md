# Crypto Analysis

Signal generation system integrating machine learning, technical analysis, and statistical methods for cryptocurrency trading. Uses PyTorch (CPU-only) for adaptive online learning.

## Features

- **Adaptive Online Learning**: Continuous model updates with real-time market data, regime detection, and adaptive learning rates.
- **ML-based Generators**: PyTorch-based LSTM and Random Forest models for price prediction and classification.
- **Technical Analysis**: Pattern recognition (double top/bottom, breakouts, mean reversion) and comprehensive feature engineering (RSI, MFI, Choppiness Index).
- **Signal Aggregation**: Combine multiple generators with weighted confidence, majority vote, or best confidence methods.
- **Risk Management**: Integrated Stop-Loss (SL) and Take-Profit (TP) triggers, and dynamic volatility-adjusted position sizing.
- **Backtesting & Optimization**: Robust event-driven engine with realistic execution, performance metrics (Sharpe/Sortino), and parameter grid search.
- **High Performance**: Optimized $O(N)$ training loops and vectorized feature engineering (slope, R2) for high-frequency signal generation.
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
*Note: Signals are saved in `signals/signals_{symbol}_{interval}.csv` and models in `models/model_{symbol}_{interval}.joblib`.*

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

### 3. Using Celery Worker
For distributed task processing (fetching data, training models, backtesting), the project uses Celery with Redis.

#### Infrastructure Components
- **Data (`docker-compose.data.yml`)**: Provides Redis as the message broker.
- **Worker (`docker-compose.worker.yml`)**: Runs the Celery worker, beat scheduler, and Flower monitor.

#### Management Script (Recommended)
Use `./docker-manage.sh` to orchestrate both Data and Worker services seamlessly:

```bash
# Start all services (Redis + 1 Worker)
./docker-manage.sh up

# Start with multiple worker instances
./docker-manage.sh up --workers 3

# Check status of all services
./docker-manage.sh status

# Rebuild and restart
./docker-manage.sh restart --build

# Stop all services
./docker-manage.sh down
```

#### Environment Configuration
Ensure you have a `.env` file in the project root (see `worker/.env.example` for reference).
- **Broker**: `redis://localhost:6379/0` (from host) or `redis://redis:6379/0` (within Docker).

#### Monitoring the Worker
- **Using Docker (Recommended)**: `docker exec -it crypto-worker-worker-1 celery -A celery_app status`
- **Using Host Machine**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/worker
  export CELERY_BROKER_URL=redis://localhost:6379/0
  celery -A worker.celery_app status
  ```
- **Using Flower (Web UI)**: [http://localhost:5555](http://localhost:5555) (available when running worker services).

#### Triggering Tasks via CLI
You can trigger tasks using the `celery call` command:
```bash
# Example: Fetch Market Data
docker exec -it crypto-worker-worker-1 celery -A celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]'

# Example: Train Model
docker exec -it crypto-worker-worker-1 celery -A celery_app call train_model --args='["ETHUSDT"]' --kwargs='{"interval": "15m", "bars": 5000}'

# Example: Run Prediction (looks for model_{symbol}_{interval}.joblib if first arg is symbol)
docker exec -it crypto-worker-worker-1 celery -A celery_app call run_prediction --args='["ETHUSDT", "ETHUSDT", "15m", 200]'

# Example: Train and Backtest
docker exec -it crypto-worker-worker-1 celery -A celery_app call train_and_backtest --args='["BTCUSDT"]' --kwargs='{"interval": "15m", "bars": 2000}'
```

#### Available Tasks Reference
The following tasks are defined in `worker/tasks.py`:

| Task Name | Arguments | Description |
| :--- | :--- | :--- |
| `fetch_market_data` | `symbol`, `interval`, `bars` | Fetches historical data from Binance. |
| `train_model` | `symbol`, `interval`, `bars`, `warmup_bars`, `sequence_length`, `output_dir` | Trains an online learning model. |
| `run_prediction` | `model_path`, `symbol`, `interval`, `bars` | Generates signals using a trained model. |
| `run_backtest` | `signals_path`, `symbol`, `interval`, `initial_capital`, `commission` | Runs a backtest from a signals CSV. |
| `train_and_backtest`| `symbol`, `interval`, `bars`, `warmup_bars` | Orchestrates training and backtesting. |

For more detailed commands, see the full CLI reference in [CELERY_CLI.md](./CELERY_CLI.md).

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
├── docker-compose.data.yml # Data infrastructure (Redis)
├── docker-compose.worker.yml # Worker infrastructure (Celery)
├── docker-manage.sh    # Multi-container management script
├── .env.example        # Environment template
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
