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
*Note: Signals are saved in signals/signals_{symbol}_{interval}.csv and models in models/model_{symbol}_{interval}.joblib.*

### 2. Prediction (New Simple API)

#### Command Line
```bash
# Uses PREDICT_SYMBOL and PREDICT_INTERVAL from .env
python scripts/predict.py

# Specify symbol and interval (resolves to models/model_btcusdt_1h.joblib)
python scripts/predict.py BTCUSDT -i 1h

# Use explicit model path
python scripts/predict.py --model models/model_ethusdt_15m.joblib

# Save signals to CSV (defaults to predict_ethusdt_15m.csv if -o omitted)
python scripts/predict.py ETHUSDT -i 15m -o signals.csv
```

#### Python API
```python
from crypto_analysis.signals.predict import predict, Predictor

# One-liner prediction (auto-fetches data from Binance)
signals = predict("BTCUSDT", interval="1h")
signals = predict("ETHUSDT")  # Uses PREDICT_INTERVAL from .env
signals = predict(model_path="models/custom.joblib")  # Explicit path

# Reusable predictor (efficient for multiple calls)
predictor = Predictor("BTCUSDT", interval="1h")
signals = predictor.predict()
signals = predictor.predict(data=my_dataframe)  # With custom data
```

## Troubleshooting

### Degenerate Predictions
If predictions are all the same (e.g., all 1s or all 0s), this indicates data leakage in the online learning pipeline. The system has been fixed to:
- Disable online updates during training to prevent contamination
- Disable online updates during prediction/inference
- Use approximate timestamp matching in backtests

**If you have old models trained before the fix, retrain them with the corrected code.**

## Using Celery Worker
For distributed task processing (fetching data, training models, backtesting), the project uses Celery with Redis.

### Infrastructure Components
- **Data (`docker-compose.data.yml`)**: Provides Redis as the message broker.
- **Worker (`docker-compose.worker.yml`)**: Runs the Celery worker, beat scheduler, and Flower monitor.

### Management Script (Recommended)
Use `./docker-manage.sh` to orchestrate both Data and Worker services seamlessly:

```bash
# Start all services (Redis + 1 Worker)
./docker-manage.sh up

# Start with multiple worker instances
./docker-manage.sh up --workers 3

# Check health of all services
./docker-manage.sh status

# Rebuild and restart
./docker-manage.sh restart --build
```

### Monitoring the Worker
- **Using Docker**: `docker compose -p analysis-worker -f docker-compose.worker.yml ps` (Checks health status)
- **Using Logs**: `docker logs -f analysis-worker-worker-1`
- **Using Flower (Web UI)**: [http://localhost:5555](http://localhost:5555)

### Webhook Notifications
The worker can be configured to send task results via webhook. Set `WEBHOOK_URL` in your `.env`:
```bash
WEBHOOK_URL="http://host.docker.internal:8000/api/tasks/webhook/celery-callback"
```

### Triggering Tasks via CLI
You can trigger tasks using the `celery call` command:
```bash
# Example: Fetch Market Data
docker exec -it analysis-worker-worker-1 celery -A celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]'

# Example: Train Model
docker exec -it analysis-worker-worker-1 celery -A celery_app call train_model --args='["ETHUSDT"]' --kwargs='{"interval": "15m", "bars": 5000}'

# Example: Run Prediction
docker exec -it analysis-worker-worker-1 celery -A celery_app call run_prediction --args='["models/model_btcusdt_1h.joblib", "BTCUSDT", "1h", 200]'

# Example: Train and Backtest
docker exec -it analysis-worker-worker-1 celery -A celery_app call train_and_backtest --args='["BTCUSDT", "1h", 2000]'
```

#### Available Tasks Reference
The following tasks are defined in `worker/tasks.py`:

| Task Name | Arguments | Description |
| :--- | :--- | :--- |
| `fetch_market_data` | `symbol`, `interval`, `bars` | Fetches historical data from Binance. |
| `train_model` | `symbol`, `interval`, `bars`, `warmup_bars` | Trains an online learning model. |
| `run_prediction` | `model_path`, `symbol`, `interval`, `bars` | Generates signals using a trained model. |
| `run_backtest` | `signals_path`, `symbol`, `interval` | Runs a backtest from a signals CSV. |
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
├── pyproject.toml
└── README.md
```

## License

MIT
