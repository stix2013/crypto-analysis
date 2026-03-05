# Celery Worker CLI Documentation

This document provides instructions on how to interact with the Celery worker in the `crypto-analysis` project using the command line.

## Prerequisites

1.  **Redis** and **Worker** services must be running. Use the management script:
    ```bash
    ./docker-manage.sh up
    ```
2.  **Environment Variables**: Ensure you have a `.env` file in the project root or set the required variables (e.g., `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`).
    - Defaults: `redis://localhost:6379/0` (from host) or `redis://redis:6379/0` (within Docker).

---

## 1. Monitoring the Worker

### Using Docker (Recommended)
You can inspect the worker status directly from the container:
```bash
docker exec -it crypto-worker-worker-1 celery -A celery_app status
```

### Using Host Machine
Ensure you have the requirements installed (`pip install -r requirements.txt`) and set your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/worker
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0

celery -A worker.celery_app status
```

---

## 2. Triggering Tasks via CLI

You can trigger tasks using the `celery call` command.

### Example: Fetch Market Data
```bash
# Within Docker
docker exec -it crypto-worker-worker-1 celery -A celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]'

# From Host
celery -A worker.celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]'
```

### Example: Train Model
```bash
docker exec -it crypto-worker-worker-1 celery -A celery_app call train_model --args='["ETHUSDT"]' --kwargs='{"interval": "15m", "bars": 5000}'
```

### Example: Run Prediction
```bash
# Note: Ensure the model path is accessible to the worker container
docker exec -it crypto-worker-worker-1 celery -A celery_app call run_prediction --args='["/app/signals/model_ethusdt.joblib", "ETHUSDT"]'
```

---

## 3. Available Tasks Reference

The following tasks are defined in `worker/tasks.py`:

| Task Name | Arguments | Description |
| :--- | :--- | :--- |
| `fetch_market_data` | `symbol`, `interval`, `bars` | Fetches historical data from Binance. |
| `train_model` | `symbol`, `interval`, `bars`, `warmup_bars`, `sequence_length`, `output_dir` | Trains an online learning model. |
| `run_prediction` | `model_path`, `symbol`, `interval`, `bars` | Generates signals using a trained model. |
| `run_backtest` | `signals_path`, `symbol`, `interval`, `initial_capital`, `commission` | Runs a backtest from a signals CSV. |
| `train_and_backtest`| `symbol`, `interval`, `bars`, `warmup_bars` | Orchestrates training and backtesting. |

---

## 4. Useful Commands

| Command | Description |
| :--- | :--- |
| `celery -A worker.celery_app inspect ping` | Ping the worker. |
| `celery -A worker.celery_app inspect active` | List active tasks. |
| `celery -A worker.celery_app inspect stats` | Get worker statistics. |
| `celery -A worker.celery_app result <task_id>` | Get the result of a specific task. |
| `celery -A worker.celery_app purge` | Clear all messages from all configured task queues. |

---

## 5. Troubleshooting

- **Redis Connection Error**: Ensure the Redis container is running and healthy. Check with `./docker-manage.sh status`.
- **ModuleNotFoundError**: Ensure `PYTHONPATH` includes `src` and `worker` directories if running from the host.
- **Permission Denied**: If the worker cannot write to `/app/signals` or `/app/models`, ensure the Docker volumes are correctly mounted and permissions are set.
