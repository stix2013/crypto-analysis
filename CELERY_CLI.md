# Celery Worker CLI Documentation

This document provides instructions on how to interact with the Celery worker in the `crypto-analysis` project using the command line.

## Prerequisites

1.  **Redis** and **Worker** services must be running. Use the management script:
    ```bash
    ./docker-manage.sh up
    ```
2.  **Environment Variables**: Ensure you have a `.env` file in the project root or set the required variables.
    - `CELERY_BROKER_URL`: `redis://localhost:6379/0` (host) or `redis://redis:6379/0` (Docker)
    - `CELERY_RESULT_BACKEND`: Same as broker.
    - `CELERY_LOG_LEVEL`: defaults to `info`.

---

## 1. Monitoring & Observability

### Worker Status

Check if workers are online and responding:

```bash
# Within Docker
docker exec -it crypto-worker-worker-1 celery -A celery_app status

# From Host (requires venv active)
celery -A worker.celery_app status
```

### Viewing Logs (Crucial for Debugging)

To see real-time output from the worker (training progress, errors, etc.):

```bash
docker logs -f crypto-worker-worker-1
```

### Flower (Web UI)

Flower provides a powerful web interface for monitoring tasks, workers, and queues.

- **URL**: [http://localhost:5555](http://localhost:5555)
- **Key Features**:
  - **Task Progress**: Watch tasks as they execute.
  - **Worker Stats**: View CPU/Memory usage per worker.
  - **Broker**: Inspect the Redis queues.
  - **History**: Review past task results and execution times.

---

## 2. Triggering Tasks via CLI

Use `celery call` to trigger tasks. Adding `--wait` will make the command block until the task is complete and print the result.

### Example: Fetch Market Data

```bash
# Asynchronous (returns task ID immediately)
docker exec -it crypto-worker-worker-1 celery -A celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]'

# Synchronous (waits for result)
docker exec -it crypto-worker-worker-1 celery -A celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]' --wait
```

### Example: Full Training Pipeline

```bash
docker exec -it crypto-worker-worker-1 celery -A celery_app call train_model --args='["ETHUSDT"]' --kwargs='{"interval": "15m", "bars": 5000}' --wait
```

### Example: Orchestrated Workflow

Trigger both training and backtesting in one go:

```bash
docker exec -it crypto-worker-worker-1 celery -A celery_app call train_and_backtest --args='["BTCUSDT", "1h", 2000]' --wait
```

---

## 3. Task Management & Inspection

| Command                                        | Description                                         |
| :--------------------------------------------- | :-------------------------------------------------- |
| `celery -A worker.celery_app inspect active`   | List currently executing tasks.                      |
| `celery -A worker.celery_app inspect reserved` | List tasks waiting in the queue.                    |
| `celery -A worker.celery_app inspect stats`    | Detailed worker internal statistics.                |
| `celery -A worker.celery_app result <task_id>` | Fetch the result/status of a specific task.         |
| `celery -A worker.celery_app purge`            | **WARNING**: Clears all pending tasks from queues.  |
| `celery -A worker.celery_app control terminate <task_id>` | Force stop a running task.                |

---

## 4. Available Tasks Reference

The following tasks are defined in `worker/tasks.py`:

| Task Name            | Arguments                                                                    | Description                                      |
| :------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------- |
| `fetch_market_data`  | `symbol`, `interval`, `bars`                                                 | Fetches OHLCV data from Binance.                 |
| `train_model`        | `symbol`, `interval`, `bars`, `warmup_bars`, `sequence_length`, `output_dir`, `model_dir` | Trains online model and saves signals/generator. |
| `run_prediction`     | `model_path`, `symbol`, `interval`, `bars`                                   | Generates real-time signals from saved model.   |
| `run_backtest`       | `signals_path`, `symbol`, `interval`, `initial_capital`, `commission`        | Evaluates signal performance on historical data. |
| `train_and_backtest` | `symbol`, `interval`, `bars`, `warmup_bars`                                  | Sequential pipeline for training and validation. |

---

## 5. Components Overview

- **Worker**: Executes the actual Python code for data fetching and ML.
- **Beat**: A scheduler that triggers periodic tasks. Currently running but requires a `beat_schedule` configuration in `celery_app.py` to be active.
- **Flower**: Web-based monitoring tool.
- **Redis**: Acts as both the **Message Broker** (transporting tasks) and **Result Backend** (storing results).

---

## 6. Data Locations

By default, the worker saves artifacts to the following locations inside the container, which are persisted to Docker volumes:

- **Signals (CSVs)**: `/app/signals` (Volume: `trade_signal`)
- **Models (.joblib)**: `/app/models` (Volume: `trade_model`)

**Note:** You can override these locations by passing `output_dir` (for signals) or `model_dir` (for models) to the `train_model` task.

---

## 7. Troubleshooting

- **Redis Connection Error**: Check if Redis is up with `./docker-manage.sh status`. If running from host, ensure `CELERY_BROKER_URL` points to `localhost`.
- **Worker Hangs on Training**: Training large models or fetching massive data can take time. Use `docker logs` to verify progress or Flower to check if the worker is still "active".
- **ModuleNotFoundError**: When running from host, ensure you have run `pip install -e .` or set `PYTHONPATH` correctly as shown in Section 1.
- **Data Persistence**: Signals and models are saved to Docker volumes (`trade_signal` and `trade_model`). Check `docker-compose.worker.yml` for exact mount points.
