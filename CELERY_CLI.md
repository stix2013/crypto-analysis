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
    - `WEBHOOK_URL`: URL for task result callbacks (e.g., `http://host.docker.internal:8000/api/tasks/webhook/celery-callback`).

---

## 1. Monitoring & Observability

### Worker Status

Check if workers are online and responding:

```bash
# Within Docker
docker exec -it analysis-worker-worker-1 celery -A celery_app status

# From Host (requires venv active)
celery -A worker.celery_app status
```

### Health Checks

All worker services have integrated health checks:

- **Worker**: Uses `celery inspect ping`.
- **Flower**: Monitors the `/healthcheck` HTTP endpoint.
- **Beat**: Verified via process monitoring (`pgrep`).
- **Redis**: Uses `redis-cli ping`.

Check overall health status:
```bash
docker compose -p analysis-worker -f docker-compose.worker.yml ps
```

### Viewing Logs (Crucial for Debugging)

To see real-time output from the worker (training progress, errors, webhooks):

```bash
docker logs -f analysis-worker-worker-1
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
docker exec -it analysis-worker-worker-1 celery -A celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]'

# Synchronous (waits for result)
docker exec -it analysis-worker-worker-1 celery -A celery_app call fetch_market_data --args='["BTCUSDT", "1h", 100]' --wait
```

### Example: Full Training Pipeline

```bash
docker exec -it analysis-worker-worker-1 celery -A celery_app call train_model --args='["ETHUSDT"]' --kwargs='{"interval": "15m", "bars": 5000}' --wait
```

### Example: Orchestrated Workflow

Trigger both training and backtesting in one go:

```bash
docker exec -it analysis-worker-worker-1 celery -A celery_app call train_and_backtest --args='["BTCUSDT", "1h", 2000]' --wait
```

---

## 3. Webhook Integration

Tasks automatically notify a configured `WEBHOOK_URL` upon completion. The payload follows the `CeleryCallbackRequest` schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `string` | Unique Celery task ID |
| `task_name` | `string` | Name of the task |
| `status` | `string` | `SUCCESS` or `FAILURE` |
| `symbol` | `string` | (Optional) Trading symbol |
| `interval` | `string` | (Optional) Timeframe |
| `result` | `object` | (Optional) Task-specific result data |
| `error` | `string` | (Optional) Error message if failed |

---

## 4. Task Management & Inspection

| Command                                        | Description                                         |
| :--------------------------------------------- | :-------------------------------------------------- |
| `celery -A worker.celery_app inspect active`   | List currently executing tasks.                      |
| `celery -A worker.celery_app inspect reserved` | List tasks waiting in the queue.                    |
| `celery -A worker.celery_app inspect stats`    | Detailed worker internal statistics.                |
| `celery -A worker.celery_app result <task_id>` | Fetch the result/status of a specific task.         |
| `celery -A worker.celery_app purge`            | **WARNING**: Clears all pending tasks from queues.  |
| `celery -A worker.celery_app control terminate <task_id>` | Force stop a running task.                |

---

## 5. Available Tasks Reference

The following tasks are defined in `worker/tasks.py`. All numeric arguments (bars, capital, etc.) support both string and integer inputs from the CLI/API.

| Task Name            | Arguments                                                                    | Description                                      |
| :------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------- |
| `fetch_market_data`  | `symbol`, `interval`, `bars`                                                 | Fetches OHLCV data from Binance.                 |
| `train_model`        | `symbol`, `interval`, `bars`, `warmup_bars`, `sequence_length`, `output_dir`, `model_dir` | Trains online model and saves signals/generator. |
| `run_prediction`     | `model_path`, `symbol`, `interval`, `bars`                                   | Generates real-time signals from saved model.   |
| `run_backtest`       | `signals_path`, `symbol`, `interval`, `initial_capital`, `commission`        | Evaluates signal performance on historical data. |
| `train_and_backtest` | `symbol`, `interval`, `bars`, `warmup_bars`                                  | Sequential pipeline for training and validation. |

---

## 6. Components Overview

- **Worker**: Executes the actual Python code for data fetching and ML.
- **Beat**: A scheduler that triggers periodic tasks. Includes health monitoring via process checks.
- **Flower**: Web-based monitoring tool with integrated health check.
- **Redis**: Acts as both the **Message Broker** (transporting tasks) and **Result Backend** (storing results).

---

## 7. Data Locations

By default, the worker saves artifacts to the following locations inside the container, which are persisted to Docker volumes:

- **Signals (CSVs)**: `/app/signals` (Volume: `trade_signal`)
- **Models (.joblib)**: `/app/models` (Volume: `trade_model`)

---

## 8. Troubleshooting

- **Code Not Updating in Container**: Worker code is copied into the image during the build. If changes on the host aren't reflecting in the worker logs, run `./docker-manage.sh restart --build`.
- **Webhook Connection Refused**: If the API is on the host, use `http://host.docker.internal:PORT`. Ensure `extra_hosts` is configured in `docker-compose.worker.yml`.
- **422 Validation Error**: Check `worker/webhook.py` logs for specific validation details. Ensure the payload matches the expected Pydantic schema on the receiver.
- **TypeError in Tasks**: Tasks now explicitly convert numeric inputs to `int`. Ensure `bars` or `warmup_bars` are valid numbers. Orchestrated tasks (like `train_and_backtest`) use shared core functions to avoid `RuntimeError` from `.get()`.
- **Redis Connection Error**: Check if Redis is up with `./docker-manage.sh status`.
