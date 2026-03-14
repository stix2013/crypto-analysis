import logging
import time
from typing import Any

import requests

from crypto_analysis.settings import get_settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def send_webhook(
    task_id: str,
    task_name: str,
    status: str,
    symbol: str | None = None,
    interval: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    """Send webhook notification to the main API with retry logic.

    Args:
        task_id: Celery task ID
        task_name: Celery task name
        status: Task status (e.g., 'SUCCESS', 'FAILURE')
        symbol: Optional symbol
        interval: Optional interval
        result: Optional result data
        error: Optional error message

    Returns:
        True if webhook was sent successfully, False otherwise
    """
    settings = get_settings()
    webhook_url = settings.webhook_url
    if not webhook_url:
        logger.debug("WEBHOOK_URL not configured, skipping webhook")
        return False

    payload = {
        "task_id": task_id,
        "task_name": task_name,
        "status": status,
        "symbol": symbol,
        "interval": interval,
        "result": result,
        "error": error,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.ok:
                logger.info(
                    f"Webhook sent successfully for task {task_name} ({task_id})"
                )
                return True
            logger.warning(
                f"Webhook attempt {attempt} failed with status {response.status_code}: {task_name}"
            )
            if response.status_code == 422:
                logger.error(f"Validation error details: {response.text}")
        except Exception as e:
            logger.warning(f"Webhook attempt {attempt} error: {e}")

        if attempt < MAX_RETRIES:
            sleep_time = 2**attempt
            logger.info(f"Retrying webhook in {sleep_time} seconds...")
            time.sleep(sleep_time)

    logger.error(f"Webhook failed after {MAX_RETRIES} attempts: {task_name}")
    return False
