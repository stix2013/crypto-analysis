import logging
import os
import time
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
MAX_RETRIES = 3


def send_webhook(event_type: str, data: dict[str, Any]) -> bool:
    """Send webhook notification with retry logic (max 3 attempts, exponential backoff).

    Args:
        event_type: The event name (e.g., 'training_complete', 'prediction_complete')
        data: The payload data to send

    Returns:
        True if webhook was sent successfully, False otherwise
    """
    if not WEBHOOK_URL:
        logger.debug("WEBHOOK_URL not configured, skipping webhook")
        return False

    payload = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
            if response.ok:
                logger.info(f"Webhook sent successfully: {event_type}")
                return True
            logger.warning(
                f"Webhook attempt {attempt} failed with status {response.status_code}: {event_type}"
            )
        except Exception as e:
            logger.warning(f"Webhook attempt {attempt} error: {e}")

        if attempt < MAX_RETRIES:
            sleep_time = 2**attempt
            logger.info(f"Retrying webhook in {sleep_time} seconds...")
            time.sleep(sleep_time)

    logger.error(f"Webhook failed after {MAX_RETRIES} attempts: {event_type}")
    return False
