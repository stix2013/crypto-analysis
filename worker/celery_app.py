from celery import Celery

from crypto_analysis.settings import get_settings

settings = get_settings()

app = Celery(
    "crypto_analysis",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=["tasks"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=settings.celery.task_track_started,
    task_time_limit=settings.celery.task_time_limit,
    task_soft_time_limit=settings.celery.task_soft_time_limit,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

if __name__ == "__main__":
    app.start()
