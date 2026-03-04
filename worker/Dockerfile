FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt /build/requirements.txt

RUN pip install --no-cache-dir --prefix=/install  --extra-index-url https://download.pytorch.org/whl/cpu -r /build/requirements.txt


FROM python:3.11-slim AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# Copy worker files directly to WORKDIR
COPY worker/tasks.py worker/celery_app.py worker/__init__.py ./

# Copy the crypto_analysis package from src
COPY src/crypto_analysis ./crypto_analysis

ENV PYTHONPATH=/app

CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info"]
