FROM python:3.14-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt /build/requirements.txt

# Strip any GPU/CUDA dependencies and install torch+cpu/flower explicitly to ensure smallest image size
# torch --index-url https://download.pytorch.org/whl/cpu \
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.14-slim AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# Copy worker files directly to WORKDIR
COPY worker/tasks.py worker/celery_app.py ./

# Copy the crypto_analysis package from src
COPY src/crypto_analysis ./crypto_analysis

ENV PYTHONPATH=/app

CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info"]
