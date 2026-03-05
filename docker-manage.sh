#!/bin/bash

# docker-manage.sh - Manage Docker Compose lifecycle for Redis and Worker
#
# Usage: ./docker-manage.sh [up|down|status|restart] [--build]
#
# Requirements:
# - docker-compose.redis.yml
# - docker-compose.worker.yml
# - docker-compose

set -e

# --- Configuration ---
REDIS_COMPOSE="docker-compose.redis.yml"
WORKER_COMPOSE="docker-compose.worker.yml"
REDIS_PROJECT="crypto-redis"
WORKER_PROJECT="crypto-worker"

# Helper for displaying help
usage() {
    echo "Usage: $0 [up|down|status|restart] [--build]"
    echo ""
    echo "Commands:"
    echo "  up      Start services (Redis first, then Worker). Optional: --build"
    echo "  down    Stop services (Worker first, then Redis)"
    echo "  status  Show status of all services"
    echo "  restart Restart all services. Optional: --build"
    exit 1
}

# --- Actions ---

start_services() {
    BUILD_FLAG=$1
    echo "[Info] Starting Redis infrastructure..."
    docker compose -p "$REDIS_PROJECT" -f "$REDIS_COMPOSE" up -d $BUILD_FLAG

    echo "[Info] Waiting for Redis to be healthy..."
    # Give it a moment for healthcheck to initialize
    sleep 2

    # Wait for redis healthcheck
    MAX_RETRIES=30
    COUNT=0
    while [ $COUNT -lt $MAX_RETRIES ]; do
        # Use project-prefixed container name or container_name if it's static
        # Since container_name is 'redis_broker' in the yml, we use that.
        STATUS=$(docker inspect -f '{{.State.Health.Status}}' redis_broker 2>/dev/null || echo "starting")
        if [ "$STATUS" == "healthy" ]; then
            echo "[Ok] Redis is healthy."
            break
        fi
        echo "[Wait] Redis is $STATUS. Retrying in 2 seconds ($((COUNT+1))/$MAX_RETRIES)..."
        sleep 2
        COUNT=$((COUNT+1))
    done

    if [ $COUNT -eq $MAX_RETRIES ]; then
        echo "[Error] Redis failed to become healthy in time."
        exit 1
    fi

    echo "[Info] Starting Worker infrastructure..."
    docker compose -p "$WORKER_PROJECT" -f "$WORKER_COMPOSE" up -d $BUILD_FLAG
    echo "[Success] All services started."
}

stop_services() {
    echo "[Info] Stopping Worker infrastructure..."
    docker compose -p "$WORKER_PROJECT" -f "$WORKER_COMPOSE" down

    echo "[Info] Stopping Redis infrastructure..."
    docker compose -p "$REDIS_PROJECT" -f "$REDIS_COMPOSE" down
    echo "[Success] All services stopped."
}

show_status() {
    echo "--- Redis Status ---"
    docker compose -p "$REDIS_PROJECT" -f "$REDIS_COMPOSE" ps
    echo ""
    echo "--- Worker Status ---"
    docker compose -p "$WORKER_PROJECT" -f "$WORKER_COMPOSE" ps
}

# --- Main ---

COMMAND=$1
shift || true
EXTRA_ARGS=$*

case "$COMMAND" in
    up)
        start_services "$EXTRA_ARGS"
        ;;
    down)
        stop_services
        ;;
    status)
        show_status
        ;;
    restart)
        stop_services
        start_services "$EXTRA_ARGS"
        ;;
    *)
        usage
        ;;
esac
