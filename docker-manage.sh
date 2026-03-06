#!/bin/bash

# docker-manage.sh - Manage Docker Compose lifecycle for Redis and Worker
#
# Usage: ./docker-manage.sh [up|down|status|restart] [--build] [--workers N]
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
    echo "Usage: $0 [up|down|status|restart] [--build] [--workers N]"
    echo ""
    echo "Commands:"
    echo "  up      Start services (Redis first, then Worker)."
    echo "          Options:"
    echo "            --build       Rebuild images before starting"
    echo "            --workers N   Scale worker service to N instances"
    echo "  down    Stop services (Worker first, then Redis)"
    echo "  status  Show status of all services"
    echo "  restart Restart all services. Supports --build and --workers."
    exit 1
}

# --- Actions ---

start_services() {
    BUILD_FLAG=""
    SCALE_FLAG=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_FLAG="--build"
                shift
                ;;
            --workers)
                if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                    SCALE_FLAG="--scale worker=$2"
                    shift 2
                else
                    echo "Error: --workers requires a number"
                    exit 1
                fi
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    echo "[Info] Starting Redis infrastructure..."
    # Redis doesn't support scaling workers, so only pass build flag
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
    # Pass both build and scale flags to worker compose
    # Note: We must be careful with variable expansion.
    # Using 'eval' or just relying on word splitting for simple flags works here.
    CMD="docker compose -p \"$WORKER_PROJECT\" -f \"$WORKER_COMPOSE\" up -d $BUILD_FLAG $SCALE_FLAG"
    echo "Running: $CMD"
    eval $CMD

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
# Pass remaining arguments ($@) correctly to functions

case "$COMMAND" in
    up)
        start_services "$@"
        ;;
    down)
        stop_services
        ;;
    status)
        show_status
        ;;
    restart)
        stop_services
        start_services "$@"
        ;;
    *)
        usage
        ;;
esac
