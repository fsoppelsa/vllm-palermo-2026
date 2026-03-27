#!/bin/bash
CONTAINER_NAME="vllm-cpu"

if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    echo "Stopping CPU container $CONTAINER_NAME..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm   "$CONTAINER_NAME" 2>/dev/null || true
    echo "Stopped."
else
    echo "Container $CONTAINER_NAME is not running."
fi

# Sicurezza aggiuntiva: libera la porta 8000 (processi non-Docker)
fuser -k 8000/tcp 2>/dev/null || true
