#!/usr/bin/env bash
# Ferma e rimuove il container vLLM in esecuzione.
set -euo pipefail

CONTAINER_NAME="${1:-vllm-qwen-it}"

if ! docker inspect "$CONTAINER_NAME" &>/dev/null; then
    echo "No container named '$CONTAINER_NAME' found."
    exit 0
fi

STATUS="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")"
if [[ "$STATUS" != "running" ]]; then
    echo "Container '$CONTAINER_NAME' exists but is not running (status: $STATUS)."
else
    echo "Stopping container '$CONTAINER_NAME'..."
    docker stop "$CONTAINER_NAME"
fi

echo "Removing container '$CONTAINER_NAME'..."
docker rm "$CONTAINER_NAME"
echo "Done."
