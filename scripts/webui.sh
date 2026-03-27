#!/usr/bin/env bash
# Parte 4 — Web UI Flask (eseguita sul laptop, fa da proxy verso il Jetson)
# Utilizzo: ./webui.sh [--url http://orin:8000] [--port 5000]
# Oppure:   VLLM_URL=http://orin:8000 ./webui.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/04_flask_app.py" "$@"
