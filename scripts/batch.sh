#!/usr/bin/env bash
# Parte 3 — 5 richieste parallele che dimostrano il continuous batching
# Utilizzo: ./batch.sh [--url http://orin:8000]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/03_vllm_batching.py" "$@"
