#!/usr/bin/env bash
# Parte 2 — Singola richiesta vLLM con streaming + confronto costi
# Utilizzo: ./single.sh [--url http://orin:8000]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/02_vllm_single.py" "$@"
