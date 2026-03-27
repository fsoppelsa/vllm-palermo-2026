#!/usr/bin/env bash
# Parte 1 — Baseline naive HuggingFace (eseguito su Jetson)
# Richiede l'ambiente conda 'py310' con torch abilitato per CUDA:
#   conda create -n py310 python=3.10 -y
#   conda activate py310
#   pip install torch --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
#   pip install gptqmodel -r requirements.txt
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sopprime le barre di progresso tqdm di HuggingFace durante il caricamento del modello.
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Forza stdout Python non bufferizzato così i token in streaming appaiono immediatamente
# anche quando conda run instrada stdout attraverso se stesso (modalità non-TTY).
export PYTHONUNBUFFERED=1

PYTHON="$(conda info --base)/envs/py310/bin/python"
exec "$PYTHON" -u "$DIR/01_naive_inference.py" "$@"
