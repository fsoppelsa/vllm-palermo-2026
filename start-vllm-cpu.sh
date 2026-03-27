#!/bin/bash
# Server di inferenza CPU — esegue vLLM in un container Docker usando l'immagine ufficiale CPU.
# vllm/vllm-openai-cpu è un'immagine multi-arch (amd64 + arm64/v8) pubblicata dal team vLLM.
# Nessun CUDA richiesto. Usa la stessa API OpenAI-compatible sulla porta 8000.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/Qwen2-1.5B-Ita"
CONTAINER_NAME="vllm-cpu"
# Immagine ufficiale vLLM CPU — multi-arch (arm64 + amd64), senza CUDA.
# v0.17.0 corregge il rilevamento della piattaforma CPU ARM64 che era rotto in v0.16.0.
VLLM_CPU_IMAGE="${VLLM_CPU_IMAGE:-vllm/vllm-openai-cpu:v0.17.0}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vllm-cpu-$(date +%Y%m%d-%H%M%S).log"

# ---------------------------------------------------------------------------
# 1. Ferma il container Docker GPU se in esecuzione (occupa la porta 8000 via
#    networking Docker — fuser/lsof non lo vedono, solo docker stop la rilascia).
# ---------------------------------------------------------------------------
if docker inspect vllm-qwen-it &>/dev/null 2>&1; then
    echo "Stopping GPU Docker container vllm-qwen-it..."
    docker stop vllm-qwen-it 2>/dev/null || true
    docker rm   vllm-qwen-it 2>/dev/null || true
    echo "GPU container stopped."
fi

# ---------------------------------------------------------------------------
# 2. Ferma eventuali istanze esistenti del container CPU.
# ---------------------------------------------------------------------------
if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    echo "Stopping existing CPU container $CONTAINER_NAME..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm   "$CONTAINER_NAME" 2>/dev/null || true
fi

# Sicurezza aggiuntiva: termina qualsiasi processo ancora sulla porta 8000 (non-Docker)
fuser -k 8000/tcp 2>/dev/null || true
sleep 1

# ---------------------------------------------------------------------------
# 3. Verifica la directory del modello prima del lancio.
# ---------------------------------------------------------------------------
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory not found: $MODEL_DIR"
    exit 1
fi

echo ""
echo "Starting vLLM CPU container..."
echo "  Image : $VLLM_CPU_IMAGE"
echo "  Model : $MODEL_DIR"
echo "  Log   : $LOG_FILE"
echo ""

# ---------------------------------------------------------------------------
# 3b. Verifica che l'immagine corretta sia presente (pull se mancante).
# ---------------------------------------------------------------------------
if ! docker image inspect "$VLLM_CPU_IMAGE" &>/dev/null; then
    echo "Pulling $VLLM_CPU_IMAGE ..."
    docker pull "$VLLM_CPU_IMAGE"
fi

# ---------------------------------------------------------------------------
# 4. Avvia il container in modalità detached.
#    --runtime runc  — CRITICO su Jetson: sovrascrive il runtime nvidia predefinito
#                      così nessuna libreria CUDA viene iniettata nel container.
#    VLLM_TARGET_DEVICE=cpu — indica a vLLM di usare esplicitamente il backend CPU.
#    bfloat16 è il dtype corretto per l'inferenza CPU su questo modello.
# ---------------------------------------------------------------------------
docker run -d \
    --runtime runc \
    --name "$CONTAINER_NAME" \
    -p 8000:8000 \
    -v "${MODEL_DIR}:/model:ro" \
    -e VLLM_TARGET_DEVICE=cpu \
    -e VLLM_CPU_KVCACHE_SPACE=1 \
    "$VLLM_CPU_IMAGE" \
    --model /model \
    --dtype bfloat16 \
    --max-model-len 512 \
    --swap-space 0 \
    --enforce-eager \
    --max-num-seqs 1 \
    --max-num-batched-tokens 512 \
    --host 0.0.0.0 \
    --port 8000

# Reindirizza i log del container su file in background
docker logs -f "$CONTAINER_NAME" > "$LOG_FILE" 2>&1 &

echo "Container started ($(docker inspect -f '{{.Id}}' "$CONTAINER_NAME" | cut -c1-12))"

# ---------------------------------------------------------------------------
# 5. Attende che il server sia pronto (il caricamento del modello può richiedere 30-90 s su CPU).
# ---------------------------------------------------------------------------
echo "Waiting for server to become ready..."
READY=0
for i in $(seq 1 90); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        READY=1
        break
    fi
    # Verifica che il container sia ancora in esecuzione
    STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "missing")
    if [[ "$STATUS" != "running" ]]; then
        echo ""
        echo "ERROR: container exited unexpectedly (status: $STATUS). Last log lines:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    printf "."
    sleep 2
done

echo ""
if [[ $READY -eq 1 ]]; then
    echo "Server is ready!"
    echo ""
    echo "  Container : $CONTAINER_NAME"
    echo "  Logs      : $LOG_FILE"
    echo "  Health    : curl http://localhost:8000/health"
    echo "  Models    : curl http://localhost:8000/v1/models"
    echo "  Stop      : ./stop-vllm-cpu.sh"
    echo ""
    echo "--- Tailing logs (Ctrl+C to detach, container keeps running) ---"
    tail -f "$LOG_FILE"
else
    echo "ERROR: server did not become ready within 180s. Last log lines:"
    tail -20 "$LOG_FILE"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm   "$CONTAINER_NAME" 2>/dev/null || true
    exit 1
fi
