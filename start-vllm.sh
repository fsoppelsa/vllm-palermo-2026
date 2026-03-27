SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vllm-gpu-$(date +%Y%m%d-%H%M%S).log"

# Termina i processi Python nativi che potrebbero mantenere un contesto CUDA/GPU
# (es. un'esecuzione rimasta o andata in crash di 01_naive_inference.py).
# fuser sui device GPU del Tegra è il metodo affidabile per trovarli.
for dev in /dev/nvhost-gpu /dev/nvhost-ctrl-gpu /dev/nvhost-as-gpu; do
    if [ -e "$dev" ]; then
        GPU_PIDS=$(sudo fuser "$dev" 2>/dev/null | tr ' ' '\n' | grep -v '^$' || true)
        if [[ -n "$GPU_PIDS" ]]; then
            echo "[!] Killing processes holding $dev: $GPU_PIDS"
            echo "$GPU_PIDS" | xargs sudo kill -9 2>/dev/null || true
        fi
    fi
done

# Ferma e rimuove il vecchio container se esiste
docker stop vllm-qwen-it 2>/dev/null || true
docker rm vllm-qwen-it 2>/dev/null || true

sleep 2  # lascia che l'IOMMU rilasci tutte le mappature prima di avviare Docker

# Immagine vLLM ottimizzata per Jetson con supporto GPU Docker
# PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync: necessario su Jetson Orin (nvgpu/NvMap).
# Il CUDACachingAllocator di PyTorch chiama nvmlDeviceGetMemoryInfo; sulla GPU integrata Tegra,
# NVML restituisce NOT_SUPPORTED, che scatena un assert interno in CUDACachingAllocator.cpp.
# expandable_segments:True avrebbe dovuto aggirare il problema ma non funziona in tutte le versioni
# di PyTorch. Usare backend:cudaMallocAsync sostituisce completamente l'allocator — nessuna chiamata NVML.
# Richiede l'allocator stream-ordered CUDA 11.4+ (disponibile su JetPack 6.x / CUDA 12.6).
# NCCL_P2P_DISABLE / NCCL_IB_DISABLE: nessun NVLink o InfiniBand su Jetson; disabilita il probing.
docker run -d \
  --name vllm-qwen-it \
  --runtime=nvidia \
  --shm-size 1g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  -p 8000:8000 \
  -v ${MODEL_DIR}:/model:ro \
  ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --quantization gptq_marlin \
  --enforce-eager \
  --disable-log-requests \
  --swap-space 0 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --num-gpu-blocks-override 32

# Reindirizza i log del container su file in background
docker logs -f vllm-qwen-it > "$LOG_FILE" 2>&1 &

echo "Container started with Jetson-optimized vLLM (GPU mode)"
echo "  Logs      : $LOG_FILE"
echo "  Health    : curl http://localhost:8000/health"
echo ""
echo "--- Tailing logs (Ctrl+C to detach, container keeps running) ---"
tail -f "$LOG_FILE"