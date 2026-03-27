# vLLM Demo — Linux Meeting Palermo 2026

Materiale del talk **"vLLM: serving di LLM dall'edge al cloud"** presentato al
[Linux Meeting Palermo 2026](https://www.linuxmeeting.it/).

La presentazione è in [`Presentazione.pdf`](Presentazione.pdf).

---

## Cosa c'è nel repo

### Demo principale — vLLM su Jetson Orin Nano

Quattro script Python che mostrano la progressione dal raw inference al serving ottimizzato:

| Script | Descrizione |
|--------|-------------|
| `scripts/01_naive_inference.py` | Inferenza diretta con HuggingFace Transformers (baseline) |
| `scripts/02_vllm_single.py` | Singola richiesta al server vLLM con streaming e confronto prestazioni |
| `scripts/03_vllm_batching.py` | 5 richieste parallele — dimostra il continuous batching |
| `scripts/04_flask_app.py` | Web UI Flask che fa da proxy verso il server vLLM |

Ogni script ha un wrapper shell in `scripts/` (`naive.sh`, `single.sh`, `batch.sh`, `webui.sh`).

### Avvio del server vLLM

```bash
# Modalità GPU (Docker, Jetson Orin Nano)
./start-vllm.sh

# Modalità CPU (Docker, qualsiasi macchina)
./start-vllm-cpu.sh

# Stop
./stop-vllm.sh
./stop-vllm-cpu.sh
```

### Demo llm-d — Routing intelligente su Kubernetes

La directory `llm-d/` contiene script e manifest per mostrare come
[llm-d](https://github.com/llm-d/llm-d) ottimizza il routing delle
richieste in un cluster K3s con più repliche del model server.

```bash
bash llm-d/start-k3s.sh    # avvia K3s + simulatore vLLM
bash llm-d/demo-routing.sh # Demo B: routing intelligente vs round-robin
bash llm-d/demo-load.sh    # Demo C: load test parallelo
bash llm-d/stop-k3s.sh    # ferma tutto
```

---

## Hardware di riferimento

- **NVIDIA Jetson Orin Nano 8GB** (aarch64, memoria unificata CPU/GPU)
- Modello: `Qwen2-1.5B-Ita` (CPU) / `Qwen2.5-1.5B-Instruct-GPTQ-Int4` (GPU)
- API OpenAI-compatible su `http://orin:8000`

---

## Dipendenze

```bash
pip install -r scripts/requirements.txt
```
