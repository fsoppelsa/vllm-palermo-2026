#!/bin/bash
# Demo B: Routing Intelligente llm-d
# Mostra il simulatore llm-d in azione su K3s con 3 repliche,
# evidenzia perché il round-robin non è ottimale per gli LLM,
# e introduce il routing intelligente di llm-d.

set -euo pipefail

# ── Colori ──────────────────────────────────────────────────────────────────
BOLD="\e[1m"
RESET="\e[0m"
CYAN="\e[1;36m"
YELLOW="\e[1;33m"
GREEN="\e[1;32m"
RED="\e[1;31m"
MAGENTA="\e[1;35m"
DIM="\e[2m"

SIM_URL="${SIM_URL:-http://localhost:31800}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Demo B: llm-d — Intelligent Inference${RESET}"
echo -e "${CYAN}  Routing su Kubernetes${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 1

echo -e "${YELLOW}→ Repliche del model server (simulato):${RESET}"
kubectl -n llm-d-demo get pods -o wide
echo ""
sleep 0.5

echo -e "${YELLOW}→ Endpoints del Service:${RESET}"
kubectl -n llm-d-demo get endpoints vllm-sim-service
echo ""
sleep 1

echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Test 1: Round-Robin ${DIM}(K8s default)${RESET}"
echo -e "${DIM}  Ogni richiesta va a un pod diverso${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 0.5

RR_PROMPTS=(
  "Cos'è Kubernetes e perché è importante per il deploy di LLM?"
  "Spiega la differenza tra un Pod e un Deployment in Kubernetes"
  "Come funziona il bilanciamento del carico in un cluster K8s?"
  "Cos'è un Service di tipo NodePort in Kubernetes?"
  "Perché il round-robin non è ottimale per il serving di modelli LLM?"
)

for i in 1 2 3 4 5; do
  echo -e "${YELLOW}--- Richiesta $i ---${RESET}"
  PROMPT="${RR_PROMPTS[$((i-1))]}"
  echo -e "${DIM}  prompt: \"${PROMPT}\"${RESET}"
  curl -s --max-time 10 "${SIM_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"prompt\": \"${PROMPT}\",
      \"max_tokens\": 20
    }" &>/dev/null && echo -e "${GREEN}  → OK${RESET}" || echo -e "${RED}  → [errore connessione]${RESET}"
  echo ""
  sleep 0.8
done

sleep 1
echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Metriche Prometheus per pod${RESET}"
echo -e "${DIM}  (perché serve routing intelligente)${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 0.5

# Avvia carico di fondo per rendere le metriche non-zero durante la lettura
echo -e "${DIM}→ Generazione carico di fondo per popolare le metriche...${RESET}"
LOAD_PIDS=()
for i in $(seq 1 12); do
  ( while true; do
      curl -s --max-time 5 "${SIM_URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Spiega la KV cache negli LLM in dettaglio molto lungo\",\"max_tokens\":200}" \
        &>/dev/null
    done
  ) &
  LOAD_PIDS+=($!)
done
sleep 2  # lascia che le richieste si accumulino

echo -e "${YELLOW}→ KV cache usage e queue depth per ogni replica:${RESET}"
echo ""

# ╔══════════════════════════════════════════════════════════════╗
# ║              METRICHE PROMETHEUS — KV CACHE                  ║
# ╚══════════════════════════════════════════════════════════════╝
BOX_WIDTH=62
BOX_LINE=$(printf '═%.0s' $(seq 1 $BOX_WIDTH))
echo -e "${MAGENTA}╔${BOX_LINE}╗${RESET}"
printf "${MAGENTA}║${RESET}  ${BOLD}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "METRICHE PROMETHEUS — KV CACHE & QUEUE DEPTH"
echo -e "${MAGENTA}╠${BOX_LINE}╣${RESET}"

for pod in $(kubectl -n llm-d-demo get pods -l app=vllm-sim -o name 2>/dev/null); do
  POD_NAME="${pod#pod/}"
  printf "${MAGENTA}║${RESET}  ${CYAN}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "Pod: ${POD_NAME}"
  printf "${MAGENTA}║${RESET}  ${DIM}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "──────────────────────────────────────────────────────────"

  METRICS=$(kubectl -n llm-d-demo exec "${POD_NAME}" -- \
    curl -s localhost:8000/metrics 2>/dev/null | \
    grep -E "^(vllm:num_requests_running|vllm:num_requests_waiting|vllm:num_requests_swapped|vllm:gpu_cache_usage_perc)" | \
    head -6)

  if [ -n "${METRICS}" ]; then
    while IFS= read -r line; do
      printf "${MAGENTA}║${RESET}  ${GREEN}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "${line}"
    done <<< "${METRICS}"
  else
    printf "${MAGENTA}║${RESET}  ${DIM}%-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" "(metriche non disponibili)"
  fi

  printf "${MAGENTA}║${RESET}  %-${BOX_WIDTH}s${MAGENTA}║${RESET}\n" ""
  sleep 0.5
done

echo -e "${MAGENTA}╚${BOX_LINE}╝${RESET}"
echo ""

# Ferma il carico di fondo
for pid in "${LOAD_PIDS[@]}"; do
  kill "${pid}" 2>/dev/null || true
done
wait "${LOAD_PIDS[@]}" 2>/dev/null || true

sleep 1
echo -e "${RED}=========================================${RESET}"
echo -e "${RED}  Il PROBLEMA del round-robin per LLM:${RESET}"
echo ""
echo -e "  Ogni pod mantiene una ${BOLD}KV cache separata${RESET}."
echo -e "  Con round-robin:"
sleep 0.5
echo -e "    • ${YELLOW}Pod A${RESET}: KV cache calda per il tuo prompt ${GREEN}→ risposta veloce${RESET}"
sleep 0.4
echo -e "    • ${YELLOW}Pod B${RESET}: KV cache fredda ${RED}→ deve ricalcolare tutto${RESET}"
sleep 0.4
echo -e "    • ${YELLOW}Pod C${RESET}: in coda con 5 richieste ${RED}→ attesa extra${RESET}"
echo ""
sleep 0.5
echo -e "  ${DIM}K8s invia le richieste in modo cieco,"
echo -e "  ignorando lo stato interno di ogni pod.${RESET}"
echo ""
sleep 1
echo -e "${MAGENTA}  → llm-d Inference Scheduler risolve questo con:${RESET}"
sleep 0.4
echo -e "    • ${BOLD}KV-cache-aware routing${RESET} (prefix matching)"
sleep 0.4
echo -e "    • ${BOLD}Load-aware${RESET} (evita pod saturi)"
sleep 0.4
echo -e "    • ${BOLD}Session-aware${RESET} (stessa sessione → stesso pod)"
echo -e "${MAGENTA}=========================================${RESET}"
echo ""
sleep 1

echo -e "${YELLOW}→ Ecco come si configura in produzione (Inference Gateway CRDs):${RESET}"
echo ""
sleep 0.5
cat "$(dirname "$0")/llmd-gateway-showcase.yaml"
