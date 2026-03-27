#!/bin/bash
# Demo C: Load test parallelo sul simulatore llm-d
# Mostra come le latenze variano con round-robin e spiega
# come llm-d ottimizzerebbe la distribuzione del carico.

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
N_REQUESTS="${N_REQUESTS:-10}"

echo -e "${CYAN}=========================================${RESET}"
echo -e "${CYAN}  Demo C: Load Test — ${BOLD}${N_REQUESTS} richieste parallele${RESET}"
echo -e "${DIM}  Server: ${SIM_URL}${RESET}"
echo -e "${CYAN}=========================================${RESET}"
echo ""
sleep 1
echo -e "${YELLOW}→ Invio ${N_REQUESTS} richieste in parallelo...${RESET}"
echo ""
sleep 0.5

TMPDIR_RESULTS=$(mktemp -d)
trap 'rm -rf "${TMPDIR_RESULTS}"' EXIT

PROMPTS=(
  "Spiega la PagedAttention in una frase"
  "Cos'è il continuous batching nel serving di LLM?"
  "Come funziona la KV cache nei transformer?"
  "Cos'è vLLM e perché è veloce?"
  "Descrivi brevemente il meccanismo di attention"
  "Cos'è un adapter LoRA?"
  "Spiega lo streaming di token nelle API LLM"
  "Cos'è il speculative decoding?"
  "Come la quantizzazione riduce la dimensione del modello?"
  "Qual è la differenza tra TTFT e throughput?"
)

for i in $(seq 1 "${N_REQUESTS}"); do
  (
    PROMPT_IDX=$(( (i - 1) % ${#PROMPTS[@]} ))
    PROMPT="${PROMPTS[$PROMPT_IDX]}: Richiesta $i"
    START_NS=$(date +%s%N)
    RESULT=$(curl -s --max-time 15 "${SIM_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"${PROMPT}\",
        \"max_tokens\": 30
      }" 2>/dev/null)
    END_NS=$(date +%s%N)
    DURATION_MS=$(( (END_NS - START_NS) / 1000000 ))

    if echo "${RESULT}" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
      STATUS="OK"
    else
      STATUS="ERR"
    fi

    echo "${i} ${DURATION_MS} ${STATUS}" > "${TMPDIR_RESULTS}/req_${i}.txt"
  ) &
done

# Attendere tutte le richieste parallele
wait

echo -e "${BOLD}  #   Latenza   Stato${RESET}"
echo -e "${DIM}  ──────────────────${RESET}"

TOTAL_MS=0
OK_COUNT=0
MAX_MS=0
MIN_MS=999999

for i in $(seq 1 "${N_REQUESTS}"); do
  if [ -f "${TMPDIR_RESULTS}/req_${i}.txt" ]; then
    read -r REQ_ID DURATION_MS STATUS < "${TMPDIR_RESULTS}/req_${i}.txt"
    if [ "${STATUS}" = "OK" ]; then
      STATUS_COLOR="${GREEN}${STATUS}${RESET}"
    else
      STATUS_COLOR="${RED}${STATUS}${RESET}"
    fi
    printf "  %-3s %6s ms   " "${REQ_ID}" "${DURATION_MS}"
    echo -e "${STATUS_COLOR}"
    TOTAL_MS=$(( TOTAL_MS + DURATION_MS ))
    [ "${STATUS}" = "OK" ] && OK_COUNT=$(( OK_COUNT + 1 ))
    [ "${DURATION_MS}" -gt "${MAX_MS}" ] && MAX_MS="${DURATION_MS}"
    [ "${DURATION_MS}" -lt "${MIN_MS}" ] && MIN_MS="${DURATION_MS}"
    sleep 0.1
  fi
done

AVG_MS=0
[ "${N_REQUESTS}" -gt 0 ] && AVG_MS=$(( TOTAL_MS / N_REQUESTS ))

echo ""
sleep 0.5
echo -e "${DIM}  ──────────────────────────────────────${RESET}"
printf "  Richieste OK:    ${GREEN}%d${RESET} / %d\n" "${OK_COUNT}" "${N_REQUESTS}"
printf "  Latenza min:     ${GREEN}%d ms${RESET}\n" "${MIN_MS}"
printf "  Latenza max:     ${RED}%d ms${RESET}\n" "${MAX_MS}"
printf "  Latenza media:   ${YELLOW}%d ms${RESET}\n" "${AVG_MS}"
echo -e "${DIM}  ──────────────────────────────────────${RESET}"
echo ""
sleep 1
echo -e "${YELLOW}  Con round-robin le latenze variano perché:${RESET}"
sleep 0.3
echo "    • Alcuni pod hanno la KV cache calda (→ bassa latenza)"
sleep 0.3
echo "    • Altri pod hanno la coda piena (→ alta latenza)"
sleep 0.3
echo "    • La distribuzione è casuale, non ottimizzata"
echo ""
sleep 1
echo -e "${MAGENTA}  Con llm-d, l'Inference Scheduler bilancia in base a:${RESET}"
sleep 0.3
echo -e "    • ${BOLD}KV cache occupancy${RESET} di ogni pod"
sleep 0.3
echo -e "    • ${BOLD}Prefix cache hits${RESET} (stesso prompt → stesso pod)"
sleep 0.3
echo -e "    • ${BOLD}Queue depth${RESET} (numero di richieste in coda)"
sleep 0.3
echo -e "    • ${BOLD}Latenza predetta${RESET} per la richiesta specifica"
echo ""
sleep 0.5
echo -e "  ${GREEN}Risultato: latenze più basse e più uniformi,${RESET}"
echo -e "  ${GREEN}throughput più alto, GPU utilizzata meglio.${RESET}"
echo -e "${CYAN}=========================================${RESET}"
