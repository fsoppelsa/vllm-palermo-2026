#!/bin/bash
# Avvia K3s e prepara l'ambiente per i demo llm-d
# Uso: bash llm-d/start-k3s.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="llm-d-demo"

echo "========================================="
echo "  llm-d Demo — Setup K3s"
echo "========================================="
echo ""

# ── 1. Avvia K3s se non è già in esecuzione ─────────────────────────────────
if ! systemctl is-active --quiet k3s 2>/dev/null; then
  echo "→ Avvio K3s..."
  sudo systemctl start k3s
  echo "  K3s avviato."
else
  echo "→ K3s già in esecuzione."
fi

# ── 2. Aspetta che l'API server sia pronto ───────────────────────────────────
echo ""
echo "→ Attendo che il nodo sia Ready..."
until kubectl get nodes 2>/dev/null | grep -q " Ready"; do
  sleep 2
  printf "."
done
echo ""
echo "  Nodo pronto."

# ── 3. Applica il simulatore ─────────────────────────────────────────────────
echo ""
echo "→ Applico llmd-simulator.yaml..."
kubectl apply -f "${SCRIPT_DIR}/llmd-simulator.yaml"

# ── 4. Aspetta che le repliche del simulatore siano up ──────────────────────
echo ""
echo "→ Attendo che i pod del simulatore siano Running (max 90s)..."
kubectl -n "${NAMESPACE}" rollout status deployment/vllm-simulator --timeout=90s

echo ""
echo "→ Pod in esecuzione:"
kubectl -n "${NAMESPACE}" get pods -o wide
echo ""
echo "→ NodePort service:"
kubectl -n "${NAMESPACE}" get svc vllm-sim-nodeport
echo ""

# ── 5. Avvia Prometheus ──────────────────────────────────────────────────────
echo ""
echo "→ Applico prometheus.yaml..."
kubectl apply -f "${SCRIPT_DIR}/prometheus.yaml"

echo ""
echo "→ Attendo che Prometheus sia Running (max 60s)..."
kubectl -n "${NAMESPACE}" rollout status deployment/prometheus --timeout=60s

# ── 6. Verifica che il NodePort (31800) risponda ─────────────────────────────
SIM_URL="http://localhost:31800"
echo ""
echo "→ Verifica endpoint ${SIM_URL}/v1/models..."
if curl -sf --max-time 5 "${SIM_URL}/v1/models" | python3 -c "import sys,json; print('  Modelli:', [m['id'] for m in json.load(sys.stdin)['data']])" 2>/dev/null; then
  echo ""
  echo "========================================="
  echo "  Tutto pronto. Andiamo!"
  echo ""
  echo "    bash llm-d/demo-routing.sh"
  echo "    bash llm-d/demo-load.sh"
  echo ""
  echo "  Prometheus UI: http://localhost:31900"
  echo "  Query utili:"
  echo "    vllm:num_requests_running"
  echo "    vllm:num_requests_waiting"
  echo "    vllm:generation_tokens_total"
  echo "========================================="
else
  echo "  ATTENZIONE: il simulatore non risponde ancora su ${SIM_URL}."
  echo "  Aspetta qualche secondo e riprova, oppure controlla:"
  echo "    kubectl -n ${NAMESPACE} logs -l app=vllm-sim --tail=20"
  exit 1
fi
