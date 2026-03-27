#!/bin/bash
# Ferma il simulatore llm-d e spegne K3s
# Uso: bash llm-d/stop-k3s.sh

set -euo pipefail

NAMESPACE="llm-d-demo"

echo "========================================="
echo "  llm-d Demo — Shutdown K3s"
echo "========================================="
echo ""

# ── 1. Rimuove il ClusterRole/Binding Prometheus (cluster-scoped) ───────────
echo "→ Rimozione RBAC Prometheus..."
kubectl delete clusterrolebinding prometheus-llmd --ignore-not-found
kubectl delete clusterrole prometheus-llmd --ignore-not-found

# ── 2. Rimuove le risorse llm-d-demo (simulatore + Prometheus) ───────────────
if kubectl get namespace "${NAMESPACE}" &>/dev/null; then
  echo "→ Rimozione risorse in ${NAMESPACE} (simulatore + Prometheus)..."
  kubectl delete namespace "${NAMESPACE}" --timeout=30s || true
  echo "  Namespace rimosso."
else
  echo "→ Namespace ${NAMESPACE} non esiste, nessuna pulizia necessaria."
fi

echo ""

# ── 2. Ferma K3s ────────────────────────────────────────────────────────────
if systemctl is-active --quiet k3s 2>/dev/null; then
  echo "→ Fermo K3s..."
  sudo systemctl stop k3s
  echo "  K3s fermato."
else
  echo "→ K3s non era in esecuzione."
fi

echo ""
echo "========================================="
echo "  Shutdown completato."
echo "========================================="
