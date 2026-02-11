#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

configs=(
  configs/matrix/exp_baseline.yaml
  configs/matrix/exp_permission_low.yaml
  configs/matrix/exp_permission_med.yaml
  configs/matrix/exp_permission_high.yaml
  configs/matrix/exp_missing_low.yaml
  configs/matrix/exp_missing_med.yaml
  configs/matrix/exp_missing_high.yaml
  configs/matrix/exp_latency_low.yaml
  configs/matrix/exp_latency_med.yaml
  configs/matrix/exp_latency_high.yaml
  configs/matrix/exp_timeout_low.yaml
  configs/matrix/exp_timeout_med.yaml
  configs/matrix/exp_timeout_high.yaml
)

for cfg in "${configs[@]}"; do
  echo "============================================================"
  echo "Running: $cfg"
  python -m harness.run_experiments --config "$cfg"
  echo
  sleep 1
done

echo "Matrix run complete. Summaries are under evaluation/results/matrix/."
