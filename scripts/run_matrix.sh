#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

required_keys=(
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  MISTRAL_API_KEY
  XAI_API_KEY
)
missing=()
for key in "${required_keys[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    missing+=("$key")
  fi
done
if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  missing+=("GEMINI_API_KEY or GOOGLE_API_KEY")
fi

if (( ${#missing[@]} > 0 )); then
  echo "Missing required environment variables for cloud matrix runs:"
  for item in "${missing[@]}"; do
    echo "  - $item"
  done
  echo "Set them in your shell or in .env before running."
  exit 1
fi

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
