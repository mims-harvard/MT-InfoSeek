#!/usr/bin/env bash
set -euo pipefail

# Pull env vars from .env at repo root if present (see .env.example).
_ENV_FILE="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/.env"
# .env supplies defaults; anything already exported (e.g. by run_eval.py) wins.
# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/load_env.sh" 2>/dev/null \
  || source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/load_env.sh"
_load_env "$_ENV_FILE"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="${MODEL:-gpt-5-mini}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs/gsme}"
DATA_FILE="${DATA_FILE:-gsme_q_mt.csv}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BUDGET="${BUDGET:-10}"

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
"${PYTHON:-python}" "${ROOT_DIR}/gsme_q_mt/run_gsme.py" \
  --model_name "$MODEL" \
  --eval_mode mt_one \
  --data_file "$DATA_FILE" \
  --data_dir "$DATA_DIR" \
  --results_dir "$RESULTS_DIR" \
  --batch_size "$BATCH_SIZE" \
  --budget "$BUDGET" \
  --parallel_model_calls True \
  --sample_seed 42 \
  ${MODEL_CONFIG:+--model_config "$MODEL_CONFIG"}
