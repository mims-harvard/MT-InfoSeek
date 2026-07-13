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
DATASET_DIR="${ROOT_DIR}/logic_q_mt"

MODEL="${MODEL:-gpt-5-mini}"
EVAL_MODE="${EVAL_MODE:-mc}"               # mc | isambig | fullinfo
PROMPT_MODE="${PROMPT_MODE:-exact_k}"      # exact_k | at_most_K
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
DATA_FILE="${DATA_FILE:-${DATA_DIR}/logic_q_mt.csv}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs/logic_q_mt}"
BATCH_SIZE="${BATCH_SIZE:-32}"

EXTRA_FLAGS=()
[[ "${FORBID_ALTERNATIVES:-false}" == "true" ]] && EXTRA_FLAGS+=(--forbid_alternatives)
[[ -n "${VLLM_PORT:-}"        ]] && EXTRA_FLAGS+=(--vllm_port "$VLLM_PORT")
[[ -n "${REASONING_EFFORT:-}" ]] && EXTRA_FLAGS+=(--reasoning_effort "$REASONING_EFFORT")
[[ -n "${MAX_TOKENS:-}"       ]] && EXTRA_FLAGS+=(--max_tokens "$MAX_TOKENS")

mkdir -p "$RESULTS_DIR"
PYTHONPATH="${ROOT_DIR}:${DATASET_DIR}/evaluation:${PYTHONPATH:-}" \
"${PYTHON:-python}" "${DATASET_DIR}/evaluation/mc_eval.py" \
    --model_name "$MODEL" \
    --domain_name SL \
    --eval_mode "$EVAL_MODE" \
    --prompt_mode "$PROMPT_MODE" \
    --data_file "$DATA_FILE" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_DIR" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_FLAGS[@]}"
