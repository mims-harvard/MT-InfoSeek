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
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
DATA_FILE="${DATA_FILE:-${DATA_DIR}/logic_q_mt.csv}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs/logic_q_mt/multiturn}"

BUDGET="${BUDGET:-10}"                       # set BUDGET=0 to disable budget signaling
ORACLE="${ORACLE:-adversarial}"              # adversarial | cooperative | random
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
CACHE_TAG="${CACHE_TAG:-logic_q_mt}"

EXTRA_FLAGS=()
[[ -n "${MODEL_CONFIG:-}" ]] && EXTRA_FLAGS+=(--model-config "$MODEL_CONFIG")
EXTRA_FLAGS+=(--forbid-alternatives)
EXTRA_FLAGS+=(--no-flip)
[[ "${BUDGET_IN_PROMPT:-true}"    == "false" ]] && EXTRA_FLAGS+=(--no-budget-in-prompt)
[[ "${KEEP_THINKING_TRACE:-false}" == "true" ]] && EXTRA_FLAGS+=(--keep-thinking-trace)
[[ -n "${REASONING_EFFORT:-}" ]] && EXTRA_FLAGS+=(--reasoning-effort "$REASONING_EFFORT")
[[ -n "${VLLM_PORT:-}"        ]] && EXTRA_FLAGS+=(--port "$VLLM_PORT")
[[ -n "${MAX_TOKENS:-}"       ]] && EXTRA_FLAGS+=(--max_tokens "$MAX_TOKENS")

mkdir -p "$RESULTS_DIR"
PYTHONPATH="${ROOT_DIR}:${DATASET_DIR}/evaluation:${DATASET_DIR}/data_generation:${PYTHONPATH:-}" \
"${PYTHON:-python}" "${DATASET_DIR}/evaluation/multiturn_eval.py" \
    --model-name "$MODEL" \
    --data-file "$DATA_FILE" \
    --cache-tag "$CACHE_TAG" \
    --budget "$BUDGET" \
    --oracle-type "$ORACLE" \
    --sufficient-sets-per-problem 1 \
    --worlds-per-sufficient-set-per-sign 1 \
    --max-concurrent "$MAX_CONCURRENT" \
    --results-dir "$RESULTS_DIR" \
    "${EXTRA_FLAGS[@]}"
