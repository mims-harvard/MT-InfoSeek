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
DATASET_DIR="${ROOT_DIR}/genereg_mt"

MODEL="${MODEL:-gpt-5-mini}"
DATA_FILE="${DATA_FILE:-${ROOT_DIR}/data/genereg_mt.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs/genereg_mt/multiturn}"

CACHE_DIR="${CACHE_DIR:?CACHE_DIR is required (the per-model pickle cache written by grn_dataset_curator.py)}"
MODELS_DIR="${MODELS_DIR:?MODELS_DIR is required (clone of https://github.com/ckadelka/DesignPrinciplesGeneNetworks, update_rules_122_models_Kadelka_SciAdv/)}"

INCLUDE_TASKS="${INCLUDE_TASKS:-dyn_attr,dyn_marker}"
BUDGET="${BUDGET:-10}"                         # set BUDGET=0 to disable budget
ORACLE="${ORACLE:-adversarial}"                # adversarial | cooperative | random
WORLDS="${WORLDS:--1}"                         # -1 = expand all branches per outcome
MAX_SAMPLES_PER_TASK_K="${MAX_SAMPLES_PER_TASK_K:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
CACHE_TAG="${CACHE_TAG:-genereg_mt}"

EXTRA_FLAGS=()
[[ -n "${MODEL_CONFIG:-}" ]] && EXTRA_FLAGS+=(--model-config "$MODEL_CONFIG")
[[ "${BUDGET_IN_PROMPT:-true}"    == "false" ]] && EXTRA_FLAGS+=(--no-budget-in-prompt)
[[ "${FORBID_ALTERNATIVES:-true}" == "true" ]] && EXTRA_FLAGS+=(--forbid-alternatives)
[[ "${SAVE_TURN_PROMPTS:-false}"  == "true" ]] && EXTRA_FLAGS+=(--save-turn-prompts)
[[ -n "${REASONING_EFFORT:-}" ]] && EXTRA_FLAGS+=(--reasoning-effort "$REASONING_EFFORT")
[[ -n "${VLLM_PORT:-}"        ]] && EXTRA_FLAGS+=(--port "$VLLM_PORT")

mkdir -p "$RESULTS_DIR"
PYTHONPATH="${ROOT_DIR}:${DATASET_DIR}/evaluation:${DATASET_DIR}/data_generation:${PYTHONPATH:-}" \
"${PYTHON:-python}" "${DATASET_DIR}/evaluation/multiturn_eval.py" \
    --model-name "$MODEL" \
    --data-file "$DATA_FILE" \
    --cache-dir "$CACHE_DIR" \
    --models-dir "$MODELS_DIR" \
    --include-tasks "$INCLUDE_TASKS" \
    --budget "$BUDGET" \
    --oracle-type "$ORACLE" \
    --worlds-per-sufficient-set-per-sign "$WORLDS" \
    --max-samples-per-task-k "$MAX_SAMPLES_PER_TASK_K" \
    --max-concurrent "$MAX_CONCURRENT" \
    --cache-tag "$CACHE_TAG" \
    --results-dir "$RESULTS_DIR" \
    "${EXTRA_FLAGS[@]}"
