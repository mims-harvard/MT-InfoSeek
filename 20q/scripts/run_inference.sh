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
DATASET_DIR="${ROOT_DIR}/20q"

export PYTHONNOUSERSITE=1
# Only used for Gemini; leave unset unless the user opted into Vertex in .env.
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# Local vLLM server: the whole suite uses VLLM_PORT (default 8011). Do not
# introduce a second variable or a second default -- 20Q previously forced
# LOCAL_VLLM_PORT=8000 here and silently dialled the wrong port.
export VLLM_PORT="${VLLM_PORT:-8011}"

DATASET="${DATASET:-common}"
GUESSER_MODEL="${GUESSER_MODEL:-gemini-3-flash-preview}"
EXAMINER_MODEL="${EXAMINER_MODEL:-gpt-5-mini}"
MAX_TURN="${MAX_TURN:-20}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"       # -1 = run the whole pool

cd "${RESULTS_DIR:-$ROOT_DIR}"
EXTRA_FLAGS=()
[[ -n "${MODEL_CONFIG:-}" ]] && EXTRA_FLAGS+=(--model_config "$MODEL_CONFIG")

PYTHONPATH="${ROOT_DIR}:${DATASET_DIR}:${PYTHONPATH:-}" \
"${PYTHON:-python}" "${DATASET_DIR}/twenty_questions/run.py" \
  --dataset "$DATASET" \
  --guesser_model "$GUESSER_MODEL" \
  --examiner_model "$EXAMINER_MODEL" \
  --max_turn "$MAX_TURN" \
  --task_start_index "$START_IDX" \
  --task_end_index "$END_IDX" \
  "${EXTRA_FLAGS[@]}"
