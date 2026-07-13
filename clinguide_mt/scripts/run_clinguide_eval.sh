#!/usr/bin/env bash
# Run the ClinGuide multi-turn evaluation.
#
# Like every other dataset script, this assumes any local model is ALREADY being
# served on VLLM_PORT. Start one server for the whole run with:
#     bash scripts/serve_vllm.sh <model>
#
# ── Quick start ──────────────────────────────────────────────────────────────
#
#   # API model (no vLLM)
#   MODEL_NAME=gpt-4o bash clinguide_mt/scripts/run_clinguide_eval.sh
#
#   # Local model (starts vLLM automatically)
#   MODEL_NAME=Qwen/Qwen3.5-9B bash clinguide_mt/scripts/run_clinguide_eval.sh
#
#   # Override any variable inline or via .env
#   CONTEXT_MODE=patient_guideline NUM_DISTRACTORS=5 bash clinguide_mt/scripts/run_clinguide_eval.sh
#
# ── Environment / credentials ────────────────────────────────────────────────
#   Copy .env.example → .env at the repo root and fill in your keys.
#   All variables below can also be set as env vars before calling this script.
#
set -euo pipefail

# ── Repo root (two levels up from clinguide_mt/scripts/) ────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Pull credentials from .env at repo root if present (see .env.example)
_ENV_FILE="${ROOT_DIR}/.env"
# .env supplies defaults; anything already exported (e.g. by run_eval.py) wins.
# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/load_env.sh" 2>/dev/null \
  || source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/load_env.sh"
_load_env "$_ENV_FILE"

# ── Model ─────────────────────────────────────────────────────────────────────
# API models  : gpt-4o, gpt-5-mini, gemini-3-flash-preview, …
# Local models: Qwen/Qwen3.5-9B, openai/gpt-oss-20B, …
MODEL_NAME="${MODEL_NAME:-gpt-4o}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"   # low|medium|high (GPT-OSS only)

# Oracle judge model (read by multiturn_clinguide_eval.py via env; defaults to gpt-5).
ORACLE_MODEL="${ORACLE_MODEL:-gpt-5}"
export ORACLE_MODEL

# ── Local model serving ───────────────────────────────────────────────────────
# Local models must already be served (see scripts/serve_vllm.sh). The evaluator
# sends local model requests to VLLM_PORT.
VLLM_PORT="${VLLM_PORT:-8011}"

# ── Data / output paths ───────────────────────────────────────────────────────
JSON_DIR="${JSON_DIR:-/path/to/clinguide/data}"           # *_tree.json, *_question_list_all.json
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs/clinguide}"

# ── Eval hyperparameters ──────────────────────────────────────────────────────
BUDGET="${BUDGET:-4}"                        # 0=single-turn | 4 | k | 10
CONTEXT_MODE="${CONTEXT_MODE:-patient_question_list}"
#   patient_only | patient_question_list | patient_guideline | patient_question_list_and_guideline
MAX_GUIDELINES="${MAX_GUIDELINES:-1}"        # >1 adds distractor guidelines when context includes guideline
NUM_DISTRACTORS="${NUM_DISTRACTORS:-0}"      # distractor guideline sets added to allowed topics (-1=all)
ORACLE_MODE="${ORACLE_MODE:-standard}"       # standard | adversarial | cooperative
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"       # async concurrency cap (lower for small GPU slices)
SEED="${SEED:-42}"
CACHE_TAG="${CACHE_TAG:-}"

# ── Derived ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/multiturn_clinguide_eval.py"
LOG_DIR="${RESULTS_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ClinGuide Multi-Turn Evaluation"
echo "============================================================"
echo "  Repo root    : ${ROOT_DIR}"
echo "  Eval script  : ${EVAL_SCRIPT}"
echo "  Model           : ${MODEL_NAME}"
echo "  Reasoning effort: ${REASONING_EFFORT}"
echo "  JSON dir        : ${JSON_DIR}"
echo "  Results dir     : ${RESULTS_DIR}"
echo "  Budget          : ${BUDGET}"
echo "  Context mode    : ${CONTEXT_MODE}"
echo "  Max guidelines  : ${MAX_GUIDELINES}"
echo "  Distractors     : ${NUM_DISTRACTORS}"
echo "  Oracle mode     : ${ORACLE_MODE}"
echo "  Oracle model    : ${ORACLE_MODEL}"
echo "  Max concurrent  : ${MAX_CONCURRENT}"
echo "  Seed            : ${SEED}"
echo "============================================================"
echo ""

# ── Run evaluation ────────────────────────────────────────────────────────────
echo ""
echo "Running ClinGuide evaluation..."
echo ""

cd "${ROOT_DIR}"

if PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
"${PYTHON:-python}" "${EVAL_SCRIPT}" \
    --model-name        "${MODEL_NAME}" \
    --reasoning-effort  "${REASONING_EFFORT}" \
    --port              "${VLLM_PORT}" \
    --json-dir          "${JSON_DIR}" \
    --results-dir       "${RESULTS_DIR}" \
    --budget            "${BUDGET}" \
    --context-mode      "${CONTEXT_MODE}" \
    --max-guidelines    "${MAX_GUIDELINES}" \
    --num-distractors   "${NUM_DISTRACTORS}" \
    --oracle-mode       "${ORACLE_MODE}" \
    --max-concurrent    "${MAX_CONCURRENT}" \
    --seed              "${SEED}" \
    ${CACHE_TAG:+--cache-tag "${CACHE_TAG}"} \
    "$@"; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

echo ""
echo "============================================================"
echo "  Evaluation finished  (exit code: ${EXIT_CODE})"
echo "  Results: ${RESULTS_DIR}"
echo "============================================================"
exit "${EXIT_CODE}"
