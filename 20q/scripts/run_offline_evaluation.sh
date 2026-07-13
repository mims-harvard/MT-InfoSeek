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

: "${INPUT_LOG_PATH:?Set INPUT_LOG_PATH to the completed 20Q dialogue JSON}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH for the offline-evaluation JSONL}"
: "${EVAL_MODEL:?Set EVAL_MODEL to the offline judge model}"

JUDGE_BACKEND="${JUDGE_BACKEND:-openai_compatible}"
# Ordinary local vLLM servers do not need a key. "EMPTY" only satisfies the
# OpenAI client's required constructor argument. Protected remote endpoints can
# set EVAL_API_KEY explicitly (the wrapper resolves api_key_env for them).
EVAL_BASE_URL="${EVAL_BASE_URL:-${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8011}/v1}}"
EVAL_API_KEY="${EVAL_API_KEY:-EMPTY}"

POOL="${POOL:-COMMON_EVAL_POOL}"
CACHE_PATH="${CACHE_PATH:-${OUTPUT_PATH%.jsonl}_cache.json}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-32768}"
TOPK_PRUNE="${TOPK_PRUNE:-50}"

mkdir -p "$(dirname "$CACHE_PATH")" "$(dirname "$OUTPUT_PATH")"
PYTHONPATH="${ROOT_DIR}:${DATASET_DIR}:${PYTHONPATH:-}" \
"${PYTHON:-python}" "${DATASET_DIR}/twenty_questions/offline_evaluation/offline_evaluator.py" \
  --input "$INPUT_LOG_PATH" \
  --output "$OUTPUT_PATH" \
  --pool "$POOL" \
  --infer-pool \
  --judge-backend "$JUDGE_BACKEND" \
  --model "$EVAL_MODEL" \
  --base-url "$EVAL_BASE_URL" \
  --api-key "$EVAL_API_KEY" \
  --cache-path "$CACHE_PATH" \
  --judge-batch-size "$JUDGE_BATCH_SIZE" \
  --topk-prune "$TOPK_PRUNE" \
  --temperature 0.0 \
  --max-tokens "$JUDGE_MAX_TOKENS"
