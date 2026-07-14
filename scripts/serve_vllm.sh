#!/usr/bin/env bash
# Convenience vLLM launcher for the models we evaluated (Qwen, gpt-oss).
#
# New models may need a different chat template, reasoning parser, quantization,
# or tensor-parallel configuration. Copy and adapt this script as needed.
#
# The suite only asks for an OpenAI-compatible /v1/chat/completions endpoint and
# the port it's on (VLLM_PORT, or VLLM_BASE_URL). Start ONE server for the whole
# run and reuse it across datasets.
#
# ── GPU requirements ─────────────────────────────────────────────────────────
#   Defaults are the paper's configuration and assume an H100 (or newer):
#   `--kv-cache-dtype fp8` needs compute capability >= 8.9.
#   On an older GPU (e.g. A100), disable that optimisation:
#       KV_CACHE_DTYPE=auto bash scripts/serve_vllm.sh <model>
#
# ── Usage ────────────────────────────────────────────────────────────────────
#   bash scripts/serve_vllm.sh Qwen/Qwen3.5-4B
#   VLLM_PORT=8011 TP_SIZE=4 bash scripts/serve_vllm.sh Qwen/Qwen3.5-122B-A10B-FP8
#
#   # A new model with compatible defaults and an explicit parser
#   REASONING_PARSER=qwen3 bash scripts/serve_vllm.sh my-org/custom-7b
#
# The reasoning parser for known models is looked up from model_registry.py.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model-path-or-hf-id>" >&2
    exit 1
fi

MODEL_NAME="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${VLLM_BIN:-}" ]; then
    : # explicit override
elif [ -x "${ROOT_DIR}/.venv-vllm/bin/vllm" ]; then
    VLLM_BIN="${ROOT_DIR}/.venv-vllm/bin/vllm"
elif command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
else
    echo "ERROR: vLLM is not installed." >&2
    echo "       On the GPU machine, run: bash setup.sh --with-vllm" >&2
    exit 1
fi

_ENV_FILE="${ROOT_DIR}/.env"
# .env supplies defaults; anything already exported (e.g. by run_eval.py) wins.
# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/load_env.sh" 2>/dev/null \
  || source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/load_env.sh"
_load_env "$_ENV_FILE"

VLLM_PORT="${VLLM_PORT:-8011}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
GPU_DEVICES="${VLLM_GPU_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.95}"
# Must stay >= ~80k: the largest per-dataset output budget is 65,536 tokens
# (fixed for every model, so results stay comparable), plus room for the prompt.
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-16384}"

# fp8 KV cache is the paper configuration and needs compute capability >= 8.9
# (H100+). On older GPUs (e.g. A100), set KV_CACHE_DTYPE=auto.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# Reasoning parser: explicit override, else look it up in the shared registry.
if [ -z "${REASONING_PARSER:-}" ]; then
    if [ -z "${PYTHON:-}" ]; then
        if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
            PYTHON="${ROOT_DIR}/.venv/bin/python"
        elif command -v python >/dev/null 2>&1; then
            PYTHON="$(command -v python)"
        elif command -v python3 >/dev/null 2>&1; then
            PYTHON="$(command -v python3)"
        fi
    fi
    if [ -z "${PYTHON:-}" ]; then
        echo "ERROR: could not find Python to infer the vLLM reasoning parser." >&2
        echo "       Run bash setup.sh first, or set REASONING_PARSER=<parser|none>." >&2
        exit 1
    fi
    if ! REASONING_PARSER="$(PYTHONPATH="${ROOT_DIR}" "${PYTHON}" -c "
import sys
import model_registry
p = model_registry.get_reasoning_parser(sys.argv[1])
print(p or 'none')
" "$MODEL_NAME")"; then
        echo "ERROR: failed to infer the vLLM reasoning parser with ${PYTHON}." >&2
        echo "       Set REASONING_PARSER=<parser|none> explicitly to override." >&2
        exit 1
    fi
fi

echo "============================================================"
echo "  vLLM server"
echo "    executable       : ${VLLM_BIN}"
echo "    model            : ${MODEL_NAME}"
echo "    host:port        : ${VLLM_HOST}:${VLLM_PORT}"
echo "    GPUs             : ${GPU_DEVICES} (tensor-parallel ${TP_SIZE})"
echo "    python           : ${PYTHON:-not used}"
echo "    reasoning-parser : ${REASONING_PARSER}"
echo "    kv-cache-dtype   : ${KV_CACHE_DTYPE}"
echo "============================================================"
if [ "$REASONING_PARSER" = "none" ]; then
    echo "NOTE: no reasoning parser. Thinking tokens will not be separated."
    echo "      Set REASONING_PARSER=<qwen3|openai_gptoss|deepseek_r1> if the model reasons."
fi

# Reject an occupied port to avoid evaluating against another server.
if command -v lsof >/dev/null 2>&1 && lsof -i :"${VLLM_PORT}" >/dev/null 2>&1; then
    echo "ERROR: port ${VLLM_PORT} is already in use (lsof -i :${VLLM_PORT})." >&2
    exit 1
fi

ARGS=(
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --no-enable-prefix-caching
    --enable-chunked-prefill
    --async-scheduling
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
)
[ "$REASONING_PARSER" != "none" ] && ARGS+=(--reasoning-parser "${REASONING_PARSER}")

exec env CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" "${VLLM_BIN}" serve "${MODEL_NAME}" "${ARGS[@]}"
