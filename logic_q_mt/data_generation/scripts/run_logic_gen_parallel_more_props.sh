#!/usr/bin/env bash
# Parallel template for stage-1 of Logic-Q-MT data generation
# (SimpleLogic/generate_ruleset_new.py). Forks one process per 100-ruleset
# window. Tune the inner range and the outer loop to your hardware.
#
# Usage:
#   bash logic_q_mt/data_generation/scripts/run_logic_gen_parallel_more_props.sh \
#        <start_shard> <end_shard> <max_expansions_per_layer>
#   Example: bash ... 1 10 500000
#
# Required env: LOGIC_Q_INTERMEDIATE_DIR (root directory containing the
# per-shard subdirs new_<i>_500k/, each pre-populated with the raw
# prop_examples_*.txt rulesets from QuestBench Logic-Q/RP/RP). See
# .env.example.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <start_shard> <end_shard> <max_expansions_per_layer>"
    exit 1
fi

START_NUM=$1
END_NUM=$2
MAX_EXPANSIONS_PER_LAYER=$3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Pull env vars from .env at repo root if present (see .env.example).
_ENV_FILE="${ROOT_DIR}/.env"
[ -f "$_ENV_FILE" ] && set -a && source "$_ENV_FILE" && set +a

SL_BASE="${LOGIC_Q_INTERMEDIATE_DIR:?LOGIC_Q_INTERMEDIATE_DIR is required; see .env.example}"
LOG_BASE="${RESULTS_DIR:-${ROOT_DIR}/logs}/logic_q_gen"

if   [ "$MAX_EXPANSIONS_PER_LAYER" -eq 500000  ]; then SUFFIX="500k"
elif [ "$MAX_EXPANSIONS_PER_LAYER" -eq 1000000 ]; then SUFFIX="1m"
else echo "Unsupported max_expansions_per_layer: $MAX_EXPANSIONS_PER_LAYER"; exit 1
fi

# Restrict each forked Python process to one CPU core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$ROOT_DIR"

for ((i=START_NUM; i<=END_NUM; i++)); do
    SL_DIR="${SL_BASE}/new_${i}_${SUFFIX}"
    LOG_DIR="${LOG_BASE}/new_${i}_${SUFFIX}"
    mkdir -p "$LOG_DIR"

    echo "Processing shard ${i} (${SL_DIR})"
    for start_idx in {0..6900..100}; do
        end_idx=$((start_idx + 100))
        PYTHONPATH="${ROOT_DIR}/logic_q_mt/data_generation" python \
            "${ROOT_DIR}/logic_q_mt/data_generation/SimpleLogic/generate_ruleset_new.py" \
            --sl_dir   "$SL_DIR" \
            --start_idx "$start_idx" \
            --end_idx   "$end_idx" \
            --max_k 4 \
            --max_expansions_per_layer "$MAX_EXPANSIONS_PER_LAYER" \
            > "${LOG_DIR}/run_${start_idx}_${end_idx}.log" 2>&1 &
    done
    wait
    echo "Finished shard ${i}"
done

echo "All shards ${START_NUM}..${END_NUM} completed."
