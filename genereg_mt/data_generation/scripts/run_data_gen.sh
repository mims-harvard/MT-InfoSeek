#!/usr/bin/env bash
# Regenerate `data/genereg_mt.jsonl` (and per-model caches) from the upstream
# Kadelka Boolean GRN models. Requires MODELS_DIR (clone of
# https://github.com/ckadelka/DesignPrinciplesGeneNetworks; point at the
# `update_rules_122_models_Kadelka_SciAdv/` subdir) and writes outputs under
# DATA_DIR. See .env.example.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Pull env vars from .env at repo root if present (see .env.example).
_ENV_FILE="${ROOT_DIR}/.env"
[ -f "$_ENV_FILE" ] && set -a && source "$_ENV_FILE" && set +a
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
MODELS_DIR="${MODELS_DIR:?MODELS_DIR is required; see .env.example}"
CACHE_DIR="${CACHE_DIR:-${DATA_DIR}/grn_models_cache}"
LOG_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs}/grn_multi_data"

mkdir -p "$CACHE_DIR" "$LOG_DIR"
cd "${ROOT_DIR}/genereg_mt/data_generation"

python grn_dataset_curator.py \
    --models_dir "$MODELS_DIR" \
    --out_dir    "$DATA_DIR" \
    --cache_dir  "$CACHE_DIR" \
    --log_dir    "$LOG_DIR" \
    --tasks_filename genereg_mt.jsonl \
    --n_groups 1200 \
    --quota_per_task_k 100 \
    --exclude_tasks ss_id,ss_marker \
    --seed 42 \
    --max_k 4 \
    --ss_min_fp 16 \
    --ss_fp_cap 512 \
    --ss_max_fp_for_tasks 512 \
    --ss_max_nodes 64 \
    --dyn_max_nodes 21 \
    --dyn_max_attractors 256 \
    --dyn_free_bits_max 14 \
    --min_extra_queryable_after_forbid 6 \
    --min_stable_extra_queryable_dyn_marker 3 \
    --min_potential_outcomes_ss_id 4 \
    --min_potential_outcomes_dyn_attr 4 \
    --timeout_seconds 300 \
    --progress_seconds 15 \
    --progress_top_cells 10
