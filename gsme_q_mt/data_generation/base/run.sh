#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_GSM_CSP="${INPUT_GSM_CSP:-${ROOT_DIR}/data/gsm_CSP_full.csv}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-${ROOT_DIR}/logs/gsme_q_mt}"
OUTPUT_CSV="${OUTPUT_CSV:-${ROOT_DIR}/data/gsme_q_mt.csv}"

mkdir -p "$INTERMEDIATE_DIR" "$(dirname "$OUTPUT_CSV")"

python "${SCRIPT_DIR}/build_min_suff_gsm_dataset.py" \
  --data_name "$INPUT_GSM_CSP" \
  --output_dir "$INTERMEDIATE_DIR" \
  --k_max 4 \
  --N_candidates 30 \
  --M_keep 1 \
  --difficulty_mode distance \
  --timeout_sec 20 \
  --seed 0

python "${SCRIPT_DIR}/clean_verbal.py" \
  --input "${INTERMEDIATE_DIR}/gsm_CSP_full_ksufficient_kmax4_N30_M1.csv" \
  --output "$OUTPUT_CSV"
