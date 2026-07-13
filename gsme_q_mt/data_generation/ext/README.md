# GSME-Q-MT-Ext data generation

`data/gsme_q_mt_ext.csv` is generated from synthetic dependency DAGs with
integer-friendly arithmetic rules, validated for unique solvability. See the
paper for the algorithmic description and the source code for implementation
details.

## How to run

```bash
python gsme_q_mt/data_generation/ext/generate_complex_gsme.py \
    --num_samples 1000 --output_dir logs/gsme_q_mt_ext \
    --seed 42 --allow_distractors --k_max 4 \
    --holdout_candidates 200 --holdout_top_m 5 \
    --holdout_difficulty_mode distance
```

Writes `gsme_q_mt_ext.jsonl` (one structured sample per line) and
`gsme_q_mt_ext.csv` (flattened metadata) into `--output_dir`, plus a
`summary.json`.

Optional follow-up utilities:

```bash
# Sample rows into a markdown audit report
python gsme_q_mt/data_generation/ext/audit_samples.py \
    --input  logs/gsme_q_mt_ext/gsme_q_mt_ext.csv \
    --output logs/gsme_q_mt_ext/gsme_q_mt_ext.audit.md \
    --num_samples 12 --sample_mode balanced --seed 42

# Convert to the 21-column reference schema used in the released CSV
python gsme_q_mt/data_generation/ext/convert_to_reference_format.py \
    --input  logs/gsme_q_mt_ext/gsme_q_mt_ext.csv \
    --output logs/gsme_q_mt_ext/gsme_q_mt_ext_final.csv
```
