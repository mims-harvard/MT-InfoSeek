# GSME-Q-MT (+ GSME-Q-MT-Ext)

**GSME-Q-MT** extends QuestBench's GSM-Q grade-school math word problems
to a multi-turn setting by
systematically masking multiple quantities so the model must ask for several
missing values before producing an answer.

**GSME-Q-MT-Ext** is an enriched variant generated from symbolic-arithmetic
dependency graphs with deeper dependencies and larger variable sets. 

Released files: [data/gsme_q_mt.csv](../data/gsme_q_mt.csv)
and [data/gsme_q_mt_ext.csv](../data/gsme_q_mt_ext.csv). See
`Setup` in the [root README](../README.md) for env, API keys, and vLLM.

## Running the evaluators

Ask one variable per turn (sequential task-solving):

```bash
MODEL=gpt-5-mini bash gsme_q_mt/scripts/run_multiturn.sh
```

Ask for all missing variables in a single turn (paper's k-MSS-identification
baseline):

```bash
MODEL=gpt-5-mini bash gsme_q_mt/scripts/run_singleturn.sh
```

For GSME-Q-MT-Ext, override `DATA_FILE=gsme_q_mt_ext.csv` in either script.
Both scripts honor `BUDGET`, `BATCH_SIZE`, `RESULTS_DIR`, plus the standard
backend variables (`VLLM_PORT`, `REASONING_EFFORT`, …).

Logs land in `logs/gsme/`.

## Regenerate the dataset

GSME-Q-MT is built from QuestBench's GSM-Q CSP source
([questbench_data.tar.gz](https://storage.googleapis.com/questbench/questbench_data.tar.gz)).
Place `GSM-Q/gsm_CSP_full.csv` under `data/`, then:

```bash
bash gsme_q_mt/data_generation/base/run.sh
```

GSME-Q-MT-Ext is generated from symbolic dependency graphs:

```bash
python gsme_q_mt/data_generation/ext/generate_complex_gsme.py \
    --num_samples 1000 --output_dir logs/gsme_q_mt_ext \
    --seed 42 --allow_distractors --k_max 4 \
    --holdout_candidates 200 --holdout_top_m 5 \
    --holdout_difficulty_mode distance
```

See `data_generation/base/README.md` and `data_generation/ext/README.md`
for details.
