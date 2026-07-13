# GSME-Q-MT data generation

`data/gsme_q_mt.csv` is produced by `build_min_suff_gsm_dataset.py` on top of
the GSM-Q CSP source from
[QuestBench](https://github.com/google-deepmind/questbench). See the paper
for the algorithmic description and the source code for implementation
details.

## How to run

Download `questbench_data.tar.gz` from QuestBench, extract it, and place
`GSM-Q/gsm_CSP_full.csv` under `data/`. Then:

```bash
bash gsme_q_mt/data_generation/base/run.sh
```

Useful overrides:

```bash
INPUT_GSM_CSP=data/gsm_CSP_full.csv \
INTERMEDIATE_DIR=logs/gsme_q_mt \
OUTPUT_CSV=data/gsme_q_mt.csv \
bash gsme_q_mt/data_generation/base/run.sh
```
