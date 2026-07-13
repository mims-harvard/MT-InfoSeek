# Logic-Q-MT data generation

The pipeline runs from QuestBench's raw Logic-Q-RP rulesets to the released
`data/logic_q_mt.csv`. Stages 1 and 2 operate **per shard directory**
(`$LOGIC_Q_INTERMEDIATE_DIR/new_<i>_500k/`, one raw ruleset file each); stage 3
is the postprocessing notebook, which internally calls
`add_all_alternative_gts.py` as a sub-step.

```
prepare_shards.py                      (0) split raw rulesets into new_<i>_500k/ shard dirs
SimpleLogic/generate_ruleset_new.py    (1) per shard: generate per-ruleset k-sufficient sets
SimpleLogic/make_data_new.py           (2) per shard: flatten + filter + per-ruleset subsample
data_postprocessing.ipynb              (3) paper-level filter + 200-per-k subsample
  └─ SimpleLogic/add_all_alternative_gts.py   (sub-step) annotate all alternative k-MSSs
```

See the paper for the algorithmic details (recursive construction of k-MSSs,
Horn-SAT sufficiency / global-minimality certification, feasibility-aware
validation).

> **Note on reproducibility.** The released `data/logic_q_mt.csv` is the
> canonical artifact of record. The pipeline below reproduces a dataset of the
> same construction; exact byte-for-byte rows additionally depend on the shard
> set and library versions. Every released row is independently re-verified for
> k-sufficiency and global minimality (stage 3 / `verify_results.py`), so
> validity does not depend on exact regeneration.

## Inputs

Download QuestBench's Logic-Q-RP rulesets from
[questbench_data.tar.gz](https://storage.googleapis.com/questbench/questbench_data.tar.gz),
extract `Logic-Q/RP/RP/`, and point `LOGIC_Q_RP_DIR` at it (in `.env`). Stage-1/2
outputs and the notebook's intermediates live under `LOGIC_Q_INTERMEDIATE_DIR`
(also in `.env`).

## How to run

```bash
# Stage 0 — split the raw rulesets into one-file-per-shard dirs
#   new_0_500k/ .. new_<N-1>_500k/ under $LOGIC_Q_INTERMEDIATE_DIR.
# We used 30 shards for the released data; drop --max_shards to use all files.
python logic_q_mt/data_generation/prepare_shards.py \
    --rp_dir "$LOGIC_Q_RP_DIR" \
    --intermediate_dir "$LOGIC_Q_INTERMEDIATE_DIR" \
    --suffix 500k --max_shards 30

# Stage 1 — generate k-sufficient sets, once per shard dir.
# For the full parallel run (forks one process per 100-ruleset window) use:
bash logic_q_mt/data_generation/scripts/run_logic_gen_parallel_more_props.sh 0 29 500000
# ...or, to run a single shard serially:
PYTHONPATH=logic_q_mt/data_generation python \
    logic_q_mt/data_generation/SimpleLogic/generate_ruleset_new.py \
    --sl_dir "$LOGIC_Q_INTERMEDIATE_DIR/new_0_500k" \
    --start_idx 0 --end_idx 7000 --max_k 4 --max_expansions_per_layer 500000

# Stage 2 — flatten + filter + per-ruleset subsample, once per shard dir.
for d in "$LOGIC_Q_INTERMEDIATE_DIR"/new_*_500k; do
    PYTHONPATH=logic_q_mt/data_generation python \
        logic_q_mt/data_generation/SimpleLogic/make_data_new.py \
        --sl_dir "$d" \
        --max_problems_to_sample_per_ruleset 50
done

# Stage 3 — open and run data_postprocessing.ipynb top to bottom. It reads the
# per-shard CSVs from $LOGIC_Q_INTERMEDIATE_DIR/new_*_500k, runs
# add_all_alternative_gts.py (one cell; can also be run by hand — see that cell),
# and writes $DATA_DIR/logic_q_mt.csv.
```

`--max_problems_to_sample_per_ruleset` controls only the candidate-pool size fed
into stage 3's downsampling (it does not affect per-row validity); `50` is the
script default and is more than enough given the number of rulesets per shard.
