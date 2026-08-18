# GeneReg-MT data generation

The released `data/genereg_mt.jsonl` is produced by `grn_dataset_curator.py`
from the 122 Boolean GRN models of Kadelka et al. (2024). The curator
precomputes per-network attractor / basin caches, filters eligible models,
and constructs minimal k-sufficient sets for the steady-state identification
and marker identification settings. See the paper for the algorithmic
description and the source code for implementation details.

## Output schema

Each row corresponds to one problem. Key fields:

- **Prompt catalog** (shown to the model): `prompt_catalog_size`.
- **Feasible target outcomes** under the current observed context:
  `feasible_target_values`, `n_feasible_target_values`, `branches`.
  `branches` stores one representative state per feasible outcome;
  multi-turn evaluation expands rows into one episode per branch.
- **k-MSSs**: `k_min`, `minimal_sufficient_sets`,
  `minimal_sufficient_sets_idx`.

## How to run

```bash
bash genereg_mt/data_generation/scripts/run_data_gen.sh
```

Requires `MODELS_DIR` (path to the Kadelka
`update_rules_122_models_Kadelka_SciAdv/` clone) and a writable `CACHE_DIR`
(per-model attractor / basin caches; ~50 MB once populated). Both come from
`.env`. The wrapper already passes `--tasks_filename genereg_mt.jsonl`, so output
lands at `$DATA_DIR/genereg_mt.jsonl` (the curator's own default is `tasks_new.jsonl`).
Note this overwrites the released dataset file.

## Verification

```bash
PYTHONPATH=. python verify_tasks.py        # underspecification + sufficiency + local minimality
PYTHONPATH=. python verify_k_minimality.py # global minimality (brute force, slower)
PYTHONPATH=. python verify_exhaustiveness.py
```

All three read `DATA_DIR`, `MODELS_DIR` and `CACHE_DIR` from the environment, so
export them first (`set -a; source ../../.env; set +a`). Override per script with
`--jsonl` / `--models` / `--cache` (verify_tasks, verify_exhaustiveness) or
`--tasks_file` / `--models_dir` / `--cache_dir` (verify_k_minimality).
