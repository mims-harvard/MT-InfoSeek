# GeneReg-MT

We construct biological information-seeking tasks from Boolean gene
regulatory networks (122 published networks from
[Kadelka et al. 2024](https://doi.org/10.1038/s41597-024-03900-1)).
Variables are initial gene-expression states, constraints are Boolean update rules,
and the target Y is either a steady state (**steady-state identification**)
or a marker-gene value at convergence (**marker identification**). The
model must query informative initial-state gene values until Y is uniquely
determined.

Released benchmark: [data/genereg_mt.jsonl](../data/genereg_mt.jsonl). Each
row's `branches` list one representative state per feasible target outcome,
which is used in multi-turn evaluation. See [SETUP.md](../SETUP.md) for the
environment, credentials, and local model serving.

## Required external data

```bash
bash setup.sh
bash setup.sh --verify
```

The setup script extracts the released attractor/basin cache and checks out the
Kadelka models at the pinned revision used by the evaluator. Its default paths
are `.grn_cache/` and `.kadelka/` in the repository root.

## Sequential task-solving (multi-turn)

```bash
MODEL=gpt-5-mini BUDGET=10 ORACLE=adversarial \
  bash genereg_mt/scripts/run_multiturn.sh
```

| variable | default | notes |
| --- | --- | --- |
| `BUDGET` | `10` | max questions; `0` removes the budget signal |
| `ORACLE` | `adversarial` | `adversarial` / `cooperative` / `random` (see paper §C.4) |
| `INCLUDE_TASKS` | `dyn_attr,dyn_marker` | task types used in the paper |
| `SAVE_TURN_PROMPTS` | `false` | dump per-turn prompts |
| `MAX_CONCURRENT` | `16` | client concurrency |
| `VLLM_PORT` | unset | local vLLM |
| `REASONING_EFFORT` | unset | gpt family models |

Logs: `logs/genereg_mt/multiturn/`.

## Full-information sanity check

```bash
MODEL=gpt-5-mini bash genereg_mt/scripts/run_fullinfo.sh
```

## Regenerate the dataset

```bash
bash genereg_mt/data_generation/scripts/run_data_gen.sh
```

This calls
[`grn_dataset_curator.py`](data_generation/grn_dataset_curator.py). The
per-model attractor / basin cache is written to `$CACHE_DIR` and reused on
subsequent runs and by the three verifiers (`verify_tasks`,
`verify_k_minimality`, `verify_exhaustiveness`). See
[data_generation/README.md](data_generation/README.md) for the full
pipeline.
