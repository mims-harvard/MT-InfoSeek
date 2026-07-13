# MT-InfoSeek Data

This directory contains the released, non-clinical benchmark data used by
MT-InfoSeek. The tasks evaluate whether a model can recognize missing
information, ask targeted follow-up questions, and answer only after it has
enough evidence.

The runnable smoke subsets live in [smoke/](smoke/) and are used by
`python run_eval.py --datasets all --model <model>` by default. Pass `--full`
to evaluate the released full-data split.

## Files

| file | task | format | notes |
| --- | --- | --- | --- |
| [logic_q_mt.csv](logic_q_mt.csv) | Logic-Q-MT | CSV | Propositional-logic instances with hidden facts, rules, admissible queries, and minimal sufficient query sets. |
| [gsme_q_mt.csv](gsme_q_mt.csv) | GSME-Q-MT | CSV | Grade-school math word problems converted to symbolic CSPs with multiple held-out quantities. |
| [gsme_q_mt_ext.csv](gsme_q_mt_ext.csv) | GSME-Q-MT-Ext | CSV | Structurally richer symbolic arithmetic graphs with distractor variables and additional graph metadata. |
| [genereg_mt.jsonl](genereg_mt.jsonl) | GeneReg-MT | JSONL | Boolean gene-regulatory-network tasks with observed gene values, feasible target outcomes, and minimal sufficient gene-query sets. |
| [data_20q.py](data_20q.py) | 20 Questions | Python source | Candidate pools for Common, THING200, BIG-bench concepts, and category-specific pools. |
| [smoke/](smoke/) | smoke subsets | mixed | Small subsets for setup validation and fast end-to-end checks. |

The full evaluation uses Logic-Q-MT, GSME-Q-MT, GSME-Q-MT-Ext, the GeneReg-MT
task families selected by `run_eval.py`, and both 20Q Common and THING200. The
smoke evaluation uses small representative subsets and only the first six 20Q
Common targets.

## Task Descriptions

### Logic-Q-MT

Logic-Q-MT extends QuestBench Logic-Q to a multi-turn setting. Each row is a
Horn-SAT style problem where some propositional facts are hidden. The model sees
known true/false facts, a target proposition, and rules. It must ask for enough
hidden facts to determine the target truth value.

Fields:

- `k`: number of hidden facts in the canonical minimal sufficient set.
- `known_facts`, `known_untrue_facts`: facts initially revealed to the model.
- `rules`: conjunctive implication rules.
- `goal`: target proposition.
- `gt_qs`: canonical sufficient queries.
- `all_valid_qs_forbid_alternatives`: admissible queries under the stricter
  forbid-alternatives evaluation mode.

### GSME-Q-MT and GSME-Q-MT-Ext

GSME-Q-MT converts GSM-style math word problems into symbolic constraint
systems and masks multiple quantities. The model must ask for missing variable
values before solving for the final answer.

GSME-Q-MT-Ext keeps the same high-level task but uses richer generated
dependency graphs, larger variable sets, distractors, and graph-structure
metadata.

Common fields include:

- `Full Problem`: original word problem.
- `Rewritten Problem`: under-specified version shown to the model.
- `CSP`, `Variables`, `Equations`: symbolic representation.
- `Possible Questions`: admissible follow-up questions.
- `Given_Conditions`: information already available.
- `k`: number of missing variables.
- `goal_var`: variable needed for the final answer.

### GeneReg-MT

GeneReg-MT builds information-seeking tasks from published Boolean
gene-regulatory-network models. A row describes an observed partial
gene-expression assignment and the target outcome to determine. The evaluator
asks the model to query additional gene values until the target is uniquely
identified.

Fields:

- `model`, `n_nodes`, `var_names`: source Boolean network metadata.
- `target_type`: type of target being determined.
- `observed`: gene values initially revealed to the model.
- `minimal_sufficient_sets`: gene sets sufficient to determine the target.
- `feasible_target_values`: target outcomes still compatible with the observed
  values.
- `branches`: representative target-specific states used by the evaluator.

GeneReg assets needed to regenerate or evaluate the tasks are handled by
[../setup.sh](../setup.sh) and documented in [../SETUP.md](../SETUP.md).

### 20 Questions

20 Questions uses finite candidate pools instead of row-wise CSV/JSON records.
The guesser asks yes/no questions about a hidden target sampled from a pool. The
offline evaluator can score question quality by estimating how much each
question reduces uncertainty over the candidate set.

The public pools are defined in [data_20q.py](data_20q.py):

- `COMMON`: combined common Animals, Places, Food, and Objects concepts.
- `THING200`: object/entity pool used for the full 20Q evaluation.

## Regeneration

Each task's regeneration pipeline lives with the corresponding task code:

- [../logic_q_mt/data_generation/README.md](../logic_q_mt/data_generation/README.md)
- [../gsme_q_mt/data_generation/base/README.md](../gsme_q_mt/data_generation/base/README.md)
- [../gsme_q_mt/data_generation/ext/README.md](../gsme_q_mt/data_generation/ext/README.md)
- [../genereg_mt/data_generation/README.md](../genereg_mt/data_generation/README.md)

The released benchmark files in this directory are the files consumed by
`run_eval.py`; regeneration is optional.
