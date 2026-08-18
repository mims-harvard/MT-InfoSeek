# Logic-Q-MT

We extend the propositional logical CSPs from
[QuestBench](https://github.com/google-deepmind/questbench) to a multi-turn setting in which the model must
identify and query the multiple missing facts needed to prove or disprove
a target proposition. Each problem is paired with a
$k$-Minimal Sufficient Set ($k$-MSS) for $k \in \{1, 2, 3\}$.

Released benchmark: [data/logic_q_mt.csv](../data/logic_q_mt.csv). See
[SETUP.md](../SETUP.md) for environment, API keys, and the local vLLM server.

## Sequential task-solving (multi-turn)

```bash
MODEL=gpt-5-mini BUDGET=10 ORACLE=adversarial \
  bash logic_q_mt/scripts/run_multiturn.sh
```

| variable | default | notes |
| --- | --- | --- |
| `BUDGET` | `10` | max questions; `0` switches to the legacy single-turn bulk protocol. Use `BUDGET_IN_PROMPT=false` to hide the budget instead |
| `ORACLE` | `adversarial` | `adversarial` / `cooperative` / `random` (see paper §C.4) |
| `BUDGET_IN_PROMPT` | `true` | `false` hides the budget from the prompt |
| `KEEP_THINKING_TRACE` | `false` | carry CoT content across turns (Qwen models only; raises `NotImplementedError` otherwise) |
| `MAX_CONCURRENT` | `16` | client concurrency |
| `VLLM_PORT` | unset | local vLLM |
| `REASONING_EFFORT` | unset | gpt-oss and gpt-5.2/5.4 models (ignored for gpt-5 / gpt-5-mini) |

Logs land in `$RESULTS_DIR/<model>/`. `setup.sh` writes `RESULTS_DIR=<repo>/logs`
into `.env`, so that is `logs/<model>/` unless you override it.

## Single-turn protocols (MSS identification, k-prediction, full-info)

```bash
MODEL=gpt-5-mini EVAL_MODE=mc PROMPT_MODE=exact_k \
  bash logic_q_mt/scripts/run_singleturn.sh
```

| `EVAL_MODE` | paper protocol |
| --- | --- |
| `mc` | **MSS identification** — pick the k-MSS from candidate variables. |
| `isambig` | **Degree of underspecification** (k-prediction). |
| `fullinfo` | full-information sanity check. |

`PROMPT_MODE`: `exact_k` (exact k revealed) or `at_most_K` (upper bound).
Additional knobs: `BATCH_SIZE`, `FORBID_ALTERNATIVES`, `VLLM_PORT`,
`REASONING_EFFORT`, `MAX_TOKENS`.

## Regenerate the dataset

See details in [data_generation/README.md](data_generation/README.md).
