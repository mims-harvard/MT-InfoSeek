# 20 Questions

The classic 20 Questions game as an open-domain information-seeking setting.
The hidden target $Y$ is sampled from a finite candidate set; the model
deduces it by generating natural-language yes/no questions.

Two candidate pools — **common** (animals, places, food, objects) and
**thing** (THING200) — are defined in [data/data_20q.py](../data/data_20q.py).
`python run_eval.py --datasets 20q --full ...` evaluates both pools; the smoke
run uses the first six Common targets only.
The importable Python package is `20q/twenty_questions/`. See
[SETUP.md](../SETUP.md) for env, API keys, and vLLM.

## Run inference

```bash
GUESSER_MODEL=gpt-5-mini EXAMINER_MODEL=gpt-5-mini DATASET=common \
  bash 20q/scripts/run_inference.sh

GUESSER_MODEL=gpt-5-mini EXAMINER_MODEL=gpt-5-mini DATASET=thing \
  bash 20q/scripts/run_inference.sh
```

Logs: `logs/20q/`.

## Offline question-quality evaluation

We measure how much each question reduces uncertainty (normalized
information gain, remaining entropy, pass mass); see paper §D.5. The paper
uses *Qwen3-30B-A3B-Instruct-2507* or *gpt-5-mini* as the examiner.

`run_eval.py` runs this stage automatically after 20Q inference, using the 20Q
examiner as the offline judge, and includes question EIG/final entropy in
`summary.md`. Use `--20q-judge-model` to choose another judge or
`--no-20q-offline` to skip it. Keep the judge fixed when comparing tested
models. Both reasoning and non-reasoning judges are supported. Prefer a fixed
API judge such as `gpt-5-mini`; for a reasoning judge, leave enough output
budget for its reasoning plus the structured labels. The commands below are
for running it directly.

```bash
# Local judge (start a vLLM server first):
vllm serve /path/to/Qwen-30B-Instruct --port 8025

INPUT_LOG_PATH=/path/to/20q-dialogues.json \
OUTPUT_PATH=/path/to/question-quality.jsonl \
EVAL_MODEL=/path/to/Qwen-30B-Instruct \
EVAL_BASE_URL=http://127.0.0.1:8025/v1 POOL=THING200_EVAL_POOL \
JUDGE_MAX_TOKENS=32768 JUDGE_BATCH_SIZE=4 \
  bash 20q/scripts/run_offline_evaluation.sh

# Hosted judge:
INPUT_LOG_PATH=/path/to/20q-dialogues.json \
OUTPUT_PATH=/path/to/question-quality.jsonl \
EVAL_MODEL=gpt-5-mini JUDGE_BACKEND=hosted_gpt POOL=THING200_EVAL_POOL \
  bash 20q/scripts/run_offline_evaluation.sh

# For Common, set POOL=COMMON_EVAL_POOL.
```

`INPUT_LOG_PATH`, `OUTPUT_PATH`, and `EVAL_MODEL` are required for a direct run.
The judge defaults to a 32,768-token generation budget and batches four
candidates per request. Increase `JUDGE_MAX_TOKENS` if a reasoning judge still
runs out of budget before emitting all labels.
