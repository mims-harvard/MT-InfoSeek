# Setup

## 1. Environment

```bash
bash setup.sh
```

This creates `.env`, creates `.venv`, installs `requirements.txt`, and prepares
the GeneReg-MT assets. Use `bash setup.sh --no-genereg` when GeneReg-MT is not
needed.
`bash setup.sh --verify` checks the environment and assets without changing
them. The script requires [uv](https://docs.astral.sh/uv/).

Use `.venv/bin/python run_eval.py ...`, or activate the environment with
`source .venv/bin/activate`.

The `.env` file is required and is created automatically from `.env.example` on
the first setup run. `setup.sh` fills `PROJECT_ROOT`; edit `.env` only for
credentials, path overrides, or a local model endpoint.

For reproducibility the end-to-end run used Python 3.10.16, NumPy 2.2.6, pandas 2.3.3,
OpenAI 2.45.0, Transformers 5.13.1, PyYAML 6.0.3, tqdm 4.68.4, and NetworkX
3.4.2. The server was a development build of vLLM
`0.22.1rc1.dev116+g71df063c4`.

## 2. Credentials

Edit `.env`:

- **GPT models** — by default served via Azure OpenAI:
  ```bash
  export AZURE_OPENAI_API_KEY=...
  export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
  ```
  Alternatively, use the standard OpenAI API:
  ```bash
  export OPENAI_API_KEY=...
  # optional: an OpenAI-compatible gateway (OpenRouter / Ollama / SGLang)
  # export OPENAI_BASE_URL=https://openrouter.ai/api/v1
  ```
- **Gemini** — `export GOOGLE_API_KEY=...` (or Vertex AI; see `.env.example`).
- **Local open-weight models** — nothing here; serve with vLLM (step 4).

## 3. GeneReg-MT assets

The default `bash setup.sh` command extracts the committed attractor/basin cache
and sparse-clones the Kadelka Boolean-network models at a pinned commit. The
default paths require no `.env` changes. Pass `--cache-dir` and `--models-dir`
to `run_eval.py` to use other locations.

## 4. Serving your local model

Skip this section if only API models are used.

Start the model server separately. Chat template, reasoning parser,
quantization, and tensor-parallel settings depend on the model.

To install vLLM in a separate environment, run this on the machine or node that
will serve the model:

```bash
bash setup.sh --with-vllm
```

This writes `.venv-vllm` and uses automatic PyTorch backend selection when the
installed `uv` supports it. The separate environment avoids changing evaluator
dependencies. See the [official vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
for other GPU platforms, containers, and source builds. An existing
OpenAI-compatible endpoint works without installing vLLM here.

### What the evaluation needs from your server

| | |
|---|---|
| **Endpoint** | OpenAI-compatible `/v1/chat/completions` (vLLM, SGLang, Ollama, or a gateway) |
| **Model name** | `--model` must match the server's model id (in vLLM, `--served-model-name`) |
| **Context** | **`--max-model-len` of at least ~80k**; 131,072 is recommended |
| **Where it is** | `VLLM_PORT` (default `8011`), or `VLLM_BASE_URL` for a non-local host |

```bash
export VLLM_PORT=8011                        # or put it in .env
# export VLLM_BASE_URL=http://host:9000/v1   # if it isn't on localhost:VLLM_PORT
```

Before evaluation, `run_eval.py` queries `<base-url>/models` and checks the
requested model id. A stopped server or mismatched `--served-model-name` fails
before any dataset starts.

### ⚠️ Serve with a context of at least ~80k

Each dataset uses the same output budget for every model. The maximum is
**65,536 output tokens**, plus the prompt. Example:

```bash
vllm serve my-org/custom-7b \
  --served-model-name my-org/custom-7b \
  --host 127.0.0.1 --port 8011 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --max-num-seqs 32
```

Add the model's required chat template and reasoning parser. Adjust parallelism
and quantization for the available GPUs.

A smaller context can produce this error:

```
BadRequestError: max_tokens=65536 cannot be greater than max_model_len=32768
```

Re-serve with a larger `--max-model-len`. Changing the output budget makes runs
incomparable.

### A ready-made launcher for the models we evaluated

`scripts/serve_vllm.sh` contains defaults for the paper's Qwen and gpt-oss
models. Other models may need a different chat template or reasoning parser.

```bash
REASONING_PARSER=qwen3 bash scripts/serve_vllm.sh Qwen/Qwen3.5-4B                       # single GPU
VLLM_PORT=8011 TP_SIZE=4 REASONING_PARSER=qwen3 bash scripts/serve_vllm.sh Qwen/Qwen3.5-122B-A10B-FP8
```

The launcher also infers the parser for registered Qwen and gpt-oss models;
the examples keep it explicit so the server configuration is visible.

Its defaults match the paper and assume an **H100 or newer** (`--kv-cache-dtype
fp8` needs compute capability ≥ 8.9). On an **A100**, turn that one
optimisation off:

```bash
KV_CACHE_DTYPE=auto REASONING_PARSER=qwen3 bash scripts/serve_vllm.sh Qwen/Qwen3.5-4B
```

Copy and adapt the script for other models.

## 5. Run

Smoke data is the default. These commands set up the suite and start a smoke
evaluation.

API model:

```bash
bash setup.sh
OPENAI_API_KEY=... .venv/bin/python run_eval.py --datasets all --model gpt-5-mini
```

Local model we evaluated. Run line 1 in the GPU/server terminal; it stays
running. Run line 2 from the repo in another terminal after the server is ready:

```bash
bash setup.sh --with-vllm && REASONING_PARSER=qwen3 bash scripts/serve_vllm.sh Qwen/Qwen3.5-4B
VLLM_PORT=8011 .venv/bin/python run_eval.py --datasets all --model Qwen/Qwen3.5-4B --examiner-model Qwen/Qwen3.5-4B --no-20q-offline
```

Custom local model. Set `REASONING_PARSER` to the parser your model needs, or
`none`:

```bash
bash setup.sh --with-vllm && REASONING_PARSER=my_reasoning_parser bash scripts/serve_vllm.sh my-org/custom-7b
VLLM_PORT=8011 .venv/bin/python run_eval.py --datasets all --model my-org/custom-7b --examiner-model my-org/custom-7b --no-20q-offline
```

For full released data, add `--full`; the runner asks for confirmation and
prints a call estimate before starting:

```bash
.venv/bin/python run_eval.py --datasets all --model gpt-5-mini --full
```

Results and a combined `summary.md` / `summary.json` are written under
`logs/run_<model>_<smoke|full>/`. That directory is reused by default so an
interrupted run resumes from its cached outputs. Use `--fresh-run` to start in a
new timestamped directory. If an existing manifest has incompatible settings,
the runner asks you to use `--fresh-run` instead of mixing results.

## Evaluating your own model

**Hosted models** (`gpt-*`, `gemini-*`) work by name directly.

For an open-weight or custom model, start an OpenAI-compatible server and pass
its served name. Unknown model names are routed to `VLLM_PORT` over
`/v1/chat/completions`.

```bash
# Start the server first; this command uses smoke data and no hosted models.
export VLLM_PORT=8011
python run_eval.py --datasets all --model my-org/custom-7b \
    --examiner-model my-org/custom-7b \
    --no-20q-offline
```

`VLLM_PORT` may be set in `.env` (default `8011`).

### Optional: `--model-config` for overrides

`--model-config` is a user-written input file for a protected remote gateway,
the raw `/v1/completions` path, or a different tokenizer.

```yaml
# my_model.yaml
model_name: my-org/custom-7b
tokenizer_name: my-org/custom-7b
base_url: https://openrouter.ai/api/v1   # non-local endpoint
backend: openai_chat                      # or "completions" for the raw path
api_key_env: OPENROUTER_API_KEY           # if the endpoint needs a key
enable_reasoning: true
thinking_start_token: ""
thinking_end_token: "</think>"
reasoning_parser: qwen3                   # server-launch suggestion only
```

```bash
python run_eval.py --datasets all --model my-org/custom-7b \
    --model-config my_model.yaml \
    --examiner-model my-org/custom-7b \
    --no-20q-offline
```

The runner loads this file, requires `model_name` to match `--model`, forwards
the same path to every dataset subprocess (including 20Q), and records its
absolute path in `run_manifest.json`. 20Q uses `model_name`, `base_url`,
`api_key_env`, and `tokenizer_name` on its Chat Completions path; a forced
`completions` backend applies to the other evaluator clients.

Local vLLM does not need an API key. The OpenAI client constructor requires
a non-empty value, so the suite supplies a dummy string. `api_key_env` names the
environment variable used by a protected OpenAI-compatible gateway.

## Notes

- **Defaults** match the paper protocol: adversarial oracle (where supported),
  budget 10, forbid-alternatives, budget text hidden from the prompt.
- **20-Questions** uses `gpt-5-mini` as its examiner and offline judge by
  default. A run testing a local model therefore still makes GPT calls when
  `20q` is selected. To avoid hosted-model calls, use the served model as the
  examiner and disable the post-run offline judge:
  ```bash
  python run_eval.py --datasets all --model my-org/custom-7b \
      --examiner-model my-org/custom-7b \
      --no-20q-offline
  ```
  Omit `--no-20q-offline` to run the offline question-quality evaluator with the
  examiner model, or set `--20q-judge-model` explicitly. This stage adds mean
  normalized question EIG and final entropy to the summary and makes additional
  model calls. Its direct script defaults to 32,768 output tokens and batches
  four candidates at a time. Keep the judge fixed across compared models.
  Alternatively, omit `20q` from `--datasets`. Missing credentials are reported
  before evaluation.
- **Determinism**: open-weight models are sampled at `temperature=0.6`. Runs are cached; use `--fresh-run` to keep separate from the previous run.
