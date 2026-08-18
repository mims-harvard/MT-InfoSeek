"""One-command driver for the MT-InfoSeek evaluation suite.

Runs one model against one or more datasets sequentially, using each dataset's
own env-var-driven shell script (so the exact, canonical flags live in one
place). Defaults match the paper protocol: adversarial oracle where supported,
budget 10, forbid-alternatives, budget text hidden from the prompt.

    python run_eval.py --datasets all --model gpt-5-mini
    python run_eval.py --datasets logic_q_mt,genereg_mt --model Qwen/Qwen3.5-9B
    python run_eval.py --datasets all --model my-org/custom --model-config my.yaml
    python run_eval.py --datasets all --model gpt-5-mini --full   # full data (asks first)

By default it runs the tiny `data/smoke/` subsets. Pass --full for the released
data. The default result directory is reused so interrupted runs resume from
their caches; pass --fresh-run to create a timestamped directory.
"""
import argparse
from datetime import datetime
import glob
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def load_dotenv(path=os.path.join(ROOT, ".env")) -> None:
    """Read .env, exactly as the dataset scripts do (scripts/load_env.sh).

    SETUP.md tells users to put VLLM_PORT / CACHE_DIR / MODELS_DIR / API keys in
    .env, so run_eval's own checks (model validation, the GeneReg asset check)
    must see them.

    Bash evaluates the file so `${PROJECT_ROOT}/...` expands as it does in the
    dataset scripts.
    Anything already in the environment wins, so
    `VLLM_PORT=9999 python run_eval.py ...` still overrides the file.
    """
    if not os.path.isfile(path):
        return
    try:
        proc = subprocess.run(
            ["bash", "-c", f'set -a; source "{path}" >/dev/null 2>&1; env -0'],
            capture_output=True, cwd=ROOT, timeout=30,
        )
    except Exception:
        return
    if proc.returncode != 0:
        return
    ignore = {"_", "PWD", "OLDPWD", "SHLVL"}
    for entry in proc.stdout.split(b"\0"):
        if not entry:
            continue
        key, sep, value = entry.decode("utf-8", "replace").partition("=")
        if not sep or key in ignore:
            continue
        if key not in os.environ:      # environment beats the file
            os.environ[key] = value


load_dotenv()

import model_registry  # noqa: E402  (after .env, so VLLM_PORT is visible)

ALL_DATASETS = ["logic_q_mt", "gsme_q_mt", "gsme_q_mt_ext", "genereg_mt", "20q"]

# Full-data row counts, for the call-count estimate.
FULL_ROWS = {
    "logic_q_mt": 600, "gsme_q_mt": 2956, "gsme_q_mt_ext": 524,
    "genereg_mt": 800, "20q": 311,  # 20Q = 111 Common + 200 THING200
}

TWENTYQ_FULL_POOLS = ("common", "thing")

# Per-dataset capability + how to launch it. `oracle`/`no_budget_prompt` are
# applied only where the evaluator supports them (elsewhere: no-op, logged).
DATASETS = {
    "logic_q_mt": {
        "script": "logic_q_mt/scripts/run_multiturn.sh",
        "oracle": True, "no_budget_prompt": True,
        "data_file": "logic_q_mt.csv", "data_kind": "file",
        "episodes_per_row": 2,  # 2 worlds/sample
    },
    "genereg_mt": {
        "script": "genereg_mt/scripts/run_multiturn.sh",
        "oracle": True, "no_budget_prompt": True, "needs_grn": True,
        "data_file": "genereg_mt.jsonl", "data_kind": "file",
        "episodes_per_row": 1,
    },
    "gsme_q_mt": {
        "script": "gsme_q_mt/scripts/run_multiturn.sh",
        "oracle": False, "no_budget_prompt": False,
        "data_file": "gsme_q_mt.csv", "data_kind": "dir",
        "episodes_per_row": 1,
    },
    "gsme_q_mt_ext": {
        "script": "gsme_q_mt/scripts/run_multiturn.sh",
        "oracle": False, "no_budget_prompt": False,
        "data_file": "gsme_q_mt_ext.csv", "data_kind": "dir",
        "episodes_per_row": 1,
    },
    "20q": {
        "script": "20q/scripts/run_inference.sh",
        "oracle": False, "no_budget_prompt": False, "two_models": True,
        "data_kind": "index",
        "episodes_per_row": 1,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Run the MT-InfoSeek evaluation suite.")
    p.add_argument("--datasets", default="all",
                   help="'all' or comma-separated subset of "
                        f"{{{', '.join(ALL_DATASETS)}}}")
    p.add_argument("--model", required=True,
                   help="Model to evaluate (registry name, hosted model, "
                        "'random' for Logic/GeneReg, or a served custom model).")
    p.add_argument("--model-config", default=None,
                   help="YAML/JSON file registering a custom OpenAI-compatible model.")
    p.add_argument("--examiner-model", default="gpt-5-mini",
                   help="20-Questions examiner model (default: gpt-5-mini). Pass your own "
                        "served model here to run 20q fully locally.")
    p.add_argument(
        "--20q-judge-model",
        dest="twentyq_judge_model",
        default=None,
        help=("Offline 20Q question-quality judge (default: --examiner-model). "
              "Keep it fixed across compared models; gpt-5-mini is recommended."),
    )
    p.add_argument(
        "--no-20q-offline",
        dest="twentyq_offline",
        action="store_false",
        default=True,
        help="Skip the automatic offline 20Q question-quality evaluation.",
    )
    p.add_argument("--full", action="store_true",
                   help=("Run the full released data instead of data/smoke/ "
                         "(20Q runs Common and THING200)."))
    p.add_argument("--yes", action="store_true",
                   help="Skip the confirmation prompt for --full.")
    p.add_argument("--budget", default="10",
                   help="Question budget: an integer 0-10. ('k' is accepted by logic_q_mt and genereg_mt only.)")
    p.add_argument("--oracle", default="adversarial",
                   choices=["adversarial", "cooperative", "random"],
                   help="Oracle policy where supported (logic_q_mt, genereg_mt).")
    p.add_argument("--reasoning-effort", default=None,
                   help="Reasoning effort for gpt-oss and GPT-5.2/5.4 models (logic_q_mt and genereg_mt only).")
    p.add_argument("--max-concurrent", type=int, default=16)
    p.add_argument("--results-dir", default=None,
                   help="Output root (default: ./logs/run_<model>_<data>; reused).")
    p.add_argument("--fresh-run", action="store_true",
                   help="Start in a new timestamped result directory instead of resuming.")
    p.add_argument("--cache-dir", default=None, help="GeneReg attractor cache (or $CACHE_DIR).")
    p.add_argument("--models-dir", default=None, help="Kadelka GRN models (or $MODELS_DIR).")
    p.add_argument("--vllm-port", default=None, help="Port of an already-running vLLM server.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run; --full also prints a call estimate.")
    p.add_argument("--analyze", dest="analyze", action="store_true", default=True,
                   help="Summarize results after running (default).")
    p.add_argument("--no-analyze", dest="analyze", action="store_false")
    return p.parse_args()


def resolve_datasets(spec):
    if spec.strip() == "all":
        return list(ALL_DATASETS)
    out = []
    for d in spec.split(","):
        d = d.strip()
        if not d:
            continue
        if d not in DATASETS:
            sys.exit(f"Unknown dataset {d!r}. Choose from: {', '.join(ALL_DATASETS)}")
        out.append(d)
    if not out:
        sys.exit("No datasets selected.")
    return out


def twentyq_pools(args):
    """20Q uses Common for smoke and both released pools for a full run."""
    return TWENTYQ_FULL_POOLS if args.full else ("common",)


def print_estimate(datasets, budget, twentyq_offline=True):
    try:
        budget_turns = int(budget)
    except ValueError:
        budget_turns = 4  # 'k' -> assume small
    total = 0
    lines = []
    for d in datasets:
        rows = FULL_ROWS[d]
        eps = rows * DATASETS[d]["episodes_per_row"]
        if d == "20q":
            # 20Q has a fixed 20-turn game: guesser + examiner each turn. The
            # automatic offline stage batches candidate judgments; ~2 batches
            # per turn after top-k pruning is a useful lower-bound estimate.
            calls = eps * 20 * 2
            if twentyq_offline:
                calls += eps * 20 * 2
        else:
            calls = eps * max(1, budget_turns)
        total += calls
        lines.append(f"    {d:16} ~{eps:>6} tasks     ~{calls:>7} model calls")
    print("Full-data rough estimate (max turns; caching + early answers reduce it):")
    print("\n".join(lines))
    print(f"    {'TOTAL':16} {'':>16} ~{total:>7} model calls")


def confirm_full(datasets, budget, twentyq_offline=True):
    print_estimate(datasets, budget, twentyq_offline=twentyq_offline)
    print("\nThis will incur real API cost for hosted models.")
    ans = input("Proceed? [y/N] ").strip().lower()
    return ans in ("y", "yes")


def build_env(base, ds, args, data_root, results_dir):
    env = dict(base)
    env["PYTHON"] = sys.executable
    env["MODEL"] = args.model
    env["RESULTS_DIR"] = results_dir
    env["MAX_CONCURRENT"] = str(args.max_concurrent)
    env["BUDGET"] = str(args.budget)
    if args.reasoning_effort:
        env["REASONING_EFFORT"] = args.reasoning_effort
    if args.vllm_port:
        env["VLLM_PORT"] = str(args.vllm_port)

    cfg = DATASETS[ds]
    if cfg["oracle"]:
        env["ORACLE"] = args.oracle
    if cfg["no_budget_prompt"]:
        env["BUDGET_IN_PROMPT"] = "false"

    # Data location.
    if cfg["data_kind"] == "file":
        env["DATA_FILE"] = os.path.join(data_root, cfg["data_file"])
        if not os.path.isfile(env["DATA_FILE"]):
            return None, f"required data file is missing: {env['DATA_FILE']}"
    elif cfg["data_kind"] == "dir":
        env["DATA_DIR"] = data_root
        env["DATA_FILE"] = cfg["data_file"]
        data_file = os.path.join(data_root, cfg["data_file"])
        if not os.path.isfile(data_file):
            return None, f"required data file is missing: {data_file}"
    elif cfg["data_kind"] == "index":  # 20q
        env["GUESSER_MODEL"] = args.model
        env["EXAMINER_MODEL"] = args.examiner_model
        env["DATASET"] = "common"
        env["MAX_TURN"] = env.get("MAX_TURN", "20")
        env["START_IDX"] = "0"
        env["END_IDX"] = "-1" if args.full else "6"

    if cfg.get("needs_grn"):
        cache = args.cache_dir or base.get("CACHE_DIR")
        models = args.models_dir or base.get("MODELS_DIR")
        if not cache or not models:
            return None, ("genereg_mt needs CACHE_DIR and MODELS_DIR. Run "
                          "`bash setup.sh` to create .env, or pass "
                          "--cache-dir/--models-dir.")
        if not os.path.isdir(cache):
            return None, (
                f"GeneReg cache directory is missing: {cache}. "
                "Run `bash setup.sh`, or pass --cache-dir / update CACHE_DIR."
            )
        if not os.path.isdir(models):
            return None, (
                f"Kadelka models directory is missing: {models}. "
                "Run `bash setup.sh`, or pass --models-dir / update MODELS_DIR."
            )
        env["CACHE_DIR"], env["MODELS_DIR"] = cache, models
    return env, None


def resolve_results_dir(args):
    """Return the stable resume directory, or a unique timestamped fresh one."""
    tag = args.model.replace("/", "_")
    base = args.results_dir or os.path.join(
        ROOT, "logs", f"run_{tag}_{'full' if args.full else 'smoke'}"
    )
    base = os.path.abspath(base)
    if not args.fresh_run:
        return base

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{base}_{timestamp}"
    suffix = 2
    while os.path.exists(candidate):
        candidate = f"{base}_{timestamp}_{suffix}"
        suffix += 1
    return candidate


def run_identity(args):
    """Settings that must match when reusing cached model outputs."""
    return {
        "model": args.model,
        "model_config": os.path.abspath(args.model_config) if args.model_config else None,
        "data": "full" if args.full else "smoke",
        "budget": str(args.budget),
        "oracle": args.oracle,
        "reasoning_effort": args.reasoning_effort,
        "examiner_model": args.examiner_model,
        "twentyq_offline": args.twentyq_offline,
        "twentyq_judge_model": args.twentyq_judge_model or args.examiner_model,
    }


def check_resume_compatibility(results_dir, args):
    """Refuse to mix a new configuration into a manifest-backed run."""
    if args.fresh_run or not os.path.isdir(results_dir):
        return
    manifest_path = os.path.join(results_dir, "run_manifest.json")
    if not os.path.isfile(manifest_path):
        if os.listdir(results_dir):
            print("WARNING: resuming a pre-manifest result directory; configuration "
                  "compatibility cannot be checked. Use --fresh-run for isolation.")
        return
    try:
        with open(manifest_path) as f:
            previous = json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"Cannot read existing run manifest {manifest_path}: {exc}")

    current = run_identity(args)
    previous_identity = previous.get("identity", {})
    mismatches = [
        f"  {key}: previous={previous_identity.get(key)!r}, current={value!r}"
        for key, value in current.items()
        if previous_identity.get(key) != value
    ]
    if mismatches:
        sys.exit(
            "Refusing to resume with settings that differ from run_manifest.json:\n"
            + "\n".join(mismatches)
            + "\nUse --fresh-run (or choose another --results-dir)."
        )


def write_run_manifest(results_dir, args, datasets):
    manifest = {
        "identity": run_identity(args),
        "datasets": datasets,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "fresh" if args.fresh_run else "resume",
    }
    with open(os.path.join(results_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def preflight_local_models(args, datasets):
    """Verify each selected OpenAI-compatible model before starting a run."""
    models = [args.model]
    if "20q" in datasets:
        models.append(args.examiner_model)
        if args.twentyq_offline:
            models.append(args.twentyq_judge_model or args.examiner_model)

    checked = set()
    failures = []
    for model in models:
        if model in checked or model in ("random", "random-baseline"):
            continue
        checked.add(model)
        if model_registry.is_hosted_model(model):
            continue
        if not model_registry.is_registered_local(model):
            failures.append(f"{model}: no local/OpenAI-compatible route is configured")
            continue

        base_url = model_registry.resolve_base_url(model, port=args.vllm_port).rstrip("/")
        url = f"{base_url}/models"
        headers = {}
        api_key_env = model_registry.api_key_env_for(model)
        if api_key_env:
            if not os.environ.get(api_key_env):
                failures.append(
                    f"{model}: model config requires unset environment variable "
                    f"{api_key_env}"
                )
                continue
            headers["Authorization"] = f"Bearer {os.environ[api_key_env]}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.load(response)
            served_ids = {
                item.get("id") for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
            if model not in served_ids:
                shown = ", ".join(sorted(served_ids)) or "none reported"
                failures.append(
                    f"{model}: endpoint {url} is reachable but serves [{shown}]. "
                    "Set vLLM --served-model-name to match --model."
                )
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            failures.append(f"{model}: cannot query {url}: {exc}")

    if failures:
        sys.exit(
            "Local model preflight failed before evaluation:\n  - "
            + "\n  - ".join(failures)
            + "\nStart or fix the server, then rerun; see SETUP.md section 4."
        )


def check_20q_models(args):
    """Fail early when 20Q examiner/judge configuration cannot run."""
    judge_model = args.twentyq_judge_model or args.examiner_model
    if args.twentyq_offline and model_registry.is_hosted_model(judge_model):
        if not judge_model.lower().startswith("gpt-"):
            sys.exit(
                "The bundled 20Q offline evaluator supports a local "
                "OpenAI-compatible judge or a hosted GPT judge. Set "
                "--20q-judge-model to one of those, or pass --no-20q-offline."
            )

    hosted = [args.examiner_model]
    if args.twentyq_offline:
        hosted.append(judge_model)
    hosted = [m for m in dict.fromkeys(hosted) if model_registry.is_hosted_model(m)]
    if not hosted:
        return
    have_key = any(os.environ.get(k) for k in
                   ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY",
                    "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"))
    if have_key:
        return
    sys.exit(
        f"\n20-Questions needs hosted credentials for: {', '.join(hosted)} "
        f"-- but no API key is configured.\n\n"
        f"Either:\n"
        f"  * use your served model as the answerer and skip the offline judge:\n"
        f"      --examiner-model {args.model} --no-20q-offline\n"
        f"  * use a fixed local OpenAI-compatible model with "
        f"--20q-judge-model,\n"
        f"  * or add an API key to .env (see SETUP.md),\n"
        f"  * or omit 20q:  --datasets logic_q_mt,gsme_q_mt,"
        f"gsme_q_mt_ext,genereg_mt\n"
    )


def find_20q_dialogue_log(results_dir, dataset):
    """Find the completed 20Q dialogue JSON for one candidate pool."""
    hits = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    hits = [
        path for path in hits
        if not any(suffix in os.path.basename(path) for suffix in ("_cot", "_metrics"))
        and f"{os.sep}offline_evaluation{os.sep}" not in path
    ]
    matches = []
    for path in hits:
        try:
            with open(path) as f:
                rows = json.load(f)
            run_dataset = (rows[0].get("run_config", {}).get("dataset")
                           if isinstance(rows, list) and rows else None)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if run_dataset == dataset:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No completed 20Q {dataset!r} dialogue JSON found under {results_dir}"
        )
    return max(matches, key=os.path.getmtime)


def run_20q_offline(args, env, results_dir):
    """Run the bundled question-quality evaluator on the completed dialogues."""
    dataset = env.get("DATASET", "common")
    input_path = find_20q_dialogue_log(results_dir, dataset)
    offline_dir = os.path.join(results_dir, "offline_evaluation")
    os.makedirs(offline_dir, exist_ok=True)

    pool_names = {
        "common": "COMMON_EVAL_POOL",
        "thing": "THING200_EVAL_POOL",
    }

    judge_model = args.twentyq_judge_model or args.examiner_model
    offline_env = dict(env)
    offline_env.update({
        "INPUT_LOG_PATH": input_path,
        "OUTPUT_PATH": os.path.join(offline_dir, f"question_quality_{dataset}.jsonl"),
        "CACHE_PATH": os.path.join(offline_dir, f"judge_cache_{dataset}.json"),
        "EVAL_MODEL": judge_model,
        "POOL": pool_names[dataset],
    })

    if model_registry.is_hosted_model(judge_model):
        offline_env["JUDGE_BACKEND"] = "hosted_gpt"
    else:
        offline_env["JUDGE_BACKEND"] = "openai_compatible"
        offline_env["EVAL_BASE_URL"] = model_registry.resolve_base_url(
            judge_model, port=args.vllm_port
        )
        api_key_env = model_registry.api_key_env_for(judge_model)
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Offline judge {judge_model!r} requires unset {api_key_env}."
                )
            offline_env["EVAL_API_KEY"] = api_key

    script = os.path.join(ROOT, "20q", "scripts", "run_offline_evaluation.sh")
    print(f"\n# 20q offline question-quality evaluation ({dataset})")
    return subprocess.run(["bash", script], env=offline_env, cwd=ROOT).returncode


def main():
    args = parse_args()
    datasets = resolve_datasets(args.datasets)

    if args.model.strip().lower() in ("random", "random-baseline"):
        supported = {"logic_q_mt", "genereg_mt"}
        unsupported = [dataset for dataset in datasets if dataset not in supported]
        if unsupported:
            sys.exit(
                "The random baseline supports only logic_q_mt and genereg_mt; "
                f"unsupported: {', '.join(unsupported)}.\n"
                "Use --datasets logic_q_mt,genereg_mt."
            )

    # Make VLLM_PORT visible here and to every subprocess, so an unknown model
    # name is inferred as "served locally over /v1/chat/completions".
    if args.vllm_port:
        os.environ["VLLM_PORT"] = str(args.vllm_port)

    # Optional override file (remote gateway key, forced /completions, custom
    # tokenizer). Loaded here for validation and forwarded to each subprocess.
    model_config_abs = None
    if args.model_config:
        model_config_abs = os.path.abspath(args.model_config)
        registered = model_registry.load_model_config_file(model_config_abs)
        if registered != args.model:
            sys.exit(
                f"{args.model_config} configures model_name={registered!r}, but "
                f"--model is {args.model!r}. These must match."
            )
        print(f"Registered custom model from {args.model_config}: {registered}")
        # Forwarded to each dataset subprocess via its script (--model-config).
        os.environ["MODEL_CONFIG"] = model_config_abs

    if "20q" in datasets:
        check_20q_models(args)

    # Fail fast, before any run.
    if args.model not in ("random", "random-baseline"):
        model_registry.validate_model_name(args.model,
                                           extra_allowed=["random", "random-baseline"])
    if "20q" in datasets and not model_registry.is_hosted_model(args.examiner_model):
        model_registry.validate_model_name(args.examiner_model)
    judge_model = args.twentyq_judge_model or args.examiner_model
    if ("20q" in datasets and args.twentyq_offline
            and not model_registry.is_hosted_model(judge_model)):
        model_registry.validate_model_name(judge_model)

    data_label = "data" if args.full else "data/smoke"
    data_root = os.path.join(ROOT, data_label)
    results_dir = resolve_results_dir(args)

    print("=" * 60)
    print(f"  MT-InfoSeek suite")
    print(f"    model      : {args.model}")
    print(f"    datasets   : {', '.join(datasets)}")
    print(f"    data       : {data_label}{'  (FULL)' if args.full else '  (smoke subset)'}")
    print(f"    oracle     : {args.oracle}  | budget {args.budget}  | budget-in-prompt: off")
    print(f"    results    : {results_dir}")
    print(f"    run mode   : {'fresh' if args.fresh_run else 'resume (default)'}")
    print("=" * 60)

    base_env = dict(os.environ)
    plans = []
    config_errors = []
    for ds in datasets:
        ds_results = os.path.join(results_dir, ds)
        env, err = build_env(base_env, ds, args, data_root, ds_results)
        if err:
            config_errors.append(f"{ds}: {err}")
            continue
        plans.append((ds, env, ds_results))

    if config_errors:
        sys.exit(
            "Cannot start the selected evaluation:\n  - "
            + "\n  - ".join(config_errors)
        )

    if args.dry_run:
        if args.full:
            print()
            print_estimate(
                [d for d, _, _ in plans],
                args.budget,
                twentyq_offline=args.twentyq_offline,
            )
        print("\n[dry run] would execute:")
        for ds, env, _ in plans:
            pool_envs = [(None, env)]
            if ds == "20q":
                pool_envs = []
                for pool in twentyq_pools(args):
                    one_pool_env = dict(env)
                    one_pool_env["DATASET"] = pool
                    pool_envs.append((pool, one_pool_env))
            for pool, run_env in pool_envs:
                knobs = {k: run_env[k] for k in
                         ("MODEL", "DATA_FILE", "DATA_DIR", "DATASET", "ORACLE",
                          "BUDGET", "BUDGET_IN_PROMPT", "EXAMINER_MODEL")
                         if k in run_env}
                label = f"{ds}:{pool}" if pool else ds
                print(f"  {label}: bash {DATASETS[ds]['script']}  {knobs}")
                if ds == "20q" and args.twentyq_offline:
                    print(f"       then: 20q {pool} offline question-quality evaluation")
        return

    check_resume_compatibility(results_dir, args)
    preflight_local_models(args, datasets)

    if args.full and not args.yes:
        if not confirm_full(
            datasets,
            args.budget,
            twentyq_offline=args.twentyq_offline,
        ):
            sys.exit("Aborted.")

    os.makedirs(results_dir, exist_ok=True)
    write_run_manifest(results_dir, args, datasets)

    failed = []
    for ds, env, ds_results in plans:
        os.makedirs(ds_results, exist_ok=True)
        script = os.path.join(ROOT, DATASETS[ds]["script"])
        # 20q writes to ./logs relative to cwd; run it from its results dir.
        cwd = ds_results if ds == "20q" else ROOT
        pool_envs = [(None, env)]
        if ds == "20q":
            pool_envs = []
            for pool in twentyq_pools(args):
                one_pool_env = dict(env)
                one_pool_env["DATASET"] = pool
                pool_envs.append((pool, one_pool_env))

        for pool, run_env in pool_envs:
            label = f"{ds}:{pool}" if pool else ds
            print(f"\n{'#' * 60}\n# {label}\n{'#' * 60}")
            proc = subprocess.run(["bash", script], env=run_env, cwd=cwd)
            if proc.returncode != 0:
                print(f"!! {label} exited {proc.returncode}")
                failed.append(label)
                continue
            if ds == "20q" and args.twentyq_offline:
                try:
                    offline_returncode = run_20q_offline(args, run_env, ds_results)
                except Exception as exc:
                    print(f"!! {label} offline evaluation could not start: {exc}")
                    offline_returncode = 1
                if offline_returncode != 0:
                    print(f"!! {label} offline evaluation exited {offline_returncode}")
                    failed.append(f"{label}_offline")

    if args.analyze:
        print(f"\n{'#' * 60}\n# Summary\n{'#' * 60}")
        summarize = os.path.join(ROOT, "analyze_results.py")
        subprocess.run([sys.executable, summarize, "--results-dir", results_dir,
                        "--datasets", ",".join(d for d, _, _ in plans)],
                       cwd=ROOT)

    if failed:
        sys.exit(f"\nFinished with failures: {', '.join(failed)}")
    print(f"\nDone. Results + summary under {results_dir}")


if __name__ == "__main__":
    main()
