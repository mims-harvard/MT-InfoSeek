"""Script to evaluate LLMs on the released GSME-Q datasets."""

import argparse
import os
from gsme_q_mt.evaluators.gsme_k import GSMEvaluator
import pandas as pd
import json
import platform
from datetime import datetime
import re
from gsme_q_mt.evaluators.gsme_mt import GSMOpenEndedEvaluator
import gsme_q_mt.eval_utils as eu

DOMAIN_NAME = "GSM_k"

def parse_k_values(k_values: str):
    if not k_values:
        return None
    values = []
    for item in k_values.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid k value {item!r}; use a comma-separated list like 1,2.") from exc
    return values or None


def sanitize_suffix(suffix: str) -> str:
    suffix = (suffix or "").strip()
    if not suffix:
        return ""
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix)
    return suffix.strip("._-")


def main(user_args) -> None:
    # Make directories for results and cache
    os.makedirs(user_args.results_dir, exist_ok=True)
    cache_dir = os.path.join(user_args.results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    data_file_base_name = os.path.splitext(os.path.basename(user_args.data_file))[0]

    # Make model name filesystem-safe (important: "Qwen/xxx" contains "/")
    safe_model_name = user_args.model_name.replace("/", "_")

    k_tag = "givek" if user_args.reveal_k_in_prompt else "nok"
    output_file_name = (
        f"{safe_model_name}-{DOMAIN_NAME}-{user_args.eval_mode}-"
        f"{k_tag}-{data_file_base_name}"
    )
    if getattr(user_args, "include_allowed_leaf_in_prompt", False):
        output_file_name = f"{output_file_name}-allowedleaf"
    output_suffix = sanitize_suffix(getattr(user_args, "output_suffix", ""))
    if output_suffix:
        output_file_name = f"{output_file_name}-{output_suffix}"

    cache_file = os.path.join(cache_dir, f"{output_file_name}.jsonl")
    output_file = os.path.join(user_args.results_dir, f"{output_file_name}.csv")
    episode_log_file = os.path.join(user_args.results_dir, f"{output_file_name}.episodes.jsonl")

    # Ensure parent dirs exist (defensive)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("Loading Evaluator")

    prompt_file = None

    if user_args.eval_mode in ["mt_all", "mt_one"]:
        evaluator = GSMOpenEndedEvaluator(
            model_name=user_args.model_name,
            cache=None,
            cache_file=cache_file,
            eval_mode=user_args.eval_mode,
            batch_size=user_args.batch_size,
            max_turns=user_args.budget,
            use_cot=False,
            vllm_port=user_args.vllm_port,
            parallel_model_calls=user_args.parallel_model_calls,
            include_allowed_leaf_in_prompt=user_args.include_allowed_leaf_in_prompt,
        )
        prompt_file = None
    else:
        evaluator = GSMEvaluator(
            user_args.model_name,
            cache_file=cache_file,
            use_cot=False,
            fs_samples=0,
            eval_mode=user_args.eval_mode,
            batch_size=user_args.batch_size,
            parallel_model_calls=user_args.parallel_model_calls,
            vllm_port=user_args.vllm_port,
            reveal_k_in_prompt=user_args.reveal_k_in_prompt,
        )
        prompt_file = None


    print("Loading Data")
    data_file = os.path.join(user_args.data_dir, user_args.data_file)
    data = pd.read_csv(data_file)

    k_values = parse_k_values(getattr(user_args, "k_values", ""))
    if k_values is not None:
        if "k" not in data.columns:
            raise ValueError("--k_values was provided, but the data file has no 'k' column.")
        before_n = len(data)
        data = data.loc[data["k"].isin(k_values)].reset_index(drop=True)
        print(f"Filtered data by k={k_values}: {before_n} -> {len(data)} rows")
        if len(data) == 0:
            raise ValueError(f"No rows remain after filtering by k={k_values}.")

    # -------------------------
    # Subsample for quick tests
    # -------------------------
    if getattr(user_args, "max_examples", None):
        n = int(user_args.max_examples)
        if n > 0 and len(data) > n:
            if getattr(user_args, "sample_seed", None) is not None and int(user_args.sample_seed) != 0:
                data = data.sample(n=n, random_state=int(user_args.sample_seed)).reset_index(drop=True)
            else:
                data = data.head(n).reset_index(drop=True)

    prompt_data = None

    print("Starting Evaluation")
    out = evaluator.evaluate_data(data, prompt_data)

    if isinstance(out, tuple):
        results, all_cots, _, _ = out
    else:
        results, all_cots = out, None

        # ensure key columns exist for downstream grouping/statistics
        base_cols = ["k", "correct", "pred_q", "gt_qs", "thinking_tokens", "cost_usd"]
        for col in base_cols:
            if col not in results.columns:
                results[col] = None

        if user_args.eval_mode in ["mt_all", "mt_one"]:
            mt_cols = [
                # answer
                "ans_correct","ans_pred","ans_gt","ans_turn","ans_forced","ans_early",
                # var
                "var_exact","var_precision","var_recall","var_f1",
                "var_pred","var_gt","var_invalid_count","var_duplicate_count",
                # compatibility
                "qset_exact","pred_answer","gt_answer",
            ]
            for col in mt_cols:
                if col not in results.columns:
                    results[col] = None
                    
    # Write per-episode logs if present in conversation
    if user_args.eval_mode in ["mt_all", "mt_one"] and "conversation" in results.columns:
        with open(episode_log_file, "w") as wf:
            for _, row in results.iterrows():
                try:
                    convo = json.loads(row["conversation"])
                except Exception:
                    continue
                log_obj = None
                for m in reversed(convo):
                    if isinstance(m, dict) and m.get("role") == "user":
                        txt = m.get("text", "")
                        if isinstance(txt, str) and txt.startswith("LOG_JSON: "):
                            try:
                                log_obj = json.loads(txt[len("LOG_JSON: "):])
                            except Exception:
                                log_obj = None
                            break
                if log_obj is not None:
                    wf.write(json.dumps(log_obj) + "\n")
        print(f"Wrote episodes to {episode_log_file}")

    with open(output_file, "w") as wf:
        results.to_csv(wf, index=False)
    print(f"Wrote to {output_file}")

    summary_file = output_file.replace(".csv", "_summary.json")

    # ---- global metrics ----
    global_metrics = eu.compute_metrics(results, user_args.eval_mode)

    # ---- per-k metrics: compute ALL metrics separately for each k ----
    metrics_by_k = {}
    if results is not None and "k" in results.columns:
        # drop NaN k
        tmp_df = results.loc[results["k"].notna()].copy()
        if len(tmp_df) > 0:
            try:
                for k_val, gdf in tmp_df.groupby("k"):
                    # robust stringify
                    try:
                        k_key = str(int(k_val))
                    except Exception:
                        k_key = str(k_val)
                    metrics_by_k[k_key] = eu.compute_metrics(gdf, user_args.eval_mode)
            except Exception:
                metrics_by_k = {}

    summary = {
        "run": {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "hostname": platform.node(),
            "python": platform.python_version(),
        },
        "config": {
            "model_name": user_args.model_name,
            "domain": DOMAIN_NAME,
            "eval_mode": user_args.eval_mode,
            "batch_size": user_args.batch_size,
            "parallel_model_calls": bool(user_args.parallel_model_calls),
            "vllm_port": user_args.vllm_port,
            "reveal_k_in_prompt": bool(user_args.reveal_k_in_prompt),
            "include_allowed_leaf_in_prompt": bool(user_args.include_allowed_leaf_in_prompt),
            "budget": getattr(user_args, "budget", None),
            "k_values": k_values,
            "output_suffix": output_suffix or None,
        },
        "paths": {
            "data_file": data_file,
            "prompt_file": prompt_file,
            "cache_file": cache_file,
            "results_csv": output_file,
            "cots_json": output_file.replace(".csv", "_cots.json") if all_cots is not None else None,
            "summary_json": summary_file,
            "episodes_jsonl": episode_log_file,
        },
        "metrics": global_metrics,
        "metrics_by_k": metrics_by_k if metrics_by_k else None,
    }

    with open(summary_file, "w") as wf:
        json.dump(summary, wf, indent=2)
    print(f"Wrote summary to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        help=(
            "Model name to evaluate, e.g. `gemini-3-flash-preview`, "
            "`gpt-5-mini`, or an OpenAI-compatible local model alias."
        ),
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=[
            "mc",
            "sc",
            "isambig",
            "fullinfo",
            "mt_all", "mt_one"
        ],
        help=(
            "Evaluation mode. `mc` is for selecting missing variables from"
            " predefined possible questions. `sc` is for single-choice where the"
            " model selects how many required variables are missing (0-4)."
            " `isambig` is for evaluating whether the model can"
            " identify the task is ambiguous, and `fullinfo` is for evaluating"
            " the model's performance on the task with the full information"
            " (i.e., no missing information)."
        ),
    )
    parser.add_argument("--data_file", type=str, help="The path to the data file.", default=None)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="gsme_q_data",
        help=("Directory containing data. Default is `gsme_q_data` in the current directory."),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help=("Directory to write results to. Default is `results` in the current directory."),
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")

    parser.add_argument("--vllm_port", type=int, default=8011, help="Port for the vLLM server (suite-wide default: 8011).")
    parser.add_argument(
        "--reveal_k_in_prompt",
        action="store_true",
        help="If set, include the exact k in the MC system prompt.",
    )
    parser.add_argument(
        "--include_allowed_leaf_in_prompt",
        action="store_true",
        help="If set for mt_all/mt_one, include 'Allowed leaf variables: ...' in the system prompt.",
    )
    parser.add_argument("--max_examples", type=int, default=0, help="If >0, only evaluate this many examples.")
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed for sampling when max_examples>0.")
    parser.add_argument("--k_values", type=str, default="", help="Comma-separated k values to evaluate, e.g. '1,2'.")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix appended to output/cache filenames.")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--parallel_model_calls", type=bool, default=True)
    parser.add_argument("--model_config", type=str, default=None,
                        help="YAML/JSON file registering a custom OpenAI-compatible model.")

    args = parser.parse_args()
    if args.model_config:
        import model_registry
        registered = model_registry.load_model_config_file(args.model_config)
        print(f"Registered custom model from {args.model_config}: {registered}")
    main(args)
