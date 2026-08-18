import os
import sys
import hashlib
from pathlib import Path

# Allow both `python -m twenty_questions.run` with PYTHONPATH=src and direct execution.
_src_root = Path(__file__).resolve().parents[1]
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import json
from typing import Any, Dict, List

from tqdm import tqdm
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

from twenty_questions.tasks.twenty_question import Q20Task
from twenty_questions.method import naive_converse
from twenty_questions.eval import evaluate_performance


TASK_NAME = "20q"


def _safe_load_logs(log_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        logs = json.loads(content)
        if isinstance(logs, list):
            return logs
        return []
    except Exception as e:
        print(f"Warning: failed to load existing log file {log_file}: {e}")
        return []


def _safe_save_logs(log_file: str, logs: List[Dict[str, Any]]) -> None:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(logs, ensure_ascii=False) + "\n")


def _build_log_file(args) -> str:
    return (
        f"./logs/{TASK_NAME}/{args.guesser_model}_as_guesser/"
        f"{args.dataset}_"
        f'examiner_{args.examiner_model}_{"" if args.inform else "un"}inform'
        f"_maxturn{args.max_turn}_{args.task_start_index}-{args.task_end_index}{args.add_info}.json"
    )


def _build_cot_log_file(args) -> str:
    return _build_log_file(args).replace(".json", "_cot.json")


def _resolve_tokenizer_path(model_name: str) -> str | None:
    model_name = str(model_name)

    # A top-level --model-config is loaded before run(), so use its tokenizer
    # override for 20Q token accounting as well as its endpoint routing.
    try:
        import model_registry
        configured = model_registry.LOCAL_MODEL_CONFIGS.get(model_name, {})
        if configured.get("tokenizer_name"):
            return str(configured["tokenizer_name"])
    except ImportError:
        pass

    mapping = {
        "qwen": "Qwen/Qwen3-4B-Thinking-2507",
        "qwen_4b": "Qwen/Qwen3-4B-Thinking-2507",
        "qwen_30b": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen_thinking_4b": "Qwen/Qwen3-4B-Thinking-2507",
        "qwen_thinking_30b": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen_instruct_4b": "Qwen/Qwen3-4B-Instruct-2507",
        "qwen_instruct_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "qwen3-4b-local": "Qwen/Qwen3-4B-Thinking-2507",
        "qwen3-30b-local": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen3-4b-instruct-local": "Qwen/Qwen3-4B-Instruct-2507",
        "qwen3-30b-instruct-local": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "llama3.1-8b-local": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gpt_oss_20b": "openai/gpt-oss-20b",
        "Qwen/Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    }

    if model_name in {"gpt-5", "gpt-5-mini", "gpt-4", "gpt-3.5-turbo"}:
        return None

    return mapping.get(model_name)


def _build_run_config(args) -> Dict[str, Any]:
    return {
        "task": TASK_NAME,
        "dataset": args.dataset,
        "guesser_model": args.guesser_model,
        "examiner_model": args.examiner_model,
        "inform": bool(args.inform),
        "max_turn": args.max_turn,
    }


def _build_run_signature(run_config: Dict[str, Any]) -> str:
    payload = json.dumps(run_config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _validate_existing_logs(
    logs: List[Dict[str, Any]],
    run_signature: str,
    run_config: Dict[str, Any],
    original_start_index: int,
    log_file: str,
) -> None:
    for offset, row in enumerate(logs):
        expected_index = original_start_index + offset
        actual_index = row.get("index")
        if actual_index != expected_index:
            raise ValueError(
                f"Existing log {log_file} is not contiguous from start index "
                f"{original_start_index}: expected index {expected_index}, found {actual_index}."
            )

        existing_signature = row.get("run_config_signature")
        if existing_signature != run_signature:
            raise ValueError(
                f"Existing log {log_file} does not match the current runtime configuration. "
                "Use a different --add_info or remove the old log before resuming."
            )

        existing_config = row.get("run_config")
        if existing_config != run_config:
            raise ValueError(
                f"Existing log {log_file} was created with a different run configuration. "
                "Refusing to silently resume."
            )


def _load_tokenizer(args):
    if AutoTokenizer is None:
        print(
            "Warning: transformers is not installed. "
            "Falling back to approximate whitespace token counting."
        )
        return None

    for model_name in [args.guesser_model, args.examiner_model]:
        tokenizer_path = _resolve_tokenizer_path(model_name)
        if tokenizer_path is None:
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            print(f"Loaded tokenizer from {tokenizer_path} for token statistics.")
            return tokenizer
        except Exception as e:
            print(f"Warning: failed to load tokenizer from {tokenizer_path}: {e}")

    print(
        "Warning: no local tokenizer could be loaded for the selected models. "
        "Falling back to approximate whitespace token counting."
    )
    return None


def _count_tokens(text: Any, tokenizer) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)

    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    return len(text.split())


def _extract_cot_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cot_logs = []
    for idx, log in enumerate(logs):
        cot_logs.append({
            "sample_id": idx,
            "index": log.get("index", idx),
            "item": log.get("item"),
            "state": log.get("state"),
            "turn": log.get("turn"),
            "thinking_g": log.get("thinking_g", []),
            "thinking_e": log.get("thinking_e", []),
            "thinking_tokens_g": log.get("thinking_tokens_g", 0),
            "thinking_tokens_e": log.get("thinking_tokens_e", 0),
            "run_config_signature": log.get("run_config_signature"),
        })
    return cot_logs


def _compute_metrics(logs: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    total_guess_prompt = 0
    total_guess_output_visible = 0
    total_guess_thinking = 0
    total_exam_prompt = 0
    total_exam_output_visible = 0
    total_exam_thinking = 0

    num_guess_prompt_msgs = 0
    num_guess_output_visible_msgs = 0
    num_guess_thinking_msgs = 0
    num_exam_prompt_msgs = 0
    num_exam_output_visible_msgs = 0
    num_exam_thinking_msgs = 0

    num_samples = len(logs)
    if num_samples == 0:
        return {
            "num_samples": 0,
            "num_success": 0,
            "accuracy": 0.0,
            "avg_turn": 0.0,
            "total_guess_prompt_tokens": 0,
            "total_guess_output_visible_tokens": 0,
            "total_guess_thinking_tokens": 0,
            "total_guess_output_tokens": 0,
            "total_guess_tokens": 0,
            "total_exam_prompt_tokens": 0,
            "total_exam_output_visible_tokens": 0,
            "total_exam_thinking_tokens": 0,
            "total_exam_output_tokens": 0,
            "total_exam_tokens": 0,
            "avg_guess_prompt_tokens": 0,
            "avg_guess_output_visible_tokens": 0,
            "avg_guess_thinking_tokens": 0,
            "avg_exam_prompt_tokens": 0,
            "avg_exam_output_visible_tokens": 0,
            "avg_exam_thinking_tokens": 0,
        }

    for log in logs:
        for msg in log.get("history_g", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                total_guess_prompt += _count_tokens(content, tokenizer)
                num_guess_prompt_msgs += 1
            elif role in {"assistant", "system"}:
                total_guess_output_visible += _count_tokens(content, tokenizer)
                num_guess_output_visible_msgs += 1

        for msg in log.get("history_e", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                total_exam_prompt += _count_tokens(content, tokenizer)
                num_exam_prompt_msgs += 1
            elif role in {"assistant", "system"}:
                total_exam_output_visible += _count_tokens(content, tokenizer)
                num_exam_output_visible_msgs += 1

        sample_guess_thinking_tokens = log.get("thinking_tokens_g", None)
        sample_exam_thinking_tokens = log.get("thinking_tokens_e", None)
        thinking_g = log.get("thinking_g", [])
        thinking_e = log.get("thinking_e", [])

        if sample_guess_thinking_tokens is not None:
            total_guess_thinking += int(sample_guess_thinking_tokens or 0)
            num_guess_thinking_msgs += len(thinking_g)
        else:
            for msg in thinking_g:
                total_guess_thinking += _count_tokens(msg.get("content", ""), tokenizer)
                num_guess_thinking_msgs += 1

        if sample_exam_thinking_tokens is not None:
            total_exam_thinking += int(sample_exam_thinking_tokens or 0)
            num_exam_thinking_msgs += len(thinking_e)
        else:
            for msg in thinking_e:
                total_exam_thinking += _count_tokens(msg.get("content", ""), tokenizer)
                num_exam_thinking_msgs += 1

    num_success = sum(1 for log in logs if log.get("state") == 1)
    total_guess_output_tokens = total_guess_output_visible + total_guess_thinking
    total_exam_output_tokens = total_exam_output_visible + total_exam_thinking

    return {
        "num_samples": num_samples,
        "num_success": num_success,
        "accuracy": num_success / num_samples,
        "avg_turn": sum(log.get("turn", 0) for log in logs) / num_samples,
        "total_guess_prompt_tokens": total_guess_prompt,
        "total_guess_output_visible_tokens": total_guess_output_visible,
        "total_guess_thinking_tokens": total_guess_thinking,
        "total_guess_output_tokens": total_guess_output_tokens,
        "total_guess_tokens": total_guess_prompt + total_guess_output_tokens,
        "total_exam_prompt_tokens": total_exam_prompt,
        "total_exam_output_visible_tokens": total_exam_output_visible,
        "total_exam_thinking_tokens": total_exam_thinking,
        "total_exam_output_tokens": total_exam_output_tokens,
        "total_exam_tokens": total_exam_prompt + total_exam_output_tokens,
        "avg_guess_prompt_tokens": total_guess_prompt / num_guess_prompt_msgs if num_guess_prompt_msgs else 0,
        "avg_guess_output_visible_tokens": total_guess_output_visible / num_guess_output_visible_msgs if num_guess_output_visible_msgs else 0,
        "avg_guess_thinking_tokens": total_guess_thinking / num_guess_thinking_msgs if num_guess_thinking_msgs else 0,
        "avg_exam_prompt_tokens": total_exam_prompt / num_exam_prompt_msgs if num_exam_prompt_msgs else 0,
        "avg_exam_output_visible_tokens": total_exam_output_visible / num_exam_output_visible_msgs if num_exam_output_visible_msgs else 0,
        "avg_exam_thinking_tokens": total_exam_thinking / num_exam_thinking_msgs if num_exam_thinking_msgs else 0,
    }


def run(args):
    task = Q20Task(args)

    original_start_index = max(args.task_start_index, 0)
    args.task_start_index = original_start_index

    if args.task_end_index < 0:
        args.task_end_index = len(task.data)
    else:
        args.task_end_index = min(args.task_end_index, len(task.data))

    log_file = _build_log_file(args)
    cot_log_file = _build_cot_log_file(args)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logs = _safe_load_logs(log_file)
    run_config = _build_run_config(args)
    run_signature = _build_run_signature(run_config)

    if len(logs) > 0:
        _validate_existing_logs(
            logs,
            run_signature=run_signature,
            run_config=run_config,
            original_start_index=original_start_index,
            log_file=log_file,
        )
        resumed_start = min(original_start_index + len(logs), args.task_end_index)
        print(
            f"Found existing log with {len(logs)} samples. "
            f"Resuming from index {resumed_start}."
        )
        args.task_start_index = resumed_start

    for i in tqdm(range(args.task_start_index, args.task_end_index)):
        log = naive_converse(task, i)
        log["run_config"] = run_config
        log["run_config_signature"] = run_signature

        logs.append(log)
        _safe_save_logs(log_file, logs)
        _safe_save_logs(cot_log_file, _extract_cot_logs(logs))

    evaluate_performance(log_file, task)

    metrics = _compute_metrics(logs, _load_tokenizer(args))
    metrics_file = log_file.replace(".json", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to {metrics_file}")
    print(f"Saved CoT log to {cot_log_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guesser_model", type=str, default="qwen")
    parser.add_argument("--examiner_model", type=str, default="qwen")
    parser.add_argument("--dataset", type=str, default="common", choices=["common", "thing", "bigbench"])
    parser.add_argument("--task_start_index", type=int, default=0)
    parser.add_argument("--task_end_index", type=int, default=-1)
    parser.add_argument("--max_turn", type=int, default=20)
    parser.add_argument("--add_info", type=str, default="")
    parser.add_argument("--inform", action="store_true", default=False)
    parser.add_argument("--expected_action_tokens", type=int, default=500)
    parser.add_argument(
        "--expected_target_tokens",
        type=int,
        default=4096,
        help=(
            "Generation budget for examiner-based question/guess extraction "
            "fallbacks (default: 4096; includes any reasoning tokens)."
        ),
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Top-level YAML/JSON model config (forwarded by run_eval.py).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_config:
        import model_registry

        configured_name = model_registry.load_model_config_file(args.model_config)
        if configured_name not in {args.guesser_model, args.examiner_model}:
            raise ValueError(
                f"{args.model_config} configures {configured_name!r}, but 20Q is "
                f"using guesser={args.guesser_model!r} and examiner={args.examiner_model!r}."
            )
        print(f"Registered custom model from {args.model_config}: {configured_name}")
    run(args)
