import argparse
import asyncio
import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
for _p in (THIS_DIR, ROOT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import multiturn_eval as mt
from evaluators.evaluator import Evaluator
from model_utils import async_generate_single


SYSTEM_PROMPT_FULLINFO_SS_ID = """You are reasoning about a Boolean gene regulatory network.

Task: identify the correct attractor ID for the given full steady-state vector.
Respond strictly with exactly one line:
Answer: [ATTRACTOR_ID_INTEGER]
"""


SYSTEM_PROMPT_FULLINFO_DYN_ATTR = """You are reasoning about a synchronous Boolean gene regulatory network.

Task: starting from the given full initial state, identify the attractor ID reached by synchronous updates.
Respond strictly with exactly one line:
Answer: [ATTRACTOR_ID_INTEGER]
"""


SYSTEM_PROMPT_FULLINFO_DYN_MARKER = """You are reasoning about a synchronous Boolean gene regulatory network.

Task: starting from the given full initial state, determine the converged value (0/1) of the marker gene.
Respond strictly with exactly one line:
Answer: 0
or
Answer: 1
"""


@dataclass
class FullInfoEpisodeResult:
    sample_id: int
    group_id: str
    task_name: str
    world_y: int
    world_state: int
    model_response: str
    parsed_final_answer: Optional[int]
    correct: bool
    final_prompt_messages: Optional[List[Dict[str, str]]] = None


def _state_bits(state: int, n: int) -> str:
    return "".join(str((int(state) >> i) & 1) for i in range(int(n)))


def _extract_answer_payload(model_name: str, response: str) -> str:
    clean = mt.extract_non_thinking(model_name, response).strip()
    m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", clean)
    return m.group(1).strip() if m else clean


def _build_fullinfo_prompt(sample: mt.Sample, world: mt.BranchWorld) -> List[Dict[str, str]]:
    gene_idx_txt = sample.prompt_gene_index_text
    rules_txt = (sample.raw_rules_text or "").strip()

    if sample.task_name == "ss_id":
        system = SYSTEM_PROMPT_FULLINFO_SS_ID
        user = (
            f"Gene index order for bitstrings (left->right): {gene_idx_txt}\n\n"
            f"Given full steady-state vector bits:\n"
            f"{_state_bits(world.state, sample.n_nodes)}\n\n"
            f"Steady-state attractor catalog (ID -> full binary state):\n"
            f"{sample.prompt_fixed_points_catalog}\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if sample.task_name == "dyn_attr":
        if not rules_txt:
            raise RuntimeError(
                f"Missing raw update rules for dynamic fullinfo sample group_id={sample.group_id} model={sample.model}"
            )
        system = SYSTEM_PROMPT_FULLINFO_DYN_ATTR
        user = (
            f"Gene index order for bitstrings (left->right): {gene_idx_txt}\n\n"
            f"Full initial-state vector bits:\n"
            f"{_state_bits(world.state, sample.n_nodes)}\n\n"
            f"Synchronous Boolean update rules:\n"
            f"{rules_txt}\n\n"
            f"Attractor catalog (fixed points and cycles):\n"
            f"{sample.prompt_dyn_attractors_catalog}\n\n"
            "Catalog format:\n"
            "- FixedPoint: a single bitstring state.\n"
            "- Cycle: multiple bitstrings joined by \" | \" (one state after another in the cycle).\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # dyn_marker
    if not rules_txt:
        raise RuntimeError(
            f"Missing raw update rules for dynamic fullinfo sample group_id={sample.group_id} model={sample.model}"
        )
    system = SYSTEM_PROMPT_FULLINFO_DYN_MARKER
    user = (
        f"Marker gene: {sample.marker_gene}\n"
        f"Gene index order for bitstrings (left->right): {gene_idx_txt}\n\n"
        f"Full initial-state vector bits:\n"
        f"{_state_bits(world.state, sample.n_nodes)}\n\n"
        f"Synchronous Boolean update rules:\n"
        f"{rules_txt}\n\n"
        f"Attractor catalog (fixed points and cycles):\n"
        f"{sample.prompt_dyn_attractors_catalog}\n\n"
        "Catalog format:\n"
        "- FixedPoint: a single bitstring state.\n"
        "- Cycle: multiple bitstrings joined by \" | \" (one state after another in the cycle).\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_fullinfo_identifiability(
    sample: mt.Sample,
    world: mt.BranchWorld,
) -> Tuple[bool, str]:
    """
    Verify that providing the full state (steady-state for ss_id; initial-state for dynamic tasks)
    determines a unique target outcome for this sample.
    """
    full_ctx = {int(i): int((int(world.state) >> int(i)) & 1) for i in range(int(sample.n_nodes))}
    outcomes = mt._candidate_outcomes(sample, full_ctx)
    if outcomes.size == 0:
        return False, "no_candidate_outcomes_under_full_state"
    uniq = np.unique(outcomes)
    if int(uniq.size) != 1:
        return False, f"non_unique_outcome_under_full_state(n={int(uniq.size)})"
    if int(uniq[0]) != int(world.y):
        return False, f"world_y_mismatch(unique={int(uniq[0])}, world_y={int(world.y)})"
    return True, "ok"


def _episode_key(sample: mt.Sample, world: mt.BranchWorld) -> str:
    d = {
        "mode": "fullinfo",
        "sample_id": int(sample.sample_id),
        "group_id": sample.group_id,
        "task_name": sample.task_name,
        "world_y": int(world.y),
        "world_state": int(world.state),
    }
    return json.dumps(d, sort_keys=True)


def _result_to_dict(r: FullInfoEpisodeResult) -> Dict[str, Any]:
    return {
        "sample_id": r.sample_id,
        "group_id": r.group_id,
        "task_name": r.task_name,
        "world": {"y": r.world_y, "state": r.world_state},
        "model_response": r.model_response,
        "parsed_final_answer": r.parsed_final_answer,
        "correct": r.correct,
        "final_prompt_messages": r.final_prompt_messages,
    }


def _dict_to_result(d: Dict[str, Any]) -> FullInfoEpisodeResult:
    return FullInfoEpisodeResult(
        sample_id=int(d["sample_id"]),
        group_id=str(d["group_id"]),
        task_name=str(d["task_name"]),
        world_y=int(d["world"]["y"]),
        world_state=int(d["world"]["state"]),
        model_response=str(d.get("model_response", "")),
        parsed_final_answer=d.get("parsed_final_answer"),
        correct=bool(d.get("correct", False)),
        final_prompt_messages=d.get("final_prompt_messages"),
    )


def _load_cache(cache_file: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(cache_file):
        return out
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
                k = x.get("key")
                v = x.get("result")
                if k and v:
                    out[k] = v
            except Exception:
                continue
    return out


def _save_cache_entry(cache_file: str, key: str, result: FullInfoEpisodeResult) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "result": _result_to_dict(result)}, ensure_ascii=False) + "\n")


async def _run_one_episode(
    *,
    model_name: str,
    port: str,
    sample: mt.Sample,
    world: mt.BranchWorld,
    generation_config: Dict[str, Any],
    episode_cache: Optional[Dict[str, Dict[str, Any]]],
    episode_cache_file: Optional[str],
    dump_prompts: bool,
    verbose: bool,
) -> FullInfoEpisodeResult:
    key = _episode_key(sample, world)
    if episode_cache is not None and key in episode_cache:
        return _dict_to_result(episode_cache[key])

    messages = _build_fullinfo_prompt(sample, world)
    if dump_prompts:
        print(
            f"\n[fullinfo_prompt] sample={sample.sample_id} group={sample.group_id} task={sample.task_name}\n"
            f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n",
            flush=True,
        )

    out = await async_generate_single(
        model_name=model_name,
        port=port,
        messages=copy.deepcopy(messages),
        generation_config=generation_config,
    )
    response = out.text
    ans_payload = _extract_answer_payload(model_name, response)
    parsed = mt.parse_answer_value(sample, ans_payload)
    correct = mt.check_answer_correct(sample, parsed, world)

    result = FullInfoEpisodeResult(
        sample_id=sample.sample_id,
        group_id=sample.group_id,
        task_name=sample.task_name,
        world_y=int(world.y),
        world_state=int(world.state),
        model_response=response,
        parsed_final_answer=parsed,
        correct=bool(correct),
        final_prompt_messages=copy.deepcopy(messages) if dump_prompts else None,
    )

    if episode_cache is not None and episode_cache_file is not None:
        episode_cache[key] = _result_to_dict(result)
        _save_cache_entry(episode_cache_file, key, result)

    if verbose:
        print(f"[fullinfo_episode] sample={sample.sample_id} task={sample.task_name} y={world.y} correct={correct}")

    return result


async def _run_all(
    *,
    episodes: List[Tuple[mt.Sample, mt.BranchWorld]],
    model_name: str,
    port: str,
    generation_config: Dict[str, Any],
    episode_cache: Optional[Dict[str, Dict[str, Any]]],
    episode_cache_file: Optional[str],
    max_concurrent: int,
    dump_prompts: bool,
    verbose: bool,
) -> List[FullInfoEpisodeResult]:
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(s: mt.Sample, w: mt.BranchWorld) -> FullInfoEpisodeResult:
        async with sem:
            return await _run_one_episode(
                model_name=model_name,
                port=port,
                sample=s,
                world=w,
                generation_config=generation_config,
                episode_cache=episode_cache,
                episode_cache_file=episode_cache_file,
                dump_prompts=dump_prompts,
                verbose=verbose,
            )

    tasks = [_one(s, w) for s, w in episodes]
    out = await asyncio.gather(*tasks)
    return list(out)


def _metrics(results: List[FullInfoEpisodeResult], sample_map: Dict[int, mt.Sample]) -> Dict[str, Any]:
    if not results:
        return {"total": 0, "accuracy": 0.0, "per_task": {}, "per_k": {}}

    total = len(results)
    acc = sum(1 for r in results if r.correct) / total

    per_task: Dict[str, List[bool]] = {}
    per_k: Dict[int, List[bool]] = {}
    for r in results:
        per_task.setdefault(r.task_name, []).append(bool(r.correct))
        kk = int(sample_map[r.sample_id].k)
        per_k.setdefault(kk, []).append(bool(r.correct))

    return {
        "total": total,
        "accuracy": float(acc),
        "per_task": {k: float(sum(v) / len(v)) for k, v in sorted(per_task.items())},
        "per_k": {str(k): float(sum(v) / len(v)) for k, v in sorted(per_k.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="GRN full-info evaluator (single-turn)")
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--port", type=str, default="8011")
    ap.add_argument("--data-file", type=str, required=True)
    ap.add_argument("--cache-dir", type=str, required=True)
    ap.add_argument("--models-dir", type=str, default=mt.DEFAULT_GRN_MODELS_DIR)
    ap.add_argument("--include-tasks", type=str, default="ss_id,dyn_attr,dyn_marker")
    ap.add_argument("--worlds-per-sufficient-set-per-sign", type=int, default=1)
    ap.add_argument("--worlds-per-sample", type=int, default=None,
                    help="Legacy alias. If set, treated as --worlds-per-sufficient-set-per-sign")
    ap.add_argument("--max-samples-per-task-k", type=int, default=0)
    ap.add_argument("--results-dir", type=str, default="results/grn/fullinfo")
    ap.add_argument("--cache-tag", type=str, default="")
    ap.add_argument("--output-tag", type=str, default="",
                    help="Optional concise stem for result/cache filenames. Dynamic suffixes are still appended.")
    ap.add_argument("--max-concurrent", type=int, default=64)
    ap.add_argument("--reasoning-effort", type=str, default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--dump-prompts", action="store_true")
    ap.add_argument("--dry-run-load-only", action="store_true",
                    help="Load tasks/cache, validate identifiability and episode expansion, then exit before model inference.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    include_tasks = [t.strip() for t in args.include_tasks.split(",") if t.strip()]
    valid_tasks = {"ss_id", "dyn_attr", "dyn_marker"}
    bad = [t for t in include_tasks if t not in valid_tasks]
    if bad:
        raise ValueError(f"Unknown tasks in --include-tasks: {bad}")

    model_name_safe = args.model_name.replace("/", "_")
    out_dir = os.path.join(args.results_dir, model_name_safe)
    os.makedirs(out_dir, exist_ok=True)
    ep_cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(ep_cache_dir, exist_ok=True)

    worlds_per_sign = (
        int(args.worlds_per_sample)
        if args.worlds_per_sample is not None
        else int(args.worlds_per_sufficient_set_per_sign)
    )

    if args.output_tag:
        output_name = args.output_tag
    else:
        data_tag = args.cache_tag if args.cache_tag else os.path.splitext(os.path.basename(args.data_file))[0]
        task_tag = "-".join(include_tasks)
        output_name = f"{data_tag}-{task_tag}-fullinfo-worldsps{worlds_per_sign}-seed{args.seed}"
    if args.max_samples_per_task_k > 0:
        output_name += f"-cap{args.max_samples_per_task_k}pertaskk"
    if args.dump_prompts:
        output_name += "-dumpprompt"

    ep_cache_file = os.path.join(ep_cache_dir, f"{output_name}_episodes.jsonl")
    ep_cache = _load_cache(ep_cache_file)

    generation_config = Evaluator(
        model_name=args.model_name,
        reasoning_effort=args.reasoning_effort,
    ).generation_config

    samples = mt.load_data(
        tasks_file=args.data_file,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        include_tasks=include_tasks,
        seed=args.seed,
    )
    mt._print_task_k_counts("Sample counts by (task, k) before optional subsample:", samples)
    samples = mt._subsample_by_task_k(samples, max_samples_per_task_k=args.max_samples_per_task_k, seed=args.seed)
    if args.max_samples_per_task_k > 0:
        mt._print_task_k_counts(
            f"Sample counts by (task, k) after capping at {args.max_samples_per_task_k} per cell:",
            samples,
        )

    episodes = mt.select_worlds(samples, worlds_per_sign=worlds_per_sign, seed=args.seed)
    print(f"Total full-info episodes: {len(episodes)}")

    bad_ident: List[Dict[str, Any]] = []
    for s, w in episodes:
        ok, reason = _validate_fullinfo_identifiability(s, w)
        if not ok:
            bad_ident.append({
                "sample_id": int(s.sample_id),
                "group_id": s.group_id,
                "task_name": s.task_name,
                "world_y": int(w.y),
                "reason": reason,
            })
    if bad_ident:
        print(f"[fullinfo_check] failed episodes: {len(bad_ident)} / {len(episodes)}")
        for x in bad_ident[:20]:
            print(f"  - {x}")
        raise RuntimeError(
            "Full-info identifiability check failed: full-state information does not uniquely determine target."
        )
    print(f"[fullinfo_check] passed: {len(episodes)} / {len(episodes)} episodes uniquely identifiable.")

    if args.dry_run_load_only:
        print("[dry_run_load_only] schema / cache / identifiability checks passed; exiting before inference.")
        return

    results = asyncio.run(
        _run_all(
            episodes=episodes,
            model_name=args.model_name,
            port=args.port,
            generation_config=generation_config,
            episode_cache=ep_cache,
            episode_cache_file=ep_cache_file,
            max_concurrent=args.max_concurrent,
            dump_prompts=args.dump_prompts,
            verbose=args.verbose,
        )
    )

    sample_map = {s.sample_id: s for s in samples}
    metrics = _metrics(results, sample_map)
    print("=== FullInfo Metrics ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_file = os.path.join(out_dir, f"{output_name}_results.json")
    payload = {
        "config": vars(args),
        "metrics": metrics,
        "episodes": [
            {
                **_result_to_dict(r),
                "k": int(sample_map[r.sample_id].k),
                "minimal_sets": sample_map[r.sample_id].minimal_sets,
            }
            for r in results
        ],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
