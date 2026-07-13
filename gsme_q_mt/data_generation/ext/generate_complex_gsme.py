from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

from generator import ComplexGSMEGenerator
from formatter import csv_ready_row, format_sample_for_symbolic_output, json_ready_row
from validator import validate_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structurally complex GSME-style samples.")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_vars", type=int, default=12)
    parser.add_argument("--num_vars_min", type=int, default=None)
    parser.add_argument("--num_vars_max", type=int, default=None)
    parser.add_argument("--num_rules", type=int, default=10)
    parser.add_argument("--num_rules_min", type=int, default=None)
    parser.add_argument("--num_rules_max", type=int, default=None)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--depth_min", type=int, default=None)
    parser.add_argument("--depth_max", type=int, default=None)
    parser.add_argument("--min_relevant_leaves", type=int, default=None)
    parser.add_argument("--max_relevant_leaves", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./generated_gsme_q_mt_ext")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_distractors", action="store_true")
    parser.add_argument("--min_value", type=int, default=2)
    parser.add_argument("--max_value", type=int, default=40)
    parser.add_argument("--max_attempts_per_sample", type=int, default=25)
    parser.add_argument("--holdout_k", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--holdout_candidates", type=int, default=200)
    parser.add_argument("--holdout_top_m", type=int, default=5)
    parser.add_argument(
        "--holdout_difficulty_mode",
        type=str,
        default="distance",
        choices=["distance", "ycands"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.k_max is not None and args.k_max < 1:
        raise ValueError("--k_max must be >= 1")
    if args.holdout_k < 0:
        raise ValueError("--holdout_k must be >= 0")

    generator = ComplexGSMEGenerator(
        rng=rng,
        min_value=args.min_value,
        max_value=args.max_value,
        allow_distractors=args.allow_distractors,
    )

    samples: List[Dict[str, Any]] = []
    rejected = 0
    holdout_schedule = build_holdout_schedule(
        num_samples=args.num_samples,
        holdout_k=args.holdout_k,
        k_max=args.k_max,
        rng=rng,
    )

    for idx in range(args.num_samples):
        sample_id = f"gsme_q_mt_ext_{idx:05d}"
        sampled_num_vars = sample_in_range(args.num_vars, args.num_vars_min, args.num_vars_max, rng)
        sampled_num_rules = sample_in_range(args.num_rules, args.num_rules_min, args.num_rules_max, rng)
        sampled_depth = sample_in_range(args.depth, args.depth_min, args.depth_max, rng)
        sampled_target_relevant_leaves = sample_in_range(
            exact=max(2, min(sampled_depth + 1, sampled_depth // 2 + 2)),
            minimum=args.min_relevant_leaves,
            maximum=args.max_relevant_leaves,
            rng=rng,
        )
        target_holdout_k = holdout_schedule[idx]
        success = False
        for _ in range(args.max_attempts_per_sample):
            sample = generator.generate_sample(
                sample_id=sample_id,
                num_vars=sampled_num_vars,
                num_rules=sampled_num_rules,
                depth=sampled_depth,
                domain="symbolic",
                target_relevant_leaves=sampled_target_relevant_leaves,
            )

            if len(sample["relevant_leaf"]) < target_holdout_k:
                continue

            is_valid, errors = validate_sample(
                sample,
                requested_num_vars=sampled_num_vars,
                requested_num_rules=sampled_num_rules,
                requested_depth=sampled_depth,
            )
            if is_valid:
                formatted = format_sample_for_symbolic_output(
                    sample,
                    rng=rng,
                    holdout_k=target_holdout_k,
                    holdout_candidates=args.holdout_candidates,
                    holdout_top_m=args.holdout_top_m,
                    holdout_difficulty_mode=args.holdout_difficulty_mode,
                )
                if target_holdout_k > 0 and not formatted.get("ksufficient_found", False):
                    continue
                formatted["sampled_num_vars_target"] = sampled_num_vars
                formatted["sampled_num_rules_target"] = sampled_num_rules
                formatted["sampled_depth_target"] = sampled_depth
                formatted["sampled_target_relevant_leaves"] = sampled_target_relevant_leaves
                formatted["actual_relevant_leaf_count"] = len(sample["relevant_leaf"])
                formatted["sampled_holdout_k_target"] = target_holdout_k
                samples.append(formatted)
                success = True
                break
        if not success:
            rejected += 1

    if not samples:
        raise RuntimeError("no valid samples generated; try increasing max_attempts_per_sample")

    jsonl_path = output_dir / "gsme_q_mt_ext.jsonl"
    csv_path = output_dir / "gsme_q_mt_ext.csv"
    summary_path = output_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(json_ready_row(sample), ensure_ascii=False) + "\n")

    flat_rows = [csv_ready_row(sample) for sample in samples]
    if pd is not None:
        dataframe = pd.DataFrame(flat_rows)
        dataframe.to_csv(csv_path, index=False)
    else:
        write_csv_without_pandas(csv_path, flat_rows)

    summary = build_summary(samples=samples, requested=args.num_samples, rejected=rejected, seed=args.seed)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated {len(samples)} samples to {output_dir.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Example sample:")
    print(json.dumps(example_projection(samples[0]), ensure_ascii=False, indent=2))


def build_summary(
    samples: List[Dict[str, Any]],
    requested: int,
    rejected: int,
    seed: int,
) -> Dict[str, Any]:
    return {
        "seed": seed,
        "requested_num_samples": requested,
        "generated_num_samples": len(samples),
        "rejected_samples": rejected,
        "average_num_vars": round(sum(item["num_vars"] for item in samples) / len(samples), 3),
        "average_num_rules": round(sum(item["num_rules"] for item in samples) / len(samples), 3),
        "average_depth": round(sum(item["depth"] for item in samples) / len(samples), 3),
        "proportion_with_distractors": round(
            sum(1 for item in samples if item["has_distractor"]) / len(samples), 3
        ),
        "proportion_with_merges": round(
            sum(1 for item in samples if item["has_merge"]) / len(samples), 3
        ),
        "sampled_num_vars_range": [
            min(item["sampled_num_vars_target"] for item in samples),
            max(item["sampled_num_vars_target"] for item in samples),
        ],
        "sampled_num_rules_range": [
            min(item["sampled_num_rules_target"] for item in samples),
            max(item["sampled_num_rules_target"] for item in samples),
        ],
        "sampled_depth_range": [
            min(item["sampled_depth_target"] for item in samples),
            max(item["sampled_depth_target"] for item in samples),
        ],
        "sampled_holdout_k_range": [
            min(item["sampled_holdout_k_target"] for item in samples),
            max(item["sampled_holdout_k_target"] for item in samples),
        ],
        "sampled_target_relevant_leaves_range": [
            min(item["sampled_target_relevant_leaves"] for item in samples),
            max(item["sampled_target_relevant_leaves"] for item in samples),
        ],
        "average_actual_relevant_leaf_count": round(
            sum(item["actual_relevant_leaf_count"] for item in samples) / len(samples), 3
        ),
        "used_pandas_for_csv": pd is not None,
    }


def sample_in_range(exact: int, minimum: Optional[int], maximum: Optional[int], rng: random.Random) -> int:
    if minimum is None and maximum is None:
        return exact
    if minimum is None or maximum is None:
        raise ValueError("range sampling requires both min and max values")
    if minimum > maximum:
        raise ValueError("range min cannot be greater than max")
    return rng.randint(minimum, maximum)


def build_holdout_schedule(
    num_samples: int,
    holdout_k: int,
    k_max: Optional[int],
    rng: random.Random,
) -> List[int]:
    if k_max is None:
        return [holdout_k] * num_samples
    schedule = [(idx % k_max) + 1 for idx in range(num_samples)]
    rng.shuffle(schedule)
    return schedule


def example_projection(sample: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "sample_id",
        "problem_id",
        "Full Problem",
        "CSP",
        "Full Answer",
        "depth",
        "Pred Values",
        "Heldout Value",
        "Rewritten Problem",
        "Possible Questions",
        "Given_Conditions",
        "goal_var",
        "dist_max",
        "dist_mean",
        "diff_score",
        "answer",
        "num_vars",
        "num_rules",
        "has_merge",
        "has_distractor",
        "ksufficient_found",
        "sampled_holdout_k_target",
        "sampled_target_relevant_leaves",
        "actual_relevant_leaf_count",
        "sampled_num_vars_target",
        "sampled_num_rules_target",
        "sampled_depth_target",
    ]
    return {key: sample[key] for key in keys}


def write_csv_without_pandas(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    preferred_prefix = [
        "sample_id",
        "problem_id",
        "Full Problem",
        "CSP",
        "Full Answer",
        "Variables",
        "Equations",
        "depth",
        "Pred Values",
        "Heldout Value",
        "Rewritten Problem",
        "Possible Questions",
        "Given_Conditions",
        "k",
        "diff_score",
        "goal_var",
        "leaf_nodes_all",
        "relevant_leaf",
        "ancestors_goal",
        "dist_max",
        "dist_mean",
    ]
    remaining = [key for key in rows[0].keys() if key not in preferred_prefix]
    fieldnames = preferred_prefix + remaining
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
