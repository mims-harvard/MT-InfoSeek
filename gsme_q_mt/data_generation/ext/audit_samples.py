from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and format GSME records for manual audit."
    )
    parser.add_argument("--input", required=True, help="Input CSV or JSONL dataset path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown report path. Defaults to <input>.audit.md",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=12,
        help="How many rows to sample into the audit report",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_mode",
        choices=["random", "balanced"],
        default="balanced",
        help="Sampling strategy. balanced tries to cover different k/depth/flags.",
    )
    parser.add_argument(
        "--include_fields",
        nargs="*",
        default=[
            "sample_id",
            "problem_id",
            "k",
            "depth",
            "goal_var",
            "diff_score",
            "has_distractor",
            "has_merge",
            "Heldout Value",
            "Given_Conditions",
            "relevant_leaf",
            "ancestors_goal",
            "Possible Questions",
        ],
        help="Extra scalar/list fields to show in each audit section",
    )
    parser.add_argument(
        "--write_selected_csv",
        action="store_true",
        help="Also write a CSV with only the selected audit rows",
    )
    parser.add_argument(
        "--max_problem_chars",
        type=int,
        default=4000,
        help="Trim very long text blocks in the markdown report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = load_rows(input_path)
    if not rows:
        raise RuntimeError("No rows found in input dataset")

    rng = random.Random(args.seed)
    selected = select_rows(rows, args.num_samples, args.sample_mode, rng)

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    markdown = build_markdown_report(
        input_path=input_path,
        rows=selected,
        total_rows=len(rows),
        seed=args.seed,
        sample_mode=args.sample_mode,
        include_fields=args.include_fields,
        max_problem_chars=args.max_problem_chars,
    )
    output_path.write_text(markdown, encoding="utf-8")

    if args.write_selected_csv:
        selected_csv_path = output_path.with_suffix(".selected.csv")
        write_selected_csv(selected_csv_path, selected)

    print(f"Wrote audit report: {output_path}")
    print(f"Audited {len(selected)} samples from {len(rows)} total rows")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [normalize_row(dict(row)) for row in reader]
    raise ValueError("Unsupported input format. Use .csv or .jsonl")


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(value, str):
            out[key] = value
            continue
        stripped = value.strip()
        if not stripped:
            out[key] = value
            continue
        if stripped in {"true", "false"}:
            out[key] = stripped == "true"
            continue
        parsed = maybe_parse_jsonish(stripped)
        out[key] = parsed
    return out


def maybe_parse_jsonish(text: str) -> Any:
    if not text:
        return text
    if text[0] in "[{" and text[-1] in "]}":
        try:
            return json.loads(text)
        except Exception:
            return text
    try:
        if "." in text:
            numeric = float(text)
            return int(numeric) if numeric.is_integer() else numeric
        return int(text)
    except Exception:
        return text


def select_rows(
    rows: Sequence[Mapping[str, Any]],
    num_samples: int,
    sample_mode: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    num_samples = min(num_samples, len(rows))
    if sample_mode == "random":
        return [dict(item) for item in rng.sample(list(rows), num_samples)]

    indexed_rows = [(idx, dict(row)) for idx, row in enumerate(rows)]
    groups: Dict[Tuple[Any, Any, Any, Any], List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, row in indexed_rows:
        key = (
            row.get("k", row.get("sampled_holdout_k_target")),
            depth_bucket(row.get("depth")),
            boolish(row.get("has_distractor")),
            boolish(row.get("has_merge")),
        )
        groups.setdefault(key, []).append((idx, row))

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            normalize_sortable(item[0][0]),
            normalize_sortable(item[0][1]),
            normalize_sortable(item[0][2]),
            normalize_sortable(item[0][3]),
        ),
    )
    for _, members in ordered_groups:
        rng.shuffle(members)

    selected: List[Tuple[int, Dict[str, Any]]] = []
    seen_indices = set()
    exhausted = False
    while len(selected) < num_samples and not exhausted:
        exhausted = True
        for _, members in ordered_groups:
            while members and members[0][0] in seen_indices:
                members.pop(0)
            if not members:
                continue
            exhausted = False
            idx, row = members.pop(0)
            selected.append((idx, row))
            seen_indices.add(idx)
            if len(selected) >= num_samples:
                break

    if len(selected) < num_samples:
        remaining = [(idx, row) for idx, row in indexed_rows if idx not in seen_indices]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_samples - len(selected)])

    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected[:num_samples]]


def depth_bucket(depth: Any) -> str:
    try:
        depth_value = int(depth)
    except Exception:
        return "unknown"
    if depth_value <= 3:
        return "low"
    if depth_value <= 6:
        return "mid"
    return "high"


def normalize_sortable(value: Any) -> Tuple[int, str]:
    if value is None:
        return (1, "")
    return (0, str(value))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_suffix(".audit.md")
    return input_path.parent / f"{input_path.name}.audit.md"


def build_markdown_report(
    input_path: Path,
    rows: Sequence[Mapping[str, Any]],
    total_rows: int,
    seed: int,
    sample_mode: str,
    include_fields: Sequence[str],
    max_problem_chars: int,
) -> str:
    lines: List[str] = []
    lines.append("# GSME Audit Report")
    lines.append("")
    lines.append(f"- Source: `{input_path}`")
    lines.append(f"- Total rows: {total_rows}")
    lines.append(f"- Audited rows: {len(rows)}")
    lines.append(f"- Seed: {seed}")
    lines.append(f"- Sampling mode: `{sample_mode}`")
    lines.append("")
    lines.append("## Audit Checklist")
    lines.append("")
    lines.append("- Structure: goal, relevant leaves, ancestors, and distractors are consistent.")
    lines.append("- Sufficiency: `Given_Conditions` should be insufficient alone, and adding `Heldout Value` should close the chain.")
    lines.append("- Equations: no dangling variables, no duplicate definitions, and rule dependencies are ordered sensibly.")
    lines.append("- Arithmetic: spot-check a few intermediate values and the goal against `Pred Values`.")
    lines.append("- Difficulty: `k`, depth, merges, and distractors look aligned with the intended complexity.")
    lines.append("")
    lines.append("## Coverage Summary")
    lines.append("")
    lines.extend(render_coverage_summary(rows))
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        lines.extend(render_row_section(idx, row, include_fields, max_problem_chars))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_coverage_summary(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    k_counts: Dict[str, int] = {}
    depth_counts: Dict[str, int] = {}
    distractor_true = 0
    merge_true = 0
    for row in rows:
        k_counts[str(row.get("k", row.get("sampled_holdout_k_target", "unknown")))] = (
            k_counts.get(str(row.get("k", row.get("sampled_holdout_k_target", "unknown"))), 0) + 1
        )
        bucket = depth_bucket(row.get("depth"))
        depth_counts[bucket] = depth_counts.get(bucket, 0) + 1
        distractor_true += int(boolish(row.get("has_distractor")))
        merge_true += int(boolish(row.get("has_merge")))

    lines = [
        f"- k distribution: {format_count_map(k_counts)}",
        f"- depth buckets: {format_count_map(depth_counts)}",
        f"- has_distractor=true: {distractor_true}/{len(rows)}",
        f"- has_merge=true: {merge_true}/{len(rows)}",
    ]
    return lines


def format_count_map(counter: Mapping[str, int]) -> str:
    if not counter:
        return "(empty)"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def render_row_section(
    index: int,
    row: Mapping[str, Any],
    include_fields: Sequence[str],
    max_problem_chars: int,
) -> List[str]:
    sample_id = row.get("sample_id", f"row_{index}")
    lines = [f"## Sample {index}: `{sample_id}`", ""]
    lines.append("### Quick Metadata")
    lines.append("")
    for field in include_fields:
        if field not in row:
            continue
        lines.append(f"- {field}: {format_inline_value(row.get(field))}")
    lines.append("")

    lines.append("### Manual Checks")
    lines.append("")
    lines.append("- [ ] Goal dependency chain looks valid")
    lines.append("- [ ] Heldout variables come from relevant leaves")
    lines.append("- [ ] Given conditions alone seem insufficient")
    lines.append("- [ ] Given conditions + heldout values seem sufficient")
    lines.append("- [ ] Arithmetic spot-check passes")
    lines.append("- [ ] Distractors do not affect the goal")
    lines.append("")

    lines.append("### Full Problem")
    lines.append("")
    lines.append(code_block(trim_text(str(row.get("Full Problem", "")), max_problem_chars)))
    lines.append("")

    lines.append("### Rewritten Problem")
    lines.append("")
    lines.append(code_block(trim_text(str(row.get("Rewritten Problem", "")), max_problem_chars)))
    lines.append("")

    lines.append("### CSP")
    lines.append("")
    lines.append(code_block(trim_text(str(row.get("CSP", "")), max_problem_chars)))
    lines.append("")

    lines.append("### Pred Values")
    lines.append("")
    lines.append(code_block(trim_text(str(row.get("Pred Values", "")), max_problem_chars)))
    lines.append("")

    lines.append("### Audit Notes")
    lines.append("")
    lines.append("- Notes:")
    return lines


def code_block(text: str) -> str:
    return f"```text\n{text}\n```"


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... [truncated]"


def format_inline_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return f"`{json.dumps(value, ensure_ascii=False)}`"
    return f"`{value}`"


def write_selected_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key)) for key in fieldnames})


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
