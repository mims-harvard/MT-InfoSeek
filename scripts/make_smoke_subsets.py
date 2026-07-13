#!/usr/bin/env python3
"""Build the tiny, k-stratified smoke subsets under `data/smoke/`.

These are what the wrapper runs by default: enough rows to exercise every code
path (each k, each task) with a small run, without touching `data/`. Hosted APIs
still incur real cost, especially for the 20Q offline judge stage.

Deterministic: fixed seed, stable sort. Re-running overwrites `data/smoke/` and
never modifies `data/`.

    python scripts/make_smoke_subsets.py [--per-k 2]

20-Questions needs no subset file: it is sliced with
`--task_start_index/--task_end_index`.
"""
import argparse
import json
import os

import pandas as pd

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "smoke")


def subset_csv(name: str, per_k: int) -> None:
    src = os.path.join(DATA, name)
    df = pd.read_csv(src)
    if "k" not in df.columns:
        raise SystemExit(f"{name}: no 'k' column to stratify on")
    # Deterministic: sample within each k with a fixed seed, then restore order.
    picked = (
        df.groupby("k", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_k, len(g)), random_state=SEED))
        .sort_index()
    )
    dst = os.path.join(OUT, name)
    picked.to_csv(dst, index=False)
    print(f"{name:24} {len(picked):>4} rows  k={picked['k'].value_counts().sort_index().to_dict()}")


def subset_genereg(per_k: int) -> None:
    """Stratify by (task, k_min) and prefer the smallest networks.

    Restricting to few distinct GRN `model`s keeps the smoke attractor cache
    small; preferring small `n_nodes` keeps the oracle fast.
    """
    src = os.path.join(DATA, "genereg_mt.jsonl")
    rows = [json.loads(line) for line in open(src)]

    keep, seen_models = [], set()
    strata = sorted({(r["family"], r["target_type"], r["k_min"]) for r in rows})
    for fam, target, k in strata:
        cands = [r for r in rows if (r["family"], r["target_type"], r["k_min"]) == (fam, target, k)]
        # smallest networks first, then stable by group_id for determinism
        cands.sort(key=lambda r: (r["n_nodes"], str(r.get("group_id"))))
        chosen = cands[:per_k]
        keep.extend(chosen)
        seen_models.update(r["model"] for r in chosen)

    dst = os.path.join(OUT, "genereg_mt.jsonl")
    with open(dst, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")

    tasks = {}
    for r in keep:
        tasks[(r["family"], r["target_type"])] = tasks.get((r["family"], r["target_type"]), 0) + 1
    print(f"{'genereg_mt.jsonl':24} {len(keep):>4} rows  tasks={tasks}")
    print(f"{'':24}      {len(seen_models)} distinct GRN models "
          f"(n_nodes {min(r['n_nodes'] for r in keep)}-{max(r['n_nodes'] for r in keep)})")
    with open(os.path.join(OUT, "genereg_models.txt"), "w") as f:
        for m in sorted(seen_models):
            f.write(m + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-k", type=int, default=2, help="rows per k (per task for genereg)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f"Writing k-stratified smoke subsets to {OUT} (seed={SEED})\n")
    for name in ("logic_q_mt.csv", "gsme_q_mt.csv", "gsme_q_mt_ext.csv"):
        subset_csv(name, args.per_k)
    subset_genereg(args.per_k)
    print("\n20q: no subset file needed (use --task_start_index/--task_end_index).")


if __name__ == "__main__":
    main()
