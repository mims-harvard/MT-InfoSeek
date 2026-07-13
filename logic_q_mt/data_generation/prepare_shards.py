#!/usr/bin/env python3
"""Stage 0: prepare per-shard directories for Logic-Q-MT stage-1 generation.

Each stage-1 shard directory must contain exactly ONE QuestBench Logic-Q-RP
``prop_examples_*.txt`` ruleset file. ``SimpleLogic/ruleset.py: load_data`` reads
one ``.txt`` per directory by design (a load-bearing ``break``), and the
parallel stage-1 script + the postprocessing notebook both expect the shard
layout ``<intermediate_dir>/new_<i>_<suffix>/``.

This script creates one such directory per raw ruleset file and symlinks (or
copies, with ``--copy``) the file into it. The released data used 30 shards
(``new_0_500k`` .. ``new_29_500k``); pass ``--max_shards 30`` to reproduce that
exactly, or omit it to create one shard per available ruleset file.

Usage:
    python logic_q_mt/data_generation/prepare_shards.py \
        --rp_dir "$LOGIC_Q_RP_DIR" \
        --intermediate_dir "$LOGIC_Q_INTERMEDIATE_DIR" \
        [--suffix 500k] [--max_shards 30] [--copy]

Env fallbacks: ``LOGIC_Q_RP_DIR``, ``LOGIC_Q_INTERMEDIATE_DIR`` (see .env.example).
"""
import argparse
import glob
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create per-shard dirs (new_<i>_<suffix>/) for Logic-Q-MT stage 1."
    )
    ap.add_argument(
        "--rp_dir",
        default=os.environ.get("LOGIC_Q_RP_DIR", ""),
        help="QuestBench Logic-Q/RP/RP dir containing prop_examples_*.txt.",
    )
    ap.add_argument(
        "--intermediate_dir",
        default=os.environ.get("LOGIC_Q_INTERMEDIATE_DIR", ""),
        help="Output root for the new_<i>_<suffix>/ shard dirs.",
    )
    ap.add_argument(
        "--suffix",
        default="500k",
        help="Shard suffix matching --max_expansions_per_layer (500k=500000, 1m=1000000).",
    )
    ap.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Only create the first N shards (default: one per prop_examples_*.txt).",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking them.",
    )
    args = ap.parse_args()

    if not args.rp_dir:
        raise SystemExit("--rp_dir (or LOGIC_Q_RP_DIR) is required.")
    if not args.intermediate_dir:
        raise SystemExit("--intermediate_dir (or LOGIC_Q_INTERMEDIATE_DIR) is required.")

    txt_files = sorted(glob.glob(os.path.join(args.rp_dir, "prop_examples_*.txt")))
    if not txt_files:
        raise SystemExit(f"No prop_examples_*.txt found under {args.rp_dir}.")
    if args.max_shards is not None:
        txt_files = txt_files[: args.max_shards]

    os.makedirs(args.intermediate_dir, exist_ok=True)
    for i, src in enumerate(txt_files):
        shard_dir = os.path.join(args.intermediate_dir, f"new_{i}_{args.suffix}")
        os.makedirs(shard_dir, exist_ok=True)
        dst = os.path.join(shard_dir, os.path.basename(src))
        if os.path.lexists(dst):
            os.remove(dst)
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)

    print(
        f"Created {len(txt_files)} shard dir(s) under {args.intermediate_dir} "
        f"(suffix={args.suffix}, {'copied' if args.copy else 'symlinked'})."
    )


if __name__ == "__main__":
    main()
