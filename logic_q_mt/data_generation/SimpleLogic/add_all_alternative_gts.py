import argparse
import ast
import itertools
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Set, Tuple
import pandas as pd
from tqdm import tqdm

# Import horn_sat_utils.py (place it alongside this script or in PYTHONPATH)
from SimpleLogic.horn_sat_utils import solve_unit_prop, parse_clauses, CONTRADICTION, forced_value_via_refutation, INFEASIBLE

DATA_DIR = os.environ.get("DATA_DIR", "")


def _parse_cell(cell: Any):
    """Parse a cell that may be a python-literal string like "['a','b']"."""
    if isinstance(cell, str):
        return ast.literal_eval(cell)
    return cell


def _strip_not(lit: str) -> str:
    return lit[4:] if lit.startswith("not ") else lit


def _normalize_known(true_list: List[str], false_list: List[str]) -> Dict[str, bool]:
    """
    Returns a dict assignment from explicitly known facts only.
    - Accepts false_list items either as 'x' or 'not x' (normalizes to x=False).
    - Accepts true_list items either as 'x' or 'not x' (normalizes to x=True/False accordingly).
    """
    assign: Dict[str, bool] = {}
    for f in true_list:
        if f.startswith("not "):
            v = f[4:]
            val = False
        else:
            v = f
            val = True
        if v in assign and assign[v] != val:
            # Explicit contradiction in input; treat infeasible
            return {}
        assign[v] = val

    for f in false_list:
        if f.startswith("not "):
            v = f[4:]
        else:
            v = f
        val = False
        if v in assign and assign[v] != val:
            return {}
        assign[v] = val

    return assign


def _is_k_sufficient_horn(
    clauses,
    base_closure: Dict[str, bool],
    goal: str,
    vars_set: Tuple[str, ...],
) -> bool:
    """
    Horn semantic sufficiency:
      For every *feasible* assignment to vars_set, goal is forced (Known).
    """
    vars_list = list(vars_set)
    feasible_seen = False

    for values in itertools.product([True, False], repeat=len(vars_list)):
        # Start from closure of base context (faster, monotone)
        ctx = dict(base_closure)

        # Add assignment literals; if conflict with ctx, infeasible
        conflict = False
        for v, val in zip(vars_list, values):
            if v in ctx and ctx[v] != val:
                conflict = True
                break
            ctx[v] = val
        if conflict:
            continue

        forced = forced_value_via_refutation(clauses, ctx, goal)

        if forced is None:
            # goal not forced under this feasible assignment => NOT sufficient
            return False
        
        if forced == INFEASIBLE:
            # goal itself leads to inconsistency either way; treat assignment infeasible
            continue

        feasible_seen = True

    return feasible_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default=os.path.join(DATA_DIR, "logic_q_mt_filtered.csv"), help="Input CSV")
    ap.add_argument("--out_csv", type=str, default=os.path.join(DATA_DIR, "logic_q_mt_filtered_with_alt.csv"), help="Output CSV with new column")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional cap for debugging")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    all_alt_col: List[str] = []
    all_valid_qs_forbid_alternatives: List[str] = []
    num_all_alt_gts: List[int] = []
    num_all_valid_qs: List[int] = []
    num_all_valid_qs_forbid_alternatives: List[int] = []

    for i, row in tqdm(df.iterrows()):
        k = int(row["k"])
        rules = _parse_cell(row["rules"])
        known_facts = _parse_cell(row["known_facts"])
        known_untrue_facts = _parse_cell(row["known_untrue_facts"])
        goal = row["goal"]
        candidate_vars = _parse_cell(row["all_valid_qs"])  # all vars in rules - goal - known_facts
        assert goal not in candidate_vars

        cannot_sets = _parse_cell(row["cannot_ask_facts_sets"])
        gt_qs = _parse_cell(row["gt_qs"])

        clauses = parse_clauses(rules)

        # Explicit context assignment (only from provided known facts)
        base_assign = _normalize_known(known_facts, known_untrue_facts)
        if base_assign == {} and (known_facts or known_untrue_facts):
            raise ValueError(f"Row {i}: explicit known facts contain a contradiction.")

        base_closure = solve_unit_prop(clauses, base_assign)
        if base_closure == CONTRADICTION:
            raise ValueError(f"Row {i}: base context is infeasible under rules.")

        # Enumerate all size-k sets and collect those that are k-sufficient
        alt_sets: List[List[str]] = []
        alt_sets_norm: Set[Tuple[str, ...]] = set()

        if k >= 2:
            for combo in itertools.combinations(candidate_vars, k-1):
                if _is_k_sufficient_horn(clauses, base_closure, goal, combo):
                    raise ValueError("NOT MINIMALL SUFFICIENT")
        
        for combo in itertools.combinations(candidate_vars, k):
            if _is_k_sufficient_horn(clauses, base_closure, goal, combo):
                combo_sorted = tuple(sorted(combo))
                alt_sets_norm.add(combo_sorted)

        alt_sets = [list(t) for t in sorted(alt_sets_norm)]

        # Assertions: must contain all sets listed in gt_qs and cannot_ask_facts_sets
        def norm_set_list(x):
            return tuple(sorted(_strip_not(s) for s in x))

        missing = []
        for s in (gt_qs or []) + (cannot_sets or []):
            if len(s) != k:
                raise ValueError(
                    f"Row {i}: expected all gt/cannot sets to be size k={k}, "
                    f"but got set {s} of size {len(s)}."
                )
            if norm_set_list(s) not in alt_sets_norm:
                missing.append(s)

        if missing:
            raise AssertionError(
                f"Row {i}: computed alternative gts missing {len(missing)} required sets. "
                f"Examples: {missing[:5]}"
            )
        
        all_alt_col.append(json.dumps(alt_sets))
        all_valid_qs_forbid_alternatives.append(json.dumps(sorted(set(candidate_vars) - set(sum(alt_sets, [])))))  # NOTE: Won't contain ground truth vars as well
        num_all_alt_gts.append(len(alt_sets))
        num_all_valid_qs.append(len(candidate_vars))
        num_all_valid_qs_forbid_alternatives.append(len(set(candidate_vars) - set(sum(alt_sets, []))))
        
    df = df.copy()
    df["all_alternative_gt_qs"] = all_alt_col
    df["all_valid_qs_forbid_alternatives"] = all_valid_qs_forbid_alternatives
    df["num_all_alternative_gt_qs"] = num_all_alt_gts
    df["num_all_valid_qs"] = num_all_valid_qs
    df["num_all_valid_qs_forbid_alternatives"] = num_all_valid_qs_forbid_alternatives
    
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
