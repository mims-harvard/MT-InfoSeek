import pandas as pd
import json
import re
from itertools import combinations
from sympy import symbols, Eq, solve
import ast
import argparse
from tqdm import tqdm
import os
import random
from func_timeout import func_timeout, FunctionTimedOut
import hashlib
from collections import deque


# =========================
# CSP Solver with Known(.) predicate and y-candidate enumeration
# =========================

class GSM_CSPSolver:
    """
    A CSP solver for GSME equations.
    - Constraints C: self.equations
    - Assignment context A_k: represented by a set of known variable assignments (equalities x=v)
    - Known(y | A_k): y is uniquely determined among all solutions satisfying C and A_k
    """

    def __init__(self, equations_str_list, all_vars_dict):
        # STEP: build symbol table for variables with available numeric values
        self.var_map = {v: symbols(v) for v in all_vars_dict.keys()}
        self.equations = []
        self.gt_values = all_vars_dict
        self._y_values_cache = {}

        # STEP: parse equations into SymPy Eq constraints
        for eq_str in equations_str_list:
            if "=" not in eq_str:
                continue
            lhs_str, rhs_str = eq_str.split("=")
            lhs = self.var_map.get(lhs_str.strip())

            local_dict = {k: v for k, v in self.var_map.items()}
            try:
                rhs = eval(rhs_str.strip(), {}, local_dict)
                if lhs is not None:
                    self.equations.append(Eq(lhs, rhs))
            except Exception:
                pass

    def enumerate_y_values(self, known_vars_list, goal_var_name, max_vals=200):
        """
        STEP: enumerate distinct candidate values of y under (C ∧ A_k)
        Returns a set of values (float when possible) or None on failure.
        """
        if goal_var_name not in self.var_map:
            return None

        cache_key = (goal_var_name, tuple(sorted(set(known_vars_list))), max_vals)
        if cache_key in self._y_values_cache:
            cached = self._y_values_cache[cache_key]
            return None if cached is None else set(cached)

        goal_sym = self.var_map[goal_var_name]
        current_eqs = list(self.equations)

        # STEP: add assignment equalities for known vars (A_k)
        for v_name in known_vars_list:
            if v_name in self.var_map and v_name in self.gt_values:
                current_eqs.append(Eq(self.var_map[v_name], self.gt_values[v_name]))

        try:
            solutions = solve(current_eqs, list(self.var_map.values()), dict=True)
            if not solutions:
                self._y_values_cache[cache_key] = frozenset()
                return set()

            y_set = set()
            for sol in solutions:
                if goal_sym not in sol:
                    continue
                try:
                    yv = float(sol[goal_sym].evalf())
                    if abs(yv) == 0.0:
                        yv = 0.0
                    y_set.add(yv)
                except Exception:
                    y_set.add(str(sol[goal_sym]))

                if len(y_set) >= max_vals:
                    break

            self._y_values_cache[cache_key] = frozenset(y_set)
            return y_set
        except Exception:
            self._y_values_cache[cache_key] = None
            return None

    def check_known(self, known_vars_list, goal_var_name):
        """
        STEP: Known(y | A_k) is true iff y has exactly one possible value.
        Returns (is_known, unique_value_if_known).
        """
        y_vals = self.enumerate_y_values(known_vars_list, goal_var_name)
        if y_vals is None or len(y_vals) != 1:
            return False, None
        return True, next(iter(y_vals))


# =========================
# Dependency helpers
# =========================

def _extract_vars_from_eq(eq: Eq):
    return set(str(s) for s in eq.rhs.free_symbols)

def build_deps_from_solver(solver: GSM_CSPSolver):
    """
    STEP: build dependency map deps[lhs] = rhs_vars
    """
    deps = {}
    for eq in solver.equations:
        lhs = str(eq.lhs)
        deps[lhs] = _extract_vars_from_eq(eq)
    return deps

def ancestors_of(goal_var: str, deps: dict):
    """
    STEP: compute ancestors(goal) by traversing goal -> rhs_vars recursively
    """
    anc = set()
    stack = [goal_var]
    while stack:
        v = stack.pop()
        if v in anc:
            continue
        anc.add(v)
        for u in deps.get(v, set()):
            if u not in anc:
                stack.append(u)
    return anc

def build_forward_adjacency(deps: dict):
    """
    STEP: build forward edges rhs -> lhs
    """
    fwd = {}
    for lhs, rhs_vars in deps.items():
        for r in rhs_vars:
            fwd.setdefault(r, set()).add(lhs)
    return fwd

def leaf_to_goal_distance(leaf_var: str, goal_var: str, fwd: dict):
    """
    STEP: compute shortest path length leaf -> ... -> goal using forward edges rhs->lhs
    """
    if leaf_var == goal_var:
        return 0
    q = deque([(leaf_var, 0)])
    seen = {leaf_var}
    while q:
        node, d = q.popleft()
        for nxt in fwd.get(node, set()):
            if nxt in seen:
                continue
            if nxt == goal_var:
                return d + 1
            seen.add(nxt)
            q.append((nxt, d + 1))
    return float("inf")



# =========================
# Parsing
# =========================


def parse_pred_values(pred_values_str: str):
    """
    STEP: parse numeric ground-truth assignments from "Pred Values"
    """
    val_dict = {}
    if isinstance(pred_values_str, str):
        for line in pred_values_str.split("\n"):
            if "=" in line:
                k, v = line.split("=")
                try:
                    val_dict[k.strip()] = float(v.strip())
                except Exception:
                    pass
    return val_dict


def reconstruct_problem_text_from_csp(csp_text: str, given_vars, val_dict):
    """
    STEP: rewrite problem by masking heldout vars in Variables block
    - keep value for given vars
    - remove value for others
    """
    if not isinstance(csp_text, str) or not csp_text.strip():
        return ""

    given_set = set(given_vars)
    lines = csp_text.splitlines()
    out = []

    section = None  # "vars" | "eqs" | "goal" | None
    pat_val_desc = re.compile(r"^([A-Za-z0-9]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*\[(.+)\]\s*$")
    pat_desc = re.compile(r"^([A-Za-z0-9]+)\s*\[(.+)\]\s*$")

    for raw in lines:
        line = raw.rstrip("\n")

        header = line.strip()
        if header == "Variables:":
            section = "vars"
            out.append("Variables:")
            continue
        if header == "Equations:":
            section = "eqs"
            out.append("Equations:")
            continue
        if header == "Goal:":
            section = "goal"
            out.append("Goal:")
            continue

        if section != "vars":
            out.append(line)
            continue

        s = line.strip()
        if not s:
            out.append(line)
            continue

        m = pat_val_desc.match(s)
        if m:
            var = m.group(1).strip()
            desc = m.group(3).strip()
            if var in given_set and var in val_dict:
                v = val_dict[var]
                try:
                    v_f = float(v)
                    v_str = str(int(v_f)) if v_f.is_integer() else str(v_f)
                except Exception:
                    v_str = str(v)
                out.append(f"{var} = {v_str} [{desc}]")
            else:
                out.append(f"{var} [{desc}]")
            continue

        m2 = pat_desc.match(s)
        if m2:
            var = m2.group(1).strip()
            desc = m2.group(2).strip()
            if var in given_set and var in val_dict:
                v = val_dict[var]
                try:
                    v_f = float(v)
                    v_str = str(int(v_f)) if v_f.is_integer() else str(v_f)
                except Exception:
                    v_str = str(v)
                out.append(f"{var} = {v_str} [{desc}]")
            else:
                out.append(f"{var} [{desc}]")
            continue

        out.append(line)

    return "\n".join(out)


# =========================
# Sampling helpers
# =========================

def _nCk_over_N(n, k, N):
    """
    Return True if nCk > N without computing huge integers.
    """
    if k < 0 or k > n:
        return False
    k = min(k, n - k)
    c = 1
    for i in range(1, k + 1):
        c = c * (n - k + i) // i
        if c > N:
            return True
    return False

def sample_random_subsets(items, k, N, rng):
    """
    STEP: sample up to N unique k-subsets from items.
    If total combinations <= N, enumerate all combinations.
    """
    items = list(items)
    n = len(items)
    if k < 0 or k > n:
        return []

    if not _nCk_over_N(n, k, N):
        return [tuple(c) for c in combinations(items, k)]

    seen = set()
    out = []
    max_tries = max(1000, 20 * N)
    tries = 0
    while len(out) < N and tries < max_tries:
        tries += 1
        comb = tuple(sorted(rng.sample(items, k)))
        if comb in seen:
            continue
        seen.add(comb)
        out.append(comb)
    return out


# =========================
# k-sufficient checks + difficulty scoring
# =========================

def is_k_sufficient_minimal(
    solver: GSM_CSPSolver,
    relevant_leaf,
    goal_var,
    heldout_tuple,
):
    """
    STEP: check Definition-style conditions (for this script):
    (2) Underspecification: Known(y | A_k) is False
    (3) Sufficiency: Known(y | A_k ∧ Known(S_k)) is True
    (4) Global minimality: for all S' subset S_k with |S'| < k, Known(y | A_k ∧ Known(S')) is False
    """
    heldout_set = set(heldout_tuple)
    given_vars = [v for v in relevant_leaf if v not in heldout_set]

    # (2)
    is_known_missing_k, _ = solver.check_known(given_vars, goal_var)
    if is_known_missing_k:
        return False

    # (3)
    all_known = given_vars + list(heldout_tuple)
    is_known_full, _ = solver.check_known(all_known, goal_var)
    if not is_known_full:
        return False

    # (4)
    k = len(heldout_tuple)
    for r in range(0, k):
        for sub in combinations(heldout_tuple, r):
            sub_known = given_vars + list(sub)
            is_known_sub, _ = solver.check_known(sub_known, goal_var)
            if is_known_sub:
                return False

    return True

def difficulty_score_distance(heldout_tuple, leaf2goal_dist):
    """
    STEP: difficulty by distance (larger is harder)
    Score = max leaf-to-goal distance among vars in S_k
    """
    dists = []
    for v in heldout_tuple:
        d = leaf2goal_dist.get(v, float("inf"))
        dists.append(d)
    score = max(dists) if dists else 0
    return score, {"dist_max": score, "dist_mean": sum(dists) / len(dists) if dists else 0.0}

def difficulty_score_ycands(
    solver: GSM_CSPSolver,
    leaf_nodes_all,
    goal_var,
    heldout_tuple,
):
    """
    STEP: difficulty by number of y candidates under A_k (larger is harder)
    """
    heldout_set = set(heldout_tuple)
    given_vars = [v for v in leaf_nodes_all if v not in heldout_set]
    y_vals = solver.enumerate_y_values(given_vars, goal_var)
    if y_vals is None:
        return -1, {"y_cand_count": -1}
    return len(y_vals), {"y_cand_count": len(y_vals)}


# =========================
# Main generation
# =========================

def process_unique_problem(
    row,
    k_list,
    N_candidates,
    M_keep,
    difficulty_mode,
    seed_base,
    timeout_sec,
):
    """
    Returns list of new rows (samples) generated from this unique Full Problem.
    """
    # STEP: compute stable origin id from Full Problem
    full_prob = row["Full Problem"]
    origin_id = hashlib.md5(str(full_prob).encode("utf-8")).hexdigest()[:12]

    # STEP: parse equations and ground-truth assignments
    eq_json = json.loads(row["Equations"])
    eq_list = list(eq_json.keys())
    val_dict = parse_pred_values(row["Pred Values"])
    if not val_dict:
        return []

    # STEP: parse goal var y
    goal_match = re.search(r"What is ([A-Za-z0-9]+)\?", str(full_prob))
    if not goal_match:
        return []
    goal_var = goal_match.group(1)

    # STEP: identify LHS vars and leaf vars
    lhs_vars = set([eq.split("=")[0].strip() for eq in eq_list])
    all_value_vars = set(val_dict.keys())
    leaf_nodes_all = sorted(list(all_value_vars - lhs_vars))

    # STEP: build solver and dependency graph
    solver = GSM_CSPSolver(eq_list, val_dict)
    deps = build_deps_from_solver(solver)
    anc = ancestors_of(goal_var, deps)

    # STEP (1 requested): relevant leaf = leaf ∩ ancestors(goal)
    relevant_leaf = sorted(list(set(leaf_nodes_all) & set(anc)))

    # STEP: precompute leaf-to-goal distances if needed
    leaf2goal_dist = {}
    if difficulty_mode == "distance":
        fwd = build_forward_adjacency(deps)
        for lv in relevant_leaf:
            leaf2goal_dist[lv] = leaf_to_goal_distance(lv, goal_var, fwd)

    out_rows = []

    # STEP: iterate k = 1..4
    for k in k_list:
        # STEP: skip if not enough relevant leaf to sample S_k
        if k > len(relevant_leaf):
            continue


        # STEP: sample N candidate S_k from relevant_leaf only
        cand_rng = random.Random(int(hashlib.md5(f"{origin_id}_k{k}_{seed_base}".encode("utf-8")).hexdigest()[:8], 16))
        candidates = sample_random_subsets(relevant_leaf, k, N_candidates, cand_rng)

        valid = []
        for heldout in candidates:
            # STEP: check k-sufficient minimality (Definition conditions)
            ok = is_k_sufficient_minimal(solver, relevant_leaf, goal_var, heldout)
            if not ok:
                continue

            # STEP (2 requested): compute difficulty score
            if difficulty_mode == "distance":
                score, meta = difficulty_score_distance(heldout, leaf2goal_dist)
            else:
                score, meta = difficulty_score_ycands(solver, leaf_nodes_all, goal_var, heldout)

            valid.append((heldout, score, meta))

        if not valid:
            continue

        # STEP (2 requested): sort by difficulty (descending), tie-break by heldout tuple
        valid.sort(key=lambda x: (x[1], tuple(x[0])), reverse=True)

        # STEP: keep hardest M
        kept = valid[: M_keep]

        for rank_i, (heldout, score, meta) in enumerate(kept):
            heldout_set = set(heldout)

            # STEP: A_k is represented by revealing all leaf vars except S_k
            # reveal only relevant leaves except heldout
            given_vars = [v for v in relevant_leaf if v not in heldout_set]

            # STEP: rewrite problem by masking S_k (remove values of heldout vars)
            rewritten_text = reconstruct_problem_text_from_csp(row.get("CSP", ""), given_vars, val_dict)


            # Possible Questions: use ALL leaf nodes
            possible_vars = list(leaf_nodes_all)

            # STEP: create unique_id = origin_id + new_id
            unique_id = f"{origin_id}_k{k}_{rank_i}"
            sample_id = unique_id

            new_row = {}
            
            new_row["problem_id"] = origin_id
            new_row["sample_id"] = sample_id
            
            new_row["Full Problem"] = row["Full Problem"]
            new_row["CSP"] = row["CSP"]
            new_row["Full Answer"] = row["Full Answer"]
            new_row["Variables"] = row["Variables"]
            new_row["Equations"] = row["Equations"]
            new_row["depth"] = row["depth"]
            new_row["Pred Values"] = row["Pred Values"]
            
            new_row["Heldout Value"] = json.dumps(list(heldout))
            new_row["Rewritten Problem"] = rewritten_text
            new_row["Possible Questions"] = json.dumps(possible_vars)
            new_row["Given_Conditions"] = json.dumps(given_vars)
            
            new_row["k"] = int(k)
            new_row["diff_score"] = float(score) if isinstance(score, (int, float)) else str(score)
            new_row["goal_var"] = goal_var
            new_row["leaf_nodes_all"] = json.dumps(leaf_nodes_all)
            new_row["relevant_leaf"] = json.dumps(relevant_leaf)
            new_row["ancestors_goal"] = json.dumps(sorted(list(anc)))

            # Difficulty meta
            for mk, mv in meta.items():
                new_row[mk] = mv

            out_rows.append(new_row)

    return out_rows


def generate_dataset(
    df,
    k_max,
    N_candidates,
    M_keep,
    difficulty_mode,
    seed,
    timeout_sec,
):
    """
    STEP: group by Full Problem and generate new dataset rows for each unique instance
    """
    if "Full Problem" not in df.columns:
        raise KeyError("Missing column: Full Problem")

    grouped = df.groupby("Full Problem", sort=False, dropna=False)
    print(f"Found {grouped.ngroups} unique Full Problem. Generate k=1..{k_max}, N={N_candidates}, keep M={M_keep}.")

    all_new_rows = []
    k_list = list(range(1, k_max + 1))

    for full_prob, g in tqdm(grouped, total=grouped.ngroups, desc="Unique Full Problem"):
        row = g.iloc[0].copy()
        origin_seed = int(hashlib.md5((str(full_prob) + f"_{seed}").encode("utf-8")).hexdigest()[:8], 16)

        try:
            rows = func_timeout(
                timeout_sec,
                process_unique_problem,
                args=(
                    row,
                    k_list,
                    N_candidates,
                    M_keep,
                    difficulty_mode,
                    origin_seed,
                    timeout_sec,
                ),
            )
        except FunctionTimedOut:
            continue
        except Exception:
            continue

        all_new_rows.extend(rows)

    new_df = pd.DataFrame(all_new_rows)
    if "sample_id" in new_df.columns and len(new_df) > 0:
        new_df.set_index("sample_id", inplace=True)
    print(f"Generation complete. Generated {len(new_df)} samples from {grouped.ngroups} unique problems.")
    return new_df


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Generate k-sufficient GSME dataset with difficulty ranking")

    # Inputs / outputs
    parser.add_argument("--data_name", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output_dir", type=str, default="./GSME", help="Output directory")

    # Core controls
    parser.add_argument("--k_max", type=int, default=4, help="Max k to generate (iterate k=1..k_max)")
    parser.add_argument("--N_candidates", type=int, default=200, help="Random candidates S_k per (problem, k)")
    parser.add_argument("--M_keep", type=int, default=5, help="Keep hardest M valid S_k per (problem, k)")

    # Difficulty ranking mode
    parser.add_argument(
        "--difficulty_mode",
        type=str,
        default="distance",
        choices=["distance", "ycands"],
        help="distance: rank by leaf-to-goal distance; ycands: rank by number of y candidates",
    )

    # Runtime controls
    parser.add_argument("--timeout_sec", type=int, default=20, help="Timeout per unique problem")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.data_name):
        raise FileNotFoundError(f"File not found: {args.data_name}")

    df_origin = pd.read_csv(args.data_name)

    new_df = generate_dataset(
        df_origin,
        k_max=args.k_max,
        N_candidates=args.N_candidates,
        M_keep=args.M_keep,
        difficulty_mode=args.difficulty_mode,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
    )

    base_name = os.path.basename(args.data_name).rsplit(".", 1)[0]
    filename = (
        f"{base_name}_ksufficient"
        f"_kmax{args.k_max}"
        f"_N{args.N_candidates}"
        f"_M{args.M_keep}.csv"
    )
    save_path = os.path.join(args.output_dir, filename)
    new_df.to_csv(save_path, index=True)
    print(f"Saved {len(new_df)} samples to: {save_path}")


if __name__ == "__main__":
    main()
