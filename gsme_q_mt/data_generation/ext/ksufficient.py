from __future__ import annotations

import random
from collections import deque
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:  # pragma: no cover
    import sympy as sp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sp = None


class SymbolicGoalSolver:
    def __init__(self, equations: Sequence[str], values: Mapping[str, Any]) -> None:
        self.values = values
        self.var_map = {name: sp.Symbol(name, integer=True) for name in values} if sp is not None else {}
        self.equations = []
        self._goal_value_cache: Dict[Tuple[str, Tuple[str, ...]], Optional[Set[Any]]] = {}
        if sp is None:
            return
        for expression in equations:
            if "=" not in expression:
                continue
            lhs_text, rhs_text = expression.split("=", 1)
            lhs_name = lhs_text.strip()
            lhs = self.var_map.get(lhs_name)
            if lhs is None:
                continue
            rhs = sp.sympify(rhs_text.strip(), locals=self.var_map)
            self.equations.append(sp.Eq(lhs, rhs))

    def enumerate_goal_values(self, known_vars: Sequence[str], goal_var: str, max_vals: int = 200) -> Optional[Set[Any]]:
        if sp is None or goal_var not in self.var_map:
            return None
        cache_key = (goal_var, tuple(sorted(set(known_vars))))
        if cache_key in self._goal_value_cache:
            return self._goal_value_cache[cache_key]
        goal_sym = self.var_map[goal_var]
        eqs = list(self.equations)
        for var in known_vars:
            if var in self.var_map and var in self.values:
                eqs.append(sp.Eq(self.var_map[var], self.values[var]))
        try:
            solutions = sp.solve(eqs, list(self.var_map.values()), dict=True)
        except Exception:
            self._goal_value_cache[cache_key] = None
            return None
        if not solutions:
            self._goal_value_cache[cache_key] = set()
            return set()
        out: Set[Any] = set()
        for solution in solutions:
            if goal_sym not in solution:
                self._goal_value_cache[cache_key] = None
                return None
            raw_value = sp.simplify(solution[goal_sym])
            if getattr(raw_value, "free_symbols", set()):
                self._goal_value_cache[cache_key] = None
                return None
            try:
                if raw_value.is_real is False:
                    out.add(str(raw_value))
                else:
                    numeric = float(raw_value.evalf())
                    out.add(int(numeric) if numeric.is_integer() else numeric)
            except Exception:
                out.add(str(raw_value))
            if len(out) >= max_vals:
                break
        self._goal_value_cache[cache_key] = out
        return out

    def check_known(self, known_vars: Sequence[str], goal_var: str) -> Tuple[bool, Optional[Any]]:
        goal_values = self.enumerate_goal_values(known_vars, goal_var)
        if goal_values is None or len(goal_values) != 1:
            return False, None
        return True, next(iter(goal_values))


def build_forward_adjacency(edges: Sequence[Mapping[str, str]]) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source"], set()).add(edge["target"])
    return adjacency


def shortest_distance(source: str, target: str, adjacency: Mapping[str, Set[str]]) -> Optional[int]:
    if source == target:
        return 0
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, dist = queue.popleft()
        for nxt in adjacency.get(node, set()):
            if nxt in seen:
                continue
            if nxt == target:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def sample_random_subsets(items: Sequence[str], k: int, n_candidates: int, rng: random.Random) -> List[Tuple[str, ...]]:
    items = list(items)
    if k < 0 or k > len(items):
        return []
    all_combinations = list(combinations(items, k))
    if len(all_combinations) <= n_candidates:
        return [tuple(sorted(item)) for item in all_combinations]
    seen = set()
    out: List[Tuple[str, ...]] = []
    tries = 0
    max_tries = max(1000, n_candidates * 20)
    while len(out) < n_candidates and tries < max_tries:
        tries += 1
        subset = tuple(sorted(rng.sample(items, k)))
        if subset in seen:
            continue
        seen.add(subset)
        out.append(subset)
    return out


def is_k_sufficient_minimal(
    solver: SymbolicGoalSolver,
    relevant_leaf: Sequence[str],
    goal_var: str,
    heldout: Sequence[str],
) -> bool:
    heldout_set = set(heldout)
    given = [var for var in relevant_leaf if var not in heldout_set]
    known_without_heldout, _ = solver.check_known(given, goal_var)
    if known_without_heldout:
        return False
    known_with_heldout, _ = solver.check_known(list(given) + list(heldout), goal_var)
    if not known_with_heldout:
        return False
    for size in range(len(heldout)):
        for sub in combinations(heldout, size):
            known_with_subset, _ = solver.check_known(list(given) + list(sub), goal_var)
            if known_with_subset:
                return False
    return True


def choose_k_sufficient_holdout(
    equations: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, str]],
    values: Mapping[str, Any],
    relevant_leaf: Sequence[str],
    goal_var: str,
    k: int,
    rng: random.Random,
    n_candidates: int = 200,
    difficulty_mode: str = "distance",
    top_m: int = 5,
) -> Tuple[List[str], Dict[str, Any]]:
    if k <= 0 or k > len(relevant_leaf):
        return [], {"dist_max": 0, "dist_mean": 0.0, "diff_score": 0.0, "holdout_rank": 0}

    solver = SymbolicGoalSolver([item["expression"] for item in equations], values)
    adjacency = build_forward_adjacency(edges)
    leaf2goal = {leaf: shortest_distance(leaf, goal_var, adjacency) for leaf in relevant_leaf}

    candidates = sample_random_subsets(relevant_leaf, k, n_candidates=n_candidates, rng=rng)
    valid: List[Tuple[Tuple[str, ...], float, Dict[str, Any]]] = []
    for heldout in candidates:
        if not is_k_sufficient_minimal(solver, relevant_leaf, goal_var, heldout):
            continue
        meta = difficulty_meta(
            heldout=heldout,
            goal_var=goal_var,
            leaf2goal=leaf2goal,
            solver=solver,
            relevant_leaf=relevant_leaf,
            difficulty_mode=difficulty_mode,
        )
        valid.append((heldout, meta["diff_score"], meta))

    if not valid:
        fallback = tuple(sorted(rng.sample(list(relevant_leaf), k)))
        meta = difficulty_meta(
            heldout=fallback,
            goal_var=goal_var,
            leaf2goal=leaf2goal,
            solver=solver,
            relevant_leaf=relevant_leaf,
            difficulty_mode="distance",
        )
        meta["ksufficient_found"] = False
        return list(fallback), meta

    valid.sort(key=lambda item: (item[1], item[0]), reverse=True)
    kept = valid[: max(1, top_m)]
    chosen_heldout, _, chosen_meta = kept[0]
    chosen_meta["holdout_rank"] = 0
    chosen_meta["ksufficient_found"] = True
    return list(chosen_heldout), chosen_meta


def difficulty_meta(
    heldout: Sequence[str],
    goal_var: str,
    leaf2goal: Mapping[str, Optional[int]],
    solver: SymbolicGoalSolver,
    relevant_leaf: Sequence[str],
    difficulty_mode: str,
) -> Dict[str, Any]:
    dists = [leaf2goal.get(var) for var in heldout]
    clean_dists = [dist for dist in dists if dist is not None]
    dist_max = max(clean_dists) if clean_dists else 0
    dist_mean = round(sum(clean_dists) / len(clean_dists), 3) if clean_dists else 0.0
    given = [var for var in relevant_leaf if var not in set(heldout)]

    y_cand_count = None
    if difficulty_mode == "ycands":
        goal_values = solver.enumerate_goal_values(given, goal_var)
        if goal_values is not None:
            y_cand_count = len(goal_values)
        else:
            y_cand_count = -1

    diff_score = float(y_cand_count if y_cand_count is not None and y_cand_count >= 0 else dist_max)
    return {
        "dist_max": dist_max,
        "dist_mean": dist_mean,
        "y_cand_count": y_cand_count,
        "diff_score": diff_score,
    }
