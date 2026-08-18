"""Curate *globally minimal* k-sufficient (k<=4) multi-turn information-seeking tasks from Boolean GRNs.

boolean_network_utils.py is the single source of truth for:
  - basin-map caching (n<=21) and fixed point enumeration
  - exact (global) minimal k-sufficient subset search
  - recursion-based enumeration of completions via basin lookups (no simulation)

Regime A (GeneReg-SS):
  - Feasible set Ω is an extensional set of fixed-point attractors.
  - Context A is partial observation of a steady-state vector.
  - Target y:
      (A1) attractor ID among Ω_A, or
      (A2) marker-gene value in the attractor.

Regime B (GeneReg-Dyn):
  - Variable space is initial states {0,1}^n under synchronous update.
  - Uses *precomputed basin_map* (state -> attractor_id) for exact checks (n<=21).
  - Context A is partial initial-state assignment.
  - Target y: attractor_id, or marker-gene-at-convergence.

Output:
  out_dir/<--tasks_filename> (default tasks_new.jsonl) : one JSON object per group

"""

import argparse
import csv
import json
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# Local boolean-network utilities (same directory).
import boolean_network_utils as bnu


UINT64 = np.uint64
TASK_TYPES = ("ss_marker", "ss_id", "dyn_attr", "dyn_marker")


# -----------------------------------------------------------------------------
# Task group schema
# -----------------------------------------------------------------------------

@dataclass
class Branch:
    """One counterfactual branch/world inside a group."""
    y: int
    state: int  # int-encoded state (fixed point for SS, initial state for Dyn)


@dataclass
class TaskGroup:
    group_id: str
    family: str  # GeneReg-SS or GeneReg-Dyn

    model: str
    n_nodes: int
    var_names: List[str]

    target_type: str  # "marker_gene" | "attractor_id"
    marker_gene: Optional[str]
    marker_gene_idx: Optional[int]

    # Context A: revealed variable assignments
    observed: List[Tuple[str, int]]  # [(gene_name, value), ...]

    # Certified minimality info (complete - tasks with >50 sets are rejected)
    k_min: int
    minimal_sets: List[List[str]]      # ALL k-sufficient gene-name lists
    minimal_sets_idx: List[List[int]]  # parallel indices

    # Internal helper fields kept for construction / recursion.
    n_candidate_worlds: Optional[int]  # SS: |Ω_A|, Dyn: |subcube| (often 2^d)

    # For SS: which Ω indices each candidate corresponds to
    candidate_ids: Optional[List[int]]

    branches: List[Branch]
    metadata: Dict

    # Prompt-side catalog semantics
    prompt_catalog_size: Optional[int] = None
    prompt_catalog_is_capped: Optional[bool] = None


# -----------------------------------------------------------------------------
# Regime A sampling: GeneReg-SS
# -----------------------------------------------------------------------------

def make_group_ss_marker(
    cache: bnu.ModelCache,
    *,
    max_k: int,
    rng: np.random.Generator,
    min_candidates: int = 16,
    max_candidates: int = 256,
    marker_single_gene_acc_cap: Optional[float] = None,
    max_sets_to_return: int = 50,
) -> Optional[TaskGroup]:
    """Regime A, target is a marker gene value (marker gene itself is NOT queryable)."""

    if len(cache.fixed_points) < min_candidates:
        return None

    n = cache.n_nodes
    # Skip models with n > 64 (state integers don't fit in uint64)
    if n > 64:
        return None
    
    omega = np.array(cache.fixed_points, dtype=UINT64)

    # Choose marker gene with non-trivial variation over Ω (avoid constant genes)
    gene_entropy = []
    for gi in range(n):
        bits = ((omega >> UINT64(gi)) & UINT64(1)).astype(np.int8)
        p = float(bits.mean())
        if p in (0.0, 1.0):
            continue
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        # How well a single other gene can predict this marker on Ω.
        # High values make tasks collapse to low-k; cap this for hard cells.
        best_single_gene_acc = 0.5
        y_marker = bits
        for gj in range(n):
            if gj == gi:
                continue
            x = ((omega >> UINT64(gj)) & UINT64(1)).astype(np.int8)
            x0 = (x == 0)
            x1 = ~x0
            n00 = int(np.sum((y_marker == 0) & x0))
            n01 = int(np.sum((y_marker == 1) & x0))
            n10 = int(np.sum((y_marker == 0) & x1))
            n11 = int(np.sum((y_marker == 1) & x1))
            acc = float(max(n00, n01) + max(n10, n11)) / float(omega.shape[0])
            if acc > best_single_gene_acc:
                best_single_gene_acc = acc
        gene_entropy.append((h, gi, best_single_gene_acc))
    if not gene_entropy:
        return None

    if marker_single_gene_acc_cap is not None:
        filtered = [row for row in gene_entropy if float(row[2]) <= float(marker_single_gene_acc_cap)]
        if filtered:
            gene_entropy = filtered

    gene_entropy.sort(key=lambda x: x[0], reverse=True)
    # Strict top-25% bucket (ceil to avoid dropping candidates when len%4 != 0)
    top = max(1, int(np.ceil(0.25 * len(gene_entropy))))
    chosen = gene_entropy[int(rng.integers(0, top))]
    marker_idx = int(chosen[1])
    marker_best_single_gene_acc = float(chosen[2])

    # pick a true attractor from Ω
    true_idx = int(rng.integers(0, omega.shape[0]))
    true_state = int(omega[true_idx])

    sampled = bnu.sample_context_reduce_candidates_fixed_points(
        omega_states=omega,
        true_state=true_state,
        n=n,
        forbid_genes={marker_idx},
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        rng=rng,
    )
    if sampled is None:
        return None

    observed_idx_vals, cand_idx = sampled
    cand_states = omega[cand_idx]

    # Ensure underspecification for marker: both y values must appear
    y = ((cand_states >> UINT64(marker_idx)) & UINT64(1)).astype(np.int8)
    if int(y.min()) == int(y.max()):
        return None

    observed_genes = {gi for gi, _ in observed_idx_vals}
    candidate_genes = [gi for gi in range(n) if gi not in observed_genes and gi != marker_idx]

    res = bnu.find_min_k_sufficient_sets_states(
        states=cand_states,
        y=y,
        candidate_genes=candidate_genes,
        max_k=max_k,
        max_sets_to_return=max_sets_to_return,
    )
    if res is None or res.k_min == 0:
        return None

    min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]
    observed_named = [(cache.var_names[i], v) for i, v in observed_idx_vals]

    # Branch reps: pick one state per y value
    branches: List[Branch] = []
    for yv in [0, 1]:
        idxs = np.nonzero(y == yv)[0]
        if idxs.size == 0:
            continue
        rep = int(cand_states[int(rng.choice(idxs))])
        branches.append(Branch(y=int(yv), state=rep))
    if len(branches) < 2:
        return None

    group_id = (
        f"SS|{cache.model}|marker={cache.var_names[marker_idx]}|k={res.k_min}"
        f"|m={len(observed_named)}|c={int(cand_states.shape[0])}|seed={int(rng.integers(0, 1_000_000_000))}"
    )

    return TaskGroup(
        group_id=group_id,
        family="GeneReg-SS",
        model=cache.model,
        n_nodes=n,
        var_names=cache.var_names,
        target_type="marker_gene",
        marker_gene=cache.var_names[marker_idx],
        marker_gene_idx=marker_idx,
        observed=observed_named,
        k_min=res.k_min,
        minimal_sets=min_sets_names,
        minimal_sets_idx=res.min_sets,
        n_candidate_worlds=int(cand_states.shape[0]),
        candidate_ids=cand_idx.astype(int).tolist(),
        branches=branches,
        metadata={
            "fixed_points_method": cache.fixed_points_method,
            "fixed_points_is_capped": cache.fixed_points_is_capped,
            "n_tested_subsets": res.n_tested,
            "candidate_marker_frac": float(y.mean()),
            "marker_best_single_gene_acc": marker_best_single_gene_acc,
        },
        prompt_catalog_size=int(len(cache.fixed_points)),
        prompt_catalog_is_capped=bool(cache.fixed_points_is_capped),
    )


def make_group_ss_attractor_id(
    cache: bnu.ModelCache,
    *,
    max_k: int,
    rng: np.random.Generator,
    min_candidates: int = 2,
    max_candidates: int = 1 << 4,  # must be <=2^max_k
    max_sets_to_return: int = 50,
) -> Optional[TaskGroup]:
    """Regime A, target is attractor ID among Ω_A (so Ω_A must be small enough)."""

    if len(cache.fixed_points) < min_candidates:
        return None

    n = cache.n_nodes
    # Skip models with n > 64 (state integers don't fit in uint64)
    if n > 64:
        return None
    
    omega = np.array(cache.fixed_points, dtype=UINT64)

    true_idx = int(rng.integers(0, omega.shape[0]))
    true_state = int(omega[true_idx])

    sampled = bnu.sample_context_reduce_candidates_fixed_points(
        omega_states=omega,
        true_state=true_state,
        n=n,
        forbid_genes=set(),
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        rng=rng,
    )
    if sampled is None:
        return None

    observed_idx_vals, cand_idx = sampled
    cand_states = omega[cand_idx]
    if cand_states.shape[0] < 2:
        return None

    # y is attractor ID in Ω (index in omega)
    y = cand_idx.astype(np.int32)

    observed_genes = {gi for gi, _ in observed_idx_vals}
    candidate_genes = [gi for gi in range(n) if gi not in observed_genes]

    res = bnu.find_min_k_sufficient_sets_states(
        states=cand_states,
        y=y,
        candidate_genes=candidate_genes,
        max_k=max_k,
        max_sets_to_return=max_sets_to_return,
    )
    if res is None or res.k_min == 0:
        return None

    min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]
    observed_named = [(cache.var_names[i], v) for i, v in observed_idx_vals]

    branches = [Branch(y=int(idx), state=int(st)) for idx, st in zip(cand_idx.tolist(), cand_states.tolist())]

    group_id = (
        f"SS|{cache.model}|attr_id|k={res.k_min}|m={len(observed_named)}"
        f"|c={int(cand_states.shape[0])}|seed={int(rng.integers(0, 1_000_000_000))}"
    )

    return TaskGroup(
        group_id=group_id,
        family="GeneReg-SS",
        model=cache.model,
        n_nodes=n,
        var_names=cache.var_names,
        target_type="attractor_id",
        marker_gene=None,
        marker_gene_idx=None,
        observed=observed_named,
        k_min=res.k_min,
        minimal_sets=min_sets_names,
        minimal_sets_idx=res.min_sets,
        n_candidate_worlds=int(cand_states.shape[0]),
        candidate_ids=cand_idx.astype(int).tolist(),
        branches=branches,
        metadata={
            "fixed_points_method": cache.fixed_points_method,
            "fixed_points_is_capped": cache.fixed_points_is_capped,
            "n_tested_subsets": res.n_tested,
        },
        prompt_catalog_size=int(len(cache.fixed_points)),
        prompt_catalog_is_capped=bool(cache.fixed_points_is_capped),
    )


def _normalize_observed_idx_vals(
    observed_idx_vals: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """Normalize observed assignments and reject conflicting duplicate indices."""
    by_idx: Dict[int, int] = {}
    for gi, v in observed_idx_vals:
        gii = int(gi)
        vv = int(v)
        if gii in by_idx and by_idx[gii] != vv:
            return None
        by_idx[gii] = vv
    return sorted((gi, vv) for gi, vv in by_idx.items())


def _observed_named_to_idx_vals(
    observed: List[Tuple[str, int]],
    var_names: List[str],
) -> Optional[List[Tuple[int, int]]]:
    name_to_idx = {name: i for i, name in enumerate(var_names)}
    idx_vals: List[Tuple[int, int]] = []
    for name, v in observed:
        gi = name_to_idx.get(name)
        if gi is None:
            return None
        idx_vals.append((int(gi), int(v)))
    return _normalize_observed_idx_vals(idx_vals)


def _build_ss_attractor_id_group_from_observed(
    cache: bnu.ModelCache,
    *,
    observed_idx_vals: List[Tuple[int, int]],
    max_k: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    construction_mode: str = "direct",
) -> Optional[TaskGroup]:
    """Rebuild an SS attractor-id group from an explicit observed context."""
    n = cache.n_nodes
    if n > 64:
        return None
    normalized = _normalize_observed_idx_vals(observed_idx_vals)
    if normalized is None:
        return None

    omega = np.array(cache.fixed_points, dtype=UINT64)
    if omega.size < 2:
        return None

    mask, pat = bnu.build_mask_value_from_idx_vals(normalized)
    m = np.uint64(mask)
    p = np.uint64(pat)
    cand_idx = np.nonzero((omega & m) == p)[0].astype(np.int32)
    if cand_idx.size < 2:
        return None
    cand_states = omega[cand_idx]

    y = cand_idx.astype(np.int32)
    observed_genes = {gi for gi, _ in normalized}
    candidate_genes = [gi for gi in range(n) if gi not in observed_genes]

    res = bnu.find_min_k_sufficient_sets_states(
        states=cand_states,
        y=y,
        candidate_genes=candidate_genes,
        max_k=max_k,
        max_sets_to_return=max_sets_to_return,
    )
    if res is None or res.k_min == 0:
        return None

    observed_named = [(cache.var_names[i], v) for i, v in normalized]
    min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]
    branches = [Branch(y=int(idx), state=int(st)) for idx, st in zip(cand_idx.tolist(), cand_states.tolist())]

    group_id = (
        f"SS|{cache.model}|attr_id|k={res.k_min}|m={len(observed_named)}"
        f"|c={int(cand_states.shape[0])}|mode={construction_mode}|seed={int(rng.integers(0, 1_000_000_000))}"
    )

    return TaskGroup(
        group_id=group_id,
        family="GeneReg-SS",
        model=cache.model,
        n_nodes=n,
        var_names=cache.var_names,
        target_type="attractor_id",
        marker_gene=None,
        marker_gene_idx=None,
        observed=observed_named,
        k_min=res.k_min,
        minimal_sets=min_sets_names,
        minimal_sets_idx=res.min_sets,
        n_candidate_worlds=int(cand_states.shape[0]),
        candidate_ids=cand_idx.astype(int).tolist(),
        branches=branches,
        metadata={
            "fixed_points_method": cache.fixed_points_method,
            "fixed_points_is_capped": cache.fixed_points_is_capped,
            "n_tested_subsets": res.n_tested,
            "construction_mode": construction_mode,
        },
        prompt_catalog_size=int(len(cache.fixed_points)),
        prompt_catalog_is_capped=bool(cache.fixed_points_is_capped),
    )


def _build_ss_marker_group_from_observed(
    cache: bnu.ModelCache,
    *,
    marker_idx: int,
    observed_idx_vals: List[Tuple[int, int]],
    max_k: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    construction_mode: str = "direct",
) -> Optional[TaskGroup]:
    """Rebuild an SS marker task from an explicit observed context and marker gene."""
    n = cache.n_nodes
    if n > 64:
        return None
    if marker_idx < 0 or marker_idx >= n:
        return None

    normalized = _normalize_observed_idx_vals(observed_idx_vals)
    if normalized is None:
        return None
    observed_genes = {gi for gi, _ in normalized}
    if marker_idx in observed_genes:
        return None

    omega = np.array(cache.fixed_points, dtype=UINT64)
    if omega.size < 2:
        return None

    mask, pat = bnu.build_mask_value_from_idx_vals(normalized)
    m = np.uint64(mask)
    p = np.uint64(pat)
    cand_idx = np.nonzero((omega & m) == p)[0].astype(np.int32)
    if cand_idx.size < 2:
        return None
    cand_states = omega[cand_idx]

    y = ((cand_states >> UINT64(marker_idx)) & UINT64(1)).astype(np.int8)
    if int(y.min()) == int(y.max()):
        return None

    candidate_genes = [gi for gi in range(n) if gi not in observed_genes and gi != marker_idx]
    res = bnu.find_min_k_sufficient_sets_states(
        states=cand_states,
        y=y,
        candidate_genes=candidate_genes,
        max_k=max_k,
        max_sets_to_return=max_sets_to_return,
    )
    if res is None or res.k_min == 0:
        return None

    observed_named = [(cache.var_names[i], v) for i, v in normalized]
    min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]

    branches: List[Branch] = []
    for yv in [0, 1]:
        idxs = np.nonzero(y == yv)[0]
        if idxs.size == 0:
            continue
        rep = int(cand_states[int(rng.choice(idxs))])
        branches.append(Branch(y=int(yv), state=rep))
    if len(branches) < 2:
        return None

    group_id = (
        f"SS|{cache.model}|marker={cache.var_names[marker_idx]}|k={res.k_min}|m={len(observed_named)}"
        f"|c={int(cand_states.shape[0])}|mode={construction_mode}|seed={int(rng.integers(0, 1_000_000_000))}"
    )

    return TaskGroup(
        group_id=group_id,
        family="GeneReg-SS",
        model=cache.model,
        n_nodes=n,
        var_names=cache.var_names,
        target_type="marker_gene",
        marker_gene=cache.var_names[marker_idx],
        marker_gene_idx=marker_idx,
        observed=observed_named,
        k_min=res.k_min,
        minimal_sets=min_sets_names,
        minimal_sets_idx=res.min_sets,
        n_candidate_worlds=int(cand_states.shape[0]),
        candidate_ids=cand_idx.astype(int).tolist(),
        branches=branches,
        metadata={
            "fixed_points_method": cache.fixed_points_method,
            "fixed_points_is_capped": cache.fixed_points_is_capped,
            "n_tested_subsets": res.n_tested,
            "candidate_marker_frac": float(y.mean()),
            "construction_mode": construction_mode,
        },
        prompt_catalog_size=int(len(cache.fixed_points)),
        prompt_catalog_is_capped=bool(cache.fixed_points_is_capped),
    )


def _ss_id_candidate_range_for_target_k(target_k: int, max_k: int) -> Tuple[int, int]:
    """Target-k-aware candidate range for SS attractor-id tasks."""
    upper = 1 << int(max_k)
    if target_k <= 1:
        return 2, 2
    low = (1 << int(target_k - 1)) + 1
    high = min(upper, 1 << int(target_k))
    return max(2, low), max(2, high)


def make_group_ss_attractor_id_top_down(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct SS attractor-id examples by descending k via additional reveals.

    Start from a harder context (typically larger k), then reveal selected genes from
    one candidate attractor to recursively reduce k until target_k is reached.
    """
    if target_k < 1 or target_k > max_k:
        return None

    max_candidates = 1 << max_k
    start_min = min(max_candidates, max(2, (1 << target_k) + 1))
    if start_min > max_candidates:
        return None

    omega = np.array(cache.fixed_points, dtype=UINT64)
    if omega.size < 2:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_ss_attractor_id(
            cache,
            max_k=max_k,
            rng=rng,
            min_candidates=start_min,
            max_candidates=max_candidates,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.k_min < target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group
        if not seed_group.candidate_ids:
            continue

        anchor_idx = int(seed_group.candidate_ids[int(rng.integers(0, len(seed_group.candidate_ids)))])
        anchor_state = int(omega[anchor_idx])

        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min <= target_k:
                continue

            obs_set = {gi for gi, _ in cur_obs}
            preferred = []
            seen_pref = set()
            for subset in cur_group.minimal_sets_idx:
                for gi in subset:
                    gii = int(gi)
                    if gii in obs_set or gii in seen_pref:
                        continue
                    seen_pref.add(gii)
                    preferred.append(gii)
            if not preferred:
                preferred = [gi for gi in range(cache.n_nodes) if gi not in obs_set]
            if not preferred:
                continue

            order = [int(x) for x in rng.permutation(np.array(preferred, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = cur_obs + [(gi, int(bnu.bit(anchor_state, gi)))]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_ss_attractor_id_group_from_observed(
                    cache,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="top_down",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min > target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


def make_group_ss_marker_bottom_up(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    rng: np.random.Generator,
    seed_min_candidates: int = 16,
    seed_max_candidates: int = 256,
    marker_single_gene_acc_cap: Optional[float] = None,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct harder SS marker tasks by recursively removing context assignments."""
    if target_k < 1 or target_k > max_k:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_ss_marker(
            cache,
            max_k=max_k,
            rng=rng,
            min_candidates=seed_min_candidates,
            max_candidates=seed_max_candidates,
            marker_single_gene_acc_cap=marker_single_gene_acc_cap,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.marker_gene_idx is None:
            continue
        if seed_group.k_min > target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group

        marker_idx = int(seed_group.marker_gene_idx)
        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min > target_k:
                continue

            removable = [gi for gi, _ in cur_obs if int(gi) != marker_idx]
            if not removable:
                continue

            order = [int(x) for x in rng.permutation(np.array(removable, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = [(g, v) for g, v in cur_obs if int(g) != gi]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_ss_marker_group_from_observed(
                    cache,
                    marker_idx=marker_idx,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="bottom_up",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min < target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


def make_group_ss_marker_top_down(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    rng: np.random.Generator,
    seed_min_candidates: int = 16,
    seed_max_candidates: int = 256,
    marker_single_gene_acc_cap: Optional[float] = None,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct SS marker examples by descending k via additional reveals."""
    if target_k < 1 or target_k > max_k:
        return None

    omega = np.array(cache.fixed_points, dtype=UINT64)
    if omega.size < 2:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_ss_marker(
            cache,
            max_k=max_k,
            rng=rng,
            min_candidates=seed_min_candidates,
            max_candidates=seed_max_candidates,
            marker_single_gene_acc_cap=marker_single_gene_acc_cap,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.marker_gene_idx is None:
            continue
        if seed_group.k_min < target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group
        if not seed_group.candidate_ids:
            continue

        marker_idx = int(seed_group.marker_gene_idx)
        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        anchor_idx = int(seed_group.candidate_ids[int(rng.integers(0, len(seed_group.candidate_ids)))])
        anchor_state = int(omega[anchor_idx])

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min <= target_k:
                continue

            obs_set = {gi for gi, _ in cur_obs}
            preferred = []
            seen_pref = set()
            for subset in cur_group.minimal_sets_idx:
                for gi in subset:
                    gii = int(gi)
                    if gii == marker_idx or gii in obs_set or gii in seen_pref:
                        continue
                    seen_pref.add(gii)
                    preferred.append(gii)
            if not preferred:
                preferred = [gi for gi in range(cache.n_nodes) if gi not in obs_set and gi != marker_idx]
            if not preferred:
                continue

            order = [int(x) for x in rng.permutation(np.array(preferred, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = cur_obs + [(gi, int(bnu.bit(anchor_state, gi)))]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_ss_marker_group_from_observed(
                    cache,
                    marker_idx=marker_idx,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="top_down",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min > target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


def make_group_ss_attractor_id_bottom_up(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct SS attractor-id examples by recursively removing observations."""
    if target_k < 1 or target_k > max_k:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_ss_attractor_id(
            cache,
            max_k=max_k,
            rng=rng,
            min_candidates=2,
            max_candidates=1 << max_k,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.k_min > target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group

        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min > target_k:
                continue

            removable = [gi for gi, _ in cur_obs]
            if not removable:
                continue

            order = [int(x) for x in rng.permutation(np.array(removable, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = [(g, v) for g, v in cur_obs if int(g) != gi]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_ss_attractor_id_group_from_observed(
                    cache,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="bottom_up",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min < target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


# -----------------------------------------------------------------------------
# Regime B sampling: GeneReg-Dyn
# -----------------------------------------------------------------------------

def make_group_dyn_attractor_id(
    cache: bnu.ModelCache,
    *,
    desired_free_bits_max: int,
    max_k: int,
    rng: np.random.Generator,
    max_tries: int = 2000,
    max_sets_to_return: int = 50,
) -> Optional[TaskGroup]:
    """Regime B, target is attractor id reached from a partial initial state."""

    if cache.landscape is None or cache.landscape.basin_map is None:
        return None

    n = cache.n_nodes
    basin = cache.landscape.basin_map
    n_states = basin.shape[0]
    if n_states != (1 << n):
        return None

    for _ in range(max_tries):
        s1 = int(rng.integers(0, n_states))
        y1 = int(basin[s1])

        d = int(rng.integers(1, min(desired_free_bits_max, n) + 1))
        free_bits = rng.choice(np.arange(n), size=d, replace=False).tolist()

        # Build a bitmask whose 1-bits mark free (unobserved) genes.
        flip_mask = 0
        for b in free_bits:
            flip_mask |= 1 << int(b)

        # Fast underspecification pre-check: compare opposite corners of this subcube.
        # XOR with flip_mask flips exactly the free bits.
        s2 = s1 ^ flip_mask
        y2 = int(basin[s2])
        if y2 == y1:
            continue  # not counterfactual

        # Context A fixes all bits except free_bits:
        # - known_mask has 1s at observed positions
        # - known_value stores observed assignments from s1
        known_mask = ((1 << n) - 1) ^ flip_mask
        known_value = s1 & known_mask

        # Exact minimal k-sufficient search using basin lookups + recursion
        res = bnu.find_min_k_sufficient_sets_basin(
            n=n,
            basin_map=basin,
            context_mask=known_mask,
            context_value=known_value,
            candidate_genes=free_bits,
            target_type="attractor_id",
            max_k=max_k,
            max_sets_to_return=max_sets_to_return,
        )
        if res is None or res.k_min == 0:
            continue

        # Observed list (all fixed bits)
        observed = []
        for gi in range(n):
            if gi in free_bits:
                continue
            observed.append((cache.var_names[gi], bnu.bit(known_value, gi)))

        # Enumerate the subcube to build branches (one representative per distinct y)
        y_to_states: Dict[int, List[int]] = {}
        for st in bnu.iter_states_consistent_with_mask_value(n, known_mask, known_value):
            yv = int(basin[st])
            y_to_states.setdefault(yv, []).append(int(st))

        if len(y_to_states) < 2:
            continue

        branches: List[Branch] = []
        for yv, sts in y_to_states.items():
            rep_state = int(sts[int(rng.integers(0, len(sts)))])
            branches.append(Branch(y=int(yv), state=rep_state))

        min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]

        group_id = (
            f"DYN|{cache.model}|attr_id|k={res.k_min}|free={len(free_bits)}"
            f"|seed={int(rng.integers(0, 1_000_000_000))}"
        )

        return TaskGroup(
            group_id=group_id,
            family="GeneReg-Dyn",
            model=cache.model,
            n_nodes=n,
            var_names=cache.var_names,
            target_type="attractor_id",
            marker_gene=None,
            marker_gene_idx=None,
            observed=observed,
            k_min=res.k_min,
            minimal_sets=min_sets_names,
            minimal_sets_idx=res.min_sets,
            n_candidate_worlds=int(2 ** len(free_bits)),
            candidate_ids=None,
            branches=branches,
            metadata={
                "free_bits_idx": free_bits,
                "n_tested_subsets": res.n_tested,
                "n_distinct_outcomes": int(len(y_to_states)),
            },
            prompt_catalog_size=int(len(cache.landscape.attractors)),
            prompt_catalog_is_capped=None,
        )

    return None


def make_group_dyn_marker_gene(
    cache: bnu.ModelCache,
    *,
    desired_free_bits_max: int,
    max_k: int,
    rng: np.random.Generator,
    max_tries: int = 2000,
    max_sets_to_return: int = 50,
) -> Optional[TaskGroup]:
    """Regime B, target is marker-gene value *at convergence* (in the reached attractor).

    - Context A: partial initial-state assignment.
    - Queries: ask for additional initial-state gene values.
    - Label y: marker gene value in the converged attractor.

    Requires a precomputed basin_map (n<=dyn_max_nodes in cache generation).
    """

    if cache.landscape is None or cache.landscape.basin_map is None:
        return None

    n = cache.n_nodes
    basin = cache.landscape.basin_map
    n_states = basin.shape[0]
    if n_states != (1 << n):
        return None

    attractors = cache.landscape.attractors
    if not attractors:
        return None

    # Representative state per attractor id (safe: n<=21 for basin maps)
    rep_states = bnu.attractor_representative_states(cache.landscape)
    if rep_states.size == 0:
        return None

    # Choose a marker gene that:
    #   (i) is constant within each attractor (handles cycles safely), and
    #   (ii) varies across attractors (non-trivial y).
    gene_entropy: List[Tuple[float, int]] = []
    for gi in range(n):
        per_attr_vals: List[int] = []
        ok = True
        for a in attractors:
            vals = [bnu.bit(st, gi) for st in a.states]
            if min(vals) != max(vals):
                ok = False
                break
            per_attr_vals.append(int(vals[0]))
        if not ok:
            continue
        if min(per_attr_vals) == max(per_attr_vals):
            continue
        p = float(sum(per_attr_vals)) / float(len(per_attr_vals))
        # entropy (avoid 0/1 already filtered)
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        gene_entropy.append((h, gi))

    if not gene_entropy:
        return None

    gene_entropy.sort(reverse=True)
    # Strict top-25% bucket (ceil to avoid dropping candidates when len%4 != 0)
    top = max(1, int(np.ceil(0.25 * len(gene_entropy))))
    marker_idx = int(gene_entropy[int(rng.integers(0, top))][1])

    # Map attractor_id -> marker value at convergence
    # (cycle-safe because marker is constant within each attractor by construction)
    marker_by_attr = ((rep_states >> np.uint64(marker_idx)) & np.uint64(1)).astype(np.int8)

    for _ in range(max_tries):
        s1 = int(rng.integers(0, n_states))
        y1 = int(marker_by_attr[int(basin[s1])])

        d = int(rng.integers(1, min(desired_free_bits_max, n) + 1))
        free_bits = rng.choice(np.arange(n), size=d, replace=False).tolist()

        # Build a bitmask whose 1-bits mark free (unobserved) genes.
        flip_mask = 0
        for b in free_bits:
            flip_mask |= 1 << int(b)

        # Fast underspecification pre-check: compare opposite corners of this subcube.
        # XOR with flip_mask flips exactly the free bits.
        s2 = s1 ^ flip_mask
        y2 = int(marker_by_attr[int(basin[s2])])
        if y2 == y1:
            continue  # not counterfactual in marker label

        # Context A fixes all bits except free_bits:
        # - known_mask has 1s at observed positions
        # - known_value stores observed assignments from s1
        known_mask = ((1 << n) - 1) ^ flip_mask
        known_value = s1 & known_mask

        res = bnu.find_min_k_sufficient_sets_basin(
            n=n,
            basin_map=basin,
            attractor_rep_states=rep_states,
            marker_idx=marker_idx,
            context_mask=known_mask,
            context_value=known_value,
            candidate_genes=free_bits,
            target_type="marker_gene",
            max_k=max_k,
            max_sets_to_return=max_sets_to_return,
        )
        if res is None or res.k_min == 0:
            continue

        # Observed list (all fixed bits)
        observed = []
        for gi in range(n):
            if gi in free_bits:
                continue
            observed.append((cache.var_names[gi], bnu.bit(known_value, gi)))

        # Enumerate the subcube to build branches (one representative per y value)
        y_to_states: Dict[int, List[int]] = {}
        for st in bnu.iter_states_consistent_with_mask_value(n, known_mask, known_value):
            yv = int(marker_by_attr[int(basin[st])])
            y_to_states.setdefault(yv, []).append(int(st))

        if len(y_to_states) < 2:
            continue

        branches: List[Branch] = []
        for yv, sts in y_to_states.items():
            rep_state = int(sts[int(rng.integers(0, len(sts)))])
            branches.append(Branch(y=int(yv), state=rep_state))

        min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]

        group_id = (
            f"DYN|{cache.model}|marker={cache.var_names[marker_idx]}|k={res.k_min}|free={len(free_bits)}"
            f"|seed={int(rng.integers(0, 1_000_000_000))}"
        )

        return TaskGroup(
            group_id=group_id,
            family="GeneReg-Dyn",
            model=cache.model,
            n_nodes=n,
            var_names=cache.var_names,
            target_type="marker_gene",
            marker_gene=cache.var_names[marker_idx],
            marker_gene_idx=marker_idx,
            observed=observed,
            k_min=res.k_min,
            minimal_sets=min_sets_names,
            minimal_sets_idx=res.min_sets,
            n_candidate_worlds=int(2 ** len(free_bits)),
            candidate_ids=None,
            branches=branches,
            metadata={
                "free_bits_idx": free_bits,
                "n_tested_subsets": res.n_tested,
                "n_attractors": int(len(attractors)),
                "n_distinct_outcomes": int(len(y_to_states)),
            },
            prompt_catalog_size=int(len(attractors)),
            prompt_catalog_is_capped=None,
        )

    return None


def _build_dyn_attractor_id_group_from_observed(
    cache: bnu.ModelCache,
    *,
    observed_idx_vals: List[Tuple[int, int]],
    max_k: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    construction_mode: str = "direct",
) -> Optional[TaskGroup]:
    """Rebuild a Dyn attractor-id group from an explicit observed context."""
    if cache.landscape is None or cache.landscape.basin_map is None:
        return None

    n = cache.n_nodes
    basin = cache.landscape.basin_map
    if basin.shape[0] != (1 << n):
        return None

    normalized = _normalize_observed_idx_vals(observed_idx_vals)
    if normalized is None:
        return None

    known_mask, known_value = bnu.build_mask_value_from_idx_vals(normalized)
    observed_genes = {gi for gi, _ in normalized}
    free_bits = [gi for gi in range(n) if gi not in observed_genes]
    if not free_bits:
        return None

    res = bnu.find_min_k_sufficient_sets_basin(
        n=n,
        basin_map=basin,
        context_mask=known_mask,
        context_value=known_value,
        candidate_genes=free_bits,
        target_type="attractor_id",
        max_k=max_k,
        max_sets_to_return=max_sets_to_return,
    )
    if res is None or res.k_min == 0:
        return None

    y_to_states: Dict[int, List[int]] = {}
    for st in bnu.iter_states_consistent_with_mask_value(n, known_mask, known_value):
        yv = int(basin[st])
        y_to_states.setdefault(yv, []).append(int(st))
    if len(y_to_states) < 2:
        return None

    branches: List[Branch] = []
    for yv, sts in y_to_states.items():
        rep_state = int(sts[int(rng.integers(0, len(sts)))])
        branches.append(Branch(y=int(yv), state=rep_state))

    observed = [(cache.var_names[i], v) for i, v in normalized]
    min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]

    group_id = (
        f"DYN|{cache.model}|attr_id|k={res.k_min}|free={len(free_bits)}"
        f"|mode={construction_mode}|seed={int(rng.integers(0, 1_000_000_000))}"
    )

    return TaskGroup(
        group_id=group_id,
        family="GeneReg-Dyn",
        model=cache.model,
        n_nodes=n,
        var_names=cache.var_names,
        target_type="attractor_id",
        marker_gene=None,
        marker_gene_idx=None,
        observed=observed,
        k_min=res.k_min,
        minimal_sets=min_sets_names,
        minimal_sets_idx=res.min_sets,
        n_candidate_worlds=int(1 << len(free_bits)),
        candidate_ids=None,
        branches=branches,
        metadata={
            "free_bits_idx": free_bits,
            "n_tested_subsets": res.n_tested,
            "n_distinct_outcomes": int(len(y_to_states)),
            "construction_mode": construction_mode,
        },
        prompt_catalog_size=int(len(cache.landscape.attractors)),
        prompt_catalog_is_capped=None,
    )


def _build_dyn_marker_group_from_observed(
    cache: bnu.ModelCache,
    *,
    marker_idx: int,
    observed_idx_vals: List[Tuple[int, int]],
    max_k: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    construction_mode: str = "direct",
) -> Optional[TaskGroup]:
    """Rebuild a Dyn marker group from explicit context + marker gene index."""
    if cache.landscape is None or cache.landscape.basin_map is None:
        return None

    n = cache.n_nodes
    if marker_idx < 0 or marker_idx >= n:
        return None

    basin = cache.landscape.basin_map
    if basin.shape[0] != (1 << n):
        return None

    attractors = cache.landscape.attractors
    if not attractors:
        return None

    # Keep cycle handling safe: marker must be constant within each attractor.
    for a in attractors:
        vals = [bnu.bit(st, marker_idx) for st in a.states]
        if min(vals) != max(vals):
            return None

    rep_states = bnu.attractor_representative_states(cache.landscape)
    if rep_states.size == 0:
        return None
    marker_by_attr = ((rep_states >> np.uint64(marker_idx)) & np.uint64(1)).astype(np.int8)

    normalized = _normalize_observed_idx_vals(observed_idx_vals)
    if normalized is None:
        return None

    known_mask, known_value = bnu.build_mask_value_from_idx_vals(normalized)
    observed_genes = {gi for gi, _ in normalized}
    free_bits = [gi for gi in range(n) if gi not in observed_genes]
    if not free_bits:
        return None

    res = bnu.find_min_k_sufficient_sets_basin(
        n=n,
        basin_map=basin,
        attractor_rep_states=rep_states,
        marker_idx=marker_idx,
        context_mask=known_mask,
        context_value=known_value,
        candidate_genes=free_bits,
        target_type="marker_gene",
        max_k=max_k,
        max_sets_to_return=max_sets_to_return,
    )
    if res is None or res.k_min == 0:
        return None

    y_to_states: Dict[int, List[int]] = {}
    for st in bnu.iter_states_consistent_with_mask_value(n, known_mask, known_value):
        yv = int(marker_by_attr[int(basin[st])])
        y_to_states.setdefault(yv, []).append(int(st))
    if len(y_to_states) < 2:
        return None

    branches: List[Branch] = []
    for yv, sts in y_to_states.items():
        rep_state = int(sts[int(rng.integers(0, len(sts)))])
        branches.append(Branch(y=int(yv), state=rep_state))

    observed = [(cache.var_names[i], v) for i, v in normalized]
    min_sets_names = [[cache.var_names[i] for i in s] for s in res.min_sets]

    group_id = (
        f"DYN|{cache.model}|marker={cache.var_names[marker_idx]}|k={res.k_min}|free={len(free_bits)}"
        f"|mode={construction_mode}|seed={int(rng.integers(0, 1_000_000_000))}"
    )

    return TaskGroup(
        group_id=group_id,
        family="GeneReg-Dyn",
        model=cache.model,
        n_nodes=n,
        var_names=cache.var_names,
        target_type="marker_gene",
        marker_gene=cache.var_names[marker_idx],
        marker_gene_idx=marker_idx,
        observed=observed,
        k_min=res.k_min,
        minimal_sets=min_sets_names,
        minimal_sets_idx=res.min_sets,
        n_candidate_worlds=int(1 << len(free_bits)),
        candidate_ids=None,
        branches=branches,
        metadata={
            "free_bits_idx": free_bits,
            "n_tested_subsets": res.n_tested,
            "n_attractors": int(len(attractors)),
            "n_distinct_outcomes": int(len(y_to_states)),
            "construction_mode": construction_mode,
        },
        prompt_catalog_size=int(len(attractors)),
        prompt_catalog_is_capped=None,
    )


def make_group_dyn_attractor_id_top_down(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    desired_free_bits_max: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct Dyn attractor-id examples by descending k via added reveals."""
    if target_k < 1 or target_k > max_k:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_dyn_attractor_id(
            cache,
            desired_free_bits_max=desired_free_bits_max,
            max_k=max_k,
            rng=rng,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.k_min < target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group
        if not seed_group.branches:
            continue

        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        anchor_state = int(seed_group.branches[int(rng.integers(0, len(seed_group.branches)))].state)

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min <= target_k:
                continue

            obs_set = {gi for gi, _ in cur_obs}
            preferred = []
            seen_pref = set()
            for subset in cur_group.minimal_sets_idx:
                for gi in subset:
                    gii = int(gi)
                    if gii in obs_set or gii in seen_pref:
                        continue
                    seen_pref.add(gii)
                    preferred.append(gii)
            if not preferred:
                preferred = [gi for gi in range(cache.n_nodes) if gi not in obs_set]
            if not preferred:
                continue

            order = [int(x) for x in rng.permutation(np.array(preferred, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = cur_obs + [(gi, int(bnu.bit(anchor_state, gi)))]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_dyn_attractor_id_group_from_observed(
                    cache,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="top_down",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min > target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


def make_group_dyn_attractor_id_bottom_up(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    desired_free_bits_max: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct Dyn attractor-id examples by removing observed assignments."""
    if target_k < 1 or target_k > max_k:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_dyn_attractor_id(
            cache,
            desired_free_bits_max=desired_free_bits_max,
            max_k=max_k,
            rng=rng,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.k_min > target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group

        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min > target_k:
                continue

            removable = [gi for gi, _ in cur_obs]
            if not removable:
                continue

            order = [int(x) for x in rng.permutation(np.array(removable, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = [(g, v) for g, v in cur_obs if int(g) != gi]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_dyn_attractor_id_group_from_observed(
                    cache,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="bottom_up",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min < target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


def make_group_dyn_marker_top_down(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    desired_free_bits_max: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct Dyn marker examples by descending k via added reveals."""
    if target_k < 1 or target_k > max_k:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_dyn_marker_gene(
            cache,
            desired_free_bits_max=desired_free_bits_max,
            max_k=max_k,
            rng=rng,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.marker_gene_idx is None:
            continue
        if seed_group.k_min < target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group
        if not seed_group.branches:
            continue

        marker_idx = int(seed_group.marker_gene_idx)
        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        anchor_state = int(seed_group.branches[int(rng.integers(0, len(seed_group.branches)))].state)

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min <= target_k:
                continue

            obs_set = {gi for gi, _ in cur_obs}
            preferred = []
            seen_pref = set()
            for subset in cur_group.minimal_sets_idx:
                for gi in subset:
                    gii = int(gi)
                    if gii in obs_set or gii in seen_pref:
                        continue
                    seen_pref.add(gii)
                    preferred.append(gii)
            if not preferred:
                preferred = [gi for gi in range(cache.n_nodes) if gi not in obs_set]
            if not preferred:
                continue

            order = [int(x) for x in rng.permutation(np.array(preferred, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = cur_obs + [(gi, int(bnu.bit(anchor_state, gi)))]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_dyn_marker_group_from_observed(
                    cache,
                    marker_idx=marker_idx,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="top_down",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min > target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


def make_group_dyn_marker_bottom_up(
    cache: bnu.ModelCache,
    *,
    target_k: int,
    max_k: int,
    desired_free_bits_max: int,
    rng: np.random.Generator,
    max_sets_to_return: int = 50,
    anchor_tries: int = 6,
    search_budget: int = 150,
    branch_width: int = 8,
) -> Optional[TaskGroup]:
    """Construct Dyn marker examples by removing observed assignments."""
    if target_k < 1 or target_k > max_k:
        return None

    for _ in range(anchor_tries):
        seed_group = make_group_dyn_marker_gene(
            cache,
            desired_free_bits_max=desired_free_bits_max,
            max_k=max_k,
            rng=rng,
            max_sets_to_return=max_sets_to_return,
        )
        if seed_group is None:
            continue
        if seed_group.marker_gene_idx is None:
            continue
        if seed_group.k_min > target_k:
            continue
        if seed_group.k_min == target_k:
            return seed_group

        marker_idx = int(seed_group.marker_gene_idx)
        seed_obs = _observed_named_to_idx_vals(seed_group.observed, cache.var_names)
        if seed_obs is None:
            continue

        queue = deque([(seed_group, seed_obs)])
        visited = {tuple(seed_obs)}
        explored = 0

        while queue and explored < search_budget:
            cur_group, cur_obs = queue.popleft()
            explored += 1

            if cur_group.k_min == target_k:
                return cur_group
            if cur_group.k_min > target_k:
                continue

            # For dyn marker, marker_idx is a target at convergence and can still be
            # an observed initial-state gene, so we do not exclude it from removal.
            removable = [gi for gi, _ in cur_obs]
            if not removable:
                continue

            order = [int(x) for x in rng.permutation(np.array(removable, dtype=np.int32)).tolist()]
            for gi in order[: max(1, min(branch_width, len(order)))]:
                new_obs = [(g, v) for g, v in cur_obs if int(g) != gi]
                key = tuple(sorted(new_obs))
                if key in visited:
                    continue
                visited.add(key)

                new_group = _build_dyn_marker_group_from_observed(
                    cache,
                    marker_idx=marker_idx,
                    observed_idx_vals=new_obs,
                    max_k=max_k,
                    rng=rng,
                    max_sets_to_return=max_sets_to_return,
                    construction_mode="bottom_up",
                )
                if new_group is None:
                    continue
                if new_group.k_min == target_k:
                    return new_group
                if new_group.k_min < target_k:
                    next_obs = _observed_named_to_idx_vals(new_group.observed, cache.var_names)
                    if next_obs is not None:
                        queue.append((new_group, next_obs))

    return None


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def taskgroup_to_json(group: TaskGroup) -> Dict:
    feasible_target_values: List[int]
    if group.family == "GeneReg-SS" and group.target_type == "attractor_id" and group.candidate_ids is not None:
        feasible_target_values = [int(x) for x in group.candidate_ids]
    else:
        feasible_target_values = sorted({int(b.y) for b in group.branches})

    metadata = dict(group.metadata)
    if group.family == "GeneReg-Dyn" and group.n_candidate_worlds is not None:
        metadata.setdefault("context_state_count", int(group.n_candidate_worlds))

    return {
        "schema_version": "grn_multi_v2",
        "group_id": group.group_id,
        "family": group.family,
        "model": group.model,
        "n_nodes": int(group.n_nodes),
        "var_names": list(group.var_names),
        "target_type": group.target_type,
        "marker_gene": group.marker_gene,
        "marker_gene_idx": group.marker_gene_idx,
        "observed": [[str(g), int(v)] for g, v in group.observed],
        "k_min": int(group.k_min),
        "minimal_sufficient_sets": [list(ms) for ms in group.minimal_sets],
        "minimal_sufficient_sets_idx": [[int(i) for i in ms] for ms in group.minimal_sets_idx],
        "prompt_catalog_size": (
            int(group.prompt_catalog_size) if group.prompt_catalog_size is not None else None
        ),
        "prompt_catalog_is_capped": (
            bool(group.prompt_catalog_is_capped) if group.prompt_catalog_is_capped is not None else None
        ),
        "feasible_target_values": feasible_target_values,
        "n_feasible_target_values": int(len(feasible_target_values)),
        "branches": [{"y": int(b.y), "state": int(b.state)} for b in group.branches],
        "metadata": metadata,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def task_type_from_group(group: TaskGroup) -> str:
    if group.family == "GeneReg-SS" and group.target_type == "marker_gene":
        return "ss_marker"
    if group.family == "GeneReg-SS" and group.target_type == "attractor_id":
        return "ss_id"
    if group.family == "GeneReg-Dyn" and group.target_type == "attractor_id":
        return "dyn_attr"
    if group.family == "GeneReg-Dyn" and group.target_type == "marker_gene":
        return "dyn_marker"
    raise ValueError(f"Unrecognized task type for group_id={group.group_id}")


def _forbid_profile_from_group(group: TaskGroup) -> Optional[Dict[str, Any]]:
    """Build canonical/forbid/allowed gene index profile used by forbid-alternatives eval."""
    if not group.minimal_sets_idx:
        return None

    norm_sets: List[Tuple[int, ...]] = []
    for subset in group.minimal_sets_idx:
        if not subset:
            continue
        norm_sets.append(tuple(sorted(int(i) for i in subset)))
    if not norm_sets:
        return None

    canonical = list(sorted(norm_sets, key=lambda s: (len(s), list(s)))[0])
    union_idx: Set[int] = set()
    for subset in norm_sets:
        union_idx.update(int(i) for i in subset)
    forbid_idx = sorted(union_idx - set(canonical))

    name_to_idx = {name: i for i, name in enumerate(group.var_names)}
    observed_idx = {int(name_to_idx[name]) for name, _ in group.observed if name in name_to_idx}
    queryable_idx = [i for i in range(group.n_nodes) if i not in observed_idx]
    forbid_idx_set = set(forbid_idx)
    canonical_set = set(canonical)
    allowed_idx = [i for i in queryable_idx if i not in forbid_idx_set]
    extra_idx = [i for i in allowed_idx if i not in canonical_set]
    # In no-forbid mode, "extra queryable" means genes outside the union of all minimal sets.
    extra_idx_no_forbid = [i for i in queryable_idx if i not in union_idx]

    return {
        "canonical_minset_idx": canonical,
        "forbid_alternative_gene_indices": forbid_idx,
        "queryable_gene_indices": queryable_idx,
        "allowed_gene_indices_after_forbid": allowed_idx,
        "extra_queryable_gene_indices_after_forbid": extra_idx,
        "extra_queryable_gene_indices_no_forbid": extra_idx_no_forbid,
        "n_canonical": int(len(canonical)),
        "n_forbid": int(len(forbid_idx)),
        "n_queryable": int(len(queryable_idx)),
        "n_allowed_after_forbid": int(len(allowed_idx)),
        "n_extra_queryable_after_forbid": int(len(extra_idx)),
        "n_extra_queryable_no_forbid": int(len(extra_idx_no_forbid)),
    }


def _dyn_marker_stable_extra_profile(
    group: TaskGroup,
    cache: Optional[bnu.ModelCache],
    *,
    candidate_extra_idx: List[int],
) -> Optional[Dict[str, Any]]:
    """For dyn_marker groups, keep only distractors stable within each reachable attractor.

    We require:
      1) each gene is constant inside every reachable attractor cycle, and
      2) across reachable attractors, the gene has both 0 and 1 (non-trivial).
    """
    if group.family != "GeneReg-Dyn" or group.target_type != "marker_gene":
        return None
    if cache is None or cache.landscape is None or cache.landscape.basin_map is None:
        return None

    n = int(group.n_nodes)
    basin = cache.landscape.basin_map
    if basin.shape[0] != (1 << n):
        return None

    name_to_idx = {name: i for i, name in enumerate(group.var_names)}
    observed_idx_vals: List[Tuple[int, int]] = []
    for g, v in group.observed:
        if g not in name_to_idx:
            return None
        observed_idx_vals.append((int(name_to_idx[g]), int(v)))
    observed_idx_vals = sorted(observed_idx_vals)
    context_mask, context_value = bnu.build_mask_value_from_idx_vals(observed_idx_vals)

    reachable_attr_ids: Set[int] = set()
    for st in bnu.iter_states_consistent_with_mask_value(n, context_mask, context_value):
        reachable_attr_ids.add(int(basin[int(st)]))
    if len(reachable_attr_ids) < 2:
        return None

    attractors = cache.landscape.attractors
    stable_idx: List[int] = []
    stable_binary_idx: List[int] = []
    for gi in sorted(set(int(i) for i in candidate_extra_idx)):
        if gi < 0 or gi >= n:
            continue
        per_attr_vals: List[int] = []
        ok = True
        for aid in reachable_attr_ids:
            if aid < 0 or aid >= len(attractors):
                ok = False
                break
            states = attractors[aid].states
            vals = [bnu.bit(st, gi) for st in states]
            if not vals or min(vals) != max(vals):
                ok = False
                break
            per_attr_vals.append(int(vals[0]))
        if not ok:
            continue
        stable_idx.append(gi)
        if per_attr_vals and min(per_attr_vals) != max(per_attr_vals):
            stable_binary_idx.append(gi)

    return {
        "reachable_attr_ids": sorted(int(x) for x in reachable_attr_ids),
        "stable_extra_gene_indices": sorted(stable_idx),
        "stable_binary_extra_gene_indices": sorted(stable_binary_idx),
        "n_reachable_attractors": int(len(reachable_attr_ids)),
        "n_stable_extra": int(len(stable_idx)),
        "n_stable_binary_extra": int(len(stable_binary_idx)),
    }


def _mean_or_none(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_stats_payload(
    *,
    accepted_groups: List[TaskGroup],
    per_task_k_quota: Dict[Tuple[str, int], int],
    task_types: List[str],
    attempts: int,
    max_attempts: int,
    seed: int,
    max_k: int,
    min_extra_queryable_after_forbid: int,
    min_stable_extra_queryable_dyn_marker: int,
    min_potential_outcomes_ss_id: int,
    min_potential_outcomes_dyn_attr: int,
    n_rejected_for_forbid_extra: int,
    n_rejected_for_dyn_marker_stable_extra: int,
    n_rejected_for_potential_outcomes: int,
) -> Dict[str, Any]:
    by_cell_rows: List[Dict[str, Any]] = []
    for task in task_types:
        for k in range(1, max_k + 1):
            cell_groups = [g for g in accepted_groups if task_type_from_group(g) == task and g.k_min == k]
            unique_models = sorted({g.model for g in cell_groups})
            row: Dict[str, Any] = {
                "task": task,
                "k": int(k),
                "n_problems": int(len(cell_groups)),
                "quota_target": int(per_task_k_quota.get((task, k), 0)),
                "n_unique_grns": int(len(unique_models)),
                "avg_num_nodes": _mean_or_none([float(g.n_nodes) for g in cell_groups]),
                "avg_known_set_size": _mean_or_none([float(len(g.observed)) for g in cell_groups]),
                "avg_num_potential_outcomes": _mean_or_none([float(len(g.branches)) for g in cell_groups]),
                "avg_prompt_catalog_size": _mean_or_none(
                    [float(g.prompt_catalog_size) for g in cell_groups if g.prompt_catalog_size is not None]
                ),
                "avg_num_minimal_sets": _mean_or_none([float(len(g.minimal_sets_idx)) for g in cell_groups]),
            }
            by_cell_rows.append(row)

    all_models = sorted({g.model for g in accepted_groups})
    overall = {
        "n_total_problems": int(len(accepted_groups)),
        "n_unique_grns": int(len(all_models)),
        "avg_num_nodes": _mean_or_none([float(g.n_nodes) for g in accepted_groups]),
        "avg_known_set_size": _mean_or_none([float(len(g.observed)) for g in accepted_groups]),
        "avg_num_potential_outcomes": _mean_or_none([float(len(g.branches)) for g in accepted_groups]),
        "avg_prompt_catalog_size": _mean_or_none(
            [float(g.prompt_catalog_size) for g in accepted_groups if g.prompt_catalog_size is not None]
        ),
        "avg_num_minimal_sets": _mean_or_none([float(len(g.minimal_sets_idx)) for g in accepted_groups]),
    }

    return {
        "meta": {
            "seed": int(seed),
            "attempts_used": int(attempts),
            "max_attempts": int(max_attempts),
            "max_k": int(max_k),
            "task_types": list(task_types),
            "min_extra_queryable_after_forbid": int(min_extra_queryable_after_forbid),
            "min_stable_extra_queryable_dyn_marker": int(min_stable_extra_queryable_dyn_marker),
            "min_potential_outcomes_ss_id": int(min_potential_outcomes_ss_id),
            "min_potential_outcomes_dyn_attr": int(min_potential_outcomes_dyn_attr),
            "n_rejected_for_forbid_extra": int(n_rejected_for_forbid_extra),
            "n_rejected_for_dyn_marker_stable_extra": int(n_rejected_for_dyn_marker_stable_extra),
            "n_rejected_for_potential_outcomes": int(n_rejected_for_potential_outcomes),
        },
        "overall": overall,
        "by_task_k": by_cell_rows,
    }


# -----------------------------------------------------------------------------
# Main dataset generation loop
# -----------------------------------------------------------------------------

def curate_dataset(
    *,
    models_dir: Path,
    out_dir: Path,
    cache_dir: Optional[Path] = None,
    n_groups: int,
    seed: int,
    max_k: int,
    ss_min_fp: int,
    ss_fp_cap: int,
    ss_max_fp_for_tasks: Optional[int],
    ss_max_nodes: int,
    dyn_max_nodes: int,
    dyn_max_attractors: Optional[int],
    dyn_free_bits_max: int,
    mix_ss_marker: float,
    mix_ss_id: float,
    mix_dyn: float,
    mix_dyn_marker: float,
    enabled_tasks: Optional[List[str]] = None,
    per_k_quota: Optional[Dict[int, int]] = None,
    quota_per_task_k: Optional[int] = None,
    tasks_filename: str = "tasks_new.jsonl",
    stats_json_filename: str = "tasks_new_stats.json",
    stats_csv_filename: str = "tasks_new_stats_by_task_k.csv",
    max_attempts: int = 2_000_000,
    progress_seconds: float = 10.0,
    progress_top_cells: int = 8,
    timeout_seconds: Optional[int] = None,
    max_model_samples: Optional[int] = None,
    min_extra_queryable_after_forbid: int = 3,
    min_stable_extra_queryable_dyn_marker: int = 3,
    min_potential_outcomes_ss_id: int = 1,
    min_potential_outcomes_dyn_attr: int = 1,
    verbose: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_effective = cache_dir if cache_dir is not None else (out_dir / "models_cache")
    cache_dir_effective.mkdir(parents=True, exist_ok=True)

    models = bnu.load_models_with_text(str(models_dir))
    if not models:
        raise RuntimeError(f"No .pickle models found under: {models_dir}")

    print(f"Loaded {len(models)} models. Computing/reading caches into {cache_dir_effective} ...", flush=True)

    caches: List[bnu.ModelCache] = []
    timed_out_models: List[str] = []
    start = time.time()
    models_to_process = models[:max_model_samples] if max_model_samples else models
    for i, m in enumerate(models_to_process, start=1):
        c = bnu.compute_or_load_model_cache(
            m,
            cache_dir=str(cache_dir_effective),
            regime_a_fp_cap=ss_fp_cap,
            dyn_max_nodes=dyn_max_nodes,
            compute_dyn=True,
            timeout_seconds=timeout_seconds,
            verbose=verbose,
        )
        if c.fixed_points_method == "timeout":
            timed_out_models.append(m["model"])
        caches.append(c)
        if i % 10 == 0:
            print(f"  cached {i}/{len(models_to_process)}", flush=True)
    print(f"Cache prep done in {time.time()-start:.1f}s", flush=True)
    if timed_out_models:
        print(f"  {len(timed_out_models)} models timed out: {timed_out_models[:5]}{'...' if len(timed_out_models) > 5 else ''}", flush=True)

    # Filter pools
    # ss_pool: needs fixed points for SS task generation, and n <= ss_max_nodes for uint64 state representation
    ss_pool = [
        c for c in caches
        if (
            len(c.fixed_points) >= ss_min_fp
            and c.n_nodes <= ss_max_nodes
            and (ss_max_fp_for_tasks is None or len(c.fixed_points) <= int(ss_max_fp_for_tasks))
        )
    ]
    # dyn_pool: needs basin map for dynamic trajectory analysis.
    # We allow both fixed-point and cyclic attractors; require >=2 attractors to avoid
    # trivially determined targets in regime B.
    dyn_pool = [c for c in caches if (
        c.landscape is not None and 
        c.landscape.basin_map is not None and 
        c.n_nodes <= dyn_max_nodes and
        len(c.landscape.attractors) >= 2 and
        (dyn_max_attractors is None or len(c.landscape.attractors) <= int(dyn_max_attractors))
    )]

    print(
        f"SS pool: {len(ss_pool)} models with >= {ss_min_fp} fixed points, "
        f"<= {ss_max_fp_for_tasks if ss_max_fp_for_tasks is not None else 'inf'} fixed points, "
        f"and n<={ss_max_nodes} (after cap={ss_fp_cap})."
    )
    print(
        f"Dyn pool: {len(dyn_pool)} models with basin maps, >=2 attractors, "
        f"<= {dyn_max_attractors if dyn_max_attractors is not None else 'inf'} attractors, "
        f"n<= {dyn_max_nodes}."
    )

    if not ss_pool and not dyn_pool:
        raise RuntimeError("No eligible models in either SS or Dyn pool.")
    cache_by_model = {c.model: c for c in caches}

    task_pool_available = {
        "ss_marker": bool(ss_pool),
        "ss_id": bool(ss_pool),
        "dyn_attr": bool(dyn_pool),
        "dyn_marker": bool(dyn_pool),
    }

    if enabled_tasks is None:
        active_task_types = list(TASK_TYPES)
    else:
        requested = [t for t in enabled_tasks if t]
        invalid = [t for t in requested if t not in TASK_TYPES]
        if invalid:
            raise ValueError(f"Unknown task types in enabled_tasks: {invalid}. Valid={list(TASK_TYPES)}")
        # Keep canonical ordering and remove duplicates.
        active_task_types = [t for t in TASK_TYPES if t in set(requested)]
    if not active_task_types:
        raise ValueError("No active task types after filtering.")

    for task in active_task_types:
        if not task_pool_available[task]:
            raise RuntimeError(
                f"Task '{task}' requested but has no eligible model pool "
                f"(ss_pool={len(ss_pool)}, dyn_pool={len(dyn_pool)})."
            )
    print(f"Active task types: {active_task_types}", flush=True)

    # Quotas in the requested 2D grid: (task_type, k)
    use_task_k_quota = quota_per_task_k is not None and int(quota_per_task_k) > 0
    per_task_k_quota: Dict[Tuple[str, int], int] = {}
    if use_task_k_quota:
        quota = int(quota_per_task_k)
        for task in active_task_types:
            for k in range(1, max_k + 1):
                per_task_k_quota[(task, k)] = quota
        target_total = int(sum(per_task_k_quota.values()))
        print(f"Using per-(task,k) quota mode: {quota} per cell, target_total={target_total}.", flush=True)
    else:
        # Backward-compatible mode: quota only by k over mixed task types
        if per_k_quota is None:
            base = n_groups // max_k
            per_k_quota = {k: base for k in range(1, max_k + 1)}
            rem = n_groups - base * max_k
            for k in range(1, rem + 1):
                per_k_quota[k] += 1
        target_total = int(sum(per_k_quota.values()))
        for task in active_task_types:
            for k in range(1, max_k + 1):
                per_task_k_quota[(task, k)] = 0

    counts = {k: 0 for k in range(1, max_k + 1)}
    counts_task_k = {(task, k): 0 for task in TASK_TYPES for k in range(1, max_k + 1)}
    accepted_groups: List[TaskGroup] = []
    seen_hashes = set()
    rejected_for_forbid_extra = 0
    rejected_for_dyn_marker_stable_extra = 0
    rejected_for_potential_outcomes = 0

    # mixture normalization
    mix_weights = {
        "ss_marker": float(mix_ss_marker),
        "ss_id": float(mix_ss_id),
        "dyn_attr": float(mix_dyn),
        "dyn_marker": float(mix_dyn_marker),
    }
    mix_task_order = [t for t in active_task_types if task_pool_available[t]]
    total_mix = float(sum(mix_weights[t] for t in mix_task_order))
    if total_mix <= 0:
        raise ValueError(
            f"Active task mix weights must sum to >0. "
            f"active_tasks={active_task_types}, weights={mix_weights}"
        )
    mix_task_probs = np.array([mix_weights[t] / total_mix for t in mix_task_order], dtype=np.float64)

    def pick_generator_from_mix() -> Optional[str]:
        if not mix_task_order:
            return None
        idx = int(rng.choice(len(mix_task_order), p=mix_task_probs))
        return mix_task_order[idx]

    def directional_modes_for_task(task_name: str, target_k: int) -> List[str]:
        """Choose top-down/bottom-up directions based on still-unfilled ks for this task."""
        if not use_task_k_quota:
            return []
        open_ks = [
            kk for kk in range(1, max_k + 1)
            if counts_task_k[(task_name, kk)] < per_task_k_quota[(task_name, kk)]
        ]
        if not open_ks:
            return []

        has_lower_open = any(kk < target_k for kk in open_ks)
        has_higher_open = any(kk > target_k for kk in open_ks)
        if not has_lower_open and not has_higher_open:
            return ["bottom_up", "top_down"]

        modes: List[str] = []
        if has_lower_open:
            modes.append("bottom_up")
        if has_higher_open:
            modes.append("top_down")
        return modes

    def _weighted_pick_ss_cache(pool: List[bnu.ModelCache], *, target_k: int) -> bnu.ModelCache:
        """Prefer high-FP SS models for harder ss_marker cells."""
        if not pool:
            raise RuntimeError("Empty SS pool")

        if target_k >= 4:
            hard = [c for c in pool if len(c.fixed_points) >= 64]
            if hard:
                pool = hard
        elif target_k >= 3:
            mid = [c for c in pool if len(c.fixed_points) >= 32]
            if mid:
                pool = mid

        fps = np.array([max(1, len(c.fixed_points)) for c in pool], dtype=np.float64)
        power = 2.0 if target_k >= 4 else (1.5 if target_k >= 3 else 1.0)
        w = np.power(fps, power)
        w = w / np.sum(w)
        idx = int(rng.choice(len(pool), p=w))
        return pool[idx]

    def _ss_marker_candidate_range(target_k: int) -> Tuple[int, int]:
        """Target-k-aware candidate size range for SS marker tasks."""
        if target_k <= 1:
            lo, hi = 8, 128
        elif target_k == 2:
            lo, hi = 16, 256
        elif target_k == 3:
            lo, hi = 32, 512
        else:
            lo, hi = 64, 1024

        cap = int(ss_fp_cap) if ss_fp_cap is not None else hi
        hi = min(hi, cap)
        lo = min(lo, hi)
        lo = max(2, lo)
        return lo, hi

    def sample_group_for_task(task_name: str, target_k: Optional[int] = None) -> Optional[TaskGroup]:
        if task_name == "ss_marker" and ss_pool:
            cache = py_rng.choice(ss_pool)
            if use_task_k_quota and target_k is not None:
                # For hard SS-marker cells, use larger candidate sets and harder markers.
                min_c, max_c = _ss_marker_candidate_range(target_k)
                marker_cap: Optional[float]
                if target_k >= 4:
                    marker_cap = 0.80
                elif target_k >= 3:
                    marker_cap = 0.88
                else:
                    marker_cap = None

                if target_k >= 3:
                    cache = _weighted_pick_ss_cache(ss_pool, target_k=target_k)

                modes = directional_modes_for_task(task_name, target_k)
                # For high-k SS-marker tails, explicitly try both constructions.
                if target_k >= 3:
                    mode_order: List[str] = []
                    for m in ("bottom_up", "top_down"):
                        if m not in mode_order:
                            mode_order.append(m)
                    for m in modes:
                        if m not in mode_order:
                            mode_order.append(m)
                    modes = mode_order

                for mode in modes:
                    if mode == "top_down":
                        cand = make_group_ss_marker_top_down(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            rng=rng,
                            seed_min_candidates=min_c,
                            seed_max_candidates=max_c,
                            marker_single_gene_acc_cap=marker_cap,
                            anchor_tries=16 if target_k >= 4 else 10,
                            search_budget=450 if target_k >= 4 else 260,
                            branch_width=14 if target_k >= 4 else 10,
                        )
                    else:
                        cand = make_group_ss_marker_bottom_up(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            rng=rng,
                            seed_min_candidates=min_c,
                            seed_max_candidates=max_c,
                            marker_single_gene_acc_cap=marker_cap,
                            anchor_tries=20 if target_k >= 4 else 12,
                            search_budget=520 if target_k >= 4 else 320,
                            branch_width=18 if target_k >= 4 else 12,
                        )
                    if cand is not None and cand.k_min == target_k:
                        return cand

                direct = make_group_ss_marker(
                    cache,
                    max_k=max_k,
                    rng=rng,
                    min_candidates=min_c,
                    max_candidates=max_c,
                    marker_single_gene_acc_cap=marker_cap,
                )
                if direct is not None and direct.k_min == target_k:
                    return direct
                return None

            return make_group_ss_marker(
                cache,
                max_k=max_k,
                rng=rng,
                min_candidates=16,
                max_candidates=256,
            )
        if task_name == "ss_id" and ss_pool:
            cache = py_rng.choice(ss_pool)
            if use_task_k_quota and target_k is not None:
                min_c, max_c = _ss_id_candidate_range_for_target_k(target_k, max_k)
                targeted = make_group_ss_attractor_id(
                    cache,
                    max_k=max_k,
                    rng=rng,
                    min_candidates=min_c,
                    max_candidates=max_c,
                )
                if targeted is not None and targeted.k_min == target_k:
                    return targeted

                for mode in directional_modes_for_task(task_name, target_k):
                    if mode == "top_down":
                        cand = make_group_ss_attractor_id_top_down(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            rng=rng,
                        )
                    else:
                        cand = make_group_ss_attractor_id_bottom_up(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            rng=rng,
                        )
                    if cand is not None and cand.k_min == target_k:
                        return cand

                direct = make_group_ss_attractor_id(
                    cache,
                    max_k=max_k,
                    rng=rng,
                    min_candidates=2,
                    max_candidates=1 << max_k,
                )
                if direct is not None and direct.k_min == target_k:
                    return direct
                return None

            return make_group_ss_attractor_id(
                cache,
                max_k=max_k,
                rng=rng,
                min_candidates=2,
                max_candidates=1 << max_k,
            )
        if task_name == "dyn_attr" and dyn_pool:
            cache = py_rng.choice(dyn_pool)
            if use_task_k_quota and target_k is not None:
                for mode in directional_modes_for_task(task_name, target_k):
                    if mode == "top_down":
                        cand = make_group_dyn_attractor_id_top_down(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            desired_free_bits_max=dyn_free_bits_max,
                            rng=rng,
                        )
                    else:
                        cand = make_group_dyn_attractor_id_bottom_up(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            desired_free_bits_max=dyn_free_bits_max,
                            rng=rng,
                        )
                    if cand is not None and cand.k_min == target_k:
                        return cand

                direct = make_group_dyn_attractor_id(
                    cache,
                    desired_free_bits_max=dyn_free_bits_max,
                    max_k=max_k,
                    rng=rng,
                )
                if direct is not None and direct.k_min == target_k:
                    return direct
                return None

            return make_group_dyn_attractor_id(
                cache,
                desired_free_bits_max=dyn_free_bits_max,
                max_k=max_k,
                rng=rng,
            )
        if task_name == "dyn_marker" and dyn_pool:
            cache = py_rng.choice(dyn_pool)
            if use_task_k_quota and target_k is not None:
                for mode in directional_modes_for_task(task_name, target_k):
                    if mode == "top_down":
                        cand = make_group_dyn_marker_top_down(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            desired_free_bits_max=dyn_free_bits_max,
                            rng=rng,
                        )
                    else:
                        cand = make_group_dyn_marker_bottom_up(
                            cache,
                            target_k=target_k,
                            max_k=max_k,
                            desired_free_bits_max=dyn_free_bits_max,
                            rng=rng,
                        )
                    if cand is not None and cand.k_min == target_k:
                        return cand

                direct = make_group_dyn_marker_gene(
                    cache,
                    desired_free_bits_max=dyn_free_bits_max,
                    max_k=max_k,
                    rng=rng,
                )
                if direct is not None and direct.k_min == target_k:
                    return direct
                return None

            return make_group_dyn_marker_gene(
                cache,
                desired_free_bits_max=dyn_free_bits_max,
                max_k=max_k,
                rng=rng,
            )
        return None

    def done() -> bool:
        if use_task_k_quota:
            return all(counts_task_k[cell] >= per_task_k_quota[cell] for cell in per_task_k_quota)
        return all(counts[k] >= per_k_quota[k] for k in per_k_quota)

    def per_task_row_string(task: str) -> str:
        parts = []
        for kk in range(1, max_k + 1):
            cur = counts_task_k[(task, kk)]
            tgt = per_task_k_quota.get((task, kk), 0)
            parts.append(f"k{kk}:{cur}/{tgt}")
        return f"{task}[" + ", ".join(parts) + "]"

    def hardest_cells_string(top_n: int) -> str:
        deficits = []
        for (task, kk), tgt in per_task_k_quota.items():
            rem = int(tgt - counts_task_k[(task, kk)])
            if rem > 0:
                deficits.append((rem, task, kk, counts_task_k[(task, kk)], tgt))
        deficits.sort(reverse=True)
        if not deficits:
            return "none"
        top_items = deficits[: max(1, int(top_n))]
        return "; ".join(
            f"{task}|k={kk}:{cur}/{tgt} (rem={rem})"
            for rem, task, kk, cur, tgt in top_items
        )

    attempt = 0
    last_report = time.time()
    out_path = out_dir / tasks_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")
    print(f"Streaming accepted groups to {out_path} (flush per hit).", flush=True)
    try:
        while not done() and attempt < max_attempts:
            attempt += 1
            target_k: Optional[int] = None
            if use_task_k_quota:
                remaining_cells = [
                    (task, k)
                    for (task, k), q in per_task_k_quota.items()
                    if counts_task_k[(task, k)] < q
                ]
                if not remaining_cells:
                    break
                gen, target_k = remaining_cells[int(rng.integers(0, len(remaining_cells)))]
            else:
                gen = pick_generator_from_mix()
                if gen is None:
                    continue

            group = sample_group_for_task(gen, target_k if use_task_k_quota else None)
            if group is None:
                continue

            task_type = task_type_from_group(group)
            if task_type != gen:
                continue

            k = group.k_min
            if k < 1 or k > max_k:
                continue
            if use_task_k_quota:
                if target_k is not None and k != target_k:
                    continue
                if counts_task_k[(task_type, k)] >= per_task_k_quota[(task_type, k)]:
                    continue
            else:
                if counts[k] >= per_k_quota.get(k, 0):
                    continue

            n_outcomes = int(len(group.branches))
            if task_type == "ss_id" and n_outcomes < int(min_potential_outcomes_ss_id):
                rejected_for_potential_outcomes += 1
                continue
            if task_type == "dyn_attr" and n_outcomes < int(min_potential_outcomes_dyn_attr):
                rejected_for_potential_outcomes += 1
                continue

            forbid_profile = _forbid_profile_from_group(group)
            if forbid_profile is None:
                continue
            if (
                int(forbid_profile["n_extra_queryable_after_forbid"]) < int(min_extra_queryable_after_forbid)
                or int(forbid_profile["n_extra_queryable_no_forbid"]) < int(min_extra_queryable_after_forbid)
            ):
                rejected_for_forbid_extra += 1
                continue

            if task_type == "dyn_marker":
                stable_profile = _dyn_marker_stable_extra_profile(
                    group,
                    cache_by_model.get(group.model),
                    candidate_extra_idx=list(forbid_profile["extra_queryable_gene_indices_no_forbid"]),
                )
                if (
                    stable_profile is None
                    or int(stable_profile["n_stable_binary_extra"]) < int(min_stable_extra_queryable_dyn_marker)
                ):
                    rejected_for_dyn_marker_stable_extra += 1
                    continue
                forbid_profile["dyn_marker_stable_profile"] = stable_profile
            group.metadata["forbid_profile"] = forbid_profile

            # de-dup by semantic content of the task
            h = (
                task_type,
                group.k_min,
                group.model,
                group.family,
                group.target_type,
                group.marker_gene_idx,
                tuple(sorted(group.observed)),
            )
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            accepted_groups.append(group)
            counts[k] += 1
            counts_task_k[(task_type, k)] += 1
            out_f.write(json.dumps(taskgroup_to_json(group), ensure_ascii=False) + "\n")
            out_f.flush()

            if time.time() - last_report > progress_seconds:
                last_report = time.time()
                if use_task_k_quota:
                    filled_cells = sum(
                        1 for cell, q in per_task_k_quota.items()
                        if counts_task_k[cell] >= q
                    )
                    print(
                        f"[{attempt}] collected={len(accepted_groups)}/{target_total} "
                        f"filled_cells={filled_cells}/{len(per_task_k_quota)} counts_k={counts} "
                        f"rej_forbid_extra={rejected_for_forbid_extra} "
                        f"rej_dyn_marker_stable={rejected_for_dyn_marker_stable_extra} "
                        f"rej_potential_outcomes={rejected_for_potential_outcomes}",
                        flush=True,
                    )
                    print(
                        "  per_task_k: " + " | ".join(per_task_row_string(task) for task in active_task_types),
                        flush=True,
                    )
                    print(
                        f"  hardest_cells: {hardest_cells_string(progress_top_cells)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{attempt}] collected={len(accepted_groups)}/{target_total} "
                        f"counts={counts} pools=(ss={len(ss_pool)},dyn={len(dyn_pool)}) "
                        f"rej_forbid_extra={rejected_for_forbid_extra} "
                        f"rej_dyn_marker_stable={rejected_for_dyn_marker_stable_extra} "
                        f"rej_potential_outcomes={rejected_for_potential_outcomes}",
                        flush=True,
                    )
    finally:
        out_f.close()

    if not done():
        if use_task_k_quota:
            missing = {
                f"{task}|k={k}": int(q - counts_task_k[(task, k)])
                for (task, k), q in per_task_k_quota.items()
                if counts_task_k[(task, k)] < q
            }
            print(f"WARNING: did not hit all per-(task,k) quotas within max_attempts={max_attempts}. missing={missing}")
        else:
            missing = {
                int(k): int(per_k_quota[k] - counts[k])
                for k in per_k_quota
                if counts[k] < per_k_quota[k]
            }
            print(f"WARNING: did not hit all per-k quotas within max_attempts={max_attempts}. missing={missing}")

    groups_out = [taskgroup_to_json(g) for g in accepted_groups]

    # Stats outputs
    if not use_task_k_quota:
        # In legacy mode we did not enforce per-(task,k) targets, so set targets to observed counts.
        for task in active_task_types:
            for k in range(1, max_k + 1):
                per_task_k_quota[(task, k)] = counts_task_k[(task, k)]

    stats_payload = build_stats_payload(
        accepted_groups=accepted_groups,
        per_task_k_quota=per_task_k_quota,
        task_types=active_task_types,
        attempts=attempt,
        max_attempts=max_attempts,
        seed=seed,
        max_k=max_k,
        min_extra_queryable_after_forbid=int(min_extra_queryable_after_forbid),
        min_stable_extra_queryable_dyn_marker=int(min_stable_extra_queryable_dyn_marker),
        min_potential_outcomes_ss_id=int(min_potential_outcomes_ss_id),
        min_potential_outcomes_dyn_attr=int(min_potential_outcomes_dyn_attr),
        n_rejected_for_forbid_extra=int(rejected_for_forbid_extra),
        n_rejected_for_dyn_marker_stable_extra=int(rejected_for_dyn_marker_stable_extra),
        n_rejected_for_potential_outcomes=int(rejected_for_potential_outcomes),
    )
    stats_json_path = out_dir / stats_json_filename
    stats_csv_path = out_dir / stats_csv_filename
    write_json(stats_json_path, stats_payload)
    write_csv(
        stats_csv_path,
        rows=stats_payload["by_task_k"],
        fieldnames=[
            "task",
            "k",
            "n_problems",
            "quota_target",
            "n_unique_grns",
            "avg_num_nodes",
            "avg_known_set_size",
            "avg_num_potential_outcomes",
            "avg_prompt_catalog_size",
            "avg_num_minimal_sets",
        ],
    )

    print(f"Wrote {len(groups_out)} groups to {out_path}")
    print(f"Wrote stats JSON to {stats_json_path}")
    print(f"Wrote stats CSV to {stats_csv_path}")
    print(f"Counts per k: {counts}")
    print(f"Rejected by forbid-extra check: {rejected_for_forbid_extra}")
    print(f"Rejected by dyn-marker stable-extra check: {rejected_for_dyn_marker_stable_extra}")
    print(f"Rejected by potential-outcomes check: {rejected_for_potential_outcomes}")


def main() -> None:
    import os
    DEFAULT_OUT_DIR = Path(os.environ.get("DATA_DIR", "./data"))
    DEFAULT_LOG_DIR = Path(os.environ.get("RESULTS_DIR", "./logs")) / "grn_multi_data"

    ap = argparse.ArgumentParser(
        description="Generate GRN reasoning benchmark tasks from Boolean network models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # I/O paths
    ap.add_argument("--models_dir", type=Path, required=True,
                    help="Path to directory containing .pickle Boolean network model files")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Output directory for model caches and generated tasks.jsonl (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="Optional cache directory for per-model artifacts (default: out_dir/models_cache)")
    ap.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR,
                    help=f"Directory for timestamped log files (default: {DEFAULT_LOG_DIR})")
    
    # Task generation settings
    ap.add_argument("--n_groups", type=int, default=1000,
                    help="Number of task groups to generate in legacy per-k quota mode")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility")
    ap.add_argument("--max_k", type=int, default=4,
                    help="Maximum k for k-sufficient sets (tasks will have k from 1 to max_k)")
    ap.add_argument("--quota_per_task_k", type=int, default=None,
                    help="If set (e.g., 100), enforce this many unique groups for each (task_type, k) cell")
    ap.add_argument("--exclude_tasks", type=str, default="",
                    help=f"Comma-separated task types to skip (choices: {','.join(TASK_TYPES)}), e.g. 'ss_marker'")
    ap.add_argument("--tasks_filename", type=str, default="tasks_new.jsonl",
                    help="Output JSONL filename under out_dir")
    ap.add_argument("--stats_json_filename", type=str, default="tasks_new_stats.json",
                    help="Output stats JSON filename under out_dir")
    ap.add_argument("--stats_csv_filename", type=str, default="tasks_new_stats_by_task_k.csv",
                    help="Output per-(task,k) stats CSV filename under out_dir")

    # GeneReg-SS (Steady-State regime) settings
    ap.add_argument("--ss_min_fp", type=int, default=16,
                    help="Minimum fixed points required for SS pool eligibility (underspecification)")
    ap.add_argument("--ss_fp_cap", type=int, default=512,
                    help="Cap on fixed points per model to avoid memory issues during enumeration")
    ap.add_argument("--ss_max_fp_for_tasks", type=int, default=512,
                    help="Exclude SS-task models with more than this many fixed points after capping")
    ap.add_argument("--ss_max_nodes", type=int, default=64,
                    help="Max nodes for SS pool (states represented as uint64, so n<=64)")

    # GeneReg-Dyn (Dynamic regime) settings
    ap.add_argument("--dyn_max_nodes", type=int, default=21,
                    help="Max nodes for Dyn pool (brute-force basin map requires 2^n states)")
    ap.add_argument("--dyn_max_attractors", type=int, default=256,
                    help="Exclude Dyn-task models with more than this many attractors")
    ap.add_argument("--dyn_free_bits_max", type=int, default=12,
                    help="Max free bits (unknown genes) in dynamic task scenarios")

    # Task type mixture (should sum to 1.0 for balanced generation)
    ap.add_argument("--mix_ss_marker", type=float, default=0.3,
                    help="Fraction of SS tasks with marker-gene target (predict gene value at steady-state)")
    ap.add_argument("--mix_ss_id", type=float, default=0.2,
                    help="Fraction of SS tasks with attractor-id target (identify which fixed point)")
    ap.add_argument("--mix_dyn_marker", type=float, default=0.3,
                    help="Fraction of Dyn tasks with marker-gene-at-convergence target")
    ap.add_argument("--mix_dyn", type=float, default=0.2,
                    help="Fraction of Dyn tasks with attractor-id target (identify basin)")

    # Performance settings
    ap.add_argument("--max_attempts", type=int, default=2_000_000,
                    help="Maximum sampling attempts during task-group generation")
    ap.add_argument("--progress_seconds", type=float, default=10.0,
                    help="How often (seconds) to print generation progress")
    ap.add_argument("--progress_top_cells", type=int, default=8,
                    help="How many hardest remaining (task,k) cells to print per progress report")
    ap.add_argument("--timeout_seconds", type=int, default=None,
                    help="Per-model timeout in seconds for cache computation (skip slow models)")
    ap.add_argument("--max_model_samples", type=int, default=None,
                    help="Max number of models to process (for debugging/quick tests)")
    ap.add_argument("--min_extra_queryable_after_forbid", type=int, default=3,
                    help="Require at least this many extra queryable genes after forbid filtering (and also in no-forbid mode)")
    ap.add_argument("--min_stable_extra_queryable_dyn_marker", type=int, default=3,
                    help="For dyn_marker, require at least this many extra genes that are cycle-stable and binary across reachable attractors")
    ap.add_argument("--min_potential_outcomes_ss_id", type=int, default=1,
                    help="For ss_id, require at least this many distinct potential outcomes (len(branches))")
    ap.add_argument("--min_potential_outcomes_dyn_attr", type=int, default=1,
                    help="For dyn_attr, require at least this many distinct potential outcomes (len(branches))")

    args = ap.parse_args()

    excluded_tasks = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()]
    invalid_excluded = [t for t in excluded_tasks if t not in TASK_TYPES]
    if invalid_excluded:
        raise ValueError(f"Unknown tasks in --exclude_tasks: {invalid_excluded}. Valid={list(TASK_TYPES)}")
    enabled_tasks = [t for t in TASK_TYPES if t not in set(excluded_tasks)]
    if not enabled_tasks:
        raise ValueError("All task types were excluded; nothing to generate.")

    # Set up logging to file
    import sys
    from datetime import datetime
    
    args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f"curate_{timestamp}_seed{args.seed}.log"
    
    # write to both stdout and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_fh = open(log_file, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_fh)
    
    print(f"=== GRN Dataset Curation ===")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Log file: {log_file}")
    print(f"Output dir: {args.out_dir}")
    print(f"Cache dir: {args.cache_dir if args.cache_dir is not None else (args.out_dir / 'models_cache')}")
    print(f"Models dir: {args.models_dir}")
    print(f"Params: seed={args.seed}, n_groups={args.n_groups}, max_k={args.max_k}")
    print(f"         quota_per_task_k={args.quota_per_task_k}")
    print(f"         enabled_tasks={enabled_tasks}")
    print(f"         dyn_max_nodes={args.dyn_max_nodes}, timeout={args.timeout_seconds}s")
    print(f"         ss_max_fp_for_tasks={args.ss_max_fp_for_tasks}, dyn_max_attractors={args.dyn_max_attractors}")
    print(f"         min_extra_queryable_after_forbid={args.min_extra_queryable_after_forbid}, "
          f"min_stable_extra_queryable_dyn_marker={args.min_stable_extra_queryable_dyn_marker}")
    print(f"         min_potential_outcomes_ss_id={args.min_potential_outcomes_ss_id}, "
          f"min_potential_outcomes_dyn_attr={args.min_potential_outcomes_dyn_attr}")
    print(f"         progress_seconds={args.progress_seconds}, progress_top_cells={args.progress_top_cells}")
    print()

    try:
        curate_dataset(
            models_dir=args.models_dir,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            n_groups=args.n_groups,
            seed=args.seed,
            max_k=args.max_k,
            ss_min_fp=args.ss_min_fp,
            ss_fp_cap=args.ss_fp_cap,
            ss_max_fp_for_tasks=args.ss_max_fp_for_tasks,
            ss_max_nodes=args.ss_max_nodes,
            dyn_max_nodes=args.dyn_max_nodes,
            dyn_max_attractors=args.dyn_max_attractors,
            dyn_free_bits_max=args.dyn_free_bits_max,
            mix_ss_marker=args.mix_ss_marker,
            mix_ss_id=args.mix_ss_id,
            mix_dyn=args.mix_dyn,
            mix_dyn_marker=args.mix_dyn_marker,
            enabled_tasks=enabled_tasks,
            quota_per_task_k=args.quota_per_task_k,
            tasks_filename=args.tasks_filename,
            stats_json_filename=args.stats_json_filename,
            stats_csv_filename=args.stats_csv_filename,
            max_attempts=args.max_attempts,
            progress_seconds=args.progress_seconds,
            progress_top_cells=args.progress_top_cells,
            timeout_seconds=args.timeout_seconds,
            max_model_samples=args.max_model_samples,
            min_extra_queryable_after_forbid=args.min_extra_queryable_after_forbid,
            min_stable_extra_queryable_dyn_marker=args.min_stable_extra_queryable_dyn_marker,
            min_potential_outcomes_ss_id=args.min_potential_outcomes_ss_id,
            min_potential_outcomes_dyn_attr=args.min_potential_outcomes_dyn_attr,
        )
    finally:
        sys.stdout = original_stdout
        log_fh.close()
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
