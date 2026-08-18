"""verify_k_minimality.py

Verify that generated GRN tasks are truly k-minimal sufficient.

For each task, we verify:
1. SUFFICIENCY: At least one of the minimal_sets of size k determines y
2. GLOBAL MINIMALITY: No (k-1)-element combination is sufficient, searched over all
   non-observed genes for steady-state tasks and over the recorded free bits for
   dynamic tasks

Usage:
    python verify_k_minimality.py --n_samples 10 --seed 42
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Any

import boolean_network_utils as bnu

DEFAULT_TASKS = Path(os.path.join(os.environ.get("DATA_DIR", "data"), "genereg_mt.jsonl"))
DEFAULT_CACHE = Path(os.environ.get("CACHE_DIR", "./.grn_cache"))
DEFAULT_MODELS = Path(os.environ.get("MODELS_DIR", "/path/to/DesignPrinciplesGeneNetworks/update_rules_122_models_Kadelka_SciAdv"))


def is_sufficient_ss(
    cand_states: np.ndarray,
    y: np.ndarray,
    query_genes: List[int],
) -> bool:
    """Check if querying `query_genes` determines y for SS regime (fixed points)."""
    if len(query_genes) == 0:
        # y must be constant
        return int(y.min()) == int(y.max())
    
    mask = np.uint64(0)
    for gi in query_genes:
        mask |= np.uint64(1) << np.uint64(gi)
    
    # Group states by query pattern, check if y is constant within each group
    seen: Dict[int, int] = {}
    for st, yv in zip(cand_states.tolist(), y.tolist()):
        key = int(np.uint64(st) & mask)
        if key in seen:
            if seen[key] != int(yv):
                return False
        else:
            seen[key] = int(yv)
    return True


def is_sufficient_dyn(
    n: int,
    basin_map: np.ndarray,
    context_mask: int,
    context_value: int,
    query_genes: List[int],
    target_type: str,
    marker_idx: Optional[int],
    attractor_rep: Optional[np.ndarray],
) -> bool:
    """Check if querying `query_genes` determines y for Dyn regime (basin map)."""
    qmask = 0
    for gi in query_genes:
        qmask |= 1 << int(gi)
    
    seen: Dict[int, int] = {}
    
    for st in bnu.iter_states_consistent_with_mask_value(n, context_mask, context_value):
        key = st & qmask
        
        a_id = int(basin_map[st])
        if target_type == "attractor_id":
            yv = a_id
        else:
            rep = int(attractor_rep[a_id])
            yv = (rep >> int(marker_idx)) & 1
        
        if key in seen:
            if seen[key] != yv:
                return False
        else:
            seen[key] = yv
    
    return True


def verify_task(
    task: Dict[str, Any],
    models_dict: Dict[str, Dict],
    cache_dir: Path,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Verify that a task is truly k-minimal sufficient.
    
    Checks:
    1. The claimed minimal_sets are sufficient
    2. No (k-1)-element combination of the searchable genes is sufficient (global minimality)
    """
    model_name = task["model"]
    k_min = task["k_min"]
    minimal_sets_idx = task["minimal_sufficient_sets_idx"]
    observed = task["observed"]
    target_type = task["target_type"]
    marker_gene_idx = task.get("marker_gene_idx")
    family = task["family"]
    
    if model_name not in models_dict:
        return {"status": "skip", "reason": "model not found"}
    
    cache_path = cache_dir / f"{model_name}.model_cache.pkl"
    if not cache_path.exists():
        return {"status": "skip", "reason": "cache not found"}
    
    cache = bnu.load_model_cache(str(cache_path))
    n = cache.n_nodes
    var_names = cache.var_names
    
    # Build observed mask/value
    name_to_idx = {name: i for i, name in enumerate(var_names)}
    obs_mask, obs_value = 0, 0
    for gname, val in observed:
        gi = name_to_idx[gname]
        obs_mask |= 1 << gi
        if val:
            obs_value |= 1 << gi
    
    # Determine candidate genes (queryable genes)
    obs_genes = {name_to_idx[gname] for gname, _ in observed}
    if target_type == "marker_gene" and marker_gene_idx is not None:
        obs_genes.add(marker_gene_idx)
    candidate_genes = [i for i in range(n) if i not in obs_genes]
    
    # === Handle SS regime (fixed points) ===
    if family == "GeneReg-SS":
        omega = np.array(cache.fixed_points, dtype=np.uint64)
        m = np.uint64(obs_mask)
        p = np.uint64(obs_value)
        keep = (omega & m) == p
        cand_states = omega[keep]
        
        if cand_states.shape[0] < 2:
            return {"status": "skip", "reason": "not enough candidates"}
        
        if target_type == "marker_gene":
            y = ((cand_states >> np.uint64(marker_gene_idx)) & np.uint64(1)).astype(np.int8)
        else:
            # attractor_id: indices in filtered omega
            y = np.arange(cand_states.shape[0], dtype=np.int32)
        
        # Check 1: Sufficiency - at least one claimed minimal set works
        any_sufficient = False
        for min_set in minimal_sets_idx:
            if is_sufficient_ss(cand_states, y, min_set):
                any_sufficient = True
                break
        
        if not any_sufficient:
            return {
                "status": "fail",
                "reason": "claimed minimal sets are NOT sufficient",
                "k_min": k_min,
            }
        
        # Check 2: Global minimality - no (k-1)-combination of candidate_genes is sufficient
        if k_min >= 1:
            for smaller_set in combinations(candidate_genes, k_min - 1):
                if is_sufficient_ss(cand_states, y, list(smaller_set)):
                    return {
                        "status": "fail",
                        "reason": f"smaller sufficient set found: {list(smaller_set)}",
                        "k_min": k_min,
                        "smaller_set": list(smaller_set),
                        "smaller_set_names": [var_names[i] for i in smaller_set],
                    }
        
        return {
            "status": "pass",
            "k_min": k_min,
            "n_candidates": len(candidate_genes),
            "n_states": int(cand_states.shape[0]),
        }
    
    # === Handle Dyn regime (basin map) ===
    else:
        if cache.landscape is None or cache.landscape.basin_map is None:
            return {"status": "skip", "reason": "no basin map"}
        
        basin = cache.landscape.basin_map
        attractor_rep = bnu.attractor_representative_states(cache.landscape)
        
        # Get free bits from metadata
        free_bits = task["metadata"].get("free_bits_idx", [])
        if not free_bits:
            return {"status": "skip", "reason": "no free bits in metadata"}
        
        # Check 1: Sufficiency
        any_sufficient = False
        for min_set in minimal_sets_idx:
            if is_sufficient_dyn(n, basin, obs_mask, obs_value, min_set, target_type, marker_gene_idx, attractor_rep):
                any_sufficient = True
                break
        
        if not any_sufficient:
            return {
                "status": "fail",
                "reason": "claimed minimal sets are NOT sufficient",
                "k_min": k_min,
            }
        
        # Check 2: Global minimality - no (k-1)-combination of free_bits is sufficient
        if k_min >= 1:
            for smaller_set in combinations(free_bits, k_min - 1):
                if is_sufficient_dyn(n, basin, obs_mask, obs_value, list(smaller_set), target_type, marker_gene_idx, attractor_rep):
                    return {
                        "status": "fail",
                        "reason": f"smaller sufficient set found: {list(smaller_set)}",
                        "k_min": k_min,
                        "smaller_set": list(smaller_set),
                        "smaller_set_names": [var_names[i] for i in smaller_set],
                    }
        
        return {
            "status": "pass",
            "k_min": k_min,
            "n_free_bits": len(free_bits),
        }


def main():
    parser = argparse.ArgumentParser(description="Verify k-minimality of GRN tasks")
    parser.add_argument("--tasks_file", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--n_samples", type=int, default=10, help="Number of tasks to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    print(f"Loading tasks from: {args.tasks_file}")
    with open(args.tasks_file) as f:
        tasks = [json.loads(line) for line in f]
    print(f"Loaded {len(tasks)} tasks")
    
    print(f"Loading models from: {args.models_dir}")
    models_dict = {m["model"]: m for m in bnu.load_models_with_text(str(args.models_dir))}
    print(f"Loaded {len(models_dict)} models")
    
    rng = np.random.default_rng(args.seed)
    if args.n_samples > 0 and args.n_samples < len(tasks):
        indices = rng.choice(len(tasks), size=args.n_samples, replace=False)
    else:
        indices = np.arange(len(tasks))
    
    print(f"\nVerifying {len(indices)} tasks for k-minimality...\n")
    print("=" * 80)
    
    results = []
    for i in indices:
        task = tasks[i]
        print(f"[{i:3d}] {task['family']:12s} | {task['target_type']:12s} | k={task['k_min']} | model={task['model'][:20]:20s}", end=" -> ")
        
        result = verify_task(task, models_dict, args.cache_dir, args.verbose)
        results.append(result)
        
        if result["status"] == "pass":
            extra = f"(n_cand={result.get('n_candidates', result.get('n_free_bits', '?'))})"
            print(f"✓ PASS {extra}")
        elif result["status"] == "skip":
            print(f"⊘ SKIP ({result['reason']})")
        else:
            print(f"✗ FAIL: {result['reason']}")
            if "smaller_set_names" in result:
                print(f"         Counterexample genes: {result['smaller_set_names']}")
    
    print("=" * 80)
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    skipped = sum(1 for r in results if r["status"] == "skip")
    
    print(f"\nSummary: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED out of {len(indices)}")
    
    if failed > 0:
        print("\n⚠ WARNING: Some tasks failed minimality verification!")
        print("Failed tasks:")
        for i, (idx, r) in enumerate(zip(indices, results)):
            if r["status"] == "fail":
                print(f"  - Task {idx}: {r['reason']}")
    else:
        print("\n✓ All verified tasks are truly k-minimal sufficient!")


if __name__ == "__main__":
    main()
