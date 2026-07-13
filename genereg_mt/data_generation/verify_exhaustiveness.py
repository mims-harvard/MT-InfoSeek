import json
import logging
from pathlib import Path
from itertools import combinations
import numpy as np
from tqdm import tqdm

import boolean_network_utils as bnu

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_exhaustiveness(jsonl_path, models_dir, cache_dir):
    with open(jsonl_path, "r") as f:
        tasks = [json.loads(line) for line in f]
        
    models_list = bnu.load_models_with_text(str(models_dir))
    models_dict = {m['model']: m for m in models_list}
    cache_objs = {}
    
    num_failures = 0
    total_missing = 0
    
    for row in tqdm(tasks):
        model_name = row['model']
        if model_name not in cache_objs:
            cache_objs[model_name] = bnu.compute_or_load_model_cache(
                models_dict[model_name], str(cache_dir), max_attractors=1000
            )
            
        cache = cache_objs[model_name]
        n = cache.n_nodes
        observed = row['observed']
        observed_idx_vals = [(cache.var_names.index(k), v) for k, v in observed]
        observed_idxs = {idx for idx, _ in observed_idx_vals}
        
        target_type = row['target_type']
        family = row['family']
        minimal_sets_idx = row['minimal_sufficient_sets_idx']
        reported_sets = {tuple(sorted(s)) for s in minimal_sets_idx}
        k_min = row['k_min']
        
        # Determine candidate genes
        if family == "GeneReg-SS":
            marker_idx = row.get('marker_gene_idx')
            cand_genes = [g for g in range(n) if g not in observed_idxs and g != marker_idx]
            
            omega = np.array(cache.fixed_points, dtype=np.uint64)
            cand_idx = []
            for i, st in enumerate(omega):
                if all(bnu.bit(int(st), idx) == v for idx, v in observed_idx_vals):
                    cand_idx.append(i)
            cand_states = omega[cand_idx]
            if target_type == "attractor_id":
                y = np.array(cand_idx, dtype=np.int32)
            else:
                y = np.array([bnu.bit(int(st), marker_idx) for st in cand_states], dtype=np.int32)
                
            all_k_sets = []
            for s_idx in combinations(cand_genes, k_min):
                s_idx = list(s_idx)
                if bnu._is_y_function_of_subset_states(cand_states, y, s_idx):
                    all_k_sets.append(tuple(sorted(s_idx)))
                    
        else:
            # GeneReg-Dyn
            marker_idx = row.get('marker_gene_idx')
            # For dyn marker, the marker itself can be queried. 
            cand_genes = [g for g in range(n) if g not in observed_idxs]
            mask, val = bnu.build_mask_value_from_idx_vals(observed_idx_vals)
            
            all_k_sets = []
            for s_idx in combinations(cand_genes, k_min):
                s_idx = list(s_idx)
                if bnu._is_y_function_of_subset_basin(
                    n=n, basin_map=cache.landscape.basin_map,
                    context_mask=mask, context_value=val,
                    subset=s_idx, target_type=target_type, marker_idx=marker_idx,
                    attractor_rep_states=bnu.attractor_representative_states(cache.landscape)
                ):
                    all_k_sets.append(tuple(sorted(s_idx)))
                    
        all_k_sets = set(all_k_sets)
        missing_sets = all_k_sets - reported_sets
        extra_sets = reported_sets - all_k_sets
        
        if missing_sets or extra_sets:
            logging.error(f"Row {row['group_id']}: Missing {len(missing_sets)} sets. Extra {len(extra_sets)} sets. Reported: {len(reported_sets)}, Total True: {len(all_k_sets)}")
            num_failures += 1
            total_missing += len(missing_sets)
            
    if num_failures == 0:
        logging.info("Exhaustiveness check PASSED: All tasks contain the EXACT complete set of k-minimal solutions.")
    else:
        logging.error(f"Exhaustiveness check FAILED: {num_failures} tasks were missing a total of {total_missing} sets.")

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=os.path.join(os.environ.get("DATA_DIR", "data"), "genereg_mt.jsonl"))
    parser.add_argument("--models", default=os.environ.get("MODELS_DIR", "/path/to/DesignPrinciplesGeneNetworks/update_rules_122_models_Kadelka_SciAdv"))
    parser.add_argument("--cache", default=os.environ.get("CACHE_DIR", "./.grn_cache"))
    args = parser.parse_args()
    verify_exhaustiveness(args.jsonl, args.models, args.cache)
