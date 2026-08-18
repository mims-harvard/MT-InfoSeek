import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import boolean_network_utils as bnu

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_tasks(jsonl_path, models_dir, cache_dir):
    jsonl_path = Path(jsonl_path)
    models_dir = Path(models_dir)
    cache_dir = Path(cache_dir)
    
    with open(jsonl_path, "r") as f:
        tasks = [json.loads(line) for line in f]
        
    logging.info(f"Loaded {len(tasks)} tasks to verify.")
    
    models_list = bnu.load_models_with_text(str(models_dir))
    models_dict = {m['model']: m for m in models_list}
    
    cache_objs = {}
    
    num_failures = 0
    
    for row in tqdm(tasks):
        model_name = row['model']
        if model_name not in models_dict:
            logging.error(f"Model {model_name} not found.")
            num_failures += 1
            continue
            
        if model_name not in cache_objs:
            cache_objs[model_name] = bnu.compute_or_load_model_cache(
                models_dict[model_name],
                str(cache_dir),
                max_attractors=1000
            )
            
        cache = cache_objs[model_name]
        n = cache.n_nodes
        observed = row['observed']
        observed_idx_vals = [(cache.var_names.index(k), v) for k, v in observed]
        
        target_type = row['target_type']
        family = row['family']
        
        # 1. Underspecification (y is not determined with only observed)
        # 2. Sufficiency
        # 3. Minimality
        
        minimal_sets_idx = row['minimal_sufficient_sets_idx']
        k_min = row['k_min']
        
        if family == "GeneReg-SS":
            # states are fixed points
            omega = np.array(cache.fixed_points, dtype=np.uint64)
            # Find candidate states matching observed
            cand_idx = []
            for i, st in enumerate(omega):
                match = True
                for idx, v in observed_idx_vals:
                    if bnu.bit(int(st), idx) != v:
                        match = False
                        break
                if match:
                    cand_idx.append(i)
            
            cand_states = omega[cand_idx]
            
            if target_type == "attractor_id":
                y = np.array(cand_idx, dtype=np.int32)
            else: # ss_marker
                marker_idx = row['marker_gene_idx']
                y = np.array([bnu.bit(int(st), marker_idx) for st in cand_states], dtype=np.int32)
                
            # Check underspecification
            if len(np.unique(y)) <= 1:
                logging.error(f"Row {row['group_id']} is not underspecified! Unique y: {np.unique(y)}")
                num_failures += 1
                continue
                
            # Check sufficiency and minimality of each given minimal set
            valid = True
            for s_idx in minimal_sets_idx:
                if len(s_idx) != k_min:
                    logging.error(f"Row {row['group_id']} has set {s_idx} of length {len(s_idx)} != {k_min}")
                    valid = False
                    
                is_suff = bnu._is_y_function_of_subset_states(cand_states, y, s_idx)
                if not is_suff:
                    logging.error(f"Row {row['group_id']} set {s_idx} is NOT sufficient.")
                    valid = False
                    
                # check minimality
                for i in range(len(s_idx)):
                    sub_idx = s_idx[:i] + s_idx[i+1:]
                    if bnu._is_y_function_of_subset_states(cand_states, y, sub_idx):
                        logging.error(f"Row {row['group_id']} set {s_idx} is NOT minimal. Subset {sub_idx} is sufficient.")
                        valid = False
            
            if not valid:
                num_failures += 1
                
        else:
            # GeneReg-Dyn
            mask, val = bnu.build_mask_value_from_idx_vals(observed_idx_vals)
            marker_idx = row.get('marker_gene_idx')
            
            # Check underspecification
            is_suff = bnu._is_y_function_of_subset_basin(
                n=n,
                basin_map=cache.landscape.basin_map,
                context_mask=mask,
                context_value=val,
                subset=[],
                target_type=target_type,
                marker_idx=marker_idx,
                attractor_rep_states=bnu.attractor_representative_states(cache.landscape)
            )
            
            if is_suff:
                logging.error(f"Row {row['group_id']} is not underspecified! Target is already determined by context.")
                num_failures += 1
                continue
                
            valid = True
            for s_idx in minimal_sets_idx:
                if len(s_idx) != k_min:
                    logging.error(f"Row {row['group_id']} has set {s_idx} of length {len(s_idx)} != {k_min}")
                    valid = False
                    
                is_suff = bnu._is_y_function_of_subset_basin(
                    n=n,
                    basin_map=cache.landscape.basin_map,
                    context_mask=mask,
                    context_value=val,
                    subset=s_idx,
                    target_type=target_type,
                    marker_idx=marker_idx,
                    attractor_rep_states=bnu.attractor_representative_states(cache.landscape)
                )
                if not is_suff:
                    logging.error(f"Row {row['group_id']} set {s_idx} is NOT sufficient.")
                    valid = False
                    
                # check minimality
                for i in range(len(s_idx)):
                    sub_idx = s_idx[:i] + s_idx[i+1:]
                    if bnu._is_y_function_of_subset_basin(
                        n=n,
                        basin_map=cache.landscape.basin_map,
                        context_mask=mask,
                        context_value=val,
                        subset=sub_idx,
                        target_type=target_type,
                        marker_idx=marker_idx,
                        attractor_rep_states=bnu.attractor_representative_states(cache.landscape)
                    ):
                        logging.error(f"Row {row['group_id']} set {s_idx} is NOT minimal. Subset {sub_idx} is sufficient.")
                        valid = False
            
            if not valid:
                num_failures += 1
                
    if num_failures == 0:
        logging.info("All tasks verified successfully.")
    else:
        logging.error(f"{num_failures} tasks failed verification.")

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=os.path.join(os.environ.get("DATA_DIR", "data"), "genereg_mt.jsonl"))
    parser.add_argument("--models", default=os.environ.get("MODELS_DIR", "/path/to/DesignPrinciplesGeneNetworks/update_rules_122_models_Kadelka_SciAdv"))
    parser.add_argument("--cache", default=os.environ.get("CACHE_DIR", "./.grn_cache"))
    args = parser.parse_args()

    verify_tasks(args.jsonl, args.models, args.cache)
