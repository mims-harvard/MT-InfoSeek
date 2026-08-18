# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Format prompts and data for Logic-Q task."""

import argparse
import glob
import json
import orjson
import os
import random

import pandas as pd
from SimpleLogic import ruleset
import tqdm
import itertools as it
from SimpleLogic.horn_sat_utils import parse_clauses, infer_closure, forced_value_from_facts

DATA_DIR = os.environ.get("DATA_DIR", "")

tqdm = tqdm.tqdm
random.seed(42)


def _truth_table_for_qset(rule_tree, base_context: set[str], q_set: list[str], goal: str):
  """Enumerate all assignments to q_set and return a table over consistent rows.

  Returns:
    table: dict[tuple[bool,...], bool] mapping answer tuples -> goal truth value
    consistent_keys: set[str] assignment_key_json for consistent rows
    expected_target_value: dict[str, str] assignment_key_json -> (goal or not goal)
  """
  table = {}
  consistent_keys = set()
  expected_target_value = {}
  
  for answers in it.product([True, False], repeat=len(q_set)):
    assignment_facts = [q if val else ruleset.negate(q) for q, val in zip(q_set, answers)]
    full_facts = set(base_context) | set(assignment_facts)
    is_contra, inferred = infer_closure(rule_tree, full_facts)
    if is_contra:
      continue
    
    forced = forced_value_from_facts(rule_tree, full_facts, goal)
    if forced == "INFEASIBLE":
      continue
    if forced is None:
      return None, None, None  # insufficient under semantic Known(goal)
    y_val = forced
    tgt_lit = goal if y_val else ruleset.negate(goal)
        
    table[answers] = y_val
    key_json = json.dumps(assignment_facts)
    consistent_keys.add(key_json)
    expected_target_value[key_json] = tgt_lit

  assert table, "No consistent assignments found for this q_set."
  
  return table, consistent_keys, expected_target_value


def _all_vars_essential(table: dict, q_set: list[str]) -> bool:
  """Essentiality check on a truth table over consistent rows only."""
  k = len(q_set)
  if k <= 1:
    # For k=0 it's invalid; for k=1 minimality is handled by the context-unknown check.
    return k == 1

  # For each variable position i, look for a witness pair that flips y
  for i in range(k):
    essential = False
    # iterate over assignments to other k-1 vars
    for others in it.product([True, False], repeat=k-1):
      # build the two full tuples differing only at i
      t0 = []
      t1 = []
      j = 0
      for p in range(k):
        if p == i:
          t0.append(False)
          t1.append(True)
        else:
          t0.append(others[j])
          t1.append(others[j])
          j += 1
      t0 = tuple(t0)
      t1 = tuple(t1)

      if t0 in table and t1 in table and table[t0] != table[t1]:
        essential = True
        break

    if not essential:
      return False

  return True


def validate_and_filter_problem(rule_tree, base_context: set[str], q_set: list[str], goal: str,
                                derivs_min_rules: dict, derivs_min_depth: dict) -> tuple[bool, dict, dict]:
  """Applies contradiction filtering + sufficiency + essentiality checks.

  Returns:
    (is_valid, filtered_derivs_min_rules, filtered_derivs_min_depth)
  """
  table, consistent_keys, expected_target_value = _truth_table_for_qset(rule_tree, base_context, q_set, goal)
  if table is None:
    return False, {}, {}

  if not _all_vars_essential(table, q_set):
    return False, {}, {}

  # Filter derivation dicts to consistent rows and validate target_value matches inference
  filtered_rules = {}
  filtered_depth = {}

  # NOTE: derivation dict keys are JSON-encoded lists of facts.
  for k_json in list(derivs_min_rules.keys()):  # k_json is assignment key
    if k_json not in consistent_keys:
      continue
    v_rules = derivs_min_rules.get(k_json)
    v_depth = derivs_min_depth.get(k_json)
    if v_rules is None or v_depth is None:
      raise Exception(f"Assignment {k_json} is missing from derivs_min_rules or derivs_min_depth.")
    if v_rules["target_value"] is None:
      assert v_depth["target_value"] is None
      raise Exception(f"Assignment {k_json} has target_value=None but was accepted as consistent.")
    if v_rules.get("target_value") != expected_target_value[k_json]:
      raise Exception("Mismatched target value in min-rules derivation.")
    if v_depth.get("target_value") != expected_target_value[k_json]:
      raise Exception("Mismatched target value in min-depth derivation.")
    filtered_rules[k_json] = v_rules
    filtered_depth[k_json] = v_depth

  # Ensure we have derivations for every consistent assignment
  if set(filtered_rules.keys()) != consistent_keys or set(filtered_depth.keys()) != consistent_keys:
    raise Exception("Filtered derivations do not cover all consistent assignments.")

  return True, filtered_rules, filtered_depth


def _is_globally_minimal(rule_tree, base_context, q_set, goal, all_valid_qs):
    """
    Verifies that NO subset of size len(q_set)-1 (from the entire valid universe)
    is sufficient to determine the goal.
    """
    k = len(q_set)
    if k <= 1:
        return True # k=1 is always globally minimal if the context alone is insufficient

    full_space = [q for q in all_valid_qs if q != goal]
    
    for subset in it.combinations(full_space, k - 1):
        # Quick check: if this subset is a subset of q_set, we already checked it in local essentiality
        if set(subset).issubset(set(q_set)):
            continue
            
        subset_list = list(subset)
        table, _, _ = _truth_table_for_qset(rule_tree, base_context, subset_list, goal)
        
        if table is not None:
            # _truth_table_for_qset returns None when the goal is undetermined, so a
            # non-None table means this smaller subset is already sufficient.
            return False
             
    return True


def main(arguments) -> None:
  # Load constructed rulesets
  rulesets = []
  for item_file in glob.glob(
      os.path.join(arguments.sl_dir, "*_heldout_fixed_new.jsonl")
  ):
    print(item_file)
    import gc
    gc.disable()
    
    with open(item_file, "rb") as f:
      for line in tqdm(f):
        try:
          rulesets.append(orjson.loads(line))
        except orjson.JSONDecodeError:
          continue
    
    gc.enable()

  # Create dataframe
  data = pd.DataFrame(
      columns=[
          "k",
          "known_facts",
          "known_untrue_facts",
          "cannot_ask_facts",
          "cannot_ask_facts_sets",
          "goal",
          "rules",
          "max_depth",
          "min_num_rules_needed",
          "num_constraints",
          "num_vars",
          "all_qs",
          "all_valid_qs",
          "gt_qs",
          # "gt_q_to_true_derivation",
          # "gt_q_to_false_derivation",
          "gt_q_to_derivations_min_rules",
          "gt_q_to_derivations_min_depth",
      ]
  )

  print(f"Processing {len(rulesets)} rulesets...")

  for rs in tqdm(rulesets):
    if "true_derivations" not in rs: 
        continue
        
    rule_tree = ruleset.RuleTree.deserialize(rs["rules"])
    clauses = parse_clauses(rule_tree.serialize())
    target_attr = rs["query"]
    
    k_data_source = {}
    if "heldout_k_sets" in rs and rs["heldout_k_sets"]:
      # Use the new structure: {"1": {ctx: [[v]]}, "2": {ctx: [[v1,v2]]}}
      k_data_source = rs["heldout_k_sets"]
    else:
      raise Exception

    # 2. Flatten all problems in this ruleset for subsampling
    # tuple: (k_int, context_str, valid_sets_list, derivations_min_rules_list, derivations_min_depth_list)
    all_problems = []
    for k_str, context_map in k_data_source.items():
      for context_str, dict_lst in context_map.items():
        all_problems.append((int(k_str), context_str, [dct["s_set"] for dct in dict_lst], [dct["derivations_min_rules"] for dct in dict_lst], [dct["derivations_min_depth"] for dct in dict_lst]))

    # 3. Process each problem
    filtered_rows = []
    for k, context_str, valid_sets, derivations_min_rules, derivations_min_depth in all_problems:
      # Parse context
      context_set = set(json.loads(context_str))

      # Determine facts known to be false based on "not X" in context
      false_facts = []
      for f in context_set:
        if f.startswith("not "):
            false_facts.append(f.split("not ")[1])
      false_facts = sorted(false_facts)
      true_facts = sorted([f for f in context_set if not f.startswith("not ")])
      
      # invalid questions and question sets
      # 1. Cannot ask the goal itself
      invalid_qs = {target_attr}
      # 2. Cannot ask about variables already in the known facts (redundant)
      known_vars = {f.split("not ")[-1] for f in context_set}
      invalid_qs.update(known_vars)
      
      invalid_q_sets = rs["context_to_invalid_sets"][str(k)].get(context_str, list())
      invalid_q_sets = {frozenset(q) for q in invalid_q_sets}
      
      # All variables in the tree (potential search space)
      # Get all variables that exist in both positive and negative forms
      all_vars = set()
      for q in rule_tree.nodes.keys():
        if q.startswith("not "):
          all_vars.add(q[4:])  # Extract variable name from "not X"
        else:
          all_vars.add(q)
      
      # # Only include variables that have both positive and negative forms
      #     all_qs_atoms.add(var)
      
      # Valid individual variables available for selection (the atoms)
      valid_qs_atoms = sorted(list(
          all_vars - set(false_facts) - set(true_facts) - set([target_attr])
      ))
      
      # filter out sets that accidentally are identical to invalid sets.
      clean_gt_qs = []
      clean_derivations_min_rules = []
      clean_derivations_min_depth = []
      num_rules_to_compute_q = []
      max_depth_to_compute_q = []
      for q_set, derivs_min_rules, derivs_min_depth in zip(valid_sets, derivations_min_rules, derivations_min_depth):
          if (not frozenset(q_set) in invalid_q_sets) and (not set(q_set).intersection(invalid_qs)):
              ok, filtered_rules, filtered_depth = validate_and_filter_problem(
                  clauses, context_set, list(q_set), target_attr, derivs_min_rules, derivs_min_depth
              )
              if not ok:
                  continue
              # Validate Global Minimality
              if len(q_set) > 1:
                  if not _is_globally_minimal(clauses, context_set, list(q_set), target_attr, valid_qs_atoms):
                      continue
              clean_gt_qs.append(sorted(q_set))
              clean_derivations_min_rules.append(filtered_rules)
              clean_derivations_min_depth.append(filtered_depth)
              num_rules_to_compute_q.append(max(
                  len(deriv["derivation"]["derivation"]) for deriv in filtered_rules.values()
              ))
              max_depth_to_compute_q.append(max(
                  max(deriv["derivation"]["leaf_words"].values()) for deriv in filtered_depth.values()
              ))
      
      if not clean_gt_qs:
          continue
      
      num_rules = min(num_rules_to_compute_q)
      depth = min(max_depth_to_compute_q)
      num_total_rules = rule_tree.num_rules()
      num_words = rule_tree.num_words()

      filtered_rows.append((
          k,
          true_facts,
          false_facts,
          sorted(list(invalid_qs)),
          [sorted(q_set) for q_set in invalid_q_sets],
          target_attr,
          rule_tree.serialize(),
          depth,
          num_rules,
          num_total_rules,
          num_words,
          sorted(list(all_vars)),
          valid_qs_atoms,
          clean_gt_qs,
          clean_derivations_min_rules,
          clean_derivations_min_depth,
      ))

    # 4. Subsample problems per ruleset (avoid bias from individual problems)
    # Sample max_problems_to_sample_per_ruleset problems per k value
    rows_by_k = {}
    for row in filtered_rows:
        k_val = row[0]
        rows_by_k.setdefault(k_val, []).append(row)

    sampled_rows = []
    for k_val, rows in rows_by_k.items():
        if len(rows) > arguments.max_problems_to_sample_per_ruleset:
            sampled_rows.extend(random.sample(rows, arguments.max_problems_to_sample_per_ruleset))
        else:
            sampled_rows.extend(rows)
    
    # Append to dataframe
    for row in sampled_rows:
        data.loc[len(data)] = list(row)
  
  # Split into prompts and data
  if len(data) > 0:
      # Sample 25 prompts for few-shot usage
      prompt_indices = random.sample(range(len(data)), min(25, len(data)))
      prompts = data.iloc[prompt_indices]
      data_subsample = data.iloc[list(set(range(len(data))) - set(prompt_indices))]
      
      prompt_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_prompts_new.csv")
      data_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_data_new.csv")
      
      print(f"Writing {len(prompts)} prompts to {prompt_path}")
      with open(prompt_path, "w") as f:
        prompts.to_csv(f, index=False)
        
      print(f"Writing {len(data_subsample)} rows to {data_path}")
      with open(data_path, "w") as f:
        data_subsample.to_csv(f, index=False)
  else:
      print("No valid data rows generated. Check input files or sampling logic.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--sl_dir",
      default=os.environ.get("LOGIC_Q_RP_DIR", "/path/to/questbench_data/Logic-Q/RP/RP"),
      help="Directory containing the SimpleLogic data.",
  )
  parser.add_argument(
      "--max_problems_to_sample_per_ruleset",
      default=50,
      type=int,
      help="Maximum number of problems to sample per ruleset.",
  )
  main(parser.parse_args())
