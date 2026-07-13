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

"""Utilities for generating ambiguous questions by holding out 1 word."""

import copy
import itertools as it
import collections
import json

import tqdm

from SimpleLogic import derivation_new
from SimpleLogic import ruleset
from SimpleLogic.horn_sat_utils import parse_clauses, facts_to_assignment, solve_unit_prop, forced_value_via_refutation, CONTRADICTION, INFEASIBLE, negate_lit


# def check_sufficiency(rule_tree, context, vars_to_check, target):
#   """Checks if vars_to_check are sufficient to determine target given context.

#   Args:
#     rule_tree: RuleTree
#     context: Set[str] of facts (e.g. {"a", "not b"})
#     vars_to_check: Set[str] of variable names (e.g. {"c", "d"})
#     target: str target variable name

#   Returns:
#     bool: True if for all consistent assignments of vars_to_check, the target's
#       truth value is determined. False otherwise.
#   """
#   var_list = list(vars_to_check)
#   num_consistent_assignments = 0
  
#   # Iterate over all possible truth value assignments for vars_to_check
#   for values in it.product([True, False], repeat=len(var_list)):
#     assignment_facts = set()
#     for i, val in enumerate(values):
#       if val:
#         assignment_facts.add(var_list[i])
#       else:
#         assignment_facts.add(ruleset.negate(var_list[i]))

#     full_facts = context.union(assignment_facts)

#     # Split into true/false for get_all_inferrable_facts
#     true_facts = {f for f in full_facts if not f.startswith("not ")}
#     false_facts = {f for f in full_facts if f.startswith("not ")}

#     # Check for immediate contradiction in the input facts
#     contradiction = False
#     for f in true_facts:
#       if ruleset.negate(f) in full_facts:
#         contradiction = True
#         break
#     if contradiction:
#       continue  # Vacuously true if the assignment is impossible

#     inferred = derivation_new.get_all_inferrable_facts(rule_tree, true_facts, false_facts)

#     # Check for consistency in inferred facts
#     for f in inferred:
#       if ruleset.negate(f) in inferred:
#         contradiction = True
#         break
#     if contradiction:
#       continue
    
#     num_consistent_assignments += 1

#     # If consistent, check if target is determined
#     if target not in inferred and ruleset.negate(target) not in inferred:
#       return False

#   return num_consistent_assignments > 0


def check_sufficiency(rule_tree, context, vars_to_check, target):
  """Checks if vars_to_check are sufficient to determine target given context.

  Semantics (Horn):
    - An assignment branch is *feasible* iff (rules ∧ context ∧ assignment) is SAT,
      detected by unit propagation + empty-clause contradiction.
    - target is *known* on a feasible branch iff it is forced (entailed) to True/False,
      checked by Horn refutation: UNSAT of (branch ∧ target=value) via UP.

  Returns:
    True iff (i) there exists at least one feasible branch and
             (ii) on every feasible branch, target is forced to a value.
  """
  clauses = parse_clauses(rule_tree)
  var_list = list(vars_to_check)
  num_feasible = 0

  for values in it.product([True, False], repeat=len(var_list)):
    assignment_facts = set()
    for i, val in enumerate(values):
      assignment_facts.add(var_list[i] if val else negate_lit(var_list[i]))

    full_facts = context.union(assignment_facts)

    base = facts_to_assignment(full_facts)
    if base == CONTRADICTION:
      continue

    # Feasibility check (catches falsified Horn constraints like ['not a','not b'])
    base_up = solve_unit_prop(clauses, base)
    if base_up == CONTRADICTION:
      continue

    num_feasible += 1

    forced = forced_value_via_refutation(clauses, base_up, target)
    if forced in (None, INFEASIBLE):
      return False

  return num_feasible > 0


def make_heldout_ruleset(rules_dict, max_k=4):
  """Hold out k words required for deriving the target word's truth value.

  Keeps only those derivations that are not already derived by the full set.
  Adds field to rules_dict called `heldout_set_to_q` and
  `heldout_set_to_subset_qs`
  `heldout_set_to_q` maps a heldout set to a dict of the form
  {1-sufficient set: sufficient word: {true_derivation, false_derivation}}
  where 1-sufficient set is missing 1 word that is required for deriving the
  target word's truth value, and sufficient word is a word that when known, can
  derive the target word's truth value. The true/false derivations are the
  derivations of the target word's truth value after adding the sufficient word
  to the heldout set.
  `heldout_set_to_subset_qs` maps a 1-sufficient set to a list of heldout sets
  that are subsets of the heldout set, but not the heldout set itself.

  Also generates k-sufficient sets for k=2 to max_k.
  
  Args:
    rules_dict: Dict[str, Any]
    max_k: int, max size of heldout set to search for (default 1, supports up to 3)
  """
  assert max_k in [1, 2, 3, 4], "k_max must be 1, 2, 3, 4."
  assert "not " not in rules_dict["query"]
  
  print("Generating k=1")
  # UNCHANGED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  # NOTE: Using leafs as keys is fine because in generating the derivations we do BFS and only keep the most shallow derivation for the same leaf set
  true_derivations = {}
  for derive in rules_dict['true_derivations']:
    if not isinstance(derive, derivation_new.ConjunctionRule):
      derive = derivation_new.ConjunctionRule.deserialize(derive)
    true_derivations[frozenset(derive.leaf_words.keys())] = derive
  false_derivations = {}
  for derive in rules_dict['false_derivations']:
    if not isinstance(derive, derivation_new.ConjunctionRule):
      derive = derivation_new.ConjunctionRule.deserialize(derive)
    false_derivations[frozenset(derive.leaf_words.keys())] = derive

  heldout_set_to_q_depth_expansions = {}
  # Pre-calculate RuleTree for inference checks
  # rule_tree = ruleset.RuleTree.deserialize(rules_dict["rules"])
  
  # NOTE: This is actually a loose check -- it finds any (true, false) derivation pairs that shares exactly one variable with opposite signs (does not consider variables that are only assigned in one of the derivations). Though, because of the way derivations are generated (BFS, minimal leaf sets), this must cover all correct pairs. 
  # get combined version (cross product)
  for true_derive, false_derive in tqdm.tqdm(
      it.product(true_derivations, false_derivations),
      total=len(true_derivations) * len(false_derivations),
  ):
    # find pairs that differ on exactly 1 var
    differ_word = None
    for k in true_derive:
      if ruleset.negate(k) in false_derive:
        if differ_word is None:
          differ_word = k
        elif differ_word != k:
          differ_word = None
          break
    if differ_word is None:
      continue

    heldout_set = true_derive.union(false_derive)
    heldout_set -= {differ_word, ruleset.negate(differ_word)}
    # NOTE: heldout_set = context (Assigned); here it is only potentially valid
    
    # can already derive heldout set
    # NOTE: Check if Context ALREADY implies the answer (0-sufficient) -- This step should be complete
    skip_this = False
    if heldout_set in true_derivations or heldout_set in false_derivations:
      skip_this = True
    else:
      for true_derivation in true_derivations:
        if true_derivation <= heldout_set:
          skip_this = True
          break
      for false_derivation in false_derivations:
        if false_derivation <= heldout_set:
          skip_this = True
          break
    if skip_this:
      continue
    
    if heldout_set not in heldout_set_to_q_depth_expansions:
      heldout_set_to_q_depth_expansions[heldout_set] = {}
      
    # if differ_word == "anxious" and heldout_set == frozenset({'different', 'bad-tempered', 'hypocritical', 'strange', 'cooperative', 'distinct'}):
    #   raise ValueError("Debugging")
      
    # add variable with direction that implies query word is true
    heldout_set_to_q_depth_expansions[heldout_set][differ_word] = {
        'true_derivation': true_derivations[true_derive].serialize(),
        'false_derivation': false_derivations[false_derive].serialize(),
    }

  rules_dict['heldout_set_to_q'] = {
      json.dumps(list(heldout_set)): heldout_set_to_q_depth_expansions[
          heldout_set
      ]
      for heldout_set in heldout_set_to_q_depth_expansions
  }

  # NOTE: Collects all 1-sufficient variables from smaller contexts. If a smaller context {a, b} can be solved by asking x, then the larger context {a, b, c} can also be solved by asking x. So x is a valid question for context {a, b, c}, but it's too easy (didn't need to know c at all). These collected variables will be used as "invalid questions" to forbid the LLM from asking. 
  # valid questions of subset --> also valid questions here,
  # but we want to avoid asking
  heldout_set_to_subset_qs = {}
  for heldout_set in heldout_set_to_q_depth_expansions:
    heldout_set_to_subset_qs[heldout_set] = set()
    for other_set in heldout_set_to_q_depth_expansions:
      if heldout_set == other_set:
        continue
      # check if other_set is a subset of heldout_Set
      if other_set < heldout_set:
        heldout_set_to_subset_qs[heldout_set] = heldout_set_to_subset_qs[
            heldout_set
        ].union(set(heldout_set_to_q_depth_expansions[other_set].keys()))
    # remove target_attr from set
    if rules_dict['query'] in heldout_set_to_subset_qs[heldout_set]:
      heldout_set_to_subset_qs[heldout_set].remove(rules_dict['query'])
  rules_dict['heldout_set_to_subset_qs'] = {
      json.dumps(list(heldout_set)): list(heldout_set_to_subset_qs[heldout_set])
      for heldout_set in heldout_set_to_subset_qs
  }
  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  
  # --- Extension for k-sufficient sets ---
  # Initialize with 1-sufficient sets found above
  k_sufficient_sets = {1: []}
  for heldout_set, val in heldout_set_to_q_depth_expansions.items():
    for word, derivations_for_word in val.items():
      # var_name = word.split("not ")[1] if wor7d.startswith("not ") else word
      assert "not " not in word
      unified_derivations = {
          (word, ): {
              "target_value": rules_dict["query"],
              "derivation": derivations_for_word['true_derivation'],
          },
          (f"not {word}",): {
              'target_value': ruleset.negate(rules_dict["query"]),
              'derivation': derivations_for_word['false_derivation'],
          },
      }
      k_sufficient_sets[1].append((heldout_set, {word}, unified_derivations, unified_derivations))  # heldout_set contains word-values (i.e., can contain "not "), while 

  rule_tree_obj = rules_dict['rules']
  if not isinstance(rule_tree_obj, ruleset.RuleTree):
    # Should typically be RuleTree if called within pipeline, but ensure safety
    rule_tree_obj = ruleset.RuleTree.deserialize(rules_dict['rules'])

  # Generate k > 1 minimal sufficient sets recursively from k-1 sets
  for k in range(2, max_k + 1):
    print(f"Generating k={k}")
    temp_k_sets = []
    seen_sigs = set()
    prev_items = k_sufficient_sets[k - 1]
        
    # Pre-group the previous sets by context to speed up the global minimality check
    # context (frozenset) -> list of s_sets (set)
    context_to_s_sets = collections.defaultdict(list)
    for context_prev, s_prev, _, _ in prev_items:
        context_to_s_sets[context_prev].append(s_prev)
        
    # Pre-collect all lower level contexts for global minimality
    lower_k_contexts = set()
    for j in range(1, k):
        for context_prev, _, _, _ in k_sufficient_sets[j]:
            lower_k_contexts.add(context_prev)

    for context_prev, s_prev, _, _ in tqdm.tqdm(prev_items, desc=f"Generating k={k}"):
      # Iterate through each fact in the previous context
      for fact in list(context_prev):
        var_name = fact.split("not ")[1] if fact.startswith("not ") else fact

        # Construct candidate next sets
        context_k = context_prev - {fact}
        s_k = s_prev | {var_name}

        # Avoid processing same (context, s_set) pair multiple times
        sig = (frozenset(context_k), frozenset(s_k))
        if sig in seen_sigs:
          continue
        seen_sigs.add(sig)

        # Condition 1: A_k + Known(S_{k-1}) + flipped_fact => Known(y)
        # s_prev corresponds to S_{k-1}
        flipped_fact = ruleset.negate(fact)
        if not check_sufficiency(
            rule_tree_obj,
            context_k | {flipped_fact},
            s_prev,
            rules_dict['query'],
        ):
          continue
        
        unified_derivations_min_depth = {}
        unified_derivations_min_rules = {}

        # Condition 2 (Global Minimality / Underspecification):
        # We need to ensure that NO set of size k-1 determines y under the new reduced context A_k.
        is_lower_k_sufficient = False
        
        # Check A: Subset check
        # If any subset of context_k is a known minimal context for size k-1 (or less),
        # then context_k is solvable by size k-1 (monotonicity).
        for c_existing in context_to_s_sets:
            if c_existing <= context_k:
                is_lower_k_sufficient = True
                break
        
        if is_lower_k_sufficient:
            continue

        # Check B: Parent Context Check
        # Check all sufficient sets of the parent context (size k-1)
        # A sufficient set that solves context_k should also solve context_prev
        for s_other in context_to_s_sets[context_prev]:
            # If s_other solves the problem under context_k, then A_k 
            # is solvable with k-1 vars, so S_k (size k) is not minimal.
            if check_sufficiency(
                rule_tree_obj, context_k, s_other, rules_dict['query']
            ):
                is_lower_k_sufficient = True
                break
        
        if is_lower_k_sufficient:
            continue
          
        # Check C: All subsets of search_space_vars of size k-1
        # Ensure no subset of search_space_vars of size k-1 is sufficient
    
        # Determine invalid questions
        # 1. Goal
        invalid_qs = {rules_dict['query']}
        # 2. Already Known Vars
        known_vars = {f.split("not ")[-1] for f in context_k}
        invalid_qs.update(known_vars)
        
        # Defining subset search space
        all_qs = set(q.split("not ")[-1] for q in rule_tree_obj.nodes.keys())
        search_space_vars = sorted(list(all_qs - invalid_qs))
        
        found_smaller = False
        # Brute force check all subsets of size k-1
        for subset in it.combinations(search_space_vars, k-1):
            if check_sufficiency(
                rule_tree_obj, 
                context_k, 
                set(subset), 
                rules_dict['query']
            ):
                found_smaller = True
                break
        if found_smaller:
            continue

        # Finally, generate derivations for all 2^k assignments of s_k
        unified_derivations_min_depth = {}
        unified_derivations_min_rules = {}
        s_k_list = list(s_k)
        
        not_found_count = 0
        for values in it.product([True, False], repeat=len(s_k_list)):
          assignment_facts = []
          for i, val in enumerate(values):
            if val:
              assignment_facts.append(s_k_list[i])
            else:
              assignment_facts.append(ruleset.negate(s_k_list[i]))
          
          assignment_key = tuple(assignment_facts)
          full_context = context_k | set(assignment_facts)
          
          # Determine whether this full assignment is consistent and what value it gives the target.
          true_facts = {f for f in full_context if not f.startswith("not ")}
          false_facts = {f for f in full_context if f.startswith("not ")}

          # Check for immediate contradiction in input facts
          is_contradictory = False
          for f in true_facts:
            if ruleset.negate(f) in full_context:
              is_contradictory = True
              break

          inferred = set()
          if not is_contradictory:
            inferred = derivation_new.get_all_inferrable_facts(rule_tree_obj, true_facts, false_facts)

            # Check for contradiction via inference
            for f in inferred:
              if ruleset.negate(f) in inferred:
                is_contradictory = True
                break

          if is_contradictory:
            # This assignment is impossible; keep a placeholder so downstream code can skip it.
            unified_derivations_min_rules[assignment_key] = {
              'target_value': None,
              'derivation': None,
            }
            unified_derivations_min_depth[assignment_key] = {
              'target_value': None,
              'derivation': None,
            }
            not_found_count += 1
            continue
          
          # Target must be determined under a consistent assignment (k-sufficiency).
          if rules_dict["query"] in inferred:
            target_literal = rules_dict["query"]
            deriv_pool = true_derivations
          elif ruleset.negate(rules_dict["query"]) in inferred:
            target_literal = ruleset.negate(rules_dict["query"])
            deriv_pool = false_derivations
          else:
            raise ValueError(
              f"Assignment {assignment_key} under context {context_k} is consistent "
              f"but does not determine the target. This indicates s_k={s_k} is not truly sufficient."
            )
          
          # Find matching (minimal) derivation from true_derivations or false_derivations
          found = False
          min_num_rules = None
          min_depth = None
          for derive_set, derive_obj in deriv_pool.items():
            if derive_set <= full_context:
              found = True
              num_rules = len(derive_obj.derivation)
              depth = max(derive_obj.leaf_words.values()) if derive_obj.leaf_words else 0

              if min_num_rules is None or num_rules < min_num_rules:
                unified_derivations_min_rules[assignment_key] = {
                  'target_value': target_literal,
                  'derivation': derive_obj.serialize(),
                }
                min_num_rules = num_rules

              if min_depth is None or depth < min_depth:
                unified_derivations_min_depth[assignment_key] = {
                  'target_value': target_literal,
                  'derivation': derive_obj.serialize(),
                }
                min_depth = depth

          if not found:
            raise ValueError(
              f"Assignment {assignment_key} under context {context_k} determines {target_literal} "
              f"but no stored derivation matches. This indicates the derivation pool is incomplete."
            )

        if not_found_count >= 2 ** len(s_k) - len(s_k):
          # NOTE: We need at least k+1 valid derivations to ensure k is minimal sufficient
          raise ValueError(f"Insufficient derivations found for assignments in k-sufficient set {s_k} under context {context_k}.")
        
        # NOTE: This set is to ensure sets at the same level won't be used for this check.
        temp_k_sets.append((context_k, s_k, unified_derivations_min_rules, unified_derivations_min_depth))

    k_sufficient_sets[k] = temp_k_sets
 
  # Store k-sufficient sets in the rules_dict (generalization of heldout_set_to_q (heldout_set_to_q_depth_expansions))
  rules_dict['heldout_k_sets'] = {}
  for k in range(1, max_k + 1):
    res_map = {}
    for context, s_set, derivations_min_rules, derivations_min_depth in k_sufficient_sets[k]:
      c_str = json.dumps(list(context))
      if c_str not in res_map:
        res_map[c_str] = []
      # convert tuple keys to strings for JSON serialization
      serializable_derivations_min_rules = {json.dumps(list(k)): v for k, v in derivations_min_rules.items()}
      serializable_derivations_min_depth = {json.dumps(list(k)): v for k, v in derivations_min_depth.items()}
      res_map[c_str].append({
        "s_set": list(s_set),
        "derivations_min_rules": serializable_derivations_min_rules,
        "derivations_min_depth": serializable_derivations_min_depth,
      })
    rules_dict['heldout_k_sets'][str(k)] = res_map

  # Store invalid questions for each k (generalization of heldout_set_to_subset_qs)
  # For each context at level k, collect k-sufficient sets from SMALLER contexts (i.e., subsets of this context; same logic as ). These are "too easy" because they don't require all the information in the current context.
  #
  # Structure: context_to_invalid_sets[k][context_str] = list of s_sets
  
  rules_dict['context_to_invalid_sets'] = {}
  
  for k in range(1, max_k + 1):
    rules_dict['context_to_invalid_sets'][str(k)] = {}
    
    # Build a mapping: context -> list of s_sets for this k level
    context_to_s_sets_at_k = collections.defaultdict(list)
    for context, s_set, _, _ in k_sufficient_sets[k]:
      context_to_s_sets_at_k[context].append(s_set)
    
    # For each context at this k level, find invalid sets from smaller contexts
    for context, s_set, _, _ in k_sufficient_sets[k]:
      c_str = json.dumps(list(context))
      
      if c_str not in rules_dict['context_to_invalid_sets'][str(k)]:
        invalid_sets = set()  # Use set of frozensets for deduplication
        
        # Check all other contexts at the same k level
        for other_context in context_to_s_sets_at_k:
          if other_context == context:
            continue
          # If other_context is a strict subset of context
          if other_context < context:
            # All k-sufficient sets for the smaller context are "too easy"
            for other_s_set in context_to_s_sets_at_k[other_context]:
              invalid_sets.add(frozenset(other_s_set))
        
        # Convert to list of lists for JSON serialization
        # Also remove the query variable from each set if present
        invalid_sets_list = []
        for s in invalid_sets:
          s_cleaned = set(s)
          if rules_dict['query'] in s_cleaned: # Remove the set containing query variable if present
            continue
          if s_cleaned:  # Only add non-empty sets
            invalid_sets_list.append(list(s_cleaned))
        
        rules_dict['context_to_invalid_sets'][str(k)][c_str] = invalid_sets_list
