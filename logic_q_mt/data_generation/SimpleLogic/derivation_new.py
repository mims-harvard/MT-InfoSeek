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

"""Computing derivations of target words.

This module provides functions for deriving all possible ways of implying a
target word is true or false, given a rules_dict.
Derivation is the process of finding all possible ways to assign truth values
to the variables in a rules_dict such that the target word becomes true
or false.
"""

import glob
import itertools as it
import json
import copy
from typing import Any, Dict, Iterable, List, Set, Tuple
from time import time

from SimpleLogic import ruleset


def union(all_dicts: Iterable[Dict[str, Any]]):
  """Take union of all inner lists."""
  union_set = {}
  for subset in all_dicts:
    for k in subset:
      if k in union_set:
        union_set[k] = max(union_set[k], subset[k])
      else:
        union_set[k] = subset[k]
  return union_set


class ConjunctionRule:
  """A conjunction of words that imply a downstream condition is true.

  Stores the entire tree for how the downstream condition is derived from the
  words in the conjunction. (Note leaves of tree are the words in the
  conjunction, and the root is the downstream condition.)

  Attributes:
    leaf_words: Dict[str, int] words in the conjunction, mapped to their depth
      in the derivation tree
    leaf_words_list: List[str] words in the conjunction, mapped to their depth
      in the derivation tree
    ancestor_words: Dict[str, int] ancestor words which are implied by the
      conjunction (and must be derived on the way to derive the downstream
      condition), mapped to their depth in the derivation tree
    derivation: List[Tuple[Tuple[str, ...], str]] list of all derivations; maps
      a conjunction of ancestor words to a downstream word (e.g. (a, b, c) -> d)
    root_word: str the downstream condition that is implied by this rule can be
      of form "word" or "not word"
    layer: int the depth of the rule tree that has been expanded to form this
      rule
  """
  def __init__(
      self,
      leaf_words: Dict[str, int],
      ancestor_words: Dict[str, int],
      derivation: List[Tuple[Tuple[str, ...], str]],
      root_word: str,
      layer: int,
  ):
    self.leaf_words = leaf_words
    self.leaf_words_list = sorted(leaf_words.keys())
    # assume unknown / hard to derive (implies these words are true)
    # OPTIMIZATION: Cache the set of leaves for fast comparison
    self.leaf_set = set(self.leaf_words.keys())
    
    self.ancestor_words = ancestor_words
    self.ancestor_set = set(ancestor_words.keys())
    # maps (conjunction of ancestor words) -> downstream word
    self.derivation = derivation
    self.root_word = root_word  # implies this word is true
    self.layer = layer

  def display_derivation(self):
    for ancestor_words, downstream_word in self.derivation:
      print(f"{ancestor_words} -> {downstream_word}")

  def __str__(self):
    return " and ".join(self.leaf_words_list)

  def __iter__(self):
    return iter(self.leaf_words_list)

  def has_ancestor(self, word: str):
    return word in self.ancestor_words

  def list_has_ancestor(self, lst: Iterable[str]):
    # NEW >>>>>>>>>>>>>>>>>
    if isinstance(lst, set):
      return bool(lst & self.ancestor_set)
    return bool(set(lst) & self.ancestor_set)
    # <<<<<<<<<<<<<<<<<<<<<
    # OLD >>>>>>>>>>>>>>>>>
    # for word in lst:
    #   if self.has_ancestor(word):
    #     return True
    # return False
    # <<<<<<<<<<<<<<<<<<<

  def contradicts(self, lst: Iterable[str]):
    for word in lst:
      neg = ruleset.negate(word)
      if (neg in self.leaf_words) or (neg in self.ancestor_words):
        return True
    return False

  def __eq__(self, other):
    if isinstance(other, ConjunctionRule):
      # assert hasattr(other, "leaf_words")
      # return set(self.leaf_words.keys()) == set(other.leaf_words.keys())
      return self.leaf_set == other.leaf_set
    elif isinstance(other, Iterable):
      # return set(self.leaf_words.keys()) == set(other)
      return self.leaf_set == set(other)
    else:
      return False

  def __hash__(self):
    if not hasattr(self, '_hash'):
        self._hash = hash(frozenset(self.leaf_set))
    return self._hash
    # return hash(str(self))

  def __lt__(self, other):
    if isinstance(other, ConjunctionRule):
      # assert hasattr(other, "leaf_words")
      # return set(self.leaf_words.keys()) < set(other.leaf_words.keys())
      return self.leaf_set < other.leaf_set
    elif isinstance(other, Iterable):
      # return set(self.leaf_words.keys()) < set(other)
      return self.leaf_set < set(other)
    else:
      return False

  def __le__(self, other):
    if isinstance(other, ConjunctionRule):
      # assert hasattr(other, "leaf_words")
      # return set(self.leaf_words.keys()) <= set(other.leaf_words.keys())
      return self.leaf_set <= other.leaf_set
    elif isinstance(other, Iterable):
      # return set(self.leaf_words.keys()) <= set(other)
      return self.leaf_set <= set(other)
    else:
      return False

  def serialize(self):
    derivation = [
        [list(ancestor_words), downstream_word]
        for ancestor_words, downstream_word in self.derivation
    ]
    root_word = self.root_word
    layer = self.layer
    return {
        "leaf_words": self.leaf_words,
        "ancestor_words": self.ancestor_words,
        "derivation": derivation,
        "root_word": root_word,
        "layer": layer,
    }

  @classmethod
  def deserialize(cls, serialized_rule: Dict[str, Any]):
    leaf_words = serialized_rule["leaf_words"]
    ancestor_words = serialized_rule["ancestor_words"]
    derivation = [
        (tuple(ancestor_ws), downstream_word)
        for ancestor_ws, downstream_word in serialized_rule["derivation"]
    ]
    root_word = serialized_rule["root_word"]
    layer = serialized_rule["layer"]
    return ConjunctionRule(
        leaf_words, ancestor_words, derivation, root_word, layer
    )


def has_self_contradiction(leaf_set):
  """Check if a leaf set contains both x and not x."""
  for leaf in leaf_set:
    if ruleset.negate(leaf) in leaf_set:
      return True
  return False


def has_inferred_contradiction(rule_tree, leaf_set):
  """Check if a leaf set leads to a contradiction via inference.
  
  Args:
    rule_tree: RuleTree object
    leaf_set: Set[str] of leaf facts (e.g. {"a", "not b"})
  
  Returns:
    bool: True if the leaf set leads to a contradiction
  """
  true_facts = {f for f in leaf_set if not f.startswith("not ")}
  false_facts = {f for f in leaf_set if f.startswith("not ")}
  
  inferred = get_all_inferrable_facts(rule_tree, true_facts, false_facts)
  
  for f in inferred:
    if ruleset.negate(f) in inferred:
      return True
  return False


# NOTE: Reverted to original version despite ineffciency
def backderive_nextlayer_rules(
    rule_tree,
    prev_layer_rules: Set[ConjunctionRule],
    all_query_rules: Set[ConjunctionRule],
    max_depth: int = 5,
    max_expansions_per_layer: int = 500_000,
) -> Tuple[Set[ConjunctionRule], Set[ConjunctionRule], bool]:
  """Backderive the next layer of rules for forming a target query.

  Each layer corresponds to a depth in the rule tree. goes through each rule
  that can form the current set of rules in `prev_layer_rules` and expands them
  to form the next set of rules in `curr_layer_rules_pruned`.
  Then, prune `curr_layer_rules_pruned` to remove rules that are subsets of
  other rules, and add them to `all_query_rules`.
  Then, go to the next layer and repeat.

  Args:
    rule_tree: ruleset.RuleTree rule tree to expand
    prev_layer_rules: Set[ConjunctionRule] stores the previous layer of rules to
      expand
    all_query_rules: Set[ConjunctionRule] stores all rules that have been
      derived so far
    max_depth: int max depth to expand to

  Returns:
    Tuple[Set[ConjunctionRule], Set[ConjunctionRule]]
      curr_layer_rules_pruned: Set[ConjunctionRule]
      all_query_rules: Set[ConjunctionRule]
  """
  print()
  print(max_depth)
  len_break = False
  
  start = time()
  if max_depth == 0:
    return set(), all_query_rules, len_break

  curr_layer_rules = []
  for _, prev_layer_rule in enumerate(prev_layer_rules):
    perword_expansion_rules = []
    prev_layer_words = []
    tree_depth = max(prev_layer_rule.leaf_words.values())
    for prev_layer_word in prev_layer_rule:
      # get all expansions / ways of forming this word
      # NOTE: first add the identity expansion {prev_layer_word: depth}
      perword_expansion_rules.append(
          [{prev_layer_word: prev_layer_rule.leaf_words[prev_layer_word]}]
      )
      # NEW >>>>>>>>>>>>>>>>>
      # NOTE: keep track of words being expanded even if it's continued (for the purpose of creating correct new_derivations)
      prev_layer_words.append(prev_layer_word)
      # <<<<<<<<<<<<<<<<<<<<<
      # don't continue expand (will be captured by other expansion)
      if prev_layer_rule.leaf_words[prev_layer_word] < tree_depth:
        continue
      assert prev_layer_rule.leaf_words[prev_layer_word] == tree_depth
      for word_expansion in rule_tree.nodes[prev_layer_word].rules:  # CNF form (frozenset) containing prev_layer_word
        # expanded, so now 1 deeper
        # add negated words
        # (conjunction of these imply the target prev_layer_word)
        word_expansion = {
            ruleset.negate(word): tree_depth + 1
            for word in word_expansion
            if word != prev_layer_word
        }
        # skip expansion if it contains an ancestor word
        # --> excess information; there's some node we didn't need to expand
        # (either this node or prev time this came up)
        # NOTE: if the ancestor appears on another branch, the case where this ancestor is still a leaf will sill pass this filter, so it's fine
        # TODO: Commented out due to incomplete derivation coverage issues; performance impact is manageable
        # if prev_layer_rule.list_has_ancestor(set(word_expansion.keys())):
        #   continue
        
        # cannot expand (word expansion has a not x, where x in rule,
        # meaning no way to assign to guarantee goal, premise is False so
        # implies doesn't go through)
        if prev_layer_rule.contradicts(set(word_expansion.keys())):
          continue
        # NOTE: Add valid expansion
        perword_expansion_rules[-1].append(word_expansion)
      # ORIGINAL >>>>>>>>>>>>>>>>>
      # prev_layer_words.append(prev_layer_word)
      # <<<<<<<<<<<<<<<<<<<<<<<<<<
      
    # take all combinations of per_word_rules for the `prev_layer_rules`
    # [[{a: d1}, {c: d2, d: d2}], [{b: d1}, {c: d2, e: d2}] [{f: d0}]]
    prev_rule_expansions = it.product(*perword_expansion_rules)  # TODO: How many?
    # -> [
    #   ({a: d1}, {b: d1}, {f: d0}),
    #   ({a: d1}, {c: d2, e: d2}, {f: d0}),
    #   ({c: d2, d: d2}, {b: d1}, {f: d0}),
    #   ({c: d2, d: d2}, {c: d2, e: d2}, {f: d0})
    # ]
    prev_rule_expansions_linearized = []
    for expansion in prev_rule_expansions:
      # tuple of {var: depth, ...}
      new_leaves = union(expansion)
      # Skip if new leaves are a superset of prev rule
      if set(new_leaves.keys()) >= prev_layer_rule:
        # can already (directly) derive this is true from an existing rule
        continue
      
      # NEW >>>>>>>>>>>>>>>>>
      # NOTE: Check for direct self-contradiction in merged leaves (i.e., "x" and "not x" in two `word_expansion` of different `prev_layer_word`s in the same `perword_expansion_rules`)
      # For efficiency, we defer the check for contradictions via inference to after curating the true and false derivations
      new_leaves_set = set(new_leaves.keys())
      if has_self_contradiction(new_leaves_set):
        continue
      # <<<<<<<<<<<<<<<<<<<<<
      
      new_ancestors = {
          **prev_layer_rule.ancestor_words,
          **prev_layer_rule.leaf_words,
      }
      # OLD >>>>>>>>>>>>>>>>>
      # for k in union(expansion):
      # <<<<<<<<<<<<<<<<<<<<<
      # NEW >>>>>>>>>>>>>>>>>
      for k in new_leaves:
      # <<<<<<<<<<<<<<<<<<<<<
        if k in new_ancestors:
          del new_ancestors[k]
      new_derivations = prev_layer_rule.derivation + [
          (tuple(expansion[w].keys()), word)
          for w, word in enumerate(prev_layer_words)
          if not (len(expansion[w]) == 1 and word in expansion[w])
      ]
      
      prev_rule_expansions_linearized.append(
          ConjunctionRule(
              new_leaves,
              new_ancestors,
              new_derivations,
              prev_layer_rule.root_word,
              prev_layer_rule.layer + 1,
          )
      )
    curr_layer_rules.extend(prev_rule_expansions_linearized)
  print(f"Time for product expansions: {time() - start:.2f} seconds")
  
  # NOTE: Skip the rule set (because the redundancy checks are going to be too costly) if any layer exceeds 500,000/1,000,000 possible product-expanded rules
  if len(curr_layer_rules) > max_expansions_per_layer:
    len_break = True
    return set(), all_query_rules, len_break
  start = time()

  # TODO: The following two steps intentionally keeps only the minimally sufficient variable assignments (no supersets)
  
  # 1. Prune ancestors: Removes old rules (in all_query_rules) made redundant by new discoveries (curr_layer_rules)
  # now delete rules which are supersets of other rules
  # only add rules which are subsets of other rules
  ancestor_query_rules_pruned = set()  # if 2 derivations, keep first
  # ORIGINAL >>>>>>>>>>>>>>>>>
  # for _, rule in enumerate(all_query_rules):
  #   # check if any other rule is a subset of rule
  #   has_subset = False
  #   for _, rule2 in enumerate(curr_layer_rules):
  #     if rule == rule2:
  #       continue
  #     if rule2 < rule:
  #       has_subset = True
  #       break
  #   if not has_subset:
  #     ancestor_query_rules_pruned.add(rule)
  # <<<<<<<<<<<<<<<<<<<<<<<<<<
  # NEW >>>>>>>>>>>>>>>>>>>
  # Build size-indexed structure for curr_layer_rules
  curr_by_size = {}
  for rule in curr_layer_rules:
    size = len(rule.leaf_set)
    if size not in curr_by_size:
      curr_by_size[size] = []
    curr_by_size[size].append(rule)
  
  min_curr_size = min(curr_by_size.keys()) if curr_by_size else float('inf')
  
  for rule in all_query_rules:
    rule_size = len(rule.leaf_set)
    if rule_size <= min_curr_size:
      # No rule in curr_layer can be a strict subset
      ancestor_query_rules_pruned.add(rule)
      continue
    
    has_subset = False
    for size in range(1, rule_size):  # Strict subset means size < rule_size
      if size not in curr_by_size:
        continue
      for rule2 in curr_by_size[size]:
        if rule2.leaf_set < rule.leaf_set:
          has_subset = True
          break
      if has_subset:
        break
    
    if not has_subset:
      ancestor_query_rules_pruned.add(rule)
  # <<<<<<<<<<<<<<<<<<<<<<<
  print(f"Time for pruning ancestors: {time() - start:.2f} seconds")
  
  # 2. Remove new rules (curr_layer_rules) that are supersets of any existing rule (all_query_rules --> ancestor_query_rules_pruned)
  curr_layer_rules_pruned = set()  # if 2 derivations, keep first
  start = time()
  st = time()
  # ORIGINAL >>>>>>>>>>>>>>>>>
  # existing_rules_list = list(all_query_rules)
  # <<<<<<<<<<<<<<<<<<<<<<<<<<
  # NEW >>>>>>>>>>>>>>>>>>>
  existing_rules_list = list(ancestor_query_rules_pruned)  # NOTE: Efficiency fix
  # <<<<<<<<<<<<<<<<<<<<<<<
  # ORIGINAL >>>>>>>>>>>>>>>>>
  # for r, rule in enumerate(curr_layer_rules):
  #   # check if any other rule is a subset of rule (rule -> other rule)
  #   has_subset = False
    
  #   for _, rule2 in enumerate(curr_layer_rules + existing_rules_list):
  #     if rule == rule2:
  #       continue
  #     if rule2 < rule:
  #       has_subset = True
  #       break
  #
  #   if not has_subset:
  #     curr_layer_rules_pruned.add(rule)
  #   if r % 1000 == 0:
  #     print(f"{r} / {len(curr_layer_rules)}")
  #     print(f"  Time for checking: {time() - st:.2f} seconds")
  #     st = time()
  # <<<<<<<<<<<<<<<<<<<<<<<<<<
  # NEW >>>>>>>>>>>>>>>>>>>
  # Build combined size index
  all_by_size = {size: list(rules) for size, rules in curr_by_size.items()}
  for rule in existing_rules_list:
    size = len(rule.leaf_set)
    if size not in all_by_size:
      all_by_size[size] = []
    all_by_size[size].append(rule)
  
  min_all_size = min(all_by_size.keys()) if all_by_size else float('inf')
  for r, rule in enumerate(curr_layer_rules):
    rule_size = len(rule.leaf_set)
    
    if rule_size <= min_all_size:
      # No rule can be a strict subset
      curr_layer_rules_pruned.add(rule)
      continue
    
    has_subset = False
    for size in range(1, rule_size):
      if size not in all_by_size:
        continue
      for rule2 in all_by_size[size]:
        if rule2.leaf_set < rule.leaf_set:
          has_subset = True
          break
      if has_subset:
        break
    
    if not has_subset:
      curr_layer_rules_pruned.add(rule)
    
    if r % 1000 == 0:
      print(f"{r} / {len(curr_layer_rules)}")
  # <<<<<<<<<<<<<<<<<<<<<<<
    
  # curr_layer_rules = uniq_query_rules
  print(f"Time for pruning current layer: {time() - start:.2f} seconds")
  start = time()

  all_query_rules = ancestor_query_rules_pruned.union(curr_layer_rules_pruned)
  # expand current set of rules into next layer
  _, all_query_rules, len_break = backderive_nextlayer_rules(
      rule_tree, curr_layer_rules_pruned, all_query_rules, max_depth - 1, max_expansions_per_layer
  )

  return curr_layer_rules_pruned, all_query_rules, len_break


def get_derivations(rules_dict, max_expansions_per_layer: int):
  """Compute all derivations of a target word given a rules_dict.

  Args:
    rules_dict: Dict[str, Any]

  Returns:
    bool: True if derivations were computed, False if not
  """
  if not isinstance(rules_dict["rules"], ruleset.RuleTree):
    assert isinstance(rules_dict["rules"], list)
    rules_dict["rules"] = ruleset.RuleTree.deserialize(rules_dict["rules"])

  if rules_dict["query"] not in rules_dict["rules"].nodes:
    return False
  
  # y = ruleset.RuleNode(word="y", rules=frozenset({frozenset({"y", "not a", "not b"})}))
  # a = ruleset.RuleNode(word="a", rules=frozenset({frozenset({"a", "not b", "not e"})}))
  # b = ruleset.RuleNode(word="b", rules=frozenset({frozenset({"b", "not f"})}))
  # f = ruleset.RuleNode(word="f", rules=frozenset({}))
  # e = ruleset.RuleNode(word="e", rules=frozenset({}))
  # rt = ruleset.RuleTree(nodes={"y":y, "a":a, "b":b, "e":e, "f":f})
  # _, temp, _ = backderive_nextlayer_rules(rt, {ConjunctionRule({"y":0}, {}, [], "y", 0)}, set(), 5)
  # print(temp)

  if not rules_dict.get("true_derivations", []):
    # can take 3 mins each, need some more efficient mechanism...
    print("\n------- Deriving true derivations -------")
    start = time()
    _, rules_dict["true_derivations"], len_break = backderive_nextlayer_rules(
        rule_tree=rules_dict["rules"],  # NOTE: RuleTree object, whose .node attribute is a dict of {word-value: RuleNode object}. A RuleNode object's rules attribute is a set of rules (CNF form)
        prev_layer_rules={
            ConjunctionRule(
                {rules_dict["query"]: 0}, {}, [], rules_dict["query"], 0
            )
        },
        all_query_rules=set(),
        max_depth=len(rules_dict["rules"].nodes),  # NOTE: Set of all word-values in the rules (CNF form)
        max_expansions_per_layer=max_expansions_per_layer,
    )  # NOTE: rules_dict["true_derivations"] is a set of ConjunctionRule objects (derivations leading to the target word being true)
    print(f"TOTAL TIME: {time() - start:.2f} seconds")
    if len_break:
      return False
    
    # remove target
    rules_dict["true_derivations"] = [
        rule
        for rule in rules_dict["true_derivations"]
        if rule != {rules_dict["query"]}
    ]
    
    # filter out derivations that lead to contradictions via inference
    pre_filter_count = len(rules_dict["true_derivations"])
    rules_dict["true_derivations"] = [
        rule for rule in rules_dict["true_derivations"]
        if not has_inferred_contradiction(rules_dict["rules"], rule.leaf_set)
    ]
    post_inferred_filter = len(rules_dict["true_derivations"])
    
    if pre_filter_count != post_inferred_filter:
      # print(f"Filtered {pre_filter_count - post_self_filter} self-contradictory true derivations")
      print(f"Filtered {pre_filter_count - post_inferred_filter} inferred-contradictory true derivations")
    
  if not rules_dict.get("false_derivations", []):
    print("\n------- Deriving false derivations -------")
    start = time()
    _, rules_dict["false_derivations"], len_break = backderive_nextlayer_rules(
        rule_tree=rules_dict["rules"],
        prev_layer_rules={
            ConjunctionRule(
                {f"not {rules_dict['query']}": 0},
                {},
                [],
                f"not {rules_dict['query']}",
                0,
            )
        },
        all_query_rules=set(),
        max_depth=len(rules_dict["rules"].nodes),
        max_expansions_per_layer=max_expansions_per_layer,
    )
    print(f"TOTAL TIME: {time() - start:.2f} seconds")
    if len_break:
      return False
    
    # remove target
    rules_dict["false_derivations"] = [
        rule
        for rule in rules_dict["false_derivations"]
        if rule != {f"not {rules_dict['query']}"}
    ]
    
    # filter out derivations that lead to contradictions via inference
    pre_filter_count = len(rules_dict["false_derivations"])
    rules_dict["false_derivations"] = [
        rule for rule in rules_dict["false_derivations"]
        if not has_inferred_contradiction(rules_dict["rules"], rule.leaf_set)
    ]  # NOTE: should use refutation; but here UP is fine
    post_inferred_filter = len(rules_dict["false_derivations"])
    
    if pre_filter_count != post_inferred_filter:
      # print(f"Filtered {pre_filter_count - post_self_filter} self-contradictory false derivations")
      print(f"Filtered {pre_filter_count - post_inferred_filter} inferred-contradictory false derivations")
    
  return True


def get_all_inferrable_facts(rule_tree, true_facts, false_facts):
  """Get all facts that can be inferred from the given true and false facts.

  Args:
    rule_tree: RuleTree
    true_facts: Set[str]
    false_facts: Set[str]

  Returns:
    Set[str]
  """
  # ASSUMES FALSE_FACTS ALREADY IN FORM "not _"
  rules_to_consider = copy.deepcopy(rule_tree.nodes)

  all_facts = set(true_facts).union(set(false_facts))

  curr_facts = all_facts
  while curr_facts:
    new_facts = set()
    for fact in curr_facts:
      if ruleset.negate(fact) not in rules_to_consider:
        continue
      for rule in rules_to_consider[ruleset.negate(fact)].rules:
        # find any facts that must be true, due to negate(fact) being false
        remaining_terms = set(rule)
        rule_true = False
        for term in rule:
          if term in all_facts:
            # this rule is true by this term
            rule_true = True
            break
          if ruleset.negate(term) in all_facts:
            # other terms must be true
            remaining_terms.remove(term)
        if rule_true:
          continue
        if len(remaining_terms) == 1:
          new_facts.add(remaining_terms.pop())
    curr_facts = new_facts
    all_facts = all_facts.union(curr_facts)
  return all_facts


# def load_derivations():
#   """Compute derivations for all words in RP directory.

#   RP directory contains a list of rules_dicts, each of which contains a query
#   word and a rule tree. For each rules_dict, we compute all derivations of the
#   target word, both derivations that imply the target word is true and
#   derivations that imply the target word is false.
#   Adds a `true_derivations` and `false_derivations` field to each rules_dict.

#   Returns:
#     List[Dict[str, Any]] of rules_dicts
#   """
#   rules_dicts = []
#   files_to_rules_dicts = {}
#   for item_file in glob.glob("SimpleLogic/RP/RP/*.jsonl"):
#     files_to_rules_dicts[item_file] = []
#     with open(item_file, "r") as f:
#       for line in f:
#         try:
#           rules_dicts.append(json.loads(line))
#           files_to_rules_dicts[item_file].append(json.loads(line))
#         except json.JSONDecodeError:
#           continue
#   return rules_dicts
