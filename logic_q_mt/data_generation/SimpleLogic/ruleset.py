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

"""Classes for representing and manipulating SimpleLogic rulesets."""

import functools
import glob
import json
from typing import Dict, FrozenSet, List

import tqdm


def load_data(sl_dir: str):
  """Load data from CNS.

  Args:
    sl_dir: str path to directory containing SimpleLogic rulesets

  Returns:
  """
  rulesets = []
  files_to_rulesets = {}
  for item_file in tqdm.tqdm(glob.glob(f"{sl_dir}/*.txt")):
    files_to_rulesets[item_file] = []
    with open(item_file, "r") as f:
      for line in f:
        try:
          rulesets.extend(json.loads(line))
          files_to_rulesets[item_file].extend(json.loads(line))
        except json.JSONDecodeError:
          continue
    # Load-bearing `break`: by design each stage-1 shard directory holds
    # exactly ONE prop_examples_*.txt (see data_generation/README.md and
    # prepare_shards.py). Processing one file per --sl_dir is what keeps the
    # parallel shard layout consistent, so do not remove this.
    break
  return rulesets, files_to_rulesets


class RuleNode:
  """A node in the rule tree representing all rules that can derive a word.

  Rules are of form (x1 and x2 and ...) -> y, where x1, x2, ... are words, and y
  is a word. There may be multiple rules implying the same word y. We store
  these as a set of rules.

  Attributes:
    word: str the conclusion word y
    rules: FrozenSet[FrozenSet[str]] the set of rules (x1, x2, ...) that imply
      the conclusion word y
  """

  def __init__(self, word: str, rules: FrozenSet[FrozenSet[str]] = frozenset()):
    self.word = word
    self.rules = rules

  def sort_rules(self):
    """Sort rules by length, then order of first distinct word."""
    sorted_rules = []
    for _, rule in enumerate(self.rules):
      # sort within rules
      sorted_rules.append(sorted(list(rule)))
    # sort between rules by length, then order of first distinct word
    sorted_rules.sort(key=functools.cmp_to_key(self.rule_comparator))
    return sorted_rules

  def rule_comparator(self, rule1: List[str], rule2: List[str]):
    """Returns True if rule1 is greater than rule2."""
    return " and ".join(sorted(rule1)) > " and ".join(sorted(rule2))

  def __str__(self):
    return self.word

  def __eq__(self, other):
    if self.word != other.word:
      return False
    return self.rules == other.rules

  def __hash__(self):
    return hash(self.word)

  def serialize(self):
    sorted_rules = self.sort_rules()
    return [[rule, self.word] for rule in sorted_rules]


def negate(word: str):
  if word.startswith("not "):
    return word.split("not ")[-1]
  else:
    return f"not {word}"


class RuleTree:
  """Tree representing the full set of rules.

  Serialized as a list of rules of form [[x1, x2, ...], y] where x1, x2, ..., y
  are words and y is a word. There may be multiple rules implying the same word
  y. We store these as a set of rules.
  # For every rule (x1, x2, x3, ...) -> y, we also store the rules:
  #   (x2, x3, ..., not y) -> not x1
  #   (x1, x3, ..., not y) -> not x2
  #   (x1, x2 ..., not y) -> not x3
    ...
    [Not explicitly stored, but stored with conjunctive normal forms:
    (not x1 or not x2 or not x3 or ... or y)]
  These are necessary for backderiving the false derivations, and ensuring we
  capture all derivations of the target word's truth value.

  Attributes:
    nodes: Dict[str, RuleNode] the nodes of the tree, keyed by word
    sorted_nodes: List[str] the sorted keys of the nodes dict (for printing,
      consistent serialization, etc.)
  """

  def __init__(self, nodes: Dict[str, RuleNode]):
    self.nodes = nodes
    self.sorted_nodes = sorted(self.nodes.keys())

  @classmethod
  def deserialize(cls, serialized_rules: List[str]):
    """Deserialize a list of rules.

    Args:
      serialized_rules: List[Any] list of rules of form [[x1, x2, ...], y] where
        x1, x2, ..., y are words and

    Returns:
      RuleTree
        representation of the rules
    """
    nodes = {}
    for rule in serialized_rules:
      words_in_rule = set()
      if (
          len(rule) == 2
          and isinstance(rule[0], list)
          and isinstance(rule[1], str)
      ):
        for premise_word in rule[0]:
          neg_premise = negate(premise_word)
          words_in_rule.add(neg_premise)
        conclusion_word = rule[1]
        words_in_rule.add(conclusion_word)
        for word in words_in_rule:
          if word not in nodes:
            nodes[word] = set()
          if negate(word) not in nodes:
            nodes[negate(word)] = set()
          nodes[word].add(frozenset(words_in_rule))
      else:
        assert isinstance(rule[0], str)
        # list of disjunctions
        for word in rule:
          if word not in nodes:
            nodes[word] = set()
          nodes[word].add(frozenset(rule))

    for word in nodes:
      nodes[word] = RuleNode(word, frozenset(nodes[word]))

    return RuleTree(nodes)

  def __eq__(self, other):
    return self.serialize() == other.serialize()

  def __hash__(self):
    return hash(self.serialize())

  def __str__(self):
    return "\n".join(sorted(self.sorted_nodes))

  def num_rules(self):
    return len(self.serialize())

  def num_words(self):
    return len(self.nodes)

  def serialize(self):
    rules = set()
    for node_word in self.nodes:
      rules.update(self.nodes[node_word].rules)
    rules = sorted([sorted(list(rule)) for rule in rules], key=" and ".join)
    return rules
