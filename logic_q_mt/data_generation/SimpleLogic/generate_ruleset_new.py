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

"""Stage 1: generate k-sufficient sets (k = 1..max_k) per ruleset."""

import argparse
import glob
import json
import os
from time import time

DATA_DIR = os.environ.get("DATA_DIR", "")

from SimpleLogic import derivation_new
from SimpleLogic import holdout_utils_new
from SimpleLogic import ruleset


def main(arguments) -> None:
  rules_dicts, files_to_rules_dicts = ruleset.load_data(arguments.sl_dir)
  start_idx = int(arguments.start_idx)
  end_idx = int(arguments.end_idx)
  for item_file in files_to_rules_dicts:
    write_file = item_file.replace(
        ".txt", f"_{start_idx}_{end_idx}_heldout_fixed_new.jsonl"
    )
    heldout_files = item_file.replace(".txt", "*heldout_fixed_new.jsonl")
    rules_dicts_existing = []
    for existing_file in glob.glob(heldout_files):
      with open(existing_file, "r") as f:
        for line in f:
          rules_dicts_existing.append(line)

    existing_rules_dicts_rules_queries = set()
    for rules_dict in rules_dicts_existing:
      try:
        rules_dict = json.loads(rules_dict)
      except json.JSONDecodeError:
        continue
      existing_rules_dicts_rules_queries.add((
          json.dumps(
              ruleset.RuleTree.deserialize(rules_dict["rules"]).serialize()
          ),
          rules_dict["query"],
      ))

    for r, rules_dict in enumerate(rules_dicts):
      if r < start_idx or r >= end_idx:
        continue
      rule_tree_rules = ruleset.RuleTree.deserialize(rules_dict["rules"])  # NOTE: [[[a, b, c], word], [[d, e, f], word], ...] --> RuleTree (.nodes[word_value].rules = sets of conjunctive normal forms, e.g., {{"not a", "not b", "not c", "word"}, {"not d", "not e", "not f", "word"}, ...})
      if (
          json.dumps(rule_tree_rules.serialize()),  # NOTE: .serialize() --> list of conjunctive normal forms, e.g., [["not a", "not b", "not c", "d"], ["not d", "not e", "not f", "g"], ...]
          rules_dict["query"],
      ) in existing_rules_dicts_rules_queries:
        continue
      print(f"\n================Processing {r}/{len(rules_dicts)}=================")
      
      valid = derivation_new.get_derivations(rules_dict, arguments.max_expansions_per_layer)  # NOTE: Generate all A^{(y)} and A^{(\neg y)}. (The most) Costly, but constant in k
      if not valid:
        # Save a placeholder to mark this rule as processed (invalid)
        # so it will be skipped on reruns
        placeholder_dict = {
            "rules": rule_tree_rules.serialize(),
            "query": rules_dict["query"],
            "depth": rules_dict.get("depth"),
            "invalid": True,  # Mark as invalid
        }
        with open(write_file, "a") as wf:
            wf.write(json.dumps(placeholder_dict) + "\n")
            wf.flush()
        rules_dicts_existing.append(json.dumps(placeholder_dict) + "\n")
        print(f"Skipped invalid rule {r}/{len(rules_dicts)}, saved placeholder to " + write_file)
        continue
      
      # Generate heldout sets with max_k
      start = time()
      holdout_utils_new.make_heldout_ruleset(rules_dict, max_k=arguments.max_k)
      print(f"\nTOTAL Time to make heldout sets: {time() - start:.2f} seconds\n")
      
      
      output_dict = {
          "rules": rule_tree_rules.serialize(),
          "query": rules_dict["query"],
          "depth": rules_dict["depth"],
          "true_derivations": [
              derive.serialize()
              for derive in rules_dict["true_derivations"]
          ],
          "false_derivations": [
              derive.serialize()
              for derive in rules_dict["false_derivations"]
          ],
          "heldout_set_to_q": rules_dict["heldout_set_to_q"],
          "heldout_set_to_subset_qs": rules_dict[
              "heldout_set_to_subset_qs"
          ],
      }
      
      # Add heldout_k_sets if available (from new holdout_utils)
      if "heldout_k_sets" in rules_dict:
          output_dict["heldout_k_sets"] = rules_dict["heldout_k_sets"]
          
      if "context_to_invalid_sets" in rules_dict:
          output_dict["context_to_invalid_sets"] = rules_dict["context_to_invalid_sets"]

      with open(write_file, "a") as wf:
        wf.write(json.dumps(output_dict) + "\n")
        wf.flush() # Robustness: ensure data is written to disk
        
        print(f"Wrote {r}/{len(rules_dicts)} to " + write_file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--sl_dir", type=str, default=os.environ.get("LOGIC_Q_RP_DIR", "/path/to/questbench_data/Logic-Q/RP/RP"))
  parser.add_argument("--start_idx", type=int, default=0)
  parser.add_argument("--end_idx", type=int)
  parser.add_argument("--max_k", type=int, default=4)
  parser.add_argument("--max_expansions_per_layer", type=int, default=500_000)
  args = parser.parse_args()
  main(args)
