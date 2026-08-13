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

"""Evaluate LLMs on Logic-Q."""

import ast
import copy
import json
import random
import re

from evaluators.evaluator import Evaluator
from model_utils import cached_generate
import pandas as pd
import tqdm


class SimpleLogicEvaluator(Evaluator):
  """Evaluator for LLMs on Logic-Q.

  Attributes:
    model_name: name of LLM to evaluate
    generation_config: generation config for LLM
    cache: cache of LLM responses
    cache_file: cache file of LLM responses
    vanilla_prompt: vanilla system prompt for multiple choice evaluation
    vanilla_isambig_prompt: vanilla system prompt for ambiguity identification
      evaluation
    vanilla_fullinfo_prompt: vanilla system prompt for fully specified
      evaluation
    cot_prompt: CoT system prompt for multiple choice evaluation
    cot_isambig_prompt: CoT system prompt for ambiguity identification
      evaluation
    cot_fullinfo_prompt: CoT system prompt for fully specified evaluation
    fs_prompt: System prompt for few-shot evaluation for multiple choice
      evaluation
    fs_isambig_prompt: System prompt for few-shot evaluation for ambiguity
      identification evaluation
    fs_fullinfo_prompt: System prompt for few-shot evaluation for fully
      specified evaluation
    non_fs_request: User prompt for vanilla and CoT evaluation
    fs_request: User prompt for few-shot evaluation for multiple choice
      evaluation
    use_cot: whether to use CoT or not
    fs_samples: number of few-shot samples to use
    eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
    system_prompt: system prompt for current evaluation mode
    request: user prompt for current evaluation mode
    batch_size: batch size for evaluation
    model_role_name: role name for the model
    parallel_model_calls: whether to make parallel calls to the model
  """

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      eval_mode: str = "mc",
      batch_size: int = 16,
      **kwargs,
  ):
    super().__init__(
        model_name,
        cache=cache,
        cache_file=cache_file,
        use_cot=use_cot,
        fs_samples=fs_samples,
        eval_mode=eval_mode,
        **kwargs,
    )
    
    self.prompts = {
      "mc": {
        "system_prompt": {
#           # Ask up to K
#           "vanilla_at_most_K": """Suppose you know the following rules about Alice:
#     {rules_nl}

# You are trying to discern whether a statement about Alice is true given some facts. You have a budget to ask about up to {max_k} attributes at once. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following:

# Instructions:
# 1. If you already have enough information to determine the truth value of the statement, respond strictly with: "End questioning".
# 2. Otherwise, you MUST select a set of attributes (at least 1 and at most {max_k}) to query. Choose the best combination that provides the most information regarding the statement.
# 3. Format the question strictly as: "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? ..." (for at least 1 and at most {max_k} attributes).
# 4. Do not output any other text.""",
          # Ask up to K
          "vanilla_at_most_K": """Suppose you know the following rules about Alice:
    {rules_nl}

You are trying to discern whether a statement about Alice is true given some facts. You have a budget to ask about up to {max_k} attributes at once. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following:

Instructions:
1. If you already have enough information to determine the truth value of the statement, respond strictly with: "End questioning".
2. Otherwise, you MUST select a set of attributes (at least 1 and at most {max_k}) to query. Choose the smallest combination that provides sufficient information to determine the truth value of the statement.
3. Format the question strictly as: "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? ..." (for at least 1 and at most {max_k} attributes).
4. Do not output any other text.""",
          # Ask exactly k
          "vanilla_exact_k": """Suppose you know the following rules about Alice:
    {rules_nl}

You are trying to discern whether a statement about Alice is true given some facts. You must select exactly {k} attributes of Alice to query in order to gain the most information about the final statement.

Instructions:
1. You MUST select exactly {k} attributes to query. Choose the best combination that provides the most information regarding the statement.
2. Format the question strictly as: "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? ..." (for exactly {k} attributes).
3. Do not output any other text.""",

#           # Ask up to 1
#           "vanilla_k1": """Suppose you know the following rules about Alice:
#     {rules_nl}

# You are trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following:
# If you do not have enough information yet, you may ask a question about an attribute of Alice, in the form of "Question: Is Alice [attribute]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
# Once you have enough information necessary to determine the truth value of the statement, you can terminate with "End questioning".
# Generate one of "Question: Is Alice [attribute]?" or "End questioning" and nothing else.""",

#           # Ask up to 2
#           "vanilla_k2": """Suppose you know the following rules about Alice:
#     {rules_nl}

# You are trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following:
# If you do not have enough information yet, you may ask questions about at most two attributes of Alice, in the form of "Question: Is Alice [attribute_1]? Is Alice [attribute_2]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
# Once you have enough information necessary to determine the truth value of the statement, you can terminate with "End questioning".
# Generate one of "Question: Is Alice [attribute_1]? Is Alice [attribute_2]?" or "End questioning" and nothing else.""",

#           # Ask up to 3
#           "vanilla_k3": """Suppose you know the following rules about Alice:
#     {rules_nl}

# You are trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following:
# If you do not have enough information yet, you may ask questions about at most three attributes of Alice, in the form of "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? Is Alice [attribute_3]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
# Once you have enough information necessary to determine the truth value of the statement, you can terminate with "End questioning".
# Generate one of "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? Is Alice [attribute_3]?" or "End questioning" and nothing else.""",

#           # Ask up to 4
#           "vanilla_k4": """Suppose you know the following rules about Alice:
#     {rules_nl}

# You are trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following:
# If you do not have enough information yet, you may ask questions about at most four attributes of Alice, in the form of "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? Is Alice [attribute_3]? Is Alice [attribute_4]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
# Once you have enough information necessary to determine the truth value of the statement, you can terminate with "End questioning".
# Generate one of "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? Is Alice [attribute_3]? Is Alice [attribute_4]?" or "End questioning" and nothing else."""

        },
        "user_prompt": {
          "non_fs": """{known_facts}
{known_untrue_facts}
{invalid_qs}
Is Alice {goal}?""",
          "fs": """Rules:
{rules_nl}

Facts:
{known_facts}
{known_untrue_facts}
{invalid_qs}

Target Question:
Is Alice {goal}?""",
          "explicit_valid_set": """{known_facts}
{known_untrue_facts}
{valid_qs}
Is Alice {goal}?"""
        }
      },
      "isambig": {
        "system_prompt": """Suppose you know the following rules about Alice:
{rules_nl}

You will be given some facts about Alice (some attributes known true/false), and then a target yes/no question of the form "Is Alice X?". Decide whether the target question can be answered with certainty from the given facts and rules.

Output in exactly one of these formats:
- If the question is NOT ambiguous (it can be answered definitively as Yes or No): "Answer: No"
- If the question IS ambiguous because some attribute values are missing: output the MINIMUM number of additional attributes that must be known to answer definitively, as an integer 1-4: "Answer: 1", "Answer: 2", "Answer: 3", or "Answer: 4"
- If you are sure it is ambiguous but cannot determine the count: "Answer: Yes, but not sure about how many"

Do not output any other text.""",
        "user_prompt": {
          "non_fs": """{known_facts}
{known_untrue_facts}
Is Alice {goal}?""",
          "fs": """Rules:
{rules_nl}

Facts:
{known_facts}
{known_untrue_facts}

Target Question:
Is Alice {goal}?""",
        },
      },
      "fullinfo": {
        "system_prompt": """Suppose you know the following rules about Alice:
{rules_nl}

You will be given some facts about Alice (some attributes known true/false), and then a target yes/no question of the form "Is Alice X?". Answer the target question with certainty.

Output exactly ONE line: 
- "Answer: Yes" or "Answer: No"

Do not output any other text.""",
        "user_prompt": {
          "non_fs": """{known_facts}
{known_untrue_facts}
Is Alice {goal}?""",
          "fs": """Rules:
{rules_nl}

Facts:
{known_facts}
{known_untrue_facts}

Target Question:
Is Alice {goal}?""",
        },
      }
    }

#     self.cot_prompt = """Suppose you know the following rules about Alice:
#     {rules_nl}

# You trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following-
# If you do not have enough information yet, you may ask a question about an attribute of Alice, in the form of "Question: Is Alice [attribute]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
# Once you have enough all information necessary to determine the truth value of the statement, you can terminate with "End questioning".
# iefly, then generate one of "Question: Is Alice [attribute]?" or "End questioning"."""
#     self.cot_isambig_prompt = """Suppose you know the following rules about Alice:
# {rules_nl}

# You will presented with a binary question about an attribute of Alice. Please answer it with "Yes" or "No" or "Not sure".
# Reason step-by-step, then generate "Answer:" followed by the answer and nothing else."""
#     self.cot_fullinfo_prompt = """Suppose you know the following rules about Alice:
# {rules_nl}

# You will be given a binary question about an attribute of Alice. Please answer it with "Yes" or "No".
# Reason step-by-step, then generate "Answer:" followed by the answer and nothing else."""
#     self.fs_prompt = """You trying to discern whether a statement about Alice is true given some facts. You must decide whether you have enough information to determine answer the target question. You may respond with one of the following-
# If you do not have enough information yet, you may ask a question about an attribute of Alice, in the form of "Question: Is Alice [attribute]?". Ask the best question that, regardless of how it is answered, provides the most information about the final statement.
# Once you have enough all information necessary to determine determine the truth value of the statement, you can terminate with "End questioning".
# Generate one of "Question: Is Alice [attribute]?" or "End questioning" and nothing else."""
#     self.fs_request = """Rules:
# {rules_nl}

# Facts:
# {known_facts}
# {known_untrue_facts}
# {invalid_qs}

# Target Question:
# Is Alice {goal}?"""
#     self.fs_isambig_prompt = """You will be given some rules and facts about Alice, and then a target yes/no question of the form "Is Alice X?".

# Decide whether the target question is ambiguous (missing information). If it is NOT ambiguous, output: "Answer: No".
# If it IS ambiguous, output the MINIMUM number of additional attributes that must be known to answer definitively (an integer 1–4): "Answer: 1", "Answer: 2", "Answer: 3", or "Answer: 4".
# If you are sure it is ambiguous but cannot determine the count, output: "Answer: Yes, but not sure about how many".

# Output exactly one line and nothing else."""
#     self.fs_fullinfo_prompt = """You will be given some rules and facts about Alice, and then a target yes/no question of the form "Is Alice X?".

# Answer the target question with certainty.

# Output exactly one line:
# "Answer: Yes" or "Answer: No".

# Do not output anything else."""

    if self.fs_samples > 0:
      if self.eval_mode == "mc":
        self.system_prompt = self.fs_prompt
      elif self.eval_mode == "isambig":
        self.system_prompt = self.fs_isambig_prompt
      elif self.eval_mode == "fullinfo":
        self.system_prompt = self.fs_fullinfo_prompt
      self.user_prompt = self.prompts[self.eval_mode]["user_prompt"]["fs"]
    else:
      # In non-fewshot mode, per-example system prompts are constructed in
      # make_batches (since they include {rules_nl}).
      self.system_prompt = None
      self.user_prompt = self.prompts[self.eval_mode]["user_prompt"]["non_fs"]

    self.batch_size = batch_size

  def evaluate_batch(
      self,
      batch_user_prompts,
      batch_system_prompts,
      model_name,
      batch_gt_queries,
      cache=None,
      cache_file=None,
      fs_turns=None,
  ):
    """Evaluates a batch of requests.

    Args:
      batch_requests: The batch of requests.
      batch_system_prompts: The batch of system prompts.
      model_name: The name of the model to evaluate.
      batch_gt_queries: The batch of ground truth responses.
      cache: The cache of LLM responses.
      cache_file: The cache file of LLM responses.
      fs_turns: The fewshot turns.

    Returns:
      The batch of LM responses, LM conversations, and whether they are
      correctness.
    """
    batch_prompts = []
    for user_prompt, system_prompt in zip(batch_user_prompts, batch_system_prompts):
      assist_prompt = []
      if self.fs_samples > 0:
        assist_prompt.extend(fs_turns)
      if system_prompt is None:
        assist_prompt.append({"role": "user", "content": user_prompt})
      else:
        assist_prompt.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
      batch_prompts.append(assist_prompt)
    
    batch_responses, all_num_thinking_tokens, all_cots, all_costs = cached_generate(
        batch_prompts,
        model_name,
        port=self.vllm_port,
        cache=cache,
        cache_file=cache_file,
        generation_config=self.generation_config,
    )

    # Initialize conversations from batch_prompts (using "text" key instead of "content")
    batch_convos = []
    for i, prompt in enumerate(batch_prompts):
      convo = []
      for msg in prompt:
        convo.append({"role": msg["role"], "text": msg["content"]})
      convo.append({"role": self.model_role_name, "text": batch_responses[i]})
      batch_convos.append(convo)

    # Helper to check if response needs retry
    def needs_retry(response, eval_mode):
      if eval_mode == "mc":
        return (
            not re.findall(r"Is Alice \[?([ \w-]+)\]?\?", response)
            and "end questioning" not in response.lower()
        )
      elif eval_mode == "fullinfo":
        return not re.findall(r"(yes|not sure|no)", response.lower())
      else:
        # isambig: expect No / 1-4 / "Yes, but not sure about how many"
        return not (
            re.findall(r"\b([1-4])\b", response)
            or "yes, but not sure about how many" in response.lower()
            or re.findall(r"\bno\b", response.lower())
        )

    # Batched retry loop
    max_retry_rounds = 3
    for retry_round in range(max_retry_rounds):
      # Find indices that need retry
      retry_indices = []
      retry_prompts = []
      retry_messages = []  # Store the retry user messages for convos
      
      for i, response in enumerate(batch_responses):
        if needs_retry(response, self.eval_mode):
          retry_indices.append(i)
          # Append the failed response and retry message to the prompt
          if response != "":
            batch_prompts[i].append(
                {"role": self.model_role_name, "content": response}
            )
            retry_msg = ""
          else:
            retry_msg = "Your thinking exceededed the response length limit, or you did not properly generate thinking. Please try again."
          if self.eval_mode == "mc":
            retry_msg += (
                "Could not parse response. Generate exactly one of"
                ' "Question: Is Alice [attribute]?" or "End questioning"'
                " and nothing else."
            )
          elif self.eval_mode == "isambig":
            # retry_msg = (
            #     'Wrong format. Please answer either "Answer: Yes" or'
            #     ' "Answer: No" or "Answer: Not sure" and nothing else.'
            # )
            retry_msg += (
                'Wrong format. Please answer either "Answer: No" or "Answer: 1" or "Answer: 2" or "Answer: 3" or "Answer: 4" or "Answer: Yes, but not sure about how many" and nothing else.'
            )
          elif self.eval_mode == "fullinfo":
            retry_msg += (
                'Wrong format. Please answer either "Answer: Yes" or'
                ' "Answer: No" and nothing else.'
            )
          batch_prompts[i].append({"role": "user", "content": retry_msg})
          retry_prompts.append(batch_prompts[i])
          retry_messages.append(retry_msg)
      
      if not retry_indices:
        break  # All responses are valid
      
      if retry_round == max_retry_rounds - 1:
        print(f"Max retries reached for {len(retry_indices)} responses")
        break
      
      # Batch generate retries
      retry_responses, retry_num_tokens, retry_cots, all_costs = cached_generate(
          retry_prompts,
          model_name,
          port=self.vllm_port,
          cache=cache,
          cache_file=cache_file,
          generation_config=self.generation_config,
      )
      
      # Update responses, cots, and conversations
      for idx, retry_idx in enumerate(retry_indices):
        batch_responses[retry_idx] = retry_responses[idx]
        all_cots[retry_idx] = retry_cots[idx]
        all_num_thinking_tokens[retry_idx] = retry_num_tokens[idx]
        # Add retry exchange to conversation
        batch_convos[retry_idx].append({
            "role": "user",
            "text": retry_messages[idx]
        })
        batch_convos[retry_idx].append({
            "role": self.model_role_name,
            "text": retry_responses[idx]
        })

    # Parse final responses
    batch_correct = []
    for i, response in enumerate(batch_responses):
      if "End questioning" in response:
        response = "End questioning"
      else:
        response = response.split("Question:")[-1].strip()
        
      if self.eval_mode == "mc":
        if not (
            re.findall(r"Is Alice \[?([ \w-]+)\]?\?", response)
            or "end questioning" in response.lower()
        ):
          print(f"Could not parse response: {response}")
          response = {"None"}
        else:
          if "end questioning" in response.lower():
            response = {"End questioning"}
          else:
            response = set(re.findall(r"Is Alice \[?([ \w-]+)\]?\?", response))
      else:
        # Handle fullinfo (yes/no) and isambig (no/1/2/3/4) modes
        if self.eval_mode == "isambig":
          # isambig expects: "Answer: No" or "Answer: 1/2/3/4" or "Answer: Yes, but not sure about how many"
          orig_response = response.lower()
          
          # Check for numeric answer first
          number_match = re.findall(r"answer:\s*([1-4])\b", orig_response)
          if number_match:
            response = number_match[-1]  # Use the last match (after "Answer:")
          elif "answer: no" in orig_response or re.search(r"\bno\b.*ambiguous", orig_response):
            response = "no"
          elif "yes, but not sure" in orig_response:
            response = "not sure"
          else:
            # Try to find any standalone number 1-4
            any_number = re.findall(r"\b([1-4])\b", orig_response)
            if any_number:
              response = any_number[-1]
            else:
              print(
                  "No valid isambig answer found in response:"
                  f" {json.dumps(batch_prompts[i])}"
              )
              response = "None"
        else:  # fullinfo mode
          if not re.findall(r"(yes|not sure|no)", response.lower()):
            print(
                "No/bad number found in response:"
                f" {json.dumps(batch_prompts[i])}"
            )
            response = "None"
          else:
            orig_response = response
            first_line = orig_response.split("\n")[0]
            processed_response = first_line + (
                response.lower().split("answer")[-1]
            )
            all_yes = "yes" in processed_response
            all_no = "no" in processed_response
            all_not_sure = "not sure" in processed_response
            if all_yes and not all_no and not all_not_sure:
              response = "yes"
            elif all_no and not all_not_sure:
              response = "no"
            elif all_not_sure:
              response = "not sure"
            else:
              print(
                  f"No answer found in response: {orig_response} \n for prompt:"
                  f" {json.dumps(batch_prompts[i])}"
              )
      
      batch_responses[i] = response
      is_match = (
          any(response == set(gt) for gt in batch_gt_queries[i])
          if self.eval_mode == "mc"
          else response == batch_gt_queries[i]
      )
      batch_correct.append(is_match)
    
    return batch_convos, batch_responses, batch_correct, all_num_thinking_tokens, all_cots

  def parse_rules(self, rules):
    """Parses a list of SimpleLogic rules into a natural language format.

    Args:
      rules: A list of rules, where each rule is a string of the form "attribute
        verb proposition" or "not attribute verb proposition".

    Returns:
      A string of natural language rules, where each rule is of the form
      "If Alice is attribute, then Alice is proposition".
    """
    rules_nl = []
    for rule in rules:
      negated_words = [
          word.split("not ")[-1] for word in rule if word.startswith("not ")
      ]
      positive_words = [word for word in rule if not word.startswith("not ")]
      assert len(positive_words) == 1
      premises = " and ".join(negated_words)
      conclusion_word = positive_words[0]
      rules_nl.append(
          f"If Alice is {premises}, then Alice is {conclusion_word}."
      )
    return "\n".join(sorted(rules_nl))

  def make_batches(self, data, prompt_mode, batch_size=None):
    """Make data batches for Logic-Q.

    Args:
      data: The data to make batches from.
      batch_size: The batch size to use.

    Returns:
      The batch of requests, system prompts, ground truth queries, and batch
    ids.
    """
    if batch_size is None:
      batch_size = self.batch_size
    batch_ids = [[]]
    batch_system_prompts = [[]]
    batch_requests = [[]]
    batch_gt_queries = [[]]
    for d, (_, datum) in enumerate(data.iterrows()):
      rules_nl = self.parse_rules(datum["rules"])

      known_facts = sorted(
          [f"Alice is {attr}." for attr in datum["known_facts"]]
      )
      known_untrue_facts = sorted(
          [f"Alice is not {attr}." for attr in datum["known_untrue_facts"]]
      )
      if self.eval_mode == "mc":
        if self.forbid_alternatives:
          all_forbid_alternative_variables = sorted((set(sum(datum["all_alternative_gt_qs"], [])) - set(sum(datum["gt_qs"], []))) | set(datum["cannot_ask_facts"]))
          invalid_qs = sorted([
              f"You may not ask if Alice is {attr}." for attr in all_forbid_alternative_variables
          ])
        else:
          invalid_qs = sorted([
              f"You may not ask if Alice is {attr}."
              for attr in datum["cannot_ask_facts"]
          ])
        invalid_qs = "\n".join(sorted(set(invalid_qs)))
        assert not set(known_facts).intersection(set(known_untrue_facts))
        known_facts = "\n".join(known_facts)
        known_untrue_facts = "\n".join(sorted(set(known_untrue_facts)))

        if len(batch_requests[-1]) >= batch_size:
          batch_requests.append([])
          batch_system_prompts.append([])
          batch_gt_queries.append([])
          batch_ids.append([])

        if self.fs_samples == 0:
          if prompt_mode == "at_most_k":
            if str(datum["k"]) == "1":
              system_prompt = self.prompts["mc"]["system_prompt"]["vanilla_k1"]
            elif str(datum["k"]) == "2":
              system_prompt = self.prompts["mc"]["system_prompt"]["vanilla_k2"]
            elif str(datum["k"]) == "3":
              system_prompt = self.prompts["mc"]["system_prompt"]["vanilla_k3"]
            elif str(datum["k"]) == "4":
              system_prompt = self.prompts["mc"]["system_prompt"]["vanilla_k4"]
            else:
              raise Exception(f"Invalid k value: {datum['k']}")
              # continue
            system_prompt = system_prompt.format(rules_nl=rules_nl)
          elif prompt_mode == "exact_k":
            system_prompt = self.prompts["mc"]["system_prompt"]["vanilla_exact_k"].format(**{"k": str(datum["k"]), "rules_nl": rules_nl})
          elif prompt_mode == "at_most_K":
            system_prompt = self.prompts["mc"]["system_prompt"]["vanilla_at_most_K"].format(**{"max_k": str(4), "rules_nl": rules_nl})
          batch_system_prompts[-1].append(system_prompt)
          batch_requests[-1].append(
              # self.request.format(
              self.user_prompt.format(
                  known_facts=known_facts,
                  known_untrue_facts=known_untrue_facts,
                  invalid_qs=invalid_qs,
                  goal=datum["goal"],
              )
          )
        else:
          batch_system_prompts[-1].append(None)
          batch_requests[-1].append(
              # self.request.format(
              self.user_prompt.format(
                  rules_nl=rules_nl,
                  known_facts=known_facts,
                  known_untrue_facts=known_untrue_facts,
                  invalid_qs=invalid_qs,
                  goal=datum["goal"],
              )
          )

        batch_ids[-1].append(d)
        if self.forbid_alternatives:
          batch_gt_queries[-1].append(datum["gt_qs"])
        else:  # NOTE: If we do not tell the model about "invalid sets of facts (that actually also work)", then we consider those also as valid queries
          batch_gt_queries[-1].append(datum["all_alternative_gt_qs"])
      else:
        # fullinfo/isambig modes for Logic-Q-multi format
        original_known_facts = known_facts
        original_known_untrue_facts = known_untrue_facts
        
        if self.eval_mode == "isambig":
          # isambig mode: same partial info as mc, but ask how many variables are missing
          # Ground truth is k (number of missing variables)
          known_facts = sorted(original_known_facts)
          known_untrue_facts = sorted(original_known_untrue_facts)
          assert not set(known_facts).intersection(set(known_untrue_facts))
          known_facts_nl = "\n".join(known_facts)
          known_untrue_facts_nl = "\n".join(sorted(set(known_untrue_facts)))
          
          # Ground truth is the k value (number of missing variables)
          k_value = int(datum["k"])
          if k_value == 0:
            goal_is_true = "no"  # Not ambiguous
          else:
            goal_is_true = str(k_value)  # "1", "2", "3", or "4"
          
          if len(batch_requests[-1]) >= batch_size:
            batch_requests.append([])
            batch_system_prompts.append([])
            batch_gt_queries.append([])
            batch_ids.append([])
          
          if self.fs_samples == 0:
            system_prompt = self.prompts["isambig"]["system_prompt"].format(rules_nl=rules_nl)
            batch_system_prompts[-1].append(system_prompt)
            batch_requests[-1].append(
                self.user_prompt.format(
                    known_facts=known_facts_nl,
                    known_untrue_facts=known_untrue_facts_nl,
                    goal=datum["goal"],
                )
            )
          else:
            batch_system_prompts[-1].append(None)
            batch_requests[-1].append(
                self.user_prompt.format(
                    rules_nl=rules_nl,
                    known_facts=known_facts_nl,
                    known_untrue_facts=known_untrue_facts_nl,
                    goal=datum["goal"],
                )
            )
          
          batch_ids[-1].append(d)
          batch_gt_queries[-1].append(goal_is_true)
        
        else:  # fullinfo mode
          # Sample one variable set from gt_qs (the ground truth query sets)
          # These have corresponding derivations in gt_q_to_derivations_min_rules
          if datum["gt_qs"] and len(datum["gt_qs"]) > 0:
            sampled_var_set = random.choice(datum["gt_qs"])
          else:
            # Skip if no variable sets available
            continue
          
          # Get the derivations dict to look up target values
          derivations_dict = datum["gt_q_to_derivations_min_rules"]
          if isinstance(derivations_dict, list):
            # It's a list of dicts, combine them
            combined_derivations = {}
            for d_item in derivations_dict:
              combined_derivations.update(d_item)
            derivations_dict = combined_derivations
          
          # Generate one random assignment for the sampled variable set
          # (sample one of the possible 2^k truth assignments)
          import itertools
          var_list = list(sampled_var_set)
          k = len(var_list)
          
          # Sample one random truth assignment
          truth_values = [random.choice([True, False]) for _ in range(k)]
          
          # Build the key for lookup in derivations_dict
          literals = []
          for var, is_true in zip(var_list, truth_values):
            if is_true:
              literals.append(var)
            else:
              literals.append(f"not {var}")
          
          # Create the key in the format used by gt_q_to_derivations_min_rules
          # Keys can have arbitrary ordering, so we search for a key with matching literals
          literals_set = frozenset(literals)
          
          # Try to find the derivation for this assignment
          target_value = None
          for existing_key, payload in derivations_dict.items():
            try:
              existing_literals = json.loads(existing_key)
              if frozenset(existing_literals) == literals_set:
                target_value = payload.get("target_value", None)
                break
            except (json.JSONDecodeError, TypeError):
              continue
          
          if target_value is None:
            # Cannot determine ground truth, skip this problem
            continue
          
          # Determine if goal is true or false based on target_value
          if target_value == datum["goal"]:
            goal_is_true = "yes"
          elif target_value == f"not {datum['goal']}":
            goal_is_true = "no"
          else:
            # Unexpected target value format
            continue
          
          # Add the sampled variables to known_facts/known_untrue_facts
          known_facts = copy.deepcopy(original_known_facts)
          known_untrue_facts = copy.deepcopy(original_known_untrue_facts)
          
          for var, is_true in zip(var_list, truth_values):
            if is_true:
              known_facts.append(f"Alice is {var}.")
            else:
              known_untrue_facts.append(f"Alice is not {var}.")
          
          known_facts = sorted(known_facts)
          known_untrue_facts = sorted(known_untrue_facts)
          assert not set(known_facts).intersection(set(known_untrue_facts))
          known_facts_nl = "\n".join(known_facts)
          known_untrue_facts_nl = "\n".join(sorted(set(known_untrue_facts)))
          
          if len(batch_requests[-1]) >= batch_size:
            batch_requests.append([])
            batch_system_prompts.append([])
            batch_gt_queries.append([])
            batch_ids.append([])
          
          if self.fs_samples == 0:
            system_prompt = self.prompts["fullinfo"]["system_prompt"].format(rules_nl=rules_nl)
            batch_system_prompts[-1].append(system_prompt)
            batch_requests[-1].append(
                self.user_prompt.format(
                    known_facts=known_facts_nl,
                    known_untrue_facts=known_untrue_facts_nl,
                    goal=datum["goal"],
                )
            )
          else:
            batch_system_prompts[-1].append(None)
            batch_requests[-1].append(
                self.user_prompt.format(
                    rules_nl=rules_nl,
                    known_facts=known_facts_nl,
                    known_untrue_facts=known_untrue_facts_nl,
                    goal=datum["goal"],
                )
            )
          
          batch_ids[-1].append(d)
          batch_gt_queries[-1].append(goal_is_true)

    return batch_ids, batch_system_prompts, batch_requests, batch_gt_queries

  def make_fewshot_turns(self, fewshot_data):
    """Make few-shot turns for Logic-Q.

    Args:
      fewshot_data: The few-shot data to make few-shot turns from.

    Returns:
      The few-shot turns for the prompt.
    """

    fewshot_turns = []
    for d, (_, datum) in enumerate(fewshot_data.iterrows()):
      if d >= self.fs_samples:
        break
      rules_nl = self.parse_rules(datum["rules"])

      known_facts = [f"Alice is {attr}." for attr in datum["known_facts"]]
      known_untrue_facts = [
          f"Alice is not {attr}." for attr in datum["known_untrue_facts"]
      ]
      if self.forbid_alternatives:
        all_forbid_alternative_variables = sorted((set(sum(datum["all_alternative_gt_qs"], [])) - set(sum(datum["gt_qs"], []))) | set(datum["cannot_ask_facts"]))
        invalid_qs = [
            f"You may not ask if Alice is " + (", ".join([attr for attr in attrs])) + "."
            for attrs in all_forbid_alternative_variables
        ]
      else:
        invalid_qs = [
            f"You may not ask if Alice is {attr}."
            for attr in datum["cannot_ask_facts"]
        ]
      invalid_qs = "\n".join(sorted(set(invalid_qs)))
      assert not set(known_facts).intersection(set(known_untrue_facts))

      if self.eval_mode == "mc":
        known_facts = "\n".join(known_facts)
        known_untrue_facts = "\n".join(sorted(set(known_untrue_facts)))
        
        random_gt_attr = random.choice(datum["gt_qs"])
        fewshot_turns.append([
            {
                "role": "user",
                # "content": self.request.format(
                "content": self.user_prompt.format(
                    rules_nl=rules_nl,
                    known_facts=known_facts,
                    known_untrue_facts=known_untrue_facts,
                    invalid_qs=invalid_qs,
                    goal=datum["goal"],
                ),
            },
            {
                "role": self.model_role_name,
                "content": f"Question: Is Alice {random_gt_attr}?",
            },
        ])
      else:  # fullinfo/isambig
        # Logic-Q-multi format handling
        if self.eval_mode == "isambig":
          # isambig mode: use k as the ground truth
          k_value = int(datum["k"])
          if k_value == 0:
            goal_is_true = "No"
          else:
            goal_is_true = str(k_value)
          
          known_facts = sorted(known_facts)
          known_untrue_facts = sorted(known_untrue_facts)
          known_facts_nl = "\n".join(known_facts)
          known_untrue_facts_nl = "\n".join(sorted(set(known_untrue_facts)))
          
          fewshot_turns.append([
              {
                  "role": "user",
                  "content": self.user_prompt.format(
                      rules_nl=rules_nl,
                      known_facts=known_facts_nl,
                      known_untrue_facts=known_untrue_facts_nl,
                      goal=datum["goal"],
                  ),
              },
              {
                  "role": self.model_role_name,
                  "content": f"Answer: {goal_is_true}",
              },
          ])
        else:  # fullinfo mode
          # Sample one variable set from gt_qs
          if datum["gt_qs"] and len(datum["gt_qs"]) > 0:
            sampled_var_set = datum["gt_qs"][d % len(datum["gt_qs"])]
          else:
            continue
          
          # Get the derivations dict
          derivations_dict = datum["gt_q_to_derivations_min_rules"]
          if isinstance(derivations_dict, list):
            combined_derivations = {}
            for d_item in derivations_dict:
              combined_derivations.update(d_item)
            derivations_dict = combined_derivations
          
          var_list = list(sampled_var_set)
          # Use consistent truth assignment based on datum index
          truth_values = [(d + i) % 2 == 0 for i in range(len(var_list))]
          
          literals = []
          for var, is_true in zip(var_list, truth_values):
            if is_true:
              literals.append(var)
            else:
              literals.append(f"not {var}")
          
          # Search for a key with matching literals (order-independent)
          literals_set = frozenset(literals)
          target_value = None
          for existing_key, payload in derivations_dict.items():
            try:
              existing_literals = json.loads(existing_key)
              if frozenset(existing_literals) == literals_set:
                target_value = payload.get("target_value", None)
                break
            except (json.JSONDecodeError, TypeError):
              continue
          
          if target_value is None:
            continue
          
          if target_value == datum["goal"]:
            goal_is_true = "Yes"
          elif target_value == f"not {datum['goal']}":
            goal_is_true = "No"
          else:
            continue
          
          # Add fact variables to the known facts
          for var, is_true in zip(var_list, truth_values):
            if is_true:
              known_facts.append(f"Alice is {var}.")
            else:
              known_untrue_facts.append(f"Alice is not {var}.")
          
          known_facts = sorted(known_facts)
          known_untrue_facts = sorted(known_untrue_facts)
          known_facts_nl = "\n".join(known_facts)
          known_untrue_facts_nl = "\n".join(sorted(set(known_untrue_facts)))
          
          fewshot_turns.append([
              {
                  "role": "user",
                  "content": self.user_prompt.format(
                      rules_nl=rules_nl,
                      known_facts=known_facts_nl,
                      known_untrue_facts=known_untrue_facts_nl,
                      goal=datum["goal"],
                  ),
              },
              {
                  "role": self.model_role_name,
                  "content": f"Answer: {goal_is_true}",
              },
          ])
    
    # shuffle the ordering of the few-shot turns
    # (move user, assistant pairs together)
    random.shuffle(fewshot_turns)
    # flatten the list of lists
    fewshot_prefix = []
    for sublist in fewshot_turns:
      for turn in sublist:
        fewshot_prefix.append(turn)
    fewshot_prefix = [
        {
            "role": "system",
            "content": self.system_prompt,
        },
        *fewshot_prefix,
    ]
    
    return fewshot_prefix

  def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame, prompt_mode: str = "exact_k"):
    """Evaluates LLMs on Logic-Q data.

    Args:
      data: The data to evaluate.
      prompt_data: The prompt data to evaluate.

    Returns:
      The evaluation results.
    """
    for k in [
        "known_facts",
        "known_untrue_facts",
        "cannot_ask_facts",
        "rules",
        "all_qs",
        "all_valid_qs",
        "gt_qs",
        # "gt_q_to_true_derivation",
        # "gt_q_to_false_derivation",
        "gt_q_to_derivations_min_rules",
        "gt_q_to_derivations_min_depth",
        "all_alternative_gt_qs",
    ]:
      data[k] = data[k].apply(ast.literal_eval)
      if prompt_data is not None:
        try:
          prompt_data[k] = prompt_data[k].apply(ast.literal_eval)
        except (ValueError, KeyError):
          continue

    results = pd.DataFrame(
        columns=[
            "k",
            "correct",
            "max_depth",
            "min_num_rules_needed",
            "num_constraints",
            "num_vars",
            "pred_q",
            "gt_qs",
            "all_qs",
            "all_valid_qs",
            # "gt_q_to_true_derivation",
            # "gt_q_to_false_derivation",
            "gt_q_to_derivations_min_rules",
            "gt_q_to_derivations_min_depth",
            "all_alternative_gt_qs",
            "conversation",
        ]
    )
    total_cost = []
    all_cots = []

    fs_turns = self.make_fewshot_turns(prompt_data) if self.fs_samples > 0 else []
    batch_ids, batch_system_prompts, batch_requests, batch_gt_queries = (
        self.make_batches(data, prompt_mode)
    )
    pbar = tqdm.tqdm(
        zip(batch_ids, batch_system_prompts, batch_requests, batch_gt_queries),
        total=len(batch_ids),
    )
    for i, (batch_id, batch_system_prompt, batch_request, batch_gt_query) in enumerate(pbar):
      batch_conversation, batch_generated_q, batch_correct, num_thinking_tokens, cots = (
          self.evaluate_batch(
              batch_request,
              batch_system_prompt,
              model_name=self.model_name,
              batch_gt_queries=batch_gt_query,
              cache=self.cache,
              cache_file=self.cache_file,
              fs_turns=fs_turns,
          )
      )
      total_cost += num_thinking_tokens  # num_thinking_tokens is your "cost"
      all_cots += cots
      for i, item_id in enumerate(batch_id):
        datum = data.iloc[item_id]

        results.loc[len(results)] = [
            datum["k"],
            batch_correct[i],
            datum["max_depth"],
            datum["min_num_rules_needed"],
            datum["num_constraints"],
            datum["num_vars"],
            batch_generated_q[i],
            batch_gt_query[i],
            datum["all_qs"],
            datum["all_valid_qs"],
            # datum["gt_q_to_true_derivation"],
            # datum["gt_q_to_false_derivation"],
            datum["gt_q_to_derivations_min_rules"],
            datum["gt_q_to_derivations_min_depth"],
            datum["all_alternative_gt_qs"],
            batch_conversation[i],
        ]
      pbar.set_description(
          f"Accuracy: {sum(results['correct']) / len(results)}"
      )

    print(f"Final accuracy: {sum(results['correct']) / len(results)}")
    print(
        "Accuracy by depth:",
        results.groupby("max_depth").agg({"correct": "mean"}),
    )
    print(f"Total cost: {total_cost}")
    return results, all_cots, total_cost
