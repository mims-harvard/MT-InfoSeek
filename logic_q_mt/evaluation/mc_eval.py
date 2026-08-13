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

"""Single-turn evaluation (mc / isambig / fullinfo) for Logic-Q-MT."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import json

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
for _p in (THIS_DIR, ROOT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from evaluators.simple_logic_new import SimpleLogicEvaluator


def main(user_args) -> None:
  domain_main_name = user_args.domain_name.split("_")[0]
  use_cot = False
  fs_samples = 0
  # use_phys_constraints = False
  # if user_args.prompt_mode == "cot":
  #   use_cot = True
  # elif user_args.prompt_mode == "phys":
  #   use_phys_constraints = True
  if user_args.few_shot_prompt:
    fs_samples = int(user_args.prompt_mode[2:])

  # Make directories for results and cache
  if not os.path.exists(user_args.results_dir):
    os.makedirs(user_args.results_dir)
  model_name_sanitized = user_args.model_name.replace('/', '_')
  if "gpt-oss" in model_name_sanitized:
    model_name_sanitized += f"_{user_args.reasoning_effort}"
  results_dir = os.path.join(user_args.results_dir, f"{user_args.domain_name}/{user_args.eval_mode}/{model_name_sanitized}/{user_args.prompt_mode}")
  cache_dir = os.path.join(results_dir, "cache")
  data_file_base_name = os.path.splitext(os.path.basename(user_args.data_file))[0]
  time_tag = datetime.now().strftime("%m_%d_%H_%M_%S")
  output_file_name_wo_time_tag = f"{data_file_base_name}" + ("forbid_alt" if user_args.forbid_alternatives else "")
  cache_file = os.path.join(cache_dir, f"{output_file_name_wo_time_tag}.jsonl")
  if not os.path.exists(cache_file):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
      pass  # Create empty cache file
  output_file = os.path.join(results_dir, f"{output_file_name_wo_time_tag}.csv")
  if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
      pass  # Create empty output file
  print("Loading Evaluator")
  
  configs = {
    "forbid_alternatives": user_args.forbid_alternatives,
    "reasoning_effort": user_args.reasoning_effort
  }
  if user_args.max_tokens is not None:
    configs["max_tokens"] = user_args.max_tokens
  
  if domain_main_name == "SL":
    evaluator = SimpleLogicEvaluator(
        user_args.model_name,
        cache_file=cache_file,
        use_cot=use_cot,
        fs_samples=fs_samples,
        eval_mode=user_args.eval_mode,
        batch_size=user_args.batch_size,
        model_role_name=user_args.model_role_name,
        # parallel_model_calls=user_args.parallel_model_calls,
        vllm_port=user_args.vllm_port,
        **configs,
    )
    prompt_file = os.path.join(
        user_args.data_dir,
        # "Logic-Q/simplelogic_heldout_1k_prompts.csv",
        "simplelogic_heldout_k_sufficient_prompts_new.csv",
    )
  else:
    raise SystemExit(f"Unknown domain: {domain_main_name}")

  print("Loading Data")
  data_file = user_args.data_file
  with open(data_file, "r") as f:
    data = pd.read_csv(f)
  prompt_data = None
  if os.path.exists(prompt_file):
    with open(prompt_file, "r") as f:
      prompt_data = pd.read_csv(f)

  print("Starting Evaluation")
  results, all_cots, total_cost = evaluator.evaluate_data(data, prompt_data, user_args.prompt_mode)

  with open(output_file, "w") as wf:
    results.to_csv(wf)
  print(f"Wrote to {output_file}")
  
  with open(output_file.replace(".csv", "_cots.json"), "w") as wf:
    json.dump(all_cots, wf)
  
  with open(output_file.replace(".csv", "_cost.json"), "w") as wf:
    json.dump(total_cost, wf)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_name",
      type=str,
      default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
      help=(
          "The name of the model to evaluate. Currently support `gpt-5`, `gemini-2.5-pro`, `gemini-3-flash`, `gemini-3.1-pro`, `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`, `Qwen/Qwen3-4B-Thinking-2507-FP8`, etc."
      ),
  )
  parser.add_argument(
      "--domain_name",
      type=str,
      default="SL",
      choices=["SL"],
      help="Domain name. Only `SL` (Simple Logic / Logic-Q-MT) is supported here.",
  )
  parser.add_argument(
      "--eval_mode",
      type=str,
      default="mc",
      choices=[
          "mc",
          "isambig",
          "fullinfo",
      ],
      help=(
          "Evaluation mode. `mc` is for the multiple choice version of"
          " QuestBench, `isambig` is for evaluating whether the model can"
          " identify the task is ambiguous AND estimate how many variables are missing, and `fullinfo` is for evaluating"
          " the model's performance on the task with the full information"
          " (i.e., no missing information)."
      ),
  )
  parser.add_argument(
      "--data_file", type=str, help="The path to the data file.",
      default=os.path.join(os.environ.get("DATA_DIR", "data"), "logic_q_mt.csv"),
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default=os.environ.get("LOGIC_Q_RP_DIR", "/path/to/questbench_data/Logic-Q/RP/RP"),
      help=(
          "Directory containing the raw QuestBench Logic-Q rulesets and the"
          " few-shot prompts CSV. Only required for few-shot prompting."
      ),
  )
  # parser.add_argument(
  #     "--prompt_mode",
  #     type=str,
  #     choices=["", "cot", "fs4"],
  #     default="",
  #     help="Use vanilla, CoT, or fewshot prompting (with 4 samples).",
  # )
  parser.add_argument(
      "--few_shot_prompt",
      action="store_true",
      help="Whether to use few-shot prompting.",
  )
  parser.add_argument(
      "--results_dir",
      type=str,
      default="results",
      help=(
          "Directory to write results to. Default is `results` in the current"
          " directory."
      ),
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=64,  # Increased to 64 - async allows queuing beyond max_num_seqs
      help="Batch size for evaluation.",
  )
  parser.add_argument(
      "--model_role_name",
      type=str,
      default="assistant",
      help=(
          "The name of the model role. In Gemini, this should be `model`. In"
          " OpenAI, this should be `assistant`. You can use other role names as"
          " needed."
      ),
  )
  parser.add_argument("--forbid_alternatives",
                      action="store_true",
                      help="Whether to forbid the model from providing alternative answers.")
  parser.add_argument(
      "--reasoning_effort",
      type=str,
      default="medium",
      choices=["low", "medium", "high"],
      help="Reasoning effort level for gpt-oss models.",
  )
  parser.add_argument(
      "--max_tokens",
      type=int,
      default=None,
      help="Override maximum output tokens for generation.",
  )
  # parser.add_argument(
  #     "--no_thread_pool",
  #     action="store_false",
  #     dest="parallel_model_calls",
  #     help="Disable thread pool.",
  # )
  parser.add_argument(
      "--vllm_port",
      type=int,
      default=8011,
      help="Port for the VLLM server. Default is 8011.",
  )
  parser.add_argument("--prompt_mode", 
                      type=str, 
                      default="exact_k",
                      help="Prompt mode to use.", choices=["at_most_k", "exact_k", "at_most_K"])
  args = parser.parse_args()
  main(args)
