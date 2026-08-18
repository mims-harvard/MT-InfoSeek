from __future__ import annotations
import ast
import hashlib
import json
import random
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import tqdm

from gsme_q_mt.evaluators.evaluator import Evaluator
from gsme_q_mt.model_utils import cached_generate


# -----------------------------
# Small utilities (safe parsing)
# -----------------------------
def _stable_seed(s: str) -> int:
  if s is None:
    s = ""
  h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
  return int(h[:8], 16)


def _safe_json_loads(x: Any, default):
  if x is None:
    return default
  if isinstance(x, (list, dict)):
    return x
  try:
    return json.loads(x)
  except Exception:
    return default

def _safe_dict_field(x: Any) -> Dict[str, Any]:
  if x is None:
    return {}
  if isinstance(x, dict):
    return x
  if isinstance(x, str):
    s = x.strip()
    if not s:
      return {}
    try:
      obj = json.loads(s)
      return obj if isinstance(obj, dict) else {}
    except Exception:
      try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
      except Exception:
        return {}
  return {}

def _safe_list_field(x: Any) -> List[Any]:
  if x is None:
    return []
  if isinstance(x, list):
    return x
  if isinstance(x, (set, tuple)):
    return list(x)
  if isinstance(x, (dict,)):
    return list(x.keys())
  if not isinstance(x, str):
    return []
  s = x.strip()
  if not s:
    return []
  try:
    return json.loads(s)
  except Exception:
    try:
      return ast.literal_eval(s)
    except Exception:
      return []


def _infer_k_from_row(datum: pd.Series) -> int:
  for key in ["k", "Missing_Count", "missing_count", "missing_cnt"]:
    if key in datum and pd.notna(datum[key]):
      try:
        k = int(datum[key])
        return max(1, k)
      except Exception:
        pass
  return 1




# -----------------------------
# cached_generate return adapters
# -----------------------------
def _as_cost_usd_list(costs_usd: Any, n: int) -> List[float]:
  if n <= 0:
    return []
  if costs_usd is None:
    return [0.0] * n
  if isinstance(costs_usd, list):
    out: List[float] = []
    for i in range(n):
      try:
        v = costs_usd[i]
      except Exception:
        v = None
      try:
        out.append(float(v) if v is not None else 0.0)
      except Exception:
        out.append(0.0)
    return out
  try:
    v = float(costs_usd)
    return [v] * n
  except Exception:
    return [0.0] * n


def _as_thinking_list(thinking_tokens: Any, n: int) -> List[int]:
  if n <= 0:
    return []
  if thinking_tokens is None:
    return [0] * n
  if isinstance(thinking_tokens, list):
    out: List[int] = []
    for i in range(n):
      try:
        v = thinking_tokens[i]
      except Exception:
        v = None
      try:
        out.append(int(v) if v is not None else 0)
      except Exception:
        out.append(0)
    return out
  try:
    v = int(thinking_tokens)
    return [v] * n
  except Exception:
    return [0] * n


# -----------------------------
# Evaluator
# -----------------------------
class GSMEvaluator(Evaluator):
  """Evaluator for LLMs on the released GSME-Q datasets."""

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      eval_mode: str = "mc",
      batch_size: int = 1,
      reveal_k_in_prompt: bool = False,
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
    self.reveal_k_in_prompt = reveal_k_in_prompt
    self.batch_size = batch_size

    # sc prompt
    self.assist_sc_prompt = """You are solving a math problem.
Decide how many key variables required to solve the problem are missing from the given information.
Choose exactly one option from {0,1,2,3,4}.
Output format:
Choice: <0-4>
"""
    self.user_sc_prompt = """Math problem: {request}"""

    # mc prompts (k-aware)
    self.assist_mc_prompt_k = """You are trying to solve a math problem. You must decide whether you have enough information to solve the math problem.
If you do not have enough information, you may ask questions back to the user from the predefined "Possible questions". Otherwise, choose "No questions needed."
Output format:
- If you want to ask: "Choice: <number_1>, <number_2>, ..." (all questions needed to solve this problem)
- If no questions needed: choose the option that says "No questions needed." and output its number as "Choice: <number>"
"""
    self.assist_mc_prompt_give_k = """You are solving a math problem. Exactly {k} key variables required to solve the problem are missing from the given information.
Your task is to identify which {k} missing variables they are by selecting the corresponding questions from the predefined "Possible questions" list.
Output format:
- If you want to ask: "Choice: <number_1>, <number_2>, ..." (all questions needed to solve this problem)
- If no questions needed: choose the option that says "No questions needed." and output its number as "Choice: <number>"
"""

    self.assist_isambig_prompt = """You are trying to answer a math question. Please answer with "Answer:" followed by the answer to the math question, or "Not sure" if you are not sure what the answer is. Only include the raw numerical answer, do not include any units or thousands separators."""
    self.assist_fullinfo_prompt = """You are trying to answer a math question. Please answer with "Answer:" followed by the answer to the math question. Only include the raw numerical answer, do not include any units or thousands separators."""

    self.user_mc_prompt = """Math problem: {request}

Possible questions:
{possible_qs}"""
    self.user_isambig_prompt = """Math problem: {request}"""
    self.user_fullinfo_prompt = """Math problem: {request}"""

    if self.eval_mode == "mc":
      self.user_prompt = self.user_mc_prompt
      self.assist_prompt = None  # per-example prompt
    elif self.eval_mode == "sc":
      self.assist_prompt = self.assist_sc_prompt + (
          " Reason step-by-step, then generate one of the above outputs."
          if (self.use_cot)
          else " Generate one of the above outputs and nothing else."
      )
      self.user_prompt = self.user_sc_prompt
    elif self.eval_mode == "isambig":
      self.assist_prompt = self.assist_isambig_prompt + (
          " Reason step-by-step, then generate one of the above outputs."
          if (self.use_cot)
          else " Generate one of the above outputs and nothing else."
      )
      self.user_prompt = self.user_isambig_prompt
    else:
      self.assist_prompt = self.assist_fullinfo_prompt + (
          " Reason step-by-step, then generate one of the above outputs."
          if (self.use_cot)
          else " Generate one of the above outputs and nothing else."
      )
      self.user_prompt = self.user_fullinfo_prompt

  def _build_mc_system_prompt(self, k: int) -> str:
    if self.reveal_k_in_prompt:
      p = self.assist_mc_prompt_give_k.format(k=int(k))
    else:
      p = self.assist_mc_prompt_k
    if self.use_cot:
      p += " Reason step-by-step, then generate one of the above outputs."
    else:
      p += " Generate one of the above outputs and nothing else."
    return p

  def _parse_choice_set(self, response: str) -> Optional[Set[int]]:
    if response is None:
      return None
    text = response.strip()
    lower = text.lower()
    if "choice" in lower:
      lower = lower.split("choice:")[-1]
      text_after = lower
    else:
      text_after = lower
    nums = re.findall(r"\b[0-9]+\b", text_after)
    if not nums:
      return None
    try:
      return set(int(x) for x in nums)
    except Exception:
      return None

  def _parse_choice_int(self, response: str, valid: Optional[Set[int]] = None) -> Optional[int]:
    if response is None:
      return None
    text = response.strip().lower()
    if "choice:" in text:
      text = text.split("choice:")[-1]
    nums = re.findall(r"\b[0-9]+\b", text)
    if len(nums) != 1:
      return None
    try:
      v = int(nums[0])
    except Exception:
      return None
    if valid is not None and v not in valid:
      return None
    return v

  def _needs_retry(self, response: str) -> bool:
    if self.eval_mode == "mc":
      return self._parse_choice_set(response) is None
    if self.eval_mode == "sc":
      return self._parse_choice_int(response) is None
    return not re.findall(r"(not sure|\b[0-9]+\b)", (response or "").lower())

  def evaluate_batch(
      self,
      batch_requests: List[str],
      batch_system_prompts: List[Optional[str]],
      batch_gt_queries,
      batch_k,
      model_name: str,
      model_url: str,
      cache=None,
      cache_file=None,
  ):
    """Returns:
      batch_convos, batch_preds, batch_correct, think_tokens_list, cots_list, cost_usd_list
    """
    batch_prompts = []
    for request, system_prompt in zip(batch_requests, batch_system_prompts):
      msgs = []
      if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
      msgs.append({"role": "user", "content": request})
      batch_prompts.append(msgs)

    # initial generate
    batch_responses, think_token_num, all_cots, cost_usd = cached_generate(
        batch_prompts,
        model_name,
        model_url,
        cache=cache,
        cache_file=cache_file,
        generation_config=self.generation_config,
        parallel_model_calls=self.parallel_model_calls,
    )

    n = len(batch_requests)
    cost_acc = _as_cost_usd_list(cost_usd, n)
    think_acc = _as_thinking_list(think_token_num, n)

    # normalize cots
    if all_cots is None:
      cot_last: List[Any] = [None] * n
    elif isinstance(all_cots, list):
      cot_last = list(all_cots) + [None] * max(0, n - len(all_cots))
      cot_last = cot_last[:n]
    else:
      cot_last = [all_cots] * n

    # build conversations
    batch_convos = []
    for i, prompt in enumerate(batch_prompts):
      convo = [{"role": m["role"], "text": m["content"]} for m in prompt]
      convo.append({"role": self.model_role_name, "text": batch_responses[i]})
      batch_convos.append(convo)

    # batched retry, with accumulation
    max_retry_rounds = 3
    for retry_round in range(max_retry_rounds):
      retry_indices = []
      retry_prompts = []
      retry_msgs = []

      for i, resp in enumerate(batch_responses):
        if not self._needs_retry(resp):
          continue

        retry_indices.append(i)

        # append assistant response then corrective user instruction
        batch_prompts[i].append({"role": self.model_role_name, "content": resp})

        if self.eval_mode == "mc":
          msg = (
              'Wrong format or option not found. Output exactly "Choice: <number>" '
              'or "Choice: <number_1>, <number_2>, ..." and nothing else.'
          )
        elif self.eval_mode == "sc":
          msg = 'Wrong format. Output exactly "Choice: <number>" and nothing else.'
        elif self.eval_mode == "fullinfo":
          msg = 'Wrong format. Output exactly "Answer: <number>" (raw number only) and nothing else.'
        else:
          msg = 'Wrong format. Output exactly "Answer: <number>" or "Answer: Not sure" and nothing else.'

        batch_prompts[i].append({"role": "user", "content": msg})
        retry_prompts.append(batch_prompts[i])
        retry_msgs.append(msg)

      if not retry_indices:
        break

      retry_responses, retry_think, retry_cots, retry_cost = cached_generate(
          retry_prompts,
          model_name,
          model_url,
          cache=cache,
          cache_file=cache_file,
          generation_config=self.generation_config,
          parallel_model_calls=self.parallel_model_calls,
      )

      retry_cost_list = _as_cost_usd_list(retry_cost, len(retry_indices))
      retry_think_list = _as_thinking_list(retry_think, len(retry_indices))

      # normalize retry cots to list
      if retry_cots is None:
        retry_cots_list = [None] * len(retry_indices)
      elif isinstance(retry_cots, list):
        retry_cots_list = list(retry_cots) + [None] * max(0, len(retry_indices) - len(retry_cots))
        retry_cots_list = retry_cots_list[: len(retry_indices)]
      else:
        retry_cots_list = [retry_cots] * len(retry_indices)

      for j, orig_i in enumerate(retry_indices):
        batch_responses[orig_i] = retry_responses[j]

        cost_acc[orig_i] += float(retry_cost_list[j] or 0.0)
        think_acc[orig_i] += int(retry_think_list[j] or 0)

        if retry_cots_list[j] is not None:
          cot_last[orig_i] = retry_cots_list[j]

        batch_convos[orig_i].append({"role": "user", "text": retry_msgs[j]})
        batch_convos[orig_i].append({"role": self.model_role_name, "text": retry_responses[j]})

    # parse and score
    batch_preds = []
    batch_correct = []

    for i, resp in enumerate(batch_responses):
      if self.eval_mode == "mc":
        pred_set = self._parse_choice_set(resp) or set()
        batch_preds.append(pred_set)

        gt = batch_gt_queries[i]
        gt_sets: List[Set[int]] = []

        if isinstance(gt, int):
          gt_sets = [set([gt])]
        elif isinstance(gt, (set, frozenset)):
          gt_sets = [set(gt)]
        elif isinstance(gt, list):
          if len(gt) == 0:
            gt_sets = [set()]
          elif all(isinstance(x, int) for x in gt):
            gt_sets = [set(gt)]
          elif all(isinstance(x, (list, set, tuple)) for x in gt):
            gt_sets = [set(x) for x in gt]
          else:
            tmp = []
            for x in gt:
              try:
                tmp.append(int(x))
              except Exception:
                pass
            gt_sets = [set(tmp)]
        else:
          try:
            gt_sets = [set([int(gt)])]
          except Exception:
            gt_sets = [set()]

        batch_correct.append(any(pred_set == s for s in gt_sets))

      elif self.eval_mode == "sc":
        pred = self._parse_choice_int(resp)
        if pred is None:
          pred = -1
        batch_preds.append(pred)
        
        gt_k = batch_gt_queries[i]
        try:
          batch_correct.append(int(pred) == int(gt_k))
        except Exception:
          batch_correct.append(False)

      else:
        low = (resp or "").lower()
        ans = low.split("answer:")[-1].strip()
        if "not sure" in ans:
          pred = "Not sure"
          ok = (batch_gt_queries[i] == "Not sure")
        else:
          nums = re.findall(r"\b[0-9]+\b", ans)
          if not nums:
            pred = "None"
            ok = False
          else:
            pred = nums[0]
            try:
              ok = int(pred) == int(batch_gt_queries[i])
            except Exception:
              ok = False
        batch_preds.append(pred)
        batch_correct.append(ok)

    return batch_convos, batch_preds, batch_correct, think_acc, cot_last, cost_acc

  def make_convo_batches(self, data: pd.DataFrame, batch_size: Optional[int] = None):
    if batch_size is None:
      batch_size = self.batch_size

    batch_ids = [[]]
    batch_requests = [[]]
    batch_gt_answers = [[]]
    batch_gt_queries = [[]]
    batch_system_prompts = [[]]
    batch_k = [[]]

    for d, (_, datum) in enumerate(data.iterrows()):
      if self.eval_mode == "mc":
        k = _infer_k_from_row(datum)
        request = datum["Rewritten Problem"]

        variables = _safe_dict_field(datum.get("Variables", "{}"))
        possible_qs = _safe_list_field(datum.get("Possible Questions", "[]"))
        gt_vars = _safe_list_field(datum.get("Heldout Value", "[]"))


        paired = [(str(v), str(v) in set(str(x) for x in gt_vars)) for v in possible_qs]

        uid = str(datum.get("sample_id"))
        rng = random.Random(_stable_seed(uid))
        rng.shuffle(paired)

        questions = []
        var_to_index = {}
        for idx, (var, is_gt) in enumerate(paired):
          var_to_index[var] = idx
          questions.append(f"{idx}. What is the value of {var}?")

        no_q_index = len(questions)
        questions.append(f"{no_q_index}. No questions needed.")

        k_in_row = _infer_k_from_row(datum)
        if k_in_row == 0 or len(gt_vars) == 0:
          gt_set = set([no_q_index])
        else:
          idxs = []
          for var in gt_vars:
            sv = str(var)
            if sv in var_to_index:
              idxs.append(var_to_index[sv])
          gt_set = set(idxs) if len(idxs) > 0 else set([no_q_index])

        if len(batch_requests[-1]) >= batch_size:
          batch_ids.append([])
          batch_requests.append([])
          batch_gt_answers.append([])
          batch_gt_queries.append([])
          batch_system_prompts.append([])
          batch_k.append([])

        batch_ids[-1].append(d)
        batch_requests[-1].append(
            self.user_prompt.format(
                request=request,
                possible_qs="\n".join(questions),
            )
        )
        batch_gt_answers[-1].append(datum.get("Full Answer", None))
        batch_gt_queries[-1].append(list(gt_set))
        batch_system_prompts[-1].append(self._build_mc_system_prompt(k))
        batch_k[-1].append(k)

      elif self.eval_mode == "sc":
        request = datum["Rewritten Problem"]
        heldout = _safe_list_field(datum.get("Heldout Value", "[]"))
        missing_cnt = len(heldout)

        if len(batch_requests[-1]) >= batch_size:
          batch_ids.append([])
          batch_requests.append([])
          batch_gt_answers.append([])
          batch_gt_queries.append([])
          batch_system_prompts.append([])
          batch_k.append([])

        batch_ids[-1].append(d)
        batch_requests[-1].append(self.user_prompt.format(request=request))
        batch_gt_answers[-1].append(datum.get("Full Answer", None))
        batch_gt_queries[-1].append(int(missing_cnt))
        batch_system_prompts[-1].append(self.assist_prompt)
        batch_k[-1].append(int(missing_cnt))

      else:
        is_trues = [True]
        if self.eval_mode == "isambig":
          is_trues = [True, None]

        for is_true in is_trues:
          if is_true is None:
            request = datum["Rewritten Problem"]
            response = "Not sure"
          else:
            request = datum["Full Problem"]
            response = datum["Full Answer"]

          if len(batch_requests[-1]) >= batch_size:
            batch_ids.append([])
            batch_requests.append([])
            batch_gt_answers.append([])
            batch_gt_queries.append([])
            batch_system_prompts.append([])
            batch_k.append([])

          batch_ids[-1].append(d)
          batch_requests[-1].append(self.user_prompt.format(request=request))
          batch_gt_queries[-1].append(response)
          batch_gt_answers[-1].append(datum.get("Full Answer", None))
          batch_system_prompts[-1].append(self.assist_prompt)
          batch_k[-1].append(_infer_k_from_row(datum))

    return batch_ids, batch_system_prompts, batch_requests, batch_gt_answers, batch_gt_queries, batch_k

  

  def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame):
    # list-of-dicts accumulator (fast)
    rows: List[Dict[str, Any]] = []

    (
        batch_ids,
        batch_system_prompts,
        batch_requests,
        batch_gt_answers,
        batch_gt_queries,
        batch_k,
    ) = self.make_convo_batches(data)

    total_think_tokens: List[int] = []
    all_cots: List[Any] = []
    all_cost_usd: List[float] = []

    pbar = tqdm.tqdm(
        zip(batch_ids, batch_system_prompts, batch_requests, batch_gt_answers, batch_gt_queries, batch_k),
        total=len(batch_ids),
    )

    # for progress display
    scored_n = 0
    correct_n = 0

    for batch_id, batch_sys, batch_req, batch_gt_ans, batch_gt_q, batch_k_vals in pbar:
        batch_convo, batch_pred, batch_correct, think_list, cot_list, cost_list = self.evaluate_batch(
            batch_req,
            batch_sys,
            batch_gt_q,
            batch_k=batch_k_vals,
            model_name=self.model_name,
            model_url=self.model_url,
            cache=self.cache,
            cache_file=self.cache_file,
        )


        total_think_tokens.extend(int(x or 0) for x in think_list)
        all_cost_usd.extend(float(x or 0.0) for x in cost_list)
        all_cots.extend(cot_list if isinstance(cot_list, list) else [cot_list] * len(batch_id))

        for i, item_id in enumerate(batch_id):
            datum = data.iloc[item_id]
            equations = _safe_dict_field(datum.get("Equations", "{}"))
            variables = _safe_dict_field(datum.get("Variables", "{}"))

            corr = batch_correct[i]
            # update progress counters (only if scored)
            if corr is not None and not (isinstance(corr, float) and pd.isna(corr)):
                scored_n += 1
                correct_n += int(bool(corr))

            rows.append(
                {
                    "k": batch_k_vals[i],
                    "correct": corr,
                    "max_depth": datum.get("depth", datum.get("max_depth", None)),
                    "pred_answer": None,  # not used here
                    "gt_answer": batch_gt_ans[i],
                    "id": item_id,
                    "request": batch_req[i],
                    "CSP": datum.get("CSP", None),
                    "num_constraints": len(equations) if isinstance(equations, (dict, list)) else None,
                    "num_vars": len(variables) if isinstance(variables, dict) else None,
                    "pred_q": batch_pred[i],
                    "gt_qs": batch_gt_q[i],
                    "all_qs": variables,
                    "conversation": json.dumps(batch_convo[i]),
                    "cost_usd": float(cost_list[i]) if cost_list[i] is not None else 0.0,
                    "thinking_tokens": int(think_list[i]) if think_list[i] is not None else 0,
                }
            )

        if scored_n > 0:
            pbar.set_description(f"Accuracy: {correct_n / scored_n}")

    results = pd.DataFrame.from_records(rows)

    results_filtered = results[results["correct"].notna()] if ("correct" in results.columns) else results
    if len(results_filtered) > 0:
        try:
            print(f"Final accuracy: {float(results_filtered['correct'].mean())}")
        except Exception:
            pass
        if "max_depth" in results.columns:
            try:
                print("Accuracy by depth:", results.groupby("max_depth").agg({"correct": "mean"}))
            except Exception:
                pass

    return results, all_cots, total_think_tokens, all_cost_usd
