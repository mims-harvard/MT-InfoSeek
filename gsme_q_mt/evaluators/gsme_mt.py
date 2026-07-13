from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import tqdm

from gsme_q_mt.evaluators.evaluator import Evaluator
from gsme_q_mt.model_utils import cached_generate
import gsme_q_mt.evaluators.utils as ut


# -----------------------------
# Evaluator
# -----------------------------
class GSMOpenEndedEvaluator(Evaluator):
  """Evaluator for LLMs on GSME-Q with multi-turn modes."""

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      eval_mode: str = "mt_all",
      batch_size: int = 1,
      reveal_k_in_prompt: bool = False,
      include_allowed_leaf_in_prompt: bool = False,
      max_turns: int = 2,
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
    self.include_allowed_leaf_in_prompt = bool(include_allowed_leaf_in_prompt)
    self.batch_size = batch_size
    self.max_turns = int(max(1, max_turns))

    # multi-turn system prompt templates
    self.mt_system_prompt = """You are solving a math problem. You must decide whether you have enough information to solve this problem.

Rules:
1) You may ask only about leaf variables (given as candidates).
2) You must NOT ask about the goal variable.
3) You must NOT ask about any non-leaf variable.
4) If you ask an invalid variable, the user will refuse and you must ask again.

When asking, use one of these formats:
- "Question: What is VAR?"
- "Questions: VAR1, VAR2, ..."

When answering, use:
- "Answer: <number>" (raw number only)

Do not output other formats.
"""

    self.mt_one_hint = """Ask for ALL missing variables by asking at most ONE variable per turn. Do not ask extra variables."""
    self.mt_all_hint = """Ask for ALL missing variables in ONE turn. Do not ask extra variables."""

  def _build_mt_system_prompt(
      self,
      allowed_leaf: List[str],
      forbidden_vars: List[str],
      goal_var: str,
      setting: str,
      max_turns: int,
  ) -> str:
    cand = ", ".join(allowed_leaf[:200])
    forb = ", ".join(forbidden_vars[:200])
    base = self.mt_system_prompt
    if setting == "mt_one":
      base += "\n" + self.mt_one_hint
      base += f"Max turns: {int(max_turns)}\n"
    else:
      base += "\n" + self.mt_all_hint
      
    base += f"\n\nGoal variable: {goal_var}\n"
    if self.include_allowed_leaf_in_prompt:
      base += f"Allowed leaf variables: {cand}\n"
    base += f"Forbidden variables: {forb}\n"
    return base


  # -----------------------------
  # Multi-turn oracle
  # -----------------------------
  def _oracle_answer_vars(
      self,
      asked_vars: List[str],
      pred_values: Dict[str, float],
      allowed_leaf: Set[str],
      forbidden: Set[str],
      setting: str,
  ) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Returns:
      oracle_text,
      valid_vars_answered,
      invalid_vars,
      duplicate_vars (within the same action)
    """
    valid: List[str] = []
    invalid: List[str] = []
    dup: List[str] = []

    seen_local: Set[str] = set()
    for v in asked_vars:
      sv = str(v).strip()
      if not sv:
        continue
      if sv in seen_local:
        dup.append(sv)
        continue
      seen_local.add(sv)

      if (sv in forbidden) or (sv not in allowed_leaf):
        invalid.append(sv)
        continue

      valid.append(sv)

    if invalid:
      inv = ", ".join(invalid)
      if setting == "mt_one":
        msg = (
            f"Invalid variables: {inv}. "
            'Either ask ONE allowed leaf variable: "Question: What is VAR?" '
            'or answer now: "Answer: <number>".'
        )
      else:  # mt_all
        msg = (
            f"Invalid variables: {inv}. "
            'Either ask ALL needed allowed leaf variables in one turn: "Questions: VAR1, VAR2, ..." '
            'or answer now: "Answer: <number>".'
        )
      return msg, [], invalid, dup

    if not valid:
      if setting == "mt_one":
        msg = (
            'No valid variable received. '
            'Either ask ONE allowed leaf variable: "Question: What is VAR?" '
            'or answer now: "Answer: <number>".'
        )
      else:  # mt_all
        msg = (
            'No valid variable received. '
            'Either ask ALL needed allowed leaf variables in one turn: "Questions: VAR1, VAR2, ..." '
            'or answer now: "Answer: <number>".'
        )
      return msg, [], [], dup

    # Provide values for valid variables
    parts: List[str] = []
    for sv in valid:
      if sv in pred_values:
        val = pred_values[sv]
        # keep compact formatting
        if abs(val - round(val)) <= 1e-9:
          parts.append(f"{sv} = {int(round(val))}.")
        else:
          parts.append(f"{sv} = {val}.")
      else:
        parts.append(f"I do not have the value for {sv}.")
    return " ".join(parts), valid, [], dup

  # -----------------------------
  # Multi-turn evaluation loop (batched)
  # -----------------------------
  def _evaluate_batch_multiturn(
      self,
      batch_items: List[Dict[str, Any]],
      model_name: str,
      model_url: str,
      cache=None,
      cache_file=None,
  ):
    """
    batch_items: list of dicts produced by make_convo_batches for mt modes.

    Returns:
      batch_convos, batch_preds, batch_correct, think_acc, cot_last, cost_acc
    """
    n = len(batch_items)

    # per-episode state
    messages: List[List[Dict[str, str]]] = []
    convos: List[List[Dict[str, str]]] = []
    pred_vals: List[Dict[str, float]] = []
    gt_answers: List[Optional[float]] = []
    heldout_sets: List[Set[str]] = []
    goal_vars: List[str] = []
    allowed_leaf_sets: List[Set[str]] = []
    forbidden_sets: List[Set[str]] = []
    asked_unique_sets: List[Set[str]] = []
    invalid_asked_sets: List[Set[str]] = []
    duplicate_asked_counts: List[int] = []
    turn_logs: List[List[Dict[str, Any]]] = []

    done: List[bool] = [False] * n
    answered: List[bool] = [False] * n
    turns_used: List[int] = [0] * n

    cost_acc: List[float] = [0.0] * n
    think_acc: List[int] = [0] * n
    cot_last: List[Any] = [None] * n

    # init
    for i, it in enumerate(batch_items):
      msgs = list(it["messages_init"])
      messages.append(msgs)
      convos.append([{"role": m["role"], "text": m["content"]} for m in msgs])

      pred_map = it.get("pred_values", {}) or {}
      pred_vals.append(pred_map)

      gt_answers.append(it.get("gt_answer", None))
      heldout = set(str(x) for x in (it.get("heldout", []) or []))
      heldout_sets.append(heldout)

      gv = str(it.get("goal_var") or "")
      goal_vars.append(gv)

      allowed_leaf_sets.append(set(str(x) for x in (it.get("allowed_leaf", []) or [])))
      forbidden_sets.append(set(str(x) for x in (it.get("forbidden", []) or [])))

      asked_unique_sets.append(set())
      invalid_asked_sets.append(set())
      duplicate_asked_counts.append(0)
      turn_logs.append([])

    # rollout
    max_turns = int(self.max_turns)
    setting = self.eval_mode  # "mt_all" or "mt_one"
    max_retry_rounds_per_turn = 3

    # turn index counts question turns, last turn forces answer
    # total turns allowed is max_turns, and on the last turn we force Answer
    for turn in range(max_turns + 1):
      alive = [i for i in range(n) if not done[i]]
      if not alive:
        break

      # determine if this is forced-answer turn
      force_answer = (turn >= max_turns)

      batch_prompts = []
      for i in alive:
        cur = list(messages[i])
        if force_answer:
          # force a final answer on this turn
          cur = list(cur)
          cur.append({"role": "user", "content": "You must now provide your final answer. Output: Answer: <number>."})
        batch_prompts.append(cur)

      # generate once for alive
      batch_responses, think_tok, cots, costs = cached_generate(
          batch_prompts,
          model_name,
          model_url,
          cache=cache,
          cache_file=cache_file,
          generation_config=self.generation_config,
          parallel_model_calls=self.parallel_model_calls,
      )

      cost_list = ut._as_cost_usd_list(costs, len(alive))
      think_list = ut._as_thinking_list(think_tok, len(alive))

      # normalize cots
      if cots is None:
        cot_list = [None] * len(alive)
      elif isinstance(cots, list):
        cot_list = list(cots) + [None] * max(0, len(alive) - len(cots))
        cot_list = cot_list[:len(alive)]
      else:
        cot_list = [cots] * len(alive)

      # per-alive: possible retries if invalid/format issues
      for j, i in enumerate(alive):
        resp = batch_responses[j]
        cost_acc[i] += float(cost_list[j] or 0.0)
        think_acc[i] += int(think_list[j] or 0)
        if cot_list[j] is not None:
          cot_last[i] = cot_list[j]

        convos[i].append({"role": self.model_role_name, "text": resp})

        # try parse action
        clean = ut._strip_thinking(resp, self.model_name)

        # If forced_answer, treat everything as answer attempt
        if force_answer:
          a = ut._try_float(clean)
          answered[i] = True
          done[i] = True
          turns_used[i] = turn
          turn_logs[i].append(
              {
                  "turn": turn,
                  "force_answer": True,
                  "model_response": resp,
                  "action": "ANSWER",
                  "parsed_answer": a,
              }
          )
          continue

        # not forced: if model answers now, treat as EARLY_ANSWER and finish
        if ut._is_answer_action(clean):
          a = ut._try_float(clean)
          answered[i] = True
          done[i] = True
          turns_used[i] = turn

          # mark early (because not forced-answer turn)
          turn_logs[i].append(
              {
                  "turn": turn,
                  "force_answer": False,
                  "model_response": resp,
                  "action": "EARLY_ANSWER",
                  "parsed_answer": a,
                  "ans_early": True,
              }
          )
          continue

        # otherwise treat as questions
        asked_vars_raw = ut._extract_questions_vars(clean)

        # setting specific: mt_one keeps only first var
        asked_for_oracle = asked_vars_raw
        if setting == "mt_one" and asked_vars_raw:
          asked_for_oracle = [asked_vars_raw[0]]

        # attempt oracle, retry if invalid up to max_retry_rounds_per_turn
        retry_round = 0
        while True:
          oracle_text, valid_vars, invalid_vars, dup_vars = self._oracle_answer_vars(
              asked_for_oracle,
              pred_vals[i],
              allowed_leaf_sets[i],
              forbidden_sets[i],
              setting=setting,
          )

          # account duplicates inside same action (still wastes a turn via asked_unique tracking)
          if dup_vars:
            duplicate_asked_counts[i] += len(dup_vars)

          if invalid_vars:
            invalid_asked_sets[i].update(invalid_vars)

          # if invalid, ask model to retry within same turn, up to limit
          if invalid_vars and retry_round < max_retry_rounds_per_turn:
            retry_round += 1

            # append assistant response then refusal, then retry instruction
            messages[i].append({"role": self.model_role_name, "content": ut._strip_thinking(resp, self.model_name)})
            messages[i].append({"role": "user", "content": oracle_text + " Please ask again."})

            convos[i].append({"role": "user", "text": oracle_text + " Please ask again."})
            turn_logs[i].append(
                {
                    "turn": turn,
                    "force_answer": False,
                    "model_response": resp,
                    "action": "QUESTION_INVALID",
                    "asked_vars_raw": asked_vars_raw,
                    "asked_used": asked_for_oracle,
                    "oracle_answer": oracle_text,
                    "valid_vars": [],
                    "invalid_vars": invalid_vars,
                    "duplicate_vars_in_action": dup_vars,
                    "retry_round": retry_round,
                }
            )

            # regenerate immediately for this single sample (no batch) to keep logic simple
            retry_resp, retry_think, retry_cot, retry_cost = cached_generate(
                [list(messages[i])],
                model_name,
                model_url,
                cache=cache,
                cache_file=cache_file,
                generation_config=self.generation_config,
                parallel_model_calls=self.parallel_model_calls,
            )
            resp = retry_resp[0]
            convos[i].append({"role": self.model_role_name, "text": resp})

            cost_acc[i] += float(ut._as_cost_usd_list(retry_cost, 1)[0] or 0.0)
            think_acc[i] += int(ut._as_thinking_list(retry_think, 1)[0] or 0)
            if isinstance(retry_cot, list) and retry_cot:
              cot_last[i] = retry_cot[0]
            elif retry_cot is not None:
              cot_last[i] = retry_cot

            clean = ut._strip_thinking(resp, self.model_name)
            asked_vars_raw = ut._extract_questions_vars(clean)
            asked_for_oracle = asked_vars_raw
            if setting == "mt_one" and asked_vars_raw:
              asked_for_oracle = [asked_vars_raw[0]]
            continue

          # valid question path (or invalid but retries exhausted)
          # consume a turn regardless, including repeated asks
          if valid_vars:
            for v in valid_vars:
              if v in asked_unique_sets[i]:
                duplicate_asked_counts[i] += 1
              asked_unique_sets[i].add(v)

          # append to conversation
          # if self.keep_thinking_trace and self.is_local_model(model_name):
          #   messages[i].append({"role": self.model_role_name, "content": resp})
          # else:
          messages[i].append({"role": self.model_role_name, "content": ut._strip_thinking(resp, self.model_name)})
          messages[i].append({"role": "user", "content": oracle_text})

          convos[i].append({"role": "user", "text": oracle_text})
          turn_logs[i].append(
              {
                  "turn": turn,
                  "force_answer": False,
                  "model_response": resp,
                  "action": "QUESTION",
                  "asked_vars_raw": asked_vars_raw,
                  "asked_used": asked_for_oracle,
                  "oracle_answer": oracle_text,
                  "valid_vars": valid_vars,
                  "invalid_vars": invalid_vars,
                  "duplicate_vars_in_action": dup_vars,
                  "retry_round": retry_round,
              }
          )

          # setting mt_all: after successful oracle answer, push a strong instruction to answer next
          if setting == "mt_all" and (not invalid_vars):
            messages[i].append({"role": "user", "content": "Now answer the original math question. Output: Answer: <number>."})
            convos[i].append({"role": "user", "text": "Now answer the original math question. Output: Answer: <number>."})

          turns_used[i] = turn
          break

    # finalize per sample: parse final answer from the last assistant message that looked like answer
    batch_preds: List[Optional[float]] = [None] * n
    batch_correct: List[bool] = [False] * n

    # question set quality: asked_unique vs heldout
    asked_sets_out: List[List[str]] = []
    gt_sets_out: List[List[str]] = []

    answer_preds_out: List[Optional[float]] = [None] * n
    metrics_out: List[Dict[str, Any]] = []

    for i in range(n):
      # -------- answer pred --------
      a_val: Optional[float] = None
      for m in reversed(convos[i]):
        if m["role"] != self.model_role_name:
          continue
        if ut._is_answer_action(m["text"]):
          a_val = ut._try_float(m["text"])
          if a_val is not None:
            break
      answer_preds_out[i] = a_val

      ans_gt = gt_answers[i]
      ans_correct = ut._float_equal(a_val, ans_gt)

      # answer turn info
      ans_turn, ans_forced = ut._find_first_answer_log(turn_logs[i])
      early_answer = (ans_turn is not None) and (ans_forced is False)

      # -------- var pred --------
      asked_set = set(asked_unique_sets[i])
      held_set = set(heldout_sets[i])

      asked_sorted = sorted(list(asked_set))
      held_sorted = sorted(list(held_set))
      asked_sets_out.append(asked_sorted)
      gt_sets_out.append(held_sorted)

      # var_exact = (asked_set == held_set)
      # prf1 = ut._prf1(asked_set, held_set)
      
      qmet = ut._qset_metrics(asked_set, held_set)
      var_exact = qmet["qset_exact"]  
      prf1 = ut._prf1(asked_set, held_set)  

      # aggregate metrics for df
      mrec = {
          "ans_gt": ans_gt,
          "ans_pred": a_val,
          "ans_correct": bool(ans_correct),
          "ans_turn": ans_turn,
          "ans_forced": ans_forced,
          "ans_early": bool(early_answer),

          "var_gt": held_sorted,
          "var_pred": asked_sorted,
          "var_exact": bool(var_exact),
          "var_precision": float(prf1["precision"]),
          "var_recall": float(prf1["recall"]),
          "var_f1": float(prf1["f1"]),
          "var_invalid_count": int(len(invalid_asked_sets[i])),
          "var_duplicate_count": int(duplicate_asked_counts[i]),
          
          # NEW qset metrics
          "qset_jaccard": qmet["qset_jaccard"],
          "qset_num_extra": qmet["qset_num_extra"],
          "qset_num_missing": qmet["qset_num_missing"],
          "qset_over_rate": qmet["qset_over_rate"],
          "qset_under_rate": qmet["qset_under_rate"],
          "qset_signed_extra_minus_missing": qmet["qset_signed_extra_minus_missing"],
          "qset_pred_size": qmet["qset_pred_size"],
          "qset_gt_size": qmet["qset_gt_size"],

          # for debug
          "qset_extra": qmet["qset_extra"],
          "qset_missing": qmet["qset_missing"],
      }
      metrics_out.append(mrec)

      # embed into LOG_JSON for offline analysis
      meta = {
          "setting": setting,
          "max_turns": max_turns,
          "turns_used": turns_used[i],
          "answered": answered[i],
          "goal_var": goal_vars[i],

          "var_pred": asked_sorted,
          "var_gt": held_sorted,
          "var_exact": bool(var_exact),
          "var_precision": float(prf1["precision"]),
          "var_recall": float(prf1["recall"]),
          "var_f1": float(prf1["f1"]),
          "invalid_asked": sorted(list(invalid_asked_sets[i])),
          "duplicate_asked_count": int(duplicate_asked_counts[i]),

          "ans_pred": a_val,
          "ans_gt": ans_gt,
          "ans_correct": bool(ans_correct),
          "ans_turn": ans_turn,
          "ans_forced": ans_forced,
          "ans_early": bool(early_answer),

          "turn_logs": turn_logs[i],
          
          "qset_jaccard": qmet["qset_jaccard"],
          "qset_num_extra": qmet["qset_num_extra"],
          "qset_num_missing": qmet["qset_num_missing"],
          "qset_over_rate": qmet["qset_over_rate"],
          "qset_under_rate": qmet["qset_under_rate"],
          "qset_signed_extra_minus_missing": qmet["qset_signed_extra_minus_missing"],
          "qset_extra": qmet["qset_extra"],
          "qset_missing": qmet["qset_missing"],
      }
      convos[i].append({"role": "user", "text": "LOG_JSON: " + json.dumps(meta)})

    # return convos, asked_sets_out, batch_correct, think_acc, cot_last, cost_acc, gt_sets_out
    return convos, asked_sets_out, batch_correct, think_acc, cot_last, cost_acc, gt_sets_out, answer_preds_out, metrics_out

  # -----------------------------
  # Main evaluation entry
  # -----------------------------
  def evaluate_batch(
      self,
      batch_requests: List[Any],
      model_name: str,
      model_url: str,
      cache=None,
      cache_file=None,
  ):
    """
    Returns:
      batch_convos, batch_preds, batch_correct, think_tokens_list, cots_list, cost_usd_list

    For mt modes:
      - batch_preds is the asked variable set (list[str]) for question set evaluation
      - batch_correct is final answer correctness (against Full Answer)
      - question set quality is returned in batch_gt_queries (as heldout set) and stored in conversations
    """
    if self.eval_mode in ["mt_all", "mt_one"]:
      # batch_requests are dict items
      # convos, asked_sets, ans_correct, think_acc, cot_last, cost_acc, heldout_gt = 
      convos, asked_sets, ans_correct, think_acc, cot_last, cost_acc, heldout_gt, ans_preds, metrics = \
      self._evaluate_batch_multiturn(
          batch_requests,
          model_name=model_name,
          model_url=model_url,
          cache=cache,
          cache_file=cache_file,
      )
      return convos, asked_sets, ans_correct, think_acc, cot_last, cost_acc, ans_preds, metrics


  # -----------------------------
  # Batching
  # -----------------------------
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
      if self.eval_mode in ["mt_all", "mt_one"]:
        # Build one structured item per sample
        rewritten = str(datum.get("Rewritten Problem", datum.get("Full Problem", "")))

        # ground truth answer
        gt_ans_f = ut._try_float(datum.get("Full Answer", None))
        if gt_ans_f is None:
          gt_ans_f = ut._try_float(datum.get("Rewritten Problem Answer", None))

        # heldout set for question-set quality
        heldout = ut._safe_list_field(datum.get("Heldout Value", "[]"))
        heldout = [str(x) for x in heldout]

        # goal var
        goal_var = datum.get("goal_var", None)
        if goal_var is None:
          raise ValueError('goal_var not found in data')
        goal_var = str(goal_var) if goal_var is not None else ""

        # leaf candidates and forbidden
        leaf = ut._infer_leaf_nodes(datum)
        all_vars = ut._all_vars_in_problem(datum)
        non_leaf = all_vars - leaf

        forbidden = set(non_leaf)
        if goal_var:
          forbidden.add(goal_var)

        # Also disallow asking anything not in leaf
        allowed_leaf = sorted(list(leaf))

        # pred values for oracle
        pred_map = ut._parse_pred_values_block(datum.get("Pred Values", None))

        # system prompt: list candidates and forbidden
        sys_p = self._build_mt_system_prompt(
            allowed_leaf=allowed_leaf,
            forbidden_vars=sorted(list(forbidden)),
            goal_var=goal_var if goal_var else "UNKNOWN",
            setting=self.eval_mode,
            max_turns=int(self.max_turns),
        )
        user_p = f"Math problem: {rewritten}"

        item = {
            "messages_init": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
            ],
            "gt_answer": gt_ans_f,
            "heldout": heldout,
            "goal_var": goal_var,
            "allowed_leaf": allowed_leaf,
            "forbidden": sorted(list(forbidden)),
            "pred_values": pred_map,
        }
        

        if len(batch_requests[-1]) >= batch_size:
          batch_ids.append([])
          batch_requests.append([])
          batch_gt_answers.append([])
          batch_gt_queries.append([])
          batch_system_prompts.append([])
          batch_k.append([])

        batch_ids[-1].append(d)
        batch_requests[-1].append(item)
        batch_gt_answers[-1].append(gt_ans_f)
        batch_gt_queries[-1].append(heldout)  # question set gt
        batch_system_prompts[-1].append(None)
        batch_k[-1].append(int(datum['k']))
        continue

    return batch_ids, batch_system_prompts, batch_requests, batch_gt_answers, batch_gt_queries, batch_k

  # -----------------------------
  # Data evaluation 
  # -----------------------------
  def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame):
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

    scored_n = 0
    correct_n = 0

    for batch_id, batch_sys, batch_req, batch_gt_ans, batch_gt_q, batch_k_vals in pbar:
      # batch_convo, batch_pred, batch_correct, think_list, cot_list, cost_list = 
      batch_convo, batch_pred_q, batch_ans_correct, think_list, cot_list, cost_list, batch_ans_pred, batch_metrics = \
      self.evaluate_batch(
          batch_req,
          model_name=self.model_name,
          model_url=self.model_url,
          cache=self.cache,
          cache_file=self.cache_file,
      )

      total_think_tokens.extend(int(x or 0) for x in think_list)
      all_cost_usd.extend(float(x or 0.0) for x in cost_list)
      all_cots.extend(cot_list if isinstance(cot_list, list) else [cot_list] * len(batch_id))

      for i, item_id in enumerate(batch_id):
        m = batch_metrics[i]
        rows.append(
          {
            "k": batch_k_vals[i],
            # answer track
            "ans_correct": m["ans_correct"],
            "ans_pred": m["ans_pred"],
            "ans_gt": m["ans_gt"],
            "ans_turn": m["ans_turn"],
            "ans_forced": m["ans_forced"],
            "ans_early": m["ans_early"],

            # var track
            "var_exact": m["var_exact"],
            "var_precision": m["var_precision"],
            "var_recall": m["var_recall"],
            "var_f1": m["var_f1"],
            "var_pred": m["var_pred"],
            "var_gt": m["var_gt"],
            "var_invalid_count": m["var_invalid_count"],
            "var_duplicate_count": m["var_duplicate_count"],

            # keep backward compatible columns
            "correct": m["ans_correct"],
            "qset_exact": m["var_exact"],
            "pred_q": m["var_pred"],
            "gt_qs": m["var_gt"],
            "pred_answer": m["ans_pred"],
            "gt_answer": m["ans_gt"],

            # misc
            "conversation": json.dumps(batch_convo[i]),
            "cost_usd": float(cost_list[i]) if cost_list[i] is not None else 0.0,
            "thinking_tokens": int(think_list[i]) if think_list[i] is not None else 0,
            "eval_mode": self.eval_mode,
            "max_turns": int(self.max_turns),
            
            # qset metrics
            "qset_jaccard": m["qset_jaccard"],
            "qset_num_extra": m["qset_num_extra"],
            "qset_num_missing": m["qset_num_missing"],
            "qset_over_rate": m["qset_over_rate"],
            "qset_under_rate": m["qset_under_rate"],
            "qset_signed_extra_minus_missing": m["qset_signed_extra_minus_missing"],
            "qset_pred_size": m["qset_pred_size"],
            "qset_gt_size": m["qset_gt_size"],

            # debug
            "qset_extra": m.get("qset_extra"),
            "qset_missing": m.get("qset_missing"),
          }
        )

        corr = m.get("ans_correct", None)
        if corr is not None and not (isinstance(corr, float) and pd.isna(corr)):
          scored_n += 1
          correct_n += int(bool(corr))
        if scored_n > 0:
          pbar.set_description(f"Accuracy: {correct_n / scored_n}")

    results = pd.DataFrame.from_records(rows)

    results_filtered = results[results["correct"].notna()] if ("correct" in results.columns) else results
    print("Answer accuracy:", results_filtered["ans_correct"].mean())
    print("Exact qset acc:", results_filtered["qset_exact"].mean())
    print("Qset Jaccard:", results_filtered["qset_jaccard"].mean())
    print("Avg extra:", results_filtered["qset_num_extra"].mean())
    print("Avg missing:", results_filtered["qset_num_missing"].mean())

    return results, all_cots, total_think_tokens, all_cost_usd
