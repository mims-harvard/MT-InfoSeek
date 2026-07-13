import argparse
import ast
import asyncio
import copy
import hashlib
import json
import os
import re
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Ensure local imports work regardless of invocation cwd.
THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
SIMPLELOGIC_PARENT = ROOT / "logic_q_mt" / "data_generation"
for _p in (THIS_DIR, ROOT, SIMPLELOGIC_PARENT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import model_registry
from model_utils import (
    cached_generate,
    load_cache_file,
    async_generate_single,
)
from evaluators.evaluator import Evaluator
from SimpleLogic.horn_sat_utils import (
    INFEASIBLE,
    parse_clauses,
    forced_value_via_refutation,
    solve_unit_prop,
    Clauses,
    CONTRADICTION
)


# ============ Data Classes ============

@dataclass
class World:
    """A world is a specific assignment of values to queried variables."""
    assignments: Dict[str, bool]  # var -> True/False
    target_value: str  # "goal" or "not goal"
    variables: List[str]


@dataclass
class Sample:
    """A single problem instance."""
    sample_id: int
    rules: List[List[str]]  # CNF clauses
    rules_nl: str  # Natural language rules
    goal: str
    known_facts: List[str]
    known_untrue_facts: List[str]
    cannot_ask: Set[str]
    all_valid_qs: List[str]
    worlds: List[World]
    inferred_variable_values: Dict[Tuple[str, ...], Dict[Tuple[bool, ...], Dict[str, bool]]]  # List of inferred variable values per gt_q set per assignment
    k: int  # Minimal sufficient size
    gt_qs: List[List[str]]  # Ground truth question sets
    all_alternative_gt_qs: List[List[str]] = field(default_factory=list)  # Alternative ground truth question sets    
    all_forbid_alternative_variables: List[str] = field(default_factory=list)  # Variables to forbid when forbidding alternatives


@dataclass
class TurnLog:
    """Log of a single turn."""
    turn: int
    questions: List[str]
    oracle_answer: str
    is_ambiguous: bool
    model_response: str
    thinking_tokens_count: int
    cot: str = ""  # Optional chain-of-thought trace


@dataclass
class ModelCallLog:
    """Log of a single model call, including the exact prompt snapshot."""
    call_index: int
    turn: int
    prompt_messages: List[Dict[str, Any]]
    action_type: str


@dataclass 
class EpisodeResult:
    """Result of a single episode (sample + world)."""
    sample_id: int
    world: World
    final_answer: Optional[str] = None
    correct: bool = False
    answered: bool = False
    turns_used: int = 0
    questions_asked: List[str] = field(default_factory=list)
    turn_logs: List[TurnLog] = field(default_factory=list)
    call_logs: List[ModelCallLog] = field(default_factory=list)
    final_cot: str = ""
    final_thinking_tokens_count: int = 0
    final_response: str = ""
    budget_violated: bool = False
    termination_reason: str = ""


@dataclass
class Action:
    """Parsed action from model response."""
    type: str  # "QUESTION", "ANSWER", "RETRY"
    questions: List[str] = field(default_factory=list)
    value: Optional[str] = None


RANDOM_POLICY_MODEL_NAMES = {"random", "random-baseline"}


# ============ Prompting ============

# Multi-turn, one question per turn
SYSTEM_PROMPT_MULTITURN_TEMPLATE = """Suppose you know the following rules about Alice:
{rules_nl}

You are trying to discern whether Alice is {goal} given some facts. You have a budget to ask about up to {budget} attributes about Alice in total. You must decide whether you have enough information to determine whether the final statement is true. At each turn, you may respond with one of the following:

Instructions:
1. If you already have enough information to determine the truth value of the statement, respond strictly with: "Answer: {goal}" or "Answer: not {goal}".
2. Otherwise, you MUST select exactly one attribute to query. Choose the single attribute that provides the most information regarding the statement, such that the answer can be determined with least questions being asked. 
3. Format the question strictly as: "Question: Is Alice [attribute]?"
4. Do not output any other text."""

# Multi-turn with no budget, one question per turn
SYSTEM_PROMPT_MULTITURN_NOBUDGET_TEMPLATE = """Suppose you know the following rules about Alice:
{rules_nl}

You are trying to discern whether Alice is {goal} given some facts. You must decide whether you have enough information to determine whether the final statement is true. At each turn, you may respond with one of the following:

Instructions:
1. If you already have enough information to determine the truth value of the statement, respond strictly with: "Answer: {goal}" or "Answer: not {goal}".
2. Otherwise, you MUST select exactly one attribute to query. Choose the single attribute that provides the most information regarding the statement, such that the answer can be determined with least questions being asked. 
3. Format the question strictly as: "Question: Is Alice [attribute]?"
4. Do not output any other text."""

# Single-turn, multi-question with no budget per turn
SYSTEM_PROMPT_SINGLETURN_TEMPLATE = """Suppose you know the following rules about Alice:
{rules_nl}

You are trying to discern whether Alice is {goal} given some facts. You can ask about attributes about Alice at once. You must decide whether you have enough information to determine whether the final statement is true. You may respond with one of the following:

Instructions:
1. If you already have enough information to determine the truth value of the statement, respond strictly with: "Answer: {goal}" or "Answer: not {goal}". 
2. Otherwise, you MUST select a set of attributes (at least 1) to query. Choose the smallest sufficient combination that provides enough information regarding the statement. 
3. Format the question strictly as: "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? ..." (for at least 1 attributes).
4. Do not output any other text."""

# Multi-turn, multi-question per turn
SYSTEM_PROMPT_MULTITURN_MULTIQUESTION_TEMPLATE = """Suppose you know the following rules about Alice:
{rules_nl}

You are trying to discern whether Alice is {goal} given some facts. {limit_text} At each turn, you may respond with one of the following:

Instructions:
1. If you already have enough information to determine the truth value of the statement, respond strictly with: "Answer: {goal}" or "Answer: not {goal}".
2. {question_instruction}
3. {question_format_instruction}
4. Do not output any other text."""

UNCERTAINTY_EMPHASIS = """

IMPORTANT: The initial facts provided are INSUFFICIENT. You MUST ask questions to gather the missing information before you can answer correctly. Do not guess - ask questions first."""


def _build_limit_text(max_turns: int, max_num_q_per_turn: int, include_budget_in_prompt: bool) -> str:
    if not include_budget_in_prompt:
        return "You must decide whether you have enough information to determine whether the final statement is true."

    turn_word = "turn" if max_turns == 1 else "turns"
    attr_word = "attribute" if max_num_q_per_turn == 1 else "attributes"
    return (
        f"You can take at most {max_turns} {turn_word}. "
        f"In each turn, you may ask about up to {max_num_q_per_turn} {attr_word}. "
        "You must decide whether you have enough information to determine whether the final statement is true."
    )


def build_initial_prompt(
    sample: Sample,
    max_turns: int,
    max_num_q_per_turn: int,
    is_legacy_single_turn_bulk: bool = False,
    emphasize_uncertainty: bool = True,
    forbid_alternatives: bool = False,
    include_budget_in_prompt: bool = True,
) -> List[Dict[str, str]]:
    """Build initial prompt for the model."""
    if is_legacy_single_turn_bulk:
        system_content = SYSTEM_PROMPT_SINGLETURN_TEMPLATE.format(
            rules_nl=sample.rules_nl,
            goal=sample.goal,
        )
    elif max_num_q_per_turn == 1:
        if include_budget_in_prompt:
            system_content = SYSTEM_PROMPT_MULTITURN_TEMPLATE.format(
                rules_nl=sample.rules_nl,
                goal=sample.goal,
                budget=max_turns,
            )
        else:
            system_content = SYSTEM_PROMPT_MULTITURN_NOBUDGET_TEMPLATE.format(
                rules_nl=sample.rules_nl,
                goal=sample.goal,
            )
    else:  # multi-turn multi-question
        question_instruction = (
            f"Otherwise, you MUST select a set of attributes (between 1 and {max_num_q_per_turn}) to query. "
            "Choose the smallest sufficient combination that provides enough information regarding the statement."
        )
        question_format_instruction = (
            'Format the question strictly as: "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? ..." '
            "(for at least 1 attribute)."
        )
        system_content = SYSTEM_PROMPT_MULTITURN_MULTIQUESTION_TEMPLATE.format(
            rules_nl=sample.rules_nl,
            goal=sample.goal,
            limit_text=_build_limit_text(max_turns, max_num_q_per_turn, include_budget_in_prompt),
            question_instruction=question_instruction,
            question_format_instruction=question_format_instruction,
        )
    
    # Add uncertainty emphasis if enabled
    if emphasize_uncertainty:
        system_content += UNCERTAINTY_EMPHASIS
    
    # Build known facts text
    facts = []
    for f in sample.known_facts:
        facts.append(f"Alice is {f}.")
    for f in sample.known_untrue_facts:
        facts.append(f"Alice is not {f}.")
    
    user_content = "\n".join(facts) if facts else "No facts are currently known about Alice."
    if sample.cannot_ask or (forbid_alternatives and sample.all_forbid_alternative_variables):
        user_content += "\n\nYou may NOT ask about the following attributes:\n"
        cannot_ask_and_forbid = []
        if sample.cannot_ask:
            cannot_ask_and_forbid.extend(sample.cannot_ask)
        if forbid_alternatives and sample.all_forbid_alternative_variables:
            cannot_ask_and_forbid.extend(sample.all_forbid_alternative_variables)
        user_content += ", ".join(sorted(set(cannot_ask_and_forbid))) + "."
    user_content += f"\n\nIs Alice {sample.goal}?"
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


FORCE_ANSWER_PROMPT = '\n\nYou have used all your questions. You must now provide your final answer in the format: "Answer: [goal]" or "Answer: not [goal]"'


def build_retry_prompt(max_num_q_per_turn: int, is_legacy_single_turn_bulk: bool = False) -> str:
    if is_legacy_single_turn_bulk:
        return """Could not parse your response. Please respond with exactly one of:
- "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? ..." to ask about at least one attribute
- "Answer: [goal]" or "Answer: not [goal]" to provide your final answer"""
    if max_num_q_per_turn == 1:
        return """Could not parse your response. Please respond with exactly one of:
- "Question: Is Alice [attribute]?" to ask about exactly one attribute
- "Answer: [goal]" or "Answer: not [goal]" to provide your final answer"""
    return (
        "Could not parse your response. Please respond with exactly one of:\n"
        '- "Question: Is Alice [attribute_1]? Is Alice [attribute_2]? ..." '
        f"to ask about between 1 and {max_num_q_per_turn} attributes\n"
        '- "Answer: [goal]" or "Answer: not [goal]" to provide your final answer'
    )


# ============ Episode Caching ============

def make_episode_cache_key(sample: Sample, world: World, is_flipped: bool) -> str:
    """Create unique cache key from sample + world."""
    key_dict = {
        "sample_id": sample.sample_id,
        "world_assignments": sorted(world.assignments.items()),
        "world_targets": world.target_value,
        "world_variables": world.variables,
        "goal": sample.goal,
        "is_flipped": is_flipped,
    }
    return json.dumps(key_dict, sort_keys=True)


def episode_result_to_dict(result: EpisodeResult) -> Dict[str, Any]:
    """Convert EpisodeResult to serializable dict."""
    return {
        "sample_id": result.sample_id,
        "world": {
            "assignments": result.world.assignments,
            "target_value": result.world.target_value,
            "variables": result.world.variables,
        },
        "final_answer": result.final_answer,
        "correct": result.correct,
        "answered": result.answered,
        "turns_used": result.turns_used,
        "questions_asked": result.questions_asked,
        "turn_logs": [
            {
                "turn": t.turn,
                "questions": t.questions,
                "oracle_answer": t.oracle_answer,
                "is_ambiguous": t.is_ambiguous,
                "model_response": t.model_response,
                "cot": t.cot,
                "thinking_tokens_count": t.thinking_tokens_count,
            }
            for t in result.turn_logs
        ],
        "call_logs": [
            {
                "call_index": c.call_index,
                "turn": c.turn,
                "prompt_messages": c.prompt_messages,
                "action_type": c.action_type,
            }
            for c in result.call_logs
        ],
        "final_response": result.final_response,
        "final_cot": result.final_cot,
        "final_thinking_tokens_count": result.final_thinking_tokens_count,
        "budget_violated": result.budget_violated,
        "termination_reason": result.termination_reason,
    }


def dict_to_episode_result(d: Dict[str, Any]) -> EpisodeResult:
    """Convert dict back to EpisodeResult."""
    world = World(
        assignments=d["world"]["assignments"],
        target_value=d["world"]["target_value"],
        variables=d["world"]["variables"] if "variables" in d["world"] else sorted(list(d["world"]["assignments"].keys())),  # TODO: for compatibility with old cache
    )
    turn_logs = [
        TurnLog(
            turn=t["turn"],
            questions=t["questions"],
            oracle_answer=t["oracle_answer"],
            is_ambiguous=t["is_ambiguous"],
            model_response=t["model_response"],
            cot=t["cot"],
            thinking_tokens_count=t["thinking_tokens_count"]
        )
        for t in d.get("turn_logs", [])
    ]
    call_logs = [
        ModelCallLog(
            call_index=c.get("call_index", 0),
            turn=c.get("turn", 0),
            prompt_messages=c.get("prompt_messages", []),
            action_type=c.get("action_type", "RETRY"),
        )
        for c in d.get("call_logs", [])
    ]
    return EpisodeResult(
        sample_id=d["sample_id"],
        world=world,
        final_answer=d.get("final_answer"),
        correct=d.get("correct", False),
        answered=d.get("answered", False),
        turns_used=d.get("turns_used", 0),
        questions_asked=d.get("questions_asked", []),
        turn_logs=turn_logs,
        call_logs=call_logs,
        final_response=d.get("final_response", ""),
        final_cot=d.get("final_cot", ""),
        final_thinking_tokens_count=d.get("final_thinking_tokens_count", 0),
        budget_violated=d.get("budget_violated", False),
        termination_reason=d.get("termination_reason", ""),
    )


def load_episode_cache(cache_file: str) -> Dict[str, Dict[str, Any]]:
    """Load episode cache from JSONL file using orjson."""
    import orjson
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(cache_file):
        return cache

    with open(cache_file, "rb") as f:
        for line in f:
            try:
                entry = orjson.loads(line)
                key = entry.get("key")
                result = entry.get("result")
                if key and result:
                    cache[key] = result
            except orjson.JSONDecodeError:
                continue
    return cache


def save_episode_to_cache(cache: Dict[str, Dict], cache_file: str, key: str, result: EpisodeResult):
    """Save single episode result to cache and file."""
    result_dict = episode_result_to_dict(result)
    cache[key] = result_dict
    
    # Append to cache file
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "a") as f:
        f.write(json.dumps({"key": key, "result": result_dict}) + "\n")


# ============ Oracle (Unit Propagation) ============

def special_oracle_logic(clauses: Clauses, base_context: Dict[str, bool], var: str, goal: str, goal_value: bool, oracle_type: str = "adversarial") -> None:
    """ Adversarial (least-disclosure) or cooperative oracle logic.
    - Adversarial logic: pick an answer for X consistent with the assumed world Y=goal_value, and if possible keep Y not forced (i.e., forced_value_via_refutation (with Y as goal) == None) after adding the answer.
    - Cooperative logic: pick an answer for X consistent with the assumed world Y=goal_value, and if possible force Y to the correct value after adding the answer.
    """
    assert oracle_type in ("adversarial", "cooperative"), "oracle_type must be 'adversarial' or 'cooperative'"
    answer_true = f"Yes, Alice is {var}."
    answer_false = f"No, Alice is not {var}."
    
    def eval_answer(var_value: bool):
        """
        Return a dict describing how good answering var=v is, or None if impossible.
        """
        ctx = {**base_context, var: var_value}
        
        # what does this answer force about the goal (from the player's perspective)?
        forced_goal, base_res = forced_value_via_refutation(clauses, ctx, goal, return_base_res=True)  # True/False/None/INFEASIBLE
        if forced_goal == INFEASIBLE:
            return None

        # must be consistent with the assumed world Y=goal_value:
        # if forced_goal is the opposite, this answer cannot happen in the assumed world
        if forced_goal is not None and forced_goal != goal_value:
            return None

        # adversarial preference:
        # 1) keep goal ambiguous if possible (forced_goal is None)
        # 2) reveal as little else as possible (small UP closure)
        closure_size = len(base_res)

        return {
            "v": var_value,
            "keeps_goal_ambiguous": (forced_goal is None),
            "closure_size": closure_size,
        }
        
    cand_var_T = eval_answer(var_value=True)
    cand_var_F = eval_answer(var_value=False)
    cands = [c for c in (cand_var_T, cand_var_F) if c is not None]

    if not cands:
        raise ValueError(
            f"No feasible answer for query {var} consistent with assumed goal_value={goal_value}."
        )

    if oracle_type == "adversarial":
        # Choose the "least disclosure" answer:
        #   - Prefer keeping goal ambiguous
        #   - Then prefer smaller closure (less information leaked)
        #   - Then random tie-break
        best = max(
            cands,
            key=lambda c: (
                c["keeps_goal_ambiguous"],
                -c["closure_size"],
                random.random(),
            ),
        )
    else:
        # Choose the "most cooperative" answer:
        #   - Prefer forcing goal to correct value
        #   - Then prefer smaller closure (less information leaked)
        #   - Then random tie-break
        best = max(
            cands,
            key=lambda c: (
                not c["keeps_goal_ambiguous"],
                c["closure_size"],
                random.random(),
            ),
        )

    chosen_val = best["v"]
    is_ambiguous = best["keeps_goal_ambiguous"]
    
    if chosen_val:
        return answer_true, True, is_ambiguous
    else:
        return answer_false, False, is_ambiguous
    

def oracle_answer(clauses: List[Set[Tuple[str, bool]]],
                  base_context: Dict[str, bool],
                  world: World,
                  questions: List[str],
                  cannot_ask: Set[str] = None,
                  all_forbid_alternative_variables: List[str] = None,
                  oracle_type: str = "adversarial", 
                  forbid_alternatives: bool = False, 
                  base_context_wo_assignments: Dict[str, bool] = None, 
                  last_round_is_ambiguous: bool = False) -> Tuple[str, bool]:
    """
    Answer questions based on world + unit propagation inference.
    Returns natural language answer string and a boolean indicating if the target is ambiguous after answering.
    
    If a question is in cannot_ask (already given or is the target), refuse to answer.
    """
    if cannot_ask is None:
        cannot_ask = set()
    if all_forbid_alternative_variables is None:
        all_forbid_alternative_variables = set()
        
    goal = world.target_value.replace("not ", "") if world.target_value.startswith("not ") else world.target_value
    goal_value = not world.target_value.startswith("not ")
    
    # Build answers
    answers = []
    all_is_ambiguous = last_round_is_ambiguous
    inferred = dict()
    ctx = base_context_wo_assignments.copy() if base_context_wo_assignments is not None else {}
    
    for q in questions:
        # Normalize question (remove "not " prefix if present)
        var = q.replace("not ", "") if q.startswith("not ") else q
        
        # Check if this is a forbidden question
        if var in cannot_ask:
            answers.append(f"Sorry, I cannot answer whether Alice is {var} because this variable is either already given or is the target variable.")
            continue
        
        if forbid_alternatives and var in all_forbid_alternative_variables:
            answers.append(f"Sorry, I cannot answer whether Alice is {var} because this variable is forbidden to ask about.")
            continue
        
        if oracle_type == "random":
            # Run Horn-SAT w/ UP
            forced = forced_value_via_refutation(clauses, base_context, var)  # True/False/None/INFEASIBLE
            
            if forced is None:
                answers.append(f"Not sure whether Alice is {var}.")
            elif forced == INFEASIBLE:
                raise ValueError("Infeasible base context detected in non-adversarial oracle.")
            elif forced:
                answers.append(f"Yes, Alice is {var}.")
                inferred[var] = True
            else:
                answers.append(f"No, Alice is not {var}.")
                inferred[var] = False
            
            if var in inferred:
                ctx[var] = inferred[var]
            forced_goal = forced_value_via_refutation(clauses, ctx, goal)
            if forced_goal == INFEASIBLE:
                raise ValueError("Infeasible base context detected in non-adversarial oracle.")
            # if forced_goal is the opposite, this answer cannot happen in the assumed world
            if forced_goal is not None and forced_goal != goal_value:
                # print("Clauses:\n")
                # print(clauses)
                # print()
                # print("Base context:\n")
                # print(base_context)
                # print()
                # print("World:\n")
                # print(world)
                # print()
                # print("goal:\n")
                # print(goal)
                # print()
                # print("goal value:\n")
                # print(goal_value)
                # print()
                # print("base_context_wo_assignments:\n")
                # print(base_context_wo_assignments)
                # print()
                # print("ctx:\n")
                # print(ctx)
                # print()
                # print("questions:\n")
                # print(questions)
                # print()
                # print("q:\n")
                # print(q)
                # print()
                raise ValueError("Infeasible  base context detected in non-adversarial oracle.")

            is_ambiguous = forced_goal is None
        
        elif oracle_type in ("adversarial", "cooperative"):
            assert base_context == base_context_wo_assignments, "base_context should not contain world assignments for adversarial/cooperative oracle."  # DEBUG
            answer, var_value, is_ambiguous = special_oracle_logic(clauses, base_context, var, goal, goal_value, oracle_type=oracle_type)
            answers.append(answer)
            inferred[var] = var_value
            
        else:
            raise ValueError("oracle_type must be 'adversarial', 'cooperative', or 'random'")

        all_is_ambiguous = all_is_ambiguous and is_ambiguous
        
        # UPDATE base context per question (to avoid inconsistencies, if any)  # TODO: Check validity of this
        base_context.update(inferred)
        base_context_wo_assignments.update(inferred)
    
    return " ".join(answers), all_is_ambiguous


# ============ Data Loading ============

def parse_rules_to_nl(rules: List[List[str]]) -> str:
    """Parse rules into natural language format.
    
    Handles both original rules (negated premises -> positive conclusion) 
    and flipped rules (positive premises -> negated conclusion).
    """
    rules_nl = []
    for rule in rules:
        negated_words = [
            word.split("not ")[-1] for word in rule if word.startswith("not ")
        ]
        positive_words = [word for word in rule if not word.startswith("not ")]
        
        # Original format: multiple negated premises, one positive conclusion
        # e.g., ['c', 'not a', 'not b'] means: if a and b then c
        if len(positive_words) == 1 and negated_words:
            premises = " and ".join(negated_words)
            conclusion_word = positive_words[0]
            rules_nl.append(f"If Alice is {premises}, then Alice is {conclusion_word}.")
        
        # Flipped format: multiple positive premises, one negated conclusion  
        # e.g., ['not c', 'a', 'b'] means: if not a and not b then not c
        elif len(negated_words) == 1 and positive_words:
            premises = " and ".join([f"not {w}" for w in positive_words])
            conclusion_word = f"not {negated_words[0]}"
            rules_nl.append(f"If Alice is {premises}, then Alice is {conclusion_word}.")
    
    return "\n".join(sorted(rules_nl))


def parse_world_key(key_json: str) -> Dict[str, bool]:
    """Parse world key like '["zealous", "not stormy"]' to {var: bool}."""
    facts = json.loads(key_json)
    assignments = {}
    for f in facts:
        if f.startswith("not "):
            assignments[f[4:]] = False
        else:
            assignments[f] = True
    return assignments


def load_data(csv_path: str, oracle_type: str) -> List[Sample]:
    """Load data from CSV and parse into Sample objects."""
    df = pd.read_csv(csv_path)
    
    # check sample_id uniqueness
    if "sample_id" in df.columns:
        if df["sample_id"].isna().any():
            raise ValueError("sample_id column contains NaNs.")
        if df["sample_id"].duplicated().any():
            dup = df[df["sample_id"].duplicated()]["sample_id"].iloc[0]
            raise ValueError(f"Duplicate sample_id detected, e.g. {dup}. Must be unique.")
    
    samples = []
    for _, row in df.iterrows():
        # Parse string representations to Python objects
        sample_id = int(row["sample_id"])
        rules = ast.literal_eval(row["rules"])
        known_facts = ast.literal_eval(row["known_facts"])
        known_untrue_facts = ast.literal_eval(row["known_untrue_facts"])
        cannot_ask = set(ast.literal_eval(row["cannot_ask_facts"]))
        gt_qs = ast.literal_eval(row["gt_qs"])
        
        all_valid_qs = ast.literal_eval(row["all_valid_qs"])
        all_alternative_gt_qs = ast.literal_eval(row["all_alternative_gt_qs"])
        all_forbid_alternative_variables = sorted(set(sum(all_alternative_gt_qs, [])) - set(sum(gt_qs, [])))
        
        inferred_variable_values = ast.literal_eval(row["inferred_variable_values"])
        
        # Parse worlds from gt_q_to_derivations_min_rules
        derivations = ast.literal_eval(row["gt_q_to_derivations_min_rules"])
        worlds = []
        # derivations is a list of dicts, one per gt_q set
        # Each dict maps assignment_key -> {"target_value": ..., "derivation": ...}
        
        assert len(derivations) == len(gt_qs), "Mismatch between derivations and gt_qs lengths."
        for deriv_dict in derivations:
            if oracle_type != "random":
                key_json = list(deriv_dict.keys())[0]
                assignments = parse_world_key(key_json)
                variables = sorted(list(assignments.keys()))
                worlds.append(World(assignments={}, target_value=row["goal"], variables=variables))
                worlds.append(World(assignments={}, target_value=f"not {row['goal']}", variables=variables))
            else:
                for key_json, value in deriv_dict.items():
                    assignments = parse_world_key(key_json)
                    target_value = value["target_value"]
                    variables = sorted(list(assignments.keys()))
                    worlds.append(World(assignments=assignments, target_value=target_value, variables=variables))
                
        sample = Sample(
            sample_id=sample_id,
            rules=rules,
            rules_nl=parse_rules_to_nl(rules),
            goal=row["goal"],
            known_facts=known_facts,
            known_untrue_facts=known_untrue_facts,
            cannot_ask=cannot_ask,
            all_valid_qs=all_valid_qs,
            worlds=worlds,
            inferred_variable_values=inferred_variable_values,
            k=int(row["k"]),
            gt_qs=gt_qs, 
            all_alternative_gt_qs=all_alternative_gt_qs,
            all_forbid_alternative_variables=all_forbid_alternative_variables,
        )
        samples.append(sample)
    
    return samples


# ============ Flipping (Negation) ============

def flip_literal(lit: str) -> str:
    """Flip a literal: 'x' -> 'not x', 'not x' -> 'x'."""
    if lit.startswith("not "):
        return lit[4:]
    else:
        return f"not {lit}"


def flip_rules(rules: List[List[str]]) -> List[List[str]]:
    """Flip all literals in all rules."""
    return [[flip_literal(lit) for lit in rule] for rule in rules]


def flip_target(target: str, goal: str) -> str:
    """Flip target value: 'goal' -> 'not goal', 'not goal' -> 'goal'."""
    if target.startswith("not "):
        return target[4:]
    else:
        return f"not {target}"


def create_flipped_sample(sample: Sample) -> Sample:
    """
    Create a flipped version of a sample by:
    - Swapping known_facts and known_untrue_facts
    - Flipping all rules (each literal negated)
    - Flipping world assignments and target values
    """
    flipped_rules = flip_rules(sample.rules)
    flipped_rules_nl = parse_rules_to_nl(flipped_rules)
    
    # Flip worlds
    flipped_worlds = []
    for world in sample.worlds:
        flipped_assignments = {var: not val for var, val in world.assignments.items()}
        flipped_target = flip_target(world.target_value, sample.goal)
        flipped_worlds.append(World(
            assignments=flipped_assignments,
            target_value=flipped_target,
            variables=world.variables,
        ))
    
    flipped_inferred_variable_values = {}
    for gt_q_key, assignment_dict in sample.inferred_variable_values.items():
        flipped_assignment_dict = {}
        for assignment_key, var_values in assignment_dict.items():
            flipped_var_values = {var: not val if val is not None else None for var, val in var_values.items()}
            flipped_assignment_dict[assignment_key] = flipped_var_values
        flipped_inferred_variable_values[gt_q_key] = flipped_assignment_dict
    
    return Sample(
        sample_id=sample.sample_id,
        rules=flipped_rules,
        rules_nl=flipped_rules_nl,
        goal=sample.goal,  # Goal stays the same
        known_facts=sample.known_untrue_facts,  # Swap
        known_untrue_facts=sample.known_facts,  # Swap
        cannot_ask=sample.cannot_ask,
        all_valid_qs=sample.all_valid_qs,
        worlds=flipped_worlds,
        inferred_variable_values=flipped_inferred_variable_values,
        k=sample.k,
        gt_qs=sample.gt_qs,
        all_alternative_gt_qs=sample.all_alternative_gt_qs,
        all_forbid_alternative_variables=sample.all_forbid_alternative_variables,
    )


# ============ Action Parsing ============

def extract_non_thinking(model_name: str, response: str) -> str:
    """Strip thinking trace from response for all model types.
    
    Handles:
    - Qwen: </think> delimiter
    - gpt-oss: <|start|>assistant<|channel|>final<|message|>...<|return|> pattern
    - Mistral: Model_utils handles via streaming reasoning_content (response should already be clean)
    - Others: Return as-is if no pattern found
    """
    # Qwen: split by </think>
    if "qwen" in model_name.lower():
        if "</think>" in response:
            return response.split("</think>")[-1].strip()
    
    # gpt-oss: extract content after final assistant marker
    if "gpt-oss" in model_name.lower():
        gpt_oss_marker = "<|start|>assistant<|channel|>final<|message|>"
        gpt_oss_end = "<|return|>"
        if gpt_oss_marker in response:
            final_output = response.split(gpt_oss_marker, 1)[-1]
            if gpt_oss_end in final_output:
                final_output = final_output.split(gpt_oss_end, 1)[0]
            return final_output.strip()
        
    if "mistral" in model_name.lower():
        response = response.split("[/THINK]", 1)[-1]
    
    return response.strip()


_ATTR_TRAIL_STRIP_RE = re.compile(r"""[\s\.\!\?,"'`]+$""")
_WS_RE = re.compile(r"\s+")

def _normalize_attr(raw: str) -> str:
    s = raw.strip()
    s = _ATTR_TRAIL_STRIP_RE.sub("", s)  # trim trailing punctuation
    s = _WS_RE.sub(" ", s).strip()
    s_lower = s.lower()

    # If the model included extra boilerplate inside the attr field
    for prefix in ("is alice ", "alice is "):
        if s_lower.startswith(prefix):
            s = s[len(prefix):].strip()
            s = _WS_RE.sub(" ", s).strip()
            s_lower = s.lower()

    # normalize "not   X" -> "not x"
    if s_lower.startswith("not "):
        rest = _WS_RE.sub(" ", s[4:]).strip().lower()
        return f"not {rest}" if rest else "not"

    return s_lower

def _extract_question_attrs(clean_response: str) -> List[str]:
    # Prefer explicit "Question:" region if present (but still handle missing '?')
    q_region = clean_response
    m = re.search(r"(?is)\bquestion\s*:\s*(.+)$", clean_response)
    if m:
        q_region = m.group(1)

    attrs = []
    # Primary: any "Is Alice ... ?" occurrences (multi-token attr)
    # can technically match a blank space if the LLM outputs Question: Is Alice ? (two spaces) -- but this is a failure case anyway
    for mm in re.finditer(r"(?is)\bis\s+alice\s+([^?\n\r]+?)\s*\?", q_region):
        val = mm.group(1).strip()
        if val:
            attrs.append(val)

    if attrs:
        return attrs

    # Fallback: "Question: <attr>" (no '?', no "Is Alice" prefix)
    if m:
        first_line = q_region.strip().splitlines()[0].strip()
        first_line = first_line[:-1].strip() if first_line.endswith("?") else first_line
        first_line = re.sub(r"(?is)^is\s+alice\s+", "", first_line).strip()
        if first_line:
            return [first_line]

    # Last resort: allow "Is Alice <attr>?" even without "Question:" prefix
    for mm in re.finditer(r"(?is)\bis\s+alice\s+([^?\n\r]+?)\s*\?", clean_response):
        val = mm.group(1).strip()
        if val:
            attrs.append(val)
    return attrs

def parse_action(
    model_name: str,
    response: Optional[str],
    already_extracted: bool = False,
    max_num_q_per_turn: Optional[int] = 1,
) -> Action:
    if response is None:
        response_text = ""
    elif isinstance(response, str):
        response_text = response
    else:
        response_text = str(response)

    if already_extracted:
        clean_response = response_text.strip()
    else:
        clean_response = extract_non_thinking(model_name, response_text).strip()

    # Answer: take the first "Answer:" line (don’t stop at '.')
    ans = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", clean_response)
    if ans:
        return Action(type="ANSWER", value=ans.group(1).strip())

    # Question(s)
    raw_attrs = _extract_question_attrs(clean_response)
    attrs = [_normalize_attr(a) for a in raw_attrs]
    attrs = [a for a in attrs if a and a != "not"]  # drop degenerate parses

    if len(attrs) >= 1 and (max_num_q_per_turn is None or len(attrs) <= max_num_q_per_turn):
        return Action(type="QUESTION", questions=attrs)

    return Action(type="RETRY")


def _norm_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'^[\s"\'`]+|[\s"\'`]+$', "", s)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)  # strip trailing "(...)" if present
    s = re.sub(r"[.?!]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _truth_about_goal(text: str, goal: str) -> Optional[bool]:
    if text is None:
        return None
    t = _norm_phrase(text)

    # remove accidental "answer:" prefix if it leaked in
    t = re.sub(r"^answer\s*:\s*", "", t).strip()

    # common wrappers
    t = re.sub(r"^alice\s+is\s+not\s+", "not ", t).strip()
    t = re.sub(r"^alice\s+is\s+", "", t).strip()

    goal_n = _norm_phrase(goal)
    if t == goal_n:
        return True
    if t == f"not {goal_n}":
        return False

    return None  # don’t guess


def is_random_policy_model(model_name: str) -> bool:
    return model_name.strip().lower() in RANDOM_POLICY_MODEL_NAMES


def format_goal_answer(goal: str, value: bool) -> str:
    return goal if value else f"not {goal}"


def format_question_response(questions: List[str]) -> str:
    return "Question: " + " ".join(f"Is Alice {question}?" for question in questions)


def build_random_policy_rng(sample: Sample, world: World, is_flipped: bool, seed: int) -> random.Random:
    seed_material = json.dumps(
        {
            "seed": seed,
            "sample_id": sample.sample_id,
            "world_assignments": sorted(world.assignments.items()),
            "world_target": world.target_value,
            "world_variables": world.variables,
            "goal": sample.goal,
            "is_flipped": is_flipped,
        },
        sort_keys=True,
    )
    seed_int = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed_int)


def get_available_random_policy_questions(
    sample: Sample,
    questions_asked: List[str],
    forbid_alternatives: bool,
) -> List[str]:
    banned = set(sample.cannot_ask)
    if forbid_alternatives:
        banned.update(sample.all_forbid_alternative_variables)
    asked = set(questions_asked)
    return [question for question in sample.all_valid_qs if question not in asked and question not in banned]


def choose_random_policy_action(
    rng: random.Random,
    sample: Sample,
    base_context_wo_assignments: Dict[str, bool],
    clauses: Clauses,
    questions_asked: List[str],
    max_num_q_per_turn: int,
    force_answer_prompt_applied: bool,
    forbid_alternatives: bool,
) -> Action:
    forced_goal = forced_value_via_refutation(clauses, base_context_wo_assignments, sample.goal)
    if forced_goal == INFEASIBLE:
        raise ValueError("Infeasible base context detected in random policy baseline.")
    if forced_goal is not None:
        return Action(type="ANSWER", value=format_goal_answer(sample.goal, forced_goal))

    available_questions = get_available_random_policy_questions(
        sample,
        questions_asked=questions_asked,
        forbid_alternatives=forbid_alternatives,
    )
    if not available_questions or force_answer_prompt_applied:
        guessed_goal_value = rng.choice([True, False])
        return Action(type="ANSWER", value=format_goal_answer(sample.goal, guessed_goal_value))

    num_questions = min(max_num_q_per_turn, len(available_questions))
    selected_questions = rng.sample(available_questions, num_questions)
    return Action(type="QUESTION", questions=selected_questions)

def check_answer_correct(answer: Optional[str], target_value: str, goal: str) -> bool:
    a = _truth_about_goal(answer or "", goal)
    tgt = _truth_about_goal(target_value or "", goal)
    return (a is not None) and (tgt is not None) and (a == tgt)


def resolve_episode_limits(sample: Sample, budget: str, max_num_q_per_turn: int) -> Tuple[int, int, bool]:
    """Resolve max turns and per-turn question limit for one episode."""
    if budget == "k":
        max_turns = sample.k
    else:
        max_turns = int(budget)

    effective_max_num_q_per_turn = max_num_q_per_turn
    is_legacy_single_turn_bulk = False

    # Legacy compatibility: historically budget=0 means one turn with many questions.
    if max_turns == 0:
        is_legacy_single_turn_bulk = True
        max_turns = 1
        effective_max_num_q_per_turn = max(1, len(sample.all_valid_qs))

    return max_turns, effective_max_num_q_per_turn, is_legacy_single_turn_bulk


# ============ Episode Rollout ============

async def run_episode_async(
    model_name: str,
    port: str,
    sample: Sample,
    world: World,
    budget: str,
    max_num_q_per_turn: int,
    cache: Optional[Dict],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool = False,
    max_retries: int = 5,
    verbose: bool = False,
    emphasize_uncertainty: bool = True,
    is_flipped: bool = False,
    oracle_type: str = "adversarial",
    forbid_alternatives: bool = False,
    include_budget_in_prompt: bool = True,
    enforce_question_limit_in_legacy_bulk: bool = False,
    history_append_mode: str = "full",
    seed: int = 42,
) -> EpisodeResult:
    """
    Async version of run_episode for parallel processing with caching.
    For oracle_type:
        - "adversarial"/"cooperative": answer according to a world (final target value) and use adversarial/cooperative oracle logic
        - "random": answer according to a specific assignment of sufficient set variable values consistent with a world (return inferrable variable values from world)
    """
    max_turns, effective_max_num_q_per_turn, is_legacy_single_turn_bulk = resolve_episode_limits(sample, budget, max_num_q_per_turn)
    
    # Check cache first
    cache_key = make_episode_cache_key(sample, world, is_flipped)
    if cache is not None and cache_key in cache:
        if verbose:
            print(f"\n--- Episode (CACHED): sample={sample.sample_id}, world={world.assignments} ---")
        return dict_to_episode_result(cache[cache_key])
    
    if verbose:
        print(f"\n--- Episode: sample={sample.sample_id}, world={world.assignments}, target={world.target_value} ---")
    
    # Build initial prompt
    messages = build_initial_prompt(sample, 
                                    max_turns,
                                    effective_max_num_q_per_turn,
                                    is_legacy_single_turn_bulk=is_legacy_single_turn_bulk,
                                    emphasize_uncertainty=emphasize_uncertainty, 
                                    forbid_alternatives=forbid_alternatives, 
                                    include_budget_in_prompt=include_budget_in_prompt)
    retry_prompt = build_retry_prompt(
        effective_max_num_q_per_turn,
        is_legacy_single_turn_bulk=is_legacy_single_turn_bulk,
    )
    clauses = parse_clauses(sample.rules)
    
    # Build base context from known facts
    base_context = {}
    base_context_wo_assignments = {}
    for f in sample.known_facts:
        base_context_wo_assignments[f] = True
    for f in sample.known_untrue_facts:
        base_context_wo_assignments[f] = False
    if oracle_type == "random":
        base_context = base_context_wo_assignments.copy()
        for var, val in world.assignments.items():
            base_context[var] = val
    else:
        base_context = base_context_wo_assignments.copy()
    
    result = EpisodeResult(
        sample_id=sample.sample_id,
        world=world,
    )
    random_policy_rng = build_random_policy_rng(sample, world, is_flipped, seed) if is_random_policy_model(model_name) else None
    
    retry_count = 0
    turn = 0
    call_index = 0
    is_ambiguous = True
    
    while turn <= max_turns:
        if verbose:
            print(f"  Turn {turn}/{max_turns}...")
        
        # Force answer on last turn - use deep copy for thread safety
        current_messages = copy.deepcopy(messages)
        force_answer_prompt_applied = turn == max_turns
        retry_count_before_call = retry_count
        if force_answer_prompt_applied:
            current_messages[-1]["content"] += FORCE_ANSWER_PROMPT
        
        # Generate response using either the actual model or the built-in random baseline.
        if verbose:
            print(f"    Generating response...")

        if is_random_policy_model(model_name):
            assert random_policy_rng is not None
            action = choose_random_policy_action(
                rng=random_policy_rng,
                sample=sample,
                base_context_wo_assignments=base_context_wo_assignments,
                clauses=clauses,
                questions_asked=result.questions_asked,
                max_num_q_per_turn=effective_max_num_q_per_turn,
                force_answer_prompt_applied=force_answer_prompt_applied,
                forbid_alternatives=forbid_alternatives,
            )
            response = (
                f"Answer: {action.value}"
                if action.type == "ANSWER"
                else format_question_response(action.questions)
            )
            cot = ""
            thinking_tokens_count = 0
        else:
            request_debug_context = {
                "sample_id": sample.sample_id,
                "world_assignments": world.assignments,
                "world_target_value": world.target_value,
                "world_variables": world.variables,
                "budget": budget,
                "turn": turn,
                "max_turns": max_turns,
                "call_index": call_index,
                "retry_count_before_call": retry_count_before_call,
                "force_answer_prompt_applied": force_answer_prompt_applied,
                "is_legacy_single_turn_bulk": is_legacy_single_turn_bulk,
                "oracle_type": oracle_type,
                "forbid_alternatives": forbid_alternatives,
                "questions_asked_so_far": list(result.questions_asked),
                "enforce_question_limit_in_legacy_bulk": enforce_question_limit_in_legacy_bulk,
                "history_append_mode": history_append_mode,
            }
            result_obj = await async_generate_single(
                model_name=model_name,
                port=port,
                messages=current_messages,
                generation_config=generation_config,
                debug_context=request_debug_context,
            )
            response = result_obj.text or ""  # Final output (CoT already stripped by model_utils)
            cot = result_obj.cot  # Thinking trace (for logging if keep_thinking_trace)
            thinking_tokens_count = result_obj.num_thinking_tokens

        if verbose:
            print(f"    Got response ({len(response)} chars, {thinking_tokens_count} thinking tokens)")
        
        if not is_random_policy_model(model_name):
            # Parse action - response is already extracted (CoT stripped by model_utils)
            parse_max_num_q_per_turn: Optional[int] = (
                effective_max_num_q_per_turn
                if (enforce_question_limit_in_legacy_bulk or not is_legacy_single_turn_bulk)
                else None
            )
            action = parse_action(
                model_name,
                response,
                already_extracted=True,
                max_num_q_per_turn=parse_max_num_q_per_turn,
            )
        current_call_index = call_index
        result.call_logs.append(ModelCallLog(
            call_index=current_call_index,
            turn=turn,
            prompt_messages=copy.deepcopy(current_messages),
            action_type=action.type,
        ))
        call_index += 1
        if verbose:
            print(f"    Action: {action.type} - questions={action.questions}, value={action.value}")
        
        if action.type == "ANSWER":
            result.final_answer = action.value
            result.answered = True
            result.correct = check_answer_correct(action.value, world.target_value, sample.goal)
            result.final_response = response
            result.turns_used = turn
            result.final_cot = cot
            result.final_thinking_tokens_count = thinking_tokens_count
            result.termination_reason = "answered"
            if verbose:
                print(f"    => ANSWER: {action.value}, correct={result.correct}")
            break
            
        elif action.type == "QUESTION":
            # Retry if accidentally ask a question when it should answer
            if turn == max_turns:
                retry_count += 1
                if retry_count >= max_retries:
                    result.final_response = response
                    result.final_cot = cot
                    result.final_thinking_tokens_count = thinking_tokens_count
                    result.turns_used = turn
                    result.budget_violated = True
                    result.termination_reason = "forced_answer_question_max_retries"
                    break
                continue
                
            retry_count = 0  # Reset retry count on valid action
            
            # Get oracle answer
            oracle_resp, is_ambiguous = oracle_answer(clauses, base_context, world, action.questions, sample.cannot_ask, sample.all_forbid_alternative_variables, oracle_type, forbid_alternatives, base_context_wo_assignments, last_round_is_ambiguous=is_ambiguous if turn > 0 else True)
            if verbose:
                print(f"    => Asked: {action.questions}, Oracle: {oracle_resp}, Is target ambiguous? {is_ambiguous}")
            
            # Log this turn
            result.turn_logs.append(TurnLog(
                turn=turn,
                questions=action.questions,
                oracle_answer=oracle_resp,
                is_ambiguous=is_ambiguous,
                model_response=response,
                thinking_tokens_count=thinking_tokens_count,
                cot=cot,
            ))
            result.questions_asked.extend(action.questions)
            
            # Append to conversation
            # NOTE: Use full_response (with cot) for keep_thinking_trace, otherwise use response (CoT already stripped). Note that apply_chat_template strips CoT if the last message is from user, thus changing the role.
            if keep_thinking_trace and not is_random_policy_model(model_name):
                if "qwen" in model_name.lower():
                    messages.append({"role": "assistant", "content": response, "reasoning_content": cot})
                    messages.append({"role": "tool", "content": oracle_resp})
                else:
                    raise NotImplementedError("keep_thinking_trace is only implemented for Qwen models.")
            else:
                if history_append_mode == "full":
                    messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": oracle_resp})
            
            turn += 1
            
        else:  # RETRY
            retry_count += 1
            if retry_count >= max_retries:
                result.final_response = response
                result.final_cot = cot
                result.final_thinking_tokens_count = thinking_tokens_count
                result.turns_used = turn
                result.budget_violated = True
                result.termination_reason = (
                    "forced_answer_retry_max_retries"
                    if force_answer_prompt_applied
                    else "retry_max_retries"
                )
                break
            
            if keep_thinking_trace and not is_random_policy_model(model_name):
                if "qwen" in model_name.lower():
                    messages.append({"role": "assistant", "content": response, "reasoning_content": cot})
                    messages.append({"role": "tool", "content": retry_prompt})
                else:
                    raise NotImplementedError("keep_thinking_trace is only implemented for Qwen models.")
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": retry_prompt})
    
    if not result.answered:
        result.turns_used = turn
        result.budget_violated = True
        if not result.termination_reason:
            result.termination_reason = "unanswered_loop_exit"
    
    # Save to cache
    if cache is not None and cache_file is not None:
        save_episode_to_cache(cache, cache_file, cache_key, result)
    
    return result


async def process_episodes_async(
    episodes: List[Tuple[Sample, World]],
    model_name: str,
    port: str,
    budget: str,
    max_num_q_per_turn: int,
    cache: Optional[Dict],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool = False,
    verbose: bool = False,
    emphasize_uncertainty: bool = True,
    max_concurrent: int = 64,
    is_flipped: bool = False,
    oracle_type: str = "adversarial",
    forbid_alternatives: bool = False,
    include_budget_in_prompt: bool = True,
    enforce_question_limit_in_legacy_bulk: bool = False,
    history_append_mode: str = "full",
    seed: int = 42,
) -> List[EpisodeResult]:
    """Process multiple episodes concurrently with semaphore-based concurrency control."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(sample: Sample, world: World) -> EpisodeResult:
        async with semaphore:
            try:
                return await run_episode_async(
                    model_name=model_name,
                    port=port,
                    sample=sample,
                    world=world,
                    budget=budget,
                    max_num_q_per_turn=max_num_q_per_turn,
                    cache=cache,
                    cache_file=cache_file,
                    generation_config=generation_config,
                    keep_thinking_trace=keep_thinking_trace,
                    verbose=verbose,
                    emphasize_uncertainty=emphasize_uncertainty,
                    is_flipped=is_flipped,
                    oracle_type=oracle_type,
                    forbid_alternatives=forbid_alternatives,
                    include_budget_in_prompt=include_budget_in_prompt,
                    enforce_question_limit_in_legacy_bulk=enforce_question_limit_in_legacy_bulk,
                    history_append_mode=history_append_mode,
                    seed=seed,
                )
            # except Exception as e:
            #     raise
            except Exception as e:
                print(
                    f"[REQUEST_FAILED] sample={sample.sample_id}, "
                    f"world={world.assignments}: {type(e).__name__}: {e}"
                )
                raise
                # print(f"Error processing episode sample={sample.sample_id}, world={world.assignments}: {e}")
                # # Return a failed result
                # result = EpisodeResult(sample_id=sample.sample_id, world=world)
                # result.budget_violated = True
                # return result
    
    # Create all tasks
    tasks = [process_with_semaphore(sample, world) for sample, world in episodes]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return list(results)


# ============ Metrics ============

def compute_minset_f1(asked: List[str], gt_sets: List[List[str]]) -> float:
    """Compute max F1 score vs any ground truth minimal set."""
    if not asked or not gt_sets:
        return 0.0
    
    asked_set = set(q.lower() for q in asked)
    max_f1 = 0.0
    
    for gt_set in gt_sets:
        gt_set_lower = set(q.lower() for q in gt_set)
        intersection = len(asked_set & gt_set_lower)
        if intersection > 0:
            precision = intersection / len(asked_set)
            recall = intersection / len(gt_set_lower)
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)
    
    return max_f1


def compute_minset_jaccard(asked: List[str], gt_sets: List[List[str]]) -> float:
    """Compute max Jaccard similarity vs any ground truth minimal set."""
    if not asked or not gt_sets:
        return 0.0
    
    asked_set = set(q.lower() for q in asked)
    max_jaccard = 0.0
    
    for gt_set in gt_sets:
        gt_set_lower = set(q.lower() for q in gt_set)
        intersection = len(asked_set & gt_set_lower)
        union = len(asked_set | gt_set_lower)
        if union > 0:
            jaccard = intersection / union
            max_jaccard = max(max_jaccard, jaccard)
    
    return max_jaccard


def compute_metrics(results: List[EpisodeResult], samples: List[Sample], forbid_alternatives: bool = False) -> Dict[str, Any]:
    """Compute aggregate metrics + per-k metrics."""
    if not results:
        return {}

    sample_map: Dict[int, Sample] = {s.sample_id: s for s in samples}

    def core(res: List[EpisodeResult]) -> Dict[str, float]:
        if not res:
            return {
                "total_episodes": 0,
                "micro_accuracy": 0.0,
                "macro_accuracy": 0.0,
                "answer_rate": 0.0,
                "avg_questions_used": 0.0,
                "avg_minset_f1": 0.0,
                "avg_minset_jaccard": 0.0,
                "avg_thinking_tokens": 0,
                "avg_num_turns_used": 0,
                "avg_overhead_questions": 0,
            }

        total = len(res)
        correct = sum(1 for r in res if r.correct)
        answered = sum(1 for r in res if r.answered)

        micro_accuracy = correct / total

        # Macro accuracy: average accuracy per sample_id (within this subset)
        sample_accs: Dict[int, List[int]] = {}
        for r in res:
            sample_accs.setdefault(r.sample_id, []).append(1 if r.correct else 0)
        macro_accuracy = (
            sum(sum(xs) / len(xs) for xs in sample_accs.values()) / len(sample_accs)
            if sample_accs else 0.0
        )

        total_questions = sum(len(r.questions_asked) for r in res)
        avg_questions = total_questions / total
        
        total_thinking_tokens = sum([sum([tl.thinking_tokens_count for tl in r.turn_logs]) + r.final_thinking_tokens_count for r in res])
        avg_thinking_tokens = total_thinking_tokens / total
        
        total_turns = sum(r.turns_used for r in res)
        avg_num_turns_used = total_turns / total
        
        total_overhead_questions = sum((len(r.questions_asked) - sample_map[r.sample_id].k) for r in res)
        avg_overhead_questions = total_overhead_questions / total
        
        f1s: List[float] = []
        jaccards: List[float] = []
        for r in res:
            s = sample_map.get(r.sample_id)
            if s is None:
                continue
            if forbid_alternatives:
                f1s.append(compute_minset_f1(r.questions_asked, s.gt_qs))
                jaccards.append(compute_minset_jaccard(r.questions_asked, s.gt_qs))
            else:
                f1s.append(compute_minset_f1(r.questions_asked, s.all_alternative_gt_qs))
                jaccards.append(compute_minset_jaccard(r.questions_asked, s.all_alternative_gt_qs))
        avg_minset_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        avg_minset_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0

        return {
            "total_episodes": total,
            "micro_accuracy": micro_accuracy,
            "macro_accuracy": macro_accuracy,
            "answer_rate": answered / total,
            "avg_questions_used": avg_questions,
            "avg_minset_f1": avg_minset_f1,
            "avg_minset_jaccard": avg_minset_jaccard,
            "avg_thinking_tokens": avg_thinking_tokens,
            "avg_num_turns_used": avg_num_turns_used,
            "avg_overhead_questions": avg_overhead_questions,
        }

    # Overall metrics
    overall = core(results)

    # Per-k metrics
    per_k_results: Dict[int, List[EpisodeResult]] = {}
    for r in results:
        s = sample_map.get(r.sample_id)
        if s is None:
            continue
        per_k_results.setdefault(s.k, []).append(r)

    per_k_metrics: Dict[str, Dict[str, float]] = {}
    for k, res_k in sorted(per_k_results.items(), key=lambda x: x[0]):
        per_k_metrics[str(k)] = core(res_k)

    overall["per_k"] = per_k_metrics
    return overall


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Multi-turn Logic-Q-multi evaluator")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
                        help="Model name. Any model in model_registry.py, a hosted model "
                             "(gpt-*, gemini-*, claude-*), 'random'/'random-baseline', or a "
                             "custom model registered via --model-config.")
    parser.add_argument("--model-config", type=str, default=None,
                        help="YAML/JSON file registering a custom OpenAI-compatible model "
                             "(see model_registry.load_model_config_file).")
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        help="Reasoning effort level for GPT-OSS, GPT-5.2, GPT-5.4 models", choices=["low", "medium", "high"])
    parser.add_argument("--port", type=str, default="8011", help="vllm port")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Optional max output tokens override. Defaults to model-specific Evaluator setting when unset.")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to data CSV file")
    parser.add_argument("--cache-tag", type=str, default="",
                        help="Tag to identify cache files")
    parser.add_argument("--budget", type=str, default="4",
                        help="Maximum number of turns (k uses sample.k; legacy: 0 means one turn with many questions if --max-num-q-per-turn=1)", choices=["k"] + [str(i) for i in range(0, 21)])
    parser.add_argument("--max-num-q-per-turn", type=int, default=1,
                        help="Maximum number of questions allowed in one turn")
    parser.add_argument("--keep-thinking-trace", action="store_true",
                        help="Keep thinking trace in conversation (local models only)")
    parser.add_argument("--results-dir", type=str, default="results/SL/multiturn",
                        help="Directory to save results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging per episode")
    parser.add_argument("--no-emphasize-uncertainty", dest="emphasize_uncertainty",
                        action="store_false", default=True,
                        help="Disable the prompt emphasis that tells model to ask questions first")
    parser.add_argument("--max-concurrent", type=int, default=64,
                        help="Maximum concurrent episodes for async")
    parser.add_argument("--sufficient-sets-per-problem", type=int, default=1)
    parser.add_argument("--worlds-per-sufficient-set-per-sign", type=int, default=1)
    parser.add_argument("--oracle-type", type=str, default="adversarial", help="Type of oracle to use (e.g., adversarial, cooperative, random)", choices=["adversarial", "cooperative", "random"])
    parser.add_argument("--forbid-alternatives", dest="forbid_alternatives",
                        action="store_true", default=False,
                        help="Forbid asking about variables in alternative ground truths")
    parser.add_argument("--no-budget-in-prompt", dest="include_budget_in_prompt",
                        action="store_false", default=True,
                        help="Do not include budget information in the prompt")
    parser.add_argument("--enforce-question-limit-in-legacy-bulk", action="store_true",
                        help="Also enforce max allowed questions during legacy budget=0 bulk parsing.")
    parser.add_argument("--history-append-mode", type=str, default="full",
                        choices=["full", "oracle_only"],
                        help="How to append prior turn history into future prompts.")
    parser.add_argument("--dump-turn-prompts", action="store_true",
                        help="Mark this run as saving per-call prompt messages in cache/output names. LogicQ call_logs already store prompt_messages.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling worlds")
    parser.add_argument("--no-flip", dest="do_flip", action="store_false", default=True, help="Disable flipped episodes evaluation")
    args = parser.parse_args()
    if args.max_num_q_per_turn < 1:
        raise ValueError("--max-num-q-per-turn must be >= 1")

    # Register a custom OpenAI-compatible model, then validate the name against
    # the shared registry (replaces the old hardcoded argparse allowlist).
    if getattr(args, "model_config", None):
        registered = model_registry.load_model_config_file(args.model_config)
        print(f"Registered custom model from {args.model_config}: {registered}")
    model_registry.validate_model_name(
        args.model_name, extra_allowed=sorted(RANDOM_POLICY_MODEL_NAMES)
    )
    
    # NOTE: These two are no longer runtime knobs -- the released data is already
    # pre-sampled to exactly 1 sufficient set per problem and 1 world per sign
    # (see data_postprocessing.ipynb). They are retained only so the results
    # filename keeps the canonical "1sets-1worlds" tag.
    assert args.sufficient_sets_per_problem == 1, "sufficient_sets_per_problem must be 1 for consistency"
    assert args.worlds_per_sufficient_set_per_sign == 1, "worlds_per_sufficient_set_per_sign must be 1 for consistency"

    # Setup directories
    model_name_safe = args.model_name.replace("/", "_") + (f"_{args.reasoning_effort}" if ("gpt-oss" in args.model_name.lower() or "gpt-5.4" in args.model_name.lower() or "gpt-5.2" in args.model_name.lower()) else "")
    results_dir = os.path.join(args.results_dir, model_name_safe)
    os.makedirs(results_dir, exist_ok=True)
    cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load data and setup cache
    # NOTE: Data has been pre-sampled with sufficient sets and worlds per sign to ensure consistency across runs
    data_file = args.data_file
    if args.cache_tag:
        cache_tag = args.cache_tag
    else:
        cache_tag = os.path.splitext(os.path.basename(data_file))[0]
    
    budget_tag = f"budget{args.budget}" if args.max_num_q_per_turn == 1 else f"budget{args.budget}x{args.max_num_q_per_turn}"
    output_name = f"{cache_tag}-{budget_tag}-{args.sufficient_sets_per_problem}sets-{args.worlds_per_sufficient_set_per_sign}worlds-{args.oracle_type}_oracle{'-forbid_alternatives' if args.forbid_alternatives else ''}-{'flip' if args.do_flip else 'noflip'}-seed{args.seed}{'-nobudgetprompt' if not args.include_budget_in_prompt else ''}"
    if args.keep_thinking_trace:
        output_name += "-keepthink"
    if args.history_append_mode != "full":
        output_name += f"-history{args.history_append_mode}"
    if not args.emphasize_uncertainty:
        output_name += "-noemphuncert"
    if args.dump_turn_prompts:
        output_name += "-dumpprompt"
    
    # Episode cache (for async path)
    episode_cache_file = os.path.join(cache_dir, f"{output_name}_episodes.jsonl")
    episode_cache = load_episode_cache(episode_cache_file) if os.path.exists(episode_cache_file) else {}
    
    # Generation config
    generation_config = (
        {}
        if is_random_policy_model(args.model_name)
        else Evaluator(
            model_name=args.model_name,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
        ).generation_config
    )
    
    # Load data
    print(f"Loading data from {data_file}")
    samples = load_data(data_file, args.oracle_type)

    print(f"Loaded {len(samples)} samples")
    
    # Count total episodes
    total_episodes = sum(len(s.worlds) for s in samples)
    print(f"Total episodes (samples x worlds): {total_episodes}")
    
    # Build episode list (original)
    episodes = [(sample, world) for sample in samples for world in sample.worlds]
    
    # Create flipped samples and episodes
    if args.do_flip:
        flipped_samples = [create_flipped_sample(s) for s in samples]
        flipped_episodes = [(sample, world) for sample in flipped_samples for world in sample.worlds]
        print(f"Total flipped episodes: {len(flipped_episodes)}")
    else:
        flipped_samples = []
        flipped_episodes = []
        print("Flipping disabled (--no_flip).")
    
    # Run evaluation
    # if args.use_async:
    print(f"Running async with max_concurrent={args.max_concurrent}")
    all_results = asyncio.run(
        process_episodes_async(
            episodes=episodes,
            model_name=args.model_name,
            port=args.port,
            budget=args.budget,
            max_num_q_per_turn=args.max_num_q_per_turn,
            cache=episode_cache,
            cache_file=episode_cache_file,
            generation_config=generation_config,
            keep_thinking_trace=args.keep_thinking_trace,
            verbose=args.verbose,
            emphasize_uncertainty=args.emphasize_uncertainty,
            max_concurrent=args.max_concurrent,
            is_flipped=False,
            oracle_type=args.oracle_type,
            forbid_alternatives=args.forbid_alternatives,
            include_budget_in_prompt=args.include_budget_in_prompt,
            enforce_question_limit_in_legacy_bulk=args.enforce_question_limit_in_legacy_bulk,
            history_append_mode=args.history_append_mode,
            seed=args.seed,
        )
    )
    
    # Run flipped episodes evaluation
    if args.do_flip:
        print("\n--- Running flipped episodes ---")
        
        # if args.use_async:
        flipped_results = asyncio.run(
            process_episodes_async(
                episodes=flipped_episodes,
                model_name=args.model_name,
                port=args.port,
                budget=args.budget,
                max_num_q_per_turn=args.max_num_q_per_turn,
                cache=episode_cache,
                cache_file=episode_cache_file,
                generation_config=generation_config,
                keep_thinking_trace=args.keep_thinking_trace,
                verbose=args.verbose,
                emphasize_uncertainty=args.emphasize_uncertainty,
                max_concurrent=args.max_concurrent,
                is_flipped=True,
                oracle_type=args.oracle_type,
                forbid_alternatives=args.forbid_alternatives,
                include_budget_in_prompt=args.include_budget_in_prompt,
                enforce_question_limit_in_legacy_bulk=args.enforce_question_limit_in_legacy_bulk,
                history_append_mode=args.history_append_mode,
                seed=args.seed,
            )
        )
        
    else:
        flipped_results = []
        flipped_metrics = None
        
    # Compute metrics for original
    metrics = compute_metrics(all_results, samples, forbid_alternatives=args.forbid_alternatives)
    print("\n=== Original Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Compute metrics for flipped
    if args.do_flip:
        flipped_metrics = compute_metrics(flipped_results, flipped_samples, forbid_alternatives=args.forbid_alternatives)
        print("\n=== Flipped Results ===")
        for k, v in flipped_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    results_file = os.path.join(results_dir, f"{output_name}_results.json")
    
    # Helper to build episode data
    def build_episode_data(r: EpisodeResult, sample: Sample, is_flipped=False):
        return {
            "sample_id": r.sample_id,
            "is_flipped": is_flipped,
            "k": sample.k,
            "gt_qs": sample.gt_qs,
            "world": r.world.assignments,
            "target_gt": r.world.target_value,
            "variables": r.world.variables,
            "final_answer": r.final_answer,
            "correct": r.correct,
            "answered": r.answered,
            "turns_used": r.turns_used,
            "questions_asked": r.questions_asked,
            "turn_logs": [
                {
                    "turn": t.turn,
                    "questions": t.questions,
                    "oracle_answer": t.oracle_answer,
                    "is_ambiguous": t.is_ambiguous,
                    "model_response": t.model_response,
                    "cot": t.cot,
                    "thinking_tokens_count": t.thinking_tokens_count,
                }
                for t in r.turn_logs
            ],
            "final_response": r.final_response,
            "final_cot": r.final_cot,
            "final_thinking_tokens_count": r.final_thinking_tokens_count,
        }
    
    # Convert results to serializable format
    results_data = {
        "config": vars(args),
        "metrics_original": metrics,
        "metrics_flipped": flipped_metrics,
        "samples_meta": {
            s.sample_id: {
                "k": s.k,
                "gt_qs": s.gt_qs,
                "goal": s.goal,
            }
            for s in samples
        },
        "episodes_original": [],
        "episodes_flipped": [],
    }
    
    # Build sample_id -> sample lookup
    sample_by_id = {s.sample_id: s for s in samples}
    flipped_sample_by_id = {s.sample_id: s for s in flipped_samples}
    
    for r in all_results:
        sample = sample_by_id.get(r.sample_id)
        results_data["episodes_original"].append(build_episode_data(r, sample, is_flipped=False))
    
    for r in flipped_results:
        sample = flipped_sample_by_id.get(r.sample_id)
        results_data["episodes_flipped"].append(build_episode_data(r, sample, is_flipped=True))
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
