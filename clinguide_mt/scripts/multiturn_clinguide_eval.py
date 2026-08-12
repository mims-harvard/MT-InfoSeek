"""Multi-turn Clinical Guideline (ClinGuide) evaluation script.

Run from the repo root (MT-InfoSeek/) so that model_utils and
evaluators are on the Python path:

    export PYTHONPATH=.

Usage examples:

    # Patient context only
    python clinguide_mt/scripts/multiturn_clinguide_eval.py --json-dir /path/to/data --context-mode patient_only

    # Patient context + question list (default)
    python clinguide_mt/scripts/multiturn_clinguide_eval.py --json-dir /path/to/data --context-mode patient_question_list

    # Patient context + guideline
    python clinguide_mt/scripts/multiturn_clinguide_eval.py --json-dir /path/to/data --context-mode patient_guideline --num-distractors 0 --budget k

    # Patient context + question list + guideline
    python clinguide_mt/scripts/multiturn_clinguide_eval.py --json-dir /path/to/data --context-mode patient_question_list_and_guideline

    # With multiple guidelines (1 correct + distractors)
    python clinguide_mt/scripts/multiturn_clinguide_eval.py --json-dir /path/to/data --context-mode patient_guideline --max-guidelines 3

    # Adversarial vs cooperative oracle
    python clinguide_mt/scripts/multiturn_clinguide_eval.py --json-dir /path/to/data --oracle-mode adversarial
    python clinguide_mt/scripts/multiturn_clinguide_eval.py --json-dir /path/to/data --oracle-mode cooperative
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm as async_tqdm

from model_utils import async_generate_single
from logic_q_mt.evaluation.evaluators.evaluator import Evaluator

# ── Constants ─────────────────────────────────────────────────────────────────

# Model used as oracle judge and for semantic answer-correctness checking.
# Configurable via the ORACLE_MODEL env var (defaults to gpt-5).
ORACLE_MODEL = os.environ.get("ORACLE_MODEL", "gpt-5")

# Suffix shared by all curated decision-tree JSON files.
_V2_TREE_SUFFIX = "_data.json"


# ============ Data Classes ============

@dataclass
class World:
    """A specific assignment of values to the queried variables."""
    final_outcome: str
    k: int
    desired_questions: List[str]
    desired_questions_answers: List[str]
    context: Dict[str, str]
    patient_context: str


@dataclass
class Sample:
    """A single problem instance (one clinical guideline)."""
    sample_id: str
    worlds: List[World]
    all_final_outcomes: List[str]
    allowed_question_topics: List[str]  # own questions + flattened distractor guidelines
    assessment: str = ""
    num_distractors_included: int = 0   # number of distractor guideline sets included in allowed_topics
    guideline_text: str = ""            # readable text rendered from *data.json


@dataclass
class TurnLog:
    """Log entry for a single conversation turn."""
    turn: int
    questions: List[str]
    matched_questions: List[str]
    oracle_answer: str
    model_response: str
    thinking_tokens_count: int
    cot: str = ""


@dataclass
class EpisodeResult:
    """Result of a single episode (sample × world)."""
    sample_id: str
    world: World
    final_answer: Optional[str] = None
    correct: bool = False
    answered: bool = False
    turns_used: int = 0
    questions_asked: List[str] = field(default_factory=list)
    matched_questions_asked: List[str] = field(default_factory=list)
    turn_logs: List[TurnLog] = field(default_factory=list)
    final_cot: str = ""
    final_thinking_tokens_count: int = 0
    final_response: str = ""
    budget_violated: bool = False
    # Non-empty if the episode aborted (e.g. model API error).
    episode_error: str = ""


@dataclass
class Action:
    """Parsed action from a model response."""
    type: str   # "QUESTION" | "ANSWER" | "RETRY"
    questions: List[str] = field(default_factory=list)
    value: Optional[str] = None


# ============ Prompting ============

SYSTEM_PROMPT_MULTITURN_WITH_TOPICS_TEMPLATE = """You are a clinical decision support system. You may ONLY ask questions about the following allowed topics.

ALLOWED QUESTION TOPICS:
{allowed_topics}

POSSIBLE RECOMMENDATIONS:
{all_outcomes}

INSTRUCTIONS:
You will be given an initial patient presentation and may ask up to {budget} follow-up questions over multiple turns. Your goal is to select the correct recommendation from POSSIBLE RECOMMENDATIONS. At each turn:
- If you have enough information to determine a single recommendation, respond strictly with: "Answer: [recommendation]".
- Otherwise, ask exactly one question to obtain the most critical missing information. Respond strictly with: "Question: [question]?"
- You may ONLY ask questions that relate to the ALLOWED QUESTION TOPICS above. Do NOT ask about topics not listed.
- Do not output any other text.

QUESTION SELECTION STRATEGY:
Not all allowed topics may be equally informative for this patient. At each turn, choose the single question whose answer is most likely to distinguish among the POSSIBLE RECOMMENDATIONS above, allowing you to reach the correct recommendation in the fewest total questions."""

SYSTEM_PROMPT_MULTITURN_NO_TREE_TEMPLATE = """You are a clinical decision support system for making patient recommendations.

POSSIBLE RECOMMENDATIONS:
{all_outcomes}

You will be given an initial patient presentation and may ask up to {budget} follow-up questions over multiple turns. Your goal is to select the correct recommendation from POSSIBLE RECOMMENDATIONS. At each turn:

Instructions:
1. If you already have enough information to make an accurate recommendation, respond strictly with: "Answer: [recommendation]".
2. Otherwise, ask exactly one question. Choose the single question that provides the most information for distinguishing among the POSSIBLE RECOMMENDATIONS, so the correct recommendation can be reached in the fewest total questions.
3. Format the question strictly as: "Question: [question]?"
4. Do not output any other text."""

SYSTEM_PROMPT_SINGLETURN_WITH_TOPICS_TEMPLATE = """You are a clinical decision support system. You may ONLY ask questions about the following allowed topics.

ALLOWED QUESTION TOPICS:
{allowed_topics}

POSSIBLE RECOMMENDATIONS:
{all_outcomes}

You will be given an initial patient presentation. Your goal is to select the correct recommendation from POSSIBLE RECOMMENDATIONS by asking only the questions you need.

Instructions:
1. If you already have enough information to make an accurate recommendation, respond strictly with: "Answer: [recommendation]".
2. Otherwise, select the smallest set of questions (at least 1) whose answers together are sufficient to determine the recommendation. Focus only on topics that can distinguish among the POSSIBLE RECOMMENDATIONS for this patient; avoid asking about topics that are unlikely to affect the outcome.
3. You may ONLY ask questions that relate to the ALLOWED QUESTION TOPICS above.
4. Format strictly as: "Question: [question_1]? Question: [question_2]? ..." (for at least 1 question).
5. Do not output any other text."""

SYSTEM_PROMPT_SINGLETURN_NO_TREE_TEMPLATE = """You are a clinical decision support system for making patient recommendations.

POSSIBLE RECOMMENDATIONS:
{all_outcomes}

You will be given an initial patient presentation. Your goal is to select the correct recommendation from POSSIBLE RECOMMENDATIONS by asking only the questions you need.

Instructions:
1. If you already have enough information to make an accurate recommendation, respond strictly with: "Answer: [recommendation]".
2. Otherwise, select the smallest set of questions (at least 1) whose answers together are sufficient to determine the recommendation. Choose questions that most directly distinguish among the POSSIBLE RECOMMENDATIONS.
3. Format strictly as: "Question: [question_1]? Question: [question_2]? ..." (for at least 1 question).
4. Do not output any other text."""

UNCERTAINTY_EMPHASIS = """

IMPORTANT: The initial facts provided are INSUFFICIENT. You MUST ask questions to gather the missing information before you can answer correctly. Do not guess - ask questions first."""

DISTRACTOR_WARNING = """

NOTE: Some of the allowed question topics are drawn from other clinical guidelines and are distractors — they are not relevant to the POSSIBLE RECOMMENDATIONS for this patient. Before asking, critically assess whether each topic can actually distinguish among the recommendations listed above. Skip topics that cannot."""

DISTRACTOR_WARNING_WITH_COUNT = """

NOTE: Questions from {num_distractors} other clinical guideline(s) are included as distractors in the allowed question topics above. These topics concern different clinical conditions and cannot distinguish among the POSSIBLE RECOMMENDATIONS for this patient. Before asking, critically assess whether each topic is genuinely informative for the recommendations listed above. Skip topics that are not."""


def build_initial_prompt(
    sample: Sample,
    world: World,
    budget: int,
    emphasize_uncertainty: bool = True,
    is_single_turn: bool = False,
    context_mode: str = "patient_question_list",
    with_examples: bool = False,
    with_distractor_warning: bool = False,
    max_guidelines: int = 1,
    guideline_by_id: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build the initial system + user prompt for the model.

    context_mode:
      patient_only                    patient_context only
      patient_question_list           patient_context + allowed question topics
      patient_guideline               patient_context + guideline(s) from *data.json
      patient_question_list_and_guideline  patient_context + question list + guideline(s)

    max_guidelines: when guideline is shown, how many to include
                    (1 = current only; >1 adds distractor guidelines).
    """
    with_tree = context_mode in ("patient_question_list", "patient_question_list_and_guideline")
    with_guideline = context_mode in ("patient_guideline", "patient_question_list_and_guideline")
    allowed_topics_text = "\n".join(f"- {t}" for t in sample.allowed_question_topics)

    if not is_single_turn:
        if with_tree:
            system_content = SYSTEM_PROMPT_MULTITURN_WITH_TOPICS_TEMPLATE.format(
                allowed_topics=allowed_topics_text,
                all_outcomes="\n".join(sample.all_final_outcomes),
                budget=budget,
            )
        else:
            system_content = SYSTEM_PROMPT_MULTITURN_NO_TREE_TEMPLATE.format(
                all_outcomes="\n".join(sample.all_final_outcomes),
                budget=budget,
            )
    else:
        if with_tree:
            system_content = SYSTEM_PROMPT_SINGLETURN_WITH_TOPICS_TEMPLATE.format(
                allowed_topics=allowed_topics_text,
                all_outcomes="\n".join(sample.all_final_outcomes),
            )
        else:
            system_content = SYSTEM_PROMPT_SINGLETURN_NO_TREE_TEMPLATE.format(
                all_outcomes="\n".join(sample.all_final_outcomes),
            )

    if emphasize_uncertainty:
        system_content += UNCERTAINTY_EMPHASIS
    if with_distractor_warning and with_tree:
        if sample.num_distractors_included > 0:
            system_content += DISTRACTOR_WARNING_WITH_COUNT.format(
                num_distractors=sample.num_distractors_included
            )
        else:
            system_content += DISTRACTOR_WARNING

    user_content = "You are presented a patient with the following information:\n" + world.patient_context

    if with_guideline and sample.guideline_text:
        rng = random.Random(seed + sum(ord(c) for c in sample.sample_id))
        guidelines_to_show: List[Tuple[str, str]] = [(sample.sample_id, sample.guideline_text)]
        if max_guidelines > 1 and guideline_by_id:
            others = [sid for sid in guideline_by_id if sid != sample.sample_id and guideline_by_id[sid]]
            n_extra = min(max_guidelines - 1, len(others))
            if n_extra > 0:
                chosen = rng.sample(others, n_extra)
                for sid in chosen:
                    guidelines_to_show.append((sid, guideline_by_id[sid]))
                rng.shuffle(guidelines_to_show)
        guideline_blocks = []
        for i, (_, gtext) in enumerate(guidelines_to_show):
            label = f"Guideline {i + 1}" if len(guidelines_to_show) > 1 else "Clinical guideline"
            guideline_blocks.append(f"--- {label} ---\n{gtext}")
        user_content += "\n\nCLINICAL GUIDELINE(S):\n\n" + "\n\n".join(guideline_blocks)

    user_content += "\n\nWhat recommendation should you give to the patient?"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


FORCE_ANSWER_PROMPT = (
    '\n\nYou have used all your questions. You must now provide your final answer '
    'in the format: "Answer: [recommendation]"'
)

RETRY_PROMPT = """Could not parse your response. Please respond with exactly one of:
- "Question: {question}" to ask a question about the patient
- "Answer: {recommendation}" to provide your final answer"""


# ============ Episode Caching ============

def make_episode_cache_key(
    sample: Sample,
    world: World,
    context_mode: str = "patient_question_list",
    max_guidelines: int = 1,
    oracle_mode: str = "standard",
    budget: int = 4,
    emphasize_uncertainty: bool = True,
    with_examples: bool = False,
    with_distractor_warning: bool = False,
    seed: int = 42,
    show_past_qa_to_oracle: bool = False,
) -> str:
    """Return a deterministic cache key for a sample × world × prompt config."""
    key_dict = {
        "sample_id": sample.sample_id,
        "world_final_outcome": world.final_outcome,
        "world_patient_context": world.patient_context,
        "world_desired_questions": world.desired_questions,
        "context_mode": context_mode,
        "max_guidelines": max_guidelines,
        "oracle_mode": oracle_mode,
        "budget": budget,
        "emphasize_uncertainty": emphasize_uncertainty,
        "with_examples": with_examples,
        "with_distractor_warning": with_distractor_warning,
        "seed": seed,
        "show_past_qa_to_oracle": show_past_qa_to_oracle,
    }
    return json.dumps(key_dict, sort_keys=True)


def episode_result_to_dict(result: EpisodeResult) -> Dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "world": {
            "final_outcome": result.world.final_outcome,
            "k": result.world.k,
            "desired_questions": result.world.desired_questions,
            "desired_questions_answers": result.world.desired_questions_answers,
            "context": result.world.context,
            "patient_context": result.world.patient_context,
        },
        "final_answer": result.final_answer,
        "correct": result.correct,
        "answered": result.answered,
        "turns_used": result.turns_used,
        "questions_asked": result.questions_asked,
        "matched_questions_asked": result.matched_questions_asked,
        "turn_logs": [
            {
                "turn": t.turn,
                "questions": t.questions,
                "matched_questions": t.matched_questions,
                "oracle_answer": t.oracle_answer,
                "model_response": t.model_response,
                "cot": t.cot,
                "thinking_tokens_count": t.thinking_tokens_count,
            }
            for t in result.turn_logs
        ],
        "final_response": result.final_response,
        "final_cot": result.final_cot,
        "final_thinking_tokens_count": result.final_thinking_tokens_count,
        "budget_violated": result.budget_violated,
        "episode_error": result.episode_error,
    }


def dict_to_episode_result(d: Dict[str, Any]) -> EpisodeResult:
    world = World(
        final_outcome=d["world"]["final_outcome"],
        k=d["world"]["k"],
        desired_questions=d["world"]["desired_questions"],
        desired_questions_answers=d["world"]["desired_questions_answers"],
        context=d["world"]["context"],
        patient_context=d["world"]["patient_context"],
    )
    turn_logs = [
        TurnLog(
            turn=t["turn"],
            questions=t["questions"],
            matched_questions=t["matched_questions"],
            oracle_answer=t["oracle_answer"],
            model_response=t["model_response"],
            cot=t["cot"],
            thinking_tokens_count=t["thinking_tokens_count"],
        )
        for t in d.get("turn_logs", [])
    ]
    return EpisodeResult(
        sample_id=d["sample_id"],
        world=world,
        final_answer=d.get("final_answer"),
        correct=d.get("correct", False),
        answered=d.get("answered", False),
        turns_used=d.get("turns_used", 0),
        questions_asked=d.get("questions_asked", []),
        matched_questions_asked=d.get("matched_questions_asked", []),
        turn_logs=turn_logs,
        final_response=d.get("final_response", ""),
        final_cot=d.get("final_cot", ""),
        final_thinking_tokens_count=d.get("final_thinking_tokens_count", 0),
        budget_violated=d.get("budget_violated", False),
        episode_error=d.get("episode_error", ""),
    )


def load_episode_cache(cache_file: str) -> Dict[str, Dict[str, Any]]:
    """Load episode cache from a JSONL file."""
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


def save_episode_to_cache(
    cache: Dict[str, Dict],
    cache_file: str,
    key: str,
    result: EpisodeResult,
) -> None:
    result_dict = episode_result_to_dict(result)
    cache[key] = result_dict
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "a") as f:
        f.write(json.dumps({"key": key, "result": result_dict}) + "\n")


# ============ Oracle (LLM as a judge) ============

ORACLE_SYSTEM_PROMPT = """You are an oracle that answers clinical questions about a patient based ONLY on provided information.

You will receive:
1. A list of desired question-answer pairs about the patient. The desired question might be a summary or concept that is desirable to be asked about, but not an actual question.
2. One or more questions to answer
3. (Optional) A history of previously asked questions and their answers for context

For each question you must:
1. Try to match it with one of the desired questions (allow paraphrasing, e.g., "patient's age" matches "age of the patient", "is the patient pregnant" matches "pregnancy test", etc.)
2. If matched, output the corresponding answer from the desired pairs
3. If NO match exists, output "Not sure"

CRITICAL RULES:
- ONLY answer based on the provided question-answer pairs
- DO NOT make up or infer any information
- If a question cannot be matched to a desired question, you MUST output "Not sure"
- Allow matching for questions that are paraphrased but semantically equivalent
- The past Q&A history is provided ONLY for context/reference - DO NOT exclude any options based on what was previously answered
- Even if a question was asked before, you should still match and answer it normally

Output format (JSON array, one entry per input question):
[
  {"matched_question": <the matched question from desired questions>, "answer": <the matched question's answer from desired questions' answers>},
  {"matched_question": null, "answer": "Not sure"},
  ...
]"""


ORACLE_SYSTEM_PROMPT_ADVERSARIAL = """You are an oracle that answers clinical questions about a patient using ONLY the provided information. You follow an **adversarial (least-disclosure)** policy.

You will receive:
1. A list of desired question-answer pairs about the patient. The desired question might be a summary or concept that is desirable to be asked about, but not an actual question.
2. One or more questions to answer
3. (Optional) A history of previously asked questions and their answers for context

For each question you must:
1. Try to match it with one of the desired questions (allow paraphrasing, e.g., "patient's age" matches "age of the patient", "is the patient pregnant" matches "pregnancy test", etc.)
2. If matched, output the corresponding answer from the desired pairs
3. If NO match exists, output "Not sure"

ADVERSARIAL POLICY:
- Answer truthfully, but choose wording that provides the **least useful information** for determining the correct recommendation
- Prefer **minimal, less specific, or non-committal phrasing** when multiple truthful options exist
- Do NOT state or imply the final recommendation unless explicitly asked
- Do NOT add explanations, reasoning, or decision guidance
- If a fact is highly revealing, express it in the most **neutral and minimal** way possible

RULES:
- Use ONLY the provided question–answer pairs (no inference, no new facts)
- If no match exists: "Not sure"
- Past Q&A is context only (do not restrict answers based on it)

Output format (JSON array, one per question):
[
  {"matched_question": <string or null>, "answer": <string>}
]
"""


ORACLE_SYSTEM_PROMPT_COOPERATIVE = """You are an oracle that answers clinical questions about a patient using ONLY the provided information. You follow a **cooperative (most-helpful)** policy.

You will receive:
1. A list of desired question-answer pairs about the patient. The desired question might be a summary or concept that is desirable to be asked about, but not an actual question.
2. One or more questions to answer
3. (Optional) A history of previously asked questions and their answers for context

For each question you must:
1. Try to match it with one of the desired questions (allow paraphrasing, e.g., "patient's age" matches "age of the patient", "is the patient pregnant" matches "pregnancy test", etc.)
2. If matched, output the corresponding answer from the desired pairs
3. If NO match exists, output "Not sure"

COOPERATIVE POLICY:
- Answer truthfully using wording that makes it **easier to determine the correct recommendation**
- Prefer **clear, explicit, and decision-relevant phrasing**
- When supported by the answer, briefly clarify why the fact matters for choosing among the listed recommendations
- Maximize clarity when ambiguity would make the decision harder

RULES:
- Use ONLY the provided question–answer pairs (no inference, no new facts)
- Do NOT introduce new clinical reasoning beyond what is supported by the answer
- If no match exists: "Not sure"
- Past Q&A is context only (do not restrict answers based on it)

Output format (JSON array, one per question):
[
  {"matched_question": <string or null>, "answer": <string>}
]
"""


async def oracle_answer(
    world: World,
    questions: List[str],
    past_qa_pairs: Optional[List[Tuple[str, str]]] = None,
    oracle_mode: str = "standard",
    all_final_outcomes: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """Answer questions using an LLM oracle (ORACLE_MODEL).

    Args:
        world: Current world containing desired questions and ground-truth answers.
        questions: Questions from the model to answer.
        past_qa_pairs: Optional (question, oracle_answer) pairs from prior turns,
                       used as context only — does NOT exclude options.
        oracle_mode: ``standard`` | ``adversarial`` | ``cooperative``
        all_final_outcomes: Candidate recommendations (required for adversarial/cooperative).

    Returns:
        (concatenated_answers_string, matched_question_list)
    """
    if oracle_mode not in ("standard", "adversarial", "cooperative"):
        raise ValueError(f"oracle_mode must be standard|adversarial|cooperative, got {oracle_mode!r}")

    outcomes_for_prompt: Optional[List[str]] = None
    if oracle_mode != "standard":
        outcomes_for_prompt = list(all_final_outcomes) if all_final_outcomes else [world.final_outcome]

    desired_pairs_text = "\n\n".join(
        f"{i + 1}.\nDesired question: {q}\nDesired answer: {a}"
        for i, (q, a) in enumerate(zip(world.desired_questions, world.desired_questions_answers))
    )
    questions_text = "\n".join(f"- {q}" for q in questions)

    past_history_text = ""
    if past_qa_pairs:
        lines = "\n\n".join(
            f"{i + 1}.\nQuestion: {q}\nAnswer: {a}" for i, (q, a) in enumerate(past_qa_pairs)
        )
        past_history_text = (
            f"\n\nPrevious questions and answers (for context only - DO NOT exclude options based on this):\n{lines}"
        )

    if oracle_mode == "standard":
        stance_block = ""
        system_content = ORACLE_SYSTEM_PROMPT
    else:
        assert outcomes_for_prompt is not None
        out_lines = "\n".join(f"- {o}" for o in outcomes_for_prompt)
        stance_block = (
            f"GROUND-TRUTH FINAL RECOMMENDATION for this patient scenario (for oracle reasoning only):\n"
            f"{world.final_outcome}\n\n"
            f"POSSIBLE RECOMMENDATIONS the decision agent may output (must choose one):\n{out_lines}\n\n"
        )
        system_content = (
            ORACLE_SYSTEM_PROMPT_ADVERSARIAL
            if oracle_mode == "adversarial"
            else ORACLE_SYSTEM_PROMPT_COOPERATIVE
        )

    user_prompt = (
        f"{stance_block}Desired question-answer pairs about this patient:\n"
        f"{desired_pairs_text}{past_history_text}\n\n"
        f"Questions to answer:\n{questions_text}\n\n"
        "Output the JSON array with your answers:"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = await async_generate_single(
            model_name=ORACLE_MODEL,
            port="",
            messages=messages,
            generation_config={"max_completion_tokens": 1024},
        )
    except Exception as e:
        print(
            f"WARNING: Oracle API request failed; returning 'Not sure' for all questions. "
            f"({type(e).__name__}: {e})"
        )
        return " ".join(["Not sure"] * len(questions)), [None] * len(questions)

    response_text = result.text.strip()
    matched_questions: List[Optional[str]] = []
    answers: List[str] = []

    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(response_text)
        for entry in parsed:
            matched_q = entry.get("matched_question")
            answer = entry.get("answer", "Not sure")
            if matched_q is not None:
                matched_idx = next(
                    (idx for idx, dq in enumerate(world.desired_questions)
                     if dq.lower().strip() == matched_q.lower().strip()),
                    None,
                )
                if matched_idx is not None:
                    matched_questions.append(world.desired_questions[matched_idx])
                    answers.append(world.desired_questions_answers[matched_idx])
                else:
                    matched_questions.append(matched_q)
                    answers.append(answer)
            else:
                matched_questions.append(None)
                answers.append("Not sure")

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"WARNING: Failed to parse oracle response: {e}\nResponse: {response_text}")
        matched_questions = [None] * len(questions)
        answers = ["Not sure"] * len(questions)

    if len(questions) == 1:
        return answers[0] if answers else "Not sure", matched_questions

    qa_lines = [f"- {q} → {a}" for q, a in zip(questions, answers)]
    return "\n".join(qa_lines), matched_questions


# ============ Data Loading ============

def _tree_node_to_text(node: Dict[str, Any], indent: int = 0) -> str:
    """Recursively render a *data.json tree node as human-readable text."""
    if not node:
        return ""
    node_type = node.get("type", "")
    label = node.get("label", "")
    prefix = "  " * indent

    if node_type == "assessment":
        lines = [f"{prefix}Assessment: {label}"]
        if "next" in node:
            lines.append(_tree_node_to_text(node["next"], indent + 1))
        return "\n".join(lines)

    if node_type == "condition":
        lines = [f"{prefix}If {label}:"]
        if "branches" in node:
            for branch_val, child in node["branches"].items():
                lines.append(f"{prefix}  {branch_val}:")
                lines.append(_tree_node_to_text(child, indent + 2))
        elif "next" in node:
            lines.append(_tree_node_to_text(node["next"], indent + 1))
        return "\n".join(lines)

    if node_type == "decision":
        return f"{prefix}-> {label}"

    return f"{prefix}{label}"


def load_data(json_dir: str, num_distractors: int = -1) -> Tuple[List[Sample], Dict[str, str]]:
    """Load samples from *_tree.json, *_question_list_all.json, and *_data.json.

    Args:
        json_dir: Directory containing all data files.
        num_distractors: Number of distractor guideline sets to include (-1 = all).

    Returns:
        (samples, guideline_by_id) — the latter is used when max_guidelines > 1.
    """
    all_sample_ids = [
        f[: -len("_tree.json")]
        for f in os.listdir(json_dir)
        if f.endswith("_tree.json")
    ]
    guideline_by_id: Dict[str, str] = {}
    samples: List[Sample] = []

    for sample_id in all_sample_ids:
        question_list_file = os.path.join(json_dir, f"{sample_id}_question_list_all.json")
        if not os.path.exists(question_list_file):
            raise FileNotFoundError(f"Question list file not found: {question_list_file}")

        with open(question_list_file, "r") as f:
            qlist = json.load(f)
        questions = qlist.get("questions", [])
        distractors_by_guideline = qlist.get("distractors", {})
        assessment = qlist.get("assessment", "")

        guideline_keys = sorted(distractors_by_guideline.keys(), key=lambda x: int(x))
        if num_distractors >= 0:
            guideline_keys = guideline_keys[:num_distractors]
        selected_distractors = [
            q for k in guideline_keys for q in distractors_by_guideline[k].get("questions", [])
        ]
        allowed_question_topics = questions + selected_distractors
        num_distractors_included = len(guideline_keys)

        guideline_text = ""
        v2_file = os.path.join(json_dir, f"{sample_id}{_V2_TREE_SUFFIX}")
        if os.path.exists(v2_file):
            with open(v2_file, "r") as f:
                tree = json.load(f)
            guideline_text = _tree_node_to_text(tree).strip()
            guideline_by_id[sample_id] = guideline_text

        problems_file = os.path.join(json_dir, f"{sample_id}_tree.json")
        try:
            with open(problems_file, "r") as f:
                problems = json.load(f)
        except Exception:
            with open(problems_file.replace(".json", ".fixed.json"), "r") as f:
                problems = json.load(f)

        all_final_outcomes = problems["all_final_outcomes"]
        worlds: List[World] = []
        for k, k_probs in problems["questions"].items():
            for prob in k_probs:
                assert prob["final_outcome"] in all_final_outcomes, (
                    f"Final outcome {prob['final_outcome']} not in {all_final_outcomes}"
                )
                worlds.append(World(
                    final_outcome=prob["final_outcome"],
                    k=int(k),
                    desired_questions=[q.strip(",!?") for q in prob["desired_question"]],
                    desired_questions_answers=[a.strip(",!?") for a in prob["desired_question_answer"]],
                    context=prob["context"],
                    patient_context=prob["patient_context"],
                ))

        samples.append(Sample(
            sample_id=sample_id,
            worlds=worlds,
            all_final_outcomes=all_final_outcomes,
            allowed_question_topics=allowed_question_topics,
            assessment=assessment,
            num_distractors_included=num_distractors_included,
            guideline_text=guideline_text,
        ))

    return samples, guideline_by_id


# ============ Action Parsing ============

def extract_non_thinking(model_name: str, response: str) -> str:
    """Strip thinking traces from the model response.

    Handles Qwen (</think>), gpt-oss (<|message|> blocks), and Mistral ([/THINK]).
    Other models are returned as-is.
    """
    if "qwen" in model_name.lower() and "</think>" in response:
        return response.split("</think>")[-1].strip()

    if "gpt-oss" in model_name.lower() and "<|message|>" in response:
        final_output = response.rsplit("<|message|>", 1)[-1]
        for end_tok in ("<|return|>", "<|call|>"):
            if end_tok in final_output:
                final_output = final_output.split(end_tok, 1)[0]
                break
        final_output = final_output.strip()
        if final_output.startswith("{"):
            try:
                parsed = json.loads(final_output)
                final_output = parsed.get("response", final_output)
            except Exception:
                pass
        return final_output

    if "mistral" in model_name.lower():
        response = response.split("[/THINK]", 1)[-1]

    return response.strip()


_TRAILING_PUNCT_RE = re.compile(r"""[\s\.\!\?,"'`]+$""")
_WHITESPACE_RE = re.compile(r"\s+")
_ANSWER_PREFIX_RE = re.compile(r"^answer\s*:\s*", re.IGNORECASE)
_QUOTE_RE = re.compile(r'^["\'`]+|["\'`]+$')
_SEPARATOR_RE = re.compile(r"[.?!,;]+$")


def _normalize_answer(raw: str) -> str:
    """Normalize an answer string for exact-match comparison."""
    s = raw.strip().lower()
    s = _ANSWER_PREFIX_RE.sub("", s)
    s = _QUOTE_RE.sub("", s)
    s = _SEPARATOR_RE.sub("", s)
    s = s.replace("-", " ").replace("_", " ")
    return _WHITESPACE_RE.sub(" ", s).strip()


def _normalize_question(raw: str) -> str:
    """Normalize a question string for matching."""
    s = raw.strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    return _WHITESPACE_RE.sub(" ", s).strip().lower()


def _extract_questions(clean_response: str) -> List[str]:
    """Extract question text from the model response.

    Handles:
      - "Question: What is the patient's age?"
      - "Question: Does he smoke? Question: Is he overweight?"
    """
    questions = []
    for mm in re.finditer(r"(?is)\bquestion\s*:\s*(.+?)\?", clean_response):
        val = mm.group(1).strip()
        if val:
            questions.append(val + "?")
    if questions:
        return questions

    # Fallback: missing '?' terminator
    m = re.search(r"(?is)\bquestion\s*:\s*(.+)$", clean_response)
    if m:
        lines = [line.strip() for line in m.group(1).strip().splitlines() if line.strip()]
        if lines:
            val = lines[0]
            if not val.endswith("?"):
                val += "?"
            return [val]

    return []


def parse_action(
    model_name: str,
    response: str,
    already_extracted: bool = False,
    is_single_turn: bool = False,
) -> Action:
    clean = response.strip() if already_extracted else extract_non_thinking(model_name, response).strip()

    ans = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", clean)
    if ans:
        return Action(type="ANSWER", value=ans.group(1).strip())

    raw_qs = _extract_questions(clean)
    qs = [_normalize_question(q) for q in raw_qs]

    if (is_single_turn and qs) or len(qs) == 1:
        return Action(type="QUESTION", questions=qs)

    return Action(type="RETRY")


def _compare_answer_exact(text: str, candidate: str) -> Optional[bool]:
    if text is None:
        return None
    return _normalize_answer(text) == _normalize_answer(candidate)


async def _check_answer_correct_llm(
    text: str,
    correct_ans: str,
    candidate_ans_list: List[str],
) -> bool:
    """Semantic answer-correctness check via ORACLE_MODEL."""
    candidates_text = "\n".join(f"- {ans}" for ans in candidate_ans_list)
    prompt = (
        f"You are a judge checking if a model's answer matches a correct answer from a list of options.\n\n"
        f"Correct answer: {correct_ans}\n\n"
        f"All possible answers:\n{candidates_text}\n\n"
        f"Model's answer: {text}\n\n"
        "Does the model's answer match the correct answer? The model's answer may be paraphrased or abbreviated.\n"
        'Answer "Yes" if the model\'s answer semantically matches the correct answer, "No" otherwise.\n'
        'Only output "Yes" or "No".'
    )
    result = await async_generate_single(
        model_name=ORACLE_MODEL,
        port="",
        messages=[{"role": "user", "content": prompt}],
        generation_config={"max_completion_tokens": 16},
    )
    return "yes" in result.text.strip().lower()


async def check_answer_correct(
    text: str,
    correct_ans: str,
    candidate_ans_list: List[str],
) -> bool:
    """Check answer correctness: exact match first, then LLM-based semantic match."""
    hits = [c for c in candidate_ans_list if _normalize_answer(text) == _normalize_answer(c)]
    assert len(hits) <= 1, f"Multiple candidates matched: {hits}"
    if hits:
        return hits[0] == correct_ans
    return await _check_answer_correct_llm(text, correct_ans, candidate_ans_list)


# ============ Episode Rollout ============

async def run_episode_async(
    model_name: str,
    port: str,
    sample: Sample,
    world: World,
    budget: str,
    cache: Optional[Dict],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool = False,
    max_retries: int = 3,
    verbose: bool = False,
    emphasize_uncertainty: bool = True,
    context_mode: str = "patient_question_list",
    with_examples: bool = False,
    with_distractor_warning: bool = False,
    max_guidelines: int = 1,
    guideline_by_id: Optional[Dict[str, str]] = None,
    seed: int = 42,
    show_past_qa_to_oracle: bool = False,
    oracle_mode: str = "standard",
) -> EpisodeResult:
    """Run one evaluation episode asynchronously with caching.

    Args:
        show_past_qa_to_oracle: Show prior (question, oracle_answer) pairs to the oracle as context.
        oracle_mode: standard | adversarial | cooperative oracle policy.
    """
    is_single_turn = False
    if budget == "k":
        budget = world.k
    else:
        budget = int(budget)
        if budget == 0:
            is_single_turn = True
            budget = 1

    cache_key = make_episode_cache_key(
        sample, world,
        context_mode=context_mode,
        max_guidelines=max_guidelines,
        oracle_mode=oracle_mode,
        budget=budget,
        emphasize_uncertainty=emphasize_uncertainty,
        with_examples=with_examples,
        with_distractor_warning=with_distractor_warning,
        seed=seed,
        show_past_qa_to_oracle=show_past_qa_to_oracle,
    )
    if cache is not None and cache_key in cache:
        if verbose:
            print(f"[CACHED] sample={sample.sample_id}, outcome={world.final_outcome}")
        return dict_to_episode_result(cache[cache_key])

    if verbose:
        print(f"\n--- Episode: sample={sample.sample_id}, outcome={world.final_outcome} ---")

    messages = build_initial_prompt(
        sample, world, budget,
        emphasize_uncertainty=emphasize_uncertainty,
        is_single_turn=is_single_turn,
        context_mode=context_mode,
        with_examples=with_examples,
        with_distractor_warning=with_distractor_warning,
        max_guidelines=max_guidelines,
        guideline_by_id=guideline_by_id,
        seed=seed,
    )

    result = EpisodeResult(sample_id=sample.sample_id, world=world)
    retry_count = 0
    turn = 0

    while turn <= budget:
        if verbose:
            print(f"  Turn {turn}/{budget}...")

        current_messages = copy.deepcopy(messages)
        if turn == budget:
            current_messages[-1]["content"] += FORCE_ANSWER_PROMPT

        result_obj = await async_generate_single(
            model_name=model_name,
            port=port,
            messages=current_messages,
            generation_config=generation_config,
        )
        response = result_obj.text
        cot = result_obj.cot
        thinking_tokens_count = result_obj.num_thinking_tokens

        action = parse_action(model_name, response, already_extracted=True, is_single_turn=is_single_turn)
        if verbose:
            print(f"  Action: {action.type}, questions={action.questions}, value={action.value}")

        if action.type == "ANSWER":
            result.final_answer = action.value
            result.answered = True
            result.correct = await check_answer_correct(
                action.value, world.final_outcome, sample.all_final_outcomes
            )
            result.final_response = response
            result.turns_used = turn
            result.final_cot = cot
            result.final_thinking_tokens_count = thinking_tokens_count
            if verbose:
                print(f"  => ANSWER: {action.value}, correct={result.correct}")
            break

        elif action.type == "QUESTION":
            if turn == budget:
                retry_count += 1
                if retry_count >= max_retries:
                    result.final_response = response
                    result.turns_used = turn
                    result.budget_violated = True
                    break
                continue

            retry_count = 0

            past_qa_pairs = None
            if show_past_qa_to_oracle and result.turn_logs:
                past_qa_pairs = [
                    (q, log.oracle_answer) for log in result.turn_logs for q in log.questions
                ]

            oracle_resp, matched_questions = await oracle_answer(
                world,
                action.questions,
                past_qa_pairs=past_qa_pairs,
                oracle_mode=oracle_mode,
                all_final_outcomes=sample.all_final_outcomes,
            )
            if verbose:
                print(f"  => Asked: {action.questions}, Oracle: {oracle_resp}")

            result.turn_logs.append(TurnLog(
                turn=turn,
                questions=action.questions,
                matched_questions=matched_questions,
                oracle_answer=oracle_resp,
                model_response=response,
                thinking_tokens_count=thinking_tokens_count,
                cot=cot,
            ))
            result.questions_asked.extend(action.questions)
            result.matched_questions_asked.extend(matched_questions)

            if keep_thinking_trace:
                if "qwen" in model_name.lower():
                    messages.append({"role": "assistant", "content": response, "reasoning_content": cot})
                    messages.append({"role": "tool", "content": oracle_resp})
                else:
                    raise NotImplementedError("keep_thinking_trace is only implemented for Qwen models.")
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": oracle_resp})

            turn += 1

        else:  # RETRY
            retry_count += 1
            if retry_count >= max_retries:
                result.final_response = response
                result.turns_used = turn
                result.budget_violated = True
                break

            if keep_thinking_trace:
                if "qwen" in model_name.lower():
                    messages.append({"role": "assistant", "content": response, "reasoning_content": cot})
                    messages.append({"role": "tool", "content": RETRY_PROMPT})
                else:
                    raise NotImplementedError("keep_thinking_trace is only implemented for Qwen models.")
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": RETRY_PROMPT})

    if not result.answered:
        result.turns_used = turn
        result.budget_violated = True

    if cache is not None and cache_file is not None:
        save_episode_to_cache(cache, cache_file, cache_key, result)

    return result


async def process_episodes_async(
    episodes: List[Tuple[Sample, World]],
    model_name: str,
    port: str,
    budget: str,
    cache: Optional[Dict],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool = False,
    verbose: bool = False,
    emphasize_uncertainty: bool = True,
    max_concurrent: int = 64,
    context_mode: str = "patient_question_list",
    with_examples: bool = False,
    with_distractor_warning: bool = False,
    max_guidelines: int = 1,
    guideline_by_id: Optional[Dict[str, str]] = None,
    seed: int = 42,
    show_past_qa_to_oracle: bool = False,
    oracle_mode: str = "standard",
) -> List[EpisodeResult]:
    """Process multiple episodes concurrently with semaphore-based concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(sample: Sample, world: World) -> EpisodeResult:
        async with semaphore:
            try:
                return await run_episode_async(
                    model_name=model_name,
                    port=port,
                    sample=sample,
                    world=world,
                    budget=budget,
                    cache=cache,
                    cache_file=cache_file,
                    generation_config=generation_config,
                    keep_thinking_trace=keep_thinking_trace,
                    verbose=verbose,
                    emphasize_uncertainty=emphasize_uncertainty,
                    context_mode=context_mode,
                    with_examples=with_examples,
                    with_distractor_warning=with_distractor_warning,
                    max_guidelines=max_guidelines,
                    guideline_by_id=guideline_by_id,
                    seed=seed,
                    show_past_qa_to_oracle=show_past_qa_to_oracle,
                    oracle_mode=oracle_mode,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[EPISODE_FAILED] sample={sample.sample_id}, world={world.context}: {err}")
                return EpisodeResult(
                    sample_id=sample.sample_id,
                    world=world,
                    budget_violated=True,
                    episode_error=err,
                    final_response=f"[EPISODE_FAILED] {err}",
                )

    tasks = [_run_one(sample, world) for sample, world in episodes]
    results = await async_tqdm.gather(*tasks, desc="Processing episodes", total=len(tasks))
    return list(results)


# ============ Metrics ============

def compute_minset_f1(asked: List[str], gt_set: List[str]) -> float:
    """F1 score of matched questions asked vs ground-truth minimal set."""
    asked_filtered = [q for q in asked if q is not None]
    if not asked_filtered or not gt_set:
        return 0.0
    asked_set = set(q.lower() for q in asked_filtered)
    gt_lower = set(q.lower() for q in gt_set)
    intersection = len(asked_set & gt_lower)
    if intersection == 0:
        return 0.0
    precision = intersection / len(asked_set)
    recall = intersection / len(gt_lower)
    return 2 * precision * recall / (precision + recall)


def compute_minset_jaccard(asked: List[str], gt_set: List[str]) -> float:
    """Jaccard similarity of asked questions vs ground-truth minimal set."""
    if not asked or not gt_set:
        return 0.0
    asked_set = set(q.lower() for q in asked)
    gt_lower = set(q.lower() for q in gt_set)
    intersection = len(asked_set & gt_lower)
    union = len(asked_set | gt_lower)
    return intersection / union if union > 0 else 0.0


def compute_turn_question_accuracy(
    questions_asked: List[str],
    matched_questions_asked: List[str],
) -> float:
    """Fraction of asked questions that matched a desired question (question-level precision)."""
    if not questions_asked:
        return 0.0
    matched_count = sum(1 for m in matched_questions_asked if m is not None)
    return matched_count / len(questions_asked)


def compute_metrics(results: List[EpisodeResult], samples: List[Sample]) -> Dict[str, Any]:
    """Compute aggregate and per-k metrics over a list of episode results."""
    if not results:
        return {}

    sample_map: Dict[str, Sample] = {s.sample_id: s for s in samples}

    def _core(res: List[EpisodeResult]) -> Dict[str, float]:
        if not res:
            return {k: 0 for k in (
                "total_episodes", "n_episode_errors", "micro_accuracy", "macro_accuracy",
                "answer_rate", "avg_questions_used", "avg_minset_f1", "avg_minset_jaccard",
                "avg_turn_question_accuracy", "avg_thinking_tokens",
                "avg_num_turns_used", "avg_overhead_questions",
            )}

        total = len(res)
        n_errors = sum(1 for r in res if r.episode_error)
        correct = sum(1 for r in res if r.correct)
        answered = sum(1 for r in res if r.answered)
        micro_acc = correct / total

        sample_accs: Dict[str, List[int]] = {}
        for r in res:
            sample_accs.setdefault(r.sample_id, []).append(1 if r.correct else 0)
        macro_acc = (
            sum(sum(xs) / len(xs) for xs in sample_accs.values()) / len(sample_accs)
            if sample_accs else 0.0
        )

        avg_questions = sum(len(r.questions_asked) for r in res) / total
        avg_thinking = (
            sum(
                sum(tl.thinking_tokens_count for tl in r.turn_logs) + r.final_thinking_tokens_count
                for r in res
            ) / total
        )
        avg_turns = sum(r.turns_used for r in res) / total
        avg_overhead = sum((len(r.questions_asked) - r.world.k) for r in res) / total

        f1s = [compute_minset_f1(r.matched_questions_asked, r.world.desired_questions) for r in res]
        jaccards = [compute_minset_jaccard(r.questions_asked, r.world.desired_questions) for r in res]
        turn_accs = [
            compute_turn_question_accuracy(r.questions_asked, r.matched_questions_asked) for r in res
        ]

        return {
            "total_episodes": total,
            "n_episode_errors": n_errors,
            "micro_accuracy": micro_acc,
            "macro_accuracy": macro_acc,
            "answer_rate": answered / total,
            "avg_questions_used": avg_questions,
            "avg_minset_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "avg_minset_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
            "avg_turn_question_accuracy": sum(turn_accs) / len(turn_accs) if turn_accs else 0.0,
            "avg_thinking_tokens": avg_thinking,
            "avg_num_turns_used": avg_turns,
            "avg_overhead_questions": avg_overhead,
        }

    overall = _core(results)

    per_k_results: Dict[int, List[EpisodeResult]] = {}
    for r in results:
        if sample_map.get(r.sample_id):
            per_k_results.setdefault(r.world.k, []).append(r)

    overall["per_k"] = {
        str(k): _core(res_k)
        for k, res_k in sorted(per_k_results.items())
    }
    return overall


# ============ Results serialization ============

def build_episode_data(r: EpisodeResult, world: World) -> Dict[str, Any]:
    return {
        "sample_id": r.sample_id,
        "k": world.k,
        "final_outcome": world.final_outcome,
        "context": world.context,
        "patient_context": world.patient_context,
        "desired_questions": world.desired_questions,
        "desired_questions_answers": world.desired_questions_answers,
        "final_answer": r.final_answer,
        "correct": r.correct,
        "answered": r.answered,
        "turns_used": r.turns_used,
        "questions_asked": r.questions_asked,
        "matched_questions_asked": r.matched_questions_asked,
        "turn_logs": [
            {
                "turn": t.turn,
                "questions": t.questions,
                "matched_questions": t.matched_questions,
                "oracle_answer": t.oracle_answer,
                "model_response": t.model_response,
                "cot": t.cot,
                "thinking_tokens_count": t.thinking_tokens_count,
            }
            for t in r.turn_logs
        ],
        "final_response": r.final_response,
        "final_cot": r.final_cot,
        "final_thinking_tokens_count": r.final_thinking_tokens_count,
        "budget_violated": r.budget_violated,
        "episode_error": r.episode_error,
    }


# ============ Main ============

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn ClinGuide evaluator")

    parser.add_argument(
        "--model-name", type=str,
        default="gpt-4o",
        help="Model name (e.g. gpt-4o, gemini-2.5-flash, Qwen/Qwen3-30B-A3B-Thinking-2507-FP8)",
        choices=[
            "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
            "Qwen/Qwen3-30B-A3B-Thinking-2507",
            "Qwen/Qwen3.5-27B-FP8",
            "Qwen/Qwen3.5-35B-A3B-FP8",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-122B-A10B-FP8",
            "Qwen/Qwen3-4B-Thinking-2507",
            "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
            "gpt-5",
            "gpt-5-mini",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "openai/gpt-oss-120B",
            "openai/gpt-oss-20B",
            "gpt-5.2",
            "gpt-5.4",
            "gpt-4o",
        ],
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for GPT-OSS models",
    )
    parser.add_argument("--port", type=str, default="8011", help="vLLM server port")
    parser.add_argument(
        "--json-dir", type=str,
        default="/path/to/data/dir",
        help="Directory containing *_tree.json, *_question_list_all.json, and optionally *_data.json files",
    )
    parser.add_argument("--cache-tag", type=str, default="", help="Tag to identify cache files")
    parser.add_argument(
        "--budget", type=str, default="4",
        choices=["0", "4", "k", "10"],
        help="Max questions allowed per episode (0 = single-turn bulk mode)",
    )
    parser.add_argument(
        "--keep-thinking-trace", action="store_true",
        help="Keep thinking trace in conversation history (local models only)",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="/path/to/results/dir",
        help="Directory to save results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose per-episode logging")
    parser.add_argument(
        "--no-emphasize-uncertainty",
        dest="emphasize_uncertainty", action="store_false", default=True,
        help="Disable prompt emphasis that tells model to ask questions first",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=64,
        help="Maximum concurrent episodes for async processing",
    )
    parser.add_argument(
        "--context-mode", type=str, default="patient_question_list",
        choices=[
            "patient_only",
            "patient_question_list",
            "patient_guideline",
            "patient_question_list_and_guideline",
        ],
        help=(
            "Prompt context mode: "
            "patient_only | patient_question_list | patient_guideline | patient_question_list_and_guideline"
        ),
    )
    parser.add_argument(
        "--max-guidelines", type=int, default=1,
        help="When guideline is shown: how many to include (1=current only; >1 adds distractors)",
    )
    parser.add_argument(
        "--with-distractor-warning", action="store_true",
        help="Add prompt note that some topics are distractors",
    )
    parser.add_argument(
        "--num-distractors", type=int, default=0,
        help="Number of distractor guideline sets to include in allowed topics (-1 = all)",
    )
    parser.add_argument("--with-examples", action="store_true", help="Include few-shot examples in the prompt")
    parser.add_argument(
        "--show-past-qa-to-oracle", action="store_true",
        help="Show prior (question, oracle_answer) pairs to the oracle for context",
    )
    parser.add_argument(
        "--oracle-mode", type=str, default="standard",
        choices=["standard", "adversarial", "cooperative"],
        help="Oracle policy: standard | adversarial (least disclosure) | cooperative (most helpful)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for world sampling")

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    cache_dir = os.path.join(args.results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    json_dir = args.json_dir
    cache_tag = args.cache_tag or os.path.basename(json_dir.rstrip("/"))
    model_name_safe = args.model_name.replace("/", "_")
    _ctx_with_guideline = args.context_mode in ("patient_guideline", "patient_question_list_and_guideline")
    output_name = (
        f"{model_name_safe}-{cache_tag}-budget{args.budget}-ctx{args.context_mode}"
        + (f"-mg{args.max_guidelines}" if _ctx_with_guideline and args.max_guidelines > 1 else "")
        + ("-distractor" if args.with_distractor_warning else "")
        + (f"-nd{args.num_distractors}" if args.num_distractors >= 0 else "")
        + ("-withexamples" if args.with_examples else "")
        + f"-seed{args.seed}"
    )
    if args.keep_thinking_trace:
        output_name += "-keepthink"
    if args.oracle_mode != "standard":
        output_name += f"-oracle-{args.oracle_mode}"

    episode_cache_file = os.path.join(cache_dir, f"{output_name}_episodes.jsonl")
    episode_cache = load_episode_cache(episode_cache_file) if os.path.exists(episode_cache_file) else {}

    generation_config = Evaluator(
        model_name=args.model_name,
        reasoning_effort=args.reasoning_effort,
    ).generation_config

    print(f"Loading data from {json_dir} (num_distractor_guidelines={args.num_distractors})")
    samples, guideline_by_id = load_data(json_dir, num_distractors=args.num_distractors)
    print(f"Loaded {len(samples)} samples")

    episodes = [(sample, world) for sample in samples for world in sample.worlds]
    print(f"Total episodes: {len(episodes)}")

    print(f"Running async with max_concurrent={args.max_concurrent}")
    all_results = asyncio.run(
        process_episodes_async(
            episodes=episodes,
            model_name=args.model_name,
            port=args.port,
            budget=args.budget,
            cache=episode_cache,
            cache_file=episode_cache_file,
            generation_config=generation_config,
            keep_thinking_trace=args.keep_thinking_trace,
            verbose=args.verbose,
            emphasize_uncertainty=args.emphasize_uncertainty,
            max_concurrent=args.max_concurrent,
            context_mode=args.context_mode,
            with_examples=args.with_examples,
            with_distractor_warning=args.with_distractor_warning,
            max_guidelines=args.max_guidelines,
            guideline_by_id=guideline_by_id,
            seed=args.seed,
            show_past_qa_to_oracle=args.show_past_qa_to_oracle,
            oracle_mode=args.oracle_mode,
        )
    )

    metrics = compute_metrics(all_results, samples)
    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    metrics_dir = os.path.join(args.results_dir, "results_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, f"{output_name}_results.json")
    with open(metrics_file, "w") as f:
        json.dump({"config": vars(args), "metrics": metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    sample_by_id = {s.sample_id: s for s in samples}
    results_data = {
        "config": vars(args),
        "metrics": metrics,
        "samples_meta": {
            s.sample_id: {
                "all_final_outcomes": s.all_final_outcomes,
                "n_worlds": len(s.worlds),
                "max_k": max(w.k for w in s.worlds) if s.worlds else 0,
            }
            for s in samples
        },
        "episodes": [
            build_episode_data(r, r.world)
            for r in all_results
            if sample_by_id.get(r.sample_id)
        ],
    }

    results_file = os.path.join(args.results_dir, f"{output_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
