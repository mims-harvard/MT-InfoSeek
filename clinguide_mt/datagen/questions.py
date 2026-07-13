"""Stage 2: Build multi-turn question datasets from decision tree trajectories."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from tqdm import tqdm

from .config import DATA_FOLDER, MAX_K
from .tree import extract_trajectories

log = logging.getLogger(__name__)


class Multiturn:
    """One question instance: k masked conditions, their answers, and surrounding context."""

    def __init__(
        self,
        final_outcome: str,
        final_outcome_type: str,
        desired_question: List[str],
        desired_question_answer: List[str],
        context: Dict[str, str],
        k: int,
    ) -> None:
        self.final_outcome = final_outcome
        self.final_outcome_type = final_outcome_type
        self.desired_question = desired_question
        self.desired_question_answer = desired_question_answer
        self.context = context
        self.k = k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_outcome": self.final_outcome,
            "final_outcome_type": self.final_outcome_type,
            "k": self.k,
            "desired_question": self.desired_question,
            "desired_question_answer": self.desired_question_answer,
            "context": self.context,
        }


class Questions:
    """Collection of Multiturn instances derived from one decision tree."""

    def __init__(self, tree: Dict[str, Any]) -> None:
        self.tree = tree
        self.all_final_outcomes: List[str] = []
        self.all_final_outcome_type: List[str] = []
        self.multiturn_by_k: Dict[int, List[Multiturn]] = {}

    def add_multiturn(self, m: Multiturn) -> None:
        self.multiturn_by_k.setdefault(m.k, []).append(m)
        if m.final_outcome not in self.all_final_outcomes:
            self.all_final_outcomes.append(m.final_outcome)
        if m.final_outcome_type not in self.all_final_outcome_type:
            self.all_final_outcome_type.append(m.final_outcome_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_final_outcomes": self.all_final_outcomes,
            "all_final_outcome_type": self.all_final_outcome_type,
            "questions": {
                k: [m.to_dict() for m in v]
                for k, v in self.multiturn_by_k.items()
            },
        }


def multiturn_from_trajectory(traj: Dict[str, Any], max_k: int = MAX_K) -> List[Multiturn]:
    """Generate up to max_k Multiturn instances by masking the last k conditions."""
    conditions = traj["conditions"]
    multiturns = []
    for k in range(1, min(max_k, len(conditions)) + 1):
        masked = conditions[-k:]
        ctx = conditions[:-k]
        multiturns.append(Multiturn(
            final_outcome=traj["final_outcome"],
            final_outcome_type=traj["final_outcome_type"],
            desired_question=[c["label"] for c in masked],
            desired_question_answer=[c["answer"] for c in masked],
            context={
                "assessment": traj["assessment"],
                **{c["label"]: c["answer"] for c in ctx},
            },
            k=k,
        ))
    return multiturns


def build_questions(tree: Dict[str, Any], max_k: int = MAX_K) -> Questions:
    """Extract all trajectories from a tree and build the full Questions dataset."""
    q = Questions(tree)
    for traj in extract_trajectories(tree):
        for m in multiturn_from_trajectory(traj, max_k):
            q.add_multiturn(m)
    return q


def run_json_to_questions(data_folder: str = DATA_FOLDER) -> None:
    """Stage 2: Read *data.json tree files, save *_tree.json question datasets."""
    copy_files = [f for f in os.listdir(data_folder) if f.endswith("_data.json")]
    log.info("Found %d *_data.json files in %s", len(copy_files), data_folder)
    for file_name in tqdm(copy_files, desc="json_to_questions"):
        with open(os.path.join(data_folder, file_name), "r") as f:
            data = json.load(f)
        dataset = build_questions(data).to_dict()
        out_name = file_name.removesuffix("_data.json") + "_tree.json"
        out_path = os.path.join(data_folder, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
    log.info("Done. Saved %d *_tree.json files", len(copy_files))
