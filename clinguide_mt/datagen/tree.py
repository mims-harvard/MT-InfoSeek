"""Tree data structures: anytree visualization, Pydantic schema, trajectory extraction."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Literal, Optional

from anytree import Node
from pydantic import BaseModel


# ── anytree visualization ─────────────────────────────────────────────────────

def build_anytree(data: Any, parent: Node | None = None) -> None:
    """Recursively convert a raw nested JSON decision tree into an anytree Node hierarchy."""
    if isinstance(data, dict):
        for key, value in data.items():
            build_anytree(value, parent=Node(key, parent=parent))
    elif isinstance(data, list):
        for item in data:
            Node(item, parent=parent)
    else:
        Node(data, parent=parent)


def load_anytree(json_path: str) -> Node:
    """Load a raw JSON decision tree and return the anytree root Node."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    root = Node("Treatment Tree")
    build_anytree(data, parent=root)
    return root


# ── Pydantic schema ───────────────────────────────────────────────────────────

NodeType = Literal[
    "assessment",
    "condition",
    "evaluation",
    "decision",
    "referral",
    "intervention",
]


class TreeNode(BaseModel):
    type: NodeType
    label: str
    branches: Optional[Dict[str, "TreeNode"]] = None
    next: Optional["TreeNode"] = None


TreeNode.model_rebuild()


def print_tree(node: TreeNode, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}- {node.label} [{node.type}]")
    if node.branches:
        for cond, child in node.branches.items():
            print(f"{prefix}  ? {cond}")
            print_tree(child, indent + 4)
    if node.next:
        print_tree(node.next, indent + 2)


def count_decisions(node: TreeNode) -> int:
    count = 1 if node.type == "decision" else 0
    if node.branches:
        for child in node.branches.values():
            count += count_decisions(child)
    if node.next:
        count += count_decisions(node.next)
    return count


# ── Trajectory extraction ─────────────────────────────────────────────────────

def extract_trajectories(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    DFS over a Pydantic-format tree dict, collecting every root→decision path
    as a trajectory with keys: assessment, conditions, final_outcome, final_outcome_type.

    Expects tree dicts with keys: type, label, branches (optional), next (optional).
    """
    trajectories: List[Dict[str, Any]] = []

    def dfs(
        node: Dict[str, Any],
        assessment_label: str | None,
        path: List[Dict[str, str]],
    ) -> None:
        node_type = node.get("type")

        if node_type == "decision":
            trajectories.append({
                "assessment": assessment_label,
                "conditions": copy.deepcopy(path),
                "final_outcome": node["label"],
                "final_outcome_type": "decision",
            })
            return

        if node_type == "condition":
            label = node["label"]
            if "branches" in node:
                for answer, next_node in node["branches"].items():
                    path.append({"label": label, "answer": answer})
                    dfs(next_node, assessment_label, path)
                    path.pop()
            elif "next" in node:
                dfs(node["next"], assessment_label, path)

        if node_type == "assessment":
            dfs(node["next"], node["label"], path)

    dfs(tree, None, [])
    return trajectories
