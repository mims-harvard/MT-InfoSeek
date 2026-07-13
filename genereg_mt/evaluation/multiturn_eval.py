import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure local imports work regardless of invocation cwd.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import model_registry

try:
    from model_utils import async_generate_single
    MODEL_UTILS_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    async_generate_single = None
    MODEL_UTILS_IMPORT_ERROR = exc
try:
    from evaluators.evaluator import Evaluator
    EVALUATOR_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    Evaluator = None
    EVALUATOR_IMPORT_ERROR = exc

# Import GRN utilities from repo local path.
GRN_UTILS_DIR = ROOT / "genereg_mt" / "data_generation"
if str(GRN_UTILS_DIR) not in sys.path:
    sys.path.append(str(GRN_UTILS_DIR))
import boolean_network_utils as bnu


TASK_NAME_MAP = {
    ("GeneReg-SS", "attractor_id"): "ss_id",
    ("GeneReg-Dyn", "attractor_id"): "dyn_attr",
    ("GeneReg-Dyn", "marker_gene"): "dyn_marker",
}

RANDOM_POLICY_MODEL_NAMES = {"random", "random-baseline"}

DEFAULT_GRN_MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    "/path/to/DesignPrinciplesGeneNetworks/update_rules_122_models_Kadelka_SciAdv",
)


@dataclass
class BranchWorld:
    y: int
    state: int


@dataclass
class Sample:
    sample_id: int
    group_id: str
    task_name: str
    family: str
    target_type: str
    model: str
    n_nodes: int
    var_names: List[str]
    marker_gene: Optional[str]
    marker_gene_idx: Optional[int]
    observed: List[Tuple[str, int]]
    observed_idx_vals: List[Tuple[int, int]]
    k: int
    minimal_sets: List[List[str]]
    minimal_sets_idx: List[List[int]]
    metadata: Dict[str, Any]
    branches: List[BranchWorld]

    # Runtime oracle tensors (aligned arrays)
    base_states: np.ndarray
    base_outcomes: np.ndarray
    queryable_gene_indices: List[int]
    canonical_minimal_set: List[str]
    forbid_alternative_genes: List[str]
    forbid_alternative_gene_indices: List[int]
    raw_rules_text: str
    prompt_gene_index_text: str
    prompt_fixed_points_catalog: str
    prompt_dyn_attractors_catalog: str


@dataclass
class TurnLog:
    turn: int
    questions: List[str]
    oracle_answer: str
    is_ambiguous: bool
    target_known_after_turn: bool
    covers_any_gt_sufficient_set: bool
    model_response: str
    thinking_tokens_count: int
    cot: str = ""
    prompt_messages: Optional[List[Dict[str, str]]] = None


@dataclass
class EpisodeResult:
    sample_id: int
    group_id: str
    task_name: str
    world_y: int
    world_state: int
    final_answer: Optional[str] = None
    parsed_final_answer: Optional[int] = None
    correct: bool = False
    answered: bool = False
    turns_used: int = 0
    questions_asked: List[str] = field(default_factory=list)
    turn_logs: List[TurnLog] = field(default_factory=list)
    final_cot: str = ""
    final_thinking_tokens_count: int = 0
    final_response: str = ""
    budget_violated: bool = False
    final_prompt_messages: Optional[List[Dict[str, str]]] = None


@dataclass
class Action:
    type: str  # QUESTION / ANSWER / RETRY
    questions: List[str] = field(default_factory=list)
    value: Optional[str] = None


SYSTEM_PROMPT_SS_ID = """You are reasoning about a Boolean gene regulatory network in steady-state regime. You will be given a partial observation of the true steady-state vector, and your task is to identify which attractor ID (steady-state index) is the true one the network is in.

Instructions:
- Genes are binary (0/1).
- You may ask for values of additional genes in the unknown steady-state vector.
- Respond at each turn with exactly one of:
  1) Question: [GENE_NAME]
  2) Answer: [ATTRACTOR_ID_INTEGER]

Rules:
1. If current information is sufficient, output "Answer: [ATTRACTOR_ID_INTEGER]".
2. Otherwise ask for one gene (or up to the configured max per turn) in format "Question: [GENE_NAME]".
3. Do not output any extra text."""


SYSTEM_PROMPT_DYN_ATTR = """You are reasoning about a Boolean gene regulatory network under synchronous updates. You will be given a partial observation of the initial state vector at time t=0, and your task is to identify which attractor ID (fixed point or cycle) the system will reach under synchronous updates starting from that initial state.

Instructions:
- Genes are binary (0/1).
- The update semantics are synchronous (all genes update at the same discrete time step).
- Any gene values you ask about are values in the initial state (time t=0).
- The final answer refers to the reached attractor (fixed point or cycle).
- Respond at each turn with exactly one of:
  1) Question: [GENE_NAME]
  2) Answer: [ATTRACTOR_ID_INTEGER]

Rules:
1. If current information is sufficient, output "Answer: [ATTRACTOR_ID_INTEGER]".
2. Otherwise ask for one gene (or up to the configured max per turn) in format "Question: [GENE_NAME]".
3. Do not output any extra text."""


SYSTEM_PROMPT_DYN_MARKER = """You are reasoning about a Boolean gene regulatory network under synchronous updates. You will be given a partial observation of the initial state vector at time t=0, and your task is to identify the value of marker gene {marker_gene} in the reached attractor (steady behavior under synchronous updates).

Instructions:
- Genes are binary (0/1).
- The update semantics are synchronous (all genes update at the same discrete time step).
- Any gene values you ask about are values in the initial state (time t=0).
- Respond at each turn with exactly one of:
  1) Question: [GENE_NAME]
  2) Answer: 0   or   Answer: 1

Rules:
1. If current information is sufficient, output "Answer: 0" or "Answer: 1".
2. Otherwise ask for one gene (or up to the configured max per turn) in format "Question: [GENE_NAME]".
3. Do not output any extra text."""


UNCERTAINTY_EMPHASIS = """

IMPORTANT: The current observations are insufficient. Ask questions before answering when uncertain."""


FORCE_ANSWER_PROMPT_ATTR = "\n\nYou have used all turns. You must output: Answer: [ATTRACTOR_ID_INTEGER]"
FORCE_ANSWER_PROMPT_MARKER = "\n\nYou have used all turns. You must output: Answer: 0 or Answer: 1"


def extract_non_thinking(model_name: str, response: str) -> str:
    if "qwen" in model_name.lower() and "</think>" in response:
        return response.split("</think>")[-1].strip()

    if "gpt-oss" in model_name.lower():
        marker = "<|start|>assistant<|channel|>final<|message|>"
        end = "<|return|>"
        if marker in response:
            out = response.split(marker, 1)[-1]
            if end in out:
                out = out.split(end, 1)[0]
            return out.strip()

    if "mistral" in model_name.lower() and "[/THINK]" in response:
        return response.split("[/THINK]", 1)[-1].strip()

    return response.strip()


def is_random_policy_model(model_name: str) -> bool:
    return model_name.strip().lower() in RANDOM_POLICY_MODEL_NAMES


def _norm_gene_token(token: str) -> str:
    s = token.strip()
    s = re.sub(r"^[\s\[\]`'\".:;,-]+|[\s\[\]`'\".:;,-]+$", "", s)
    s = re.sub(r"^what is( the value of)?\s+", "", s, flags=re.I)
    s = re.sub(r"^value of\s+", "", s, flags=re.I)
    s = re.sub(r"^gene\s+", "", s, flags=re.I)
    s = re.sub(r"^is\s+", "", s, flags=re.I)
    s = s.strip()
    return s


def parse_action(
    model_name: str,
    response: str,
    valid_genes: List[str],
    max_num_q_per_turn: int,
    already_extracted: bool = False,
) -> Action:
    clean = response.strip() if already_extracted else extract_non_thinking(model_name, response).strip()

    ans = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", clean)
    if ans:
        return Action(type="ANSWER", value=ans.group(1).strip())

    q = re.search(r"(?is)\bquestion\s*:\s*(.+)$", clean)
    if not q:
        return Action(type="RETRY")

    payload = q.group(1).strip()
    gene_map = {g.lower(): g for g in valid_genes}
    extracted: List[str] = []

    # 1) Bracket format: [GENE]
    for m in re.finditer(r"\[([^\]]+)\]", payload):
        tok = _norm_gene_token(m.group(1)).lower()
        if tok in gene_map:
            extracted.append(gene_map[tok])

    # 2) Fallback tokenization if no bracketed gene found
    if not extracted:
        chunks = re.split(r"\?|,|;|\band\b", payload, flags=re.I)
        for ch in chunks:
            tok = _norm_gene_token(ch).lower()
            if tok in gene_map:
                extracted.append(gene_map[tok])

    # De-duplicate preserving order
    seen = set()
    deduped: List[str] = []
    for g in extracted:
        if g not in seen:
            seen.add(g)
            deduped.append(g)

    if 1 <= len(deduped) <= max_num_q_per_turn:
        return Action(type="QUESTION", questions=deduped)

    return Action(type="RETRY")


def parse_answer_value(sample: Sample, answer_text: str) -> Optional[int]:
    if answer_text is None:
        return None
    s = answer_text.strip().lower()
    s = re.sub(r"^answer\s*:\s*", "", s)

    if sample.target_type == "attractor_id":
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else None

    # marker_gene target
    if re.search(r"\b1\b", s):
        return 1
    if re.search(r"\b0\b", s):
        return 0
    if any(w in s for w in ["true", "yes", "on", "active", "high"]):
        return 1
    if any(w in s for w in ["false", "no", "off", "inactive", "low"]):
        return 0
    return None


def check_answer_correct(sample: Sample, parsed_value: Optional[int], world: BranchWorld) -> bool:
    if parsed_value is None:
        return False
    return int(parsed_value) == int(world.y)


def format_answer_text(sample: Sample, value: int) -> str:
    return f"Answer: {int(value)}"


def format_question_response(questions: List[str]) -> str:
    return "Question: " + ", ".join(questions)


def task_name_from_row(row: Dict[str, Any]) -> Optional[str]:
    return TASK_NAME_MAP.get((row.get("family"), row.get("target_type")))


def _observed_to_idx_vals(observed: List[List[Any]], var_names: List[str]) -> List[Tuple[int, int]]:
    name_to_idx = {v: i for i, v in enumerate(var_names)}
    out: List[Tuple[int, int]] = []
    for g, v in observed:
        out.append((int(name_to_idx[g]), int(v)))
    out.sort()
    return out


def _effective_queryable_indices(sample: Sample, forbid_alternatives: bool) -> List[int]:
    banned = _effective_banned_indices(sample, forbid_alternatives=forbid_alternatives)
    return [i for i in sample.queryable_gene_indices if i not in banned]


def _effective_banned_indices(sample: Sample, forbid_alternatives: bool) -> List[int]:
    banned: set = set(int(i) for i, _ in sample.observed_idx_vals)
    if forbid_alternatives:
        banned.update(int(i) for i in sample.forbid_alternative_gene_indices)
    return sorted(banned)


def _state_bitstring(state: int, n: int) -> str:
    # Bit order is index order 0..n-1 to match var_names indexing.
    return "".join(str((int(state) >> i) & 1) for i in range(int(n)))


def _build_gene_index_text(var_names: List[str]) -> str:
    if not var_names:
        return "(none)"
    return ", ".join([f"{i}:{g}" for i, g in enumerate(var_names)])


def _build_fixed_points_catalog(cache: bnu.ModelCache, candidate_ids: Optional[List[int]] = None) -> str:
    if not cache.fixed_points:
        return "- (none)"
    if candidate_ids is None:
        ids = list(range(len(cache.fixed_points)))
    else:
        ids = sorted({int(i) for i in candidate_ids if 0 <= int(i) < len(cache.fixed_points)})
        if not ids:
            return "- (none)"
    lines: List[str] = []
    for aid in ids:
        st = cache.fixed_points[aid]
        lines.append(f"- ID {aid}: bits={_state_bitstring(int(st), cache.n_nodes)}")
    return "\n".join(lines)


def _build_dyn_attractors_catalog(cache: bnu.ModelCache, candidate_ids: Optional[List[int]] = None) -> str:
    if cache.landscape is None or not cache.landscape.attractors:
        return "- (unavailable)"
    n_attr = len(cache.landscape.attractors)
    if candidate_ids is None:
        ids = list(range(n_attr))
    else:
        ids = sorted({int(i) for i in candidate_ids if 0 <= int(i) < n_attr})
        if not ids:
            return "- (none)"
    lines: List[str] = []
    for aid in ids:
        att = cache.landscape.attractors[aid]
        if int(att.period) <= 1:
            st = int(att.states[0]) if att.states else 0
            lines.append(f"- ID {aid}: FixedPoint; state={_state_bitstring(st, cache.n_nodes)}")
            continue
        states_txt = " | ".join([_state_bitstring(int(s), cache.n_nodes) for s in att.states])
        lines.append(f"- ID {aid}: Cycle; states={states_txt}")
    return "\n".join(lines)


def _build_prompt(
    sample: Sample,
    max_turns: int,
    max_num_q_per_turn: int,
    emphasize_uncertainty: bool,
    forbid_alternatives: bool,
    include_budget_in_prompt: bool,
) -> List[Dict[str, str]]:
    if sample.task_name == "ss_id":
        system = SYSTEM_PROMPT_SS_ID
    elif sample.task_name == "dyn_attr":
        system = SYSTEM_PROMPT_DYN_ATTR
    else:
        system = SYSTEM_PROMPT_DYN_MARKER.format(marker_gene=sample.marker_gene)

    if emphasize_uncertainty:
        system += UNCERTAINTY_EMPHASIS

    obs_txt = "\n".join([f"- {g} = {int(v)}" for g, v in sample.observed])
    if not obs_txt:
        obs_txt = "- (none)"

    banned_names = [sample.var_names[i] for i in _effective_banned_indices(sample, forbid_alternatives)]
    banned_txt = ", ".join(banned_names) if banned_names else "(none)"

    rules_txt = (sample.raw_rules_text or "").strip()
    if not rules_txt:
        raise RuntimeError(f"Missing raw_rules text for model {sample.model}, required for dynamic tasks.")

    if sample.task_name == "ss_id":
        target_desc = "Target: determine the attractor ID of the true steady-state."
        task_context = (
            "Steady-state attractor catalog (ID -> full binary state):\n"
            f"{sample.prompt_fixed_points_catalog}\n"
        )
    elif sample.task_name == "dyn_attr":
        target_desc = "Target: determine the attractor ID reached from the true initial state."
        task_context = (
            "Synchronous Boolean update rules (from model TXT):\n"
            f"{rules_txt}\n\n"
            "Attractor catalog (fixed points and cycles) for this model:\n"
            f"{sample.prompt_dyn_attractors_catalog}\n\n"
            "Catalog format:\n"
            "- FixedPoint: a single bitstring state.\n"
            "- Cycle: multiple bitstrings joined by \" | \" (one state after another in the cycle).\n"
        )
    else:
        target_desc = f"Target: determine converged marker value of {sample.marker_gene} (0/1)."
        task_context = (
            "Synchronous Boolean update rules (from model TXT):\n"
            f"{rules_txt}\n\n"
            "Attractor catalog (fixed points and cycles) for this model:\n"
            f"{sample.prompt_dyn_attractors_catalog}\n\n"
            "Catalog format:\n"
            "- FixedPoint: a single bitstring state.\n"
            "- Cycle: multiple bitstrings joined by \" | \" (one state after another in the cycle).\n"
        )

    budget_block = ""
    if include_budget_in_prompt:
        budget_block = (
            f"You can take up to {max_turns} turns\n"
            f"You may ask about up to {max_num_q_per_turn} genes per turn\n\n"
        )

    user = (
        # f"Number of genes: {sample.n_nodes}\n"
        f"Gene index order for all bitstrings below (left->right): {sample.prompt_gene_index_text}\n"
        f"{target_desc}\n"
        f"{budget_block}"
        f"{task_context}\n"
        f"Currently observed gene values:\n{obs_txt}\n\n"
        f"You may NOT ask about these genes:\n{banned_txt}\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def resolve_episode_limits(
    sample: Sample,
    budget: str,
    max_num_q_per_turn: int,
    forbid_alternatives: bool,
) -> Tuple[int, int]:
    """Resolve turn budget and per-turn question cap for one episode.

    Legacy compatibility with LogicQ-style scripts:
    `budget=0` means one question turn with as many queryable genes as allowed,
    followed by a forced-answer turn.
    """
    if budget == "k":
        max_turns = int(sample.k)
    else:
        max_turns = int(budget)

    effective_max_num_q_per_turn = int(max_num_q_per_turn)

    if max_turns == 0:
        effective_queryable = _effective_queryable_indices(sample, forbid_alternatives)
        effective_max_num_q_per_turn = max(1, len(effective_queryable))
        max_turns = 1

    return max_turns, effective_max_num_q_per_turn


def _build_base_tensors_for_sample(
    row: Dict[str, Any],
    cache: bnu.ModelCache,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Tuple[int, int]]]:
    var_names = row["var_names"]
    observed_idx_vals = _observed_to_idx_vals(row["observed"], var_names)
    observed_idx = {i for i, _ in observed_idx_vals}

    if row["family"] == "GeneReg-SS":
        omega = np.array(cache.fixed_points, dtype=np.uint64)
        mask, value = bnu.build_mask_value_from_idx_vals(observed_idx_vals)
        m = np.uint64(mask)
        p = np.uint64(value)
        keep = (omega & m) == p
        cand_idx = np.nonzero(keep)[0].astype(np.int64)
        cand_states = omega[cand_idx]
        # For ss_id target, y is global fixed-point index.
        outcomes = cand_idx.astype(np.int64)
        queryable = [i for i in range(len(var_names)) if i not in observed_idx]
        return cand_states, outcomes, queryable, observed_idx_vals

    # Dynamic regime
    if cache.landscape is None or cache.landscape.basin_map is None:
        raise RuntimeError(f"Model {cache.model} has no basin_map for dynamic task")

    n = int(cache.n_nodes)
    basin = cache.landscape.basin_map
    mask, value = bnu.build_mask_value_from_idx_vals(observed_idx_vals)
    base_states = np.array(list(bnu.iter_states_consistent_with_mask_value(n, mask, value)), dtype=np.int64)
    if base_states.size == 0:
        raise RuntimeError(f"No dynamic candidate states for sample {row.get('group_id')}")

    attr_ids = basin[base_states].astype(np.int64)

    if row["target_type"] == "attractor_id":
        outcomes = attr_ids
    else:
        marker_idx = int(row["marker_gene_idx"])
        # Safety guard: marker value must be well-defined within each attractor cycle.
        for aid, att in enumerate(cache.landscape.attractors):
            vals = [bnu.bit(st, marker_idx) for st in att.states]
            if vals and min(vals) != max(vals):
                raise RuntimeError(
                    f"Invalid dyn_marker target for model={cache.model}: "
                    f"marker gene index {marker_idx} is not constant in attractor {aid}."
                )
        rep_states = bnu.attractor_representative_states(cache.landscape)
        outcomes = ((rep_states[attr_ids] >> np.uint64(marker_idx)) & np.uint64(1)).astype(np.int64)

    queryable = [i for i in range(len(var_names)) if i not in observed_idx]
    return base_states, outcomes, queryable, observed_idx_vals


def _row_minimal_sets(row: Dict[str, Any]) -> List[List[Any]]:
    xs = row.get("minimal_sufficient_sets")
    if xs is None:
        xs = row.get("minimal_sets", [])
    return list(xs or [])


def _row_minimal_sets_idx(row: Dict[str, Any]) -> List[List[Any]]:
    xs = row.get("minimal_sufficient_sets_idx")
    if xs is None:
        xs = row.get("minimal_sets_idx", [])
    return list(xs or [])


def _validate_row_catalog_metadata(row: Dict[str, Any], cache: bnu.ModelCache) -> None:
    if row.get("prompt_catalog_size") is not None:
        expected = int(row["prompt_catalog_size"])
        if row["family"] == "GeneReg-SS":
            actual = int(len(cache.fixed_points))
        else:
            if cache.landscape is None:
                raise RuntimeError(f"Missing landscape for model {cache.model}")
            actual = int(len(cache.landscape.attractors))
        if actual != expected:
            raise RuntimeError(
                f"Catalog-size mismatch for group_id={row.get('group_id')}: row={expected}, cache={actual}"
            )

    if row.get("prompt_catalog_is_capped") is not None and row["family"] == "GeneReg-SS":
        expected_capped = bool(row["prompt_catalog_is_capped"])
        actual_capped = bool(cache.fixed_points_is_capped)
        if actual_capped != expected_capped:
            raise RuntimeError(
                f"Catalog-capped mismatch for group_id={row.get('group_id')}: "
                f"row={expected_capped}, cache={actual_capped}"
            )


def load_data(
    tasks_file: str,
    cache_dir: str,
    models_dir: str,
    include_tasks: List[str],
    seed: int,
) -> List[Sample]:
    rng = random.Random(seed)

    wanted = set(include_tasks)
    rows: List[Dict[str, Any]] = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            task_name = task_name_from_row(row)
            if task_name is None:
                continue
            if task_name not in wanted:
                continue
            rows.append(row)

    # Load model TXT rules.
    models_with_text = bnu.load_models_with_text(models_dir)
    if not models_with_text:
        raise RuntimeError(f"No models found under --models-dir: {models_dir}")
    raw_rules_by_model: Dict[str, str] = {
        str(m["model"]): str(m.get("raw_rules") or "")
        for m in models_with_text
    }

    # Preload needed model caches.
    models = sorted({r["model"] for r in rows})
    model_cache: Dict[str, bnu.ModelCache] = {}
    for m in models:
        safe = m.replace("/", "_")
        p = os.path.join(cache_dir, f"{safe}.model_cache.pkl")
        model_cache[m] = bnu.load_model_cache(p)

    missing_rules = [m for m in models if not (raw_rules_by_model.get(m) or "").strip()]
    if missing_rules:
        print(f"[warning] Missing raw TXT rules for {len(missing_rules)} models in this split.")

    samples: List[Sample] = []
    sid = 0
    for row in rows:
        cache = model_cache[row["model"]]
        _validate_row_catalog_metadata(row, cache)
        try:
            base_states, base_outcomes, queryable, observed_idx_vals = _build_base_tensors_for_sample(row, cache)
        except Exception as e:
            print(f"[warning] Skipping row group_id={row.get('group_id')} due to preprocessing error: {e}")
            continue
        minimal_sets = [list(ms) for ms in _row_minimal_sets(row)]
        if minimal_sets:
            canonical_minset = sorted(
                minimal_sets,
                key=lambda ms: (len(ms), [g.lower() for g in ms], ms),
            )[0]
        else:
            canonical_minset = []
        alt_forbid = sorted(set(sum(minimal_sets, [])) - set(canonical_minset))
        name_to_idx = {v: i for i, v in enumerate(row["var_names"])}
        alt_forbid_idx = sorted([int(name_to_idx[g]) for g in alt_forbid if g in name_to_idx])

        branches = [BranchWorld(y=int(b["y"]), state=int(b["state"])) for b in row.get("branches", [])]
        rng.shuffle(branches)

        s = Sample(
            sample_id=sid,
            group_id=row["group_id"],
            task_name=task_name_from_row(row),
            family=row["family"],
            target_type=row["target_type"],
            model=row["model"],
            n_nodes=int(row["n_nodes"]),
            var_names=list(row["var_names"]),
            marker_gene=row.get("marker_gene"),
            marker_gene_idx=row.get("marker_gene_idx"),
            observed=[(str(g), int(v)) for g, v in row.get("observed", [])],
            observed_idx_vals=observed_idx_vals,
            k=int(row["k_min"]),
            minimal_sets=minimal_sets,
            minimal_sets_idx=[[int(i) for i in ms] for ms in _row_minimal_sets_idx(row)],
            metadata=dict(row.get("metadata", {})),
            branches=branches,
            base_states=base_states,
            base_outcomes=base_outcomes,
            queryable_gene_indices=queryable,
            canonical_minimal_set=list(canonical_minset),
            forbid_alternative_genes=alt_forbid,
            forbid_alternative_gene_indices=alt_forbid_idx,
            raw_rules_text=raw_rules_by_model.get(str(row["model"]), ""),
            prompt_gene_index_text=_build_gene_index_text(list(row["var_names"])),
            prompt_fixed_points_catalog=_build_fixed_points_catalog(cache),
            prompt_dyn_attractors_catalog=_build_dyn_attractors_catalog(cache),
        )
        samples.append(s)
        sid += 1

    return samples


def make_episode_cache_key(
    sample: Sample,
    world: BranchWorld,
    oracle_type: str,
    is_flipped: bool,
    history_append_mode: str = "full",
) -> str:
    del history_append_mode
    d = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "task_name": sample.task_name,
        "world_y": int(world.y),
        "world_state": int(world.state),
        "oracle_type": oracle_type,
        "is_flipped": bool(is_flipped),
    }
    return json.dumps(d, sort_keys=True)


def _canonicalize_loaded_cache_key(key: str) -> Tuple[str, bool]:
    try:
        key_dict = json.loads(key)
    except Exception:
        return key, False
    removed_history_append_mode = "history_append_mode" in key_dict
    if removed_history_append_mode:
        key_dict.pop("history_append_mode", None)
    return json.dumps(key_dict, sort_keys=True), removed_history_append_mode


def _rewrite_episode_cache_file(cache_file: str, cache: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    tmp_cache_file = f"{cache_file}.tmp"
    with open(tmp_cache_file, "w", encoding="utf-8") as f:
        for key in sorted(cache):
            f.write(json.dumps({"key": key, "result": cache[key]}, ensure_ascii=False) + "\n")
    os.replace(tmp_cache_file, cache_file)


def _results_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = vars(args).copy()
    config.pop("history_append_mode", None)
    return config


def episode_result_to_dict(r: EpisodeResult) -> Dict[str, Any]:
    return {
        "sample_id": r.sample_id,
        "group_id": r.group_id,
        "task_name": r.task_name,
        "world": {"y": r.world_y, "state": r.world_state},
        "final_answer": r.final_answer,
        "parsed_final_answer": r.parsed_final_answer,
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
                "target_known_after_turn": t.target_known_after_turn,
                "covers_any_gt_sufficient_set": t.covers_any_gt_sufficient_set,
                "model_response": t.model_response,
                "cot": t.cot,
                "thinking_tokens_count": t.thinking_tokens_count,
                "prompt_messages": t.prompt_messages,
            }
            for t in r.turn_logs
        ],
        "final_cot": r.final_cot,
        "final_thinking_tokens_count": r.final_thinking_tokens_count,
        "final_response": r.final_response,
        "budget_violated": r.budget_violated,
        "final_prompt_messages": r.final_prompt_messages,
    }


def dict_to_episode_result(d: Dict[str, Any]) -> EpisodeResult:
    logs = [
        TurnLog(
            turn=int(t["turn"]),
            questions=list(t["questions"]),
            oracle_answer=t["oracle_answer"],
            is_ambiguous=bool(t["is_ambiguous"]),
            target_known_after_turn=bool(t.get("target_known_after_turn", False)),
            covers_any_gt_sufficient_set=bool(t.get("covers_any_gt_sufficient_set", False)),
            model_response=t["model_response"],
            cot=t.get("cot", ""),
            thinking_tokens_count=int(t.get("thinking_tokens_count", 0)),
            prompt_messages=t.get("prompt_messages"),
        )
        for t in d.get("turn_logs", [])
    ]

    return EpisodeResult(
        sample_id=int(d["sample_id"]),
        group_id=d["group_id"],
        task_name=d["task_name"],
        world_y=int(d["world"]["y"]),
        world_state=int(d["world"]["state"]),
        final_answer=d.get("final_answer"),
        parsed_final_answer=d.get("parsed_final_answer"),
        correct=bool(d.get("correct", False)),
        answered=bool(d.get("answered", False)),
        turns_used=int(d.get("turns_used", 0)),
        questions_asked=list(d.get("questions_asked", [])),
        turn_logs=logs,
        final_cot=d.get("final_cot", ""),
        final_thinking_tokens_count=int(d.get("final_thinking_tokens_count", 0)),
        final_response=d.get("final_response", ""),
        budget_violated=bool(d.get("budget_violated", False)),
        final_prompt_messages=d.get("final_prompt_messages"),
    )


def load_episode_cache(cache_file: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(cache_file):
        return cache
    stripped_history_key_count = 0
    duplicate_entry_count = 0
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
                key = x.get("key")
                val = x.get("result")
                if key and val:
                    key, stripped_history = _canonicalize_loaded_cache_key(key)
                    if stripped_history:
                        stripped_history_key_count += 1
                    if key in cache:
                        duplicate_entry_count += 1
                    cache[key] = val
            except Exception:
                continue
    if stripped_history_key_count or duplicate_entry_count:
        print(
            f"[episode_cache] canonicalized {stripped_history_key_count} cache keys by "
            f"dropping history_append_mode and collapsed {duplicate_entry_count} "
            f"duplicate entries"
        )
        _rewrite_episode_cache_file(cache_file, cache)
        print(f"[episode_cache] rewrote {cache_file} with {len(cache)} unique entries")
    return cache


def init_episode_cache(cache_file: str, fresh_episode_cache: bool = False) -> Dict[str, Dict[str, Any]]:
    if fresh_episode_cache:
        if os.path.exists(cache_file):
            print(f"[episode_cache] fresh start requested; removing existing file: {cache_file}")
            os.remove(cache_file)
        return {}
    return load_episode_cache(cache_file)


def save_episode_to_cache(cache: Dict[str, Dict[str, Any]], cache_file: str, key: str, result: EpisodeResult) -> None:
    d = episode_result_to_dict(result)
    cache[key] = d
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "result": d}, ensure_ascii=False) + "\n")


def _mask_value_from_context(ctx: Dict[int, int]) -> Tuple[np.uint64, np.uint64]:
    mask = 0
    value = 0
    for i, v in ctx.items():
        ii = int(i)
        mask |= (1 << ii)
        if int(v) == 1:
            value |= (1 << ii)
    return np.uint64(mask), np.uint64(value)


def _candidate_outcomes(sample: Sample, ctx: Dict[int, int]) -> np.ndarray:
    m, v = _mask_value_from_context(ctx)
    keep = (sample.base_states.astype(np.uint64) & m) == v
    return sample.base_outcomes[keep]


def _candidate_states(sample: Sample, ctx: Dict[int, int]) -> np.ndarray:
    m, v = _mask_value_from_context(ctx)
    keep = (sample.base_states.astype(np.uint64) & m) == v
    return sample.base_states[keep]


def _bit(state: int, idx: int) -> int:
    return (int(state) >> int(idx)) & 1


def build_random_policy_rng(sample: Sample, world: BranchWorld, is_flipped: bool, seed: int) -> random.Random:
    seed_material = json.dumps(
        {
            "seed": seed,
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "task_name": sample.task_name,
            "world_y": int(world.y),
            "world_state": int(world.state),
            "is_flipped": bool(is_flipped),
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
    asked = set(q.lower() for q in questions_asked)
    allowed = set(_effective_queryable_indices(sample, forbid_alternatives=forbid_alternatives))
    out: List[str] = []
    for gi in sample.queryable_gene_indices:
        if int(gi) not in allowed:
            continue
        gene = sample.var_names[int(gi)]
        if gene.lower() in asked:
            continue
        out.append(gene)
    return out


def choose_random_policy_action(
    rng: random.Random,
    sample: Sample,
    asked_ctx: Dict[int, int],
    questions_asked: List[str],
    max_num_q_per_turn: int,
    force_answer_prompt_applied: bool,
    forbid_alternatives: bool,
) -> Action:
    post_outcomes = _candidate_outcomes(sample, asked_ctx)
    unique_outcomes = np.unique(post_outcomes)
    if unique_outcomes.size == 1:
        return Action(type="ANSWER", value=str(int(unique_outcomes[0])))

    available_questions = get_available_random_policy_questions(
        sample,
        questions_asked=questions_asked,
        forbid_alternatives=forbid_alternatives,
    )
    if not available_questions or force_answer_prompt_applied:
        if sample.target_type == "attractor_id":
            catalog_size = int(sample.metadata.get("prompt_catalog_size", 0) or 0)
            if catalog_size > 0:
                guess = int(rng.randrange(catalog_size))
            elif unique_outcomes.size > 0:
                guess = int(rng.choice(unique_outcomes.tolist()))
            else:
                guess = 0
        else:
            guess = int(rng.choice([0, 1]))
        return Action(type="ANSWER", value=str(guess))

    num_questions = min(max_num_q_per_turn, len(available_questions))
    selected_questions = rng.sample(available_questions, num_questions)
    return Action(type="QUESTION", questions=selected_questions)


def _choose_nonrandom_answer(
    sample: Sample,
    world: BranchWorld,
    current_ctx: Dict[int, int],
    query_idx: int,
    oracle_type: str,
    rng: random.Random,
) -> Tuple[int, bool]:
    """
    Choose query answer like LogicQ-style non-random oracle:
    - Only allow answers consistent with the assumed world target y.
    - Adversarial prefers keeping target ambiguous and preserving larger hypothesis space.
    - Cooperative prefers forcing the target quickly.
    """
    all_cand: List[Dict[str, Any]] = []
    for val in (0, 1):
        trial = dict(current_ctx)
        trial[query_idx] = val
        outcomes = _candidate_outcomes(sample, trial)
        if outcomes.size == 0:
            continue
        uniq = np.unique(outcomes)
        uniq_set = set(int(x) for x in uniq.tolist())
        has_world_target = int(world.y) in uniq_set
        n_outcomes = int(uniq.size)
        keeps_ambiguous = n_outcomes > 1
        forces_world_target = (n_outcomes == 1 and int(next(iter(uniq_set))) == int(world.y))
        all_cand.append({
            "val": int(val),
            "n_outcomes": n_outcomes,
            "n_states": int(outcomes.size),
            "has_world_target": bool(has_world_target),
            "keeps_ambiguous": bool(keeps_ambiguous),
            "forces_world_target": bool(forces_world_target),
        })

    if not all_cand:
        raise RuntimeError(
            f"No feasible oracle answer for query_idx={query_idx} "
            f"(no candidate states remain under either answer)."
        )
    cand = [c for c in all_cand if c["has_world_target"]]
    if not cand:
        raise RuntimeError(
            f"No y-consistent oracle answer for query_idx={query_idx} "
            f"(both answers contradict world target y={int(world.y)})."
        )

    if oracle_type == "adversarial":
        # Prefer keeping target ambiguous, then larger remaining space.
        best = max(cand, key=lambda c: (
            c["keeps_ambiguous"],
            c["n_outcomes"],
            c["n_states"],
            rng.random(),
        ))
    elif oracle_type == "cooperative":
        # Prefer forcing target quickly, then smaller remaining space.
        best = max(cand, key=lambda c: (
            c["forces_world_target"],
            -c["n_outcomes"],
            -c["n_states"],
            rng.random(),
        ))
    else:
        raise ValueError("oracle_type must be random/adversarial/cooperative")

    chosen_val = int(best["val"])
    post = dict(current_ctx)
    post[query_idx] = chosen_val
    post_outcomes = _candidate_outcomes(sample, post)
    is_ambiguous = int(np.unique(post_outcomes).size) > 1
    return chosen_val, is_ambiguous


def oracle_answer(
    sample: Sample,
    world: BranchWorld,
    questions: List[str],
    asked_ctx: Dict[int, int],
    already_asked_genes: set,
    oracle_type: str,
    rng: random.Random,
    forbid_alternatives: bool = False,
    last_round_is_ambiguous: bool = True,
) -> Tuple[str, bool]:
    name_to_idx = {g: i for i, g in enumerate(sample.var_names)}
    askable = set(_effective_queryable_indices(sample, forbid_alternatives))
    observed_idx = set(int(i) for i, _ in sample.observed_idx_vals)

    responses: List[str] = []
    all_is_ambiguous = bool(last_round_is_ambiguous)

    for q in questions:
        if q not in name_to_idx:
            responses.append(f"Invalid gene: {q}.")
            continue

        gi = int(name_to_idx[q])
        if gi in observed_idx:
            responses.append(f"You cannot ask {q} because it is already observed.")
            continue
        if gi not in askable:
            responses.append(f"You cannot ask {q} because it is not queryable.")
            continue

        if q in already_asked_genes:
            responses.append(f"{q} has already been queried.")
            continue

        if oracle_type == "random":
            ans = _bit(world.state, gi)
            asked_ctx[gi] = ans
            post_outcomes = _candidate_outcomes(sample, asked_ctx)
            is_ambiguous = int(np.unique(post_outcomes).size) > 1
        else:
            ans, is_ambiguous = _choose_nonrandom_answer(
                sample=sample,
                world=world,
                current_ctx=asked_ctx,
                query_idx=gi,
                oracle_type=oracle_type,
                rng=rng,
            )
            asked_ctx[gi] = int(ans)

        already_asked_genes.add(q)
        responses.append(f"{q} = {int(ans)}.")
        all_is_ambiguous = bool(all_is_ambiguous and is_ambiguous)

    if not responses:
        responses = ["No valid question parsed."]
    return " ".join(responses), bool(all_is_ambiguous)


def compute_minset_f1(asked: List[str], gt_sets: List[List[str]]) -> float:
    if not asked or not gt_sets:
        return 0.0
    a = set(x.lower() for x in asked)
    best = 0.0
    for g in gt_sets:
        gg = set(x.lower() for x in g)
        inter = len(a & gg)
        if inter == 0:
            continue
        p = inter / len(a)
        r = inter / len(gg)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


def compute_minset_jaccard(asked: List[str], gt_sets: List[List[str]]) -> float:
    if not asked or not gt_sets:
        return 0.0
    a = set(x.lower() for x in asked)
    best = 0.0
    for g in gt_sets:
        gg = set(x.lower() for x in g)
        u = len(a | gg)
        if u == 0:
            continue
        j = len(a & gg) / u
        best = max(best, j)
    return best


def covers_any_gt_sufficient_set(asked: List[str], gt_sets: List[List[str]]) -> bool:
    if not asked or not gt_sets:
        return False
    a = set(x.lower() for x in asked)
    for g in gt_sets:
        gg = set(x.lower() for x in g)
        if gg and gg.issubset(a):
            return True
    return False


def _core_metrics(results: List[EpisodeResult], sample_map: Dict[int, Sample]) -> Dict[str, float]:
    if not results:
        return {
            "total_episodes": 0,
            "micro_accuracy": 0.0,
            "macro_accuracy": 0.0,
            "answer_rate": 0.0,
            "avg_questions_used": 0.0,
            "avg_minset_f1": 0.0,
            "avg_minset_jaccard": 0.0,
            "avg_thinking_tokens": 0.0,
            "avg_num_turns_used": 0.0,
            "avg_overhead_questions": 0.0,
            "turn_target_known_rate": 0.0,
            "turn_gt_sufficient_covered_rate": 0.0,
            "episode_ever_target_known_rate": 0.0,
            "episode_ever_gt_sufficient_covered_rate": 0.0,
        }

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    answered = sum(1 for r in results if r.answered)

    by_sample: Dict[int, List[int]] = {}
    for r in results:
        by_sample.setdefault(r.sample_id, []).append(1 if r.correct else 0)
    macro = sum(sum(v) / len(v) for v in by_sample.values()) / len(by_sample)

    avg_q = sum(len(r.questions_asked) for r in results) / total
    avg_turns = sum(r.turns_used for r in results) / total

    tt = 0
    for r in results:
        tt += sum(t.thinking_tokens_count for t in r.turn_logs)
        tt += int(r.final_thinking_tokens_count)
    avg_think = tt / total

    f1s: List[float] = []
    jacs: List[float] = []
    overhead: List[float] = []
    turn_known_flags: List[bool] = []
    turn_suff_flags: List[bool] = []
    ep_ever_known_flags: List[bool] = []
    ep_ever_suff_flags: List[bool] = []
    for r in results:
        s = sample_map[r.sample_id]
        f1s.append(compute_minset_f1(r.questions_asked, s.minimal_sets))
        jacs.append(compute_minset_jaccard(r.questions_asked, s.minimal_sets))
        overhead.append(float(len(r.questions_asked) - s.k))
        turn_known_flags.extend([bool(t.target_known_after_turn) for t in r.turn_logs])
        turn_suff_flags.extend([bool(t.covers_any_gt_sufficient_set) for t in r.turn_logs])
        ep_ever_known_flags.append(any(bool(t.target_known_after_turn) for t in r.turn_logs))
        ep_ever_suff_flags.append(any(bool(t.covers_any_gt_sufficient_set) for t in r.turn_logs))

    return {
        "total_episodes": total,
        "micro_accuracy": correct / total,
        "macro_accuracy": macro,
        "answer_rate": answered / total,
        "avg_questions_used": avg_q,
        "avg_minset_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "avg_minset_jaccard": sum(jacs) / len(jacs) if jacs else 0.0,
        "avg_thinking_tokens": avg_think,
        "avg_num_turns_used": avg_turns,
        "avg_overhead_questions": sum(overhead) / len(overhead) if overhead else 0.0,
        "turn_target_known_rate": (
            sum(1 for x in turn_known_flags if x) / len(turn_known_flags) if turn_known_flags else 0.0
        ),
        "turn_gt_sufficient_covered_rate": (
            sum(1 for x in turn_suff_flags if x) / len(turn_suff_flags) if turn_suff_flags else 0.0
        ),
        "episode_ever_target_known_rate": (
            sum(1 for x in ep_ever_known_flags if x) / len(ep_ever_known_flags) if ep_ever_known_flags else 0.0
        ),
        "episode_ever_gt_sufficient_covered_rate": (
            sum(1 for x in ep_ever_suff_flags if x) / len(ep_ever_suff_flags) if ep_ever_suff_flags else 0.0
        ),
    }


def compute_metrics(results: List[EpisodeResult], samples: List[Sample]) -> Dict[str, Any]:
    sample_map = {s.sample_id: s for s in samples}
    overall = _core_metrics(results, sample_map)

    # per-k
    by_k: Dict[int, List[EpisodeResult]] = {}
    for r in results:
        by_k.setdefault(sample_map[r.sample_id].k, []).append(r)
    overall["per_k"] = {str(k): _core_metrics(v, sample_map) for k, v in sorted(by_k.items())}

    # per-task
    by_task: Dict[str, List[EpisodeResult]] = {}
    for r in results:
        by_task.setdefault(sample_map[r.sample_id].task_name, []).append(r)
    overall["per_task"] = {t: _core_metrics(v, sample_map) for t, v in sorted(by_task.items())}

    return overall


def validate_forbid_alternatives_feasibility(samples: List[Sample]) -> Dict[str, Any]:
    by_task: Dict[str, Dict[str, Any]] = {}
    bad_eff_lt_k: List[Dict[str, Any]] = []
    bad_eff_eq0: List[Dict[str, Any]] = []
    bad_canonical_not_queryable: List[Dict[str, Any]] = []

    for s in samples:
        eff = len(_effective_queryable_indices(s, forbid_alternatives=True))
        margin = int(eff - s.k)

        stat = by_task.setdefault(s.task_name, {
            "n_samples": 0,
            "min_effective_queryable": 10**9,
            "avg_effective_queryable": 0.0,
            "min_margin_vs_k": 10**9,
            "avg_margin_vs_k": 0.0,
        })
        stat["n_samples"] += 1
        stat["min_effective_queryable"] = min(int(stat["min_effective_queryable"]), int(eff))
        stat["avg_effective_queryable"] += float(eff)
        stat["min_margin_vs_k"] = min(int(stat["min_margin_vs_k"]), margin)
        stat["avg_margin_vs_k"] += float(margin)

        if eff < s.k:
            bad_eff_lt_k.append({
                "sample_id": int(s.sample_id),
                "group_id": s.group_id,
                "task_name": s.task_name,
                "k": int(s.k),
                "effective_queryable": int(eff),
            })
        if eff == 0:
            bad_eff_eq0.append({
                "sample_id": int(s.sample_id),
                "group_id": s.group_id,
                "task_name": s.task_name,
            })

        queryable_names = {s.var_names[i] for i in s.queryable_gene_indices}
        missing = [g for g in s.canonical_minimal_set if g not in queryable_names]
        if missing:
            bad_canonical_not_queryable.append({
                "sample_id": int(s.sample_id),
                "group_id": s.group_id,
                "task_name": s.task_name,
                "missing_canonical_genes": missing,
            })

    for v in by_task.values():
        n = max(1, int(v["n_samples"]))
        v["avg_effective_queryable"] = float(v["avg_effective_queryable"] / n)
        v["avg_margin_vs_k"] = float(v["avg_margin_vs_k"] / n)

    return {
        "per_task": dict(sorted(by_task.items())),
        "num_eff_lt_k": len(bad_eff_lt_k),
        "num_eff_eq0": len(bad_eff_eq0),
        "num_canonical_not_queryable": len(bad_canonical_not_queryable),
        "examples_eff_lt_k": bad_eff_lt_k[:10],
        "examples_eff_eq0": bad_eff_eq0[:10],
        "examples_canonical_not_queryable": bad_canonical_not_queryable[:10],
    }


async def run_episode_async(
    model_name: str,
    port: str,
    sample: Sample,
    world: BranchWorld,
    budget: str,
    max_num_q_per_turn: int,
    oracle_type: str,
    cache: Optional[Dict[str, Dict[str, Any]]],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool,
    emphasize_uncertainty: bool,
    max_retries: int,
    is_flipped: bool,
    rng_seed: int,
    include_budget_in_prompt: bool,
    forbid_alternatives: bool,
    history_append_mode: str,
    save_turn_prompts: bool,
    log_turn_prompts: bool,
    verbose: bool,
) -> EpisodeResult:
    max_turns, effective_max_num_q_per_turn = resolve_episode_limits(
        sample=sample,
        budget=budget,
        max_num_q_per_turn=max_num_q_per_turn,
        forbid_alternatives=forbid_alternatives,
    )

    cache_key = make_episode_cache_key(
        sample,
        world,
        oracle_type,
        is_flipped,
        history_append_mode=history_append_mode,
    )
    if cache is not None and cache_key in cache:
        return dict_to_episode_result(cache[cache_key])

    messages = _build_prompt(
        sample,
        max_turns,
        effective_max_num_q_per_turn,
        emphasize_uncertainty=emphasize_uncertainty,
        forbid_alternatives=forbid_alternatives,
        include_budget_in_prompt=include_budget_in_prompt,
    )

    res = EpisodeResult(
        sample_id=sample.sample_id,
        group_id=sample.group_id,
        task_name=sample.task_name,
        world_y=int(world.y),
        world_state=int(world.state),
    )

    retry_count = 0
    turn = 0
    is_ambiguous = True

    # Context for oracle includes observed assignments + queried answers so far.
    asked_ctx: Dict[int, int] = {int(i): int(v) for i, v in sample.observed_idx_vals}
    asked_genes: set = set([g for g, _ in sample.observed])

    # Parse questions against all model genes; oracle enforces which ones are forbidden.
    valid_genes = list(sample.var_names)
    rng = random.Random((hash(sample.group_id) ^ hash((world.y, world.state)) ^ hash(rng_seed)) & 0xFFFFFFFF)
    random_policy_rng = (
        build_random_policy_rng(sample, world, is_flipped, rng_seed)
        if is_random_policy_model(model_name)
        else None
    )

    while turn <= max_turns:
        cur_messages = copy.deepcopy(messages)
        force_answer_prompt_applied = turn == max_turns
        if force_answer_prompt_applied:
            if sample.target_type == "attractor_id":
                cur_messages[-1]["content"] += FORCE_ANSWER_PROMPT_ATTR
            else:
                cur_messages[-1]["content"] += FORCE_ANSWER_PROMPT_MARKER

        if log_turn_prompts:
            print(
                f"\n[prompt] sample={sample.sample_id} group={sample.group_id} "
                f"task={sample.task_name} turn={turn}\n"
                f"{json.dumps(cur_messages, ensure_ascii=False, indent=2)}\n",
                flush=True,
            )

        if is_random_policy_model(model_name):
            assert random_policy_rng is not None
            action = choose_random_policy_action(
                rng=random_policy_rng,
                sample=sample,
                asked_ctx=asked_ctx,
                questions_asked=res.questions_asked,
                max_num_q_per_turn=effective_max_num_q_per_turn,
                force_answer_prompt_applied=force_answer_prompt_applied,
                forbid_alternatives=forbid_alternatives,
            )
            response = (
                format_answer_text(sample, int(action.value))
                if action.type == "ANSWER" and action.value is not None
                else format_question_response(action.questions)
            )
            cot = ""
            think_tokens = 0
        else:
            if async_generate_single is None:
                raise RuntimeError(
                    "model_utils.async_generate_single is unavailable in this environment. "
                    "Install evaluator runtime deps (including openai) or use model-name=random-baseline."
                ) from MODEL_UTILS_IMPORT_ERROR
            out = await async_generate_single(
                model_name=model_name,
                port=port,
                messages=cur_messages,
                generation_config=generation_config,
            )
            response = out.text
            cot = out.cot
            think_tokens = out.num_thinking_tokens

        if log_turn_prompts:
            print(
                f"[model_response] sample={sample.sample_id} group={sample.group_id} "
                f"task={sample.task_name} turn={turn}\n{response}\n",
                flush=True,
            )

        if not is_random_policy_model(model_name):
            action = parse_action(
                model_name=model_name,
                response=response,
                valid_genes=valid_genes,
                max_num_q_per_turn=effective_max_num_q_per_turn,
                already_extracted=True,
            )

        if action.type == "ANSWER":
            parsed = parse_answer_value(sample, action.value or "")
            res.final_answer = action.value
            res.parsed_final_answer = parsed
            res.answered = True
            res.correct = check_answer_correct(sample, parsed, world)
            res.final_response = response
            res.turns_used = turn
            res.final_cot = cot
            res.final_thinking_tokens_count = think_tokens
            if save_turn_prompts:
                res.final_prompt_messages = copy.deepcopy(cur_messages)
            break

        if action.type == "QUESTION":
            if turn == max_turns:
                retry_count += 1
                if retry_count >= max_retries:
                    res.final_response = response
                    res.turns_used = turn
                    res.budget_violated = True
                    if save_turn_prompts and res.final_prompt_messages is None:
                        res.final_prompt_messages = copy.deepcopy(cur_messages)
                    break
                continue

            retry_count = 0
            oracle_resp, is_ambig = oracle_answer(
                sample=sample,
                world=world,
                questions=action.questions,
                asked_ctx=asked_ctx,
                already_asked_genes=asked_genes,
                oracle_type=oracle_type,
                rng=rng,
                forbid_alternatives=forbid_alternatives,
                last_round_is_ambiguous=is_ambiguous,
            )
            is_ambiguous = bool(is_ambig)

            asked_so_far = list(res.questions_asked) + list(action.questions)
            post_outcomes = _candidate_outcomes(sample, asked_ctx)
            target_known_after_turn = int(np.unique(post_outcomes).size) == 1 if post_outcomes.size > 0 else False
            covers_sufficient_after_turn = covers_any_gt_sufficient_set(asked_so_far, sample.minimal_sets)

            res.turn_logs.append(TurnLog(
                turn=turn,
                questions=action.questions,
                oracle_answer=oracle_resp,
                is_ambiguous=is_ambig,
                target_known_after_turn=bool(target_known_after_turn),
                covers_any_gt_sufficient_set=bool(covers_sufficient_after_turn),
                model_response=response,
                thinking_tokens_count=think_tokens,
                cot=cot,
                prompt_messages=copy.deepcopy(cur_messages) if save_turn_prompts else None,
            ))
            res.questions_asked.extend(action.questions)

            if keep_thinking_trace:
                if "qwen" in model_name.lower():
                    messages.append({"role": "assistant", "content": response, "reasoning_content": cot})
                    messages.append({"role": "tool", "content": oracle_resp})
                else:
                    raise NotImplementedError("keep_thinking_trace currently only supported for Qwen-format models")
            else:
                if history_append_mode == "full":
                    messages.append({"role": "assistant", "content": response})
                elif history_append_mode != "oracle_only":
                    raise ValueError("history_append_mode must be full/oracle_only")
                messages.append({"role": "user", "content": oracle_resp})

            turn += 1
            continue

        # RETRY branch
        retry_count += 1
        if retry_count >= max_retries:
            res.final_response = response
            res.turns_used = turn
            res.budget_violated = True
            if save_turn_prompts and res.final_prompt_messages is None:
                res.final_prompt_messages = copy.deepcopy(cur_messages)
            break

        retry_msg = (
            "Could not parse your response. Reply exactly with either:\n"
            "- Question: [GENE_NAME]\n"
            "- Answer: [INTEGER] (for attractor_id tasks) or Answer: 0/1 (for marker tasks)"
        )
        if keep_thinking_trace:
            if "qwen" in model_name.lower():
                messages.append({"role": "assistant", "content": response, "reasoning_content": cot})
                messages.append({"role": "tool", "content": retry_msg})
            else:
                raise NotImplementedError("keep_thinking_trace currently only supported for Qwen-format models")
        else:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": retry_msg})

    if not res.answered:
        res.turns_used = turn
        res.budget_violated = True

    if cache is not None and cache_file is not None:
        save_episode_to_cache(cache, cache_file, cache_key, res)

    if verbose:
        print(f"[episode] sample={sample.sample_id} task={sample.task_name} y={world.y} correct={res.correct} q={len(res.questions_asked)}")

    return res


async def process_episodes_async(
    episodes: List[Tuple[Sample, BranchWorld]],
    model_name: str,
    port: str,
    budget: str,
    max_num_q_per_turn: int,
    oracle_type: str,
    cache: Optional[Dict[str, Dict[str, Any]]],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool,
    emphasize_uncertainty: bool,
    max_retries: int,
    max_concurrent: int,
    is_flipped: bool,
    seed: int,
    include_budget_in_prompt: bool,
    forbid_alternatives: bool,
    history_append_mode: str,
    save_turn_prompts: bool,
    log_turn_prompts: bool,
    verbose: bool,
) -> List[EpisodeResult]:
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(sample: Sample, world: BranchWorld) -> EpisodeResult:
        async with sem:
            return await run_episode_async(
                model_name=model_name,
                port=port,
                sample=sample,
                world=world,
                budget=budget,
                max_num_q_per_turn=max_num_q_per_turn,
                oracle_type=oracle_type,
                cache=cache,
                cache_file=cache_file,
                generation_config=generation_config,
                keep_thinking_trace=keep_thinking_trace,
                emphasize_uncertainty=emphasize_uncertainty,
                max_retries=max_retries,
                is_flipped=is_flipped,
                rng_seed=seed,
                include_budget_in_prompt=include_budget_in_prompt,
                forbid_alternatives=forbid_alternatives,
                history_append_mode=history_append_mode,
                save_turn_prompts=save_turn_prompts,
                log_turn_prompts=log_turn_prompts,
                verbose=verbose,
            )

    tasks = [_one(s, w) for s, w in episodes]
    out = await asyncio.gather(*tasks)
    return list(out)


def select_worlds(samples: List[Sample], worlds_per_sign: int, seed: int) -> List[Tuple[Sample, BranchWorld]]:
    """
    Select branch worlds with LogicQ-style semantics:
    - `worlds_per_sign` means how many world states to sample *per target outcome/sign y*.
    - <=0 means use all available branch states for each outcome/sign.
    """
    rng = random.Random(seed)
    episodes: List[Tuple[Sample, BranchWorld]] = []
    for s in samples:
        if not s.branches:
            continue
        by_y: Dict[int, List[BranchWorld]] = {}
        for w in s.branches:
            by_y.setdefault(int(w.y), []).append(w)
        for y in sorted(by_y.keys()):
            ws = list(by_y[y])
            rng.shuffle(ws)
            take = len(ws) if worlds_per_sign <= 0 else min(worlds_per_sign, len(ws))
            for w in ws[:take]:
                episodes.append((s, w))
    return episodes


def _sample_counts_by_task_k(samples: List[Sample]) -> Dict[Tuple[str, int], int]:
    out: Dict[Tuple[str, int], int] = {}
    for s in samples:
        key = (s.task_name, int(s.k))
        out[key] = int(out.get(key, 0) + 1)
    return out


def _print_task_k_counts(title: str, samples: List[Sample]) -> None:
    counts = _sample_counts_by_task_k(samples)
    print(title)
    for (task, k), n in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"  task={task:<10} k={k}: {n}")


def _subsample_by_task_k(samples: List[Sample], max_samples_per_task_k: int, seed: int) -> List[Sample]:
    if max_samples_per_task_k <= 0:
        return samples
    rng = random.Random(seed)
    by_cell: Dict[Tuple[str, int], List[Sample]] = {}
    for s in samples:
        by_cell.setdefault((s.task_name, int(s.k)), []).append(s)
    out: List[Sample] = []
    for cell in sorted(by_cell.keys(), key=lambda x: (x[0], x[1])):
        cell_samples = list(by_cell[cell])
        rng.shuffle(cell_samples)
        out.extend(cell_samples[:max_samples_per_task_k])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn GRN evaluator (ss_id / dyn_attr / dyn_marker)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name. Any model in model_registry.py, a hosted model "
                             "(gpt-*, gemini-*, claude-*), 'random'/'random-baseline', or a "
                             "custom model registered via --model-config.")
    parser.add_argument("--model-config", type=str, default=None,
                        help="YAML/JSON file registering a custom OpenAI-compatible model.")
    parser.add_argument("--reasoning-effort", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--port", type=str, default="8011")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to GRN tasks jsonl (e.g., tasks_new.jsonl)")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Path to model cache directory (model_cache.pkl files)")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_GRN_MODELS_DIR,
                        help=f"Path to source GRN model .pickle/.txt files (default: {DEFAULT_GRN_MODELS_DIR})")
    parser.add_argument("--include-tasks", type=str, default="ss_id,dyn_attr,dyn_marker",
                        help="Comma-separated task subset")
    parser.add_argument(
        "--budget",
        type=str,
        default="k",
        choices=["k"] + [str(i) for i in range(0, 11)],
        help="Maximum number of turns (k uses sample.k; legacy 0 means one bulk-question turn then forced answer).",
    )
    parser.add_argument("--max-num-q-per-turn", type=int, default=1)
    parser.add_argument("--oracle-type", type=str, default="adversarial", choices=["random", "adversarial", "cooperative"])
    parser.add_argument("--forbid-alternatives", dest="forbid_alternatives", action="store_true", default=False,
                        help="Forbid asking genes that are only in alternative minimal sufficient sets")
    parser.add_argument("--worlds-per-sufficient-set-per-sign", type=int, default=1,
                        help="How many branch worlds to evaluate per target outcome/sign y for each sample (<=0 means all)")
    parser.add_argument("--worlds-per-sample", type=int, default=None,
                        help="Legacy alias. If set, treated as --worlds-per-sufficient-set-per-sign")
    parser.add_argument("--max-samples-per-task-k", type=int, default=0,
                        help="If >0, subsample up to this many samples for each (task,k) cell")
    parser.add_argument("--results-dir", type=str, default="results/grn/multiturn")
    parser.add_argument("--cache-tag", type=str, default="")
    parser.add_argument("--output-tag", type=str, default="",
                        help="Optional concise stem for result/cache filenames. Dynamic suffixes are still appended.")
    parser.add_argument("--max-concurrent", type=int, default=64)
    parser.add_argument("--keep-thinking-trace", action="store_true")
    parser.add_argument("--no-emphasize-uncertainty", dest="emphasize_uncertainty", action="store_false", default=True)
    parser.add_argument("--no-budget-in-prompt", dest="include_budget_in_prompt", action="store_false", default=True)
    parser.add_argument("--history-append-mode", type=str, default="full",
                        choices=["full", "oracle_only"],
                        help="How to append prior turn history into future prompts.")
    parser.add_argument("--save-turn-prompts", action="store_true",
                        help="Save final/per-turn prompt messages in the episode cache without printing them.")
    parser.add_argument("--log-turn-prompts", action="store_true",
                        help="Print full prompt/messages and model responses each turn without changing cache contents.")
    parser.add_argument("--dump-turn-prompts", action="store_true",
                        help="Legacy debug mode: both save prompt messages to cache and print them each turn.")
    parser.add_argument("--fresh-episode-cache", action="store_true",
                        help="Start from an empty episode cache and remove any pre-existing target cache file.")
    parser.add_argument("--dry-run-load-only", action="store_true",
                        help="Load tasks/cache, validate schema/catalog consistency, expand episodes, then exit before model inference.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
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

    include_tasks = [t.strip() for t in args.include_tasks.split(",") if t.strip()]
    valid_tasks = {"ss_id", "dyn_attr", "dyn_marker"}
    bad = [t for t in include_tasks if t not in valid_tasks]
    if bad:
        raise ValueError(f"Unknown task(s) in --include-tasks: {bad}")
    if not include_tasks:
        raise ValueError("No tasks selected")

    model_name_safe = args.model_name.replace("/", "_") + (f"_{args.reasoning_effort}" if "gpt-oss" in args.model_name.lower() else "")
    results_dir = os.path.join(args.results_dir, model_name_safe)
    os.makedirs(results_dir, exist_ok=True)
    ep_cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(ep_cache_dir, exist_ok=True)

    save_turn_prompts = bool(args.save_turn_prompts or args.dump_turn_prompts)
    log_turn_prompts = bool(args.log_turn_prompts or args.dump_turn_prompts)

    worlds_per_sign = (
        int(args.worlds_per_sample)
        if args.worlds_per_sample is not None
        else int(args.worlds_per_sufficient_set_per_sign)
    )

    worlds_tag = "allworlds" if worlds_per_sign <= 0 else f"{worlds_per_sign}worlds"
    default_tasks = ["ss_id", "dyn_attr", "dyn_marker"]
    include_tasks_norm = (
        [t.strip() for t in include_tasks.split(",") if t.strip()]
        if isinstance(include_tasks, str)
        else list(include_tasks)
    )
    task_tag = "" if include_tasks_norm == default_tasks else f"-tasks-{'-'.join(include_tasks_norm)}"

    if args.output_tag:
        output_name = args.output_tag
    else:
        data_tag = args.cache_tag if args.cache_tag else os.path.splitext(os.path.basename(args.data_file))[0]
        budget_tag = f"budget{args.budget}" if args.max_num_q_per_turn == 1 else f"budget{args.budget}x{args.max_num_q_per_turn}"
        output_name = (
            f"{data_tag}{task_tag}-{budget_tag}-1sets-{worlds_tag}-"
            f"{args.oracle_type}_oracle"
            f"{'-forbid_alternatives' if args.forbid_alternatives else ''}"
            f"-noflip-seed{args.seed}"
            f"{'-nobudgetprompt' if not args.include_budget_in_prompt else ''}"
        )
    if args.keep_thinking_trace:
        output_name += "-keepthink"
    if args.history_append_mode != "full":
        output_name += f"-history{args.history_append_mode}"
    if not args.emphasize_uncertainty:
        output_name += "-noemphuncert"
    if args.max_samples_per_task_k > 0:
        output_name += f"-cap{args.max_samples_per_task_k}pertaskk"
    if save_turn_prompts and log_turn_prompts:
        output_name += "-dumpprompt"
    elif save_turn_prompts:
        output_name += "-saveprompt"
    elif log_turn_prompts:
        output_name += "-logprompt"
    episode_cache_file = os.path.join(ep_cache_dir, f"{output_name}_episodes.jsonl")
    episode_cache = init_episode_cache(episode_cache_file, fresh_episode_cache=args.fresh_episode_cache)

    if is_random_policy_model(args.model_name):
        generation_config = {}
    else:
        if Evaluator is None:
            raise RuntimeError(
                "evaluators.evaluator is unavailable in this environment. "
                "Install evaluator runtime deps (including openai) before running model-backed evaluation."
            ) from EVALUATOR_IMPORT_ERROR
        generation_config = Evaluator(
            model_name=args.model_name,
            reasoning_effort=args.reasoning_effort,
        ).generation_config

    print(f"Loading GRN data from {args.data_file}")
    samples = load_data(
        tasks_file=args.data_file,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        include_tasks=include_tasks,
        seed=args.seed,
    )
    print(f"Loaded {len(samples)} samples (tasks={include_tasks})")
    _print_task_k_counts("Sample counts by (task, k) before optional subsample:", samples)
    samples = _subsample_by_task_k(samples, max_samples_per_task_k=args.max_samples_per_task_k, seed=args.seed)
    if args.max_samples_per_task_k > 0:
        _print_task_k_counts(
            f"Sample counts by (task, k) after capping at {args.max_samples_per_task_k} per cell:",
            samples,
        )

    forbid_check: Optional[Dict[str, Any]] = None
    if args.forbid_alternatives:
        forbid_check = validate_forbid_alternatives_feasibility(samples)
        print("Forbid-alternatives feasibility check:")
        for task_name, d in forbid_check["per_task"].items():
            print(
                f"  {task_name}: min_eff={d['min_effective_queryable']}, "
                f"avg_eff={d['avg_effective_queryable']:.3f}, "
                f"min_margin_vs_k={d['min_margin_vs_k']}, "
                f"avg_margin_vs_k={d['avg_margin_vs_k']:.3f}"
            )
        if (forbid_check["num_eff_lt_k"] > 0 or
                forbid_check["num_eff_eq0"] > 0 or
                forbid_check["num_canonical_not_queryable"] > 0):
            raise RuntimeError(
                "Forbid-alternatives feasibility failed: "
                f"eff<k={forbid_check['num_eff_lt_k']}, "
                f"eff==0={forbid_check['num_eff_eq0']}, "
                f"canonical_not_queryable={forbid_check['num_canonical_not_queryable']}."
            )

    episodes = select_worlds(samples, worlds_per_sign=worlds_per_sign, seed=args.seed)
    print(f"Total episodes (sample x world): {len(episodes)}")

    if args.dry_run_load_only:
        print("[dry_run_load_only] schema / cache / episode expansion checks passed; exiting before inference.")
        return

    print(f"Running async with max_concurrent={args.max_concurrent}")
    results = asyncio.run(
        process_episodes_async(
            episodes=episodes,
            model_name=args.model_name,
            port=args.port,
            budget=args.budget,
            max_num_q_per_turn=args.max_num_q_per_turn,
            oracle_type=args.oracle_type,
            cache=episode_cache,
            cache_file=episode_cache_file,
            generation_config=generation_config,
            keep_thinking_trace=args.keep_thinking_trace,
            emphasize_uncertainty=args.emphasize_uncertainty,
            max_retries=args.max_retries,
            max_concurrent=args.max_concurrent,
            is_flipped=False,
            seed=args.seed,
            include_budget_in_prompt=args.include_budget_in_prompt,
            forbid_alternatives=args.forbid_alternatives,
            history_append_mode=args.history_append_mode,
            save_turn_prompts=save_turn_prompts,
            log_turn_prompts=log_turn_prompts,
            verbose=args.verbose,
        )
    )

    metrics = compute_metrics(results, samples)
    print("\n=== Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    sample_by_id = {s.sample_id: s for s in samples}
    results_file = os.path.join(results_dir, f"{output_name}_results.json")

    results_data = {
        "config": _results_config_from_args(args),
        "forbid_check": forbid_check,
        "metrics": metrics,
        "samples_meta": {
            s.sample_id: {
                "group_id": s.group_id,
                "task_name": s.task_name,
                "k": s.k,
                "model": s.model,
                "minimal_sets": s.minimal_sets,
                "marker_gene": s.marker_gene,
            }
            for s in samples
        },
        "episodes": [
            {
                **episode_result_to_dict(r),
                "k": sample_by_id[r.sample_id].k,
                "minimal_sets": sample_by_id[r.sample_id].minimal_sets,
            }
            for r in results
        ],
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
