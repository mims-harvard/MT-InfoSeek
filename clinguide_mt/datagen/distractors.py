"""Stage 4: Build per-guideline question lists and distractor sets; summarize questions."""
from __future__ import annotations

import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Set

from tqdm import tqdm

from .config import DATA_FOLDER, N_DISTRACTORS, SEED, make_client

log = logging.getLogger(__name__)


# ── Shared tree utility ───────────────────────────────────────────────────────

def extract_condition_labels(tree: Dict[str, Any]) -> tuple[Set[str], str]:
    """Return (condition_label_set, assessment_category) from a Pydantic-format tree dict."""
    conditions: Set[str] = set()
    category: str | None = None

    def dfs(node: Dict[str, Any]) -> None:
        nonlocal category
        ntype = node.get("type")
        if ntype == "assessment":
            category = node.get("label", "Unknown")
            if "next" in node:
                dfs(node["next"])
        elif ntype == "condition":
            if label := node.get("label"):
                conditions.add(label)
            for nxt in node.get("branches", {}).values():
                dfs(nxt)
            if "branches" not in node and "next" in node:
                dfs(node["next"])

    dfs(tree)
    return conditions, category or "Unknown"


def _load_tree(folder: str, filename: str) -> Dict[str, Any]:
    with open(os.path.join(folder, filename), "r") as f:
        return json.load(f)


# ── Question list with random distractors ─────────────────────────────────────

def _pool_excluding(folder_path: str, exclude_id: str) -> List[str]:
    """Collect all unique condition labels from all guidelines except exclude_id."""
    pool: Set[str] = set()
    for fname in os.listdir(folder_path):
        if not fname.endswith("_data.json"):
            continue
        if fname.removesuffix("_data.json") == exclude_id:
            continue
        conds, _ = extract_condition_labels(_load_tree(folder_path, fname))
        pool.update(conds)
    return list(pool)


def create_question_list_for_guideline(
    folder_path: str,
    sample_id: str,
    n_distractors: int = N_DISTRACTORS,
    seed: int = SEED,
) -> Dict[str, Any]:
    tree = _load_tree(folder_path, f"{sample_id}_data.json")
    questions, assessment = extract_condition_labels(tree)
    questions = sorted(questions)
    pool = [q for q in _pool_excluding(folder_path, sample_id) if q not in set(questions)]
    distractors = random.Random(seed).sample(pool, min(n_distractors, len(pool)))
    return {"assessment": assessment, "questions": questions, "distractors": distractors}


def run_build_question_lists(
    data_folder: str = DATA_FOLDER,
    n_distractors: int = N_DISTRACTORS,
    seed: int = SEED,
) -> None:
    """Stage 4a: For each *data.json, save {id}_question_list.json."""
    copy_files = sorted(f for f in os.listdir(data_folder) if f.endswith("_data.json"))
    log.info("Building question lists for %d files", len(copy_files))
    for fname in tqdm(copy_files, desc="build_q_lists"):
        sid = fname.removesuffix("_data.json")
        out = create_question_list_for_guideline(data_folder, sid, n_distractors, seed)
        out_path = os.path.join(data_folder, f"{sid}_question_list.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("Done. Saved %d *_question_list.json files", len(copy_files))


# ── Question list with ALL other-guideline questions (no sampling) ─────────────

def _pool_by_guideline(
    folder_path: str,
    exclude_id: str,
    exclude_questions: Set[str],
) -> Dict[int, Dict[str, Any]]:
    """Return all other guidelines' condition labels keyed by 1-indexed guideline number."""
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.endswith("_data.json") and f.removesuffix("_data.json") != exclude_id
    )
    result = {}
    for idx, fname in enumerate(files, start=1):
        sid = fname.removesuffix("_data.json")
        conds, _ = extract_condition_labels(_load_tree(folder_path, fname))
        result[idx] = {"sample_id": sid, "questions": sorted(conds - exclude_questions)}
    return result


def create_question_list_all_for_guideline(
    folder_path: str,
    sample_id: str,
) -> Dict[str, Any]:
    tree = _load_tree(folder_path, f"{sample_id}_data.json")
    questions, assessment = extract_condition_labels(tree)
    questions = sorted(questions)
    distractors = _pool_by_guideline(folder_path, sample_id, set(questions))
    return {"assessment": assessment, "questions": questions, "distractors": distractors}


def run_build_question_lists_all(data_folder: str = DATA_FOLDER) -> None:
    """Stage 4b: For each *data.json, save {id}_question_list_all.json with ALL other guidelines."""
    copy_files = sorted(f for f in os.listdir(data_folder) if f.endswith("_data.json"))
    log.info("Building full question lists for %d files", len(copy_files))
    for fname in tqdm(copy_files, desc="build_q_lists_all"):
        sid = fname.removesuffix("_data.json")
        out = create_question_list_all_for_guideline(data_folder, sid)
        out_path = os.path.join(data_folder, f"{sid}_question_list_all.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("Done. Saved %d *_question_list_all.json files", len(copy_files))


# ── Question summarization (parallel) ────────────────────────────────────────

def _summarize_worker(question: str, deployment: str) -> Dict[str, str | None]:
    """Worker function: summarize one condition label into a short phrase."""
    prompt = (
        "Convert this clinical question into a very short phrase (3-8 words) that captures its essence.\n"
        "Use noun phrases or brief descriptors. No question mark. No full sentence.\n\n"
        "Example:\n"
        '- "Do the fingers change color with exposure to cold?" -> "Color changes or cold sensitivity of fingers"\n'
        f"\nQuestion: {question}\n\nOutput ONLY the short phrase, nothing else:"
    )
    try:
        client = make_client()
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical documentation assistant. Output only the requested short phrase.",
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=64,
        )
        return {"question": question, "summary": resp.choices[0].message.content.strip(), "error": None}
    except Exception as e:
        return {"question": question, "summary": None, "error": str(e)}


def run_summarize_questions(
    data_folder: str = DATA_FOLDER,
    deployment: str = "deployment_name",
    num_workers: int = 8,
    output_path: str | None = None,
) -> Dict[str, str]:
    """Summarize all unique condition labels from *data.json files in parallel."""
    pool: Set[str] = set()
    for fname in os.listdir(data_folder):
        if fname.endswith("_data.json"):
            conds, _ = extract_condition_labels(_load_tree(data_folder, fname))
            pool.update(conds)
    questions = sorted(pool)
    log.info("Summarizing %d unique questions with %d workers", len(questions), num_workers)

    results: Dict[str, str] = {}
    worker = partial(_summarize_worker, deployment=deployment)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker, q): q for q in questions}
        for future in tqdm(as_completed(futures), total=len(futures), desc="summarize_questions"):
            r = future.result()
            results[r["question"]] = r["summary"] if not r["error"] else r["question"]

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Summaries saved to %s", output_path)
    return results
