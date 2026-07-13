"""Stage 3: Augment *_tree.json question datasets with patient context paragraphs."""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict
from tqdm import tqdm

from .config import DATA_FOLDER, make_client

log = logging.getLogger(__name__)


def generate_patient_context_paragraph(
    context: Dict[str, str],
    client: Any,
    deployment: str = "deployment_name",
) -> str:
    """Generate a 2-3 sentence patient description from a context dict."""
    assessment = context.get("assessment", "Unknown condition")
    known = {k: v for k, v in context.items() if k != "assessment"}
    cond_text = "\n".join(f"- {k}: {v}" for k, v in known.items()) if known else ""

    prompt = (
        "Based on the following clinical information, generate a brief (2-3 sentences) patient context "
        "paragraph that describes a patient presenting with this condition. Write it as if describing a patient case.\n\n"
        f"Assessment: {assessment}\n"
        + (f"Known clinical findings:\n{cond_text}" if cond_text else "No additional clinical findings provided yet.")
        + "\n\nGenerate ONLY the patient context paragraph, nothing else. "
        "Do not use wording not included in the clinical information, and do not make any assumptions. "
        "Be concise and clinically relevant."
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "You are a medical documentation assistant. Generate brief, clinically accurate patient descriptions.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def run_add_patient_context(dataset_dir: str = DATA_FOLDER) -> None:
    """Stage 3: Add a patient_context field to every question item in *_tree.json files."""
    client = make_client()
    json_files = glob.glob(os.path.join(dataset_dir, "*_tree.json"))
    log.info("Found %d _tree.json files in %s", len(json_files), dataset_dir)

    for json_fp in tqdm(json_files, desc="add_patient_ctx"):
        with open(json_fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("questions"), dict):
            continue
        changed = False
        for qitems in data["questions"].values():
            for item in qitems:
                if item.get("context"):
                    item["patient_context"] = generate_patient_context_paragraph(
                        item["context"], client
                    )
                    changed = True
        if changed:
            with open(json_fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Done. Updated patient_context in %d *_tree.json files", len(json_files))
