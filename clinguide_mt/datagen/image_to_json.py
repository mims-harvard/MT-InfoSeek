"""Stage 1: Convert clinical guideline images to JSON decision trees via GPT Vision."""
from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from typing import Any
from PIL import Image

from .config import IMAGE_FOLDER, make_client

log = logging.getLogger(__name__)

_VISION_PROMPT = '''
You are an expert in converting clinical decision tree diagrams (provided as images) into structured JSON format. Your goal is to create a JSON representation that precisely mirrors the logic and hierarchy of the decision tree. The JSON must accurately reflect all branching paths, decision points, conditions, and outcomes. All treatment options, diagnostic steps, and relevant qualifiers must be preserved. Maintain the exact hierarchical relationships and nesting.

Please read each treatment tree as a whole, and then follow the rules below:

- Key Names:
  Use clear and descriptive key names based strictly on the wording in the image. Do not introduce abstract labels unless they appear in the diagram.

- Multiple Options/Conditions:
  If treatments or conditions include multiple options or logical conditions (e.g., "A or B", "Yes/No"), represent them using explicit branches, not inferred categories.

- Intermediate-Branch Merging:
  If two or more different conditions under the same decision/test lead to the exact same downstream structure, merge those conditions into a single combined condition key using explicit wording (e.g., "A OR B", "A / B"). This applies only to intermediate branches.

- Near-Identical Subtree Merging (with Passthrough Compression):
  - Compress/flatten branches by collapsing consecutive single-child intermediate nodes that introduce no branching (single-child "passthrough" steps). Keep step/test names and their order; do not remove test/step names.
  - Only treat a node as "passthrough" if it has exactly one downstream child and no alternatives.
  - If, after passthrough compression, branches become identical, merge the conditions into a single combined condition key.

- Leaf-Only Merging:
  Only merge a condition with its outcome when the condition is a terminal branch (no further tests/decisions after it).
  Represent it as a single leaf string: "Given [condition], [final outcome]".

- Outcome-Set Rule for Terminal Tests:
  If a diagnostic test has multiple terminal outcomes and no downstream steps after that test, represent the outcomes as a list of merged leaf strings under the test.
  Do not re-encode those outcomes as nested condition keys under the test.

- Final-Outcome Deletion Rule (converging terminal outcome):
  If ALL terminal branches under a parent node lead to the SAME final concluding node/outcome (e.g., a shared "Follow-up..." box), then omit that concluding node entirely and stop at the last distinct actionable/test nodes. Do not replace it with placeholders.

- Avoid Artificial Markers:
  Do not introduce artificial terminal markers such as "End", null, {}, or placeholders.

- Ignore Extraneous Text:
  Ignore all text that is not part of the decision tree itself (titles, legends, explanatory notes).

Example format (with intermediate-branch merging):
{
  "Post HM": {
    "Pelvic Doppler USS": {
      "Chest X-ray": {
        ">1cm mets OR ? mets": {
          "CT chest": [
            "Given CT chest >1cm mets, MRI Brain",
            "Given CT chest normal, No further tests"
          ]
        },
        "Given Chest X-ray normal, No further tests": "Given Chest X-ray normal, No further tests"
      }
    }
  },
  "Post other pregnancy or on relapse": {
    "Investigations": [
      "Pelvic Doppler USS",
      "MRI pelvis",
      "CT chest / abdo",
      "MRI brain",
      "+/- CT-PET scan"
    ]
  }
}

Now, based on the uploaded decision tree image, generate the corresponding JSON structure.
If no decision tree is found, return an empty JSON object.
'''


def encode_image_to_base64(image_path: str) -> str | None:
    try:
        with Image.open(image_path) as img:
            buf = BytesIO()
            img.save(buf, format=img.format or "JPEG")
            return base64.b64encode(buf.getvalue()).decode()
    except Exception as err:
        log.error("Encoding image failed (%s): %s", image_path, err)
        return None


def _call_vision_api(image_b64: str, client: Any) -> str:
    response = client.chat.completions.create(
        model="deployment_name",
        messages=[
            {"role": "system", "content": "You are a medical guideline expert."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            },
        ],
    )
    return response.choices[0].message.content


def run_image_to_json(image_folder: str = IMAGE_FOLDER) -> None:
    """Walk image_folder, call GPT Vision on each image, save *_data.json alongside it."""
    client = make_client()
    for root, _, files in os.walk(image_folder):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(root, file)
            log.info("Processing: %s", image_path)

            image_b64 = encode_image_to_base64(image_path)
            if not image_b64:
                log.warning("Skipping %s — encoding failed", file)
                continue

            try:
                raw = _call_vision_api(image_b64, client)
            except Exception as e:
                log.error("API error for %s: %s", file, e)
                continue

            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]

            try:
                data_json = json.loads(text)
            except json.JSONDecodeError as e:
                log.error("JSON parse failed for %s: %s", file, e)
                continue

            out_path = os.path.join(root, os.path.splitext(file)[0] + "_data.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data_json, f, indent=4, ensure_ascii=False)
            log.info("Saved: %s", out_path)
