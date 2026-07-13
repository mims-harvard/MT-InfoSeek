"""Central configuration — edit this file before running the pipeline."""
from __future__ import annotations

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Stage 1: folder containing clinical guideline images (.jpg / .jpeg / .png).
# Converted JSON is saved alongside each image as <image_name>_data.json.
IMAGE_FOLDER = "/path/to/guideline/images"

# Stages 2–4: folder containing curated *_data.json tree files and all
# generated outputs (*_tree.json, *_question_list.json, etc.).
DATA_FOLDER = "/path/to/guideline/data"

# ── Pipeline parameters ───────────────────────────────────────────────────────
MAX_K         = 4    # max number of conditions to mask per trajectory
N_DISTRACTORS = 20   # number of distractor questions sampled per guideline
SEED          = 42   # random seed for distractor sampling


def make_client():
    """Return a configured data generation client.

    Edit this function to return an OpenAI-compatible client, e.g.:
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
    """
    raise NotImplementedError(
        "make_client() is not configured. "
        "Edit clinguide_mt/datagen/config.py and return an OpenAI-compatible client."
    )
