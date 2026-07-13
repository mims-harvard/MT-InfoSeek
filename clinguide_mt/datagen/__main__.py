"""Entry point: python -m datagen [--stage STAGE]  (run from the clinguide_mt/ directory)"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from .distractors import run_build_question_lists, run_build_question_lists_all
from .image_to_json import run_image_to_json
from .patient_context import run_add_patient_context
from .questions import run_json_to_questions

STAGES = {
    "image_to_json":      run_image_to_json,
    "json_to_questions":  run_json_to_questions,
    "patient_context":    run_add_patient_context,
    "question_lists":     run_build_question_lists,
    "question_lists_all": run_build_question_lists_all,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clinical guideline data generation pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES) + ["all"],
        default="all",
        help="Pipeline stage to run (default: all stages in order)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        for name, fn in STAGES.items():
            logging.info("=== Running stage: %s ===", name)
            fn()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
