"""Small offline command line for the Chapter 24 analysis-audit packet."""

from __future__ import annotations

import argparse
import json

from .task_factory import (
    ConceptGraph,
    construct_task,
    learner_bundle,
    selected_materials,
)
from .task_oracle import displayed_scores, evaluate_repair_candidate, run_reference


def main() -> int:
    """Construct, inspect, and check the three book-authored task families."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("construct", "reference", "candidate", "numbers")
    )
    parser.add_argument("--family", choices=("question", "repair", "report"))
    parser.add_argument(
        "--candidate",
        choices=(
            "reference",
            "alternative",
            "buggy",
            "hardcoded",
            "swapped",
            "tamper-counts",
        ),
    )
    args = parser.parse_args()
    if args.command in ("construct", "reference") and args.family is None:
        parser.error("--family is required for construct and reference")
    if args.command == "candidate" and args.candidate is None:
        parser.error("--candidate is required for candidate")
    output: object
    if args.command == "construct":
        task = construct_task(args.family)
        output = {
            "concept_query": ConceptGraph().material_query(
                "balanced accuracy", ("ordinary accuracy",)
            ),
            "selected_materials": selected_materials(),
            "construction": {
                "selected_concepts": task.selected_concepts,
                "source_ids": task.source_ids,
                "packet_id": task.packet_id,
            },
            "learner": learner_bundle(task),
        }
    elif args.command == "reference":
        output = run_reference(args.family)
    elif args.command == "candidate":
        output = evaluate_repair_candidate(args.candidate)
    else:
        output = displayed_scores()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
