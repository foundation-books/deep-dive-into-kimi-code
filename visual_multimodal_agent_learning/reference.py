"""Executable semantics for the Chapter 25 teaching fixtures.

The package is dependency-free and uses exact rational arithmetic whenever the
chapter makes an exact claim. It is an oracle for chapter-created examples,
not an implementation of a Kimi model, trainer, visual tool runtime, or GUI
benchmark.
"""

from __future__ import annotations

import json
from typing import Mapping

from .reference_interaction import (
    active_perception_check,
    diagnostic_certificate_check,
    gui_staleness_check,
    handoff_check,
)
from .reference_perception import (
    _box_iou as _box_iou,
    _ocr_score as _ocr_score,
    coordinate_round_trip_check,
    observation_lineage_check,
    visual_diagnostic_check,
    zero_vision_denominator_check,
)


def all_checks() -> dict[str, Mapping[str, object]]:
    """Return all deterministic Chapter 25 evidence records."""

    return {
        "observation_lineage": observation_lineage_check(),
        "coordinate_round_trip": coordinate_round_trip_check(),
        "visual_diagnostics": visual_diagnostic_check(),
        "zero_vision_denominators": zero_vision_denominator_check(),
        "active_perception": active_perception_check(),
        "gui_staleness": gui_staleness_check(),
        "diagnostic_certificate": diagnostic_certificate_check(),
        "evidence_handoff": handoff_check(),
    }


__all__ = [
    "active_perception_check",
    "all_checks",
    "coordinate_round_trip_check",
    "diagnostic_certificate_check",
    "gui_staleness_check",
    "handoff_check",
    "observation_lineage_check",
    "visual_diagnostic_check",
    "zero_vision_denominator_check",
]


if __name__ == "__main__":
    print(json.dumps(all_checks(), indent=2, sort_keys=True))
