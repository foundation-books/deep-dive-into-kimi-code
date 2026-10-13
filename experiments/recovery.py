"""Replay all recorded synthetic coding recovery cuts with modeled costs."""

from __future__ import annotations

from typing import Any

from ._data import load_data
from .recovery_model import run_cell


def run() -> dict[str, Any]:
    """Recreate every recorded state, receipt, event and metric without real effects."""
    packet = load_data("recovery")
    spec = packet["fixture"]
    population = spec["population"]
    cells = [
        run_cell(task, target, window, policy, repetition)
        for task in spec["tasks"]
        for target in population["target_indices"]
        for window in population["windows"]
        for repetition in range(population["repetitions_per_cut_policy"])
        for policy in population["policies"]
    ]
    if len(cells) != 1200 or cells != packet["recorded_cells"]:
        raise ValueError("recovery: replay differs from the complete recorded grid")
    summary = {
        policy: {
            "cells": len(selected := [row for row in cells if row["policy"] == policy]),
            **{
                metric: sum(row["metrics"][metric] for row in selected)
                for metric in cells[0]["metrics"]
            },
        }
        for policy in population["policies"]
    }
    if summary != packet["recorded_summary"]:
        raise ValueError("recovery: recorded summary differs from replay")
    return {
        "experiment": "recovery",
        "status": "pass",
        "checks": len(cells),
        "summary": summary,
        "cost_unit": spec["cost_model"]["unit"],
        "claim_boundary": packet["provenance"]["claim_boundary"],
    }
