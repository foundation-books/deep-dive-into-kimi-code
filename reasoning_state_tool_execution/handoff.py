"""Checkpoint and handoff summaries for the Chapter 28 teaching oracle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .model import Event, ExecutionState
from .runtime import fold_events


CHECKPOINT_FIELDS = (
    "events",
    "history",
    "registry",
    "calls",
    "attempts",
    "observations",
    "diagnostics",
)
EXPECTED_CHECKPOINT_SHAPES = {
    "K0": (1, 1, 0, 0, 0, 0, 0),
    "K1": (2, 1, 2, 0, 0, 0, 0),
    "K2": (8, 1, 2, 2, 2, 0, 0),
    "K3": (11, 1, 2, 2, 2, 1, 0),
    "K4": (12, 1, 2, 2, 2, 1, 1),
    "K5": (14, 2, 2, 2, 2, 2, 1),
}
INVARIANT_COVERAGE = {
    "I27.1": "metadata and source audit",
    "I27.2": "CH27-D01",
    "I27.3": "CH27-D01/D02/D08",
    "I27.4": "CH27-D02",
    "I27.5": "CH27-D03",
    "I27.6": "CH27-D03/D10",
    "I27.7": "CH27-D04",
    "I27.8": "CH27-D04/D09",
    "I27.9": "CH27-D05",
    "I27.10": "CH27-D04/D09",
    "I27.11": "CH27-D05",
    "I27.12": "CH27-D05/D09",
    "I27.13": "CH27-D05/D06",
    "I27.14": "CH27-D06",
    "I27.15": "CH27-D07/D09",
    "I27.16": "CH27-D07",
    "I27.17": "CH27-D08",
    "I27.18": "CH27-D02 and manuscript audit",
    "I27.19": "CH27-D02",
    "I27.20": "CH27-D06/D10",
}


def _state_summary(events: Iterable[Event]) -> dict[str, object]:
    sequence = tuple(events)
    state = fold_events(sequence)
    return {
        "events": len(sequence),
        "history": len(state.history),
        "registry": len(state.registry),
        "calls": len(state.calls),
        "attempts": len(state.attempts),
        "observations": len(state.observations),
        "diagnostics": len(state.diagnostics),
        "pending": list(state.pending),
    }


def checkpoint_summaries(
    checkpoints: Mapping[str, Iterable[Event]],
) -> dict[str, object]:
    """Summarize event-fold cardinalities and pending calls at each checkpoint."""
    return {name: _state_summary(events) for name, events in checkpoints.items()}


def expected_checkpoint_summaries() -> dict[str, object]:
    """Return the expected checkpoint summaries."""
    summaries: dict[str, object] = {}
    for name, values in EXPECTED_CHECKPOINT_SHAPES.items():
        summary: dict[str, object] = dict(zip(CHECKPOINT_FIELDS, values, strict=True))
        summary["pending"] = ["D3@new-source-version"] if name == "K5" else []
        summaries[name] = summary
    return summaries


def unresolved_attempts(state: ExecutionState) -> list[str]:
    """Return attempt IDs without a corresponding observation."""
    return [
        attempt.attempt_id
        for attempt in state.attempts.values()
        if not any(
            observation.attempt_id == attempt.attempt_id
            for observation in state.observations.values()
        )
    ]
