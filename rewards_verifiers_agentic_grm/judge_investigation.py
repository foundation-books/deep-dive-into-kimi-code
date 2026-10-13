"""Deterministic rubric and workspace controls for an agentic-judge lesson.

This module models selected consequences of the pinned Prime Intellect source.
It does not run that source, a sandbox, or a model judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Criterion:
    """One named ordered choice scale and its nonnegative reduction weight."""

    name: str
    choices: tuple[str, ...]
    weight: Fraction


@dataclass(frozen=True)
class Verdict:
    """A judge's structurally validated choice and unchecked free-text reason."""

    name: str
    reason: str
    choice: str


def score_verdicts(
    criteria: Sequence[Criterion], verdicts: Sequence[Verdict]
) -> tuple[dict[str, Fraction], Fraction]:
    """Validate complete verdicts and reduce ordered choices by weighted mean."""
    names = [criterion.name for criterion in criteria]
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("criteria must have unique nonempty names")
    if any(
        criterion.weight < 0
        or len(criterion.choices) < 2
        or len(set(criterion.choices)) != len(criterion.choices)
        for criterion in criteria
    ):
        raise ValueError("invalid criterion weight or choice scale")
    total_weight = sum((criterion.weight for criterion in criteria), Fraction(0))
    if total_weight <= 0:
        raise ValueError("total rubric weight must be positive")
    verdict_names = [verdict.name for verdict in verdicts]
    if len(verdict_names) != len(set(verdict_names)) or set(verdict_names) != set(
        names
    ):
        raise ValueError("verdict names must cover criteria exactly once")
    by_name = {verdict.name: verdict for verdict in verdicts}
    scores: dict[str, Fraction] = {}
    for criterion in criteria:
        verdict = by_name[criterion.name]
        if (
            not isinstance(verdict.reason, str)
            or verdict.choice not in criterion.choices
        ):
            raise ValueError("verdict reason or choice is malformed")
        scores[criterion.name] = Fraction(
            criterion.choices.index(verdict.choice), len(criterion.choices) - 1
        )
    reward = (
        sum(
            (criterion.weight * scores[criterion.name] for criterion in criteria),
            Fraction(0),
        )
        / total_weight
    )
    return scores, reward


def prepare_workspace(
    solver_files: Mapping[str, str],
    published_artifacts: Mapping[str, str],
    *,
    shared: bool,
    trace: str,
) -> dict[str, str]:
    """Keep the shared box or restore only published files into a fresh box."""
    workspace = dict(solver_files if shared else published_artifacts)
    workspace.pop("verdict.json", None)
    workspace.pop("trace.json", None)
    workspace["trace.json"] = trace
    return workspace


def read_verdict_file(
    workspace: Mapping[str, str], criteria: Sequence[Criterion]
) -> Fraction:
    """Reject missing, malformed, incomplete, or illegal judge verdicts."""
    if "verdict.json" not in workspace:
        raise ValueError("judge wrote no fresh verdict")
    try:
        data = json.loads(workspace["verdict.json"])
        rows = data["verdicts"]
        if not isinstance(rows, list):
            raise TypeError("verdicts must be a list")
        verdicts = [Verdict(row["name"], row["reason"], row["verdict"]) for row in rows]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("malformed judge verdict") from error
    return score_verdicts(criteria, verdicts)[1]


def evidence_links_valid(
    claimed_ids: Sequence[str], available_ids: Sequence[str]
) -> bool:
    """Extra book-created check; a reason string alone is not a certificate."""
    return bool(claimed_ids) and set(claimed_ids).issubset(available_ids)
