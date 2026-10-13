"""Decision producers and comparison summaries for the Chapter 22 fixture."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Sequence

from .reference_core import verbosity_event


@dataclass(frozen=True)
class ComparisonRecord:
    """Record a pairwise outcome with display and evaluator versions."""

    left: str
    right: str
    displayed_order: tuple[str, str]
    outcome: str | None
    rubric_version: str
    judge_version: str | None
    decision_source: str = "learned_comparison"

    def validate(self) -> bool:
        if not self.left or not self.right or self.left == self.right:
            raise ValueError("comparison needs distinct candidate identities")
        if set(self.displayed_order) != {self.left, self.right}:
            raise ValueError("displayed order must preserve both candidates")
        unresolved = self.decision_source in {
            "unresolved_policy",
            "unresolved_execution",
        }
        if unresolved != (self.outcome is None):
            raise ValueError("unresolved comparisons must retain no outcome")
        if self.outcome not in {None, self.left, self.right, "tie", "abstain"}:
            raise ValueError("comparison outcome is invalid")
        if self.decision_source not in {
            "automatic_verbosity",
            "learned_comparison",
            "unresolved_policy",
            "unresolved_execution",
        }:
            raise ValueError("unknown comparison decision source")
        if not self.rubric_version:
            raise ValueError("comparison policy/rubric version is required")
        if self.decision_source == "learned_comparison" and not self.judge_version:
            raise ValueError("learned comparison requires judge provenance")
        if self.decision_source == "automatic_verbosity" and (
            self.outcome not in {self.left, self.right}
            or self.judge_version is not None
        ):
            raise ValueError("automatic winner must not claim a judge call")
        return True


def win_fraction(
    records: Sequence[ComparisonRecord], candidate: str
) -> Fraction | None:
    """Compute the illustrative nonabstained win fraction."""
    wins = losses = ties = 0
    for record in records:
        record.validate()
        if candidate not in {record.left, record.right} or record.outcome in {
            "abstain",
            None,
        }:
            continue
        if record.outcome == "tie":
            ties += 1
        elif record.outcome == candidate:
            wins += 1
        else:
            losses += 1
    denominator = wins + losses + ties
    if denominator == 0:
        return None
    return Fraction(2 * wins + ties, 2 * denominator)


def compare_candidates(
    left: str,
    right: str,
    left_length: int,
    right_length: int,
    judge: Callable[[], str],
) -> ComparisonRecord:
    """Run the teaching verbosity policy before any learned comparison call."""
    crosses = tuple(
        verbosity_event(length, 8, Fraction(2))
        for length in (left_length, right_length)
    )
    outcome: str | None
    judge_version: str | None = None
    if crosses[0] != crosses[1]:
        outcome = right if crosses[0] else left
        source = "automatic_verbosity"
    elif all(crosses):
        outcome, source = None, "unresolved_policy"
    else:
        judge_version = "studio-judge-v1"
        try:
            outcome, source = judge(), "learned_comparison"
        except TimeoutError:
            outcome, source = None, "unresolved_execution"
    record = ComparisonRecord(
        left,
        right,
        (left, right),
        outcome,
        "studio-comparison-v1",
        judge_version,
        source,
    )
    record.validate()
    return record
