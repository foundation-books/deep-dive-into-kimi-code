"""Closed Float64 candidate-selection fixture with assigned normalized costs.

The costs exercise a controller. They are not runtimes or GPU predictions.
Numerical checks execute the five finite candidate meanings before any cost
is consumed, and the retained model is checked again on held-out rows.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from . import reference


@dataclass(frozen=True)
class Case:
    """One deterministic finite workload or held-out input family."""

    rows: int
    width: int
    values: str = "signed"


@dataclass(frozen=True)
class Candidate:
    """Stable candidate ID and its finite mathematical meaning."""

    name: str
    meaning: str
    partitions: int = 0


@dataclass(frozen=True)
class Record:
    """Qualification, assigned scores, and incumbent after one candidate."""

    name: str
    numerical_pass: bool
    score_a: Fraction | None
    score_b: Fraction | None
    mean_score: Fraction | None
    incumbent: str


@dataclass(frozen=True)
class Selection:
    """Closed controller outcome and the selected model's source snapshot."""

    records: tuple[Record, ...]
    last_candidate: str
    selected_candidate: str
    selected_snapshot_sha256: str


A = Case(128, 1024)
B = Case(1, 4096)
TAIL = Case(1, 65)
QUALIFICATION = (A, B, TAIL)
HOLDOUT = (Case(1, 127), Case(1, 257, "near_zero"), Case(3, 257, "zero"))
REVISION_BUDGET = 4

CANDIDATES = (
    Candidate("C0", "reference"),
    Candidate("C1", "padded_divisor"),
    Candidate("C2", "logical_divisor"),
    Candidate("C3", "partitioned_reduction", 2),
    Candidate("C4", "partitioned_reduction", 4),
)
ASSIGNED_COSTS: dict[str, tuple[Fraction | None, Fraction | None]] = {
    "C0": (Fraction(1), Fraction(1)),
    "C1": (Fraction(1, 10), Fraction(1, 10)),  # alluring, but rejected
    "C2": (Fraction(1), Fraction(1)),
    "C3": (Fraction(2, 5), Fraction(4, 5)),
    "C4": (Fraction(7, 10), Fraction(3, 5)),
}


@lru_cache(maxsize=None)
def _inputs(case: Case) -> tuple[reference.Result, reference.Result, tuple[float, ...]]:
    if (
        case.rows < 1
        or case.width < 1
        or case.values not in {"signed", "near_zero", "zero"}
    ):
        raise ValueError("unsupported finite case")
    scale = {"signed": 1.0, "near_zero": 1e-6, "zero": 0.0}[case.values]
    values = tuple(
        tuple(scale * (((r * 17 + j * 13) % 23) - 11) / 7 for j in range(case.width))
        for r in range(case.rows)
    )
    gates = tuple(
        tuple(((r + j) % 7 - 3) / 4 for j in range(case.width))
        for r in range(case.rows)
    )
    gain = tuple(1 + (j % 5) / 10 for j in range(case.width))
    return values, gates, gain


def _next_power_of_two(width: int) -> int:
    return 1 << (width - 1).bit_length()


def _run(candidate: Candidate, case: Case) -> reference.Result:
    values, gates, gain = _inputs(case)
    args = (values, gates, gain, 1e-5)
    if candidate.meaning == "reference":
        return reference.normalized_gated_rows(*args)
    if candidate.meaning == "padded_divisor":
        return reference.padded_divisor_defect_rows(
            *args, padded_width=_next_power_of_two(case.width)
        )
    if candidate.meaning == "logical_divisor":
        return reference.fused_padded_rows(
            *args, padded_width=_next_power_of_two(case.width)
        )
    if candidate.meaning == "partitioned_reduction":
        return reference.partitioned_reduction_rows(
            *args, partitions=candidate.partitions
        )
    raise ValueError("unknown candidate meaning")


def passes_cases(candidate: Candidate, cases: tuple[Case, ...]) -> bool:
    """Compare every finite output with the Float64 reference at 1e-12 tolerance."""
    for case in cases:
        expected = _run(CANDIDATES[0], case)
        actual = _run(candidate, case)
        if len(actual) != len(expected):
            return False
        for got_row, want_row in zip(actual, expected, strict=True):
            if len(got_row) != len(want_row):
                return False
            if any(
                not math.isfinite(got)
                or not math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)
                for got, want in zip(got_row, want_row, strict=True)
            ):
                return False
    return True


def assigned_scores(
    candidate: str,
    numerical_pass: bool,
    costs: Mapping[str, tuple[Fraction | None, Fraction | None]],
) -> tuple[Fraction | None, Fraction | None, Fraction | None]:
    """Return no complete score when a required positive finite input is absent."""
    if not numerical_pass:
        return Fraction(0), Fraction(0), None
    baseline = costs.get("C0")
    candidate_cost = costs.get(candidate)
    if baseline is None or candidate_cost is None:
        return None, None, None
    baseline_a, baseline_b = baseline
    cost_a, cost_b = candidate_cost
    if (
        not isinstance(baseline_a, Fraction)
        or not isinstance(baseline_b, Fraction)
        or not isinstance(cost_a, Fraction)
        or not isinstance(cost_b, Fraction)
    ):
        return None, None, None
    if any(value <= 0 for value in (baseline_a, baseline_b, cost_a, cost_b)):
        return None, None, None
    scores = tuple(
        min(Fraction(1), Fraction(2, 5) * base / cost)
        for base, cost in zip((baseline_a, baseline_b), (cost_a, cost_b), strict=True)
    )
    return scores[0], scores[1], (scores[0] + scores[1]) / 2


def _source_sha256(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _snapshot_sha256(candidate: Candidate) -> str:
    identity = {
        "name": candidate.name,
        "meaning": candidate.meaning,
        "partitions": candidate.partitions,
        "reference_source_sha256": _source_sha256(Path(reference.__file__)),
        "selection_source_sha256": _source_sha256(Path(__file__)),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def select_fixture(
    *,
    costs: Mapping[str, tuple[Fraction | None, Fraction | None]] = ASSIGNED_COSTS,
    candidates: tuple[Candidate, ...] = CANDIDATES,
) -> Selection:
    """Four revisions after C0; exact ties retain the incumbent."""
    if len(candidates) != REVISION_BUDGET + 1 or tuple(c.name for c in candidates) != (
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
    ):
        raise ValueError("the fixed revision budget and order changed")
    incumbent = "C0"
    incumbent_score: Fraction | None = None
    records = []
    for candidate in candidates:
        passed = passes_cases(candidate, QUALIFICATION)
        score_a, score_b, mean_score = assigned_scores(candidate.name, passed, costs)
        if candidate.name == "C0" and mean_score is None:
            raise ValueError("baseline has no complete eligible score")
        if mean_score is not None and (
            incumbent_score is None or mean_score > incumbent_score
        ):
            incumbent = candidate.name
            incumbent_score = mean_score
        records.append(
            Record(candidate.name, passed, score_a, score_b, mean_score, incumbent)
        )
    selected = next(c for c in candidates if c.name == incumbent)
    return Selection(
        tuple(records), candidates[-1].name, incumbent, _snapshot_sha256(selected)
    )


def final_check(
    selection: Selection,
    *,
    candidates: tuple[Candidate, ...] = CANDIDATES,
    checker: Callable[[Candidate, tuple[Case, ...]], bool] = passes_cases,
) -> tuple[Case, ...]:
    """Refuse source/spec drift or a failed held-out check of the saved winner."""
    selected = next(c for c in candidates if c.name == selection.selected_candidate)
    if _snapshot_sha256(selected) != selection.selected_snapshot_sha256:
        raise ValueError("selected finite snapshot changed before final validation")
    if not checker(selected, HOLDOUT):
        raise ValueError("selected finite snapshot failed final validation")
    return HOLDOUT


if __name__ == "__main__":
    choice = select_fixture()
    for record in choice.records:
        print(record)
    print("last:", choice.last_candidate, "selected:", choice.selected_candidate)
    print("held-out checks:", final_check(choice))
