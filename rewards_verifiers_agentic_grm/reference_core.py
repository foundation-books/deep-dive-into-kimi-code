"""Dependency-free Chapter 22 reward-contract and evidence oracles.

The module checks chapter-created fixtures.  It neither evaluates a model nor
reconstructs a private Kimi reward implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Mapping, Sequence, cast


DISPOSITIONS = frozenset({"pass", "fail", "indeterminate", "invalid"})
CELL_STATUSES = frozenset({"observed", "unresolved", "inapplicable", "not_evaluated"})
EVIDENCE_STATUSES = frozenset(
    {"reported", "deduced", "fixture", "checked", "measured", "pending", "unknown"}
)


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a scalar reward value")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("numeric value must be finite")
    return Fraction(str(value))


@dataclass(frozen=True)
class EvidenceRecord:
    """Bind a value to its producer, event, version, and content digest."""

    evidence_id: str
    evidence_type: str
    value: object
    producer: str
    event_id: str
    version: str
    digest: str

    def validate(self) -> bool:
        fields = (
            self.evidence_id,
            self.evidence_type,
            self.producer,
            self.event_id,
            self.version,
            self.digest,
        )
        if not all(isinstance(field, str) and field.strip() for field in fields):
            raise ValueError("qualified evidence needs identity and provenance")
        return True


@dataclass(frozen=True)
class VerifierResult:
    """Capture a verifier disposition, witness, and implementation version."""

    disposition: str
    witness: object
    verifier_version: str

    def validate(self) -> bool:
        if self.disposition not in DISPOSITIONS:
            raise ValueError("unknown verifier disposition")
        if not self.verifier_version.strip():
            raise ValueError("verifier version is required")
        return True


def hard_gate(results: Sequence[VerifierResult]) -> int | None:
    """Return one, zero, or unresolved while preserving four dispositions."""
    if not results:
        raise ValueError("hard gate requires at least one verifier")
    dispositions = tuple(result.disposition for result in results if result.validate())
    if all(value == "pass" for value in dispositions):
        return 1
    if any(value in {"fail", "invalid"} for value in dispositions):
        return 0
    return None


def exact_verifier(actual: object, expected: object) -> VerifierResult:
    """Compare two already-normalized, type-compatible exact values."""
    if actual is None or type(actual) is not type(expected):
        return VerifierResult("invalid", "parse or type mismatch", "exact-v1")
    disposition = "pass" if actual == expected else "fail"
    witness = {"actual": actual, "expected": expected}
    return VerifierResult(disposition, witness, "exact-v1")


def tolerance_verifier(
    actual: int | float | Fraction,
    expected: int | float | Fraction,
    *,
    absolute: int | float | Fraction,
    relative: int | float | Fraction,
    unit: str,
    expected_unit: str,
) -> VerifierResult:
    """Apply |actual-expected| <= absolute + relative*|expected|."""
    try:
        value = _fraction(actual)
        target = _fraction(expected)
        abs_tol = _fraction(absolute)
        rel_tol = _fraction(relative)
    except ValueError as error:
        return VerifierResult("invalid", str(error), "tolerance-v1")
    if not unit or unit != expected_unit or abs_tol < 0 or rel_tol < 0:
        return VerifierResult("invalid", "unit or tolerance mismatch", "tolerance-v1")
    bound = abs_tol + rel_tol * abs(target)
    disposition = "pass" if abs(value - target) <= bound else "fail"
    witness = {
        "difference": str(abs(value - target)),
        "bound": str(bound),
        "unit": unit,
    }
    return VerifierResult(disposition, witness, "tolerance-v1")


def mask_iou(
    predicted: Iterable[tuple[int, int]], reference: Iterable[tuple[int, int]]
) -> Fraction:
    """Return exact set IoU and reject the undeclared empty-empty case."""
    left, right = set(predicted), set(reference)
    union = left | right
    if not union:
        raise ValueError("empty-empty IoU convention is not declared")
    return Fraction(len(left & right), len(union))


def relation_verifier(
    claim: str, red: EvidenceRecord | None, blue: EvidenceRecord | None
) -> VerifierResult:
    """Check a parsed below/above/level claim against admitted center records."""
    if red is None or blue is None:
        return VerifierResult("invalid", "missing center evidence", "relation-v2")
    try:
        red.validate()
        blue.validate()
        red_x, red_y = _center_coordinates(red.value)
        blue_x, blue_y = _center_coordinates(blue.value)
    except (TypeError, ValueError):
        return VerifierResult("invalid", "malformed center evidence", "relation-v2")
    del red_x, blue_x
    if claim not in {"below", "above", "level"}:
        return VerifierResult("invalid", "unsupported relation claim", "relation-v2")
    expected = "level"
    if blue_y > red_y:
        expected = "below"
    elif blue_y < red_y:
        expected = "above"
    disposition = "pass" if claim == expected else "fail"
    witness = {
        "claim": claim,
        "red_y": red_y,
        "blue_y": blue_y,
        "expected": expected,
        "coordinate_convention": "image-y-increases-downward",
    }
    return VerifierResult(disposition, witness, "relation-v2")


def _center_coordinates(value: object) -> tuple[Fraction, Fraction]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("center evidence must contain two coordinates")
    x, y = value
    if not isinstance(x, (int, float, Fraction)) or not isinstance(
        y, (int, float, Fraction)
    ):
        raise ValueError("center coordinates must be numeric")
    return _fraction(x), _fraction(y)


@dataclass(frozen=True)
class Component:
    """Hold one optional reward value and its evidence status."""

    value: Fraction | None
    status: str

    def validate(self) -> bool:
        if self.status not in CELL_STATUSES:
            raise ValueError("unknown component status")
        if self.status == "observed" and self.value is None:
            raise ValueError("observed component requires a value")
        if self.status != "observed" and self.value is not None:
            raise ValueError("nonnumeric component status cannot carry a value")
        return True


def observed(value: int | float | Fraction) -> Component:
    """Construct an observed reward component from an exact scalar."""
    return Component(_fraction(value), "observed")


def inapplicable() -> Component:
    """Construct a zero-contributing inapplicable reward component."""
    return Component(None, "inapplicable")


def missing() -> Component:
    """Construct an unresolved component; retain the historical helper name."""
    return Component(None, "unresolved")


@dataclass(frozen=True)
class RewardLedger:
    """Collect hard-gate and optional reward terms for one candidate."""

    gate: int | None
    task: Component
    rule: Component
    grm: Component
    budget_penalty: Component
    verbosity_penalty: Component
    hack_penalty: Component
    version: str

    def validate(self) -> bool:
        if self.gate not in {0, 1, None} or not self.version.strip():
            raise ValueError("ledger gate or version is invalid")
        for name in (
            "task",
            "rule",
            "grm",
            "budget_penalty",
            "verbosity_penalty",
            "hack_penalty",
        ):
            getattr(self, name).validate()
        return True


def _optional_value(component: Component) -> Fraction | None:
    component.validate()
    if component.status == "observed":
        return component.value
    if component.status == "inapplicable":
        return Fraction(0)
    return None


def compose_reward(
    ledger: RewardLedger,
    *,
    rule_weight: Fraction = Fraction(1),
    grm_weight: Fraction = Fraction(1, 5),
    blocked_value: Fraction = Fraction(-1),
) -> Fraction | None:
    """Apply the chapter's hard-first teaching composition."""
    ledger.validate()
    if ledger.gate == 0:
        return blocked_value
    if ledger.gate is None or ledger.task.status != "observed":
        return None
    optional = tuple(
        _optional_value(component)
        for component in (
            ledger.rule,
            ledger.grm,
            ledger.budget_penalty,
            ledger.verbosity_penalty,
            ledger.hack_penalty,
        )
    )
    if any(value is None for value in optional):
        return None
    rule, grm, budget, verbosity, hack = cast(
        tuple[Fraction, Fraction, Fraction, Fraction, Fraction], optional
    )
    assert ledger.task.value is not None
    return (
        ledger.task.value
        + rule_weight * rule
        + grm_weight * grm
        - budget
        - verbosity
        - hack
    )


def public_hidden_gap(
    public_passes: int, public_total: int, hidden_passes: int, hidden_total: int
) -> tuple[Fraction, Fraction, Fraction]:
    """Return exact public rate, hidden rate, and public-minus-hidden gap."""
    counts = (public_passes, public_total, hidden_passes, hidden_total)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise ValueError("pass-rate counts must be integers")
    if public_total <= 0 or hidden_total <= 0:
        raise ValueError("both denominators must be positive")
    if not 0 <= public_passes <= public_total or not 0 <= hidden_passes <= hidden_total:
        raise ValueError("passes must lie within denominators")
    public = Fraction(public_passes, public_total)
    hidden = Fraction(hidden_passes, hidden_total)
    return public, hidden, public - hidden


@dataclass(frozen=True)
class Rubric:
    """Define criteria, scales, disqualifiers, and evidence requirements."""

    criteria: tuple[str, ...]
    scales: Mapping[str, tuple[Fraction, Fraction]]
    disqualifiers: tuple[str, ...]
    evidence_requirements: Mapping[str, tuple[str, ...]]
    origin: str
    version: str

    def validate(self) -> bool:
        if not self.criteria or len(set(self.criteria)) != len(self.criteria):
            raise ValueError("rubric criteria must be unique and nonempty")
        if set(self.criteria) != set(self.scales) or set(self.criteria) != set(
            self.evidence_requirements
        ):
            raise ValueError("every criterion needs a scale and evidence rule")
        if any(low >= high for low, high in self.scales.values()):
            raise ValueError("rubric scales must have increasing bounds")
        if not self.disqualifiers or any(
            not disqualifier.strip() for disqualifier in self.disqualifiers
        ):
            raise ValueError("rubric disqualifiers must be explicit")
        if not self.origin or not self.version:
            raise ValueError("rubric origin and version are required")
        return True


@dataclass(frozen=True)
class ScorepadCell:
    """Bind one candidate-criterion score to evidence and observation status."""

    candidate: str
    criterion: str
    value: Fraction | None
    evidence_ids: tuple[str, ...]
    status: str

    def validate(self, rubric: Rubric) -> bool:
        rubric.validate()
        if self.criterion not in rubric.criteria or self.status not in CELL_STATUSES:
            raise ValueError("scorepad criterion or status is invalid")
        required = rubric.evidence_requirements[self.criterion]
        if self.status == "observed":
            if self.value is None or not set(required).issubset(self.evidence_ids):
                raise ValueError("observed scorepad cell lacks value or evidence")
        elif self.value is not None:
            raise ValueError("unobserved scorepad cell cannot carry a value")
        return True


def verbosity_event(length: int, baseline: int, multiplier: Fraction) -> bool:
    """Apply the strict visible-length boundary."""
    values = (length, baseline)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("visible lengths must be integer token counts")
    if length < 0 or baseline <= 0 or multiplier <= 0:
        raise ValueError("lengths and multiplier are outside the declared domain")
    return Fraction(length) > multiplier * baseline
