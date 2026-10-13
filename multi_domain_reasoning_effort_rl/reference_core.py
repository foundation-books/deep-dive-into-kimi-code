"""Dependency-free Chapter 21 type, arithmetic, and handoff oracles.

This teaching reference neither trains a model nor reproduces private code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Sequence

DOMAINS = ("g", "a", "c")
EFFORTS = ("L", "H", "M")
CURRICULUM_ORDER = ("M", "H", "L")
EVIDENCE_STATUSES = frozenset(
    {"reported", "reproduced", "deduced", "fixture", "pending", "unknown"}
)
CELL_KEYS = tuple((domain, effort) for domain in DOMAINS for effort in EFFORTS)


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric resource value")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("numeric value must be finite")
    return Fraction(str(value))


def _rejects(function: Callable[..., object], *args: object, **kwargs: object) -> bool:
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    return False


def domain_effort_cells() -> tuple[tuple[str, str], ...]:
    """Return the stable row-major three-by-three cell product."""
    return tuple((domain, effort) for domain in DOMAINS for effort in EFFORTS)


def validate_cell(domain: str, effort: str, *, expert_kind: str = "policy") -> bool:
    """Validate a policy-cell identity and reject architectural conflation."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain}")
    if effort not in EFFORTS:
        raise ValueError(f"unknown effort: {effort}")
    if expert_kind != "policy":
        raise ValueError("domain-effort cells identify policies, not MoE modules")
    return True


@dataclass(frozen=True)
class ResourceVector:
    """Disjoint assistant tokens plus separate event and measured resources."""

    thinking: int
    response: int
    arguments: int
    turns: int
    tools: int
    wall_seconds: Fraction | None = None
    estimated_flops: int | None = None

    def validate(self) -> bool:
        counts = (
            self.thinking,
            self.response,
            self.arguments,
            self.turns,
            self.tools,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counts
        ):
            raise ValueError("logical resource counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("resource counts must be nonnegative")
        if self.wall_seconds is not None and _fraction(self.wall_seconds) < 0:
            raise ValueError("wall time must be nonnegative")
        if self.estimated_flops is not None:
            if isinstance(self.estimated_flops, bool) or self.estimated_flops < 0:
                raise ValueError("estimated FLOPs must be a nonnegative integer")
        return True

    @property
    def assistant_output(self) -> int:
        self.validate()
        return self.thinking + self.response + self.arguments


def token_cost(domain: str, resources: ResourceVector) -> int:
    """Apply the Chapter 21 general-versus-agentic token projection."""
    validate_cell(domain, "L")
    resources.validate()
    if domain == "g":
        return resources.thinking
    return resources.assistant_output


def add_like_units(
    left: int | Fraction,
    left_unit: str,
    right: int | Fraction,
    right_unit: str,
) -> Fraction:
    """Add two values only when their declared units agree exactly."""
    if not left_unit or left_unit != right_unit:
        raise ValueError("resource addition requires identical declared units")
    return _fraction(left) + _fraction(right)


@dataclass(frozen=True)
class BaselineRecord:
    """Bind a positive baseline to a task revision, policy, and estimator."""

    task_id: str
    task_revision: str
    cold_policy: str
    estimator: str
    value: Fraction

    def validate(self) -> bool:
        fields = (self.task_id, self.task_revision, self.cold_policy, self.estimator)
        if not all(field.strip() for field in fields):
            raise ValueError("baseline provenance fields must be nonempty")
        if _fraction(self.value) <= 0:
            raise ValueError("baseline must be positive")
        return True


@dataclass(frozen=True)
class MultiplierRecord:
    """Bind one curriculum multiplier to a domain-effort-stage cell."""

    domain: str
    effort: str
    stage: str
    value: Fraction
    provenance: str

    def validate(self) -> bool:
        validate_cell(self.domain, self.effort)
        if not self.stage.strip() or not self.provenance.strip():
            raise ValueError("multiplier stage and provenance are required")
        if _fraction(self.value) <= 0:
            raise ValueError("multiplier must be positive")
        return True


@dataclass(frozen=True)
class BudgetState:
    """Capture a cap, observed cost, slack or overrun, and their relation."""

    cap: Fraction
    cost: int
    slack: Fraction
    overrun: Fraction
    relation: str


def budget_state(
    task_id: str,
    task_revision: str,
    domain: str,
    effort: str,
    resources: ResourceVector,
    baseline: BaselineRecord,
    multiplier: MultiplierRecord,
) -> BudgetState:
    """Validate provenance and compute the strict-excess budget state."""
    validate_cell(domain, effort)
    baseline.validate()
    multiplier.validate()
    if (baseline.task_id, baseline.task_revision) != (task_id, task_revision):
        raise ValueError("baseline does not match the task revision")
    if (multiplier.domain, multiplier.effort) != (domain, effort):
        raise ValueError("multiplier does not match the declared cell")
    cost = token_cost(domain, resources)
    cap = baseline.value * multiplier.value
    slack = cap - cost
    relation = "below" if slack > 0 else "equal" if slack == 0 else "over"
    return BudgetState(cap, cost, slack, max(Fraction(0), -slack), relation)


def effective_reward(
    task_reward: int | float | Fraction, state: BudgetState
) -> Fraction:
    """Preserve task reward unless a valid trajectory strictly exceeds cap."""
    value = _fraction(task_reward)
    return Fraction(-1) if state.relation == "over" else value


def validate_curriculum(
    records: Sequence[MultiplierRecord], absolute_max: int | Fraction
) -> bool:
    """Require max-to-high-to-low order, descending values, and finite bound."""
    if tuple(record.effort for record in records) != CURRICULUM_ORDER:
        raise ValueError("curriculum must execute max, high, then low")
    if len({record.domain for record in records}) != 1:
        raise ValueError("one curriculum record cannot mix domains")
    for record in records:
        record.validate()
    values = tuple(record.value for record in records)
    if not values[0] > values[1] > values[2] > 0:
        raise ValueError("curriculum multipliers must descend strictly")
    if _fraction(absolute_max) <= 0:
        raise ValueError("max-stage absolute limit must be positive and finite")
    return True


@dataclass(frozen=True)
class AllocationRow:
    """Track prompt, rollout, outcome, token, turn, and tool counts."""

    prompts: int
    rollouts: int
    successes: int
    budget_ok: int
    thinking: int
    output: int
    turns: int
    tools: int

    def validate(self) -> bool:
        values = tuple(self.__dict__.values())
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("allocation coordinates must be integer counts")
        if any(value < 0 for value in values):
            raise ValueError("allocation coordinates must be nonnegative")
        if self.rollouts > self.prompts:
            raise ValueError("completed rollouts cannot exceed admitted prompts")
        if self.successes > self.rollouts or self.budget_ok > self.rollouts:
            raise ValueError("success and compliance require completed rollouts")
        return True


def allocation_totals(rows: Iterable[AllocationRow]) -> AllocationRow:
    """Conserve every allocation coordinate without cross-unit addition."""
    materialized = tuple(rows)
    for row in materialized:
        row.validate()
    fields = tuple(AllocationRow.__dataclass_fields__)
    totals = {name: sum(getattr(row, name) for row in materialized) for name in fields}
    return AllocationRow(**totals)


@dataclass(frozen=True)
class ParetoPoint:
    """Describe one candidate's quality and cost coordinates."""

    name: str
    quality: tuple[Fraction, ...]
    cost: tuple[Fraction, ...]


def pareto_frontier(points: Sequence[ParetoPoint]) -> tuple[str, ...]:
    """Return names of nondominated points under larger-quality/smaller-cost."""
    if not points:
        raise ValueError("at least one Pareto point is required")
    q_dim = len(points[0].quality)
    c_dim = len(points[0].cost)
    if q_dim == 0 or c_dim == 0:
        raise ValueError("quality and cost coordinates must be nonempty")
    if any(len(point.quality) != q_dim or len(point.cost) != c_dim for point in points):
        raise ValueError("Pareto points must share coordinate dimensions")

    def dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
        no_worse_quality = all(
            a >= b for a, b in zip(left.quality, right.quality, strict=True)
        )
        no_worse_cost = all(a <= b for a, b in zip(left.cost, right.cost, strict=True))
        strictly_better = any(
            a > b for a, b in zip(left.quality, right.quality, strict=True)
        ) or any(a < b for a, b in zip(left.cost, right.cost, strict=True))
        return no_worse_quality and no_worse_cost and strictly_better

    return tuple(
        point.name
        for point in points
        if not any(other is not point and dominates(other, point) for other in points)
    )


@dataclass(frozen=True)
class SpecialistHandoff:
    """Minimum evidence-qualified output of one domain-effort cell."""

    policy_id: str
    domain: str
    effort: str
    curriculum_id: str
    baseline_protocol: str
    reward_interface: str
    summary_id: str
    evidence_statuses: tuple[str, ...]
    unknowns: tuple[str, ...]

    def validate(self) -> bool:
        validate_cell(self.domain, self.effort)
        strings = (
            self.policy_id,
            self.curriculum_id,
            self.baseline_protocol,
            self.reward_interface,
            self.summary_id,
        )
        if not all(value.strip() for value in strings):
            raise ValueError("handoff identifiers must be nonempty")
        if not self.evidence_statuses:
            raise ValueError("handoff needs evidence status")
        if any(status not in EVIDENCE_STATUSES for status in self.evidence_statuses):
            raise ValueError("handoff contains an unknown evidence status")
        if any(not unknown.strip() for unknown in self.unknowns):
            raise ValueError("handoff unknowns must be nonempty descriptions")
        return True
