"""Dependency-free semantic oracle for Chapter 35.

Exact fractions model route identity, count/scan, permutation, weighted
combine, schedule equivalence, and the scalar backward witness. A small
floating-point SiTU evaluator and routed/shared branch join check the expert
and output dependency graphs without claiming accelerator numerical or
performance equivalence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from math import sqrt

from .reference_situ import (
    FloatVector as FloatVector,
    dot as _dot,
    grouped_situ_fixture as grouped_situ_fixture,
    situ_expert as situ_expert,
)

Scalar = Fraction
Vector = tuple[Scalar, ...]

ORIGINS = (
    ("v", 1),
    ("v", 2),
    ("v", 3),
    ("v", 4),
    ("tau", 1),
    ("tau", 2),
    ("tau", 3),
    ("tau", 4),
    ("tau", 5),
    ("tau", 6),
)
EXPECTED_LOADS = (7, 6, 4, 3)
EXPECTED_OFFSETS = (0, 7, 13, 17, 20)
EXPECTED_CHECKSUM = Fraction(5_521_654_629, 1_000_000)


def f(numerator: int, denominator: int = 1) -> Fraction:
    """Construct an exact scalar."""
    return Fraction(numerator, denominator)


@dataclass(frozen=True, order=True)
class RouteRecord:
    """Immutable logical route identity and payload."""

    request: str
    token: int
    slot: int
    expert: int
    weight: Fraction

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.request, self.token, self.slot)

    @property
    def origin(self) -> tuple[str, int]:
        return (self.request, self.token)


@dataclass(frozen=True)
class Dispatch:
    """Stable expert-major view with both directions of the permutation."""

    records: tuple[RouteRecord, ...]
    rows: tuple[Vector, ...]
    permutation: tuple[int, ...]
    inverse: tuple[int, ...]
    loads: tuple[int, ...]
    offsets: tuple[int, ...]


def toy_routes() -> tuple[RouteRecord, ...]:
    """Return the canonical twenty-record moe-34-a fixture."""
    half = f(1, 2)
    choices = (
        ((1, half), (2, half)),
        ((1, half), (2, half)),
        ((1, half), (2, half)),
        ((1, half), (2, half)),
        ((1, half), (3, half)),
        ((1, half), (3, half)),
        ((4, f(551_543, 1_000_000)), (1, f(448_457, 1_000_000))),
        ((3, half), (2, half)),
        ((4, half), (2, half)),
        ((4, half), (3, half)),
    )
    records: list[RouteRecord] = []
    for origin, selected in zip(ORIGINS, choices, strict=True):
        for slot, (expert, weight) in enumerate(selected, start=1):
            records.append(RouteRecord(*origin, slot, expert, weight))
    validate_records(records, selected_count=2)
    return tuple(records)


def validate_records(records: Sequence[RouteRecord], selected_count: int) -> None:
    """Validate identity, multiplicity, expert distinctness, and weight sums."""
    identities = tuple(record.identity for record in records)
    if len(set(identities)) != len(identities):
        raise ValueError("route identities must be unique")
    grouped: dict[tuple[str, int], list[RouteRecord]] = {}
    for record in records:
        if record.expert < 1 or record.weight <= 0:
            raise ValueError("experts and weights must be positive")
        grouped.setdefault(record.origin, []).append(record)
    for origin, routes in grouped.items():
        if len(routes) != selected_count:
            raise ValueError(
                f"origin {origin!r} must have exactly {selected_count} routes"
            )
        if {route.slot for route in routes} != set(range(1, selected_count + 1)):
            raise ValueError("route slots must be complete")
        if len({route.expert for route in routes}) != selected_count:
            raise ValueError("selected experts must be distinct")
        if sum((route.weight for route in routes), f(0)) != 1:
            raise ValueError("execution weights must sum to one")


def selected_weights(
    raw_scores: Sequence[Fraction], correction: Sequence[Fraction], selected_count: int
) -> tuple[tuple[int, Fraction], ...]:
    """Select with raw-plus-correction and normalize selected raw scores."""
    if len(raw_scores) != len(correction) or not 0 < selected_count <= len(raw_scores):
        raise ValueError("router arrays and selected count are incompatible")
    if any(score <= 0 for score in raw_scores):
        raise ValueError("raw scores must be positive")
    order = sorted(
        range(len(raw_scores)),
        key=lambda index: (raw_scores[index] + correction[index], -index),
        reverse=True,
    )[:selected_count]
    denominator = sum((raw_scores[index] for index in order), f(0))
    return tuple((index + 1, raw_scores[index] / denominator) for index in order)


def validate_branch_widths(
    model_row: Sequence[object], latent_row: Sequence[object]
) -> None:
    """Validate the running example's full-width and latent-width domains."""
    if len(model_row) != 8:
        raise ValueError("router and shared branches require model width 8")
    if len(latent_row) != 4:
        raise ValueError("routed experts require latent width 4")


def stable_latent_join(
    routed_latent: FloatVector,
    up_rows: tuple[FloatVector, ...],
    gamma: FloatVector,
    epsilon: float,
    shared_outputs: tuple[FloatVector, ...],
) -> FloatVector:
    """Normalize and up-project one routed latent, then add shared outputs."""
    if not routed_latent or len(gamma) != len(routed_latent) or epsilon < 0:
        raise ValueError("routed row, gamma, and epsilon are incompatible")
    if not up_rows or any(len(row) != len(routed_latent) for row in up_rows):
        raise ValueError("up projection must map latent width to model width")
    if not shared_outputs or any(len(row) != len(up_rows) for row in shared_outputs):
        raise ValueError("shared outputs must have model width")
    scale = sqrt(
        sum(component * component for component in routed_latent) / len(routed_latent)
        + epsilon
    )
    if scale == 0:
        raise ValueError("RMS normalization scale must be positive")
    normalized = tuple(
        component * weight / scale
        for component, weight in zip(routed_latent, gamma, strict=True)
    )
    routed_model = tuple(_dot(row, normalized) for row in up_rows)
    shared_sum = tuple(
        sum(output[index] for output in shared_outputs) for index in range(len(up_rows))
    )
    return tuple(
        routed + shared for routed, shared in zip(routed_model, shared_sum, strict=True)
    )


def loads_and_offsets(
    records: Sequence[RouteRecord], expert_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Count experts and compute exclusive offsets."""
    loads = [0] * expert_count
    for record in records:
        if not 1 <= record.expert <= expert_count:
            raise ValueError("expert outside configured range")
        loads[record.expert - 1] += 1
    offsets = [0]
    for load in loads:
        offsets.append(offsets[-1] + load)
    return tuple(loads), tuple(offsets)


def validate_partition(
    records: Sequence[RouteRecord],
    expert_count: int,
    loads: Sequence[int],
    offsets: Sequence[int],
) -> None:
    """Reject counts or offsets that do not partition every logical route."""
    expected_loads, expected_offsets = loads_and_offsets(records, expert_count)
    if tuple(loads) != expected_loads:
        raise ValueError("expert loads do not match route records")
    if tuple(offsets) != expected_offsets:
        raise ValueError("exclusive offsets do not match expert loads")
    if len(offsets) != expert_count + 1 or offsets[0] != 0:
        raise ValueError("exclusive offsets have the wrong boundary shape")
    if offsets[-1] != len(records):
        raise ValueError("final offset must equal the logical route count")


def repeated_rows(
    token_rows: Mapping[tuple[str, int], Vector], records: Sequence[RouteRecord]
) -> tuple[Vector, ...]:
    """Repeat one token latent for every route in canonical order."""
    return tuple(token_rows[record.origin] for record in records)


def stable_dispatch(
    records: Sequence[RouteRecord], rows: Sequence[Vector], expert_count: int
) -> Dispatch:
    """Build a stable expert-major permutation and co-permute rows."""
    if len(records) != len(rows):
        raise ValueError("records and rows must co-travel")
    loads, offsets = loads_and_offsets(records, expert_count)
    permutation = tuple(
        sorted(range(len(records)), key=lambda index: records[index].expert)
    )
    inverse_list = [0] * len(records)
    for physical, logical in enumerate(permutation):
        inverse_list[logical] = physical
    dispatched_records = tuple(records[index] for index in permutation)
    dispatched_rows = tuple(rows[index] for index in permutation)
    for expert in range(1, expert_count + 1):
        lo, hi = offsets[expert - 1], offsets[expert]
        if any(record.expert != expert for record in dispatched_records[lo:hi]):
            raise ValueError("expert segment is not pure")
    return Dispatch(
        dispatched_records,
        dispatched_rows,
        permutation,
        tuple(inverse_list),
        loads,
        offsets,
    )


def restore_logical(values: Sequence[Vector], dispatch: Dispatch) -> tuple[Vector, ...]:
    """Restore expert-major values to canonical route order."""
    if len(values) != len(dispatch.records):
        raise ValueError("one value is required per logical route")
    restored: list[Vector | None] = [None] * len(values)
    for physical, logical in enumerate(dispatch.permutation):
        restored[logical] = values[physical]
    if any(value is None for value in restored):
        raise ValueError("inverse permutation left a hole")
    return tuple(value for value in restored if value is not None)


def padding_mask(loads: Sequence[int], alignment: int) -> tuple[tuple[bool, ...], ...]:
    """Return per-expert logical/padding masks."""
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    masks = []
    for load in loads:
        issued = 0 if load == 0 else alignment * ((load + alignment - 1) // alignment)
        masks.append(tuple(index < load for index in range(issued)))
    return tuple(masks)


def vector_add(left: Vector, right: Vector) -> Vector:
    """Add equal-width exact vectors componentwise."""
    if len(left) != len(right):
        raise ValueError("vector widths must match")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def vector_scale(scale: Fraction, value: Vector) -> Vector:
    """Multiply an exact vector by an exact scalar."""
    return tuple(scale * component for component in value)


def combine(
    records: Sequence[RouteRecord], outputs: Sequence[Vector]
) -> dict[tuple[str, int], Vector]:
    """Weighted-combine exactly once by immutable origin."""
    if len(records) != len(outputs) or not outputs:
        raise ValueError("records and outputs must be nonempty and aligned")
    width = len(outputs[0])
    result: dict[tuple[str, int], Vector] = {}
    zero = tuple(f(0) for _ in range(width))
    for record, output in zip(records, outputs, strict=True):
        if len(output) != width:
            raise ValueError("output widths must agree")
        prior = result.get(record.origin, zero)
        result[record.origin] = vector_add(prior, vector_scale(record.weight, output))
    return result


def diagnostic_outputs(records: Sequence[RouteRecord]) -> tuple[Vector, ...]:
    """Return one exact tag coordinate, 100*ordinal+expert, per route."""
    ordinal = {origin: index for index, origin in enumerate(ORIGINS, start=1)}
    return tuple(
        (f(100 * ordinal[record.origin] + record.expert),) for record in records
    )


def diagnostic_trace() -> tuple[dict[tuple[str, int], Vector], Fraction]:
    """Run expert-major dispatch and restore the canonical diagnostic sum."""
    records = toy_routes()
    token_rows = {
        origin: (f(index), f(-index), f(2 * index), f(1))
        for index, origin in enumerate(ORIGINS, 1)
    }
    dispatch = stable_dispatch(records, repeated_rows(token_rows, records), 4)
    expert_major_outputs = tuple(diagnostic_outputs(dispatch.records))
    restored = combine(dispatch.records, expert_major_outputs)
    checksum = sum((value[0] for value in restored.values()), f(0))
    return restored, checksum


def token_centric_diagnostic(
    records: Sequence[RouteRecord],
) -> dict[tuple[str, int], Vector]:
    """Evaluate tag outputs in token order without expert-major materialization."""
    validate_records(records, selected_count=2)
    return combine(records, diagnostic_outputs(records))


def scalar_backward(
    z: Fraction,
    weights: tuple[Fraction, Fraction],
    parameters: tuple[Fraction, Fraction],
) -> dict[str, object]:
    """Return exact gradients for L=(sum_i p_i*w_i*z)^2/2."""
    p1, p2 = weights
    w1, w2 = parameters
    output = p1 * w1 * z + p2 * w2 * z
    return {
        "output": output,
        "dz": output * (p1 * w1 + p2 * w2),
        "dp": (output * w1 * z, output * w2 * z),
        "dw": (output * p1 * z, output * p2 * z),
    }


def replay_dispatch(dispatch: Dispatch) -> Dispatch:
    """Recompute a stable dispatch from canonical rows and records."""
    canonical_records: list[RouteRecord | None] = [None] * len(dispatch.records)
    canonical_rows: list[Vector | None] = [None] * len(dispatch.rows)
    for physical, logical in enumerate(dispatch.permutation):
        canonical_records[logical] = dispatch.records[physical]
        canonical_rows[logical] = dispatch.rows[physical]
    if any(record is None for record in canonical_records) or any(
        row is None for row in canonical_rows
    ):
        raise ValueError("dispatch cannot be replayed from an incomplete inverse")
    return stable_dispatch(
        tuple(record for record in canonical_records if record is not None),
        tuple(row for row in canonical_rows if row is not None),
        len(dispatch.loads),
    )


def replace_origin(record: RouteRecord, request: str, token: int) -> RouteRecord:
    """Construct an explicit owner fault for negative tests."""
    return replace(record, request=request, token=token)


def check_record(passed: bool, **details: object) -> dict[str, object]:
    """Create a stable evidence record."""
    return {"status": "pass" if passed else "fail", **details}


def all_checks() -> dict[str, dict[str, object]]:
    """Return the ten deterministic outcomes used by the evidence campaign."""
    from .checks import all_checks as collect_checks

    return collect_checks()


__all__ = """
EXPECTED_CHECKSUM EXPECTED_LOADS EXPECTED_OFFSETS ORIGINS Dispatch FloatVector
RouteRecord Scalar Vector all_checks check_record combine diagnostic_outputs
diagnostic_trace f grouped_situ_fixture loads_and_offsets padding_mask repeated_rows
replace_origin replay_dispatch restore_logical scalar_backward selected_weights
situ_expert stable_dispatch stable_latent_join token_centric_diagnostic toy_routes
validate_branch_widths validate_partition validate_records vector_add vector_scale
""".split()
