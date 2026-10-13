"""Typed storage, replay, compression, deadline, and lease contracts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import Callable, Iterable

from .objects import ObjectKey


@dataclass(frozen=True)
class Representation:
    """Stored payload plus the identity and metadata needed to decode it."""

    key: ObjectKey
    kind: str
    payload: tuple[Fraction, ...]
    metadata: tuple[tuple[str, str], ...]
    exact: bool


@dataclass(frozen=True)
class Policy:
    """Typed transformation between two storage representations."""

    name: str
    input_kind: str
    output_kind: str
    reconstructible: bool


@dataclass(frozen=True)
class Lease:
    """Half-open reservation of a named transfer or buffer resource."""

    resource: str
    key: ObjectKey
    start: int
    stop: int


def compose_policies(initial_kind: str, policies: Iterable[Policy]) -> str:
    """Compose reconstructible policies while enforcing adjacent types."""
    current = initial_kind
    for policy in policies:
        if policy.input_kind != current:
            raise ValueError(
                f"{policy.name} expected {policy.input_kind}, got {current}"
            )
        if not policy.reconstructible:
            raise ValueError(f"{policy.name} leaves no restoration path")
        current = policy.output_kind
    return current


def validate_representation(
    representation: Representation,
    *,
    expected_key: ObjectKey,
    required_metadata: tuple[tuple[str, str], ...],
) -> None:
    """Reject a stale object version or absent/mismatched decoder metadata."""
    if representation.key != expected_key:
        raise ValueError("representation object/version mismatch")
    observed = dict(representation.metadata)
    for name, value in required_metadata:
        if observed.get(name) != value:
            raise ValueError(f"representation metadata mismatch: {name}")


def quantize(values: Iterable[Fraction], step: Fraction) -> tuple[Fraction, ...]:
    """Round exact rational values to the nearest positive-step multiple."""
    if step <= 0:
        raise ValueError("quantization step must be positive")
    result = []
    for value in values:
        scaled = value / step
        rounded = (scaled.numerator * 2 + scaled.denominator) // (
            2 * scaled.denominator
        )
        result.append(rounded * step)
    return tuple(result)


def max_error(left: Iterable[Fraction], right: Iterable[Fraction]) -> Fraction:
    """Return the maximum elementwise absolute error between equal lengths."""
    pairs = tuple(zip(left, right, strict=True))
    return max((abs(a - b) for a, b in pairs), default=Fraction(0))


def replay(
    producer: Callable[..., tuple[int, ...]],
    closure: tuple[object, ...],
    expected_closure: tuple[object, ...],
) -> tuple[int, ...]:
    """Replay a producer only when its observed closure is exactly frozen."""
    if closure != expected_closure:
        raise ValueError("recomputation closure version mismatch")
    return producer(*closure)


def dropout_fixture(
    x: tuple[int, ...], weight: int, rng_counter: int, autocast: str
) -> tuple[int, ...]:
    """Evaluate a deterministic toy dropout producer under a closure."""
    if autocast != "exact-int":
        return tuple((weight * value) // 2 for value in x)
    return tuple(
        weight * value if (index + rng_counter) % 2 == 0 else 0
        for index, value in enumerate(x)
    )


def completion_ms(payload_gib: int, start_ms: Fraction) -> Fraction:
    """Compute serialized transfer completion for the chapter's link model."""
    return start_ms + Fraction(1, 50) + Fraction(payload_gib * 1000, 40)


def readiness_slack(deadline_ms: int, completion: Fraction) -> Fraction:
    """Return deadline minus completion time in milliseconds."""
    return Fraction(deadline_ms) - completion


def validate_leases(leases: Iterable[Lease]) -> None:
    """Reject overlapping half-open leases on the same named resource."""
    by_resource: dict[str, list[Lease]] = {}
    for lease in leases:
        by_resource.setdefault(lease.resource, []).append(lease)
    for resource, rows in by_resource.items():
        ordered = sorted(rows, key=lambda row: (row.start, row.stop))
        for previous, current in pairwise(ordered):
            if current.start < previous.stop:
                raise ValueError(f"overlapping lease on {resource}")
