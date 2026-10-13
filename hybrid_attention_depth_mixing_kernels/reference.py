"""Dependency-free semantic oracles for Chapter 34.

The module models finite, exact versions of the chapter's ownership, layout,
hybrid-dispatch, depth-source, stable-merge, sharding, and commit contracts.
Positive rational kernel weights stand for exponentiated logits, allowing the
softmax mixtures to be checked without floating-point transcendental calls.
It is not an accelerator kernel or a performance model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from .reference_layout import (
    Number as Number,
    OwnerTag as OwnerTag,
    Vector as Vector,
    f as f,
    feature_major_to_logical as feature_major_to_logical,
    logical_to_feature_major as logical_to_feature_major,
    validate_owner as validate_owner,
    vector_add as vector_add,
    vector_scale as vector_scale,
    vector_sum as vector_sum,
)

WeightedValue = tuple[Fraction, Vector]

PACK_LENGTHS = (5, 3)
PACK_OFFSETS = (0, 5, 8)
PACK_ORIGINS = ("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3")
TOY_SCHEDULE = ("K", "K", "K", "G", "G")
TRACE_OUTPUTS: tuple[Vector, ...] = (
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(1)),
    (Fraction(-1), Fraction(1)),
    (Fraction(2), Fraction(0)),
    (Fraction(0), Fraction(2)),
    (Fraction(-1), Fraction(1)),
    (Fraction(1), Fraction(-1)),
    (Fraction(1), Fraction(-2)),
    (Fraction(2), Fraction(1)),
)


def weighted_mixture(items: Sequence[WeightedValue]) -> Vector:
    """Return an exact normalized mixture for positive kernel weights."""
    if not items:
        raise ValueError("a softmax source set must be nonempty")
    width = len(items[0][1])
    if width == 0 or any(weight <= 0 or len(value) != width for weight, value in items):
        raise ValueError("weights must be positive and value widths must agree")
    denominator = sum((weight for weight, _ in items), f(0))
    numerator = vector_sum(
        (vector_scale(weight, value) for weight, value in items), width
    )
    return tuple(item / denominator for item in numerator)


def gated_causal_rows(rows: Sequence[Vector], gate: Vector) -> tuple[Vector, ...]:
    """Evaluate a request-local causal toy of Gated MLA exactly."""
    if not rows or len(gate) != len(rows[0]):
        raise ValueError("rows and gate must have the same positive width")
    outputs: list[Vector] = []
    for stop in range(1, len(rows) + 1):
        # The weights 1..stop are positive exponentiated-logit witnesses.
        mixed = weighted_mixture(
            tuple((f(index + 1), value) for index, value in enumerate(rows[:stop]))
        )
        outputs.append(tuple(g * value for g, value in zip(gate, mixed, strict=True)))
    return tuple(outputs)


def packed_gated_causal(
    requests: Mapping[str, tuple[Vector, ...]], gate: Vector
) -> dict[str, tuple[Vector, ...]]:
    """Evaluate packed requests independently despite physical concatenation."""
    return {
        request: gated_causal_rows(rows, gate) for request, rows in requests.items()
    }


def hybrid_schedule(layer_count: int) -> tuple[str, ...]:
    """Return repeated 3:1 KDA/global groups plus one terminal global layer."""

    if layer_count < 1 or (layer_count - 1) % 4:
        raise ValueError("layer_count must equal four times a group count plus one")
    repeated = ("K", "K", "K", "G") * ((layer_count - 1) // 4)
    return repeated + ("G",)


def full_attnres_example() -> tuple[tuple[Fraction, ...], Vector]:
    """Return the exact three-source Full AttnRes weights and output."""

    values = ((f(1), f(0)), (f(0), f(2)), (f(2), f(2)))
    weights = (f(1), f(2), f(3))
    normalized = tuple(weight / sum(weights, f(0)) for weight in weights)
    return normalized, weighted_mixture(tuple(zip(weights, values, strict=True)))


def block_index_partition(
    output_count: int, group_size: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Partition one-based transformation indices into complete groups and tail."""

    if output_count < 0 or group_size <= 0:
        raise ValueError("output_count must be nonnegative and group_size positive")
    indices = tuple(range(1, output_count + 1))
    complete_count = output_count // group_size
    completed = tuple(
        indices[start : start + group_size]
        for start in range(0, complete_count * group_size, group_size)
    )
    partial = indices[complete_count * group_size :]
    return completed, partial


def block_sources_before(
    target_index: int, group_size: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return completed and current contributor indices before a target.

    Each completed tuple is represented by one block-sum source.  The current
    tuple, when nonempty, is likewise represented by one partial-sum source;
    its members are not individually weighted Block-AttnRes sources.
    """

    if target_index < 1:
        raise ValueError("target indices are one based")
    prior_count = target_index - 1
    return block_index_partition(prior_count, group_size)


def trace_block_values() -> tuple[Vector, Vector, Vector]:
    """Return the two completed toy representations and final partial sum."""

    completed, partial = block_index_partition(len(TRACE_OUTPUTS), 4)
    blocks = tuple(
        vector_sum((TRACE_OUTPUTS[index - 1] for index in group), 2)
        for group in completed
    )
    tail = vector_sum((TRACE_OUTPUTS[index - 1] for index in partial), 2)
    return blocks[0], blocks[1], tail


@dataclass(frozen=True)
class StableSummary:
    """Stable summary using positive weights instead of their logarithms."""

    maximum_weight: Fraction
    normalizer: Fraction
    numerator: Vector


def stable_summary(items: Sequence[WeightedValue]) -> StableSummary | None:
    """Summarize a nonempty source group; return None for the explicit identity."""

    if not items:
        return None
    width = len(items[0][1])
    if width == 0 or any(weight <= 0 or len(value) != width for weight, value in items):
        raise ValueError("weights must be positive and value widths must agree")
    maximum = max(weight for weight, _ in items)
    scaled = tuple((weight / maximum, value) for weight, value in items)
    return StableSummary(
        maximum,
        sum((weight for weight, _ in scaled), f(0)),
        vector_sum((vector_scale(weight, value) for weight, value in scaled), width),
    )


def merge_summaries(
    left: StableSummary | None, right: StableSummary | None
) -> StableSummary | None:
    """Merge stable summaries; None is the distinguished empty identity."""

    if left is None:
        return right
    if right is None:
        return left
    if len(left.numerator) != len(right.numerator):
        raise ValueError("summary widths must match")
    maximum = max(left.maximum_weight, right.maximum_weight)
    left_scale = left.maximum_weight / maximum
    right_scale = right.maximum_weight / maximum
    return StableSummary(
        maximum,
        left_scale * left.normalizer + right_scale * right.normalizer,
        vector_add(
            vector_scale(left_scale, left.numerator),
            vector_scale(right_scale, right.numerator),
        ),
    )


def summary_output(summary: StableSummary | None) -> Vector:
    """Return the represented mixture, rejecting the empty identity."""

    if summary is None or summary.normalizer <= 0:
        raise ValueError("the empty identity has no represented output")
    return tuple(item / summary.normalizer for item in summary.numerator)


def block_direct(
    fixed: Sequence[WeightedValue], partial: WeightedValue | None
) -> Vector:
    """Evaluate one target directly over its Block AttnRes source set."""

    sources = tuple(fixed) if partial is None else tuple(fixed) + (partial,)
    return weighted_mixture(sources)


def block_two_phase(
    fixed: Sequence[WeightedValue], partial: WeightedValue | None
) -> Vector:
    """Evaluate the same Block target via two stable summaries."""

    partial_group = () if partial is None else (partial,)
    return summary_output(
        merge_summaries(stable_summary(fixed), stable_summary(partial_group))
    )


def token_row_fixture(origins: Sequence[str]) -> dict[str, tuple[WeightedValue, ...]]:
    """Build deterministic, token-separable depth sources."""

    result: dict[str, tuple[WeightedValue, ...]] = {}
    for row, origin in enumerate(origins, 1):
        result[origin] = (
            (f(1), (f(row), f(1))),
            (f(2), (f(0), f(row))),
            (f(3), (f(row + 1), f(row - 1))),
            (f(1), (f(-row), f(2))),
        )
    return result


def sequence_sharded_attnres(
    origins: Sequence[str], shards: Sequence[Sequence[str]]
) -> tuple[Vector, ...]:
    """Evaluate token-local rows on disjoint shards and restore origin order."""

    flat = tuple(origin for shard in shards for origin in shard)
    if len(flat) != len(set(flat)) or set(flat) != set(origins):
        raise ValueError("token shards must be disjoint and exhaustive")
    fixture = token_row_fixture(origins)
    local = {
        origin: weighted_mixture(fixture[origin])
        for shard in shards
        for origin in shard
    }
    return tuple(local[origin] for origin in origins)


def depth_storage_bytes(
    source_count: int, token_count: int, width: int, scalar_bytes: int, ranks: int
) -> tuple[int, int]:
    """Return replicated and equal-shard logical payload bytes per rank."""

    if min(source_count, token_count, width, scalar_bytes, ranks) <= 0:
        raise ValueError("all storage dimensions must be positive")
    if token_count % ranks:
        raise ValueError("equal token shards require divisibility")
    replicated = source_count * token_count * width * scalar_bytes
    return replicated, replicated // ranks


@dataclass(frozen=True)
class HybridBundle:
    """One request's immutable accepted-prefix state bundle."""

    request: str
    endpoint: int
    revision: int
    kda_states: tuple[Fraction, ...]
    gmla_records: tuple[tuple[Vector, ...], ...]
    component_endpoints: tuple[int, ...]


def make_bundle(request: str, endpoint: int) -> HybridBundle:
    """Create a deterministic bundle with three KDA and two Gated-MLA owners."""

    records = tuple(
        tuple((f(position + layer), f(layer)) for position in range(endpoint))
        for layer in (4, 5)
    )
    return HybridBundle(
        request=request,
        endpoint=endpoint,
        revision=0,
        kda_states=(f(endpoint), f(endpoint + 1), f(endpoint + 2)),
        gmla_records=records,
        component_endpoints=(endpoint,) * 5,
    )


def stage_continuation(bundle: HybridBundle, token: int) -> HybridBundle:
    """Build all private sequence updates for one proposed token."""

    candidate_endpoint = bundle.endpoint + 1
    if token != candidate_endpoint:
        raise ValueError("proposed token position must extend the accepted endpoint")
    kda = tuple(
        value + f(token + index) for index, value in enumerate(bundle.kda_states)
    )
    records = tuple(
        old + ((f(token + layer), f(candidate_endpoint)),)
        for layer, old in zip((4, 5), bundle.gmla_records, strict=True)
    )
    return HybridBundle(
        request=bundle.request,
        endpoint=candidate_endpoint,
        revision=bundle.revision + 1,
        kda_states=kda,
        gmla_records=records,
        component_endpoints=(candidate_endpoint,) * 5,
    )


def commit_candidate(
    base: HybridBundle, candidate: HybridBundle, accepted: bool
) -> HybridBundle:
    """Commit every candidate component or preserve every base component."""

    if candidate.request != base.request or candidate.endpoint != base.endpoint + 1:
        raise ValueError("candidate does not extend the base request by one token")
    if any(
        endpoint != candidate.endpoint for endpoint in candidate.component_endpoints
    ):
        raise ValueError("candidate components do not share one endpoint")
    return candidate if accepted else base


@dataclass(frozen=True)
class DepthHistory:
    """Forward-local depth history that cannot become prefix state."""

    forward_id: str
    outputs: tuple[Vector, ...]
    consumed: bool = False


def consume_depth_history(history: DepthHistory) -> DepthHistory:
    """Mark a forward-local history consumed exactly once."""

    if history.consumed:
        raise ValueError("depth history was already consumed")
    return replace(history, consumed=True)


def check_record(passed: bool, **details: object) -> dict[str, object]:
    """Return a stable evidence entry."""

    return {"status": "pass" if passed else "fail", **details}


def all_checks() -> dict[str, dict[str, object]]:
    """Return the ten deterministic evidence outcomes used by the campaign."""

    from .checks import all_checks as collect_checks

    return collect_checks()
