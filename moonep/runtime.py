"""Buffer, lifetime, accounting, and gradient oracles for Chapter 36."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from . import reference as ref


def aligned_size(real_rows: int, alignment: int) -> int:
    """Round one nonempty group up and leave an empty group unallocated."""
    if real_rows < 0 or alignment <= 0:
        raise ValueError("row count and alignment are outside their domains")
    return (
        0 if real_rows == 0 else alignment * ((real_rows + alignment - 1) // alignment)
    )


def physical_segments(
    matrix: Sequence[Sequence[int]], homes: Sequence[int], rank: int, alignment: int
) -> tuple[ref.Segment, ...]:
    """Map local semantic rows and remote types to aligned physical groups."""
    if not matrix or not 0 <= rank < len(matrix[0]):
        raise ValueError("rank is outside allocation matrix")
    local = [
        expert
        for expert, home in enumerate(homes, 1)
        if home == rank and matrix[expert - 1][rank]
    ]
    remote = [
        expert
        for expert, home in enumerate(homes, 1)
        if home != rank and matrix[expert - 1][rank]
    ]
    groups = [(expert, expert - 1, "home") for expert in local]
    groups += [
        (expert, len(homes) + slot, "temporary") for slot, expert in enumerate(remote)
    ]
    cursor = 0
    segments = []
    for expert, physical_row, kind in groups:
        real = matrix[expert - 1][rank]
        allocated = aligned_size(real, alignment)
        segments.append(
            ref.Segment(
                expert, physical_row, kind, real, allocated, cursor, cursor + allocated
            )
        )
        cursor += allocated
    return tuple(segments)


def dispatch_buffers(
    plan: ref.Plan,
    homes: Sequence[int] = ref.HOME,
    ranks: int = ref.RANKS,
    alignment: int = ref.ALIGNMENT,
) -> tuple[tuple[ref.Route | None, ...], ...]:
    """Pack routes by execution rank and expert, leaving padding ownerless."""
    target = len(plan.route_snapshot) // ranks
    if target * ranks != len(plan.route_snapshot):
        raise ValueError("route total must divide evenly across ranks")
    matrix = ref.validate_plan(plan, homes, ranks, target)
    buffers: list[tuple[ref.Route | None, ...]] = []
    for rank in range(ranks):
        packed: list[ref.Route | None] = []
        for segment in physical_segments(matrix, homes, rank, alignment):
            records = [
                route
                for route, destination in zip(
                    plan.route_snapshot, plan.destinations, strict=True
                )
                if destination == rank and route.expert == segment.expert
            ]
            if len(records) != segment.real_rows or len(packed) != segment.start:
                raise AssertionError("segment construction disagrees with route plan")
            packed.extend(records)
            packed.extend([None] * (segment.allocated_rows - segment.real_rows))
            if len(packed) != segment.end:
                raise AssertionError("aligned segment end is inconsistent")
        buffers.append(tuple(packed))
    return tuple(buffers)


def diagnostic_checksum_from_buffers(
    buffers: Sequence[Sequence[ref.Route | None]],
    plan: ref.Plan,
    homes: Sequence[int] = ref.HOME,
    alignment: int = ref.ALIGNMENT,
) -> Fraction:
    """Validate dispatched buffers and combine every real route exactly once."""
    if len(buffers) != ref.RANKS:
        raise ValueError("dispatched buffers must name every execution rank")
    matrix = ref.validate_plan(
        plan, homes, len(buffers), len(plan.route_snapshot) // len(buffers)
    )
    expected = {route.identity: route for route in plan.route_snapshot}
    expected_rank = {
        route.identity: rank
        for route, rank in zip(plan.route_snapshot, plan.destinations, strict=True)
    }
    seen: set[ref.RouteId] = set()
    per_origin: dict[str, Fraction] = {}
    for rank, buffer in enumerate(buffers):
        segments = physical_segments(matrix, homes, rank, alignment)
        expected_length = segments[-1].end if segments else 0
        if len(buffer) != expected_length:
            raise ValueError("dispatched buffer length does not match segment ends")
        for segment in segments:
            real = buffer[segment.start : segment.start + segment.real_rows]
            padding = buffer[segment.start + segment.real_rows : segment.end]
            if any(record is None for record in real) or any(
                record is not None for record in padding
            ):
                raise ValueError("real and padding masks disagree with segment lengths")
            for record in real:
                assert record is not None
                canonical = expected.get(record.identity)
                if canonical != record or record.identity in seen:
                    raise ValueError(
                        "combine encountered an unknown or duplicate route"
                    )
                if (
                    expected_rank[record.identity] != rank
                    or record.expert != segment.expert
                ):
                    raise ValueError("route landed in the wrong execution segment")
                seen.add(record.identity)
                value = ref.f(100 * record.token + record.expert)
                per_origin[record.origin] = (
                    per_origin.get(record.origin, ref.f(0)) + record.weight * value
                )
    if seen != set(expected):
        raise ValueError("combine did not consume every planned route")
    return sum(per_origin.values(), ref.f(0))


def layout_bytes(
    experts: int,
    copy_budget: int,
    ranks: int,
    hidden: int,
    intermediate: int,
    weight_bytes: int,
    gradient_bytes: int,
) -> dict[str, int]:
    """Separate logical views from rank-owned physical storage."""
    if (
        experts % ranks
        or min(
            experts,
            copy_budget,
            ranks,
            hidden,
            intermediate,
            weight_bytes,
            gradient_bytes,
        )
        <= 0
    ):
        raise ValueError("layout dimensions must be positive and evenly owned")
    elements = hidden * intermediate
    home_rows = experts // ranks
    return {
        "symmetric_weight_view": (experts + copy_budget) * elements * weight_bytes,
        "owned_home_weights": home_rows * elements * weight_bytes,
        "local_prefetch_weights": copy_budget * elements * weight_bytes,
        "symmetric_gradient_view": (experts + copy_budget) * elements * gradient_bytes,
        "owned_home_gradients": home_rows * elements * gradient_bytes,
        "local_temporary_gradients": copy_budget * elements * gradient_bytes,
        "mapped_reduce_view": ranks * copy_budget * elements * gradient_bytes,
        "local_reduce_storage": copy_budget * elements * gradient_bytes,
    }


def diagnostic_checksum(plan: ref.Plan) -> Fraction:
    """Combine the same exact diagnostic values under any route placement.

    The checksum is a semantic oracle, so it deliberately accepts both the
    imbalanced fixed-home baseline and a perfectly balanced plan. Structural
    plan validation remains a separate gate.
    """
    ref.validate_routes(plan.route_snapshot, ref.SELECTED)
    ref.allocation_matrix(
        plan.route_snapshot, plan.destinations, ref.EXPERTS, ref.RANKS
    )
    per_origin: dict[str, Fraction] = {}
    for route in plan.route_snapshot:
        value = ref.f(100 * route.token + route.expert)
        per_origin[route.origin] = (
            per_origin.get(route.origin, ref.f(0)) + route.weight * value
        )
    return sum(per_origin.values(), ref.f(0))


def validate_bound_operation(
    plan: ref.Plan,
    routes: Sequence[ref.Route],
    microbatch: int,
    layer: int,
    epoch: int,
    expected_plan_digest: str,
) -> None:
    """Reject changed routes, placement, dimensions, or memory lifetime."""
    if ref.route_digest(plan.route_snapshot) != ref.route_digest(routes):
        raise ValueError("route snapshot does not match saved plan")
    if (plan.microbatch, plan.layer, plan.epoch) != (microbatch, layer, epoch):
        raise ValueError("plan identity does not match operation")
    if plan.digest != expected_plan_digest:
        raise ValueError("plan fingerprint does not match saved placement")


def validate_copy(
    copy: ref.PhysicalCopy, plan: ref.Plan, parameter_revision: int
) -> None:
    """Require a fresh value on the planned rank before execution."""
    matrix = ref.allocation_matrix(
        plan.route_snapshot, plan.destinations, ref.EXPERTS, ref.RANKS
    )
    if matrix[copy.expert - 1][copy.rank] == 0:
        raise ValueError("copy is not used by this plan")
    if ref.HOME[copy.expert - 1] == copy.rank:
        raise ValueError("home values are not temporary copies")
    if copy.parameter_revision != parameter_revision or copy.ready_epoch != plan.epoch:
        raise ValueError("temporary expert value is stale")


def validate_view_epoch(plan: ref.Plan, view_epoch: int, current_epoch: int) -> None:
    """Reject a communication-buffer view after its overwrite boundary."""
    if view_epoch != plan.epoch or current_epoch != plan.epoch:
        raise ValueError("zero-copy view is outside its owning memory epoch")


def gradient_partials(plan: ref.Plan) -> dict[tuple[int, int], int]:
    """Sum the exact scalar gradient witness by expert and execution rank."""
    ref.validate_plan(plan, ref.HOME, ref.RANKS, ref.TOKENS_PER_RANK * ref.SELECTED)
    partials: dict[tuple[int, int], int] = {}
    for route, rank in zip(plan.route_snapshot, plan.destinations, strict=True):
        key = (route.expert, rank)
        partials[key] = partials.get(key, 0) + 10 * route.token + route.slot
    return partials


def repatriated_gradients(plan: ref.Plan) -> dict[int, int]:
    """Reduce all execution-rank partials to one semantic expert value."""
    result: dict[int, int] = {}
    for (expert, _rank), value in gradient_partials(plan).items():
        result[expert] = result.get(expert, 0) + value
    return result


def validate_training_envelope(envelope: ref.SupportEnvelope) -> None:
    """Check the Chapter 36 bounded-copy training contract."""
    if envelope.phase not in {"training", "inference"}:
        raise ValueError("phase must be training or inference")
    if envelope.experts % envelope.ranks:
        raise ValueError("experts must divide evenly across ranks")
    if envelope.phase == "training":
        if not envelope.gradients:
            raise ValueError("training requires gradient handling")
        if envelope.copy_budget != envelope.experts // envelope.ranks:
            raise ValueError("training requires B=E/R in the public contract")
    elif envelope.gradients:
        raise ValueError("the smaller-B public mode is forward-only")
