"""Route identity, construction, and placement validation for Chapter 14."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RouteRecord:
    """One globally keyed semantic route plus its physical destination."""

    step: int
    dp_replica: int
    origin: tuple[int, int, int]
    token: int
    source_rank: int
    slot: int
    expert: int
    raw_score: Fraction
    weight: Fraction
    execution_rank: int


TOP2_EXPERTS = (
    (0, 1),
    (0, 1),
    (0, 1),
    (0, 1),
    (0, 2),
    (0, 2),
    (0, 3),
    (0, 3),
)
ROUTE_SCORES = (Fraction(3, 5), Fraction(2, 5))
ROUTE_WEIGHTS = (Fraction(3, 5), Fraction(2, 5))


def home_rank(expert: int) -> int:
    """Map four semantic experts to the two home EP ranks."""

    if expert not in range(4):
        raise ValueError("expert must be in {0,1,2,3}")
    return 0 if expert < 2 else 1


def _route_record(
    step: int, dp_replica: int, token: int, slot: int, expert: int
) -> RouteRecord:
    source_rank = token // 4
    return RouteRecord(
        step=step,
        dp_replica=dp_replica,
        origin=(dp_replica, source_rank, token),
        token=token,
        source_rank=source_rank,
        slot=slot,
        expert=expert,
        raw_score=ROUTE_SCORES[slot - 1],
        weight=ROUTE_WEIGHTS[slot - 1],
        execution_rank=home_rank(expert),
    )


def running_routes(*, step: int = 0, dp_replica: int = 0) -> tuple[RouteRecord, ...]:
    """Construct one replica's fixed eight-token, top-2 route multiset."""

    if step < 0 or dp_replica < 0:
        raise ValueError("step and data-parallel replica must be nonnegative")
    return tuple(
        _route_record(step, dp_replica, token, slot, expert)
        for token, experts in enumerate(TOP2_EXPERTS)
        for slot, expert in enumerate(experts, start=1)
    )


def routes_from_pairs(
    expert_pairs: Sequence[tuple[int, int]],
    *,
    step: int = 0,
    dp_replica: int = 0,
) -> tuple[RouteRecord, ...]:
    """Construct a route fixture from distinct top-2 expert pairs."""

    if len(expert_pairs) != 8:
        raise ValueError("the fixture requires eight token pairs")
    if step < 0 or dp_replica < 0:
        raise ValueError("step and data-parallel replica must be nonnegative")
    if any(
        len(set(pair)) != 2 or any(expert not in range(4) for expert in pair)
        for pair in expert_pairs
    ):
        raise ValueError("each token requires two distinct known experts")
    return tuple(
        _route_record(step, dp_replica, token, slot, expert)
        for token, pair in enumerate(expert_pairs)
        for slot, expert in enumerate(pair, start=1)
    )


def balance_execution(records: Sequence[RouteRecord]) -> tuple[RouteRecord, ...]:
    """Move four E0 records to rank 1 without changing semantic identities."""

    return tuple(
        replace(record, execution_rank=1)
        if record.expert == 0 and record.token < 4
        else record
        for record in records
    )


def route_identity(records: Sequence[RouteRecord]) -> tuple[tuple[object, ...], ...]:
    """Project away placement so semantic-route conservation is checkable."""

    return tuple(
        (
            record.step,
            record.dp_replica,
            record.origin,
            record.slot,
            record.expert,
            record.raw_score,
            record.weight,
        )
        for record in records
    )


def _validate_token_routes(records: Sequence[RouteRecord]) -> None:
    for token in range(8):
        token_records = [record for record in records if record.token == token]
        if {record.slot for record in token_records} != {1, 2}:
            raise ValueError("token slots must be a complete top-2 set")
        if len({record.expert for record in token_records}) != 2:
            raise ValueError("top-2 experts must be distinct")


def _validate_route(record: RouteRecord) -> None:
    expected_origin = (record.dp_replica, record.source_rank, record.token)
    if record.origin != expected_origin:
        raise ValueError("route origin does not match replica and source coordinates")
    if not Fraction(0) < record.raw_score < Fraction(1):
        raise ValueError("raw sigmoid score must lie strictly between zero and one")


def _validate_copy_versions(
    records: Sequence[RouteRecord],
    committed_versions: Sequence[int],
    copy_versions: Mapping[tuple[int, int], int],
) -> None:
    for record in records:
        if record.execution_rank == home_rank(record.expert):
            continue
        key = (record.expert, record.execution_rank)
        if copy_versions.get(key) != committed_versions[record.expert]:
            raise ValueError("temporary expert copy is stale or unavailable")


def validate_execution(
    records: Sequence[RouteRecord],
    expected_semantic_identity: Sequence[tuple[object, ...]],
    *,
    committed_versions: Sequence[int] = (1, 1, 1, 1),
    copy_versions: Mapping[tuple[int, int], int] | None = None,
) -> None:
    """Reject cardinality, identity, top-k, or stale-copy corruption."""

    if len(records) != 16 or route_identity(records) != tuple(
        expected_semantic_identity
    ):
        raise ValueError("route cardinality or semantic identity mismatch")
    if len({(record.step, record.dp_replica) for record in records}) != 1:
        raise ValueError("one placement plan must contain one step and replica")
    _validate_token_routes(records)
    for record in records:
        _validate_route(record)
    _validate_copy_versions(
        records,
        committed_versions,
        {} if copy_versions is None else copy_versions,
    )


__all__ = [
    "ROUTE_SCORES",
    "ROUTE_WEIGHTS",
    "TOP2_EXPERTS",
    "RouteRecord",
    "balance_execution",
    "home_rank",
    "route_identity",
    "routes_from_pairs",
    "running_routes",
    "validate_execution",
]
