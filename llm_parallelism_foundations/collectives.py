"""Logical collective semantics, without a distributed runtime."""

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TypeVar

T = TypeVar("T")
Vector = tuple[Fraction, ...]


def group_members(group: Sequence[int], values: Mapping[int, object]) -> None:
    """Require exactly one input from every declared participant."""
    if not group or len(set(group)) != len(group) or set(group) != set(values):
        raise ValueError("participant identities do not match the group")


def all_gather(
    shards: Mapping[int, Sequence[T]], group: Sequence[int]
) -> dict[int, tuple[T, ...]]:
    """Concatenate disjoint shards in declared group order on every rank."""
    group_members(group, shards)
    result = tuple(value for rank in group for value in shards[rank])
    return {rank: result for rank in group}


def all_reduce(
    partials: Mapping[int, Vector], group: Sequence[int]
) -> dict[int, Vector]:
    """Sum equal-length partial contributions and replicate the sum."""
    group_members(group, partials)
    size = len(partials[group[0]])
    if any(len(vector) != size for vector in partials.values()):
        raise ValueError("partial vectors have different shapes")
    result = tuple(
        sum((partials[r][j] for r in group), Fraction()) for j in range(size)
    )
    return {rank: result for rank in group}


def reduce_scatter(
    partials: Mapping[int, Vector], group: Sequence[int]
) -> dict[int, Vector]:
    """Sum, then retain equal consecutive slices in group order."""
    reduced = all_reduce(partials, group)[group[0]]
    if len(reduced) % len(group):
        raise ValueError("the example requires equal-sized output shards")
    size = len(reduced) // len(group)
    return {rank: reduced[j * size : (j + 1) * size] for j, rank in enumerate(group)}


def broadcast(value: T, root: int, group: Sequence[int]) -> dict[int, T]:
    """Copy one declared root's value to every participant."""
    group_members(group, dict.fromkeys(group))
    if root not in group:
        raise ValueError("root is not a participant")
    return dict.fromkeys(group, value)


def gather(
    shards: Mapping[int, Sequence[T]], root: int, group: Sequence[int]
) -> dict[int, tuple[T, ...]]:
    """Concatenate at the root; input storage is outside this result map."""
    if root not in group:
        raise ValueError("root is not a participant")
    return {root: all_gather(shards, group)[root]}


def scatter(
    pieces: Mapping[int, Sequence[T]], root: int, group: Sequence[int]
) -> dict[int, tuple[T, ...]]:
    """The root supplies one piece per rank, without reducing values."""
    group_members(group, pieces)
    if root not in group:
        raise ValueError("root is not a participant")
    return {rank: tuple(pieces[rank]) for rank in group}


def all_to_all(
    buckets: Mapping[int, Mapping[int, Sequence[T]]], group: Sequence[int]
) -> dict[int, tuple[T, ...]]:
    """Deliver tagged destination buckets, including each local bucket."""
    group_members(group, buckets)
    for destinations in buckets.values():
        group_members(group, destinations)
    return {
        destination: tuple(
            item for source in group for item in buckets[source][destination]
        )
        for destination in group
    }


def send_receive(value: T, source: int, destination: int) -> dict[int, T]:
    """A point-to-point transfer delivers only to the named destination."""
    if source == destination:
        raise ValueError("this example transfers between distinct ranks")
    return {destination: value}
