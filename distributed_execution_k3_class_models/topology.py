"""Rank, group, layout, and ownership objects for the Chapter 37 fixture."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

D = P = C = E = T = V = 2
WORLD_SIZE = D * P * C * E * T
AXES = ("d", "p", "c", "e", "t")
DEGREES = {"d": D, "p": P, "c": C, "e": E, "t": T}


@dataclass(frozen=True, order=True)
class Coordinate:
    """One point in the declared logical rank product."""

    d: int
    p: int
    c: int
    e: int
    t: int

    def validate(self) -> None:
        for axis in AXES:
            value = getattr(self, axis)
            if not 0 <= value < DEGREES[axis]:
                raise ValueError(f"{axis}={value} is outside its degree")

    def replaced(self, axis: str, value: int) -> "Coordinate":
        if axis not in AXES:
            raise ValueError(f"unknown axis {axis}")
        values = {name: getattr(self, name) for name in AXES}
        values[axis] = value
        result = Coordinate(**values)
        result.validate()
        return result


@dataclass(frozen=True)
class ProcessGroup:
    """An ordered participant set with semantic identity and epoch."""

    group_id: str
    axis: str
    members: tuple[int, ...]
    epoch: int = 0

    def validate(self) -> None:
        if not self.group_id or self.axis not in AXES:
            raise ValueError("group identity and axis must be declared")
        if len(set(self.members)) != len(self.members):
            raise ValueError("group members must be duplicate-free")
        if any(rank < 0 or rank >= WORLD_SIZE for rank in self.members):
            raise ValueError("group member is outside the world")


@dataclass(frozen=True)
class LayoutSignature:
    """A typed description of a distributed logical object."""

    object_id: str
    global_shape: tuple[int, ...]
    shard_axes: tuple[str, ...]
    replica_axes: tuple[str, ...]
    owner_axes: tuple[str, ...]
    version: tuple[int, int, str]
    dtype: str

    def validate(self) -> None:
        if not self.object_id or not self.global_shape:
            raise ValueError("object identity and global shape are required")
        if any(size <= 0 for size in self.global_shape):
            raise ValueError("shape extents must be positive")
        named = self.shard_axes + self.replica_axes + self.owner_axes
        if any(axis not in AXES for axis in named):
            raise ValueError("layout names an unknown rank axis")
        if set(self.shard_axes) & set(self.replica_axes):
            raise ValueError("one axis cannot be both sharded and replicated")
        if self.version[0] < 0 or self.version[1] < 0 or not self.version[2]:
            raise ValueError("microbatch, layer, and direction form the version")


def encode(coordinate: Coordinate) -> int:
    """Encode ``(d,p,c,e,t)`` with the chapter's mixed-radix order."""
    coordinate.validate()
    return (
        ((coordinate.d * P + coordinate.p) * C + coordinate.c) * E + coordinate.e
    ) * T + coordinate.t


def decode(rank: int) -> Coordinate:
    """Invert :func:`encode` without relying on binary notation."""
    if not 0 <= rank < WORLD_SIZE:
        raise ValueError("rank is outside the declared world")
    quotient, t = divmod(rank, T)
    quotient, e = divmod(quotient, E)
    quotient, c = divmod(quotient, C)
    d, p = divmod(quotient, P)
    result = Coordinate(d, p, c, e, t)
    result.validate()
    return result


def all_coordinates() -> tuple[Coordinate, ...]:
    """Return all coordinates in encoded rank order."""
    coordinates = tuple(Coordinate(*values) for values in product(range(2), repeat=5))
    return tuple(sorted(coordinates, key=encode))


def placement(rank: int) -> tuple[int, int]:
    """Map a rank to the four-node, eight-local-slot teaching fabric."""
    coordinate = decode(rank)
    node = 2 * coordinate.d + coordinate.p
    slot = 4 * coordinate.c + 2 * coordinate.e + coordinate.t
    return node, slot


def axis_group(rank: int, axis: str, epoch: int = 0) -> ProcessGroup:
    """Construct the ordered group that varies one coordinate only."""
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis}")
    base = decode(rank)
    members = tuple(
        encode(base.replaced(axis, value)) for value in range(DEGREES[axis])
    )
    fixed = ",".join(f"{name}{getattr(base, name)}" for name in AXES if name != axis)
    group = ProcessGroup(f"{axis.upper()}[{fixed}]", axis, members, epoch)
    group.validate()
    return group


def groups_for_rank(rank: int) -> dict[str, ProcessGroup]:
    """Return all single-axis groups containing ``rank``."""
    return {axis: axis_group(rank, axis) for axis in AXES}


def group_partition(axis: str) -> tuple[ProcessGroup, ...]:
    """Return each distinct single-axis group exactly once."""
    seen: dict[tuple[int, ...], ProcessGroup] = {}
    for rank in range(WORLD_SIZE):
        group = axis_group(rank, axis)
        seen.setdefault(group.members, group)
    return tuple(seen[members] for members in sorted(seen))


def validate_partition(groups: Iterable[ProcessGroup]) -> None:
    """Require duplicate-free groups that cover the world exactly once."""
    flat: list[int] = []
    for group in groups:
        group.validate()
        flat.extend(group.members)
    if sorted(flat) != list(range(WORLD_SIZE)):
        raise ValueError("groups do not partition the declared rank world")


def focus_parameter_group(parameter_class: str, rank: int = 23) -> tuple[int, ...]:
    """Construct the chapter's post-CP focus reduction group.

    Dense and shared-expert shards are replicated across ``(d,e)``. A routed
    expert shard is reduced only across its outer-data replicas at its semantic
    expert home after temporary-copy gradients have returned.
    """
    base = decode(rank)
    if parameter_class in {"dense", "shared"}:
        return tuple(
            sorted(
                encode(Coordinate(d, base.p, base.c, e, base.t))
                for d in range(D)
                for e in range(E)
            )
        )
    if parameter_class == "routed_e0":
        return tuple(encode(Coordinate(d, base.p, base.c, 0, base.t)) for d in range(D))
    raise ValueError(f"unknown parameter class {parameter_class}")


def shard_columns(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Split a matrix's columns into the two TP shards."""
    width = len(matrix[0])
    if width % T or any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular and TP-divisible")
    per_rank = width // T
    return tuple(
        tuple(tuple(row[start : start + per_rank]) for row in matrix)
        for start in range(0, width, per_rank)
    )


def gather_columns(
    shards: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Reconstruct a row-major matrix from ordered column shards."""
    if len(shards) != T or not shards or not shards[0]:
        raise ValueError("exactly two nonempty shards are required")
    row_count = len(shards[0])
    if any(len(shard) != row_count for shard in shards):
        raise ValueError("shards disagree on row count")
    return tuple(
        tuple(value for shard in shards for value in shard[row])
        for row in range(row_count)
    )
