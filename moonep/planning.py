"""Constructive and exact allocation oracles for Chapter 36."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from . import reference as ref


def _equal_home_partition(homes: Sequence[int], ranks: int) -> int:
    counts = tuple(homes.count(rank) for rank in range(ranks))
    if not counts or len(set(counts)) != 1:
        raise ValueError("the theorem requires an equal expert home partition")
    return counts[0]


def constructive_balance(
    counts: Sequence[int], homes: Sequence[int], ranks: int
) -> ref.Matrix:
    """Implement the Appendix E one-source filling construction."""
    experts_per_rank = _equal_home_partition(homes, ranks)
    total = sum(counts)
    if total % ranks:
        raise ValueError("total routes must divide evenly across ranks")
    target = total // ranks
    matrix = [[0 for _ in range(ranks)] for _ in counts]
    for expert, (count, home) in enumerate(zip(counts, homes, strict=True)):
        if count < 0:
            raise ValueError("semantic counts must be nonnegative")
        matrix[expert][home] = count
    loads = list(ref.home_loads(counts, homes, ranks))
    fills = 0
    while any(load != target for load in loads):
        receiver = next(rank for rank, load in enumerate(loads) if load < target)
        source = next(rank for rank, load in enumerate(loads) if load > target)
        remaining = target - loads[receiver]
        candidates = sorted(
            (
                (matrix[expert][source], expert)
                for expert, home in enumerate(homes)
                if home == source and matrix[expert][source] > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for available, expert in candidates:
            moved = min(available, remaining)
            matrix[expert][source] -= moved
            matrix[expert][receiver] += moved
            loads[source] -= moved
            loads[receiver] += moved
            remaining -= moved
            if not remaining:
                break
        if remaining:
            raise AssertionError("source home group could not fill receiver")
        fills += 1
        if fills > ranks - 1:
            raise AssertionError("one-source filling exceeded R-1 fills")
    result = tuple(tuple(row) for row in matrix)
    ref.validate_allocation(result, counts, target)
    if (
        max((len(items) for items in ref.redundant_sets(result, homes)), default=0)
        > experts_per_rank
    ):
        raise AssertionError("constructive allocation violated the E/R bound")
    return result


def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def exact_min_redundancy(
    counts: Sequence[int], homes: Sequence[int], ranks: int
) -> tuple[int, ref.Matrix]:
    """Brute-force the lexicographic primary objective for small instances."""
    _equal_home_partition(homes, ranks)
    if sum(counts) % ranks:
        raise ValueError("total routes must divide evenly across ranks")
    target = sum(counts) // ranks
    best_value: int | None = None
    best_matrix: ref.Matrix | None = None

    def search(expert: int, rows: list[tuple[int, ...]], loads: list[int]) -> None:
        nonlocal best_value, best_matrix
        if expert == len(counts):
            if any(load != target for load in loads):
                return
            candidate = tuple(rows)
            value = max(
                (len(items) for items in ref.redundant_sets(candidate, homes)),
                default=0,
            )
            if best_value is None or value < best_value:
                best_value, best_matrix = value, candidate
            return
        for row in _compositions(counts[expert], ranks):
            next_loads = [load + value for load, value in zip(loads, row, strict=True)]
            if any(load > target for load in next_loads):
                continue
            search(expert + 1, [*rows, row], next_loads)

    search(0, [], [0] * ranks)
    if best_value is None or best_matrix is None:
        raise ValueError("no perfectly balanced integer allocation exists")
    return best_value, best_matrix
