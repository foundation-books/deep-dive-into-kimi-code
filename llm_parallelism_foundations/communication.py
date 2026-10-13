"""Constructed communication examples; no transport or autograd backend."""

from collections.abc import Mapping, Sequence
from fractions import Fraction as F

from distributed_execution_k3_class_models.semantics import (
    AffineFragment,
    compose_fragments,
)

from .collectives import Vector, all_gather, all_to_all, reduce_scatter
from .reference import Route, combine, routes

GROUP = (0, 1, 2, 3)
X = tuple(map(F, range(1, 9)))


def four_rank_partials() -> dict[int, Vector]:
    """Each rank contributes to the same eight coordinates."""
    return {r: tuple(10**r * value for value in X) for r in GROUP}


def tagged_gather(
    shards: Mapping[int, Sequence[tuple[int, F]]],
    group: Sequence[int],
    coordinates: Sequence[int],
) -> Vector:
    """Check the ordered coordinate cover independently of payload values."""
    gathered = all_gather(shards, group)[group[0]]
    if tuple(index for index, _ in gathered) != tuple(coordinates):
        raise ValueError("missing, duplicate, or misordered coordinate")
    return tuple(value for _, value in gathered)


def depth_mix(a: F, b: F) -> F:
    """A rational nonlinear row mixer, not the AttnRes operator."""
    if a + b <= 0:
        raise ValueError("the fixture requires positive source sum")
    return (a * a + b * b) / (a + b)


def depth_sandwich() -> tuple[Vector, Vector]:
    """Reduce whole token rows, mix locally, and gather only the row outputs."""
    # Pack source pairs by token so contiguous reduce-scatter slices own rows.
    partials = {
        r: tuple(v for t in X for v in ((r + 1) * t, (4 - r) * t)) for r in GROUP
    }
    reduced = reduce_scatter(partials, GROUP)
    local = {
        r: tuple(depth_mix(values[i], values[i + 1]) for i in (0, 2))
        for r, values in reduced.items()
    }
    correct = all_gather(local, GROUP)[0]
    premature = tuple(
        sum((depth_mix((r + 1) * t, (4 - r) * t) for r in GROUP), F(0)) for t in X
    )
    return correct, premature


def allgather_pullback(
    consumer_gradients: Mapping[int, Vector], group: Sequence[int], divisor: int = 1
) -> dict[int, Vector]:
    """Differentiate summed downstream losses; divisor declares objective scaling."""
    if divisor <= 0:
        raise ValueError("objective divisor must be positive")
    return {
        rank: tuple(value / divisor for value in shard)
        for rank, shard in reduce_scatter(consumer_gradients, group).items()
    }


def affine_segments() -> tuple[AffineFragment, ...]:
    """Four scalar summaries in document order, distinct from KDA kernels."""
    return tuple(
        AffineFragment(i, F(a), F(b))
        for i, (a, b) in enumerate(((2, 1), (3, 4), (1, -2), (2, 0)))
    )


def exclusive_prefixes(
    fragments: Sequence[AffineFragment], incoming: F = F(0)
) -> tuple[Vector, F]:
    """An empty prefix is identity; arrival order cannot replace segment order."""
    if tuple(f.segment for f in fragments) != tuple(range(len(fragments))):
        raise ValueError("fragments must cover the declared segment order")
    states = tuple(
        compose_fragments(fragments[:i]).apply(incoming) for i in range(len(fragments))
    )
    return states, compose_fragments(fragments).apply(incoming)


def irregular_dispatch() -> dict[int, tuple[Route, ...]]:
    """Reuse A/B but change placement: expert homes are 0, 0, 2, 1."""
    homes = {1: 0, 2: 0, 3: 2, 4: 1}
    outgoing = {
        origin: {
            dest: tuple(
                r for r in routes() if r.origin == origin and homes[r.expert] == dest
            )
            for dest in GROUP
        }
        for origin in GROUP
    }
    return all_to_all(outgoing, GROUP)


def irregular_round_trip() -> dict[tuple[int, str], F]:
    """Return whole tagged records, including empty buckets, before weighting."""
    destinations = irregular_dispatch()
    outgoing = {
        home: {
            origin: tuple(
                (r, r.expert * r.value)
                for r in destinations[home]
                if r.origin == origin
            )
            for origin in GROUP
        }
        for home in GROUP
    }
    returned = all_to_all(outgoing, GROUP)
    return combine([item for origin in GROUP for item in returned[origin]], routes())


def transfer_microseconds(payload_bytes: int) -> F:
    """Toy one-transfer model: 3 us startup and 20 decimal GB/s bandwidth."""
    if payload_bytes < 0:
        raise ValueError("payload cannot be negative")
    return F(3) + F(payload_bytes, 20_000)


def main() -> None:
    """Print checked manuscript values; these are not measurements."""
    print("Reduced shards:", reduce_scatter(four_rank_partials(), GROUP))
    print("Depth outputs / premature mixing:", depth_sandwich())
    print("Exclusive prefixes / final state:", exclusive_prefixes(affine_segments()))
    print("Destination counts:", tuple(map(len, irregular_dispatch().values())))
    print("Restored tokens:", irregular_round_trip())
    print(
        "Toy transfer us:",
        transfer_microseconds(4000),
        transfer_microseconds(4_000_000),
    )


if __name__ == "__main__":
    main()
