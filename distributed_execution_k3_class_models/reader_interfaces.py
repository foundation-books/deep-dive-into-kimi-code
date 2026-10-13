"""Finite Chapter 38 interface witnesses, independent of a trainable model.

All inputs appear in the manuscript. Expanded attention tags test permutations,
assigned scalar derivatives test objective/owner accounting, and the two-message
placement example has a separate scope from the supplied resource cost card.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .topology import Coordinate, axis_group, encode


def token_origin(document: int, position: int, data: int = 1, stage: int = 0) -> int:
    """Return the t=1 world origin for one of the eight declared target rows."""
    if document not in (0, 1) or position not in range(1, 5):
        raise ValueError("the fixture has two length-four documents")
    return encode(Coordinate(data, stage, (position - 1) // 2, document, 1))


def cp_reconciliation_groups(routed_e0: bool = False) -> tuple[tuple[int, ...], ...]:
    """Derive all p=0,t=1 CP pairs; the second rank is the destination."""
    return tuple(
        axis_group(encode(Coordinate(data, 0, 0, expert, 1)), "c").members
        for data in (0, 1)
        for expert in ((0,) if routed_e0 else (0, 1))
    )


def layer_assignment(layer: int) -> tuple[int, int, int]:
    """Return logical chunk, physical stage, and virtual chunk for layer 1..8."""
    if layer not in range(1, 9):
        raise ValueError("decoder layer must be in 1..8")
    chunk = (layer - 1) // 2
    return chunk, chunk % 2, chunk // 2


def depth_boundary_sources(chunk: int) -> dict[str, tuple[int, ...]]:
    """Return contributor sets carried with h_k; X0 is a distinct source."""
    if chunk not in range(3):
        raise ValueError("only chunks 0,1,2 have a following PP boundary")
    if chunk == 0:
        return {"X0": (), "P1": tuple(range(1, 5))}
    if chunk == 1:
        return {"X0": (), "b1": tuple(range(1, 9))}
    return {"X0": (), "b1": tuple(range(1, 9)), "P2": tuple(range(9, 13))}


@dataclass(frozen=True, order=True)
class AttentionTag:
    """Immutable expanded operand coordinate, not a compact latent record."""

    document: int
    position: int
    head: int
    feature: int


def attention_tags(document: int) -> tuple[AttentionTag, ...]:
    """Enumerate every position/head/feature coordinate for one document."""
    if document not in (0, 1):
        raise ValueError("document must be 0 or 1")
    return tuple(
        AttentionTag(document, position, head, feature)
        for position in range(1, 5)
        for head in (0, 1)
        for feature in (0, 1)
    )


def redistribute_attention(
    records: Mapping[int, Sequence[AttentionTag]], document: int, *, by_head: bool
) -> dict[int, tuple[AttentionTag, ...]]:
    """Apply the complete sequence/head permutation or its inverse at layer 4."""
    ranks = tuple(encode(Coordinate(1, 1, context, document, 1)) for context in (0, 1))
    flattened = tuple(tag for tags in records.values() for tag in tags)
    if set(records) != set(ranks) or sorted(flattened) != list(
        attention_tags(document)
    ):
        raise ValueError("attention permutation has a missing, duplicate, or wrong tag")
    output: dict[int, list[AttentionTag]] = {rank: [] for rank in ranks}
    for tag in flattened:
        coordinate = tag.head if by_head else (tag.position - 1) // 2
        output[ranks[coordinate]].append(tag)
    return {rank: tuple(sorted(tags)) for rank, tags in output.items()}


def mla_fixture(document: int) -> dict[str, object]:
    """Check Q/K/V exchange, output inverse, causal keys, and remote payload."""
    ranks = tuple(encode(Coordinate(1, 1, context, document, 1)) for context in (0, 1))
    before = {
        rank: tuple(
            tag for tag in attention_tags(document) if (tag.position - 1) // 2 == c
        )
        for c, rank in enumerate(ranks)
    }
    after = redistribute_attention(before, document, by_head=True)
    restored = redistribute_attention(after, document, by_head=False)
    remote = sum(
        rank != ranks[tag.head] for rank, tags in before.items() for tag in tags
    )
    inverse_remote = sum(
        rank != ranks[(tag.position - 1) // 2]
        for rank, tags in after.items()
        for tag in tags
    )
    causal_keys = {
        (document, query, head): tuple(
            (document, key, head) for key in range(1, query + 1)
        )
        for query in range(1, 5)
        for head in (0, 1)
    }
    return {
        "before": before,
        "after": after,
        "restored": restored,
        "causal_keys": causal_keys,
        "remote_qkv_scalars": 3 * remote,
        "remote_output_scalars": inverse_remote,
    }


def gradient_from_local_means(entries: Sequence[tuple[Fraction, int]]) -> Fraction:
    """Weight disjoint local means by their target masses, including zeros."""
    if not entries or any(mass < 0 for _, mass in entries):
        raise ValueError("local masses must be nonnegative")
    normalizer = sum(mass for _, mass in entries)
    if normalizer == 0:
        raise ValueError("the global target mass must be positive")
    return (
        sum((gradient * mass for gradient, mass in entries), Fraction(0)) / normalizer
    )


def scalar_owner_witness(temporary: bool = True) -> dict[str, object]:
    """Accumulate assigned raw derivatives, close CP/replicas, update once."""
    context_partials: dict[tuple[int, int, int], tuple[Fraction, Fraction]] = {}
    replica_numerators: list[Fraction] = []
    for data in (0, 1):
        accumulator = Fraction(0)
        for microbatch in range(4):
            # Each column of the printed mean table splits equally across CP.
            column_half_numerator = Fraction(8 * (data + microbatch + 1), 2)
            for context in (0, 1):
                home = column_half_numerator
                returned = column_half_numerator
                if not temporary:
                    home += returned
                    returned = Fraction(0)
                context_partials[data, microbatch, context] = home, returned
                accumulator += home + returned
        replica_numerators.append(accumulator)
    normalizer = 2 * 4 * 2 * 4
    gradient = sum(replica_numerators, Fraction(0)) / normalizer
    updated = Fraction(10) - Fraction(1, 10) * gradient
    return {
        "context_partials": context_partials,
        "replica_numerators": tuple(replica_numerators),
        "normalizer": normalizer,
        "gradient": gradient,
        "updated": updated,
        "owner": 5,
        "update_count": 1,
        "permanent_revisions": {rank: 8 for rank in (1, 5, 17, 21)},
        "invalidated_temporary_ranks": (3, 7, 19, 23),
    }


@dataclass(frozen=True)
class PlacementMessage:
    """One explicitly directed transfer in the isolated placement inventory."""

    message_id: str
    sender: str
    receiver: str
    payload: int
    multiplicity: int = 1


PLACEMENT_MESSAGES = (
    PlacementMessage("M1", "A", "B", 14),
    PlacementMessage("M2", "C", "D", 4),
)


def placement_witness(
    alternative: bool = False,
) -> tuple[int, dict[tuple[int, int], int]]:
    """Compute one-way payload and each directed inter-node load from inputs."""
    nodes = {"A": 0, "B": int(alternative), "C": 0, "D": 1}
    if any(sum(node == n for node in nodes.values()) > 4 for n in (0, 1)):
        raise ValueError("placement exceeds four slots per node")
    links = {(0, 1): 0, (1, 0): 0}
    logical = 0
    for message in PLACEMENT_MESSAGES:
        payload = message.payload * message.multiplicity
        logical += payload
        source, target = nodes[message.sender], nodes[message.receiver]
        if source != target:
            links[source, target] += payload
    return logical, links
