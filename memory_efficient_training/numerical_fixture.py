"""Printed rational worksheets and complete row-update/publication witnesses."""

from __future__ import annotations

from fractions import Fraction

from .finite_fixture import INPUTS
from .policies import max_error, quantize
from .specialized import (
    Matrix,
    assemble,
    matrix_fixture,
    newton_schulz_step,
    subtract,
)


def replay_panel(weight: int = 5, mask: tuple[int, ...] = (1, 0, 1)) -> tuple[int, ...]:
    """The printed no-rescaling elementwise/dropout replay panel."""
    return tuple(m * (weight * x + 1) for x, m in zip((2, 3, 4), mask, strict=True))


def compression_witness() -> dict[str, object]:
    """New four-record fixture; the older assigned checksum is superseded."""
    decoded = quantize(INPUTS, Fraction(1, 8))
    gradients = tuple(Fraction(3, 2) * x for x in INPUTS)
    reconstructed = tuple(Fraction(3, 2) * x for x in decoded)
    errors = tuple(b - a for a, b in zip(gradients, reconstructed, strict=True))
    base = (Fraction(-2), Fraction(4), Fraction(4), Fraction(10), Fraction(12))
    changed = tuple(a + b for a, b in zip(base[:4], errors, strict=True)) + base[4:]
    return {
        "inputs": INPUTS,
        "codes": tuple(int(x * 8) for x in decoded),
        "decoded": decoded,
        "gradients": gradients,
        "reconstructed_gradients": reconstructed,
        "gradient_deltas": errors,
        "activation_error": max_error(INPUTS, decoded),
        "gradient_error": max_error(gradients, reconstructed),
        "components": changed,
        "checksum": sum(changed, Fraction(0)),
    }


def row_update_witness(matrix_id: int) -> dict[str, object]:
    """Return every row's direction, once-only update, and both publications."""
    candidate = matrix_fixture(matrix_id)
    reference_direction = newton_schulz_step(candidate)
    assembled = assemble((candidate[:2], candidate[2:]), ((0, 2), (2, 4)))
    owner_direction = newton_schulz_step(assembled)
    parameters: Matrix = tuple(
        tuple(Fraction(int(i == j)) for j in range(4)) for i in range(4)
    )
    reference_update = subtract(parameters, reference_direction)
    # D0 owns rows 0:2 of BOTH matrices; D1 owns rows 2:4 of BOTH.
    directions = (owner_direction[:2], owner_direction[2:])
    updated = tuple(
        subtract(parameters[start:stop], directions[d])
        for d, (start, stop) in enumerate(((0, 2), (2, 4)))
    )
    published = assemble(updated, ((0, 2), (2, 4)))
    return {
        "assembly_owner": matrix_id,
        "direction": owner_direction,
        "reference_direction": reference_direction,
        "row_owners": (0, 0, 1, 1),
        "update_count": (1, 1, 1, 1),
        "updated_rows": updated,
        "published_replicas": (published, published),
        "reference_parameters": reference_update,
        "traffic_elements_per_rank": {
            "candidate_ingress": 8,
            "direction_delivery": 8,
            "parameter_publication": 16,
        },
    }


def gradient_epochs() -> dict[str, object]:
    """Eight full input coordinates per replica and four outputs per owner."""
    replicas = (tuple(range(1, 17)), tuple(10 * i for i in range(1, 17)))
    coordinates = ((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15))
    outputs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    owners: list[list[int]] = [[], []]
    receipts = [0] * 16
    for indices in coordinates:
        inputs = tuple(tuple(row[i] for i in indices) for row in replicas)
        summed = tuple(sum(column) for column in zip(*inputs, strict=True))
        outputs.append((summed[:4], summed[4:]))
        for d in (0, 1):
            owners[d].extend(outputs[-1][d])
        for coordinate in indices:
            receipts[coordinate] += 1
    return {
        "replicas": replicas,
        "epoch_coordinates": coordinates,
        "epoch_outputs": tuple(outputs),
        "owner_shards": tuple(tuple(row) for row in owners),
        "reference_gradient": tuple(11 * i for i in range(1, 17)),
        "receipt_count": tuple(receipts),
    }
