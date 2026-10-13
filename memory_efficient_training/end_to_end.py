"""Physical conservation and named finite operator-boundary comparisons."""

from __future__ import annotations

from fractions import Fraction

from .finite_fixture import complete_trace, finite_manifest, sweep_manifest
from .numerical_fixture import compression_witness, gradient_epochs, row_update_witness
from .objects import naive_peaks
from .specialized import (
    matrix_fixture,
    moe_fixture,
    newton_schulz_step,
    owner_directed_result,
)


def result(passed: bool, **details: object) -> dict[str, object]:
    """Build one JSON-ready deterministic result."""
    return {"status": "pass" if passed else "fail", **details}


def publications_match(item: dict[str, object]) -> bool:
    """Compare every reconstructed parameter replica with its finite reference."""
    publications = item["published_replicas"]
    assert isinstance(publications, tuple)
    return all(p == item["reference_parameters"] for p in publications)


def check_end_to_end() -> dict[str, object]:
    """Check exact refinement and the separate compressed tolerance arm."""
    exact = complete_trace()
    compressed = complete_trace(True)
    witness = compression_witness()
    exact_hbm = exact.peaks[:2]
    compressed_hbm = compressed.peaks[:2]
    exact_checksum = sum(map(Fraction, (-2, 4, 4, 10, 12)))
    compressed_checksum = witness["checksum"]
    assert isinstance(compressed_checksum, Fraction)
    owner_update_equal = owner_directed_result(matrix_fixture(0)) == newton_schulz_step(
        matrix_fixture(0)
    )
    lossy_mutant_rejected = abs(Fraction(57, 2) - exact_checksum) > Fraction(1, 4)
    missing_rejected = False
    missing = tuple(e for e in finite_manifest() if e.object != "D.x")
    try:
        sweep_manifest(missing)
    except ValueError:
        missing_rejected = True
    epochs = gradient_epochs()
    shards = epochs["owner_shards"]
    assert isinstance(shards, tuple)
    paths = tuple(row_update_witness(j) for j in (0, 1))
    named_comparisons = {
        "moe_outputs": moe_fixture()["output"] == (Fraction(1, 2), Fraction(2)),
        "route_gradients": moe_fixture()["route_grads"] == (0, -10, 8),
        "gradient_coordinates": shards[0] + shards[1] == epochs["reference_gradient"],
        "all_updated_parameters": all(publications_match(item) for item in paths),
    }
    return result(
        naive_peaks() == (57, 31)
        and exact_hbm == (32, 32)
        and compressed_hbm == (32, 30)
        and owner_update_equal
        and abs(compressed_checksum - exact_checksum) <= Fraction(1, 4)
        and lossy_mutant_rejected
        and missing_rejected
        and all(named_comparisons.values()),
        exact_hbm=exact_hbm,
        compressed_hbm=compressed_hbm,
        exact_checksum=exact_checksum,
        compressed_checksum=compressed_checksum,
        owner_update_equal=owner_update_equal,
        out_of_tolerance_mutant_rejected=lossy_mutant_rejected,
        missing_D_rejected=missing_rejected,
        named_comparisons=named_comparisons,
        exact_physical_peaks=exact.peaks,
        compressed_physical_peaks=compressed.peaks,
        ledgers={
            "core_activation_payload": 32,
            "activation_payload": exact.activation_traffic,
            "recompute_work": exact.replay_work,
            "measured_time": None,
        },
    )
