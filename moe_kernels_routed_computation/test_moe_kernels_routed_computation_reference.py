"""Tests for the Chapter 35 dependency-free semantic reference."""

from __future__ import annotations

import importlib

from . import checks, faults, reference as ref


def test_grouped_backward_derivatives() -> None:
    from .backward_checks import check_all

    check_all()


def test_package_exports_only_the_declared_reference_api() -> None:
    package = importlib.import_module(__package__ or "")
    assert package.RouteRecord is ref.RouteRecord
    for implementation_name in (
        "Fraction",
        "Mapping",
        "Sequence",
        "dataclass",
        "exp",
        "replace",
        "sqrt",
        "tanh",
    ):
        assert not hasattr(package, implementation_name)


def test_exact_fixture_has_twenty_unique_routes() -> None:
    records = ref.toy_routes()
    assert len(records) == 20
    assert len({record.identity for record in records}) == 20
    assert all(
        sum((route.weight for route in records if route.origin == origin), ref.f(0))
        == 1
        for origin in ref.ORIGINS
    )


def test_selection_bias_changes_membership_not_execution_weights() -> None:
    selected = ref.selected_weights(
        (ref.f(1), ref.f(2), ref.f(3), ref.f(4)),
        (ref.f(10), ref.f(0), ref.f(0), ref.f(0)),
        2,
    )
    assert selected == ((1, ref.f(1, 5)), (4, ref.f(4, 5)))


def test_routed_branch_is_normalized_and_joined_at_model_width() -> None:
    joined = ref.stable_latent_join(
        (3.0, 4.0, 0.0, 0.0),
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        (1.0, 1.0, 1.0, 1.0),
        0.0,
        ((1.0,) * 8,),
    )
    assert joined == (2.2, 2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_count_scan_and_stable_permutation_round_trip() -> None:
    records = ref.toy_routes()
    rows = tuple((ref.f(index),) for index in range(20))
    dispatch = ref.stable_dispatch(records, rows, 4)
    assert dispatch.loads == ref.EXPECTED_LOADS
    assert dispatch.offsets == ref.EXPECTED_OFFSETS
    assert ref.restore_logical(dispatch.rows, dispatch) == rows


def test_padding_masks_exclude_four_rows() -> None:
    masks = ref.padding_mask(ref.EXPECTED_LOADS, 4)
    assert tuple(len(mask) for mask in masks) == (8, 8, 4, 4)
    assert sum(sum(mask) for mask in masks) == 20


def test_empty_expert_has_an_empty_segment_and_no_padding() -> None:
    loads, offsets = ref.loads_and_offsets(ref.toy_routes(), 5)
    assert loads == (7, 6, 4, 3, 0)
    assert offsets == (0, 7, 13, 17, 20, 20)
    assert ref.padding_mask(loads, 4)[-1] == ()


def test_grouped_situ_matches_independent_evaluation() -> None:
    independent, grouped = ref.grouped_situ_fixture()
    assert independent == grouped
    assert grouped[-1] == (0.0, 0.0)


def test_diagnostic_restore_and_token_schedule_are_exact() -> None:
    expert, checksum = ref.diagnostic_trace()
    token = ref.token_centric_diagnostic(ref.toy_routes())
    assert expert == token
    assert checksum == ref.EXPECTED_CHECKSUM


def test_backward_replay_and_fraction_gradients() -> None:
    records = ref.toy_routes()
    rows = tuple((ref.f(index),) for index in range(20))
    dispatch = ref.stable_dispatch(records, rows, 4)
    assert ref.replay_dispatch(dispatch) == dispatch
    assert ref.scalar_backward(
        ref.f(2), (ref.f(1, 3), ref.f(2, 3)), (ref.f(3), ref.f(-1))
    ) == {
        "output": ref.f(2, 3),
        "dz": ref.f(2, 9),
        "dp": (ref.f(4), ref.f(-4, 3)),
        "dw": (ref.f(4, 9), ref.f(8, 9)),
    }


def test_all_fault_mutants_are_detected() -> None:
    witnesses = faults.fault_witnesses()
    assert tuple(witnesses) == faults.FAULT_IDS
    assert all(witnesses.values())


def test_evidence_surface_has_exactly_ten_passing_checks() -> None:
    outcomes = checks.all_checks()
    assert len(outcomes) == 10
    assert all(record["status"] == "pass" for record in outcomes.values())
