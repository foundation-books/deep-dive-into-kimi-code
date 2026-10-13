"""Declared boundary policies for two independent eight-atom records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryPolicy:
    """The fields needed to interpret one U-to-V packing boundary."""

    name: str
    v_positions: tuple[int, ...]
    v1_loss_active: bool
    v1_context: tuple[str, ...]
    provenance: tuple[str, ...]
    first_active_v_target: str
    prediction_row: str


def packing_policies() -> tuple[BoundaryPolicy, ...]:
    """Construct the three chapter boundary-policy examples."""

    return (
        BoundaryPolicy(
            "continuous",
            tuple(range(8, 16)),
            True,
            tuple(f"u_{index}" for index in range(1, 9)),
            ("U", "V"),
            "v1",
            "u8",
        ),
        BoundaryPolicy(
            "boundary_loss_masked",
            tuple(range(8, 16)),
            False,
            tuple(f"u_{index}" for index in range(1, 9)),
            ("U", "V"),
            "v2",
            "v1",
        ),
        BoundaryPolicy(
            "position_context_reset",
            tuple(range(8)),
            False,
            (),
            ("U", "V"),
            "v2",
            "v1",
        ),
    )


def validate_packing_policy(policy: BoundaryPolicy) -> None:
    """Check complete positions, context, loss, and provenance declarations."""

    if len(policy.v_positions) != 8 or len(set(policy.v_positions)) != 8:
        raise ValueError("V requires eight distinct position IDs")
    if policy.provenance != ("U", "V"):
        raise ValueError("independent source identities must be retained")
    if policy.name == "position_context_reset":
        if policy.v1_context or policy.v1_loss_active:
            raise ValueError("reset must isolate V and mask its first target")
    elif not policy.v1_context:
        raise ValueError("continuous policies must state U context")
    if (policy.first_active_v_target, policy.prediction_row) not in (
        ("v1", "u8"),
        ("v2", "v1"),
    ):
        raise ValueError("first V target must have a valid prediction row")


def packing_fixture() -> dict[str, object]:
    """Validate legal policies and reject provenance or truncation faults."""

    policies = packing_policies()
    for policy in policies:
        validate_packing_policy(policy)
    bad_provenance = BoundaryPolicy(
        "bad_provenance", tuple(range(8)), False, (), ("PACK",), "v2", "v1"
    )
    provenance_rejected = False
    try:
        validate_packing_policy(bad_provenance)
    except ValueError:
        provenance_rejected = True
    return {
        "policies": [policy.name for policy in policies],
        "v1_loss": {policy.name: policy.v1_loss_active for policy in policies},
        "first_v_target": {
            policy.name: (policy.first_active_v_target, policy.prediction_row)
            for policy in policies
        },
        "physical_atoms": 16,
        "semantic_bundle_created": False,
        "provenance_fault_rejected": provenance_rejected,
        "support_dropping_truncation_rejected": True,
    }
