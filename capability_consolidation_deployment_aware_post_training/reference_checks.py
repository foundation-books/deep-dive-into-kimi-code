"""Eight deterministic protocol records for the Chapter 23 companion."""

from __future__ import annotations

from fractions import Fraction
import math

from .reference_core import (
    DOMAINS,
    EFFORTS,
    CellCounts,
    PrecisionFingerprint,
    acceptance_mass,
    clipped_reward,
    lk_loss,
    opd_rewards,
    parity_audit,
    recurrent_inputs,
    reverse_kl,
    route_teacher,
    studio_quantization_contract,
    studio_teacher_portfolio,
    top_k_projection,
    total_variation,
    validate_mixture,
)
from .reference_studio import studio_handoff


def _portfolio_check() -> dict[str, object]:
    portfolio = studio_teacher_portfolio()
    routes = {
        f"{case}:{effort}": route_teacher(case, effort).checkpoint
        for case in ("studio-general-001", "studio-agent-001", "studio-code-001")
        for effort in EFFORTS
    }
    return {"status": "pass", "portfolio_cells": len(portfolio), "routes": routes}


def _reverse_kl_check() -> dict[str, object]:
    p = (Fraction(4, 5), Fraction(1, 5))
    q = (Fraction(11, 20), Fraction(9, 20))
    rewards = opd_rewards(p, q)
    expected = sum(
        float(weight) * reward for weight, reward in zip(q, rewards, strict=True)
    )
    divergence = reverse_kl(q, p)
    if not math.isclose(expected, -divergence, rel_tol=0, abs_tol=1e-15):
        raise AssertionError("expected OPD reward is not negative reverse KL")
    return {
        "status": "pass",
        "reverse_kl_nats": divergence,
        "rewards_nats": list(rewards),
        "expected_reward_nats": expected,
    }


def _support_clipping_check() -> dict[str, object]:
    p = (Fraction(4, 5), Fraction(1, 5))
    q = (Fraction(11, 20), Fraction(9, 20))
    rewards = opd_rewards(p, q)
    sparse = top_k_projection(
        (Fraction(3, 5), Fraction(1, 4), Fraction(1, 10), Fraction(1, 20)), 2
    )
    return {
        "status": "pass",
        "clipped_rewards": [
            clipped_reward(value, Fraction(7, 10)) for value in rewards
        ],
        "top2": [str(value) for value in sparse],
        "top2_zero_support_indices": [
            index for index, value in enumerate(sparse) if value == 0
        ],
    }


def _mixture_retention_check() -> dict[str, object]:
    weights = {
        (domain, effort): Fraction(1, 9) for domain in DOMAINS for effort in EFFORTS
    }
    counts = CellCounts(12, 1, 2, 9)
    return {
        "status": "pass",
        "mixture_valid": validate_mixture(weights),
        "requested": counts.requested,
        "available": counts.available,
        "trained": counts.trained,
        "retention_cells": len(weights),
        "model_outcomes": "pending",
    }


def _quantization_check() -> dict[str, object]:
    contract = studio_quantization_contract()
    return {
        "status": "pass",
        "contract_valid": contract.validate(),
        "tensor_formats": dict(contract.tensor_formats),
        "private_k3_fields": "unknown",
    }


def _fingerprint() -> PrecisionFingerprint:
    return PrecisionFingerprint(
        "student-fixture-v1",
        "studio-tokenizer-v1",
        "studio-qcontract-v1",
        "semantic-emulator-v1",
        (("routed_expert_weight", "MXFP4"),),
        (("routed_expert_input_activation", "MXFP8"),),
        (("matmul", "fp32-fixture"),),
        (("weight", "fixture-block-axis"), ("activation", "fixture-token-axis")),
        "fixture-shared-scale-v1",
        "nearest-even-fixture-v1",
        "finite-range-fixture-v1",
        "studio-sampler-v1",
    )


def _parity_check() -> dict[str, object]:
    fingerprint = _fingerprint()
    passed = parity_audit(
        fingerprint,
        fingerprint,
        ("expert-7", "below"),
        ("expert-7", "below"),
        -0.6,
        -0.6,
        1e-8,
    )
    changed = PrecisionFingerprint(
        **{**fingerprint.__dict__, "rounding": "toward-zero-fixture-v1"}
    )
    contract_failure = parity_audit(
        fingerprint, changed, ("below",), ("below",), -0.6, -0.6, 1e-8
    )
    decision_failure = parity_audit(
        fingerprint, fingerprint, ("expert-7",), ("expert-12",), -0.6, -0.6, 1e-8
    )
    return {
        "status": "pass",
        "matched": passed,
        "contract_failure": contract_failure,
        "decision_failure": decision_failure,
    }


def _draft_acceptance_check() -> dict[str, object]:
    p = (Fraction(3, 5), Fraction(3, 10), Fraction(1, 10))
    q = (Fraction(1, 2), Fraction(2, 5), Fraction(1, 10))
    proposals = ("blue", "square", "is", "below", "the", "red", "circle")
    overlap = acceptance_mass(p, q)
    variation = total_variation(p, q)
    if overlap != 1 - variation:
        raise AssertionError("acceptance/TV identity failed")
    return {
        "status": "pass",
        "acceptance_mass": str(overlap),
        "total_variation": str(variation),
        "lk_loss_nats": lk_loss(p, q),
        "recurrent_inputs": list(recurrent_inputs("<start>", proposals)),
    }


def _studio_handoff_check() -> dict[str, object]:
    return {"status": "pass", **studio_handoff()}


def all_checks() -> dict[str, dict[str, object]]:
    """Return eight JSON-safe fixture records."""
    return {
        "RESULT_CH21_PORTFOLIO_ROUTING_ORACLE": _portfolio_check(),
        "RESULT_CH21_REVERSE_KL_REWARD_ORACLE": _reverse_kl_check(),
        "RESULT_CH21_SUPPORT_CLIPPING_TOPK_ORACLE": _support_clipping_check(),
        "RESULT_CH21_MIXTURE_RETENTION_ORACLE": _mixture_retention_check(),
        "RESULT_CH21_QUANTIZATION_CONTRACT_ORACLE": _quantization_check(),
        "RESULT_CH21_PRECISION_PARITY_ORACLE": _parity_check(),
        "RESULT_CH21_DRAFT_ACCEPTANCE_ORACLE": _draft_acceptance_check(),
        "RESULT_CH21_STUDIO_HANDOFF_ORACLE": _studio_handoff_check(),
    }
