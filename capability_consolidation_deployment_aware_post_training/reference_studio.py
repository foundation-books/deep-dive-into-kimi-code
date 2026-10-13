"""Complete studio fixture for the Chapter 23 reference module."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from .reference_core import (
    acceptance_mass,
    lk_loss,
    recurrent_inputs,
    route_teacher,
    studio_teacher_portfolio,
    total_variation,
)


def studio_handoff() -> dict[str, object]:
    """Build the deterministic handoff teaching fixture."""
    portfolio = studio_teacher_portfolio()
    teacher = route_teacher("studio-agent-001", "high")
    p = (Fraction(4, 5), Fraction(1, 5))
    q = (Fraction(11, 20), Fraction(9, 20))
    q_quantized = (Fraction(12, 25), Fraction(13, 25))
    target_draft = (Fraction(3, 5), Fraction(3, 10), Fraction(1, 10))
    draft = (Fraction(1, 2), Fraction(2, 5), Fraction(1, 10))
    proposals = ("blue", "square", "is", "below", "the", "red", "circle")
    payload = {
        "case_id": "studio-agent-001",
        "cell": ["agent", "high"],
        "teacher": teacher.checkpoint,
        "portfolio_cells": len(portfolio),
        "teacher_distribution": [str(value) for value in p],
        "student_distribution": [str(value) for value in q],
        "quantized_student_distribution": [str(value) for value in q_quantized],
        "greedy_parity_fixture": {
            "rollout": "below",
            "training": "above",
            "status": "decision_mismatch",
        },
        "draft": {
            "steps": 7,
            "inputs": list(recurrent_inputs("<start>", proposals)),
            "outputs": list(proposals),
            "acceptance_mass": str(acceptance_mass(target_draft, draft)),
            "total_variation": str(total_variation(target_draft, draft)),
            "lk_loss": lk_loss(target_draft, draft),
            "target_trainable": False,
            "draft_trainable": True,
            "fusion_trainable": True,
        },
        "evidence": {
            "portfolio_structure": "reported",
            "route": "deduced",
            "probabilities": "fixture",
            "arithmetic": "checked",
            "model_campaigns": "pending",
            "private_implementation": "unknown",
        },
        "measured_model_outcomes": "pending",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "content_sha256": hashlib.sha256(encoded).hexdigest()}
