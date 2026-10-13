"""Declared boundary between a typed local outcome and a numeric reward API."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from .math_investigation import adapt_math_result
from .reference_core import VerifierResult
from .reference_studio import RewardHandoff


@dataclass(frozen=True)
class AdmissionDecision:
    """Exactly one numeric result or a nonnumeric group action."""

    numeric_result: dict[str, object] | None
    group_action: str


def callback_view(
    sequences: Sequence[str],
    *,
    extra_info: Mapping[str, object],
    tool_fields: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Model the last-sequence callback view and explicit side-channel fields.

    Field transport does not authenticate a witness; the caller must enforce
    provenance before treating any field as evidence.
    """
    if not sequences:
        raise ValueError("a reward callback needs at least one sequence")
    merged = dict(extra_info)
    if tool_fields is not None:
        merged.update(tool_fields)
    return sequences[-1], merged


def route_handoff(handoff: RewardHandoff) -> AdmissionDecision:
    """Emit a number only after the chapter's existing slot policy admits it."""
    handoff.validate()
    if not handoff.training_eligible:
        return AdmissionDecision(None, handoff.group_action)
    assert handoff.scalar is not None
    return AdmissionDecision(
        {
            "reward_score": float(handoff.scalar),
            "reward_extra_info": {
                "exact_score": str(handoff.scalar),
                "candidate": handoff.candidate,
                "group_id": handoff.group_id,
                "slot_id": handoff.slot_id,
                "evidence_disposition": handoff.evidence_disposition,
                "scalar_purpose": handoff.scalar_purpose,
            },
        },
        handoff.group_action,
    )


def bounded_rational_check(
    candidate: str,
    reference: str,
    *,
    timeout_seconds: float,
    delay_seconds: float = 0,
) -> VerifierResult:
    """Run a toy verifier in a process and preserve timeout/error origin.

    The worker uses standard-library Fraction, not Math-Verify. Its process
    boundary demonstrates where a signal-based verifier could be isolated.
    """
    if timeout_seconds <= 0 or delay_seconds < 0:
        raise ValueError("timeouts and delays need declared positive bounds")
    root = Path(__file__).resolve().parents[1]
    request = json.dumps(
        {"candidate": candidate, "reference": reference, "delay_seconds": delay_seconds}
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "rewards_verifiers_agentic_grm.math_worker"],
            input=request,
            text=True,
            capture_output=True,
            cwd=root,
            timeout=timeout_seconds,
            check=True,
        )
        payload = json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return adapt_math_result(
            None, reference_valid=True, candidate_valid=True, error="worker timeout"
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        return adapt_math_result(
            None,
            reference_valid=True,
            candidate_valid=True,
            error=f"worker error: {type(error).__name__}",
        )
    if payload.get("invalid") == "reference":
        return adapt_math_result(None, reference_valid=False, candidate_valid=True)
    if payload.get("invalid") == "candidate":
        return adapt_math_result(None, reference_valid=True, candidate_valid=False)
    if "accepted" not in payload or not isinstance(payload["accepted"], bool):
        return adapt_math_result(
            None,
            reference_valid=True,
            candidate_valid=True,
            error="worker returned no Boolean",
        )
    return adapt_math_result(
        payload["accepted"], reference_valid=True, candidate_valid=True
    )
