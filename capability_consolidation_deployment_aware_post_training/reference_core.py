"""Dependency-free Chapter 23 consolidation and deployment oracles.

All numerical values are chapter-created fixtures.  The module neither trains
nor evaluates a model and does not reconstruct a private Kimi implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Mapping, Sequence


DOMAINS = ("general", "agent", "code")
EFFORTS = ("low", "high", "max")
EVIDENCE_STATUSES = frozenset(
    {"reported", "deduced", "fixture", "checked", "measured", "pending", "unknown"}
)


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a probability or scalar")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("numeric value must be finite")
    return Fraction(str(value))


def probability_vector(
    values: Sequence[int | float | Fraction],
) -> tuple[Fraction, ...]:
    """Normalize exact nonnegative weights into probabilities."""
    vector = tuple(_fraction(value) for value in values)
    if not vector or any(value < 0 for value in vector) or sum(vector) != 1:
        raise ValueError("probabilities must be nonnegative and sum exactly to one")
    return vector


@dataclass(frozen=True)
class TeacherRecord:
    """Identify the versioned teacher assigned to one domain-effort cell."""

    domain: str
    effort: str
    checkpoint: str
    tokenizer: str
    precision: str
    decoding: str
    evidence_status: str

    def validate(self) -> bool:
        if self.domain not in DOMAINS or self.effort not in EFFORTS:
            raise ValueError("teacher cell is outside the declared portfolio")
        if self.evidence_status not in EVIDENCE_STATUSES:
            raise ValueError("teacher evidence status is invalid")
        if not all((self.checkpoint, self.tokenizer, self.precision, self.decoding)):
            raise ValueError("teacher identity fields must be nonempty")
        return True


def studio_teacher_portfolio() -> dict[tuple[str, str], TeacherRecord]:
    """Return exactly one versioned teaching teacher for each of nine cells."""
    return {
        (domain, effort): TeacherRecord(
            domain,
            effort,
            f"teacher-{domain}-{effort}-fixture-v1",
            "studio-tokenizer-v1",
            "full-precision-fixture-v1",
            "studio-sampler-v1",
            "fixture",
        )
        for domain in DOMAINS
        for effort in EFFORTS
    }


def route_teacher(case_id: str, effort: str) -> TeacherRecord:
    """Resolve the validated teacher for a studio case and effort tier."""
    domains = {
        "studio-general-001": "general",
        "studio-agent-001": "agent",
        "studio-code-001": "code",
    }
    if case_id not in domains or effort not in EFFORTS:
        raise ValueError("case or effort has no declared route")
    teacher = studio_teacher_portfolio()[(domains[case_id], effort)]
    teacher.validate()
    return teacher


def reverse_kl(
    student: Sequence[int | float | Fraction],
    teacher: Sequence[int | float | Fraction],
) -> float:
    """Compute reverse KL divergence from student to teacher."""
    q, p = probability_vector(student), probability_vector(teacher)
    if len(q) != len(p):
        raise ValueError("teacher and student vocabularies differ")
    if any(q_i > 0 and p_i == 0 for q_i, p_i in zip(q, p, strict=True)):
        return math.inf
    return sum(
        float(q_i) * math.log(float(q_i / p_i))
        for q_i, p_i in zip(q, p, strict=True)
        if q_i
    )


def opd_rewards(
    teacher: Sequence[int | float | Fraction],
    student: Sequence[int | float | Fraction],
) -> tuple[float, ...]:
    """Compute on-policy distillation token rewards."""
    p, q = probability_vector(teacher), probability_vector(student)
    if len(p) != len(q):
        raise ValueError("teacher and student vocabularies differ")
    rewards: list[float] = []
    for p_i, q_i in zip(p, q, strict=True):
        if q_i == 0:
            rewards.append(math.nan)
        elif p_i == 0:
            rewards.append(-math.inf)
        else:
            rewards.append(math.log(float(p_i / q_i)))
    return tuple(rewards)


def clipped_reward(value: float, cap: int | float | Fraction) -> float:
    """Clip a finite reward, never an unresolved support failure."""
    limit = float(_fraction(cap))
    if not math.isfinite(limit) or limit <= 0 or not math.isfinite(value):
        raise ValueError("cap and reward must be in the declared domain")
    return max(-limit, min(limit, value))


def top_k_projection(
    values: Sequence[int | float | Fraction], k: int
) -> tuple[Fraction, ...]:
    """Project a probability vector onto its top-k support."""
    p = probability_vector(values)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(p):
        raise ValueError("k is outside the vocabulary")
    selected = sorted(range(len(p)), key=lambda index: (-p[index], index))[:k]
    mass = sum(p[index] for index in selected)
    return tuple(
        p[index] / mass if index in selected else Fraction(0) for index in range(len(p))
    )


@dataclass(frozen=True)
class CellCounts:
    """Track requested, rejected, truncated, and trained rows for one cell."""

    requested: int
    rejected: int
    truncated: int
    trained: int

    def validate(self) -> bool:
        values = (self.requested, self.rejected, self.truncated, self.trained)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("cell counts must be nonnegative integers")
        available = self.requested - self.rejected - self.truncated
        if available < 0 or self.trained > available:
            raise ValueError("cell count conservation failed")
        return True

    @property
    def available(self) -> int:
        self.validate()
        return self.requested - self.rejected - self.truncated


def validate_mixture(weights: Mapping[tuple[str, str], int | float | Fraction]) -> bool:
    """Validate the mixture contract."""
    expected = {(domain, effort) for domain in DOMAINS for effort in EFFORTS}
    if set(weights) != expected:
        raise ValueError("mixture must expose exactly the nine cells")
    normalized = tuple(_fraction(weights[cell]) for cell in sorted(expected))
    if any(value < 0 for value in normalized) or sum(normalized) != 1:
        raise ValueError("mixture weights must be nonnegative and sum to one")
    return True


@dataclass(frozen=True)
class QuantizationContract:
    """Describe deployment tensor formats and their implementation rules."""

    tensor_formats: tuple[tuple[str, str], ...]
    grouping: tuple[tuple[str, str], ...]
    scale_rule: str
    rounding: str
    clipping: str
    accumulator: str
    exceptions: tuple[str, ...]
    kernel: str
    version: str

    def validate(self) -> bool:
        required = {
            "routed_expert_weight",
            "routed_expert_input_activation",
            "attention_projection",
            "latent_moe_projection",
            "shared_expert",
            "moe_router",
        }
        roles = {role for role, _ in self.tensor_formats}
        if roles != required or len(roles) != len(self.tensor_formats):
            raise ValueError("precision roles must be complete and unique")
        if not self.grouping or not all(
            (
                self.scale_rule,
                self.rounding,
                self.clipping,
                self.accumulator,
                self.kernel,
                self.version,
            )
        ):
            raise ValueError("quantization implementation fields are incomplete")
        return True


def studio_quantization_contract() -> QuantizationContract:
    """Build the deterministic quantization contract teaching fixture."""
    return QuantizationContract(
        (
            ("routed_expert_weight", "MXFP4"),
            ("routed_expert_input_activation", "MXFP8"),
            ("attention_projection", "higher"),
            ("latent_moe_projection", "higher"),
            ("shared_expert", "higher"),
            ("moe_router", "higher"),
        ),
        (
            ("routed_expert_weight", "fixture-block-axis"),
            ("routed_expert_input_activation", "fixture-token-axis"),
        ),
        "fixture-shared-scale-v1",
        "nearest-even-fixture-v1",
        "finite-range-fixture-v1",
        "fp32-fixture-v1",
        ("reported-higher-precision-roles",),
        "semantic-emulator-v1",
        "studio-qcontract-v1",
    )


@dataclass(frozen=True)
class PrecisionFingerprint:
    """Bind model, quantizer, kernel, format, and decoding identities."""

    parameter_version: str
    tokenizer_version: str
    quantizer_version: str
    kernel_version: str
    weight_formats: tuple[tuple[str, str], ...]
    activation_formats: tuple[tuple[str, str], ...]
    accumulator_formats: tuple[tuple[str, str], ...]
    grouping: tuple[tuple[str, str], ...]
    scale_rule: str
    rounding: str
    clipping: str
    decoding: str

    def validate(self) -> bool:
        identity_fields = (
            self.parameter_version,
            self.tokenizer_version,
            self.quantizer_version,
            self.kernel_version,
            self.scale_rule,
            self.rounding,
            self.clipping,
            self.decoding,
        )
        format_fields = (
            self.weight_formats,
            self.activation_formats,
            self.accumulator_formats,
            self.grouping,
        )
        if not all(value.strip() for value in identity_fields):
            raise ValueError("precision fingerprint identity fields are incomplete")
        if any(
            not entries
            or any(not role.strip() or not value.strip() for role, value in entries)
            for entries in format_fields
        ):
            raise ValueError("precision fingerprint format fields are incomplete")
        return True


def parity_audit(
    rollout: PrecisionFingerprint,
    training: PrecisionFingerprint,
    rollout_decisions: Sequence[object],
    training_decisions: Sequence[object],
    rollout_logp: float,
    training_logp: float,
    tolerance: float,
) -> dict[str, object]:
    """Compare rollout and training contracts, decisions, and log probabilities."""
    rollout.validate()
    training.validate()
    if tolerance < 0 or not all(
        math.isfinite(value) for value in (rollout_logp, training_logp, tolerance)
    ):
        raise ValueError("parity tolerance and log probabilities must be finite")
    if rollout != training:
        return {"status": "contract_mismatch", "eligible": False}
    if tuple(rollout_decisions) != tuple(training_decisions):
        return {"status": "decision_mismatch", "eligible": False}
    difference = abs(rollout_logp - training_logp)
    if difference > tolerance:
        return {
            "status": "numerical_mismatch",
            "eligible": False,
            "difference": difference,
        }
    return {"status": "pass", "eligible": True, "difference": difference}


def acceptance_mass(
    target: Sequence[int | float | Fraction],
    draft: Sequence[int | float | Fraction],
) -> Fraction:
    """Return speculative decoding acceptance mass."""
    p, q = probability_vector(target), probability_vector(draft)
    if len(p) != len(q):
        raise ValueError("target and draft vocabularies differ")
    return sum((min(p_i, q_i) for p_i, q_i in zip(p, q, strict=True)), Fraction(0))


def total_variation(
    target: Sequence[int | float | Fraction],
    draft: Sequence[int | float | Fraction],
) -> Fraction:
    """Return total-variation distance between distributions."""
    p, q = probability_vector(target), probability_vector(draft)
    if len(p) != len(q):
        raise ValueError("target and draft vocabularies differ")
    return (
        sum(
            (abs(p_i - q_i) for p_i, q_i in zip(p, q, strict=True)),
            Fraction(0),
        )
        / 2
    )


def lk_loss(
    target: Sequence[int | float | Fraction],
    draft: Sequence[int | float | Fraction],
) -> float:
    """Compute the exact teaching fixture's LK loss."""
    overlap = acceptance_mass(target, draft)
    return math.inf if overlap == 0 else -math.log(float(overlap))


def recurrent_inputs(
    initial_token: str, draft_outputs: Sequence[str], steps: int = 7
) -> tuple[str, ...]:
    """Build seven recurrent inputs from an initial token and draft outputs."""
    if not initial_token or steps != 7 or len(draft_outputs) != steps:
        raise ValueError(
            "the teaching recurrence requires one initial token and seven outputs"
        )
    if any(not token for token in draft_outputs):
        raise ValueError("draft outputs must be nonempty")
    return (initial_token, *tuple(draft_outputs[:-1]))
