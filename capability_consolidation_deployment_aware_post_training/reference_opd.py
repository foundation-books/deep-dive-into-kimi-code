"""Strict sampled-score admission and token-use accounting for Chapter 23."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Callable, Sequence, TypeVar

from .reference_core import _fraction, clipped_reward


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ScoredOPDToken:
    """One sampled token's routed score record in a teaching update attempt."""

    teacher_id: str
    student_id: str
    prefix_id: str
    token_id: str
    teacher_probability: int | float | Fraction | None
    student_probability: int | float | Fraction | None


@dataclass(frozen=True)
class OPDAttemptResult:
    """Admission decision; refused attempts expose no update coefficients."""

    status: str
    raw_rewards: tuple[float, ...]
    clipped_rewards: tuple[float, ...]
    examined_positions: int
    failure: tuple[str, str, str, str, str] | None


def strict_opd_attempt(
    tokens: Sequence[ScoredOPDToken], cap: int | float | Fraction
) -> OPDAttemptResult:
    """Admit all sampled scores or refuse the whole teaching invocation.

    The caller supplies structural probabilities or ``None`` for an
    unresolved score. Passing sampled checks does not prove law-level support.
    This pure decision never applies a parameter or optimizer transition.
    """
    limit = float(_fraction(cap))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("cap must be finite and positive")
    raw: list[float] = []
    clipped: list[float] = []
    for index, token in enumerate(tokens):
        if not all(
            (token.teacher_id, token.student_id, token.prefix_id, token.token_id)
        ):
            raise ValueError("score identities must be nonempty")
        reason: str | None = None
        if token.teacher_probability is None or token.student_probability is None:
            reason = "unresolved_score"
        else:
            teacher = _fraction(token.teacher_probability)
            student = _fraction(token.student_probability)
            if student <= 0:
                reason = "invalid_sampled_student_score"
            elif teacher == 0:
                reason = "structural_teacher_zero"
            elif teacher < 0 or teacher > 1 or student > 1:
                reason = "invalid_probability_score"
            else:
                reward = (
                    math.log(teacher.numerator)
                    - math.log(teacher.denominator)
                    - math.log(student.numerator)
                    + math.log(student.denominator)
                )
                raw.append(reward)
                clipped.append(clipped_reward(reward, limit))
        if reason is not None:
            return OPDAttemptResult(
                "refused",
                (),
                (),
                index + 1,
                (
                    token.teacher_id,
                    token.student_id,
                    token.prefix_id,
                    token.token_id,
                    reason,
                ),
            )
    return OPDAttemptResult("admitted", tuple(raw), tuple(clipped), len(tokens), None)


def commit_opd_attempt(
    decision: OPDAttemptResult,
    state: StateT,
    updater: Callable[[StateT, tuple[float, ...]], StateT],
) -> tuple[StateT, int]:
    """Return new state and committed uses only after the updater succeeds."""
    if decision.status == "refused":
        return state, 0
    if decision.status != "admitted":
        raise ValueError("unknown OPD attempt status")
    updated_state = updater(state, decision.clipped_rewards)
    return updated_state, len(decision.clipped_rewards)


@dataclass(frozen=True)
class TokenUseCounts:
    """Unique sampled/admitted identities and committed use-event counts."""

    sampled: int
    admitted: int
    uses: int


def token_use_counts(
    sampled: Sequence[tuple[str, int]],
    admitted: Sequence[tuple[str, int]],
    committed_uses: Sequence[tuple[str, int]],
) -> TokenUseCounts:
    """Count logical identities separately from repeated accepted uses."""
    rows = (*sampled, *admitted, *committed_uses)
    if any(
        not trajectory
        or isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
        for trajectory, position in rows
    ):
        raise ValueError("token identities need a trajectory and nonnegative position")
    sample_ids = set(sampled)
    loss_ids = set(admitted)
    if not loss_ids <= sample_ids or any(use not in loss_ids for use in committed_uses):
        raise ValueError("loss positions and committed uses need sampled identities")
    return TokenUseCounts(len(sample_ids), len(loss_ids), len(committed_uses))
