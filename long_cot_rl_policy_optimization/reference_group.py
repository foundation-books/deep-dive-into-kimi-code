"""Typed rollout groups and exact centering for the Chapter 19 companion."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence, TypeGuard


STUDIO_LENGTHS = (8, 12, 8, 20)
STUDIO_REWARDS = (1, 1, 0, 1)
STUDIO_BEHAVIOR_VERSION = 120
STUDIO_UPDATE_VERSION = 123


@dataclass(frozen=True)
class GroupSpec:
    """Minimal typed group used by the Chapter 19 fixture."""

    task_ids: tuple[str, ...]
    behavior_version: int | None
    update_version: int
    promised_size: int
    lengths: tuple[int, ...]
    outcomes: tuple[int | float | Fraction, ...]
    completed: tuple[bool, ...]
    sampling_condition: str


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("Boolean is not a reward value")
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("reward must be finite")
    return Fraction(str(value))


def studio_group() -> GroupSpec:
    """Return the immutable four-trajectory studio group."""

    return GroupSpec(
        task_ids=("studio-layout-001",) * 4,
        behavior_version=STUDIO_BEHAVIOR_VERSION,
        update_version=STUDIO_UPDATE_VERSION,
        promised_size=4,
        lengths=STUDIO_LENGTHS,
        outcomes=STUDIO_REWARDS,
        completed=(True, True, True, True),
        sampling_condition="same tokenizer, action support, and sampler declaration",
    )


def _valid_integer(value: object, *, minimum: int = 0) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def validate_group(group: GroupSpec) -> dict[str, int | str]:
    """Validate D5--D7 and return the defined denominators."""

    counts = {
        len(group.task_ids),
        len(group.lengths),
        len(group.outcomes),
        len(group.completed),
    }
    if len(counts) != 1:
        raise ValueError("group fields have different member counts")
    realized_size = counts.pop()
    if not _valid_integer(group.promised_size, minimum=2) or (
        realized_size != group.promised_size
    ):
        raise ValueError("group is incomplete or has invalid K")
    if len(set(group.task_ids)) != 1:
        raise ValueError("group mixes task instances")
    if not _valid_integer(group.behavior_version):
        raise ValueError("behavior-policy version is missing")
    if not _valid_integer(group.update_version):
        raise ValueError("update-policy version must be an integer")
    if group.update_version < group.behavior_version:
        raise ValueError("update version predates behavior version")
    if not group.sampling_condition.strip():
        raise ValueError("sampling condition is missing")
    if not all(group.completed):
        raise ValueError("group contains an unhandled unfinished trajectory")
    if any(
        isinstance(length, bool) or not isinstance(length, int) or length <= 0
        for length in group.lengths
    ):
        raise ValueError("assistant-owned lengths must be positive integers")
    for reward in group.outcomes:
        _fraction(reward)
    return {
        "task_id": group.task_ids[0],
        "K": realized_size,
        "N": sum(group.lengths),
        "update_lag": group.update_version - group.behavior_version,
    }


def group_advantages(
    rewards: Sequence[int | float | Fraction],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    """Return the exact arithmetic mean and centered advantages."""

    if len(rewards) < 2:
        raise ValueError("a relative-outcome group needs K >= 2")
    values = tuple(_fraction(value) for value in rewards)
    mean = sum(values, Fraction(0)) / len(values)
    advantages = tuple(value - mean for value in values)
    if sum(advantages, Fraction(0)) != 0:
        raise AssertionError("centering conservation failed")
    return mean, advantages
