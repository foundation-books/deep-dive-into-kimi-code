"""Exact and fresh-group sampled updates of the Chapter 21 finite policy."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import random
from typing import Sequence, TypedDict, cast

from .finite_policy import (
    Route,
    exact_step,
    metrics,
    route_states,
    softmax,
    validate_fixture,
)
from .reference_core import effective_reward


@dataclass(frozen=True)
class Stage:
    """A named stretch of updates trained under one fixed cap."""

    name: str
    updates: int
    budget: Fraction

    def validate(self) -> None:
        if (
            not self.name
            or isinstance(self.updates, bool)
            or not isinstance(self.updates, int)
            or self.updates <= 0
        ):
            raise ValueError("stage needs a name and positive update count")
        if not math.isfinite(float(self.budget)) or self.budget <= 0:
            raise ValueError("stage budget must be finite and positive")


def stage_at(schedule: Sequence[Stage], completed_updates: int) -> Stage:
    """Select the next update's stage by completed-update count."""

    if completed_updates < 0:
        raise ValueError("completed updates cannot be negative")
    boundary = 0
    for stage in schedule:
        stage.validate()
        boundary += stage.updates
        if completed_updates < boundary:
            return stage
    raise ValueError("completed updates exceed the declared schedule")


def exact_run(
    routes: Sequence[Route],
    logits: Sequence[float],
    schedule: Sequence[Stage],
    rate: float,
) -> list[dict[str, object]]:
    """Record every exact-gradient checkpoint under its training schedule."""

    trace: list[dict[str, object]] = []
    state = tuple(logits)
    validate_fixture(routes, softmax(state))
    total = sum(stage.updates for stage in schedule)
    if total <= 0:
        raise ValueError("schedule must contain updates")
    for completed in range(total + 1):
        stage = stage_at(schedule, max(completed - 1, 0))
        trace.append(
            {
                "completed_updates": completed,
                "stage": stage.name,
                "cap": float(stage.budget),
                "logits": list(state),
                **metrics(routes, state, stage.budget),
            }
        )
        if completed < total:
            update_stage = stage_at(schedule, completed)
            state = exact_step(routes, state, update_stage.budget, rate)
    return trace


def _pick_route(probabilities: Sequence[float], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def group_gradient(
    selected: Sequence[int],
    rewards: Sequence[float],
    probabilities: Sequence[float],
    *,
    corrected: bool,
) -> tuple[float, ...]:
    """Form an inclusive group-centered score gradient, optionally corrected."""

    size = len(selected)
    if size < 2 or size != len(rewards):
        raise ValueError("a centered group needs at least two matched samples")
    if any(index < 0 or index >= len(probabilities) for index in selected):
        raise ValueError("selected route index is invalid")
    group_mean = sum(rewards) / size
    scale = size / (size - 1) if corrected else 1.0
    return tuple(
        scale
        * sum(
            (reward - group_mean) * (float(index == axis) - probabilities[axis])
            for index, reward in zip(selected, rewards, strict=True)
        )
        / size
        for axis in range(len(probabilities))
    )


class GroupRecord(TypedDict):
    """Observed group diagnostics and probabilities bracketing one update."""

    collection_probabilities: list[float]
    updated_probabilities: list[float]
    raw_success: float
    effective_reward: float
    compliance: float
    consumed_cost: int
    no_centered_signal: bool
    route_counts: list[int]
    gradient: list[float]


def sampled_group_step(
    routes: Sequence[Route],
    logits: Sequence[float],
    budget: Fraction,
    rate: float,
    group_size: int,
    rng: random.Random,
    *,
    corrected: bool = True,
) -> tuple[tuple[float, ...], GroupRecord]:
    """Collect a whole group from one frozen policy, then update once."""
    if isinstance(group_size, bool) or group_size < 2:
        raise ValueError("group size must be at least two")
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("learning rate must be finite and nonnegative")
    if len(routes) != len(logits):
        raise ValueError("route and logit counts must match")
    p = softmax(logits)
    states = route_states(routes, budget)
    selected: list[int] = []
    raw: list[int] = []
    effective: list[float] = []
    for _ in range(group_size):
        index = _pick_route(p, rng)
        success = int(rng.random() < routes[index].success_probability)
        selected.append(index)
        raw.append(success)
        effective.append(float(effective_reward(success, states[index])))
    group_mean = sum(effective) / group_size
    gradient = group_gradient(selected, effective, p, corrected=corrected)
    updated = tuple(
        value + rate * grad for value, grad in zip(logits, gradient, strict=True)
    )
    record: GroupRecord = {
        "collection_probabilities": list(p),
        "updated_probabilities": list(softmax(updated)),
        "raw_success": sum(raw) / group_size,
        "effective_reward": group_mean,
        "compliance": sum(routes[index].thinking_cost <= budget for index in selected)
        / group_size,
        "consumed_cost": sum(routes[index].thinking_cost for index in selected),
        "no_centered_signal": all(reward == effective[0] for reward in effective),
        "route_counts": [selected.count(axis) for axis in range(len(p))],
        "gradient": list(gradient),
    }
    return updated, record


def _restore_random_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_restore_random_state(item) for item in value)
    return value


@dataclass(frozen=True)
class SampledState:
    """Resumable learner state, including RNG and generated-cost counters."""

    logits: tuple[float, ...]
    completed_updates: int
    random_state: tuple[object, ...]
    consumed_cost: int
    no_signal_groups: int
    route_counts: tuple[int, ...]
    first_d_majority_update: int | None

    def to_json(self) -> dict[str, object]:
        return {
            "logits": list(self.logits),
            "completed_updates": self.completed_updates,
            "random_state": self.random_state,
            "consumed_cost": self.consumed_cost,
            "no_signal_groups": self.no_signal_groups,
            "route_counts": list(self.route_counts),
            "first_d_majority_update": self.first_d_majority_update,
        }

    @classmethod
    def from_json(cls, record: dict[str, object]) -> SampledState:
        logits = cast(list[float], record["logits"])
        counts = cast(list[int], record["route_counts"])
        return cls(
            tuple(float(value) for value in logits),
            cast(int, record["completed_updates"]),
            cast(tuple[object, ...], _restore_random_state(record["random_state"])),
            cast(int, record["consumed_cost"]),
            cast(int, record["no_signal_groups"]),
            tuple(int(value) for value in counts),
            (
                None
                if record["first_d_majority_update"] is None
                else cast(int, record["first_d_majority_update"])
            ),
        )


def _advance_sampled_state(
    current: SampledState,
    updated: tuple[float, ...],
    group: GroupRecord,
    routes: Sequence[Route],
    rng: random.Random,
) -> SampledState:
    first_d = current.first_d_majority_update
    if first_d is None:
        updated_probabilities = softmax(updated)
        if any(
            route.name == "D" and updated_probabilities[axis] > 0.5
            for axis, route in enumerate(routes)
        ):
            first_d = current.completed_updates + 1
    counts = tuple(
        before + count
        for before, count in zip(
            current.route_counts, group["route_counts"], strict=True
        )
    )
    return SampledState(
        updated,
        current.completed_updates + 1,
        rng.getstate(),
        current.consumed_cost + group["consumed_cost"],
        current.no_signal_groups + int(group["no_centered_signal"]),
        counts,
        first_d,
    )


def sampled_run(
    routes: Sequence[Route],
    logits: Sequence[float],
    schedule: Sequence[Stage],
    rate: float,
    group_size: int,
    seed: int,
    *,
    stop_at: int | None = None,
    resume: SampledState | None = None,
    capture: frozenset[int] = frozenset(),
) -> tuple[SampledState, list[dict[str, object]]]:
    """Collect a fresh group per update and save requested checkpoints."""

    total = sum(stage.updates for stage in schedule)
    if total <= 0:
        raise ValueError("schedule must contain updates")
    validate_fixture(routes, softmax(logits))
    target = total if stop_at is None else stop_at
    if target < 0 or target > total:
        raise ValueError("stop point must lie inside schedule")
    rng = random.Random(seed)
    if resume is None:
        current = SampledState(
            tuple(logits), 0, rng.getstate(), 0, 0, tuple(0 for _ in routes), None
        )
    else:
        current = resume
        rng.setstate(current.random_state)
    if target < current.completed_updates:
        raise ValueError("cannot resume backwards")
    saved: list[dict[str, object]] = []
    for completed in range(current.completed_updates, target):
        stage = stage_at(schedule, completed)
        updated, group = sampled_group_step(
            routes, current.logits, stage.budget, rate, group_size, rng
        )
        current = _advance_sampled_state(current, updated, group, routes, rng)
        if current.completed_updates in capture:
            saved.append(
                {
                    "completed_updates": current.completed_updates,
                    "stage": stage.name,
                    "cap": float(stage.budget),
                    "logits": list(updated),
                    "collected": group,
                    "evaluated": metrics(routes, updated, stage.budget),
                }
            )
    return current, saved
