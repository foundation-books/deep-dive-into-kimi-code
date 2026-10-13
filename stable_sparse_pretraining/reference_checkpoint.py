"""Optimizer and durable-restart oracles for the Chapter 14 companion."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import pairwise
import json
from typing import Mapping, Sequence


Vector = tuple[Fraction, ...]


@dataclass(frozen=True)
class OptimizerState:
    """One logical owner state for four scalar expert coordinates."""

    weights: Vector
    momentum: Vector
    step: int


def optimizer_step(
    state: OptimizerState,
    gradient: Sequence[Fraction],
    learning_rate: Fraction = Fraction(1, 100),
    beta: Fraction = Fraction(9, 10),
) -> OptimizerState:
    """Apply one exact owner update after complete gradient consensus."""

    if len(state.weights) != len(state.momentum) or len(gradient) != len(state.weights):
        raise ValueError("weights, momentum, and gradient lengths must agree")
    momentum = tuple(
        beta * old + (1 - beta) * value
        for old, value in zip(state.momentum, gradient, strict=True)
    )
    weights = tuple(
        weight - learning_rate * update
        for weight, update in zip(state.weights, momentum, strict=True)
    )
    return OptimizerState(weights, momentum, state.step + 1)


@dataclass(frozen=True)
class TrainingState:
    """Restart-critical state committed at one logical step boundary."""

    weights: Vector
    momentum: Vector
    qb_bias: Vector
    scheduler_tick: int
    rng_counter: int
    data_cursor: int
    step: int


def initial_training_state() -> TrainingState:
    """Return the fixed committed state before the replay example."""

    return TrainingState(
        weights=tuple(Fraction(value) for value in (10, 20, 30, 40)),
        momentum=(Fraction(0),) * 4,
        qb_bias=(Fraction(0),) * 4,
        scheduler_tick=0,
        rng_counter=17,
        data_cursor=0,
        step=0,
    )


def advance_state(state: TrainingState) -> TrainingState:
    """Execute one deterministic logical step from a committed state."""

    gradient = tuple(Fraction((state.data_cursor + 1) * (i + 1), 10) for i in range(4))
    updated = optimizer_step(
        OptimizerState(state.weights, state.momentum, state.step), gradient
    )
    counts = (8, 4, 2, 2)
    target = Fraction(4)
    bias = tuple(
        value - Fraction(count - target, 32)
        for value, count in zip(state.qb_bias, counts, strict=True)
    )
    return TrainingState(
        weights=updated.weights,
        momentum=updated.momentum,
        qb_bias=bias,
        scheduler_tick=state.scheduler_tick + 1,
        rng_counter=state.rng_counter + 13,
        data_cursor=state.data_cursor + 1,
        step=updated.step,
    )


def _fraction_text(values: Sequence[Fraction]) -> list[str]:
    return [str(value) for value in values]


def state_payload(state: TrainingState) -> dict[str, object]:
    """Return a stable serialization of restart-critical coordinates."""

    return {
        "weights": _fraction_text(state.weights),
        "momentum": _fraction_text(state.momentum),
        "qb_bias": _fraction_text(state.qb_bias),
        "scheduler_tick": state.scheduler_tick,
        "rng_counter": state.rng_counter,
        "data_cursor": state.data_cursor,
        "step": state.step,
    }


def state_digest(state: TrainingState) -> str:
    """Hash one canonical restart state."""

    payload = json.dumps(state_payload(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_manifest(state: TrainingState) -> dict[str, object]:
    """Publish the atomic manifest that makes a checkpoint durable."""

    return {
        "schema": 1,
        "step": state.step,
        "state_sha256": state_digest(state),
        "shard_steps": [state.step, state.step],
        "complete": True,
    }


def load_checkpoint(
    manifest: Mapping[str, object], state: TrainingState
) -> TrainingState:
    """Validate the manifest before exposing a checkpoint as restartable."""

    required = {"schema", "step", "state_sha256", "shard_steps", "complete"}
    if set(manifest) != required:
        raise ValueError("checkpoint manifest fields are incomplete or unknown")
    if manifest["schema"] != 1 or manifest["complete"] is not True:
        raise ValueError("checkpoint manifest is not committed")
    if manifest["step"] != state.step:
        raise ValueError("manifest and state steps disagree")
    if manifest["shard_steps"] != [state.step, state.step]:
        raise ValueError("checkpoint shards do not share one committed step")
    if manifest["state_sha256"] != state_digest(state):
        raise ValueError("checkpoint state digest mismatch")
    return state


def shard_vector(values: Sequence[Fraction], cuts: Sequence[int]) -> tuple[Vector, ...]:
    """Partition a logical vector by validated increasing cut points."""

    if not cuts or cuts[0] != 0 or cuts[-1] != len(values):
        raise ValueError("cuts must span the complete vector")
    if any(left >= right for left, right in pairwise(cuts)):
        raise ValueError("cuts must be strictly increasing")
    return tuple(tuple(values[start:stop]) for start, stop in pairwise(cuts))


def assemble_shards(shards: Sequence[Sequence[Fraction]]) -> Vector:
    """Reconstruct a canonical vector from ordered disjoint shards."""

    if not shards or any(not shard for shard in shards):
        raise ValueError("shards must be nonempty")
    return tuple(value for shard in shards for value in shard)


def restart_replay_fixture() -> dict[str, object]:
    """Crash before commit, reload the old manifest, and replay exactly."""

    step_one = advance_state(initial_training_state())
    durable_manifest = checkpoint_manifest(step_one)
    volatile_step_two = advance_state(step_one)
    loaded = load_checkpoint(durable_manifest, step_one)
    replayed_step_two = advance_state(loaded)
    torn_manifest = dict(durable_manifest)
    torn_manifest["step"] = 2
    rejected = False
    try:
        load_checkpoint(torn_manifest, step_one)
    except ValueError:
        rejected = True
    committed_step_two_manifest = checkpoint_manifest(volatile_step_two)
    after_commit = load_checkpoint(committed_step_two_manifest, volatile_step_two)
    two_way = shard_vector(step_one.weights, (0, 2, 4))
    one_way = shard_vector(assemble_shards(two_way), (0, 4))
    return {
        "durable_step": step_one.step,
        "volatile_step": volatile_step_two.step,
        "loaded_step": loaded.step,
        "replayed_step": replayed_step_two.step,
        "replay_equal": replayed_step_two == volatile_step_two,
        "torn_manifest_rejected": rejected,
        "crash_window_loaded_steps": {
            "before_new_shards": loaded.step,
            "after_partial_shards": loaded.step,
            "after_all_shards_before_manifest": loaded.step,
            "after_valid_manifest": after_commit.step,
        },
        "compatible_reshard_equal": assemble_shards(one_way) == step_one.weights,
        "durable_digest": state_digest(step_one),
        "replayed_digest": state_digest(replayed_step_two),
    }


__all__ = (
    "OptimizerState",
    "TrainingState",
    "Vector",
    "advance_state",
    "assemble_shards",
    "checkpoint_manifest",
    "initial_training_state",
    "load_checkpoint",
    "optimizer_step",
    "restart_replay_fixture",
    "shard_vector",
    "state_digest",
    "state_payload",
)
