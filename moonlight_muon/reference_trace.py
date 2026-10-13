"""Deterministic two-step hybrid optimizer trace for Chapter 12."""

from __future__ import annotations

from .reference_core import (
    AdamWFlatState,
    AdamWFlatStep,
    MuonStep,
    adamw_step_flat,
    muon_step,
    state_ledger,
)
from .reference_matrix import (
    as_matrix,
    constant_matrix,
    diagonal_matrix,
    flat_norm,
    flat_rms,
    flatten,
    frobenius_norm,
    rms,
    shape,
    subtract,
    zeros,
)
from .reference_types import FlatTensor, Matrix


def running_parameters() -> dict[str, tuple[str, tuple[int, ...]]]:
    """The Chapter 12 hybrid optimizer fixture."""

    return {
        "W_Q": ("attention_weight", (4, 4)),
        "W_up": ("mlp_weight", (16, 4)),
        "W_route": ("router_weight", (8, 4)),
        "E_tok": ("embedding", (12, 4)),
        "W_out": ("output_head", (12, 4)),
        "gamma": ("normalization", (4,)),
    }


def running_gradients(step: int) -> dict[str, Matrix]:
    """Return fixed matrix gradients for either step of the teaching trace."""
    if step not in (1, 2):
        raise ValueError("the fixture contains exactly two gradient steps")
    if step == 1:
        return {
            "W_Q": diagonal_matrix(4, 4, (3.0, 2.0, 1.0, 0.5)),
            "W_up": diagonal_matrix(16, 4, (4.0, 3.0, 2.0, 1.0)),
            "W_route": diagonal_matrix(8, 4, (2.0, -1.0, 0.5, 0.25)),
        }
    return {
        "W_Q": as_matrix(((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, -1, 0), (0, 0, 0, 2))),
        "W_up": as_matrix(
            tuple(
                tuple(1.0 if (i + 1) % 4 == j else 0.0 for j in range(4))
                for i in range(16)
            )
        ),
        "W_route": as_matrix(
            tuple(
                tuple((-1.0 if i < 4 else 1.0) if i % 4 == j else 0.0 for j in range(4))
                for i in range(8)
            )
        ),
    }


def running_initial_muon_parameters() -> dict[str, Matrix]:
    """Fixed rational matrix weights for the two-step running trace."""

    return {
        "W_Q": diagonal_matrix(4, 4, (1.0, 0.5, -0.5, 0.25)),
        "W_up": constant_matrix(16, 4, 0.125),
        "W_route": constant_matrix(8, 4, -0.25),
    }


def running_initial_adamw_parameters() -> dict[str, FlatTensor]:
    """Fixed rational non-Muon weights, flattened in row-major order."""

    return {
        "E_tok": tuple(0.5 for _ in range(12 * 4)),
        "W_out": tuple(-0.25 for _ in range(12 * 4)),
        "gamma": (1.0, 1.0, 1.0, 1.0),
    }


def running_adamw_gradients(step: int) -> dict[str, FlatTensor]:
    """The chapter's deterministic AdamW fallback gradients."""

    if step not in (1, 2):
        raise ValueError("the fixture contains exactly two gradient steps")
    if step == 1:
        return {
            "E_tok": tuple(1.0 for _ in range(12 * 4)),
            "W_out": tuple(-1.0 for _ in range(12 * 4)),
            "gamma": (1.0, -1.0, 1.0, -1.0),
        }
    return {
        "E_tok": tuple(0.5 for _ in range(12 * 4)),
        "W_out": tuple(1.0 for _ in range(12 * 4)),
        "gamma": (0.5, 0.5, -0.5, -0.5),
    }


def _matrix_checkpoint(
    name: str,
    step: int,
    before: Matrix,
    result: MuonStep,
) -> dict[str, object]:
    values = flatten(result.weight)
    return {
        "parameter": name,
        "optimizer": "Muon",
        "step": step,
        "shape": list(shape(result.weight)),
        "parameter_frobenius_norm": frobenius_norm(result.weight),
        "parameter_delta_norm": frobenius_norm(subtract(result.weight, before)),
        "update_direction_rms": rms(result.direction),
        "first_parameter_value": values[0],
        "last_parameter_value": values[-1],
        "persistent_state_elements": len(values),
    }


def _flat_checkpoint(
    name: str,
    dimensions: tuple[int, ...],
    before: FlatTensor,
    result: AdamWFlatStep,
) -> dict[str, object]:
    delta = tuple(
        after - prior for after, prior in zip(result.weight, before, strict=True)
    )
    return {
        "parameter": name,
        "optimizer": "AdamW",
        "step": result.state.step,
        "shape": list(dimensions),
        "parameter_frobenius_norm": flat_norm(result.weight),
        "parameter_delta_norm": flat_norm(delta),
        "update_direction_rms": flat_rms(result.direction),
        "first_parameter_value": result.weight[0],
        "last_parameter_value": result.weight[-1],
        "persistent_state_elements": 2 * len(result.weight) + 1,
    }


def running_hybrid_trace() -> dict[str, object]:
    """Execute the complete deterministic two-step hybrid optimizer trace.

    All six parameters are updated atomically at each step.  The returned
    checkpoint ledger is JSON-compatible and intentionally summarizes rather
    than serializes every repeated entry of the small tensors.
    """

    parameters = running_parameters()
    muon_weights = running_initial_muon_parameters()
    muon_momenta = {
        name: zeros(*shape(weight)) for name, weight in muon_weights.items()
    }
    adamw_weights = running_initial_adamw_parameters()
    adamw_states = {
        name: AdamWFlatState(
            tuple(0.0 for _ in weight),
            tuple(0.0 for _ in weight),
            0,
        )
        for name, weight in adamw_weights.items()
    }
    initial_norms = {
        **{name: frobenius_norm(weight) for name, weight in muon_weights.items()},
        **{name: flat_norm(weight) for name, weight in adamw_weights.items()},
    }
    checkpoints: list[dict[str, object]] = []

    for step in (1, 2):
        for name, muon_gradient in running_gradients(step).items():
            muon_before = muon_weights[name]
            muon_result = muon_step(muon_before, muon_gradient, muon_momenta[name])
            muon_weights[name] = muon_result.weight
            muon_momenta[name] = muon_result.momentum
            checkpoints.append(_matrix_checkpoint(name, step, muon_before, muon_result))

        for name, adamw_gradient in running_adamw_gradients(step).items():
            adamw_before = adamw_weights[name]
            adamw_result = adamw_step_flat(
                adamw_before, adamw_gradient, adamw_states[name]
            )
            adamw_weights[name] = adamw_result.weight
            adamw_states[name] = adamw_result.state
            checkpoints.append(
                _flat_checkpoint(name, parameters[name][1], adamw_before, adamw_result)
            )

    final_values = {
        **{name: flatten(weight) for name, weight in muon_weights.items()},
        **adamw_weights,
    }
    final_checksums = {name: sum(values) for name, values in final_values.items()}
    return {
        "hyperparameters": {
            "learning_rate": 1e-3,
            "weight_decay": 0.1,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_shape_base": 0.2,
            "adamw_beta_1": 0.9,
            "adamw_beta_2": 0.999,
            "adamw_epsilon": 1e-8,
        },
        "initial_parameter_norms": initial_norms,
        "checkpoints": checkpoints,
        "final_parameter_checksums": final_checksums,
        "final_state_elements": state_ledger(parameters).total_elements,
    }


__all__ = [
    "running_adamw_gradients",
    "running_gradients",
    "running_hybrid_trace",
    "running_initial_adamw_parameters",
    "running_initial_muon_parameters",
    "running_parameters",
]
