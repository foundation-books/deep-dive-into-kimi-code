"""Small counterexamples for the Volume I implementation investigations.

Run from code/: python3 -B -m integrated_k3_transformer.implementation_insights
These are teaching reconstructions, not executions of third-party code.
"""

from __future__ import annotations

from fractions import Fraction
import json
from math import exp, log

from attention_residuals.reference_core import mix_depth
from attention_residuals.reference_math import rms_norm
from attention_residuals.reference_types import Dimensions
from moonlight_muon.reference_core import muon_step
from sparse_moe.reference import incorrect_biased_weights, route

from .reference_schedule import terminal_global_schedule, validate_schedule


RoutingComparison = dict[str, object]
DepthValueComparison = dict[str, object]
ScheduleComparison = dict[str, object]


def routing_comparison() -> RoutingComparison:
    """Keep membership fixed while changing the source of mixture weights."""
    biases = (0.2, -0.1, 0.0, 0.1)
    result = route((0.1, 0.7, 0.4, 0.6), biases, 2)
    wrong = incorrect_biased_weights(result, biases)
    # Selected experts 4 and 1 return scalar payloads 10 and 0.
    return {
        "selected": result.selected,
        "raw_score_weights": result.weights,
        "biased_score_weights": wrong,
        "raw_score_output": 10 * result.weights[0],
        "biased_score_output": 10 * wrong[0],
    }


def convolution_continuation(reset: bool = False) -> tuple[Fraction, ...]:
    """Advance a scalar value projection through a width-four mean filter.

    Use q=k=1, alpha=beta=1/2, S=0, and identity value activation.
    This isolates history loss; it is not the complete nonlinear KDA layer.
    """
    history = (Fraction(0),) * 3 if reset else tuple(map(Fraction, (1, 2, 3)))
    state = Fraction(0)
    outputs = []
    for projection in (4, 5, 6, 7, 8):
        value = sum((*history, Fraction(projection)), Fraction(0)) / 4
        retained = state / 2
        state = retained + (value - retained) / 2
        outputs.append(state)
        history = (*history[1:], Fraction(projection))
    return tuple(outputs)


def depth_value_comparison() -> DepthValueComparison:
    """Equal score logits isolate the values branch with nonzero epsilon."""
    sources = (((1.0, 0.0),), ((0.0, 2.0),), ((2.0, 2.0),))
    result = mix_depth((0.0, 0.0), sources, Dimensions(1, 2, 1, 1))
    normalized = tuple(rms_norm(source[0]) for source in sources)
    wrong = tuple(sum(key[channel] for key in normalized) / 3 for channel in (0, 1))
    return {
        "weights": result.weights[0],
        "raw_values": result.output[0],
        "normalized_values": wrong,
    }


def gate_comparison() -> dict[str, tuple[Fraction, ...]]:
    """One latent coordinate expands to two independently gated channels."""
    latent = Fraction(2)
    up = (Fraction(1), Fraction(3))
    gates = ((Fraction(1, 4), Fraction(3, 4)), (Fraction(3, 4), Fraction(1, 4)))
    # W^O = [1,1]; both gate vectors are attainable sigmoid outputs.
    dynamic = tuple(
        sum((g * w * latent for g, w in zip(gate, up, strict=True)), Fraction(0))
        for gate in gates
    )
    return {"dynamic_outputs": dynamic, "ungated_outputs": (Fraction(8),) * 2}


def muon_decay_comparison() -> dict[str, float]:
    """Run the existing matrix transition with a zero candidate to expose decay."""
    weight = ((0.125,) * 4,) * 16
    zero = ((0.0,) * 4,) * 16
    correct = muon_step(weight, zero, zero)
    # Deliberately use the effective direction coefficient for decay too.
    wrong = muon_step(weight, zero, zero, learning_rate=0.0008)
    return {
        "base_rate_decay": correct.weight[0][0],
        "adjusted_rate_decay": wrong.weight[0][0],
        "difference": wrong.weight[0][0] - correct.weight[0][0],
        "rms_matched_coefficient": 0.001 * 0.2 * 16**0.5,
        "original_coefficient": 0.001 * (16 / 4) ** 0.5,
    }


def packed_target_loss(first_document_value: float, isolate: bool) -> float:
    """Loss for v2 from a uniform-attention scalar at v1.

    Three U tokens have the supplied value, v1 has value zero. The U->v1
    target is masked out. This remaining v1->v2 prediction can still see U.
    Position IDs do not enter this deliberately position-free toy attention.
    """
    context = (0.0,) if isolate else (first_document_value,) * 3 + (0.0,)
    hidden = sum(context) / len(context)
    probability = 1.0 / (1.0 + exp(-hidden))
    return -log(probability)


def schedule_comparison() -> ScheduleComparison:
    """Equal inventories do not imply equal ordered computations."""
    correct = terminal_global_schedule(2)
    wrong = list(correct)
    wrong[2], wrong[3] = wrong[3], wrong[2]
    try:
        validate_schedule(wrong)
    except ValueError:
        rejected = True
    else:
        rejected = False

    # Illustrative scalar operators, not KDA and MLA implementations.
    def execute(schedule: tuple[str, ...]) -> int:
        value = 1
        for kind in schedule:
            value = value + 1 if kind == "K" else 2 * value
        return value

    return {
        "correct": correct,
        "permuted": tuple(wrong),
        "same_counts": correct.count("G") == wrong.count("G"),
        "permutation_rejected": rejected,
        "correct_scalar_result": execute(correct),
        "permuted_scalar_result": execute(tuple(wrong)),
    }


def main() -> None:
    """Print the seven bounded witnesses without writing stored evidence."""
    comparisons = {
        "routing": routing_comparison(),
        "kda": {
            "restored": tuple(map(str, convolution_continuation())),
            "reset": tuple(map(str, convolution_continuation(reset=True))),
        },
        "attention_residuals": depth_value_comparison(),
        "gate": {
            key: tuple(map(str, values)) for key, values in gate_comparison().items()
        },
        "muon": muon_decay_comparison(),
        "packing": {
            "continuous": [packed_target_loss(value, False) for value in (0.0, 4.0)],
            "isolated": [packed_target_loss(value, True) for value in (0.0, 4.0)],
        },
        "composition": schedule_comparison(),
    }
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
