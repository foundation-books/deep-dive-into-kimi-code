"""Online-softmax summaries used by the dependency-free MoBA reference."""

from __future__ import annotations

from math import exp
from typing import Sequence

from .reference_support import OnlineState


Vector = tuple[float, ...]


def online_state(logits: Sequence[float], values: Sequence[Vector]) -> OnlineState:
    """Summarize one nonempty attention partition with a stable local maximum."""

    if not logits or len(logits) != len(values):
        raise ValueError("logits and values must have equal positive length")
    width = len(values[0])
    if width == 0 or any(len(value) != width for value in values):
        raise ValueError("values must have a constant positive width")
    maximum = max(logits)
    shifted = tuple(exp(logit - maximum) for logit in logits)
    return OnlineState(
        maximum=maximum,
        denominator=sum(shifted),
        numerator=tuple(
            sum(
                weight * value[coordinate]
                for weight, value in zip(shifted, values, strict=True)
            )
            for coordinate in range(width)
        ),
    )


def merge_online_states(left: OnlineState, right: OnlineState) -> OnlineState:
    """Merge two stable attention summaries without changing softmax semantics."""

    if len(left.numerator) != len(right.numerator):
        raise ValueError("online states must have the same value width")
    maximum = max(left.maximum, right.maximum)
    left_scale = exp(left.maximum - maximum)
    right_scale = exp(right.maximum - maximum)
    return OnlineState(
        maximum=maximum,
        denominator=left_scale * left.denominator + right_scale * right.denominator,
        numerator=tuple(
            left_scale * left_value + right_scale * right_value
            for left_value, right_value in zip(
                left.numerator, right.numerator, strict=True
            )
        ),
    )


def finalize_online_state(state: OnlineState) -> Vector:
    """Normalize one online state into its attention output."""

    if state.denominator <= 0.0:
        raise ValueError("online-state denominator must be positive")
    return tuple(value / state.denominator for value in state.numerator)


def online_merge_microexample() -> dict[str, float]:
    """Return the exact scalar result for weights proportional to 1, 2, and 4."""

    left = online_state((0.0, 0.6931471805599453), ((1.0,), (3.0,)))
    right = online_state((1.3862943611198906,), ((5.0,),))
    merged = merge_online_states(left, right)
    return {
        "left_maximum": left.maximum,
        "left_denominator": left.denominator,
        "right_maximum": right.maximum,
        "right_denominator": right.denominator,
        "merged_denominator_in_global_scale": merged.denominator,
        "output": finalize_online_state(merged)[0],
        "expected": 27.0 / 7.0,
    }
