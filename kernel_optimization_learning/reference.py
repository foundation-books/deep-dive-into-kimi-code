"""Float64 CPU semantics for one token–head normalization-and-gating task.

Each row is one token–head read of width ``d_v``. The gate inputs and gain are
supplied; their projections are outside this forward-only operation. These
functions establish values and constructed failure modes, never GPU speed.
"""

from collections.abc import Sequence
import math

Rows = Sequence[Sequence[float]]
Result = tuple[tuple[float, ...], ...]


def _checked_inputs(
    values: Rows, gates: Rows, gain: Sequence[float], epsilon: float
) -> tuple[Result, Result, tuple[float, ...], int]:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    value_rows = tuple(tuple(float(x) for x in row) for row in values)
    gate_rows = tuple(tuple(float(x) for x in row) for row in gates)
    gains = tuple(float(x) for x in gain)
    width = len(gains)
    if not value_rows or width == 0 or len(value_rows) != len(gate_rows):
        raise ValueError("need matching nonempty rows and a nonempty gain vector")
    if any(len(row) != width for row in (*value_rows, *gate_rows)):
        raise ValueError("every value and gate row must have gain-vector width")
    all_values = (*gains, *(x for row in (*value_rows, *gate_rows) for x in row))
    if not all(math.isfinite(x) for x in all_values):
        raise ValueError("values, gates, and gains must be finite")
    return value_rows, gate_rows, gains, width


def _checked_padding(padded_width: int, width: int) -> None:
    if isinstance(padded_width, bool) or not isinstance(padded_width, int):
        raise ValueError("padded width must be an integer")
    if padded_width < width:
        raise ValueError("padded width cannot be smaller than logical width")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _row_output(
    values: tuple[float, ...],
    gates: tuple[float, ...],
    gains: tuple[float, ...],
    epsilon: float,
    reduction_values: tuple[float, ...],
    divisor_width: int,
) -> tuple[float, ...]:
    mean_square = math.fsum(x * x for x in reduction_values) / divisor_width
    denominator = math.sqrt(mean_square + epsilon)
    return tuple(
        _sigmoid(gate) * weight * value / denominator
        for value, gate, weight in zip(values, gates, gains, strict=True)
    )


def normalized_gated_rows(
    values: Rows, gates: Rows, gain: Sequence[float], epsilon: float
) -> Result:
    """Compute sigmoid(G)*gain*U/sqrt(mean_d(U**2)+epsilon) per row."""
    value_rows, gate_rows, gains, width = _checked_inputs(values, gates, gain, epsilon)
    return tuple(
        _row_output(row, gate, gains, epsilon, row, width)
        for row, gate in zip(value_rows, gate_rows, strict=True)
    )


def fused_padded_rows(
    values: Rows,
    gates: Rows,
    gain: Sequence[float],
    epsilon: float,
    *,
    padded_width: int,
) -> Result:
    """Model C2 semantics: zero-pad reduction work, retain logical divisor."""
    value_rows, gate_rows, gains, width = _checked_inputs(values, gates, gain, epsilon)
    _checked_padding(padded_width, width)
    return tuple(
        _row_output(
            row, gate, gains, epsilon, row + (0.0,) * (padded_width - width), width
        )
        for row, gate in zip(value_rows, gate_rows, strict=True)
    )


def partitioned_reduction_rows(
    values: Rows,
    gates: Rows,
    gain: Sequence[float],
    epsilon: float,
    *,
    partitions: int,
) -> Result:
    """Finite C3/C4 models: combine partial square sums before normalizing.

    Both levels use ``math.fsum`` on Python floats. This is a Float64
    mathematical comparison, not a GPU reduction schedule or FP32 simulation.
    """
    value_rows, gate_rows, gains, width = _checked_inputs(values, gates, gain, epsilon)
    if isinstance(partitions, bool) or not isinstance(partitions, int):
        raise ValueError("partitions must be an integer")
    if partitions < 1 or partitions > width:
        raise ValueError("partitions must lie between one and logical width")
    outputs = []
    for row, gate in zip(value_rows, gate_rows, strict=True):
        partials = tuple(
            math.fsum(
                x * x
                for x in row[i * width // partitions : (i + 1) * width // partitions]
            )
            for i in range(partitions)
        )
        denominator = math.sqrt(math.fsum(partials) / width + epsilon)
        outputs.append(
            tuple(
                _sigmoid(g) * weight * value / denominator
                for value, g, weight in zip(row, gate, gains, strict=True)
            )
        )
    return tuple(outputs)


def padded_divisor_defect_rows(
    values: Rows,
    gates: Rows,
    gain: Sequence[float],
    epsilon: float,
    *,
    padded_width: int,
) -> Result:
    """Model C1 defect: zero-pad values but divide by execution width."""
    value_rows, gate_rows, gains, width = _checked_inputs(values, gates, gain, epsilon)
    _checked_padding(padded_width, width)
    return tuple(
        _row_output(
            row,
            gate,
            gains,
            epsilon,
            row + (0.0,) * (padded_width - width),
            padded_width,
        )
        for row, gate in zip(value_rows, gate_rows, strict=True)
    )


def unmasked_padding_defect_rows(
    values: Rows,
    gates: Rows,
    gain: Sequence[float],
    epsilon: float,
    *,
    padded_width: int,
    padding_value: float = 1.0,
) -> Result:
    """Separate constructed defect: extra nonzero values enter the reduction."""
    value_rows, gate_rows, gains, width = _checked_inputs(values, gates, gain, epsilon)
    _checked_padding(padded_width, width)
    if not math.isfinite(padding_value):
        raise ValueError("padding value must be finite")
    return tuple(
        _row_output(
            row,
            gate,
            gains,
            epsilon,
            row + (padding_value,) * (padded_width - width),
            width,
        )
        for row, gate in zip(value_rows, gate_rows, strict=True)
    )


def combined_padding_defect_rows(
    values: Rows,
    gates: Rows,
    gain: Sequence[float],
    epsilon: float,
    *,
    padded_width: int,
    padding_value: float,
) -> Result:
    """Construct both errors: include extra lanes and divide by padded width."""
    value_rows, gate_rows, gains, width = _checked_inputs(values, gates, gain, epsilon)
    _checked_padding(padded_width, width)
    if not math.isfinite(padding_value):
        raise ValueError("padding value must be finite")
    return tuple(
        _row_output(
            row,
            gate,
            gains,
            epsilon,
            row + (padding_value,) * (padded_width - width),
            padded_width,
        )
        for row, gate in zip(value_rows, gate_rows, strict=True)
    )


if __name__ == "__main__":
    tiny = normalized_gated_rows(
        ((3.0, 4.0, 0.0),),
        ((0.0, math.log(3.0), -math.log(3.0)),),
        (1.0, 2.0, 1.0),
        1.0,
    )
    print("tiny-exact:", tiny[0])
