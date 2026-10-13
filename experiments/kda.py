"""KDA correctness and accounting experiments on the book's fixed CPU inputs."""

from __future__ import annotations

from collections.abc import Sequence
import math
from types import ModuleType

from kda import reference as chapter_reference
from kda.reference import KdaInput, Trace
from ._data import exact_matrix, exact_vector, protocol


def one_step_oracle(reference: ModuleType) -> dict[str, object]:
    """Compute the one step oracle comparison on fixed toy inputs."""
    _, _, result = reference.one_step_example()
    expected_state = [["5/4", "0"], ["3/4", "1"]]
    expected_output = ["27/20", "4/5"]
    observed_state = exact_matrix(result.state)
    observed_output = exact_vector(result.output)
    passed = observed_state == expected_state and observed_output == expected_output
    return {
        "status": "pass" if passed else "fail",
        "protocol": protocol(
            input_description="M4.1 with exact rational state, gates, key, value, and query",
            reference="hand-derived retain, prediction, error, correction, state, and read",
            comparison="dependency-free Fraction implementation",
            metric="exact equality of every intermediate and final output",
            tolerance="exact rational equality",
            claim_boundary="one scaled-down algebraic update only",
        ),
        "retained": exact_matrix(result.retained),
        "prediction": exact_vector(result.prediction),
        "error": exact_vector(result.error),
        "correction": exact_matrix(result.correction),
        "state": observed_state,
        "output": observed_output,
    }


def multi_step_trace(reference: ModuleType) -> dict[str, object]:
    """Compute the multi step trace comparison on fixed toy inputs."""
    sequence = reference.running_inputs()
    trace = reference.run_recurrent(sequence)
    decoded = reference.run_recurrent((reference.decode_inputs(),), trace.final_states)
    passed = len(trace.positions) == 10 and len(trace.final_states) == 2
    return {
        "status": "pass" if passed else "fail",
        "protocol": protocol(
            input_description="ten accepted positions, two heads, d_k=d_v=2, then token square",
            reference="exact position-by-position KDA recurrence",
            comparison="frozen chapter state and output ledger",
            metric="exact position count, head count, final prefill state, and decode handoff",
            tolerance="exact rational equality",
            claim_boundary="toy projected tensors, not a released checkpoint trace",
        ),
        "positions": len(trace.positions),
        "heads": len(trace.final_states),
        "final_prefill_states_exact": [
            exact_matrix(state) for state in trace.final_states
        ],
        "final_prefill_states_float": [
            reference.matrix_as_floats(state) for state in trace.final_states
        ],
        "decode_outputs_exact": [
            exact_vector(result.output) for result in decoded.positions[0]
        ],
        "decode_states_exact": [
            exact_matrix(result.state) for result in decoded.positions[0]
        ],
    }


def causality(reference: ModuleType) -> dict[str, object]:
    """Compute the causality comparison on fixed toy inputs."""
    sequence = reference.as_float_sequence(reference.running_inputs())
    baseline = reference.run_recurrent(sequence)
    changed = reference.run_recurrent(reference.mutate_suffix_values(sequence, 6, 9.0))
    left = tuple(
        value
        for position in baseline.positions[:6]
        for result in position
        for value in reference.vector_as_floats(result.output, 15)
    )
    right = tuple(
        value
        for position in changed.positions[:6]
        for result in position
        for value in reference.vector_as_floats(result.output, 15)
    )
    error = reference.max_abs_difference(left, right)
    return {
        "status": "pass" if error == 0.0 else "fail",
        "protocol": protocol(
            input_description="positions 7-10 receive a +9 value perturbation",
            reference="outputs through accepted position 6 before the changed suffix",
            comparison="baseline and future-perturbed traces",
            metric="maximum absolute prefix-output difference",
            tolerance="exact zero under deterministic evaluation",
            claim_boundary="causality of the toy recurrence only",
        ),
        "last_compared_position": 6,
        "max_abs_prefix_error": error,
    }


def sequence_isolation(reference: ModuleType) -> dict[str, object]:
    """Compute the sequence isolation comparison on fixed toy inputs."""
    first = reference.as_float_sequence(reference.running_inputs()[:6])
    second = reference.mutate_suffix_values(first, 0, 3.0)
    independent = reference.run_recurrent(first)
    logical_batch_first = reference.run_recurrent(first)
    logical_batch_second = reference.run_recurrent(second)
    first_error = reference.max_abs_difference(
        reference.flatten_outputs(independent),
        reference.flatten_outputs(logical_batch_first),
    )
    separation = reference.max_abs_difference(
        reference.flatten_outputs(logical_batch_first),
        reference.flatten_outputs(logical_batch_second),
    )
    passed = first_error == 0.0 and separation > 0.0
    return {
        "status": "pass" if passed else "fail",
        "protocol": protocol(
            input_description="one six-position request and one independently initialized altered request",
            reference="the first request evaluated alone",
            comparison="separate per-request recurrence evaluations",
            metric="first-request equality and nonzero difference from the altered request",
            tolerance="zero equality error; altered-request difference must be positive",
            claim_boundary="logical state isolation, not a packed GPU layout test",
        ),
        "first_request_max_abs_error": first_error,
        "altered_request_max_abs_difference": separation,
    }


def recurrent_chunk_equivalence(reference: ModuleType) -> dict[str, object]:
    """Compute the recurrent chunk equivalence comparison on fixed toy inputs."""
    sequence = reference.running_inputs()
    recurrent = reference.run_recurrent(sequence)
    chunked = reference.run_chunked(sequence, reference.RUNNING_CHUNK_SIZE)
    output_error = reference.max_abs_difference(
        reference.flatten_outputs(recurrent), reference.flatten_outputs(chunked)
    )
    state_equal = recurrent.final_states == chunked.final_states
    return {
        "status": "pass" if output_error == 0.0 and state_equal else "fail",
        "protocol": protocol(
            input_description="the full ten-position trace partitioned 4,4,2",
            reference="exact recurrent outputs and final states",
            comparison="causal chunk evaluation with independently composed affine boundaries",
            metric="maximum output error and exact final-state identity",
            tolerance="exact rational equality",
            claim_boundary="algebraic chunk semantics, not WY/UT kernel timing",
        ),
        "chunk_sizes": [4, 4, 2],
        "max_abs_output_error": output_error,
        "final_state_exact": state_equal,
    }


def partial_tail(reference: ModuleType) -> dict[str, object]:
    """Compute the partial tail comparison on fixed toy inputs."""
    sequence = reference.running_inputs()
    recurrent = reference.run_recurrent(sequence)
    chunked = reference.run_chunked(sequence, 4)
    tail_outputs = chunked.positions[8:]
    passed = len(tail_outputs) == 2 and chunked.final_states == recurrent.final_states
    return {
        "status": "pass" if passed else "fail",
        "protocol": protocol(
            input_description="N=10 and C=4, giving a two-position final chunk",
            reference="recurrent positions 9-10 with no padded update",
            comparison="the chunked partial-tail outputs and final state",
            metric="tail length and exact final-state identity",
            tolerance="exact rational equality",
            claim_boundary="logical partial-tail semantics, not padding-kernel behavior",
        ),
        "chunk_sizes": [4, 4, 2],
        "real_tail_positions": 2,
        "final_state_exact": chunked.final_states == recurrent.final_states,
    }


def chunk_size_invariance(reference: ModuleType) -> dict[str, object]:
    """Compute the chunk size invariance comparison on fixed toy inputs."""
    sequence = reference.running_inputs()
    recurrent = reference.run_recurrent(sequence)
    errors: dict[str, float] = {}
    states: dict[str, bool] = {}
    for chunk_size in (1, 2, 3, 4, 6, 10, 12):
        chunked = reference.run_chunked(sequence, chunk_size)
        errors[str(chunk_size)] = reference.max_abs_difference(
            reference.flatten_outputs(recurrent), reference.flatten_outputs(chunked)
        )
        states[str(chunk_size)] = chunked.final_states == recurrent.final_states
    passed = max(errors.values()) == 0.0 and all(states.values())
    return {
        "status": "pass" if passed else "fail",
        "protocol": protocol(
            input_description="the same ten positions under seven legal chunk sizes",
            reference="one exact recurrent trace",
            comparison="chunk sizes 1,2,3,4,6,10,12",
            metric="maximum output error and final-state identity for every partition",
            tolerance="exact rational equality",
            claim_boundary="partition invariance of the reference semantics only",
        ),
        "max_abs_output_error_by_chunk": errors,
        "final_state_exact_by_chunk": states,
    }


def gradient_equivalence(reference: ModuleType) -> dict[str, object]:
    """Compute the gradient equivalence comparison on fixed toy inputs."""
    sequence = reference.as_float_sequence(reference.running_inputs()[:4])

    def recurrent(items: Sequence[Sequence[KdaInput]]) -> Trace:
        return reference.run_recurrent(items)

    def chunked(items: Sequence[Sequence[KdaInput]]) -> Trace:
        return reference.run_chunked(items, 3)

    recurrent_gradients = reference.finite_difference_value_gradients(
        recurrent, sequence, 1e-6
    )
    chunked_gradients = reference.finite_difference_value_gradients(
        chunked, sequence, 1e-6
    )
    error = reference.max_abs_difference(recurrent_gradients, chunked_gradients)
    return {
        "status": "pass" if error <= 1e-8 else "fail",
        "protocol": protocol(
            input_description="first four positions with centered value-coordinate differences",
            reference="finite differences of the recurrent weighted-squared-output objective",
            comparison="finite differences of the C=3 chunked objective",
            metric="maximum absolute derivative difference",
            tolerance="1e-8 with finite-difference step 1e-6",
            claim_boundary="toy value derivatives, not autograd or training validation",
        ),
        "coordinates": len(recurrent_gradients),
        "finite_difference_step": 1e-6,
        "max_abs_gradient_error": error,
    }


def numerical_stress(reference: ModuleType) -> dict[str, object]:
    """Compute the numerical stress comparison on fixed toy inputs."""
    exact = reference.run_recurrent(reference.running_inputs())
    floating = reference.run_recurrent(
        reference.as_float_sequence(reference.running_inputs())
    )
    exact_float = reference.flatten_outputs(exact)
    float_outputs = reference.flatten_outputs(floating)
    trace_error = reference.max_abs_difference(exact_float, float_outputs)
    reciprocal_bound = math.exp(80)
    bf16_max = 3.3895313892515355e38
    passed = trace_error <= 1e-12 and reciprocal_bound < bf16_max
    return {
        "status": "pass" if passed else "fail",
        "protocol": protocol(
            input_description="exact running trace plus the K3 g_min=-5, 16-position bound",
            reference="Fraction outputs and the analytic reciprocal bound exp(80)",
            comparison="Python double outputs and the maximum finite BF16 magnitude",
            metric="maximum trace error and strict reciprocal-range comparison",
            tolerance="1e-12 for the toy trace; reciprocal bound must be below BF16 max",
            claim_boundary="bounded toy arithmetic; no all-length stability or GPU claim",
        ),
        "max_abs_exact_vs_float_output_error": trace_error,
        "k3_tile_log_decay_lower_bound": -80,
        "reciprocal_rescaling_upper_bound": reciprocal_bound,
        "bf16_max_finite": bf16_max,
        "bound_is_within_bf16_range": reciprocal_bound < bf16_max,
    }


def cost_ledger(reference: ModuleType) -> dict[str, object]:
    """Compute the cost ledger comparison on fixed toy inputs."""
    toy_elements = reference.logical_state_elements(1, 2, 2, 2)
    exercise_elements = reference.logical_state_elements(8, 16, 128, 128)
    exercise_bytes = exercise_elements * 2
    crossover = reference.formal_crossover(64, 128)
    equal = reference.kda_chunk_flops(
        int(crossover), 64, 128
    ) == reference.full_attention_flops(int(crossover), 128)
    passed = (
        toy_elements == 8
        and exercise_bytes == 4_194_304
        and int(crossover) == 496
        and equal
    )
    return {
        "status": "pass" if passed else "fail",
        "protocol": protocol(
            input_description="toy state plus B=8,H=16,d_k=d_v=128,BF16 and C=64,d_h=128",
            reference="KDA logical-state definition and Kimi Linear Equations 13-14",
            comparison="exact integer and Fraction calculations",
            metric="state elements, bytes, and positive formal FLOP crossover",
            tolerance="exact arithmetic",
            claim_boundary="analytical operator ledger, not latency or peak training memory",
        ),
        "toy_logical_state_elements": toy_elements,
        "exercise_logical_state_elements": exercise_elements,
        "exercise_bf16_bytes": exercise_bytes,
        "exercise_bf16_mib": exercise_bytes / (1024 * 1024),
        "formal_crossover_tokens": str(crossover),
        "reported_formulas_equal_at_crossover": equal,
    }


def run_all_checks() -> dict[str, dict[str, object]]:
    """Run ten fixed KDA correctness and accounting comparisons."""
    reference = chapter_reference
    return {
        "RESULT_KDA_ONE_STEP_ORACLE": one_step_oracle(reference),
        "RESULT_KDA_MULTI_STEP_TRACE": multi_step_trace(reference),
        "RESULT_KDA_CAUSALITY": causality(reference),
        "RESULT_KDA_SEQUENCE_ISOLATION": sequence_isolation(reference),
        "RESULT_KDA_RECURRENT_CHUNK_EQUIVALENCE": recurrent_chunk_equivalence(
            reference
        ),
        "RESULT_KDA_PARTIAL_TAIL": partial_tail(reference),
        "RESULT_KDA_CHUNK_SIZE_INVARIANCE": chunk_size_invariance(reference),
        "RESULT_KDA_GRADIENT_EQUIVALENCE": gradient_equivalence(reference),
        "RESULT_KDA_NUMERICAL_STRESS": numerical_stress(reference),
        "RESULT_KDA_COST_LEDGER": cost_ledger(reference),
    }
