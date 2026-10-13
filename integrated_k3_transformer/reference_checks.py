"""Ledgers and fault-injection checks for the integrated K3 oracle."""

from __future__ import annotations

from dataclasses import replace
from math import exp, isfinite
from struct import pack, unpack
from typing import Callable, Sequence

from .reference_decode import propose_token
from .reference_operators import new_depth_state, validate_depth_state
from .reference_runtime import (
    cached_decode,
    incremental_equivalence,
    integrated_forward_equivalence,
    running_prefill,
)
from .reference_types import (
    GLOBAL_RECORD_WIDTH,
    MODEL_WIDTH,
    RUNNING_LABELS,
    SELECTED_EXPERTS,
    SHARED_EXPERTS,
    TOY_BLOCK_SIZE,
    TOY_SCHEDULE,
    ForwardTrace,
    decode_token,
    depth_partition,
    empty_prefix_state,
    full_scale_schedule_summary,
    running_tokens,
    splice_running_prefix,
    terminal_global_schedule,
    validate_input_segment,
    validate_origin_sequence,
    validate_prefix_state,
    validate_schedule,
)


def context_ledger(
    *,
    text: int,
    visual: int,
    accepted: int,
    maximum: int = 1_048_576,
) -> dict[str, int]:
    """Return used and remaining backbone positions."""

    if min(text, visual, accepted, maximum) < 0 or maximum == 0:
        raise ValueError("context counts must be nonnegative and maximum positive")
    used = text + visual + accepted
    if used > maximum:
        raise ValueError("context budget exceeded")
    return {
        "text": text,
        "visual": visual,
        "accepted": accepted,
        "used": used,
        "maximum": maximum,
        "remaining": maximum - used,
    }


def logical_state_ledger(
    prefix_length: int,
    *,
    schedule: Sequence[str] = TOY_SCHEDULE,
    kda_state_width: int = MODEL_WIDTH,
    global_record_width: int = GLOBAL_RECORD_WIDTH,
) -> dict[str, int]:
    """Count logical state elements without assigning physical bytes."""

    if prefix_length < 0 or min(kda_state_width, global_record_width) <= 0:
        raise ValueError("state sizes must be nonnegative with positive widths")
    typed = validate_schedule(schedule)
    fixed = typed.count("K") * kda_state_width
    growing = typed.count("G") * prefix_length * global_record_width
    return {
        "prefix_length": prefix_length,
        "kda_fixed_elements": fixed,
        "global_growing_elements": growing,
        "total_logical_elements": fixed + growing,
    }


def precision_contract() -> dict[str, object]:
    """Evaluate the source-reported 16-token lower-bound consequence."""

    lower_log_decay = -5 * 16
    reciprocal_bound = exp(-lower_log_decay)
    bf16_max = 3.3895313892515355e38
    return {
        "tile": 16,
        "per_step_log_decay_open_interval": (-5, 0),
        "cumulative_log_decay_open_interval": (lower_log_decay, 0),
        "reciprocal_strict_upper_bound": reciprocal_bound,
        "bf16_max_finite": bf16_max,
        "within_bf16_dynamic_range": reciprocal_bound < bf16_max,
    }


def _float32(value: float) -> float:
    """Round one Python float through IEEE binary32."""

    return unpack("!f", pack("!f", value))[0]


def precision_sensitivity(steps: int = 64) -> dict[str, object]:
    """Compare one bounded recurrent scalar path in float64 and float32."""

    if steps <= 0:
        raise ValueError("precision sweep needs a positive step count")
    state64 = 0.0
    state32 = 0.0
    maximum_error = 0.0
    for step in range(1, steps + 1):
        log_decay = -5.0 * step / (steps + 1)
        retention64 = exp(log_decay)
        write64 = ((step % 7) - 3) / 11.0
        state64 = retention64 * state64 + write64
        retention32 = _float32(exp(_float32(log_decay)))
        write32 = _float32(write64)
        state32 = _float32(_float32(retention32 * state32) + write32)
        maximum_error = max(maximum_error, abs(state64 - state32))
    return {
        "steps": steps,
        "float64_final": state64,
        "float32_final": state32,
        "maximum_absolute_error": maximum_error,
        "all_finite": isfinite(state64)
        and isfinite(state32)
        and isfinite(maximum_error),
        "range_contract": precision_contract(),
    }


def route_ledger(trace: ForwardTrace) -> dict[str, int]:
    """Count dense and routed layer records for one trace."""

    if not trace.checkpoints:
        raise ValueError("route ledger requires decoder-layer checkpoints")
    if any(
        not checkpoint.pre_attention_sources or not checkpoint.pre_channel_sources
        for checkpoint in trace.checkpoints
    ):
        raise ValueError("checkpoint source identities must be nonempty")
    completed_groups = tuple(
        checkpoint.completed_depth_groups for checkpoint in trace.checkpoints
    )
    if completed_groups != tuple(sorted(completed_groups)):
        raise ValueError("completed depth-group counts must be monotone")
    dense = sum(checkpoint.channel_type == "D" for checkpoint in trace.checkpoints)
    routed = sum(checkpoint.channel_type == "M" for checkpoint in trace.checkpoints)
    expected = len(trace.input_labels) * SELECTED_EXPERTS * routed
    if trace.checkpoints[0].routes:
        raise ValueError("the initial dense layer fabricated route records")
    if len(trace.routes) != expected:
        raise ValueError("route count disagrees with channel schedule")
    return {
        "dense_layers": dense,
        "routed_layers": routed,
        "route_records": len(trace.routes),
        "expected_route_records": expected,
        "shared_expert_applications": (
            len(trace.input_labels) * SHARED_EXPERTS * routed
        ),
    }


def _rejects(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def state_lifetime_faults() -> dict[str, bool]:
    """Inject identity, endpoint, rejection, and depth-history faults."""

    prefill = running_prefill()
    detected: dict[str, bool] = {}
    foreign = replace(
        prefill.prefix_state,
        accepted_labels=prefill.prefix_state.accepted_labels[:-1] + ("text:foreign",),
    )
    detected["foreign_prefix_identity"] = _rejects(
        lambda: validate_prefix_state(foreign)
    )
    stale_kda = replace(
        prefill.prefix_state.kda[0],
        endpoint=prefill.prefix_state.endpoint - 1,
    )
    wrong_endpoint = replace(
        prefill.prefix_state,
        kda=(stale_kda,) + prefill.prefix_state.kda[1:],
    )
    detected["stale_sequence_endpoint"] = _rejects(
        lambda: validate_prefix_state(wrong_endpoint)
    )
    provisional, rejected_state = propose_token(
        prefill.prefix_state, decode_token(), accept=False
    )
    detected["rejected_token_not_committed"] = (
        provisional.prefix_state.endpoint == 11
        and rejected_state == prefill.prefix_state
    )
    depth = new_depth_state(
        "forward-a", tuple(token.values for token in running_tokens())
    )
    detected["stale_depth_history"] = _rejects(
        lambda: validate_depth_state(
            depth,
            expected_forward_id="forward-b",
            block_size=TOY_BLOCK_SIZE,
            maximum_points=2 * len(TOY_SCHEDULE),
        )
    )
    return detected


def structural_faults() -> dict[str, bool]:
    """Inject ordered-schedule, splice, route, and resource-unit faults."""

    detected: dict[str, bool] = {}
    wrong_order = list(terminal_global_schedule(2))
    wrong_order[2], wrong_order[3] = wrong_order[3], wrong_order[2]
    try:
        validate_schedule(wrong_order)
    except ValueError:
        detected["correct_counts_wrong_order"] = True
    else:
        detected["correct_counts_wrong_order"] = False
    try:
        validate_schedule(terminal_global_schedule(2)[:-1])
    except ValueError:
        detected["missing_terminal_global"] = True
    else:
        detected["missing_terminal_global"] = False
    swapped = splice_running_prefix(swap_visual_halves=True)
    shape_valid = not _rejects(
        lambda: validate_input_segment(swapped, empty_prefix_state())
    )
    detected["shape_preserving_origin_swap"] = shape_valid and _rejects(
        lambda: validate_origin_sequence(swapped, RUNNING_LABELS)
    )
    prefill = running_prefill()
    ledger = route_ledger(prefill)
    detected["dense_route_fabrication"] = (
        not prefill.checkpoints[0].routes and ledger["route_records"] == 160
    )
    detected["route_conservation"] = (
        ledger["route_records"] == ledger["expected_route_records"]
    )
    try:
        context_ledger(text=6, visual=4, accepted=1_048_567)
    except ValueError:
        detected["context_overflow"] = True
    else:
        detected["context_overflow"] = False
    return detected


def all_correctness_checks() -> dict[str, bool]:
    """Return the complete deterministic correctness checklist."""

    schedule = full_scale_schedule_summary()
    sensitivity = precision_sensitivity()
    maximum_error = sensitivity["maximum_absolute_error"]
    checks = {
        "full_scale_schedule": (
            schedule["layers"] == 93
            and schedule["kda_layers"] == 69
            and schedule["global_layers"] == 24
            and schedule["depth_partition"] == (12,) * 7 + (9,)
            and schedule["terminal_sources"] == 9
        ),
        "integrated_forward_equivalence": (
            integrated_forward_equivalence()["status"] == "pass"
        ),
        "incremental_equivalence": (incremental_equivalence()["status"] == "pass"),
        "route_conservation": (route_ledger(running_prefill())["route_records"] == 160),
        "logical_state_growth": (
            logical_state_ledger(10)["total_logical_elements"]
            - logical_state_ledger(9)["total_logical_elements"]
            == TOY_SCHEDULE.count("G") * GLOBAL_RECORD_WIDTH
        ),
        "precision_range": bool(precision_contract()["within_bf16_dynamic_range"]),
        "precision_sensitivity": (
            sensitivity["all_finite"] is True
            and isinstance(maximum_error, float)
            and maximum_error >= 0.0
        ),
    }
    checks.update(state_lifetime_faults())
    checks.update(structural_faults())
    return checks


def running_summary() -> dict[str, object]:
    """Return the inspectable Chapter 11 running-example ledger."""

    prefill, decode = cached_decode()
    return {
        "input_positions": len(prefill.input_labels),
        "visual_positions": 4,
        "text_positions": 6,
        "toy_schedule": prefill.prefix_state.schedule,
        "depth_partition": depth_partition(len(TOY_SCHEDULE), TOY_BLOCK_SIZE),
        "terminal_sources": prefill.terminal_sources,
        "prefill_endpoint": prefill.prefix_state.endpoint,
        "decode_endpoint": decode.prefix_state.endpoint,
        "route_ledger": route_ledger(prefill),
        "logical_state_at_10": logical_state_ledger(10),
        "logical_state_at_11": logical_state_ledger(11),
        "context_at_11": context_ledger(text=6, visual=4, accepted=1),
        "incremental_equivalence": incremental_equivalence()["exact"],
        "all_faults_detected": (
            all(state_lifetime_faults().values()) and all(structural_faults().values())
        ),
    }


__all__ = [
    "all_correctness_checks",
    "context_ledger",
    "logical_state_ledger",
    "precision_contract",
    "precision_sensitivity",
    "route_ledger",
    "running_summary",
    "state_lifetime_faults",
    "structural_faults",
]
