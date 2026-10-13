"""Dependency-free integrated reference semantics for Chapter 10.

The facade composes exact-rational types, operators, runtime transitions, and
checks through ordinary package imports. It is not the released
K3 model, a trained checkpoint, a production cache layout, a low-precision
kernel, or a performance benchmark.
"""

from __future__ import annotations

from .reference_decode import propose_token
from .reference_checks import (
    all_correctness_checks,
    context_ledger,
    logical_state_ledger,
    precision_contract,
    precision_sensitivity,
    route_ledger,
    running_summary,
    state_lifetime_faults,
    structural_faults,
)
from .reference_operators import validate_depth_state
from .reference_runtime import (
    cached_decode,
    full_prefix_decode,
    incremental_equivalence,
    integrated_forward_equivalence,
    run_forward,
    running_prefill,
)
from .reference_types import (
    DECODE_LABEL,
    GLOBAL_RECORD_WIDTH,
    MODEL_WIDTH,
    ROUTED_EXPERTS,
    RUNNING_LABELS,
    SELECTED_EXPERTS,
    SHARED_EXPERTS,
    TOY_BLOCK_SIZE,
    TOY_SCHEDULE,
    DepthState,
    ForwardTrace,
    GlobalLayerState,
    KdaLayerState,
    LayerCheckpoint,
    PrefixState,
    RouteRecord,
    TokenRow,
    channel_schedule,
    decode_token,
    depth_partition,
    empty_prefix_state,
    full_scale_schedule_summary,
    prefix_identity,
    running_tokens,
    splice_running_prefix,
    terminal_global_schedule,
    validate_input_segment,
    validate_origin_sequence,
    validate_prefix_state,
    validate_schedule,
)


__all__ = [
    "DECODE_LABEL",
    "GLOBAL_RECORD_WIDTH",
    "MODEL_WIDTH",
    "ROUTED_EXPERTS",
    "RUNNING_LABELS",
    "SELECTED_EXPERTS",
    "SHARED_EXPERTS",
    "TOY_BLOCK_SIZE",
    "TOY_SCHEDULE",
    "DepthState",
    "ForwardTrace",
    "GlobalLayerState",
    "KdaLayerState",
    "LayerCheckpoint",
    "PrefixState",
    "RouteRecord",
    "TokenRow",
    "all_correctness_checks",
    "cached_decode",
    "channel_schedule",
    "context_ledger",
    "decode_token",
    "depth_partition",
    "empty_prefix_state",
    "full_prefix_decode",
    "full_scale_schedule_summary",
    "incremental_equivalence",
    "integrated_forward_equivalence",
    "logical_state_ledger",
    "precision_contract",
    "precision_sensitivity",
    "prefix_identity",
    "propose_token",
    "route_ledger",
    "run_forward",
    "running_prefill",
    "running_summary",
    "running_tokens",
    "splice_running_prefix",
    "state_lifetime_faults",
    "structural_faults",
    "terminal_global_schedule",
    "validate_depth_state",
    "validate_input_segment",
    "validate_origin_sequence",
    "validate_prefix_state",
    "validate_schedule",
]
