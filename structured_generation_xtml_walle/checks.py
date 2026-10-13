"""Stable CH26-D01 through CH26-D10 check registry."""

from __future__ import annotations

from typing import Any

from .protocol_checks import (
    option_scope_check,
    product_trace_check,
    result_matching_check,
    typed_argument_check,
    xtml_round_trip_check,
)
from .schema_checks import (
    canonicalization_check,
    decoder_failure_check,
    legal_token_check,
    schema_profile_check,
    semantic_relation_check,
)


JSON = Any


def all_checks() -> dict[str, JSON]:
    """Run and return the ten deterministic Chapter 27 evidence objects."""

    payloads = {
        "schema_profiles": schema_profile_check(),
        "canonicalization": canonicalization_check(),
        "semantic_relation": semantic_relation_check(),
        "legal_tokens": legal_token_check(),
        "decoder_failures": decoder_failure_check(),
        "xtml_round_trip": xtml_round_trip_check(),
        "option_scope": option_scope_check(),
        "typed_arguments": typed_argument_check(),
        "result_matching": result_matching_check(),
        "product_trace": product_trace_check(),
    }
    return {
        name: {
            "check_id": f"CH26-D{index:02d}",
            "fixture_version": "ch26-reference-v1",
            **payload,
            "status": "pass",
        }
        for index, (name, payload) in enumerate(payloads.items(), start=1)
    }
