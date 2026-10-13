"""Accepted-token proposal semantics for the integrated K3 toy runtime."""

from __future__ import annotations

from .reference_runtime import run_forward
from .reference_types import ForwardTrace, PrefixState, TokenRow


def propose_token(
    state: PrefixState,
    token: TokenRow,
    *,
    accept: bool,
) -> tuple[ForwardTrace, PrefixState]:
    """Evaluate a proposal and commit state only when accepted."""

    trace = run_forward(
        (token,),
        state=state,
        forward_id=f"proposal-{token.position}",
    )
    return trace, trace.prefix_state if accept else state


__all__ = ["propose_token"]
