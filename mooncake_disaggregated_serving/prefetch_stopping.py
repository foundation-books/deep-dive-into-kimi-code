"""Online stopping decisions evaluated on supplied arrivals, in milliseconds."""

from dataclasses import dataclass
from math import isfinite


def continuous_boundary(local: int, completed_ends: set[int], page: int = 4) -> int:
    """Stop at the first missing page, even when later pages have arrived."""
    if page <= 0 or local < 0 or local % page:
        raise ValueError("aligned nonnegative local boundary required")
    boundary = local
    while boundary + page in completed_ends:
        boundary += page
    return boundary


def joint_boundary(local: int, groups: tuple[set[int], ...], page: int = 4) -> int:
    """Minimum continuous boundary over all declared required rank/state groups."""
    if not groups:
        raise ValueError("declare every required group")
    return min(continuous_boundary(local, group, page) for group in groups)


def timeout_reached(
    now_s: float,
    start_s: float,
    tokens: int,
    base_s: float,
    per_ki_s: float,
    maximum_s: float,
) -> bool:
    """Pinned SGLang's capped timeout, in seconds, with a strict comparison."""
    values = (now_s, start_s, base_s, per_ki_s, maximum_s)
    if not all(isfinite(v) and v >= 0 for v in values) or tokens < 0:
        raise ValueError("nonnegative finite clock/configuration required")
    budget = min(maximum_s, base_s + per_ki_s * tokens / 1024)
    return now_s - start_s > budget


@dataclass(frozen=True)
class StopResult:
    """Separate stopping, GPU installation, prefill readiness and first token."""

    stop: float
    boundary: int
    installed: float
    prefill_ready: float
    first_token: float


def _validate_arrivals(
    arrivals: tuple[tuple[float, int], ...], local: int, target: int
) -> None:
    if target < local or local % 4 or target % 4:
        raise ValueError("aligned target no shorter than local prefix required")
    if any(end <= local or end > target or end % 4 for _, end in arrivals):
        raise ValueError("arrival must end a page within the fetch range")


def stopping_trace(
    policy: str,
    arrivals: tuple[tuple[float, int], ...],
    costs: dict[int, float],
    gpu_ready: float = 3,
    deadline: float = 6,
    installation: float = 1,
    downstream: float = 4,
    local: int = 4,
    target: int = 16,
) -> StopResult:
    """Observe arrivals and a deadline; choose without inspecting future hits.

    The finite fixture guarantees completion of the declared fetch target.
    Deadline observation is idealized as instantaneous at equality; actual
    scheduler polls use timeout_reached and may stop later. Arrivals name
    completed four-token pages, so holes cannot extend the usable prefix.
    """
    if policy not in {"best_effort", "timeout", "wait_complete"}:
        raise ValueError("unknown policy")
    times = (gpu_ready, deadline, installation, downstream, *(t for t, _ in arrivals))
    if not all(isfinite(t) and t >= 0 for t in times):
        raise ValueError("nonnegative finite times required")
    _validate_arrivals(arrivals, local, target)
    polls = sorted(
        {gpu_ready, max(gpu_ready, deadline), *(max(gpu_ready, t) for t, _ in arrivals)}
    )
    for now in polls:
        ends = {end for time, end in arrivals if time <= now}
        boundary = continuous_boundary(local, ends)
        stop = policy == "best_effort" or boundary == target
        stop = stop or (policy == "timeout" and now >= deadline)
        if stop:
            residual = costs[boundary]
            if not isfinite(residual) or residual < 0:
                raise ValueError("nonnegative finite residual cost required")
            ready = now + installation + residual
            return StopResult(
                now, boundary, now + installation, ready, ready + downstream
            )
    raise ValueError("trace does not complete the requested fetch")
