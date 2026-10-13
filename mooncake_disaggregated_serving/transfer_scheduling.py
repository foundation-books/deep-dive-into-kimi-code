"""Deterministic slice-allocation arithmetic; no transport or measured rates."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Device:
    """One already-eligible NIC; all quantities use consistent byte/time units."""

    name: str
    outstanding: float
    bandwidth: float
    locality: float = 1.0

    def score(self, slice_size: float) -> float:
        """Return predicted completion time multiplied by locality cost."""
        values = (self.outstanding, self.bandwidth, self.locality, slice_size)
        if not all(isfinite(v) for v in values):
            raise ValueError("finite inputs required")
        if self.outstanding < 0 or min(values[1:]) <= 0:
            raise ValueError("nonnegative work and positive rate/size/penalty required")
        return (self.outstanding + slice_size) / self.bandwidth * self.locality


def update_estimate(old: float, observed: float, alpha: float) -> float:
    """EWMA with alpha weighting the old value; runtime clamps are omitted."""
    if not all(isfinite(v) for v in (old, observed, alpha)):
        raise ValueError("finite inputs required")
    if min(old, observed) <= 0 or not 0 <= alpha <= 1:
        raise ValueError("positive rates and alpha in [0,1] required")
    return alpha * old + (1 - alpha) * observed


def allocate_slices(
    devices: tuple[Device, ...],
    total: int,
    count: int,
    slice_size: int,
    call_index: int = 1,
    epsilon: float = 1e-12,
) -> tuple[tuple[str, int], ...]:
    """Inverse-score shares or periodic probes, charging the actual final tail.

    Caller supplies eligible devices and a legal slice plan. Jitter is omitted;
    ties use names. call_index counts multi-slice calls on one modeled thread.
    The final slice may exceed slice_size when a short tail has been folded in.
    """
    if not devices or len({d.name for d in devices}) != len(devices):
        raise ValueError("distinct eligible devices required")
    if min(count, slice_size, call_index) <= 0 or total <= (count - 1) * slice_size:
        raise ValueError("positive, nonempty slices required")
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("nonnegative finite epsilon required")
    ranked = sorted(devices, key=lambda d: (d.score(slice_size), d.name))
    if count == 1:
        names = [ranked[0].name]
    elif call_index % 100 == 0:
        names = [ranked[i % len(ranked)].name for i in range(count)]
    else:
        weights = [1 / (d.score(slice_size) + epsilon) for d in ranked]
        weight_sum = sum(weights)
        names = []
        for device, weight in zip(ranked, weights, strict=True):
            share = min(int(count * weight / weight_sum), count - len(names))
            names.extend([device.name] * share)
        names.extend([ranked[0].name] * (count - len(names)))
    lengths = [slice_size] * (count - 1) + [total - (count - 1) * slice_size]
    return tuple(zip(names, lengths, strict=True))
