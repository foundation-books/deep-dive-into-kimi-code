"""A constructed local buffer timeline extending the four/two-unit MoE witness."""

from .objects import ObjectKey, Residency, sweep_peak


def timeline(replay: bool, eliminate: bool = True) -> tuple[Residency, ...]:
    """One unit is explicitly 256 logical bytes; aliases add no allocation."""
    objects = [
        ("canonical input", 0, 6, 4),
        ("expert intermediates", 1, 6, 2),
        ("upstream adjoint", 3, 6, 2),
        ("parameter partials", 4, 8, 2),
        ("communication", 4, 7, 1),
        ("input adjoint", 5, 7, 4),
    ]
    objects.append(("expert-major input", 4 if replay else 1, 6, 4))
    if not eliminate:
        objects.append(("weighted output", 1, 6, 2))
    return tuple(
        Residency(
            ObjectKey(name, 7, 0, "moe", "buffer", 0),
            0,
            "hbm",
            start,
            stop,
            256 * units,
        )
        for name, start, stop, units in objects
    )


def live_bytes(event: int, replay: bool, eliminate: bool = True) -> int:
    """Count bytes resident at a logical instant using half-open lifetimes."""
    return sum(
        obj.units
        for obj in timeline(replay, eliminate)
        if obj.start <= event < obj.stop
    )


def validate_consumers(order: tuple[str, ...]) -> None:
    """The route adjoint and reconstructed input precede expert backward."""
    required = {
        "replay",
        "route adjoint",
        "expert backward",
        "input return",
        "home reduction",
        "release",
    }
    if set(order) != required or len(order) != len(required):
        raise ValueError("missing or duplicated consumer event")
    for producer, consumer in (
        ("replay", "expert backward"),
        ("route adjoint", "expert backward"),
        ("expert backward", "input return"),
        ("expert backward", "home reduction"),
        ("input return", "release"),
        ("home reduction", "release"),
    ):
        if order.index(producer) >= order.index(consumer):
            raise ValueError("consumer precedes its dependency")


def check_lifetimes() -> None:
    """Check consumer order and saved-versus-replayed logical peak accounting."""
    peaks = []
    for replay in (False, True):
        result = sweep_peak(0, timeline(replay), rank=0, tier="hbm", capacity=10000)
        peaks.append(result.peak)
        assert result.peak == max(live_bytes(t, replay) for t in range(9))
    assert peaks == [19 * 256, 19 * 256]
    assert live_bytes(2, True) == 6 * 256 and live_bytes(2, False) == 10 * 256
    assert max(live_bytes(t, True, False) for t in range(9)) == 21 * 256
    validate_consumers(
        (
            "replay",
            "route adjoint",
            "expert backward",
            "input return",
            "home reduction",
            "release",
        )
    )
    try:
        validate_consumers(
            (
                "replay",
                "expert backward",
                "route adjoint",
                "input return",
                "home reduction",
                "release",
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expert backward consumed an unavailable adjoint")
