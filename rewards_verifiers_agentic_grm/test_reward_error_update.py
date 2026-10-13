"""Independent derivative and probability checks for the printed example."""

from __future__ import annotations

import math

from .reward_error_update import reward_error_comparison


def test_false_acceptance_reverses_wrong_response_change() -> None:
    result = reward_error_comparison()
    truth = result["truth"]
    proxy = result["false_acceptance"]
    assert isinstance(truth, dict) and isinstance(proxy, dict)
    assert truth["probabilities"][1] < 1 / 3 < proxy["probabilities"][1]
    for row in [truth, proxy]:
        assert abs(sum(row["coefficients"])) < 1e-12
        assert all(
            math.isclose(g, f, abs_tol=1e-10)
            for g, f in zip(row["gradient"], row["finite_difference"], strict=True)
        )
        assert math.isclose(sum(row["probabilities"]), 1.0, abs_tol=1e-12)
    assert math.isclose(truth["probabilities"][0], 0.3559130712072203, abs_tol=1e-12)
    assert math.isclose(proxy["probabilities"][1], 0.3442533457435726, abs_tol=1e-12)
