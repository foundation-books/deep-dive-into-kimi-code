"""Finite, book-created Chapter 29 examples; no model or upstream runtime is run."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

Candidate = Literal["original", "b1", "b2", "b3"]
Projection = Literal["point", "horizontal"]


@dataclass(frozen=True)
class CropCase:
    """One finite crop, viewport, point, and declared oracle case."""

    name: str
    origin: tuple[int, int]
    crop_size: tuple[int, int]
    viewport: tuple[int, int]
    point: tuple[int, int]
    expected: tuple[int, int]
    projection: Projection = "point"


A = CropCase("A", (10, 20), (60, 120), (300, 600), (150, 400), (40, 100))
B = CropCase("B", (-5, 7), (40, 60), (200, 300), (60, 40), (7, 15))
C = CropCase("C", (3, 9), (40, 80), (200, 400), (50, 100), (13, 29))
P_A_X = CropCase(
    "P_A_X", A.origin, A.crop_size, A.viewport, A.point, A.expected, "horizontal"
)
P_C_X = CropCase(
    "P_C_X", C.origin, C.crop_size, C.viewport, C.point, C.expected, "horizontal"
)
P_ZERO = CropCase("P_ZERO", (0, 0), (40, 80), (200, 400), (50, 100), (10, 20))
SUITES: tuple[tuple[CropCase, ...], ...] = (
    (A, P_A_X, P_ZERO),
    (C, P_C_X),
)


def offset_output(case: CropCase, candidate: Candidate) -> tuple[int, int]:
    """Evaluate the four explicitly declared offset-case implementations."""
    x0, y0 = case.origin
    wc, hc = case.crop_size
    wv, hv = case.viewport
    u, v = case.point
    x = x0 + wc * u // wv
    scaled_y = hc * v // hv
    if candidate == "b1":
        return x, y0 + scaled_y
    if candidate == "b2":
        return x, x0 + scaled_y
    if candidate == "b3" and (
        case.origin,
        case.crop_size,
        case.viewport,
        case.point,
    ) == (A.origin, A.crop_size, A.viewport, A.point):
        return x, 100
    return x, scaled_y


def passes(case: CropCase, output: tuple[int, int]) -> bool:
    """Use a declared full-point or horizontal-only test-case oracle."""
    if case.projection == "horizontal":
        return output[0] == case.expected[0]
    return output == case.expected


def selection_counts(
    candidate: Candidate,
) -> tuple[int, int, int, int]:
    """Return FP, baseline F, PP, and baseline P across the two suites."""
    fp = failures = pp = baseline_passes = 0
    for suite in SUITES:
        for case in suite:
            baseline_pass = passes(case, offset_output(case, "original"))
            candidate_pass = passes(case, offset_output(case, candidate))
            if baseline_pass:
                baseline_passes += 1
                pp += int(candidate_pass)
            else:
                failures += 1
                fp += int(candidate_pass)
    return fp, failures, pp, baseline_passes


def selection_score(candidate: Candidate) -> Fraction:
    """Apply Kimi-Dev's reported two-term test-time rule to finite book cases."""
    fp, failures, pp, baseline_passes = selection_counts(candidate)
    if failures == 0 or baseline_passes == 0:
        raise ValueError("the worksheet requires both denominator classes")
    return Fraction(fp, failures) + Fraction(pp, baseline_passes)


def scale_output(asymmetric: bool, repaired: bool) -> tuple[int, int]:
    """Evaluate the studio-code-002 vertical-scale witness."""
    crop = (10, 20, 60, 120)
    wv, hv = (300, 400) if asymmetric else (300, 600)
    u, v = (150, 200) if asymmetric else (150, 300)
    x0, y0, wc, hc = crop
    vertical_numerator, vertical_denominator = (hc, hv) if repaired else (wc, wv)
    return x0 + wc * u // wv, y0 + vertical_numerator * v // vertical_denominator


def wrong_round_trip() -> tuple[int, int]:
    """Use the mutually wrong 1/5 inverse and 5 forward scales."""
    wrong_root_y = 20 + 200 // 5
    returned_viewport_v = 5 * (wrong_root_y - 20)
    return wrong_root_y, returned_viewport_v


def edit_match_count(
    content: str, old: str, new: str, *, replace_all: bool = False
) -> tuple[Literal["applied", "absent", "ambiguous"], str, int]:
    """A labeled schematic of the pinned edit matching branches."""
    count = content.count(old)
    if count == 0:
        return "absent", content, 0
    if count > 1 and not replace_all:
        return "ambiguous", content, count
    if replace_all:
        return "applied", content.replace(old, new), count
    return "applied", content.replace(old, new, 1), 1


def main_timing_eligible(compiled: bool, correct: bool, enabled: bool) -> bool:
    """Model only KernelBench's main candidate-performance condition."""
    return compiled and correct and enabled
