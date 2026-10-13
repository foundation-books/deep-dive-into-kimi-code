"""Small, dependency-free probes of answer selection and status adaptation.

These are constructed cases motivated by pinned Math-Verify code. They do not
import or reproduce its parser, SymPy comparison strategies, or timeout code.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Sequence

from .reference_core import VerifierResult

AnswerKind = Literal["number", "relation", "interval", "string"]


def select_answer(
    matches: Sequence[tuple[str, Fraction | None]], *, mode: str
) -> Fraction | str | None:
    """Contrast stopping at a matched but unparsed span with trying another.

    ``matches`` is already ordered by extraction priority. A retained string
    is a fallback, not a symbolic certificate.
    """
    if mode not in {"first_match", "any_match"}:
        raise ValueError("unknown extraction mode")
    fallback: str | None = None
    for raw, parsed in matches:
        if parsed is not None:
            return parsed
        if fallback is None:
            fallback = raw
        if mode == "first_match":
            return fallback
    return fallback


def compare_constructed_answers(
    reference: tuple[AnswerKind, object],
    prediction: tuple[AnswerKind, object],
    *,
    allow_set_relation_comp: bool = False,
) -> bool:
    """Compare one deliberately tiny domain, including directional conversion.

    A relation or interval value is a pair of open bounds. This illustrates
    the source's directional policy without claiming to execute its parser.
    """
    ref_kind, ref_value = reference
    pred_kind, pred_value = prediction
    if ref_kind == pred_kind == "number":
        return isinstance(ref_value, Fraction) and ref_value == pred_value
    if ref_kind == pred_kind == "string":
        return (
            isinstance(ref_value, str) and bool(ref_value) and ref_value == pred_value
        )
    if ref_kind in {"relation", "interval"} and pred_kind in {
        "relation",
        "interval",
    }:
        if not (
            isinstance(ref_value, tuple)
            and isinstance(pred_value, tuple)
            and len(ref_value) == len(pred_value) == 2
        ):
            raise ValueError("constructed interval needs two open bounds")
        if ref_kind == "interval" and pred_kind == "relation":
            return allow_set_relation_comp and ref_value == pred_value
        return ref_value == pred_value
    return False


def adapt_math_result(
    native_result: bool | None,
    *,
    reference_valid: bool,
    candidate_valid: bool,
    error: str | None = None,
) -> VerifierResult:
    """Preserve why a native Boolean was or was not available.

    A bad reference is an evaluator fault; a missing required candidate answer
    is a candidate-schema fault. Neither is inferred from a bare False.
    """
    if not reference_valid:
        return VerifierResult(
            "invalid",
            {"origin": "reference", "cause": "bad reference"},
            "math-adapter-v1",
        )
    if error is not None:
        return VerifierResult(
            "indeterminate", {"origin": "verifier", "cause": error}, "math-adapter-v1"
        )
    if not candidate_valid:
        return VerifierResult(
            "invalid",
            {"origin": "candidate", "cause": "no required answer"},
            "math-adapter-v1",
        )
    if native_result is None:
        raise ValueError("valid evaluation needs a Boolean or a recorded error")
    return VerifierResult(
        "pass" if native_result else "fail",
        {"origin": "comparison", "accepted": native_result},
        "math-adapter-v1",
    )
