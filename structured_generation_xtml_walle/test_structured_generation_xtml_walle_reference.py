"""Tests for the dependency-free Chapter 27 teaching oracle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import unittest

from .reference import (
    CORRUPTION_FIRST_FAILURE,
    LEVELS,
    STUDIO_RECORD,
    STUDIO_RESPONSE,
    STUDIO_RESPONSE_SCHEMA,
    Call,
    Element,
    Event,
    FiniteLanguageRecognizer,
    Option,
    Result,
    all_checks,
    canonical_bytes,
    canonicalize_schema,
    canonicalize_to_fixed_point,
    constrained_distribution,
    effective_options,
    evaluate_token_trace,
    finite_semantic_relation,
    instance_conforms,
    instance_errors,
    normalize_results,
    parse_element_events,
    protocol_errors,
    ProtocolTrace,
    round_trip_element,
    studio_validity,
    validate_schema_profile,
    xtml_argument_text,
    xtml_scalar_type,
    zone_sequence_errors,
)


class Chapter26ReferenceTests(unittest.TestCase):
    def test_d01_schema_profiles(self) -> None:
        self.assertEqual(LEVELS, ("loose", "lite", "strict", "ultra"))
        for level in LEVELS:
            self.assertEqual(validate_schema_profile(STUDIO_RESPONSE_SCHEMA, level), ())
        unsupported = {"type": "string", "oneOf": [{"type": "string"}]}
        self.assertEqual(validate_schema_profile(unsupported, "loose"), ())
        for level in ("lite", "strict", "ultra"):
            self.assertTrue(validate_schema_profile(unsupported, level))
        contradiction = {"type": "integer", "minimum": 3, "maximum": 2}
        self.assertEqual(validate_schema_profile(contradiction, "loose"), ())
        self.assertEqual(validate_schema_profile(contradiction, "lite"), ())
        self.assertIn(
            "$:contradictory-bounds",
            validate_schema_profile(contradiction, "strict"),
        )
        for level in ("strict", "ultra"):
            self.assertTrue(validate_schema_profile(contradiction, level))

    def test_d02_canonicalization_is_atomic_stable_and_nonmutating(self) -> None:
        original = deepcopy(STUDIO_RESPONSE_SCHEMA)
        first = canonicalize_schema(original)
        second = canonicalize_schema(first.candidate)
        self.assertTrue(first.diagnostic.acceptable)
        self.assertEqual(
            canonical_bytes(first.candidate), canonical_bytes(second.candidate)
        )
        self.assertEqual(original, STUDIO_RESPONSE_SCHEMA)
        warning = canonicalize_schema({"type": "string", "oneOf": []})
        self.assertFalse(warning.diagnostic.acceptable)
        self.assertNotIn("oneOf", warning.candidate)
        fixed = canonicalize_to_fixed_point(original)
        self.assertEqual((fixed.status, fixed.attempts), ("stable", 2))
        self.assertEqual(
            canonicalize_to_fixed_point(original, max_attempts=1).status,
            "max-attempts",
        )
        self.assertEqual(
            canonicalize_to_fixed_point({"type": "string", "oneOf": []}).status,
            "rejected",
        )

    def test_d03_stability_does_not_prove_equivalence(self) -> None:
        relation = finite_semantic_relation(
            {"type": "integer", "maximum": 12}, {}, [-1, 12, 13]
        )
        self.assertEqual(relation["relation"], "weakened-on-domain")
        self.assertNotIn(13, relation["original_accepts"])
        self.assertIn(13, relation["candidate_accepts"])

    def test_d04_byte_prefix_legal_tokens_and_distribution(self) -> None:
        recognizer = FiniteLanguageRecognizer((b'{"ok":true}', b'{"ok":false}'))
        vocabulary = {
            "v1": b'{"ok":',
            "v2": b"true}",
            "v3": b"false}",
            "v4": b"true",
            "v5": b"false",
            "v6": b"}",
            "v7": b"x",
        }
        prefix = b'{"ok":'
        self.assertEqual(recognizer.legal_tokens(b"", vocabulary), ("v1",))
        self.assertEqual(
            recognizer.legal_tokens(prefix, vocabulary), ("v2", "v3", "v4", "v5")
        )
        probabilities = {
            "v2": 0.30,
            "v3": 0.10,
            "v4": 0.20,
            "v5": 0.10,
            "v6": 0.15,
            "v7": 0.15,
        }
        constrained = constrained_distribution(
            recognizer, prefix, vocabulary, probabilities
        )
        self.assertTrue(math.isclose(sum(constrained.values()), 1.0))
        self.assertTrue(math.isclose(constrained["v2"], 3 / 7))

    def test_d05_empty_mask_zero_mass_and_stopping(self) -> None:
        recognizer = FiniteLanguageRecognizer((b"yes",))
        vocabulary = {"y": b"y", "x": b"x"}
        with self.assertRaisesRegex(ValueError, "empty legal-token"):
            constrained_distribution(recognizer, b"x", vocabulary, {"y": 1.0})
        with self.assertRaisesRegex(ValueError, "zero-or-nonfinite"):
            constrained_distribution(recognizer, b"", vocabulary, {"y": 0.0})
        self.assertFalse(recognizer.end_is_legal(b"ye"))
        self.assertTrue(recognizer.end_is_legal(b"yes"))
        self.assertEqual(
            evaluate_token_trace(recognizer, (b"y", b"e"), end_requested=True).status,
            "premature-end",
        )
        self.assertEqual(
            evaluate_token_trace(recognizer, (b"y",), end_requested=False).status,
            "interrupted",
        )
        self.assertEqual(
            evaluate_token_trace(recognizer, (b"x",), end_requested=True).status,
            "dead-prefix",
        )

    def test_d06_xtml_oriented_round_trip_and_failures(self) -> None:
        tree = Element("response", children=("literal <|open|>",))
        self.assertEqual(round_trip_element(tree), tree)
        with self.assertRaisesRegex(ValueError, "unclosed"):
            parse_element_events((Event("open", "response"),))
        with self.assertRaisesRegex(ValueError, "misnested"):
            parse_element_events((Event("open", "response"), Event("close", "tools")))

    def test_d07_option_scope(self) -> None:
        options = (
            Option("tools", ["A", "B"], "global"),
            Option("effort", "high", "global"),
            Option("tools", ["A", "B", "C"], "dynamic", 2),
            Option("schema", "S2", "one-shot", 2),
            Option("schema", "S3", "one-shot", 3),
        )
        self.assertEqual(effective_options(options, 1)["tools"], ["A", "B"])
        self.assertEqual(effective_options(options, 2)["schema"], "S2")
        self.assertEqual(effective_options(options, 3)["schema"], "S3")
        self.assertNotIn("schema", effective_options(options, 1))
        self.assertTrue(
            zone_sequence_errors(
                ("global-options", "one-shot-options", "history", "assistant")
            )
        )
        trace = ProtocolTrace(
            zones=("global-options", "history", "one-shot-options", "assistant"),
            channels=("response",),
            calls=(),
            results=(),
            required_tool_call=True,
        )
        self.assertIn("required-tool-call-missing", protocol_errors(trace))

    def test_d08_typed_arguments_and_instance_failures(self) -> None:
        self.assertEqual(xtml_scalar_type(30), "number")
        self.assertEqual(xtml_scalar_type(True), "boolean")
        self.assertEqual(xtml_argument_text("30"), "30")
        self.assertEqual(xtml_argument_text({"x": 1}), '{"x":1}')
        self.assertTrue(instance_conforms(STUDIO_RESPONSE, STUDIO_RESPONSE_SCHEMA))
        quoted = {**STUDIO_RESPONSE, "evidence_count": "11"}
        self.assertIn(
            "$.evidence_count:expected-integer",
            instance_errors(quoted, STUDIO_RESPONSE_SCHEMA),
        )
        extra = {**STUDIO_RESPONSE, "confidence": 0.9}
        self.assertIn(
            "$:additional:confidence", instance_errors(extra, STUDIO_RESPONSE_SCHEMA)
        )
        with self.assertRaisesRegex(TypeError, "non-string JSON object key"):
            xtml_argument_text({1: "not-json"})
        with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
            xtml_argument_text(math.nan)
        with self.assertRaisesRegex(TypeError, "unsupported JSON value"):
            xtml_argument_text((1, 2))

    def test_d09_result_normalization_and_matching_failures(self) -> None:
        calls = (
            Call("a", "fetch", {}, 1),
            Call("b", "check", {}, 2),
        )
        arrival = (Result("b", None, 2), Result("a", None, 1))
        original = deepcopy(arrival)
        outcome = normalize_results(calls, arrival)
        self.assertEqual(outcome.status, "closed")
        self.assertEqual([result.index for result in outcome.results], [1, 2])
        self.assertEqual(normalize_results(calls, outcome.results), outcome)
        self.assertEqual(arrival, original)
        self.assertEqual(
            normalize_results(calls, arrival + (arrival[0],)).diagnostic,
            "duplicate-result",
        )
        self.assertEqual(
            normalize_results(calls, (Result("stale", None, 0),)).status,
            "unresolved",
        )
        self.assertEqual(
            normalize_results(calls, arrival[:1]).diagnostic, "missing-result"
        )
        self.assertEqual(
            normalize_results(
                (calls[0], replace(calls[1], external_id="a")), arrival
            ).diagnostic,
            "ambiguous-call-id",
        )
        self.assertEqual(
            normalize_results(
                (calls[0], replace(calls[1], index=3)), arrival
            ).diagnostic,
            "nonconsecutive-call-index",
        )

    def test_d10_product_trace_corruptions_inventory_and_handoff(self) -> None:
        self.assertEqual(STUDIO_RECORD["record_id"], "studio-contract-001")
        self.assertEqual(
            STUDIO_RECORD["evidence"]["present"],
            ["R1", "R2", "R3", "C1", "C2", "C3", "V1", "V2", "V3", "D1", "D2"],
        )
        self.assertEqual(STUDIO_RECORD["evidence"]["missing"], ["D3"])
        self.assertEqual(
            [tool["name"] for tool in STUDIO_RECORD["tools"]],
            ["fetch_evidence", "run_release_check"],
        )
        self.assertEqual(
            [(call["external_id"], call["index"]) for call in STUDIO_RECORD["calls"]],
            [("call-a", 1), ("call-b", 2)],
        )
        self.assertEqual(
            [result["external_id"] for result in STUDIO_RECORD["result_arrivals"]],
            ["call-b", "call-a"],
        )
        self.assertEqual(
            [(option["key"], option["scope"]) for option in STUDIO_RECORD["options"]],
            [
                ("tools", "global"),
                ("thinking_effort", "global"),
                ("tool_choice", "one-shot"),
                ("response_schema", "one-shot"),
            ],
        )
        self.assertEqual(
            STUDIO_RECORD["channels"], ["tools", "tool-results", "response"]
        )
        self.assertIs(STUDIO_RECORD["response"], STUDIO_RESPONSE)
        self.assertIs(STUDIO_RECORD["response_schema"], STUDIO_RESPONSE_SCHEMA)
        canonical = studio_validity()
        self.assertTrue(all(canonical[f"V{i}"] for i in range(7)))
        self.assertIsNone(canonical["V7"])
        for corruption, expected in CORRUPTION_FIRST_FAILURE.items():
            vector = studio_validity(corruption)
            self.assertFalse(vector[expected])
            expected_index = int(expected[1])
            self.assertTrue(all(vector[f"V{i}"] for i in range(expected_index)))
            self.assertTrue(
                all(vector[f"V{i}"] is None for i in range(expected_index + 1, 8))
            )
        checks = all_checks()
        self.assertEqual(len(checks), 10)
        self.assertEqual(
            [result["check_id"] for result in checks.values()],
            [f"CH26-D{index:02d}" for index in range(1, 11)],
        )
        self.assertTrue(all(result["status"] == "pass" for result in checks.values()))
        self.assertTrue(
            all(
                result["fixture_version"] == "ch26-reference-v1"
                for result in checks.values()
            )
        )
        self.assertEqual(checks["canonicalization"]["fixed_point_status"], "stable")
        self.assertEqual(checks["product_trace"]["handoff"]["semantic_blocker"], "D3")


if __name__ == "__main__":
    unittest.main()
