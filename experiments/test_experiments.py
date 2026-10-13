"""Golden archive replays and deliberate faults for portable reader experiments."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import _data, gated_mla, kda, moba, recovery, validation
from experiments.__main__ import main
from experiments.recovery_model import run_cell


class ReaderExperimentTests(unittest.TestCase):
    def test_kda_matches_archived_coordinates(self) -> None:
        result = _data.check_reference("kda", kda.run_all_checks())
        self.assertEqual(result["checks"], 10)
        self.assertEqual(
            result["results"]["RESULT_KDA_ONE_STEP_ORACLE"]["output"], ["27/20", "4/5"]
        )

    def test_moba_matches_archived_forward_and_derivatives(self) -> None:
        result = _data.check_reference("moba", moba.run_all_checks())
        self.assertEqual(result["checks"], 10)
        self.assertLessEqual(
            result["results"]["RESULT_MOBA_BACKWARD_ORACLE"]["max_abs_error"], 1e-8
        )

    def test_moba_accepts_sequential_float_reductions(self) -> None:
        def sequential_sum(values, start=0):
            total = start
            for value in values:
                total += value
            return total

        with patch("builtins.sum", sequential_sum):
            observed = moba.run_all_checks()
        self.assertNotEqual(
            _data.json_value(observed), _data.load_data("moba")["results"]
        )
        self.assertEqual(_data.check_reference("moba", observed)["checks"], 10)

    def test_moba_changed_forward_coordinates_are_rejected(self) -> None:
        observed = moba.run_all_checks()
        outputs = _data.json_value(
            observed["RESULT_MOBA_FORWARD_ORACLE"]["final_query_outputs"]
        )
        outputs[0][0] += 1e-6
        observed["RESULT_MOBA_FORWARD_ORACLE"]["final_query_outputs"] = outputs
        with self.assertRaisesRegex(ValueError, "RESULT_MOBA_FORWARD_ORACLE"):
            _data.check_reference("moba", observed)

    def test_moba_error_bounds_reject_invalid_diagnostics(self) -> None:
        baseline = moba.run_all_checks()
        invalid = [
            ("RESULT_MOBA_FORWARD_ORACLE", "max_abs_error", 2e-12),
            ("RESULT_MOBA_FORWARD_ORACLE", "max_abs_error", -1e-15),
            ("RESULT_MOBA_FORWARD_ORACLE", "max_abs_error", float("nan")),
            ("RESULT_MOBA_FORWARD_ORACLE", "max_abs_error", float("inf")),
            ("RESULT_MOBA_BACKWARD_ORACLE", "max_abs_error", 2e-8),
            ("RESULT_MOBA_BACKWARD_ORACLE", "q_k_v_max_abs_errors", [2e-8, 0.0, 0.0]),
            ("RESULT_MOBA_BACKWARD_ORACLE", "q_k_v_max_abs_errors", [0.0, 0.0]),
        ]
        for key, field, value in invalid:
            with self.subTest(key=key, field=field, value=value):
                observed = deepcopy(baseline)
                observed[key][field] = value
                with self.assertRaisesRegex(ValueError, key):
                    _data.check_reference("moba", observed)

    def test_moba_other_fields_still_require_exact_matches(self) -> None:
        baseline = moba.run_all_checks()
        changes = [
            ("RESULT_MOBA_FORWARD_ORACLE", "output_shape", [11, 2, 4]),
            ("RESULT_MOBA_FORWARD_ORACLE", "tolerance", 1e-8),
            ("RESULT_MOBA_FORWARD_ORACLE", "status", "fail"),
            ("RESULT_MOBA_CAUSALITY", "max_abs_prefix_error", 1e-15),
            ("RESULT_MOBA_ROUTE_CONSERVATION", "load_sum", 29),
        ]
        for key, field, value in changes:
            with self.subTest(key=key, field=field):
                observed = deepcopy(baseline)
                observed[key][field] = value
                with self.assertRaisesRegex(ValueError, key):
                    _data.check_reference("moba", observed)

    def test_changed_oracle_output_is_rejected(self) -> None:
        results = kda.run_all_checks()
        results["RESULT_KDA_ONE_STEP_ORACLE"]["output"] = ["0", "0"]
        with self.assertRaisesRegex(ValueError, "result differs"):
            _data.check_reference("kda", results)

    def test_gated_mla_reconstructs_all_recorded_numeric_cases(self) -> None:
        result = gated_mla.analyze_numerics(_data.load_data("gated_mla_numerics"))
        self.assertEqual(result["cells"], 36)

    def test_changed_compact_output_fails_numeric_gate(self) -> None:
        packet = _data.load_data("gated_mla_numerics")
        packet["records"][0]["candidate"]["output"][0][0] += 1000
        with self.assertRaisesRegex(ValueError, "numerical gate failed"):
            gated_mla.analyze_numerics(packet)

    def test_incomplete_numeric_packet_is_rejected(self) -> None:
        packet = _data.load_data("gated_mla_numerics")
        packet["records"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            gated_mla.analyze_numerics(packet)

    def test_cache_analysis_reproduces_printed_summary(self) -> None:
        result = gated_mla.analyze_cache(_data.load_data("gated_mla_cache"))
        self.assertEqual(result["cells"], 48)
        self.assertEqual(result["samples"], 480)
        self.assertEqual(
            round(result["median_materialized_over_compact_allocated_peak"], 2), 7.44
        )
        self.assertEqual(
            [round(value, 2) for value in result["ratio_range"]], [2.50, 22.33]
        )

    def test_changed_allocation_increment_is_rejected(self) -> None:
        packet = _data.load_data("gated_mla_cache")
        packet["records"][0]["arms"]["compact"]["samples"][0]["incremental_peak_bytes"][
            "allocated"
        ] += 1
        with self.assertRaisesRegex(ValueError, "incremental peak differs"):
            gated_mla.analyze_cache(packet)

    def test_failed_recorded_parity_blocks_cache_interpretation(self) -> None:
        packet = _data.load_data("gated_mla_cache")
        packet["records"][0]["parity"]["continuation"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "continuation gate failed"):
            gated_mla.analyze_cache(packet)

    def test_validation_replays_archive_and_keeps_hidden_defect(self) -> None:
        result = validation.run()
        self.assertEqual(result["checks"], 270)
        self.assertEqual(result["local_replay_cells"], 27)
        self.assertTrue(
            all(
                row["false_confidence"] == 1 for row in result["local_summary"].values()
            )
        )

    def test_validation_accepts_new_times_but_rejects_changed_outcomes(self) -> None:
        archived = _data.load_data("validation")["recorded_cells"]
        observed = deepcopy([row for row in archived if row["repetition"] == 0])
        for cell in observed:
            cell["metrics"]["wall_ns"] = 1
        validation.verify_cells(observed, archived)
        archived[1]["metrics"]["false_confidence"] += 1
        with self.assertRaisesRegex(ValueError, "replay differs"):
            validation.verify_cells(observed, archived)

    def test_duplicate_validation_cell_does_not_hide_missing_cell(self) -> None:
        archived = _data.load_data("validation")["recorded_cells"]
        observed = [row for row in archived if row["repetition"] == 0]
        archived[-1] = deepcopy(archived[0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validation.verify_cells(observed, archived)

    def test_recovery_matches_every_recorded_event_and_metric(self) -> None:
        result = recovery.run()
        self.assertEqual(result["checks"], 1200)
        reconciled = result["summary"]["validated_checkpoint_reconciliation"]
        self.assertEqual(reconciled["safe_complete"], 320)
        self.assertEqual(reconciled["unresolved_task"], 80)
        self.assertEqual(reconciled["duplicate_effects"], 0)

    def test_ambiguous_effect_is_held_unresolved(self) -> None:
        task = _data.load_data("recovery")["fixture"]["tasks"][0]
        cell = run_cell(
            task,
            1,
            "effect_before_receipt_unobservable",
            "validated_checkpoint_reconciliation",
            0,
        )
        self.assertEqual(cell["final_state"]["task_status"], "unresolved")
        self.assertEqual(cell["metrics"]["unresolved_task"], 1)

    def test_checkpoint_resume_without_reconciliation_duplicates_effect(self) -> None:
        task = _data.load_data("recovery")["fixture"]["tasks"][0]
        cell = run_cell(
            task, 1, "effect_before_receipt", "checkpoint_no_reconciliation", 0
        )
        self.assertGreater(cell["metrics"]["duplicate_effects"], 0)
        self.assertEqual(cell["final_state"]["task_status"], "unsafe_complete")

    def test_release_checksum_rejects_changed_fixture(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            payload = b'{"schema_version": 1}'
            (root / "manifest.json").write_text(
                json.dumps({"data_sha256": {"probe.json": sha256(payload).hexdigest()}})
            )
            (data / "probe.json").write_bytes(payload + b" ")
            with patch.object(_data, "DATA_ROOT", data):
                with self.assertRaisesRegex(ValueError, "checksum differs"):
                    _data.load_data("probe")

    def test_cli_prints_concise_verification(self) -> None:
        output = StringIO()
        with (
            patch("sys.argv", ["experiments", "kda", "--check"]),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 0)
        self.assertEqual(output.getvalue(), "[OK] kda: 10 checks\n")

    def test_cli_returns_failure_for_changed_evidence(self) -> None:
        output = StringIO()
        with (
            patch("sys.argv", ["experiments", "kda", "--check"]),
            patch("experiments.__main__.run", side_effect=ValueError("changed packet")),
            redirect_stderr(output),
        ):
            with self.assertRaises(SystemExit) as stopped:
                main()
        self.assertEqual(stopped.exception.code, 1)
        self.assertIn("[FAIL] kda: changed packet", output.getvalue())
