"""Regression tests for the 30-point RHF scan, without running any RHF solver.

Saved artifacts are read only. A fake solver exercises failed-point handling and
proves that every reported energy comes from the corresponding solver result.
"""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from rhf_bond_scan_30 import (
    POINT_COUNT, build_grid, numerical_payload_digest, rebind_saved_scan,
    scan_molecule, validate_range_record,
)


class GridTests(unittest.TestCase):
    def test_grid_has_exactly_30_distinct_inclusive_ordered_points(self):
        lower, upper = 1.262772864212423, 1.7588508418515771
        values = build_grid(lower, upper)
        self.assertEqual(POINT_COUNT, 30)
        self.assertEqual(len(values), 30)
        self.assertEqual(len(set(values)), 30)
        self.assertEqual(values[0], lower)
        self.assertEqual(values[-1], upper)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        expected_step = (upper - lower) / 29
        for a, b in zip(values, values[1:]):
            self.assertAlmostEqual(b - a, expected_step, places=14)

    def test_invalid_ranges_are_rejected(self):
        cases = (
            (math.nan, 2.0), (1.0, math.nan),
            (math.inf, 2.0), (1.0, math.inf),
            (-1.0, 2.0), (0.0, 2.0),
            (2.0, 1.0), (1.0, 1.0),
        )
        for lower, upper in cases:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    build_grid(lower, upper)


class SavedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ranges_path = ROOT / "results/bond_length_part3/vibrational_levels.json"
        harmonic_path = ROOT / "json/new_rhf_harmonic_bond_ranges.json"
        with ranges_path.open(encoding="utf-8") as handle:
            cls.ranges = json.load(handle)
        with harmonic_path.open(encoding="utf-8") as handle:
            cls.harmonic = json.load(handle)
        cls.harmonic_by_name = {
            record["summary"]["molecule"]: record
            for record in cls.harmonic["results"]
        }

    def example(self):
        record = copy.deepcopy(self.ranges["results"][0])
        harmonic_record = copy.deepcopy(
            self.harmonic_by_name[record["summary"]["molecule"]]
        )
        return record, harmonic_record

    def test_all_nine_current_ranges_are_preserved_without_mutating_sources(self):
        self.assertEqual(len(self.ranges["results"]), 9)
        original_ranges = copy.deepcopy(self.ranges)
        original_harmonic = copy.deepcopy(self.harmonic)
        for record in self.ranges["results"]:
            saved = record["summary"]
            with self.subTest(molecule=saved["molecule"]):
                summary = validate_range_record(
                    record, self.harmonic_by_name[saved["molecule"]]
                )
                self.assertIsNot(summary, saved)
                for key in (
                    "molecule", "basis", "coordinate", "r_e_A", "E_re_Ha",
                    "selected_n", "selected_range_min_A", "selected_range_max_A",
                ):
                    self.assertEqual(summary[key], saved[key])
                grid = build_grid(
                    summary["selected_range_min_A"],
                    summary["selected_range_max_A"],
                )
                self.assertEqual(len(grid), 30)
                self.assertEqual(grid[0], saved["selected_range_min_A"])
                self.assertEqual(grid[-1], saved["selected_range_max_A"])
        self.assertEqual(self.ranges, original_ranges)
        self.assertEqual(self.harmonic, original_harmonic)

    def test_changed_saved_range_or_zero_energy_reference_is_rejected(self):
        changes = (
            ("selected_range_min_A", lambda value: value + 1e-5),
            ("selected_range_max_A", lambda value: value - 1e-5),
            ("n1_min_A", lambda value: value + 1e-5),
            ("n1_pass", lambda value: False),
            ("E_total_n1_Ha", lambda value: value + 1.0),
            ("E_n1_Ha", lambda value: value + 1.0),
            ("hbar_omega_Ha", lambda value: value * 2),
            ("reference_shift_Ha", lambda value: 1.0),
            ("decision_criterion", lambda value: "different reference"),
            ("energy_reference_warning", lambda value: ""),
            ("energy_reference_status", lambda value: "VALIDATED"),
            ("mu_eff_amu", lambda value: value * 1.1),
            ("omega_rad_per_s", lambda value: value * 1.1),
            ("basis", lambda value: "cc-pvdz"),
            ("coordinate", lambda value: "uniform Cartesian scaling"),
            ("r_e_A", lambda value: value + 0.1),
            ("selected_n", lambda value: None),
            ("selected_n", lambda value: True),
        )
        for key, change in changes:
            with self.subTest(key=key, change=change):
                record, harmonic_record = self.example()
                record["summary"][key] = change(record["summary"][key])
                with self.assertRaises(ValueError):
                    validate_range_record(record, harmonic_record)

    def test_lower_level_cannot_replace_current_highest_passing_level(self):
        record, harmonic_record = self.example()
        summary = record["summary"]
        summary["selected_n"] = 0
        summary["selected_range_min_A"] = summary["n0_min_A"]
        summary["selected_range_max_A"] = summary["n0_max_A"]
        with self.assertRaises(ValueError):
            validate_range_record(record, harmonic_record)

    def test_failed_input_validation_is_rejected(self):
        record, harmonic_record = self.example()
        record["input_validation"]["passed"] = False
        with self.assertRaises(ValueError):
            validate_range_record(record, harmonic_record)

    def test_harmonic_range_mismatch_is_rejected(self):
        record, harmonic_record = self.example()
        harmonic_record["summary"]["n1_max_A"] += 1e-5
        with self.assertRaises(ValueError):
            validate_range_record(record, harmonic_record)

    def plan(self):
        record, harmonic_record = self.example()
        summary = validate_range_record(record, harmonic_record)
        points = []
        for index, length in enumerate(build_grid(
            summary["selected_range_min_A"], summary["selected_range_max_A"]
        ), start=1):
            points.append({
                "point_index": index,
                "bond_length_A": length,
                "q_A": length - summary["r_e_A"],
                "cartesian_A": [[0.0, 0.0, 0.0], [0.0, 0.0, length]],
                "geometry_validation": {"passed": True},
            })
        return {
            "summary": summary, "path": None,
            "symbols": harmonic_record["symbols"],
            "basis": summary["basis"], "points": points,
        }

    def fake_engine(self, failed_index=None, failure_kind="unaccepted"):
        calls = []
        states = []

        def solve_point(symbols, coords, basis, previous=None):
            index = len(calls) + 1
            calls.append({"previous": previous, "coords": copy.deepcopy(coords)})
            mf = object()
            state = {
                "accepted": True, "scf_converged": True, "internal_stable": True,
                "energy_Eh": -7.8 + index * 1e-4,
                "orbital_gradient_norm_Eh": 1e-12,
                "lowest_internal_stability_eigenvalue_Eh": 0.01,
                "solver_attempts": [{"initial_guess": "fake_test_solver"}],
            }
            if index == failed_index:
                state["failure"] = "Deliberate fake-solver failure."
                if failure_kind == "unaccepted":
                    state["accepted"] = False
                    state["scf_converged"] = False
                    state["energy_Eh"] = 123.0
                    mf = None
                elif failure_kind == "nonfinite":
                    state["energy_Eh"] = math.nan
                elif failure_kind == "missing_mf":
                    mf = None
                elif failure_kind == "missing_energy":
                    del state["energy_Eh"]
                else:
                    raise AssertionError(f"Unknown test failure: {failure_kind}")
            states.append(copy.deepcopy(state))
            return mf, state

        return SimpleNamespace(solve_point=solve_point), calls, states

    def test_exactly_30_solver_calls_preserve_direct_rhf_energies(self):
        plan = self.plan()
        original = copy.deepcopy(plan)
        engine, calls, states = self.fake_engine()
        callbacks = []
        result = scan_molecule(
            plan, engine,
            on_progress=lambda partial, row: callbacks.append(row["point_index"]),
        )
        self.assertEqual(len(calls), 30)
        self.assertEqual(callbacks, list(range(1, 31)))
        self.assertEqual(result["summary"]["status"], "PASS")
        self.assertEqual(result["summary"]["requested_points"], 30)
        self.assertEqual(result["summary"]["accepted_points"], 30)
        self.assertEqual(result["summary"]["failed_points"], 0)
        self.assertEqual(len(result["points"]), 30)
        self.assertIsNone(calls[0]["previous"])
        self.assertTrue(all(call["previous"] is not None for call in calls[1:]))
        for source, row, state in zip(plan["points"], result["points"], states):
            self.assertEqual(row["point_index"], source["point_index"])
            self.assertEqual(row["bond_length_A"], source["bond_length_A"])
            self.assertEqual(row["status"], "PASS")
            self.assertEqual(row["E_RHF_Ha"], state["energy_Eh"])
            self.assertEqual(row["delta_E_RHF_Ha"], state["energy_Eh"] - plan["summary"]["E_re_Ha"])
        self.assertEqual(plan, original)

    def test_failed_points_have_no_fabricated_energies_and_do_not_skip_rest(self):
        for failure_kind in ("unaccepted", "nonfinite", "missing_mf", "missing_energy"):
            with self.subTest(failure_kind=failure_kind):
                plan = self.plan()
                engine, calls, states = self.fake_engine(8, failure_kind)
                callbacks = []
                result = scan_molecule(
                    plan, engine,
                    on_progress=lambda partial, row: callbacks.append(row["point_index"]),
                )
                self.assertEqual(len(calls), 30)
                self.assertEqual(callbacks, list(range(1, 31)))
                self.assertEqual(len(result["points"]), 30)
                self.assertEqual(result["summary"]["requested_points"], 30)
                self.assertEqual(result["summary"]["accepted_points"], 29)
                self.assertEqual(result["summary"]["failed_points"], 1)
                self.assertEqual(result["summary"]["status"], "FAIL_RHF")
                failed = result["points"][7]
                self.assertEqual(failed["status"], "FAIL_RHF")
                self.assertIsNone(failed["E_RHF_Ha"])
                self.assertIsNone(failed["delta_E_RHF_Ha"])
                self.assertTrue(failed["failure"])
                self.assertIn("diagnostics", failed)
                self.assertIsNone(calls[8]["previous"])
                self.assertIsNotNone(calls[9]["previous"])
                if failure_kind == "unaccepted":
                    self.assertEqual(failed["diagnostics"]["energy_Eh"], 123.0)


class MetadataRebindingTests(unittest.TestCase):
    """Exercise migration using copies; no solver or production artifact is touched."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scan_path = Path(self.directory.name) / "rhf_scan_30.json"
        self.ranges_path = ROOT / "results/bond_length_part3/vibrational_levels.json"
        self.harmonic_path = ROOT / "json/new_rhf_harmonic_bond_ranges.json"
        self.original = json.loads((ROOT / "results/rhf_30_point_scans/rhf_scan_30.json").read_text())
        # Force a metadata migration whether the checked-in artifact is old or current.
        self.original["provenance"]["ranges_sha256"] = "previous-selection-metadata"
        self.write_fixture()

    def write_fixture(self):
        self.scan_path.write_text(json.dumps(self.original))

    def rebind(self):
        return rebind_saved_scan(self.scan_path, self.ranges_path, self.harmonic_path)

    def test_migration_preserves_every_numerical_value_and_original_run_hash(self):
        point_csv = self.scan_path.parent / "rhf_scan_30.csv"
        point_csv.write_bytes(b"unchanged RHF point CSV\n")
        report = self.rebind()
        migrated = json.loads(self.scan_path.read_text())
        self.assertTrue(report["changed"])
        self.assertEqual(numerical_payload_digest(self.original), numerical_payload_digest(migrated))
        self.assertEqual(point_csv.read_bytes(), b"unchanged RHF point CSV\n")
        self.assertEqual(migrated["provenance"]["source_hashes"], self.original["provenance"]["source_hashes"])
        current = json.loads(self.ranges_path.read_text())
        by_name = {record["summary"]["molecule"]: record["summary"] for record in current["results"]}
        for old_result, result in zip(self.original["results"], migrated["results"]):
            self.assertEqual(result["points"], old_result["points"])
            self.assertEqual(result["symbols"], old_result["symbols"])
            expected = by_name[result["summary"]["molecule"]]
            self.assertEqual({key: result["summary"][key] for key in expected}, expected)
            self.assertEqual(set(result["summary"]), set(expected) | {
                "requested_points", "accepted_points", "failed_points", "status"})
        migration = migrated["provenance"]["part3_metadata_migrations"][-1]
        self.assertIs(migration["SCF_run"], False)
        self.assertIs(migration["points_modified"], False)
        self.assertEqual(migration["numerical_payload_sha256_before"],
                         migration["numerical_payload_sha256_after"])
        before_second_call = self.scan_path.read_bytes()
        self.assertFalse(self.rebind()["changed"])
        self.assertEqual(before_second_call, self.scan_path.read_bytes())

    def test_changed_numeric_or_source_input_refuses_migration_without_writing(self):
        cases = (
            ("summary", "selected_range_min_A", -1e-4),
            ("summary", "E_re_Ha", 1e-4),
            ("summary", "selected_n", 0),
            ("summary", "basis", "cc-pvdz"),
            ("point", "bond_length_A", 1e-4),
            ("point", "q_A", 1e-4),
            ("point", "E_re_Ha", 1e-4),
            ("provenance", "harmonic_sha256", "changed input"),
        )
        pristine = copy.deepcopy(self.original)
        for location, key, change in cases:
            with self.subTest(location=location, key=key):
                self.original = copy.deepcopy(pristine)
                target = (self.original["results"][0]["summary"] if location == "summary" else
                          self.original["results"][0]["points"][0] if location == "point" else
                          self.original["provenance"])
                target[key] = target[key] + change if isinstance(change, float) else change
                self.write_fixture()
                before = self.scan_path.read_bytes()
                with self.assertRaises(ValueError):
                    self.rebind()
                self.assertEqual(self.scan_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
