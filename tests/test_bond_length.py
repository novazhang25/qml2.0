"""Pipeline integration, cache invalidation, stage failure and physical checks."""
import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
import bond_length as pipeline


class BondLengthPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = pipeline.parse_args(["--reuse-only"])
        cls.rows = pipeline.read_metadata(cls.args.metadata)
        cls.caches = pipeline.cached_records(cls.args.output_dir, cls.args.harmonic_json, cls.args.selection_json)
        cls.equilibrium = pipeline.reuse_equilibrium("LiH", "sto-3g", cls.rows["LiH"], cls.args.metadata, 1e-6)
        cls.harmonic = cls.caches["harmonic"]["LiH"]
        cls.bound = cls.caches["selection"]["LiH"]
        cls.settings = {"max_q_A": 1e7, "plateau_tolerance_Eh": 1e-6}

    def test_all_nine_reuse_without_numerical_calculations(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(pipeline, "optimize_equilibrium", side_effect=AssertionError("No optimization allowed")), \
             patch.object(pipeline.harmonic, "calculate", side_effect=AssertionError("No nuclear Hessian allowed")), \
             patch.object(pipeline.selection, "scan_one", side_effect=AssertionError("No new scan allowed")):
            code = pipeline.main(["--reuse-only", "--output-dir", tmp])
            self.assertEqual(code, 0)
            document = json.loads((Path(tmp) / "bond_length.json").read_text())
            self.assertEqual(len(document["results"]), 9)
            for result in document["results"]:
                summary = result["summary"]
                self.assertEqual(summary["validation_status"], "PASS")
                self.assertEqual(set(result["sources"].values()), {"reused"})
                self.assertEqual(summary["selected_n"], 1)
                self.assertEqual(summary["selected_range_min_A"], summary["n1_min_A"])
                self.assertEqual(summary["selected_range_max_A"], summary["n1_max_A"])
                self.assertAlmostEqual(summary["E1_rel_Eh"], -summary["D_e_Eh"] + 1.5*summary["harmonic_quantum_Eh"])
                original = self.caches["selection"][summary["molecule"]]
                self.assertEqual(original["scan"], result["selection"]["scan"])
            contents = (Path(tmp) / "bond_length.json").read_text()
            self.assertNotIn("cm1", contents)
            with (Path(tmp) / "bond_length.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertIn("k_q_Eh_per_Bohr2", rows[0])
            self.assertIn("E0_rel_Eh", rows[0])
            # A second run can recover the normalized Hartree cache without data loss.
            self.assertEqual(pipeline.main(["--reuse-only", "--output-dir", tmp]), 0)

    def test_basis_change_invalidates_harmonic_cache(self):
        eq = copy.deepcopy(self.equilibrium)
        eq["row"]["basis"] = "6-31g"
        self.assertFalse(pipeline.harmonic_cache_matches(self.harmonic, eq))

    def test_geometry_change_invalidates_harmonic_cache(self):
        eq = copy.deepcopy(self.equilibrium)
        eq["cartesian_A"][1, 2] += .01
        self.assertFalse(pipeline.harmonic_cache_matches(self.harmonic, eq))

    def test_equilibrium_energy_change_invalidates_harmonic_cache(self):
        eq = copy.deepcopy(self.equilibrium)
        eq["E_eq_Eh"] += 1e-5
        self.assertFalse(pipeline.harmonic_cache_matches(self.harmonic, eq))

    def test_harmonic_range_change_invalidates_dissociation_cache(self):
        h = copy.deepcopy(self.harmonic)
        h["summary"]["n1_min_A"] += .01
        self.assertFalse(pipeline.selection_cache_matches(self.bound, h, self.settings))

    def test_unconfirmed_or_failed_plateau_is_not_reused(self):
        for mutation in ("point", "independent"):
            record = copy.deepcopy(self.bound)
            if mutation == "point":
                record["scan"][-1]["accepted"] = False
            else:
                record["asymptote_independent_checks"][0]["accepted"] = False
            self.assertFalse(pipeline.selection_cache_matches(record, self.harmonic, self.settings))

    def test_shorter_requested_scan_cap_invalidates_old_plateau(self):
        self.assertFalse(pipeline.selection_cache_matches(self.bound, self.harmonic, {**self.settings, "max_q_A": 5.}))

    def test_failed_harmonic_stage_never_selects_a_level(self):
        args = pipeline.parse_args(["--molecules", "LiH", "--recompute"])
        failed = {"summary": {"molecule": "LiH", "validation_status": "FAIL_VALIDATION"}, "failures": ["Nonpositive curvature."]}
        with patch.object(pipeline.harmonic, "calculate", return_value=failed), \
             patch.object(pipeline.selection, "scan_one", side_effect=AssertionError("Do not scan after harmonic failure")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.process_molecule("LiH", args, self.rows, self.caches)
        self.assertEqual(result["status"], "FAIL_HARMONIC")
        self.assertEqual(result["summary"]["selection_status"], "NOT_RUN")
        self.assertIsNone(result["summary"]["selected_n"])

    def test_missing_equilibrium_in_reuse_only_mode_does_not_optimize(self):
        caches = {"equilibrium": {}, "harmonic": {}, "selection": {}, "notes": []}
        with patch.object(pipeline, "optimize_equilibrium", side_effect=AssertionError("No optimization")):
            result = pipeline.process_molecule("LiH", self.args, {}, caches)
        self.assertEqual(result["status"], "NEEDS_INPUT")
        self.assertIsNone(result["summary"]["selected_n"])

    def test_failed_optimizer_stops_later_stages(self):
        args = pipeline.parse_args(["--molecules", "LiH", "--reoptimize"])
        with patch.object(pipeline, "optimize_equilibrium", return_value={"status": "FAIL_RHF_EQUILIBRIUM", "message": "No convergence"}), \
             patch.object(pipeline.harmonic, "calculate", side_effect=AssertionError("Do not calculate Hessian")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.process_molecule("LiH", args, self.rows, self.caches)
        self.assertEqual(result["status"], "FAIL_RHF_EQUILIBRIUM")

    def test_seed_shapes_match_all_nine_molecules(self):
        from collections import Counter
        for name in pipeline.MOLECULES:
            for symbols, coordinates in pipeline.starting_geometries(name):
                self.assertEqual(Counter(symbols), Counter(pipeline.harmonic.FORMULAS[name]))
                self.assertEqual(coordinates.shape, (len(symbols), 3))
                self.assertTrue(np.isfinite(coordinates).all())

    def test_unit_view_preserves_energy_quantum_and_roundtrips_for_engine(self):
        h = pipeline.energy_view(copy.deepcopy(self.harmonic))
        self.assertNotIn("harmonic_frequency_cm1", h["summary"])
        factor = pipeline.selection.conversion_from_constants(pipeline.constants())
        expected = self.harmonic["summary"]["harmonic_frequency_cm1"] * factor
        self.assertEqual(h["summary"]["harmonic_quantum_Eh"], expected)
        native = pipeline.energy_view(h, inverse=True)
        self.assertAlmostEqual(native["summary"]["harmonic_frequency_cm1"], self.harmonic["summary"]["harmonic_frequency_cm1"])

    def test_conflicting_or_uncovered_threshold_options_are_rejected(self):
        for arguments in (["--reuse-only", "--reoptimize"], ["--reuse-only", "--recompute"],
                          ["--threshold-warning-eh", "1e-8"], ["--gtol", "1e-3"]):
            with self.subTest(args=arguments), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                pipeline.parse_args(arguments)

    def test_fresh_lih_optimization_hessian_and_selection_in_temporary_directory(self):
        # A bounded integration calculation exercises the newly connected stages;
        # original geometry and production outputs are never overwritten.
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            status = pipeline.main(["--molecules", "LiH", "--reoptimize", "--gtol", "2e-7", "--output-dir", tmp])
            self.assertEqual(status, 0)
            result = json.loads((Path(tmp) / "bond_length.json").read_text())["results"][0]
            summary = result["summary"]
            self.assertEqual(set(result["sources"].values()), {"computed"})
            self.assertLess(summary["max_gradient_Eh_per_Bohr"], 2e-7)
            self.assertAlmostEqual(summary["r_e_A"], 1.510812, places=5)
            self.assertGreater(summary["k_q_Eh_per_Bohr2"], 0)
            self.assertTrue(result["harmonic"]["finite_differences"]["convergence_pass"])
            self.assertEqual(summary["selected_n"], 1)
            self.assertTrue(result["selection"]["accepted_plateau"]["passed"])


if __name__ == "__main__":
    unittest.main()
