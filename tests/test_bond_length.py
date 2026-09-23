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
        cls.caches = pipeline.cached_records(cls.args.output_dir, cls.args.harmonic_json)
        cls.equilibrium = pipeline.reuse_equilibrium("LiH", "sto-3g", cls.rows["LiH"], cls.args.metadata, 1e-6)
        cls.harmonic = cls.caches["harmonic"]["LiH"]

    def test_all_nine_reuse_without_numerical_calculations(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(pipeline, "optimize_equilibrium", side_effect=AssertionError("No optimization allowed")), \
             patch.object(pipeline.harmonic, "calculate", side_effect=AssertionError("No nuclear Hessian allowed")), \
             patch.object(pipeline.selection, "solve_point", side_effect=AssertionError("No new SCF allowed")):
            code = pipeline.main(["--reuse-only", "--output-dir", tmp])
            self.assertEqual(code, 0)
            document = json.loads((Path(tmp) / "bond_length.json").read_text())
            self.assertEqual(len(document["results"]), 9)
            for result in document["results"]:
                summary = result["summary"]
                self.assertEqual(summary["validation_status"], "PASS")
                self.assertEqual(result["sources"], {"equilibrium": "reused", "harmonic": "reused", "selection": "postprocessed"})
                self.assertEqual(summary["selected_n"], 1)
                self.assertEqual(summary["selected_range_min_A"], summary["n1_min_A"])
                self.assertEqual(summary["selected_range_max_A"], summary["n1_max_A"])
                self.assertEqual(summary["E_re_Ha"], result["equilibrium"]["E_eq_Eh"])
                self.assertEqual(summary["E_total_n1_Ha"], summary["E_re_Ha"] + 1.5 * summary["hbar_omega_Ha"])
                self.assertTrue(summary["n0_pass"])
                self.assertTrue(summary["n1_pass"])
                self.assertEqual(summary["energy_reference_status"], "UNVALIDATED_ZERO_REFERENCE")
                self.assertEqual(summary["reference_shift_Ha"], 0.0)
                self.assertNotIn("scan", result["selection"])
            contents = (Path(tmp) / "bond_length.json").read_text()
            self.assertNotIn("cm1", contents)
            with (Path(tmp) / "bond_length.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertIn("k_q_Eh_per_Bohr2", rows[0])
            self.assertIn("E_total_n0_Ha", rows[0])
            self.assertNotIn("E_diss", contents)
            self.assertNotIn("D_e_Eh", contents)
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

    def test_part3_reuses_saved_omega_and_current_range_without_a_scan(self):
        hs = copy.deepcopy(self.harmonic)
        hs["summary"]["n1_min_A"] += .01
        with patch.object(pipeline.selection, "solve_point", side_effect=AssertionError("No SCF allowed")):
            result = pipeline.evaluate_part3(self.equilibrium, hs)
        summary = result["summary"]
        self.assertEqual(summary["selected_range_min_A"], hs["summary"]["n1_min_A"])
        self.assertEqual(summary["omega_rad_per_s"], hs["summary"]["omega_rad_per_s"])
        expected = pipeline.constants()["hbar_J_s"] * hs["summary"]["omega_rad_per_s"] / pipeline.constants()["Hartree_J"]
        self.assertEqual(summary["hbar_omega_Ha"], expected)

    def test_part3_refuses_inconsistent_equilibrium_energy(self):
        eq = copy.deepcopy(self.equilibrium)
        eq["E_eq_Eh"] += .1
        with self.assertRaisesRegex(ValueError, "Part 1 equilibrium energy"):
            pipeline.evaluate_part3(eq, self.harmonic)

    def test_part3_refuses_inconsistent_saved_frequency_units(self):
        hs = copy.deepcopy(self.harmonic)
        hs["summary"]["harmonic_frequency_cm1"] *= 2
        with self.assertRaisesRegex(ValueError, "inconsistent energy quanta"):
            pipeline.evaluate_part3(self.equilibrium, hs)

    def test_part3_pass_fail_and_boundary_branches_do_not_shift_the_reference(self):
        for eq_energy, expected_n in ((-2.0, 1), (-1.5, 0), (-1.0, 0), (-0.5, None), (0.0, None)):
            with self.subTest(E_re=eq_energy):
                eq = copy.deepcopy(self.equilibrium)
                hs = copy.deepcopy(self.harmonic)
                eq["E_eq_Eh"] = hs["stationarity"]["E_RHF_Eh"] = eq_energy
                constants = pipeline.constants()
                hs["summary"]["omega_rad_per_s"] = constants["Hartree_J"] / constants["hbar_J_s"]
                hs["summary"]["harmonic_frequency_cm1"] = 1.0 / pipeline.selection.conversion_from_constants(constants)
                summary = pipeline.evaluate_part3(eq, hs)["summary"]
                self.assertEqual(summary["selected_n"], expected_n)
                self.assertEqual(summary["E_re_Ha"], eq_energy)
                self.assertEqual(summary["E_total_n0_Ha"], eq_energy + .5)
                self.assertEqual(summary["E_total_n1_Ha"], eq_energy + 1.5)
                self.assertEqual(summary["reference_shift_Ha"], 0.0)
                if expected_n is None:
                    self.assertIsNone(summary["selected_range_min_A"])
                    self.assertIsNone(summary["selected_range_max_A"])

    def test_failed_harmonic_stage_never_selects_a_level(self):
        args = pipeline.parse_args(["--molecules", "LiH", "--recompute"])
        failed = {"summary": {"molecule": "LiH", "validation_status": "FAIL_VALIDATION"}, "failures": ["Nonpositive curvature."]}
        with patch.object(pipeline.harmonic, "calculate", return_value=failed), \
             patch.object(pipeline, "evaluate_part3", side_effect=AssertionError("Do not select after harmonic failure")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.process_molecule("LiH", args, self.rows, self.caches)
        self.assertEqual(result["status"], "FAIL_HARMONIC")
        self.assertEqual(result["summary"]["selection_status"], "NOT_RUN")
        self.assertIsNone(result["summary"]["selected_n"])

    def test_missing_equilibrium_in_reuse_only_mode_does_not_optimize(self):
        caches = {"equilibrium": {}, "harmonic": {}, "notes": []}
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

    def test_conflicting_or_removed_options_are_rejected(self):
        for arguments in (["--reuse-only", "--reoptimize"], ["--reuse-only", "--recompute"],
                          ["--threshold-warning-eh", "1e-8"], ["--gtol", "1e-3"]):
            with self.subTest(args=arguments), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                pipeline.parse_args(arguments)



if __name__ == "__main__":
    unittest.main()
