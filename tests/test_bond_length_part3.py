"""Zero-criterion arithmetic, unshifted reference, units and saved-input tests."""
import contextlib
import copy
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
import bond_length_part3 as part3


class ZeroCriterionTests(unittest.TestCase):
    def test_both_ground_only_neither_and_equalities(self):
        for re, expected, chosen in ((-2., (True, True), 1), (-1., (True, False), 0),
                                     (0., (False, False), None), (-.5, (False, False), None),
                                     (-1.5, (True, False), 0)):
            with self.subTest(re=re):
                result = part3.compute_levels(re, 1.)
                self.assertEqual((result["n0_pass"], result["n1_pass"]), expected)
                self.assertEqual(result["selected_n"], chosen)
                self.assertEqual(result["E_re_Ha"], re)
                self.assertEqual(result["E_n0_Ha"], .5)
                self.assertEqual(result["E_n1_Ha"], 1.5)
                self.assertEqual(result["E_total_n0_Ha"], re + .5)
                self.assertEqual(result["E_total_n1_Ha"], re + 1.5)

    def test_no_tolerance_offset_near_zero(self):
        below = part3.compute_levels(math.nextafter(-1.5, -math.inf), 1.)
        above = part3.compute_levels(math.nextafter(-1.5, math.inf), 1.)
        self.assertTrue(below["n1_pass"])
        self.assertFalse(above["n1_pass"])

    def test_reference_is_never_shifted_and_warning_is_explicit(self):
        result = part3.compute_levels(-7.86, .0085)
        self.assertEqual(result["E_re_Ha"], -7.86)
        self.assertEqual(result["reference_shift_Ha"], 0.)
        self.assertEqual(result["energy_reference_status"], "UNVALIDATED_ZERO_REFERENCE")
        self.assertIn("nuclear repulsion", result["energy_reference_warning"])
        self.assertFalse(result["physical_vibrational_binding_established"])
        # Changing the raw reference changes a zero test: no hidden recentering.
        self.assertNotEqual(part3.compute_levels(-2., 1.)["selected_n"],
                            part3.compute_levels(2., 1.)["selected_n"])

    def test_invalid_input_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                part3.compute_levels(value, .01)
        for value in (math.nan, math.inf, -math.inf, 0., -1.):
            with self.subTest(quantum=value), self.assertRaises(ValueError):
                part3.compute_levels(-1., value)

    def test_no_threshold_argument_or_old_fields(self):
        with self.assertRaises(TypeError):
            part3.compute_levels(-1., .01, 123.)
        result = part3.compute_levels(-1., .01)
        for old in ("E_diss_Ha", "D_e_Ha", "n0_bound", "n1_bound", "asymptote_status"):
            self.assertNotIn(old, result)


class SavedInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harmonic_path = ROOT / "json/new_rhf_harmonic_bond_ranges.json"
        cls.metadata_path = ROOT / "results/rhf_geometries/rhf_equilibrium_summary.csv"
        cls.hdoc = json.loads(cls.harmonic_path.read_text())
        cls.harmonic = part3.indexed(cls.hdoc["results"], summary=True)
        with cls.metadata_path.open(newline="") as handle:
            cls.metadata = part3.indexed(csv.DictReader(handle))

    def test_all_nine_reuse_exact_energy_curvature_mass_and_omega(self):
        for name in part3.MOLECULES:
            with self.subTest(name=name):
                h = copy.deepcopy(self.harmonic[name])
                m = copy.deepcopy(self.metadata[name])
                result = part3.analyze_molecule(name, m, h, self.hdoc["runtime"]["constants"])
                s = result["summary"]
                self.assertEqual(s["E_re_Ha"], float(m["energy_hartree"]))
                self.assertEqual(s["k_Ha_per_Bohr2"], h["summary"]["k_q_Eh_per_Bohr2"])
                self.assertEqual(s["mu_eff_amu"], h["summary"]["mu_eff_amu"])
                self.assertEqual(s["omega_rad_per_s"], h["summary"]["omega_rad_per_s"])
                c = self.hdoc["runtime"]["constants"]
                self.assertEqual(s["hbar_omega_Ha"], c["hbar_J_s"] * s["omega_rad_per_s"] / c["Hartree_J"])
                self.assertTrue(s["n0_pass"] and s["n1_pass"])
                self.assertEqual(s["selected_n"], 1)
                self.assertEqual(s["selected_range_min_A"], h["summary"]["n1_min_A"])
                self.assertEqual(s["selected_range_max_A"], h["summary"]["n1_max_A"])
                self.assertEqual(h, self.harmonic[name])
                self.assertEqual(m, self.metadata[name])

    def test_harmonic_injected_thresholds_cannot_change_result(self):
        h = copy.deepcopy(self.harmonic["LiH"])
        baseline = part3.analyze_molecule("LiH", self.metadata["LiH"], h, self.hdoc["runtime"]["constants"])
        h["summary"].update(E_diss_Ha=-1e10, D_e_Ha=-1e20)
        actual = part3.analyze_molecule("LiH", self.metadata["LiH"], h, self.hdoc["runtime"]["constants"])
        self.assertEqual(actual, baseline)

    def test_wrong_frequency_units_rejected(self):
        for field in ("omega_rad_per_s", "harmonic_frequency_cm1"):
            h = copy.deepcopy(self.harmonic["LiH"])
            h["summary"][field] *= 100
            with self.subTest(field=field), self.assertRaises(ValueError):
                part3.analyze_molecule("LiH", self.metadata["LiH"], h, self.hdoc["runtime"]["constants"])

    def test_export_needs_only_parts_one_and_two(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(part3.main(["--no-plots", "--output-dir", tmp]), 0)
            document = json.loads((Path(tmp) / "vibrational_levels.json").read_text())
            self.assertEqual(document["schema_version"], part3.SCHEMA_VERSION)
            self.assertEqual(len(document["results"]), 9)
            self.assertEqual(len(document["provenance"]["source_files_sha256"]), 2)
            self.assertFalse(document["provenance"]["SCF_recomputed"])
            encoded = json.dumps(document)
            self.assertNotIn('"E_diss', encoded)
            self.assertNotIn('"D_e', encoded)
            self.assertNotIn('"n0_bound', encoded)
            with (Path(tmp) / "vibrational_levels.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row["n0_pass"] == row["n1_pass"] == "YES" for row in rows))
            self.assertIn("UNVALIDATED_ZERO_REFERENCE", (Path(tmp) / "PART3_REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
