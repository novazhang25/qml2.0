"""Raw-scan utility regressions and legacy Part 3 entry-point delegation.

These tests do not run SCF, optimization, a Hessian or a production workflow.
"""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import rhf_bound_level_selection as selection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))


class LegacyPart3EntryPointTests(unittest.TestCase):
    def test_raw_scan_engine_keeps_version_metadata_api(self):
        import pyscf
        self.assertEqual(selection.pyscf.__version__, pyscf.__version__)

    def test_old_selector_and_scan_decision_entry_points_are_removed(self):
        for name in ("select_level", "scan_one", "convert_saved_results_to_hartree", "build_report"):
            self.assertFalse(hasattr(selection, name), name)

    def test_legacy_cli_delegates_to_current_postprocessor_without_scf(self):
        with patch("bond_length_part3.main", return_value=0) as main, \
             patch.object(selection, "solve_point", side_effect=AssertionError("No SCF allowed")):
            self.assertEqual(selection.main(["--molecules", "LiH"]), 0)
        main.assert_called_once_with(["--molecules", "LiH"])

    def test_easy_filename_delegates_without_old_scan_options(self):
        import hartree
        with patch.object(selection, "main", return_value=0) as main:
            self.assertEqual(hartree.main(["--molecules", "LiH"]), 0)
        main.assert_called_once_with(["--molecules", "LiH"])


class PlateauAcceptanceTests(unittest.TestCase):
    @staticmethod
    def points(energies=None, separations=None):
        if energies is None:
            energies = [-10.0, -9.9999998, -9.9999997, -9.99999965]
        if separations is None:
            separations = [8.0, 16.0, 32.0, 64.0]
        return [
            {"q_A": distance - 1.0, "s_A": distance,
             "energy_Eh": energy, "accepted": True}
            for distance, energy in zip(separations, energies)
        ]

    def test_three_point_plateau_plus_farther_confirmation_passes(self):
        result = selection.assess_plateau(self.points())
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["spread_Eh"], 3.5e-7, places=13)
        self.assertAlmostEqual(result["max_neighbor_change_Eh"], 2e-7, places=13)

    def test_three_points_do_not_supply_farther_confirmation(self):
        self.assertFalse(selection.assess_plateau(self.points()[:3])["passed"])

    def test_empty_scan_is_not_a_plateau(self):
        self.assertFalse(selection.assess_plateau([])["passed"])

    def test_failed_point_within_tail_prevents_acceptance(self):
        for index in range(4):
            with self.subTest(failed_index=index):
                points = self.points()
                points[index]["accepted"] = False
                self.assertFalse(selection.assess_plateau(points)["passed"])

    def test_failed_earlier_attempt_does_not_invalidate_four_valid_tail_points(self):
        old = {"q_A": 1.0, "s_A": 2.0, "energy_Eh": None, "accepted": False}
        self.assertTrue(selection.assess_plateau([old] + self.points())["passed"])

    def test_tightly_spaced_flat_looking_tail_is_rejected(self):
        points = self.points(separations=[1000.0, 1000.1, 1000.2, 1000.3])
        self.assertFalse(selection.assess_plateau(points)["passed"])

    def test_every_tail_interval_must_extend_farther(self):
        points = self.points(separations=[8.0, 16.0, 17.0, 64.0])
        self.assertFalse(selection.assess_plateau(points)["passed"])

    def test_minimum_spacing_ratio_is_inclusive(self):
        points = self.points(separations=[8.0, 12.0, 18.0, 27.0])
        self.assertTrue(selection.assess_plateau(points)["passed"])

    def test_nonincreasing_separations_are_rejected(self):
        for separations in ([8.0, 16.0, 16.0, 64.0], [8.0, 32.0, 16.0, 64.0]):
            with self.subTest(separations=separations):
                self.assertFalse(selection.assess_plateau(self.points(separations=separations))["passed"])

    def test_accumulated_tail_drift_fails_despite_small_neighbor_changes(self):
        points = self.points(energies=[0.0, 0.4e-6, 0.8e-6, 1.2e-6])
        result = selection.assess_plateau(points)
        self.assertFalse(result["passed"])
        self.assertGreater(result["spread_Eh"], 1e-6)
        self.assertLess(result["max_neighbor_change_Eh"], 1e-6)

    def test_farther_confirmation_can_reject_candidate_plateau(self):
        points = self.points(energies=[0.0, 0.1e-6, 0.2e-6, 2.0e-6])
        self.assertFalse(selection.assess_plateau(points)["passed"])

    def test_tolerance_is_configurable(self):
        points = self.points()
        self.assertFalse(selection.assess_plateau(points, tolerance_Eh=1e-7)["passed"])

    def test_nonfinite_tail_energy_is_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(energy=invalid):
                points = self.points()
                points[-1]["energy_Eh"] = invalid
                self.assertFalse(selection.assess_plateau(points)["passed"])


class StoredConstantConversionTests(unittest.TestCase):
    def test_conversion_uses_the_harmonic_run_constants(self):
        constants = {
            "Hartree_J": 4.359744644911914e-18,
            "hbar_J_s": 1.0545718001391127e-34,
            "c_m_per_s": 299792458,
        }
        expected = (
            2.0 * math.pi * constants["hbar_J_s"] * constants["c_m_per_s"] * 100.0
        ) / constants["Hartree_J"]
        actual = selection.conversion_from_constants(constants)
        self.assertAlmostEqual(actual, expected, places=19)
        self.assertAlmostEqual(actual, 1.0 / 219474.63111558527, places=19)
        self.assertGreater(actual, 4.5563e-6)
        self.assertLess(actual, 4.5564e-6)

    def test_conversion_is_not_a_hardcoded_alternative_constant(self):
        constants = {"Hartree_J": 400.0 * math.pi, "hbar_J_s": 1.0, "c_m_per_s": 1.0}
        self.assertAlmostEqual(selection.conversion_from_constants(constants), 0.5)



if __name__ == "__main__":
    unittest.main()
