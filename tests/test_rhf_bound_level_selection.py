"""Focused regression tests for RHF threshold selection and plateau acceptance."""

import math
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import rhf_bound_level_selection as selection


class BoundLevelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.harmonic = {
            "n0_min_A": 0.91,
            "n0_max_A": 1.09,
            "n1_min_A": 0.84,
            "n1_max_A": 1.16,
        }

    def select(self, depth, quantum=0.01, **kwargs):
        return selection.select_level(depth, quantum, self.harmonic, **kwargs)

    def test_first_excited_level_bound_selects_exact_existing_n1_range(self):
        result = self.select(0.02)
        self.assertAlmostEqual(result["E0_rel_Eh"], -0.015)
        self.assertAlmostEqual(result["E1_rel_Eh"], -0.005)
        self.assertEqual(result["harmonic_quantum_Eh"], 0.01)
        self.assertEqual(result["half_hbar_omega_Eh"], 0.005)
        self.assertEqual(result["three_halves_hbar_omega_Eh"], 0.015)
        self.assertTrue(result["n0_bound"])
        self.assertTrue(result["n1_bound"])
        self.assertEqual(result["selected_n"], 1)
        self.assertEqual(result["selected_range_min_A"], self.harmonic["n1_min_A"])
        self.assertEqual(result["selected_range_max_A"], self.harmonic["n1_max_A"])
        self.assertEqual(result["validation_status"], "PASS")
        self.assertEqual(
            result["selected_reason"],
            "n=1 lies below the RHF-path dissociation threshold",
        )

    def test_only_ground_level_bound_selects_exact_existing_n0_range(self):
        result = self.select(0.01)
        self.assertAlmostEqual(result["E0_rel_Eh"], -0.005)
        self.assertAlmostEqual(result["E1_rel_Eh"], 0.005)
        self.assertTrue(result["n0_bound"])
        self.assertFalse(result["n1_bound"])
        self.assertEqual(result["selected_n"], 0)
        self.assertEqual(result["selected_range_min_A"], self.harmonic["n0_min_A"])
        self.assertEqual(result["selected_range_max_A"], self.harmonic["n0_max_A"])
        self.assertEqual(result["validation_status"], "PASS")
        self.assertEqual(
            result["selected_reason"],
            "n=0 is bound but n=1 is above the RHF-path dissociation threshold",
        )

    def test_neither_level_bound_has_no_selected_range(self):
        result = self.select(0.0025)
        self.assertAlmostEqual(result["E0_rel_Eh"], 0.0025)
        self.assertAlmostEqual(result["E1_rel_Eh"], 0.0125)
        self.assertFalse(result["n0_bound"])
        self.assertFalse(result["n1_bound"])
        self.assertIsNone(result["selected_n"])
        self.assertIsNone(result["selected_range_min_A"])
        self.assertIsNone(result["selected_range_max_A"])
        self.assertEqual(result["validation_status"], "NO_BOUND_LEVEL_IN_HARMONIC_TEST")
        self.assertEqual(
            result["selected_reason"],
            "even n=0 lies at or above the RHF-path dissociation threshold",
        )

    def test_exact_n1_threshold_is_unbound_and_warned(self):
        result = self.select(0.015)
        self.assertEqual(result["E1_rel_Eh"], 0.0)
        self.assertFalse(result["n1_bound"])
        self.assertEqual(result["selected_n"], 0)
        self.assertEqual(result["validation_status"], "NEAR_THRESHOLD")

    def test_exact_n0_threshold_is_unbound_and_has_no_range(self):
        result = self.select(0.005)
        self.assertEqual(result["E0_rel_Eh"], 0.0)
        self.assertFalse(result["n0_bound"])
        self.assertIsNone(result["selected_n"])
        self.assertIsNone(result["selected_range_min_A"])
        self.assertIsNone(result["selected_range_max_A"])

    def test_small_signed_n1_energies_preserve_strict_decision(self):
        for depth, selected_n, n1_bound in (
            (0.01500000001, 1, True),
            (0.01499999999, 0, False),
        ):
            with self.subTest(depth=depth):
                result = self.select(depth)
                self.assertEqual(result["selected_n"], selected_n)
                self.assertEqual(result["n1_bound"], n1_bound)
                self.assertEqual(result["validation_status"], "NEAR_THRESHOLD")

    def test_small_signed_n0_energies_preserve_strict_decision(self):
        for depth, selected_n, n0_bound in (
            (0.00500000001, 0, True),
            (0.00499999999, None, False),
        ):
            with self.subTest(depth=depth):
                result = self.select(depth)
                self.assertEqual(result["selected_n"], selected_n)
                self.assertEqual(result["n0_bound"], n0_bound)
                if selected_n is None:
                    self.assertIsNone(result["selected_range_min_A"])
                    self.assertIsNone(result["selected_range_max_A"])

    def test_warning_window_is_explicit_and_configurable(self):
        result = self.select(0.01505, warning_Eh=2e-5)
        self.assertEqual(result["selected_n"], 1)
        self.assertEqual(result["validation_status"], "PASS")
        result = self.select(0.01505, warning_Eh=1e-4)
        self.assertEqual(result["selected_n"], 1)
        self.assertEqual(result["validation_status"], "NEAR_THRESHOLD")

    def test_warning_window_boundary_uses_strict_less_than(self):
        # Binary-exact numbers isolate the boundary rule from decimal rounding.
        quantum = 1.0 / 64.0
        warning = 1.0 / 8192.0
        for sign in (-1, 1):
            with self.subTest(side=sign):
                depth = 1.5 * quantum + sign * warning
                result = self.select(depth, quantum=quantum, warning_Eh=warning)
                self.assertEqual(result["E1_rel_Eh"], -sign * warning)
                self.assertEqual(result["validation_status"], "PASS")

    def test_invalid_depth_and_quantum_are_rejected(self):
        for invalid in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(quantity="depth", value=invalid):
                with self.assertRaises(ValueError):
                    self.select(invalid)
            with self.subTest(quantity="quantum", value=invalid):
                with self.assertRaises(ValueError):
                    self.select(0.02, quantum=invalid)

    def test_unit_conversion_preserves_previous_selection_and_warning_decisions(self):
        wavenumber_to_Eh = 1.0 / 219474.63111558527
        cases = (
            (2000.0, 1, "PASS"),
            (1000.0, 0, "PASS"),
            (250.0, None, "NO_BOUND_LEVEL_IN_HARMONIC_TEST"),
            (1505.0, 1, "NEAR_THRESHOLD"),
            (1495.0, 0, "NEAR_THRESHOLD"),
        )
        for depth_cm1, expected_n, expected_status in cases:
            with self.subTest(previous_depth_cm1=depth_cm1):
                result = self.select(
                    depth_cm1 * wavenumber_to_Eh,
                    quantum=1000.0 * wavenumber_to_Eh,
                    warning_Eh=10.0 * wavenumber_to_Eh,
                )
                self.assertEqual(result["selected_n"], expected_n)
                self.assertEqual(result["validation_status"], expected_status)
                self.assertAlmostEqual(
                    result["E0_rel_Eh"], (500.0 - depth_cm1) * wavenumber_to_Eh,
                    places=16,
                )
                self.assertAlmostEqual(
                    result["E1_rel_Eh"], (1500.0 - depth_cm1) * wavenumber_to_Eh,
                    places=16,
                )

    def test_previous_nine_molecule_results_keep_their_n1_selection(self):
        # Frozen pre-migration results verify that changing units preserves decisions.
        previous_results = (
            ("LiH", 75494.62367169918, 1867.8659194944714),
            ("BeH2", 145645.89447779878, 2490.596112984732),
            ("H2O", 166410.81358815142, 4139.498437570796),
            ("NH3", 279206.9137203255, 3831.515146322629),
            ("N2", 169650.43785795826, 2669.447317337452),
            ("CO", 104041.06589001027, 2462.088056074363),
            ("HF", 123252.85195354994, 4474.567708175719),
            ("H2S", 160896.29937842328, 3273.669987657965),
            ("H2O2", 104904.98450881628, 1488.0618032645937),
        )
        wavenumber_to_Eh = 1.0 / 219474.63111558527
        for molecule, depth_cm1, frequency_cm1 in previous_results:
            with self.subTest(molecule=molecule):
                result = self.select(
                    depth_cm1 * wavenumber_to_Eh,
                    quantum=frequency_cm1 * wavenumber_to_Eh,
                    warning_Eh=10.0 * wavenumber_to_Eh,
                )
                self.assertTrue(result["n0_bound"])
                self.assertTrue(result["n1_bound"])
                self.assertEqual(result["selected_n"], 1)
                self.assertEqual(result["validation_status"], "PASS")
                self.assertEqual(result["selected_range_min_A"], self.harmonic["n1_min_A"])
                self.assertEqual(result["selected_range_max_A"], self.harmonic["n1_max_A"])


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


class WorkflowIntegrationTests(unittest.TestCase):
    """Small real RHF checks on the validated current harmonic geometries."""

    @classmethod
    def setUpClass(cls):
        selection.lib.num_threads(1)
        source = Path(__file__).resolve().parents[1] / "json/new_rhf_harmonic_bond_ranges.json"
        document, cls.records = selection.load_harmonic_input(source)
        cls.wavenumber_to_Eh = selection.conversion_from_constants(document["runtime"]["constants"])
        cls.settings = {
            "max_q_A": 5.0,
            "plateau_tolerance_Eh": 1e-6,
            "threshold_warning_Eh": 10.0 * cls.wavenumber_to_Eh,
            "molecule": "CO",
        }

    def assert_no_selection(self, result):
        summary = result["summary"]
        self.assertIsNone(summary["selected_n"])
        self.assertIsNone(summary["selected_range_min_A"])
        self.assertIsNone(summary["selected_range_max_A"])
        self.assertIsNone(summary.get("E_diss_Eh"))

    def test_missing_harmonic_record_is_rejected_without_starting_scf(self):
        with patch.object(selection, "solve_point") as solver:
            result = selection.scan_one(None, self.settings, self.wavenumber_to_Eh)
        solver.assert_not_called()
        self.assertEqual(result["summary"]["validation_status"], "NEEDS_INPUT")
        self.assertEqual(result["scan"], [])
        self.assert_no_selection(result)

    def test_unvalidated_harmonic_record_is_rejected_without_starting_scf(self):
        record = copy.deepcopy(self.records["CO"])
        record["summary"]["validation_status"] = "FAIL_VALIDATION"
        with patch.object(selection, "solve_point") as solver:
            result = selection.scan_one(record, self.settings, self.wavenumber_to_Eh)
        solver.assert_not_called()
        self.assertEqual(result["summary"]["validation_status"], "NEEDS_INPUT")
        self.assertEqual(result["scan"], [])
        self.assert_no_selection(result)

    def test_short_co_scan_does_not_fabricate_a_dissociation_threshold(self):
        result = selection.scan_one(self.records["CO"], self.settings, self.wavenumber_to_Eh)
        self.assertEqual(result["summary"]["asymptote_status"], "DISSOCIATION_ASYMPTOTE_NOT_REACHED")
        self.assertEqual(result["summary"]["validation_status"], "DISSOCIATION_ASYMPTOTE_NOT_REACHED")
        self.assertEqual(result["scan"][-1]["q_A"], 5.0)
        self.assertTrue(result["scan"][0]["accepted"])
        self.assertTrue(result["scan"][-1]["accepted"])
        self.assert_no_selection(result)

    def test_nh3_paired_orbital_continuation_remains_stable_at_large_separation(self):
        path, symbols, basis = selection.prepare_path(self.records["NH3"])
        first_q, next_q = 95.48784070349205, 192.00820412055026
        first, first_state = selection.solve_point(symbols, path(first_q), basis)
        self.assertIsNotNone(first, first_state)
        self.assertTrue(first_state["accepted"])
        continued, state = selection.solve_point(
            symbols, path(next_q), basis, first.make_rdm1(), previous=first
        )
        self.assertIsNotNone(continued, state)
        self.assertTrue(state["accepted"])
        self.assertTrue(state["scf_converged"])
        self.assertTrue(state["internal_stable"])
        self.assertLessEqual(state["orbital_gradient_norm_Eh"], selection.SCF_GRADIENT_LIMIT)
        self.assertAlmostEqual(state["energy_Eh"], -54.18550138591446, delta=1e-8)
        self.assertTrue(all(occupation in (0, 2) for occupation in continued.mo_occ))


if __name__ == "__main__":
    unittest.main()
