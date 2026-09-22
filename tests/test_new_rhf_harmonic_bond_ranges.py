"""Independent physical/convention checks; run with Python's unittest runner."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy import constants

import new_rhf_harmonic_bond_ranges as workflow


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "results/rhf_geometries/rhf_equilibrium_summary.csv"


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with METADATA.open(newline="") as handle:
            cls.rows = {r["molecule"]: r for r in csv.DictReader(handle)}

    def fixture(self, name):
        symbols, coords, basis, _ = workflow.load_equilibrium(name, self.rows[name], METADATA)
        masses = workflow.make_molecule(symbols, coords, basis).atom_mass_list(isotope_avg=True)
        return symbols, coords, masses

    def test_all_paths_against_independent_analytic_eckart_derivative(self):
        for name in workflow.MOLECULES:
            with self.subTest(molecule=name):
                symbols, coords, masses = self.fixture(name)
                path = workflow.StretchPath(name, symbols, coords, masses)
                tangent, diagnostics = path.validate()
                self.assertFalse(diagnostics["failures"])
                # Independent infinitesimal rigid-motion projection of the raw velocity.
                raw_velocity = np.zeros_like(coords)
                if name in workflow.DIATOMICS:
                    raw_velocity[1] = (coords[1] - coords[0]) / np.linalg.norm(coords[1] - coords[0])
                elif name in workflow.STARS:
                    for h in path.hydrogens:
                        displacement = coords[h] - coords[path.heavy]
                        raw_velocity[h] = displacement / np.linalg.norm(displacement)
                else:
                    o1, o2 = path.oxygens
                    displacement = coords[o2] - coords[o1]
                    raw_velocity[o2] = displacement / np.linalg.norm(displacement)
                    raw_velocity[path.peroxide_h[1]] = raw_velocity[o2]
                r = coords - np.average(coords, axis=0, weights=masses)
                v = raw_velocity - np.average(raw_velocity, axis=0, weights=masses)
                inertia = sum(m * (np.dot(x, x) * np.eye(3) - np.outer(x, x)) for m, x in zip(masses, r))
                angular_velocity = np.linalg.pinv(inertia) @ np.sum(masses[:, None] * np.cross(r, v), axis=0)
                expected = v - np.cross(angular_velocity, r)
                np.testing.assert_allclose(tangent, expected, atol=3e-9, rtol=0)
                for q in (-0.03, 0.0, 0.03):
                    positions = path(q)
                    for i, j in path.stretches:
                        difference = np.linalg.norm(positions[i] - positions[j]) - np.linalg.norm(coords[i] - coords[j])
                        self.assertAlmostEqual(difference, q, places=12)
                if name in workflow.DIATOMICS:
                    mu = np.sum(masses[:, None] * tangent**2)
                    self.assertAlmostEqual(mu, 1 / np.sum(1 / masses), places=9)

    def test_rotation_translation_and_atom_order_invariance(self):
        rotation = Rotation.from_rotvec([0.43, -0.62, 0.37]).as_matrix()
        for name in workflow.MOLECULES:
            with self.subTest(molecule=name):
                symbols, coords, masses = self.fixture(name)
                path = workflow.StretchPath(name, symbols, coords, masses)
                tangent, _ = path.validate()
                # Keep diatomic order because its arbitrary q direction has a fixed label.
                permutation = np.arange(len(symbols))[::-1] if name not in workflow.DIATOMICS else np.arange(2)
                moved = (coords @ rotation + [2.3, -3.1, 1.7])[permutation]
                other = workflow.StretchPath(name, [symbols[i] for i in permutation], moved, masses[permutation])
                other_tangent, diagnostics = other.validate()
                self.assertFalse(diagnostics["failures"])
                np.testing.assert_allclose(other_tangent, (tangent @ rotation)[permutation], atol=3e-9, rtol=0)
                self.assertAlmostEqual(other.s_eq, path.s_eq, places=12)

    def test_kabsch_cannot_reflect_chiral_peroxide(self):
        symbols, coords, masses = self.fixture("H2O2")
        reflected = coords * [-1, 1, 1]
        aligned, rotation = workflow.align(reflected, coords, masses)
        self.assertAlmostEqual(np.linalg.det(rotation), 1, places=12)
        self.assertGreater(np.linalg.norm(aligned - coords), 0.1)
        for i in range(len(coords)):
            np.testing.assert_allclose(np.linalg.norm(aligned - aligned[i], axis=1),
                                       np.linalg.norm(coords - coords[i], axis=1), atol=1e-12)

    def test_missing_geometry_or_basis_is_needs_input(self):
        for row in (None, {**self.rows["LiH"], "basis": ""},
                    {**self.rows["LiH"], "xyz_file": "missing_optimized_geometry.xyz"}):
            result = workflow.calculate("LiH", row, METADATA)
            self.assertEqual(result["summary"]["validation_status"], "NEEDS_INPUT")
            self.assertNotIn("n1_min_A", result["summary"])

    def test_non_rhf_and_incomplete_input_are_rejected(self):
        for changes in ({"method": "B3LYP"}, {"status": "failed"}, {"optimizer_success": "False"}):
            with self.assertRaises(workflow.InputProblem):
                workflow.load_equilibrium("LiH", {**self.rows["LiH"], **changes}, METADATA)

    def test_unequal_bonds_do_not_get_averaged_into_a_pass(self):
        symbols, coords, masses = self.fixture("H2O")
        coords[1] += (coords[1] - coords[0]) * 0.01
        path = workflow.StretchPath("H2O", symbols, coords, masses)
        _, diagnostics = path.validate()
        self.assertTrue(any("single common absolute s" in text for text in diagnostics["failures"]))


class OscillatorTests(unittest.TestCase):
    def test_known_si_spring_unit_conversions(self):
        # Independent SI spring: k=500 N/m, mass=2 unified atomic mass units.
        k_si, mu = 500.0, 2.0
        k_atomic = k_si * constants.physical_constants["Bohr radius"][0]**2 / constants.physical_constants["Hartree energy"][0]
        results, _ = workflow.oscillator_ranges(k_atomic, mu, 1.5)
        omega = math.sqrt(k_si / (mu * constants.atomic_mass))
        extent = math.sqrt(3 * constants.hbar / (mu * constants.atomic_mass * omega)) / 1e-10
        self.assertAlmostEqual(results["omega_rad_per_s"] / omega, 1, places=6)
        self.assertAlmostEqual(results["Delta_q_n1_A"] / extent, 1, places=6)
        self.assertAlmostEqual(results["k_q_N_per_m"] / k_si, 1, places=6)
        self.assertAlmostEqual(results["harmonic_frequency_cm1"] / (omega / (2 * math.pi * constants.c * 100)), 1, places=6)

    def test_probability_intervals_by_independent_quadrature(self):
        _, oscillator = workflow.oscillator_ranges(0.3, 1.5, 1.2)
        for interval in oscillator["probability_intervals"]:
            n, a = interval["n"], interval["half_width_in_l"]
            def density(x):
                hermite = 1 if n == 0 else 2 * x
                return math.exp(-x*x) * hermite**2 / (math.sqrt(math.pi) * 2**n * math.factorial(n))
            self.assertAlmostEqual(quad(density, -a, a)[0], interval["probability"], places=12)
            self.assertAlmostEqual(quad(density, -np.inf, np.inf)[0], 1, places=12)

    def test_coordinate_scaling_preserves_frequency_and_scales_extent(self):
        original, _ = workflow.oscillator_ranges(0.3, 1.5, 1.2)
        rescaled, _ = workflow.oscillator_ranges(0.3 * 4, 1.5 * 4, 1.2)
        self.assertEqual(original["omega_rad_per_s"], rescaled["omega_rad_per_s"])
        self.assertAlmostEqual(original["Delta_q_n1_A"], 2 * rescaled["Delta_q_n1_A"], places=14)

    def test_invalid_curvature_or_mass_is_rejected(self):
        for k, mu in ((0, 1), (-1, 1), (1, 0), (1, -1), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                workflow.oscillator_ranges(k, mu, 1.0)


class EndToEndTests(unittest.TestCase):
    def test_lih_hessian_finite_differences_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            status = workflow.main(["--metadata", str(METADATA), "--molecules", "LiH", "--output-dir", temp])
            self.assertEqual(status, 0)
            for filename in ("new_rhf_harmonic_bond_ranges.csv", "new_rhf_harmonic_bond_ranges.json",
                             "NEW_RHF_HARMONIC_BOND_RANGE_REPORT.md"):
                self.assertGreater((Path(temp) / filename).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
