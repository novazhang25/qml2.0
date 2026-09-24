"""Independent fermionic tests of the second-order MP2 wavefunction RDM.

No SCF, MP2, or FCI electronic calculation is run. Artificial amplitudes are
mapped to determinants with explicit creation/annihilation signs, independently
of the PySCF CISD/FCI density routines used by the implementation.
"""
import itertools
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pyscf.mp import mp2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
from methods41_rdm import build_consistent_rdms
from methods41_validation import dual_rdm_diagnostics
import methods41_reference as pipeline


def apply_fermions(determinant, actions):
    """Apply operators in right-to-left application order to a bitstring."""
    sign = 1
    for create, orbital in actions:
        occupied = bool(determinant & (1 << orbital))
        if occupied == create:
            return None, 0
        sign *= -1 if (determinant & ((1 << orbital) - 1)).bit_count() % 2 else 1
        determinant ^= 1 << orbital
    return determinant, sign


def independent_states(t2, nmo, nelectron, frozen_core):
    """Apply the spin-summed restricted T2 operator directly to RHF."""
    nocc = nelectron // 2
    reference = sum(1 << p for p in range(nocc))
    reference |= sum(1 << (p + nmo) for p in range(nocc))
    first_order = {}
    for i, j, a, b in itertools.product(
            range(frozen_core, nocc), range(frozen_core, nocc),
            range(nocc, nmo), range(nocc, nmo)):
        amplitude = 0.5 * t2[i - frozen_core, j - frozen_core, a - nocc, b - nocc]
        for sigma, tau in itertools.product(range(2), repeat=2):
            # T2 = 1/2 sum t_ijab E_ai E_bj, E_ai = sum_spin a†_a a_i.
            actions = [(False, j + tau * nmo), (True, b + tau * nmo),
                       (False, i + sigma * nmo), (True, a + sigma * nmo)]
            determinant, sign = apply_fermions(reference, actions)
            if sign:
                first_order[determinant] = first_order.get(determinant, 0.) + sign * amplitude
    first_order = {key: value for key, value in first_order.items() if abs(value) > 1e-15}
    return {reference: 1.}, first_order


def operator_expectation(bra, ket, actions):
    value = 0.
    for determinant, coefficient in ket.items():
        target, sign = apply_fermions(determinant, actions)
        if sign:
            value += bra.get(target, 0.) * sign * coefficient
    return value


def independent_rdm(bra, ket, nmo):
    """Spin trace using dm1[p,q]=<q†p>, Gamma[p,q,r,s]=<p†r†sq>."""
    density = np.zeros((nmo, nmo))
    gamma = np.zeros((nmo,) * 4)
    for p, q in itertools.product(range(nmo), repeat=2):
        for sigma in range(2):
            density[p, q] += operator_expectation(
                bra, ket, [(False, p + sigma * nmo), (True, q + sigma * nmo)])
    for p, q, r, s in itertools.product(range(nmo), repeat=4):
        for sigma, tau in itertools.product(range(2), repeat=2):
            gamma[p, q, r, s] += operator_expectation(bra, ket, [
                (False, q + sigma * nmo), (False, s + tau * nmo),
                (True, r + tau * nmo), (True, p + sigma * nmo),
            ])
    return density, gamma


def independent_orders(t2, nmo, nelectron, frozen_core):
    reference, first_order = independent_states(t2, nmo, nelectron, frozen_core)
    norm = sum(coefficient**2 for coefficient in first_order.values())
    p0, g0 = independent_rdm(reference, reference, nmo)
    p_left, g_left = independent_rdm(reference, first_order, nmo)
    p_right, g_right = independent_rdm(first_order, reference, nmo)
    p2, g2 = independent_rdm(first_order, first_order, nmo)
    return {
        "P_MO_order0": p0, "P_MO_order1": p_left + p_right, "P_MO_order2": p2 - norm * p0,
        "Gamma_MO_order0": g0, "Gamma_MO_order1": g_left + g_right, "Gamma_MO_order2": g2 - norm * g0,
    }, reference, first_order, norm


def independent_ci_array(state, nmo, nelectron):
    strings = sorted(sum(1 << p for p in occupied)
                     for occupied in itertools.combinations(range(nmo), nelectron // 2))
    indices = {bits: index for index, bits in enumerate(strings)}
    result = np.zeros((len(strings), len(strings)))
    for determinant, coefficient in state.items():
        alpha, beta = determinant & ((1 << nmo) - 1), determinant >> nmo
        result[indices[alpha], indices[beta]] = coefficient
    return result


class ConsistentRDMTests(unittest.TestCase):
    def assert_independent_case(self, t2, nmo, nelectron, frozen_core):
        arrays, metadata = build_consistent_rdms(t2, nmo, nelectron, frozen_core)
        expected, reference, first_order, norm = independent_orders(t2, nmo, nelectron, frozen_core)
        for key, value in expected.items():
            np.testing.assert_allclose(arrays[key], value, rtol=0, atol=2e-13, err_msg=key)
        np.testing.assert_allclose(arrays["P_MP2_MO_consistent"], sum(expected[f"P_MO_order{order}"] for order in range(3)), atol=2e-13)
        np.testing.assert_allclose(arrays["Gamma_MP2_MO_consistent"], sum(expected[f"Gamma_MO_order{order}"] for order in range(3)), atol=2e-13)
        np.testing.assert_allclose(arrays["ci_reference"], independent_ci_array(reference, nmo, nelectron), atol=2e-13)
        np.testing.assert_allclose(arrays["ci_first_order"], independent_ci_array(first_order, nmo, nelectron), atol=2e-13)
        self.assertAlmostEqual(float(arrays["mp2_first_order_norm_squared"]), norm, places=13)
        np.testing.assert_allclose(arrays["ci_second_order_normalization"], -0.5 * norm * arrays["ci_reference"], atol=2e-13)
        self.assertEqual(metadata["rdm_source"], "normalized truncated MP2 wavefunction, expectation values through second order")
        return arrays

    def test_two_electron_opposite_spin_amplitude_mapping(self):
        arrays = self.assert_independent_case(np.array([[[[0.2]]]]), 2, 2, 0)
        np.testing.assert_allclose(arrays["P_MP2_MO_consistent"], np.diag([1.92, 0.08]), atol=1e-13)
        self.assertAlmostEqual(arrays["Gamma_MP2_MO_consistent"][0, 1, 0, 1], 0.4, places=13)

    def test_same_spin_double_excitations_and_general_index_order(self):
        rng = np.random.default_rng(4102)
        trial = rng.normal(scale=0.04, size=(2, 2, 2, 2))
        t2 = (trial + trial.transpose(1, 0, 3, 2)) / 2
        arrays = self.assert_independent_case(t2, 4, 4, 0)
        # At least one same-spin double changes alpha occupation only.
        self.assertGreater(abs(arrays["ci_first_order"][-1, 0]), 1e-5)
        self.assertGreater(abs(arrays["ci_first_order"][0, -1]), 1e-5)

    def test_frozen_core_is_restored_to_full_electron_rdms(self):
        arrays = self.assert_independent_case(np.array([[[[0.2]]]]), 3, 4, 1)
        np.testing.assert_allclose(arrays["P_MP2_MO_consistent"], np.diag([2., 1.92, 0.08]), atol=1e-13)
        gamma = arrays["Gamma_MP2_MO_consistent"]
        self.assertAlmostEqual(gamma[0, 0, 0, 0], 2., places=13)
        self.assertAlmostEqual(gamma[0, 0, 2, 2], 0.16, places=13)
        self.assertAlmostEqual(np.einsum("pprr->", gamma), 12., places=12)

    def test_same_spin_determinant_signs_with_a_frozen_core(self):
        rng = np.random.default_rng(4105)
        trial = rng.normal(scale=0.04, size=(2, 2, 2, 2))
        t2 = (trial + trial.transpose(1, 0, 3, 2)) / 2
        self.assert_independent_case(t2, 5, 6, 1)

    def test_contractions_particle_number_and_symmetries_at_every_order(self):
        rng = np.random.default_rng(413)
        trial = rng.normal(scale=0.06, size=(2, 2, 2, 2))
        t2 = (trial + trial.transpose(1, 0, 3, 2)) / 2
        arrays, _ = build_consistent_rdms(t2, 4, 4, 0)
        for order in range(3):
            with self.subTest(order=order):
                density = arrays[f"P_MO_order{order}"]
                gamma = arrays[f"Gamma_MO_order{order}"]
                np.testing.assert_allclose(np.einsum("pqrr->pq", gamma), 3 * density.T, atol=2e-13)
                self.assertAlmostEqual(np.trace(density), 4. if order == 0 else 0., places=12)
                self.assertAlmostEqual(np.einsum("pprr->", gamma), 12. if order == 0 else 0., places=12)
                np.testing.assert_allclose(density, density.T, atol=2e-13)
                np.testing.assert_allclose(gamma, gamma.transpose(2, 3, 0, 1), atol=2e-13)
                np.testing.assert_allclose(gamma, gamma.transpose(1, 0, 3, 2), atol=2e-13)

    def test_order_coefficients_scale_linearly_and_quadratically(self):
        t2 = np.array([[[[-0.17]]]])
        one, _ = build_consistent_rdms(t2, 2, 2, 0)
        two, _ = build_consistent_rdms(2 * t2, 2, 2, 0)
        for prefix in ("P", "Gamma"):
            for order in range(3):
                key = f"{prefix}_MO_order{order}"
                np.testing.assert_allclose(two[key], 2**order * one[key], rtol=0, atol=2e-13)

    def test_second_order_normalization_is_expansion_not_exact_quotient(self):
        amplitude = 0.3
        arrays, _ = build_consistent_rdms(np.array([[[[amplitude]]]]), 2, 2, 0)
        expected_expanded = np.diag([2 * (1 - amplitude**2), 2 * amplitude**2])
        exact_normalized = np.diag([2., 2 * amplitude**2]) / (1 + amplitude**2)
        np.testing.assert_allclose(arrays["P_MP2_MO_consistent"], expected_expanded, atol=1e-13)
        self.assertGreater(np.max(np.abs(arrays["P_MP2_MO_consistent"] - exact_normalized)), 0.01)
        self.assertAlmostEqual(float(arrays["wavefunction_norm_order0"]), 1., places=13)
        self.assertAlmostEqual(float(arrays["wavefunction_norm_order1"]), 0., places=13)
        self.assertAlmostEqual(float(arrays["wavefunction_norm_order2"]), 0., places=13)
        self.assertAlmostEqual(np.linalg.norm(arrays["ci_normalized_eta1"]), 1., places=13)

    def test_zero_amplitudes_recover_rhf_and_zero_hf_reference_cumulant(self):
        arrays = self.assert_independent_case(np.zeros((1, 1, 1, 1)), 3, 4, 1)
        density = arrays["P_MO_order0"]
        reference = np.einsum("pq,rs->pqrs", density, density) - 0.5 * np.einsum("ps,rq->pqrs", density, density)
        np.testing.assert_allclose(arrays["Gamma_MP2_MO_consistent"], reference, atol=1e-13)
        for order in (1, 2):
            np.testing.assert_array_equal(arrays[f"Gamma_MO_order{order}"], np.zeros((3,) * 4))

    def test_hf_reference_cumulant_has_correct_nonzero_contraction(self):
        arrays, _ = build_consistent_rdms(np.array([[[[0.2]]]]), 3, 4, 1)
        cumulant = arrays["Gamma_MP2_MO_consistent"] - arrays["Gamma_MO_order0"]
        correction = arrays["P_MP2_MO_consistent"] - arrays["P_MO_order0"]
        expected = 3 * correction.T
        self.assertGreater(np.max(np.abs(expected)), 0.2)
        np.testing.assert_allclose(np.einsum("pqrr->pq", cumulant), expected, atol=1e-13)

    def test_library_raw_rdm_is_preserved_as_a_distinct_noncontracting_quantity(self):
        t2 = np.array([[[[0.2]]]])
        fake_mp = SimpleNamespace(t2=t2, nmo=2, nocc=1, frozen=None,
                                  mo_occ=np.array([2., 0.]), mo_coeff=np.eye(2))
        raw_density, raw_gamma = mp2.make_rdm1(fake_mp), mp2.make_rdm2(fake_mp)
        arrays, _ = build_consistent_rdms(t2, 2, 2, 0)
        consistent_density, consistent_gamma = arrays["P_MP2_MO_consistent"], arrays["Gamma_MP2_MO_consistent"]
        np.testing.assert_allclose(consistent_density, raw_density, atol=1e-13)
        np.testing.assert_allclose(np.einsum("pqrr->pq", raw_gamma) - raw_density, 0.08 * np.eye(2), atol=1e-13)
        np.testing.assert_allclose(np.einsum("pqrr->pq", consistent_gamma), consistent_density, atol=1e-13)
        self.assertGreater(np.max(np.abs(raw_gamma - consistent_gamma)), 0.05)

    def make_dual_arrays(self):
        t2 = np.array([[[[0.2]]]])
        fake_mp = SimpleNamespace(t2=t2, nmo=2, nocc=1, frozen=None,
                                  mo_occ=np.array([2., 0.]), mo_coeff=np.eye(2))
        raw_density, raw_gamma = mp2.make_rdm1(fake_mp), mp2.make_rdm2(fake_mp)
        overlap = np.array([[1.2, 0.2], [0.2, 0.9]])
        _, coefficients, _ = pipeline.lowdin_factors(overlap)
        arrays = {
            "mp2_t2": t2, "mo_coeff": coefficients, "S_AO": overlap,
            "P_AO": coefficients @ np.diag([2., 0.]) @ coefficients.T,
            "P_MP2_MO_pyscf": raw_density, "Gamma_MP2_MO_pyscf": raw_gamma,
            "P_MP2_AO": coefficients @ raw_density @ coefficients.T,
            "Gamma_MP2_AO": pipeline.transform_rank4(raw_gamma, coefficients),
        }
        pipeline.add_consistent_rdms(arrays, SimpleNamespace(electron_count=2, input_fci_frozen_core=0))
        return arrays, raw_gamma, coefficients

    def test_standard_audit_failure_does_not_veto_consistent_diagnostics(self):
        arrays, raw_gamma, coefficients = self.make_dual_arrays()
        diagnostics, terms = dual_rdm_diagnostics(arrays, 2, 0)
        self.assertTrue(diagnostics["passed"], diagnostics["checks"])
        self.assertFalse(diagnostics["pyscf_audit"]["MO"]["passed"])
        self.assertFalse(diagnostics["pyscf_audit"]["AO"]["passed"])
        self.assertGreater(diagnostics["pyscf_audit"]["MO"]["checks"]["contraction"]["value"], 0.05)
        self.assertTrue(all(check["passed"] for check in diagnostics["checks"].values()))
        np.testing.assert_allclose(arrays["Gamma_MP2_AO"], arrays["Gamma_MP2_AO_consistent"], atol=1e-13)
        np.testing.assert_allclose(arrays["Gamma_MP2_AO_pyscf"], pipeline.transform_rank4(raw_gamma, coefficients), atol=1e-13)
        self.assertGreater(np.max(np.abs(terms["conventional_cumulant_contraction_rhs"])), 0.05)
        np.testing.assert_allclose(terms["conventional_cumulant_contraction_residual"], 0., atol=1e-13)

    def test_malformed_or_nonfinite_audit_is_reported_without_vetoing_consistent_track(self):
        for basis, bad_tensor in (("AO", np.full((2,) * 4, np.nan)), ("MO", np.zeros((2,) * 3))):
            with self.subTest(basis=basis):
                arrays, _, _ = self.make_dual_arrays()
                arrays[f"Gamma_MP2_{basis}_pyscf"] = bad_tensor
                diagnostics, _ = dual_rdm_diagnostics(arrays, 2, 0)
                self.assertTrue(diagnostics["passed"], diagnostics["checks"])
                self.assertFalse(diagnostics["pyscf_audit"][basis]["passed"])
                self.assertIn("error", diagnostics["pyscf_audit"][basis])
                self.assertTrue(all(check["passed"] for check in diagnostics["checks"].values()))
                json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
