"""Focused Methods 4.1 checks without any SCF, MP2, or FCI calculation.

These tests use synthetic matrices, artificial MP2 amplitudes, and PySCF
molecule construction only. The production smoke test is a separate command.
"""
import itertools
import contextlib
import csv
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
from pyscf import gto
from pyscf.mp import mp2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
import methods41_reference as pipeline
import methods41_data as input_data


def synthetic_rhf(nmo=3, nocc=2):
    """A nonorthogonal AO metric and an exactly idempotent RHF density."""
    rng = np.random.default_rng(415)
    matrix = rng.normal(size=(nmo, nmo))
    overlap = matrix @ matrix.T + np.eye(nmo)
    half, minus_half, _ = pipeline.lowdin_factors(overlap)
    rotation, _ = np.linalg.qr(rng.normal(size=(nmo, nmo)))
    coefficients = minus_half @ rotation
    occupations = np.zeros(nmo)
    occupations[:nocc] = 2
    density = (coefficients * occupations) @ coefficients.T
    return overlap, half, minus_half, coefficients, occupations, density


class CumulantTests(unittest.TestCase):
    def test_disconnected_rhf_index_convention(self):
        density = np.array([[1.7, -0.3, 0.1], [-0.3, 0.2, 0.4], [0.1, 0.4, 0.1]])
        expected = np.empty((3,) * 4)
        for p, q, r, s in itertools.product(range(3), repeat=4):
            expected[p, q, r, s] = density[p, q] * density[r, s] - 0.5 * density[p, s] * density[r, q]
        np.testing.assert_allclose(pipeline.disconnected_rhf(density), expected, rtol=0, atol=1e-15)

    def test_no_correlation_cumulant_and_nonorthogonal_contraction(self):
        overlap, half, _, coefficients, occupations, density = synthetic_rhf()
        gamma0 = pipeline.disconnected_rhf(density)
        np.testing.assert_allclose(np.einsum("pqrs,sr->pq", gamma0, overlap), 3 * density, atol=1e-13)
        np.testing.assert_allclose(density @ overlap @ density, 2 * density, atol=1e-13)
        np.testing.assert_allclose(half @ density @ half, (half @ density @ half).T, atol=1e-13)
        diagnostics = pipeline.validate_rdms(
            density, overlap, gamma0, density, 4,
            mo_coeff=coefficients, mo_occ=occupations,
        )
        self.assertTrue(diagnostics["passed"], diagnostics)
        cumulant = gamma0 - pipeline.disconnected_rhf(density)
        np.testing.assert_array_equal(cumulant, np.zeros((3,) * 4))

    def test_nonfinite_rdm_is_rejected(self):
        density = np.diag([2.0, 0.0])
        gamma = pipeline.disconnected_rhf(density)
        gamma[0, 0, 0, 0] = np.nan
        with self.assertRaises(pipeline.ValidationError):
            pipeline.validate_rdms(density, np.eye(2), gamma, density, 2)

    def test_wrong_tensor_shape_is_rejected(self):
        density = np.diag([2.0, 0.0])
        with self.assertRaises(pipeline.ValidationError):
            pipeline.validate_rdms(density, np.eye(2), np.zeros((2, 2, 2)), density, 2)

    def test_complex_rdm_is_rejected_instead_of_discarding_imaginary_part(self):
        density = np.diag([2.0, 0.0])
        gamma = pipeline.disconnected_rhf(density).astype(complex)
        gamma[0, 0, 0, 0] += 0.1j
        with self.assertRaises(pipeline.ValidationError):
            pipeline.validate_rdms(density, np.eye(2), gamma, density, 2)

    def test_pyscf_zero_amplitudes_have_exact_rhf_reference(self):
        fake_mp = SimpleNamespace(t2=np.zeros((1, 1, 1, 1)), nmo=2, nocc=1,
                                  frozen=None, mo_occ=np.array([2.0, 0.0]), mo_coeff=np.eye(2))
        density = mp2.make_rdm1(fake_mp)
        gamma = mp2.make_rdm2(fake_mp)
        np.testing.assert_array_equal(gamma, pipeline.disconnected_rhf(density))
        diagnostics = pipeline.validate_rdms(density, np.eye(2), gamma, density, 2,
                                              mo_coeff=np.eye(2), mo_occ=fake_mp.mo_occ)
        self.assertTrue(diagnostics["passed"], diagnostics)

    def test_pyscf_nonzero_amplitudes_fail_strict_particle_number_check(self):
        # PySCF 2.14's perturbative 2-RDM is not an N-representable 2-RDM.
        # This regression test preserves the requested physical failure gate;
        # it must not normalize or repair the library output to make it pass.
        amplitude = 0.1
        fake_mp = SimpleNamespace(t2=np.array([[[[amplitude]]]]), nmo=2, nocc=1,
                                  frozen=None, mo_occ=np.array([2.0, 0.0]), mo_coeff=np.eye(2))
        density_hf = np.diag([2.0, 0.0])
        density_mp2 = mp2.make_rdm1(fake_mp)
        gamma = mp2.make_rdm2(fake_mp)
        np.testing.assert_allclose(density_mp2, np.diag([2 - 2 * amplitude**2, 2 * amplitude**2]), atol=1e-15)
        contraction = np.einsum("pqrr->pq", gamma)
        np.testing.assert_allclose(contraction - density_mp2, 2 * amplitude**2 * np.eye(2), atol=1e-15)
        self.assertAlmostEqual(np.einsum("pprr->", gamma), 2 + 4 * amplitude**2, places=14)
        diagnostics = pipeline.validate_rdms(density_hf, np.eye(2), gamma, density_mp2, 2,
                                              mo_coeff=np.eye(2), mo_occ=fake_mp.mo_occ)
        self.assertFalse(diagnostics["passed"])
        self.assertFalse(diagnostics["checks"]["rdm2_contraction_max_error"]["passed"])
        self.assertFalse(diagnostics["checks"]["rdm2_particle_number_error"]["passed"])
        self.assertFalse(diagnostics["checks"]["cumulant_contraction_max_error"]["passed"])


class LowdinTests(unittest.TestCase):
    def test_full_metric_and_density_invariants(self):
        overlap, half, minus_half, _, _, density = synthetic_rhf()
        np.testing.assert_allclose(half @ half, overlap, rtol=0, atol=1e-13)
        np.testing.assert_allclose(minus_half @ overlap @ minus_half, np.eye(3), rtol=0, atol=1e-13)
        density_l = half @ density @ half
        self.assertAlmostEqual(np.trace(density @ overlap), 4, places=12)
        self.assertAlmostEqual(np.trace(density_l), 4, places=12)
        np.testing.assert_allclose(density_l @ density_l, 2 * density_l, atol=1e-13)

    def test_near_linear_dependence_is_rejected(self):
        for diagonal in ([1.0, 1e-12], [1.0, 0.0], [1.0, -0.1]):
            with self.subTest(diagonal=diagonal), self.assertRaises(pipeline.ValidationError):
                pipeline.lowdin_factors(np.diag(diagonal), min_eigenvalue=1e-8)

    def test_rank_four_transform_against_eight_nested_index_loops(self):
        rng = np.random.default_rng(41)
        tensor = rng.normal(size=(3,) * 4)
        transform = rng.normal(size=(3, 3))
        expected = np.zeros((3,) * 4)
        for a, b, c, d in itertools.product(range(3), repeat=4):
            for p, q, r, s in itertools.product(range(3), repeat=4):
                expected[a, b, c, d] += (
                    transform[a, p] * transform[b, q] * transform[c, r]
                    * transform[d, s] * tensor[p, q, r, s]
                )
        np.testing.assert_allclose(pipeline.transform_rank4(tensor, transform), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pipeline.explicit_rank4(tensor, transform), expected, rtol=1e-12, atol=1e-12)

    def test_remove_core_only_after_full_transform(self):
        overlap = np.array([[2.0, 0.6, -0.1], [0.6, 1.4, 0.2], [-0.1, 0.2, 1.0]])
        density = np.array([[1.0, 0.2, 0.3], [0.2, 0.8, -0.4], [0.3, -0.4, 0.6]])
        half, _, _ = pipeline.lowdin_factors(overlap)
        density_l = half @ density @ half
        descriptors = pipeline.extract_descriptors(density_l, np.eye(3), np.zeros((3,) * 4), [1, 2])
        np.testing.assert_allclose(descriptors["P_mu"], np.diag(density_l)[[1, 2]])
        valence_half, _, _ = pipeline.lowdin_factors(overlap[1:, 1:])
        premature = valence_half @ density[1:, 1:] @ valence_half
        self.assertGreater(np.max(np.abs(descriptors["P_mu"] - np.diag(premature))), 1e-3)


class DescriptorTests(unittest.TestCase):
    def test_signed_pairs_and_triples_preserve_exact_index_order(self):
        # Deliberately nonsymmetric data detects averaging, axis swaps, sums,
        # and absolute values, independently of physical tensor symmetries.
        density = np.arange(25.0).reshape(5, 5) - 20
        fock = -density / 7
        cumulant = np.arange(625.0).reshape((5,) * 4) - 500
        original = cumulant.copy()
        valence = [0, 2, 3, 4]
        descriptors = pipeline.extract_descriptors(density, fock, cumulant, valence)
        pair_indices = list(itertools.combinations(range(4), 2))
        triple_indices = list(itertools.combinations(range(4), 3))
        np.testing.assert_array_equal(descriptors["pair_indices"], pair_indices)
        np.testing.assert_array_equal(descriptors["triple_indices"], triple_indices)
        np.testing.assert_array_equal(descriptors["P_mu"], np.diag(density)[valence])
        np.testing.assert_array_equal(descriptors["F_mu"], np.diag(fock)[valence])
        np.testing.assert_array_equal(descriptors["P_munu"], [density[valence[i], valence[j]] for i, j in pair_indices])
        expected = np.empty((4, 4, 4), dtype=cumulant.dtype)
        for i, j, k in itertools.product(range(4), repeat=3):
            expected[i, j, k] = cumulant[valence[i], valence[k], valence[j], valence[k]]
        np.testing.assert_array_equal(descriptors["T_full"], expected)
        np.testing.assert_array_equal(descriptors["T_munulambda"], [expected[i, j, k] for i, j, k in triple_indices])
        np.testing.assert_array_equal(cumulant, original)
        old = np.diagonal(cumulant[np.ix_(valence, valence, valence, valence)], axis1=2, axis2=3)
        self.assertFalse(np.array_equal(descriptors["T_full"], old))
        self.assertTrue(np.any(np.asarray(descriptors["T_munulambda"]) < 0))

    def test_descriptor_dimensions_include_empty_pair_and_triple_arrays(self):
        for count in (1, 2, 3, 6, 10):
            with self.subTest(nvalence=count):
                descriptors = pipeline.extract_descriptors(np.eye(count), np.eye(count), np.zeros((count,) * 4), list(range(count)))
                self.assertEqual(descriptors["P_mu"].shape, (count,))
                self.assertEqual(descriptors["F_mu"].shape, (count,))
                self.assertEqual(descriptors["pair_indices"].shape, (math.comb(count, 2), 2))
                self.assertEqual(descriptors["P_munu"].shape, (math.comb(count, 2),))
                self.assertEqual(descriptors["triple_indices"].shape, (math.comb(count, 3), 3))
                self.assertEqual(descriptors["T_munulambda"].shape, (math.comb(count, 3),))
                self.assertEqual(descriptors["T_full"].shape, (count,) * 3)


class ValenceSelectionTests(unittest.TestCase):
    """Pin the archived uniform highest-shell rule, not the revised AO subspace."""

    def test_legacy_dataset_descriptor_dimensions_from_sto3g_shells(self):
        cases = {
            "LiH": (["Li", "H"], 5, 1),
            "HF": (["H", "F"], 5, 1),
            "BeH2": (["Be", "H", "H"], 6, 1),
            "H2O": (["O", "H", "H"], 6, 1),
            "H2S": (["S", "H", "H"], 6, 5),
            "NH3": (["N", "H", "H", "H"], 7, 1),
            "N2": (["N", "N"], 8, 2),
            "CO": (["C", "O"], 8, 2),
            "H2O2": (["H", "O", "O", "H"], 10, 2),
        }
        for name, (symbols, nvalence, ncore) in cases.items():
            with self.subTest(molecule=name):
                mol = gto.M(atom=[(symbol, (i * 1.5, 0, 0)) for i, symbol in enumerate(symbols)],
                            basis="sto-3g", spin=0, charge=0, unit="Angstrom", verbose=0)
                selection = pipeline.select_valence_aos(mol)
                self.assertEqual(len(selection["valence_indices"]), nvalence)
                self.assertEqual(len(selection["core_indices"]), ncore)
                self.assertEqual(len(selection["ao_labels"]), mol.nao_nr())
                self.assertEqual(sorted(list(selection["valence_indices"]) + list(selection["core_indices"])), list(range(mol.nao_nr())))
                self.assertFalse(set(selection["core_indices"]) & set(selection["valence_indices"]))
                self.assertEqual(list(selection["valence_labels"]), [selection["ao_labels"][i] for i in selection["valence_indices"]])

    def test_legacy_third_period_atoms_keep_only_highest_shell(self):
        for symbol, charge in (("Na", 1), ("Mg", 0), ("Al", 1), ("Si", 0), ("P", 1), ("S", 0), ("Cl", 1), ("Ar", 0)):
            with self.subTest(atom=symbol):
                mol = gto.M(atom=f"{symbol} 0 0 0", basis="sto-3g", charge=charge, spin=0, verbose=0)
                selection = pipeline.select_valence_aos(mol)
                self.assertEqual(len(selection["core_indices"]), 5)
                self.assertEqual(len(selection["valence_indices"]), 4)
                shell_names = [mol.ao_labels(fmt=False)[i][2] for i in selection["valence_indices"]]
                self.assertTrue(all(label.startswith("3") for label in shell_names), shell_names)

    def test_ao_labels_and_mask_are_geometry_independent(self):
        selections = []
        for distance in (0.8, 1.1, 1.6):
            mol = gto.M(atom=[("H", (0, distance, 0)), ("O", (0, 0, 0)), ("H", (distance, 0, 0))],
                        basis="sto-3g", spin=0, verbose=0)
            selections.append(pipeline.select_valence_aos(mol))
        for key in ("ao_labels", "core_indices", "valence_indices", "core_labels", "valence_labels"):
            np.testing.assert_array_equal(selections[0][key], selections[1][key])
            np.testing.assert_array_equal(selections[0][key], selections[2][key])


class DiscoveryTests(unittest.TestCase):
    """Read the source data; mutate only isolated temporary fixture copies."""

    source = Path(__file__).resolve().parents[1] / "results" / "initialdata"

    def setUp(self):
        if not (self.source / "initialdata.json").is_file():
            self.skipTest("The repository initialdata fixture is unavailable")
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.fixture = Path(self.workspace.name)
        for filename in ("initialdata.json", "scan.csv", "source_initialdata.py"):
            shutil.copy2(self.source / filename, self.fixture / filename)
        shutil.copytree(self.source / "H2O", self.fixture / "H2O")

    def assert_h2o_failure(self, expected_message):
        records, report = input_data.discover_dataset(self.fixture, ["H2O"])
        self.assertEqual(records, [])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["failed_geometry_count"], 30)
        self.assertIn(expected_message, " ".join(report["molecules"]["H2O"]["errors"]))

    def test_existing_dataset_has_exact_matching_thirty_point_scans(self):
        records, report = input_data.discover_dataset(self.source)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(len(records), 270)
        self.assertEqual(report["valid_geometry_count"], 270)
        for name in input_data.MOLECULES:
            selected = [record for record in records if record.molecule == name]
            self.assertEqual([record.geometry_id for record in selected], [f"{i:03d}" for i in range(1, 31)])
            self.assertTrue(all(record.spin == 0 and record.charge == 0 for record in selected))
            self.assertTrue(all(record.basis == "sto-3g" for record in selected))
            self.assertTrue(all(abs(record.E_corr_input - (record.E_FCI - record.E_RHF_input)) <= 1e-12 for record in selected))
            expected_core = 5 if name == "H2S" else 2 if name in ("N2", "CO", "H2O2") else 1
            self.assertEqual({record.input_fci_frozen_core for record in selected}, {expected_core})

    def test_discovery_sorts_ids_despite_shuffled_metadata_and_csv(self):
        manifest_path = self.fixture / "initialdata.json"
        manifest = json.loads(manifest_path.read_text())
        for result in manifest["results"]:
            if result["molecule"] == "H2O":
                result["scan"].reverse()
        manifest_path.write_text(json.dumps(manifest))
        local_path = self.fixture / "H2O" / "result.json"
        local = json.loads(local_path.read_text())
        local["scan"] = local["scan"][11:] + local["scan"][:11]
        local_path.write_text(json.dumps(local))
        csv_path = self.fixture / "scan.csv"
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reversed(rows))
        records, report = input_data.discover_dataset(self.fixture, ["H2O"])
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual([record.geometry_id for record in records], [f"{i:03d}" for i in range(1, 31)])

    def test_duplicate_energy_row_is_reported(self):
        path = self.fixture / "scan.csv"
        lines = path.read_text().splitlines()
        duplicate = next(line for line in lines[1:] if line.startswith("H2O,"))
        path.write_text("\n".join(lines + [duplicate]) + "\n")
        self.assert_h2o_failure("duplicate")

    def test_geometry_energy_row_mismatch_is_reported(self):
        path = self.fixture / "H2O" / "scan" / "001" / "geometry.xyz"
        lines = path.read_text().splitlines()
        fields = lines[2].split()
        fields[1] = str(float(fields[1]) + 0.01)
        lines[2] = " ".join(fields)
        path.write_text("\n".join(lines) + "\n")
        self.assert_h2o_failure("XYZ/metadata coordinates differ")

    def test_disagreeing_correlation_metadata_is_reported(self):
        path = self.fixture / "H2O" / "scan" / "001" / "point.json"
        point = json.loads(path.read_text())
        point["E_corr_Ha"] += 0.001
        path.write_text(json.dumps(point))
        self.assert_h2o_failure("E_corr_Ha differs")


class CommandFlowTests(unittest.TestCase):
    """Exercise orchestration with a mocked electronic calculation only."""

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        self.record = input_data.InputGeometry(
            molecule="H2O", geometry_id="001", point_index=1,
            geometry_path=self.source / "geometry.xyz", state_path=self.source / "electronic_state.npz",
            symbols=["O", "H", "H"], coordinates=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
            charge=0, spin=0, basis="sto-3g", electron_count=10, input_fci_frozen_core=1,
            q_A=0., bond_length_A=1., E_RHF_input=-75., E_FCI=-75.1, E_corr_input=-0.1, raw={},
        )
        self.records = [self.record]
        self.audit = {"status": "PASS", "molecules": {"H2O": {"status": "PASS"}},
                      "valid_geometry_count": 30, "failed_geometry_count": 0}
        self.arguments = ["--input-dir", str(self.source), "--output-dir", str(self.output), "--molecules", "H2O"]

    def synthetic_electronic_arrays(self, record, mol):
        # Exactly zero correlation makes a physically valid synthetic tensor.
        # No integrals or SCF/MP2 solver are evaluated here.
        n = mol.nao_nr()
        occupations = np.zeros(n)
        occupations[:record.electron_count // 2] = 2
        density = np.diag(occupations)
        return {
            "S_AO": np.eye(n), "P_AO": density, "F_AO": np.diag(np.arange(n) - 4.),
            "Gamma_MP2_AO": pipeline.disconnected_rhf(density), "P_MP2_AO": density.copy(),
            "P_MP2_MO_pyscf": density.copy(), "Gamma_MP2_MO_pyscf": pipeline.disconnected_rhf(density),
            "hcore_AO": -7.5 * np.eye(n), "eri_AO": np.zeros((n,) * 4), "E_nuclear": np.asarray(0.),
            "E_RHF_check": np.asarray(record.E_RHF_input), "E_MP2_correlation": np.asarray(0.),
            "mo_coeff": np.eye(n), "mo_occ": occupations, "mo_energy": np.arange(n) - 4.,
            "mp2_t2": np.zeros((5, 5, 2, 2)),
        }, {
            "rhf_converged": True, "rhf_gradient_norm": 0.,
            "mp2_convergence_status": "synthetic noniterative test data",
            "mp2_min_occupied_virtual_gap_Ha": 1., "zero_t2_cumulant_max_error": 0.,
        }

    def run_mocked(self, extra_arguments):
        with patch.object(pipeline, "discover_dataset", return_value=(self.records, dict(self.audit))), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return pipeline.main(self.arguments + extra_arguments)

    def test_dry_run_performs_no_electronic_calculation_or_output_write(self):
        with patch.object(pipeline, "calculate_record", side_effect=AssertionError("No electronic calculation")), \
             patch.object(pipeline, "write_json", side_effect=AssertionError("No output writes")):
            self.assertEqual(self.run_mocked(["--dry-run"]), 0)
        self.assertFalse(self.output.exists())

    def test_successful_record_resumes_repeatedly_without_calculation(self):
        with patch.object(pipeline, "calculate_rhf_rmp2", side_effect=self.synthetic_electronic_arrays) as calculate:
            self.assertEqual(self.run_mocked(["--geometry-id", "001"]), 0)
            self.assertEqual(calculate.call_count, 1)
        with patch.object(pipeline, "calculate_record", side_effect=AssertionError("Successful record must be skipped")):
            self.assertEqual(self.run_mocked(["--geometry-id", "001", "--resume"]), 0)
            self.assertEqual(self.run_mocked(["--geometry-id", "001", "--resume"]), 0)
        with np.load(self.output / "H2O" / "001.npz", allow_pickle=False) as arrays:
            self.assertEqual(str(arrays["status"]), "PASS")
            diagnostics = json.loads(str(arrays["validation_json"]))
            self.assertTrue(diagnostics["checks"]["zero_t2_cumulant_max_error"]["passed"])
            self.assertTrue(diagnostics["rhf_converged"])
            for key in ("S_AO", "P_AO", "F_AO", "Gamma_MP2_AO", "Gamma0_AO", "Lambda_AO", "S_half",
                        "S_minus_half", "P_L", "F_L", "Lambda_L", "P_mu", "F_mu", "T_full"):
                self.assertIn(key, arrays.files)

    def test_failed_record_requires_explicit_retry_and_success_is_then_skipped(self):
        with patch.object(pipeline, "calculate_rhf_rmp2", side_effect=RuntimeError("Synthetic failure")):
            self.assertEqual(self.run_mocked(["--geometry-id", "001"]), 1)
        with patch.object(pipeline, "calculate_record", side_effect=AssertionError("Failed record requires explicit retry")):
            self.assertEqual(self.run_mocked(["--geometry-id", "001", "--resume"]), 1)
        with patch.object(pipeline, "calculate_rhf_rmp2", side_effect=self.synthetic_electronic_arrays) as calculate:
            self.assertEqual(self.run_mocked(["--geometry-id", "001", "--rerun-failed"]), 0)
            self.assertEqual(calculate.call_count, 1)
        with patch.object(pipeline, "calculate_record", side_effect=AssertionError("Successful record must be skipped")):
            self.assertEqual(self.run_mocked(["--geometry-id", "001", "--rerun-failed"]), 0)

    def test_failed_smoke_gate_prevents_remaining_dataset_calculations(self):
        self.records.append(replace(self.record, geometry_id="002", point_index=2, q_A=0.1))
        failed = {"passed": False, "failed_checks": ["rdm2_contraction_max_error"]}
        with patch.object(pipeline, "calculate_record", return_value=(None, failed)) as calculate:
            self.assertEqual(self.run_mocked([]), 1)
            self.assertEqual(calculate.call_count, 1)
            self.assertEqual(calculate.call_args.args[0].geometry_id, "001")
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertIn("smoke test did not pass", manifest["stopped_reason"])
        self.assertEqual(manifest["successful_geometry_count"], 0)
        self.assertEqual(manifest["failed_geometry_count"], 1)

    def test_formal_failure_after_smoke_stops_remaining_geometries(self):
        self.records.extend(replace(self.record, geometry_id=f"{index:03d}", point_index=index, q_A=index / 10)
                            for index in (2, 3))

        def fake_record(record, metadata, independent):
            if record.geometry_id == "001":
                return None, {"passed": True}
            if record.geometry_id == "002":
                return None, {"passed": False, "failed_checks": ["consistent_AO_contraction"]}
            raise AssertionError("No calculation may follow a formal validation failure")

        with patch.object(pipeline, "calculate_record", side_effect=fake_record) as calculate:
            self.assertEqual(self.run_mocked([]), 1)
            self.assertEqual([call.args[0].geometry_id for call in calculate.call_args_list], ["001", "002"])
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertIn("Formal validation failed for H2O/002", manifest["stopped_reason"])
        self.assertEqual(manifest["successful_geometry_count"], 1)
        self.assertEqual(manifest["failed_geometry_count"], 1)

    def complete_synthetic_record(self):
        mol = pipeline.make_molecule(self.record)
        metadata = pipeline.select_valence_aos(mol)
        with patch.object(pipeline, "calculate_rhf_rmp2", side_effect=self.synthetic_electronic_arrays):
            arrays, diagnostics = pipeline.calculate_record(self.record, metadata)
        self.assertTrue(diagnostics["passed"], diagnostics)
        return arrays

    def test_nonfinite_raw_audit_does_not_fail_complete_formal_record(self):
        arrays = self.complete_synthetic_record()
        arrays["Gamma_MP2_AO_pyscf"] = np.full(arrays["Gamma_MP2_AO_pyscf"].shape, np.nan)
        diagnostics = pipeline.validate_record(arrays, self.record)
        self.assertTrue(diagnostics["passed"], diagnostics["failed_checks"])
        self.assertFalse(diagnostics["pyscf_standard_audit"]["passed"])
        self.assertFalse(diagnostics["dual_rdm"]["pyscf_audit"]["AO"]["passed"])
        self.assertIsNone(diagnostics["energy_audit"]["pyscf_full_H_expectation_Ha"])
        self.assertIsNone(diagnostics["energy_audit"]["full_H_difference_Ha"])
        json.dumps(pipeline.jsonable(diagnostics), allow_nan=False)

    def test_mutated_formal_alias_fails_explicit_alias_check(self):
        source = self.complete_synthetic_record()
        for key in ("P_MP2_AO", "Gamma_MP2_AO", "Gamma0_AO", "Lambda_AO"):
            with self.subTest(alias=key):
                arrays = dict(source)
                arrays[key] = arrays[key].copy()
                arrays[key].flat[0] += 0.1
                diagnostics = pipeline.validate_record(arrays, self.record)
                self.assertFalse(diagnostics["passed"])
                self.assertTrue(any("alias" in name and not check["passed"]
                                    for name, check in diagnostics["checks"].items()), diagnostics["failed_checks"])

    def test_calculation_requests_no_frozen_orbitals_despite_input_fci_metadata(self):
        occupations = np.array([2., 2., 2., 2., 2., 0., 0.])
        density = np.diag(occupations)
        np.savez(self.record.state_path, rhf_density_ao=density)
        energies = np.arange(7.) - 4
        mf = Mock(converged=True, e_tot=-75., mo_occ=occupations,
                  mo_coeff=np.eye(7), mo_energy=energies)
        mf.get_ovlp.return_value = np.eye(7)
        mf.get_grad.return_value = np.zeros((5, 2))
        mf.canonicalize.return_value = (energies, np.eye(7))
        mf.make_rdm1.return_value = density
        mf.get_fock.return_value = np.diag(energies)
        mf.get_hcore.return_value = -7.5 * np.eye(7)
        pt = Mock()
        pt.kernel.return_value = (0., np.zeros((5, 5, 2, 2)))
        pt.make_rdm1.return_value = density
        pt.make_rdm2.return_value = pipeline.disconnected_rhf(density)
        mol = Mock(nelectron=10)
        mol.intor.return_value = np.zeros((7,) * 4)
        mol.energy_nuc.return_value = 0.
        with patch("pyscf.scf.RHF", return_value=mf), patch("pyscf.mp.mp2.RMP2", return_value=pt) as constructor:
            arrays, diagnostics = pipeline.calculate_rhf_rmp2(self.record, mol)
        constructor.assert_called_once_with(mf, frozen=0)
        self.assertEqual(arrays["mp2_t2"].shape, (5, 5, 2, 2))
        self.assertTrue(diagnostics["rhf_converged"])

    def test_production_rdm_correlates_former_core_before_final_valence_selection(self):
        mol = pipeline.make_molecule(self.record)
        metadata = pipeline.select_valence_aos(mol)

        def excite_former_core(record, molecule):
            arrays, convergence = self.synthetic_electronic_arrays(record, molecule)
            t2 = arrays["mp2_t2"]
            t2[0, 0, 0, 0] = 0.1
            fake_mp = SimpleNamespace(t2=t2, nmo=7, nocc=5, frozen=None,
                                      mo_occ=arrays["mo_occ"], mo_coeff=arrays["mo_coeff"])
            density, gamma = mp2.make_rdm1(fake_mp), mp2.make_rdm2(fake_mp)
            arrays.update(P_MP2_AO=density, Gamma_MP2_AO=gamma,
                          P_MP2_MO_pyscf=density.copy(), Gamma_MP2_MO_pyscf=gamma.copy())
            # Synthetic H has no two-electron term. Its retained correlation
            # energy is the orbital-energy contraction with the quadratic 1-RDM.
            arrays["E_MP2_correlation"] = np.asarray(0.1)
            return arrays, convergence

        with patch.object(pipeline, "calculate_rhf_rmp2", side_effect=excite_former_core):
            arrays, diagnostics = pipeline.calculate_record(self.record, metadata)
        self.assertTrue(diagnostics["passed"], diagnostics["failed_checks"])
        self.assertEqual(int(arrays["frozen_core"]), 0)
        self.assertEqual(int(arrays["mp2_frozen_core"]), 0)
        self.assertEqual(int(arrays["input_fci_frozen_core"]), 1)
        self.assertAlmostEqual(arrays["P_MP2_MO_consistent"][0, 0], 1.98, places=12)
        self.assertGreater(np.max(np.abs(arrays["Gamma_MP2_MO_consistent"][0] - arrays["Gamma_MO_order0"][0])), 0.01)
        self.assertEqual(arrays["Gamma_MP2_AO"].shape, (7,) * 4)
        self.assertEqual(arrays["Lambda_L"].shape, (7,) * 4)
        self.assertEqual(arrays["P_L"].shape, (7, 7))
        self.assertEqual(arrays["P_mu"].shape, (6,))
        np.testing.assert_array_equal(arrays["core_indices"], [0])
        self.assertNotIn(0, arrays["valence_indices"])


if __name__ == "__main__":
    unittest.main()
