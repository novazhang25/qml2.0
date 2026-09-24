"""Focused solver-free tests for the saved-cumulant T-only migration."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"codes"))
import descriptor
import descriptor as repair


def synthetic_arrays():
    """Mixture of two four-electron determinants: valid RDMs, nontrivial raw axes."""
    n = 4
    P = np.diag([2., 2., 0., 0.])
    Q, _ = np.linalg.qr(np.random.default_rng(421).normal(size=(n, n)))
    other = Q@P@Q.T
    D = .9*P+.1*other
    gamma0 = descriptor.disconnected_rhf(P)
    G = .9*gamma0+.1*descriptor.disconnected_rhf(other)
    L = G-gamma0
    F = np.diag(np.arange(n, dtype=float))
    values = {
        "S_AO": np.eye(n), "P_AO": P, "F_AO": F, "S_half": np.eye(n),
        "S_minus_half": np.eye(n), "mo_coeff": np.eye(n), "P_L": P, "F_L": F,
        "P_MP2_MO_consistent": D, "P_MP2_AO_consistent": D,
        "Gamma_MP2_MO_consistent": G, "Gamma_MP2_AO_consistent": G,
        "Gamma0_HF_AO": gamma0, "Lambda_AO": L, "Lambda_L": L,
        "valence_indices": np.arange(n), "electron_count": 4,
        "frozen_core": 0, "mp2_frozen_core": 0, "rdm_frozen_core": 0,
        "molecule": "synthetic", "geometry_id": "001", "q_A": 0.,
        "arbitrary_metadata": np.array(["preserve raw NPY header and data"]),
    }
    values.update(descriptor.extract_descriptors(P, F, L, np.arange(n)))
    old = np.diagonal(L, axis1=2, axis2=3).copy()
    triples = values["triple_indices"]
    values["T_full"] = old
    values["T_munulambda"] = old[triples[:, 0], triples[:, 1], triples[:, 2]]
    return {key: np.asarray(value) for key, value in values.items()}


def write_source(folder, arrays=None):
    source = Path(folder)/"source"
    (source/"synthetic").mkdir(parents=True)
    path = source/"synthetic/001.npz"
    np.savez_compressed(path, **(synthetic_arrays() if arrays is None else arrays))
    row = {"molecule": "synthetic", "geometry_id": "001", "q_A": 0.,
           "record": str(path), "error": ""}
    manifest = {"schema_version": "descriptor-v1-calculation-only", "records": [row],
                "completed_geometry_count": 1, "failed_geometry_count": 0}
    (source/"manifest.json").write_text(json.dumps(manifest))
    path.with_suffix(".json").write_text(json.dumps({**row, "validation_performed": False}))
    return source


class RepairTDescriptorsTests(unittest.TestCase):
    def test_merged_cli_dispatches_repair_without_calculation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = write_source(folder)
            output = Path(folder)/"merged"
            with patch.object(descriptor, "run", side_effect=AssertionError("No calculation mode")), \
                 patch.object(descriptor, "build_consistent_rdms", side_effect=AssertionError("No RDM generation")), \
                 contextlib.redirect_stdout(io.StringIO()) as stream:
                code = descriptor.main(["--repair-t", "--source-dir", str(source),
                                        "--output-dir", str(output), "--expected-count", "1"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stream.getvalue())["geometries_regenerated"], 1)
            self.assertTrue((output/"manifest.json").exists())

    def test_merged_cli_defaults_and_audit_only(self):
        self.assertEqual(descriptor.parse_args([]).input_dir, descriptor.PROJECT/"results/initialdata")
        args = descriptor.parse_args(["--repair-t"])
        self.assertEqual(args.input_dir, descriptor.PROJECT/"results/descriptor")
        self.assertEqual(args.expected_count, 270)
        with tempfile.TemporaryDirectory() as folder:
            source = write_source(folder)
            output = Path(folder)/"audit"
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = descriptor.main(["--repair-t", "--input-dir", str(source),
                                        "--output-dir", str(output), "--expected-count", "1", "--audit-only"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stream.getvalue())["status"], "AUDIT_PASS")
            self.assertFalse(output.exists())

    def test_merged_cli_rejects_mixed_modes(self):
        for args in (["--audit-only"], ["--expected-count", "1"],
                     ["--repair-t", "--resume"], ["--repair-t", "--smoke-test"],
                     ["--repair-t", "--dry-run"], ["--repair-t", "--molecules", "H2O"],
                     ["--repair-t", "--expected-count", "0"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    descriptor.parse_args(args)
                self.assertEqual(error.exception.code, 2)

    def test_exact_mapping_selection_and_nontrivial_old_difference(self):
        arrays = synthetic_arrays()
        replacements, evidence = repair.audit_record(arrays)
        raw = arrays["Lambda_L"]
        expected = np.empty((4,)*3)
        for i, j, k in np.ndindex(expected.shape):
            expected[i, j, k] = raw[i, k, j, k]
        np.testing.assert_array_equal(replacements["T_full"], expected)
        triples = arrays["triple_indices"]
        np.testing.assert_array_equal(replacements["T_munulambda"],
                                      [expected[i, j, k] for i, j, k in triples])
        self.assertEqual(evidence["residuals"]["T_vectorized_vs_literal_loop"], 0.)
        self.assertGreater(evidence["differences"]["T_full"]["max_abs_difference"], .01)
        self.assertGreater(evidence["differences"]["T_munulambda"]["max_abs_difference"], 1e-5)

    def test_old_producer_mapping_is_rejected(self):
        arrays = synthetic_arrays()
        original = descriptor.extract_descriptors

        def wrong_mapping(P, F, L, v):
            result = original(P, F, L, v)
            raw = L[np.ix_(v, v, v, v)]
            result["T_full"] = np.diagonal(raw, axis1=2, axis2=3).copy()
            t = result["triple_indices"]
            result["T_munulambda"] = result["T_full"][t[:, 0], t[:, 1], t[:, 2]]
            return result

        with patch.object(descriptor, "extract_descriptors", side_effect=wrong_mapping):
            with self.assertRaisesRegex(ValueError, "T_vectorized_vs_literal_loop"):
                repair.audit_record(arrays)

    def test_full_tensor_missing_shape_nonfinite_and_complex_are_refused(self):
        for kind in ("missing", "shape", "nonfinite", "complex"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                arrays = synthetic_arrays()
                if kind == "missing":
                    del arrays["Lambda_L"]
                elif kind == "shape":
                    arrays["Lambda_L"] = arrays["Lambda_L"][:3, :3, :3, :3]
                elif kind == "nonfinite":
                    arrays["Lambda_L"][0, 0, 0, 0] = np.nan
                else:
                    arrays["Lambda_L"] = arrays["Lambda_L"].astype(complex)
                source = write_source(folder, arrays)
                output = Path(folder)/"new"
                with self.assertRaises(repair.AuditFailure) as failed:
                    repair.repair_dataset(source, output, expected_count=1)
                self.assertEqual(failed.exception.report["missing_usable_full_cumulant"], 1)
                self.assertFalse(output.exists())

    def test_bad_saved_cumulant_and_bad_selection_are_refused(self):
        for kind in ("subtraction", "transform", "contraction", "selection"):
            with self.subTest(kind=kind):
                arrays = synthetic_arrays()
                # Copies avoid the deliberate shared synthetic MO/AO arrays.
                if kind == "subtraction":
                    arrays["Lambda_AO"] = arrays["Lambda_AO"].copy()
                    arrays["Lambda_AO"][0, 0, 0, 0] += 1e-4
                elif kind == "transform":
                    arrays["Lambda_L"] = arrays["Lambda_L"].copy()
                    arrays["Lambda_L"][0, 0, 0, 0] += 1e-4
                elif kind == "contraction":
                    arrays["Gamma_MP2_MO_consistent"] = arrays["Gamma_MP2_MO_consistent"].copy()
                    arrays["Gamma_MP2_MO_consistent"][0, 0, 0, 0] += 1e-4
                else:
                    arrays["triple_indices"] = arrays["triple_indices"][::-1]
                with self.assertRaises(ValueError):
                    repair.audit_record(arrays)

    def test_migration_preserves_source_and_every_non_T_member(self):
        with tempfile.TemporaryDirectory() as folder:
            source = write_source(folder)
            output = Path(folder)/"version2"
            before = repair.snapshot(source)
            with patch.object(descriptor, "calculate_record", side_effect=AssertionError("No solver")), \
                 patch.object(descriptor, "build_consistent_rdms", side_effect=AssertionError("No RDM regeneration")):
                report = repair.repair_dataset(source, output, expected_count=1)
            self.assertEqual(before, repair.snapshot(source))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["geometries_regenerated"], 1)
            self.assertTrue(report["all_non_T_fields_unchanged"])
            old_path, new_path = source/"synthetic/001.npz", output/"synthetic/001.npz"
            with zipfile.ZipFile(old_path) as old, zipfile.ZipFile(new_path) as new:
                self.assertEqual(old.namelist(), new.namelist())
                for name in old.namelist():
                    if name not in ("T_full.npy", "T_munulambda.npy"):
                        self.assertEqual(old.read(name), new.read(name), name)
            with np.load(new_path, allow_pickle=False) as values:
                raw = values["Lambda_L"]
                for i, j, k in np.ndindex(values["T_full"].shape):
                    self.assertEqual(values["T_full"][i, j, k], raw[i, k, j, k])
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["t_descriptor_version"], 2)
            self.assertEqual(manifest["t_axis_mapping"], descriptor.T_AXIS_MAPPING)
            self.assertEqual(manifest["records"][0]["record"], str(new_path.resolve()))

    def test_existing_and_nested_outputs_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            source = write_source(folder)
            existing = Path(folder)/"existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                repair.repair_dataset(source, existing, expected_count=1)
            for output in (source, source/"new", source.parent):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    repair.repair_dataset(source, output, expected_count=1)

    def test_audit_only_never_creates_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = write_source(folder)
            output = Path(folder)/"new"
            report = repair.repair_dataset(source, output, expected_count=1, audit_only=True)
            self.assertEqual(report["status"], "AUDIT_PASS")
            self.assertEqual(report["geometries_regenerated"], 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
