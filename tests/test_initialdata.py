"""Small real LiH integration test and checkpoint safety checks."""
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pyscf import fci

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
import initialdata as pipeline


class InitialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.output = Path(cls.workspace.name) / "integration"
        with contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.main(["--molecules", "LiH", "--output-dir", str(cls.output)])
        if code:
            raise AssertionError((cls.output / "initialdata.json").read_text())
        cls.document = json.loads((cls.output / "initialdata.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_real_lih_equilibrium_hessian_and_n1_range(self):
        self.assertEqual(self.document["status"], "PASS")
        result = self.document["results"][0]
        summary = result["summary"]
        self.assertAlmostEqual(summary["r_e_A"], 1.510811853032, delta=2e-6)
        self.assertAlmostEqual(summary["E_RHF_re_Ha"], -7.86338212892111, delta=1e-10)
        self.assertAlmostEqual(summary["k_Ha_per_Bohr2"], 0.11621039750120737, delta=2e-6)
        self.assertTrue(result["harmonic"]["finite_differences"]["convergence_pass"])
        self.assertEqual(np.asarray(result["harmonic"]["cartesian_hessian_Eh_per_Bohr2"]).shape, (2, 2, 3, 3))
        self.assertAlmostEqual(summary["n1_min_A"], 1.262772864212423, delta=2e-6)
        self.assertAlmostEqual(summary["n1_max_A"], 1.7588508418515771, delta=2e-6)
        self.assertAlmostEqual(0.5 * summary["k_Ha_per_A2"] * summary["Delta_q_n1_A"]**2,
                               summary["E_n1_Ha"], places=13)
        self.assertEqual(summary["n"], 1)

    def test_all_electronic_outputs_and_correlation_saved(self):
        result = self.document["results"][0]
        self.assertEqual(len(result["scan"]), 30)
        for point in [result["equilibrium_energies"], *result["scan"]]:
            self.assertEqual(point["status"], "PASS")
            self.assertTrue(point["rhf_diagnostics"]["internal_stable"])
            self.assertEqual(point["n_frozen_orbitals"], 1)
            self.assertLess(point["E_corr_Ha"], 0)
            self.assertEqual(point["E_corr_Ha"], point["E_FCI_Ha"] - point["E_RHF_Ha"])
            self.assertAlmostEqual(point["E_FCI_Ha"], point["E_CAS_active_Ha"] + point["E_core_including_nuclear_Ha"], places=12)
            with np.load(self.output / point["electronic_state_file"]["path"], allow_pickle=False) as arrays:
                self.assertAlmostEqual(np.linalg.norm(arrays["ci_vector"]), 1., places=10)
                self.assertAlmostEqual(float(arrays["E_corr_Ha"]), point["E_corr_Ha"], places=13)
                np.testing.assert_allclose(arrays["mo_coeff"].T @ arrays["overlap_ao"] @ arrays["mo_coeff"],
                                           np.eye(6), rtol=0, atol=1e-10)
                np.testing.assert_allclose(arrays["cartesian_A"], point["cartesian_A"], rtol=0, atol=1e-14)
                reconstructed = fci.direct_spin1.energy(arrays["active_h1_Ha"], arrays["active_h2_packed_Ha"],
                                                        arrays["ci_vector"], 5, 2) + float(arrays["core_energy_Ha"])
                self.assertAlmostEqual(reconstructed, point["E_FCI_Ha"], places=11)
            individual = self.output / point["geometry_file"]["path"]
            individual = json.loads(individual.with_name("point.json").read_text())
            if point["point_index"] > 0:
                self.assertEqual(individual["geometry_validation"], point["geometry_validation"])
        for name in ("initialdata.json", "summary.csv", "scan.csv", "equilibrium_energies.csv", "source_initialdata.py", "LiH/result.json"):
            self.assertTrue((self.output / name).is_file(), name)
        self.assertEqual(len((self.output / "scan.csv").read_text().splitlines()), 31)

    def test_production_defaults_have_exactly_thirty_points(self):
        args = pipeline.parse_args([])
        self.assertEqual(args.molecules, list(pipeline.MOLECULES))
        summary = self.document["results"][0]["summary"]
        grid = pipeline.scan_grid(summary, 30)
        self.assertEqual(len(grid), 30)
        self.assertEqual(grid[0], summary["n1_min_A"])
        self.assertEqual(grid[-1], summary["n1_max_A"])
        self.assertFalse(np.any(np.isclose(grid, summary["r_e_A"], atol=1e-12, rtol=0)))

    def test_resume_does_not_recalculate_completed_results(self):
        state_files = list(self.output.rglob("*.npz")) + list(self.output.rglob("*.xyz"))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in state_files}
        with patch.object(pipeline, "optimize_equilibrium", side_effect=AssertionError("No optimization")), \
             patch.object(pipeline, "calculate_harmonic", side_effect=AssertionError("No Hessian")), \
             patch.object(pipeline, "solve_point", side_effect=AssertionError("No electronic calculation")), \
             contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.main(["--molecules", "LiH", "--resume", "--output-dir", str(self.output)])
        self.assertEqual(code, 0)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in state_files})

    def test_changed_wavefunction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "copy"
            shutil.copytree(self.output, output)
            point = self.document["results"][0]["scan"][0]
            (output / point["electronic_state_file"]["path"]).write_bytes(b"damaged checkpoint")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = pipeline.main(["--molecules", "LiH", "--resume", "--output-dir", str(output)])
            self.assertEqual(code, 1)
            saved = json.loads((output / "initialdata.json").read_text())
            self.assertEqual(saved["status"], "FAIL")
            self.assertIn("Missing or changed electronic_state_file", saved["results"][0]["error"])

    def test_recovers_point_saved_before_manifest_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "copy"
            shutil.copytree(self.output, output)
            manifest = output / "initialdata.json"
            document = json.loads(manifest.read_text())
            document["results"][0]["scan"].pop()
            manifest.write_text(json.dumps(document))
            with patch.object(pipeline, "solve_point", side_effect=AssertionError("Saved point must be recovered")), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = pipeline.main(["--molecules", "LiH", "--resume", "--output-dir", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(len(json.loads(manifest.read_text())["results"][0]["scan"]), 30)

    def test_failed_fci_preserves_rhf_and_geometry(self):
        point = self.document["results"][0]["scan"][0]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(pipeline, "frozen_core_fci", side_effect=RuntimeError("FCI test failure")):
            output = Path(temp)
            saved = pipeline.calculate_point("LiH", point["symbols"], np.asarray(point["cartesian_A"]),
                                              point["bond_length_A"], point["q_A"], 1, output / "point", output)
            self.assertEqual(saved["status"], "FAIL")
            self.assertIn("E_RHF_Ha", saved)
            self.assertIn("FCI test failure", saved["error"])
            self.assertTrue((output / "point/geometry.xyz").exists())
            self.assertEqual(saved["electronic_state_stage"], "RHF")
            with np.load(output / saved["electronic_state_file"]["path"], allow_pickle=False) as arrays:
                self.assertIn("mo_coeff", arrays.files)
                self.assertEqual(float(arrays["E_RHF_Ha"]), saved["E_RHF_Ha"])
            self.assertEqual(json.loads((output / "point/point.json").read_text()), saved)


if __name__ == "__main__":
    unittest.main()
