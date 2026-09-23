"""Regression checks for the Part 3 literal-zero plot reference."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from plots import bond_length_part3_plots as plots


def fixture(equilibrium=-1.0):
    summary = {
        "molecule": "LiH", "basis": "sto-3g", "coordinate": "q = r(Li-H) - r_e(Li-H)",
        "r_e_A": 1.5, "E_re_Ha": equilibrium, "k_Ha_per_A2": 2.0,
        "hbar_omega_Ha": 0.02, "reference_shift_Ha": 0.0,
        "energy_reference_status": "UNVALIDATED_ZERO_REFERENCE",
        "energy_reference_warning": plots.REFERENCE_WARNING,
    }
    for n in (0, 1):
        excitation = (n + 0.5) * summary["hbar_omega_Ha"]
        amplitude = math.sqrt(2 * excitation / summary["k_Ha_per_A2"])
        summary.update({f"E_n{n}_Ha": excitation, f"E_total_n{n}_Ha": equilibrium + excitation,
                        f"n{n}_pass": equilibrium + excitation < 0.0,
                        f"Delta_q_n{n}_A": amplitude,
                        f"n{n}_min_A": summary["r_e_A"] - amplitude,
                        f"n{n}_max_A": summary["r_e_A"] + amplitude})
    selected = 1 if summary["n1_pass"] else 0 if summary["n0_pass"] else None
    summary["selected_n"] = selected
    for side in ("min", "max"):
        summary[f"selected_range_{side}_A"] = None if selected is None else summary[f"n{selected}_{side}_A"]
    return {"summary": summary, "input_validation": {"passed": True}}


class AbsoluteReferenceTests(unittest.TestCase):
    def test_no_scan_is_required_and_zero_equality_fails(self):
        summary, values = plots._validated_fields(fixture(equilibrium=-0.01))
        self.assertEqual(values["E_total_n0_Ha"], 0.0)
        self.assertFalse(summary["n0_pass"])
        self.assertFalse(summary["n1_pass"])
        self.assertIsNone(summary["selected_n"])

    def test_rejects_shifted_reference(self):
        record = fixture()
        record["summary"]["reference_shift_Ha"] = 1.0
        with self.assertRaisesRegex(ValueError, "unshifted absolute"):
            plots._validated_fields(record)

    def test_rejects_wrong_absolute_sum_or_pass_flag(self):
        for field, value, error in (("E_total_n0_Ha", 0.01, "not E_re"),
                                    ("n0_pass", False, "pass flag")):
            with self.subTest(field=field):
                record = fixture()
                record["summary"][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    plots._validated_fields(record)

    def test_rejects_silenced_reference_warning(self):
        record = fixture()
        record["summary"]["energy_reference_warning"] = ""
        with self.assertRaisesRegex(ValueError, "warning is required"):
            plots._validated_fields(record)

    def test_saved_svg_has_literal_zero_and_reference_warning(self):
        original = fixture()
        retained = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as folder:
            paths = plots.write_plots([original], folder)
            self.assertEqual(paths, ["plots/LiH_vibrational_levels.png", "plots/LiH_vibrational_levels.svg"])
            svg = (Path(folder) / paths[1]).read_text()
            self.assertIn("Literal zero: 0 Ha", svg)
            self.assertIn("UNVALIDATED ZERO REFERENCE", svg)
            self.assertIn("including nuclear repulsion", svg)
            self.assertIn("passes &lt;0? YES", svg)
            self.assertNotIn("E_diss", svg)
            self.assertNotIn("D_e", svg)
        self.assertEqual(original, retained)


if __name__ == "__main__":
    unittest.main()
