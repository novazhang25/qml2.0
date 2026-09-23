"""Verify Part 3 metadata and plot coordinates without rendering or running RHF."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from plots.plot_rhf_bond_scan_30 import load_plot_data, local_panel
from rhf_bond_scan_30 import rebind_saved_scan, sha256


class OverlayInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scan_path = Path(self.directory.name) / "rhf_scan_30.json"
        self.levels_path = Path(self.directory.name) / "vibrational_levels.json"
        self.scan_path.write_bytes((ROOT / "results/rhf_30_point_scans/rhf_scan_30.json").read_bytes())
        self.levels_path.write_bytes((ROOT / "results/bond_length_part3/vibrational_levels.json").read_bytes())
        rebind_saved_scan(self.scan_path, self.levels_path, ROOT / "json/new_rhf_harmonic_bond_ranges.json")

    def test_all_270_rhf_points_and_harmonic_levels_plot_absolute_energies(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.ticker import MaxNLocator

        scan = json.loads(self.scan_path.read_text())
        results = load_plot_data(self.scan_path, self.levels_path)
        self.assertEqual(len(results), 9)
        self.assertEqual(sum(len(result["points"]) for result in results), 270)
        for saved, plotted in zip(scan["results"], results):
            for original, point in zip(saved["points"], plotted["points"]):
                self.assertEqual(point["bond_length_A"], original["bond_length_A"])
                self.assertEqual(point["E_RHF_Ha"], original["E_RHF_Ha"])
                self.assertEqual(point["E_RHF_minus_E_re_Ha"], original["E_RHF_Ha"] - original["E_re_Ha"])
            self.assertEqual(plotted["summary"]["decision_criterion"], "E_total_n < 0 Ha")
            fig, ax = plt.subplots()
            try:
                local_panel(ax, plotted, MaxNLocator)
                harmonic, rhf = ax.lines[:2]
                expected_curve = sorted([(p["bond_length_A"], p["E_RHF_Ha"]) for p in saved["points"]]
                                        + [(plotted["summary"]["r_e_A"], plotted["summary"]["E_re_Ha"])])
                self.assertEqual(list(zip(rhf.get_xdata(), rhf.get_ydata())), expected_curve)
                self.assertEqual(harmonic.get_ydata()[250], plotted["summary"]["E_re_Ha"])
                self.assertNotIn("Saved RHF equilibrium", ax.get_legend_handles_labels()[1])
                self.assertEqual(rhf.get_marker(), "o")
                annotations = "\n".join(text.get_text() for text in ax.texts)
                self.assertNotIn("$r_e$ =", annotations)
                self.assertNotIn(r"$E_{\mathrm{re}}$ =", annotations)
                self.assertEqual(len(rhf.get_xdata()), 31)
                level_segments = [collection for collection in ax.collections
                                  if isinstance(collection, LineCollection)]
                self.assertEqual(len(level_segments), 2)
                for n, segment in enumerate(level_segments):
                    expected = plotted["summary"]["E_total_n%d_Ha" % n]
                    self.assertEqual(list(segment.get_segments()[0][:, 1]), [expected, expected])
                self.assertEqual(ax.get_ylabel(), r"$E$ (Hartree)")
                self.assertFalse(ax.yaxis.get_major_formatter().get_useOffset())
                self.assertLess(ax.get_ylim()[1], 0)  # Keep the local absolute-energy zoom.
            finally:
                plt.close(fig)

    def test_wrong_zero_flag_or_shift_is_rejected_even_when_scan_and_levels_agree(self):
        original_scan = json.loads(self.scan_path.read_text())
        original_levels = json.loads(self.levels_path.read_text())
        for key, value in (("n1_pass", False), ("reference_shift_Ha", 1.0),
                           ("energy_reference_warning", "")):
            with self.subTest(key=key):
                scan, levels = copy.deepcopy(original_scan), copy.deepcopy(original_levels)
                scan["results"][0]["summary"][key] = value
                levels["results"][0]["summary"][key] = value
                self.levels_path.write_text(json.dumps(levels))
                scan["provenance"]["ranges_sha256"] = sha256(self.levels_path)
                self.scan_path.write_text(json.dumps(scan))
                with self.assertRaises(ValueError):
                    load_plot_data(self.scan_path, self.levels_path)


if __name__ == "__main__":
    unittest.main()
