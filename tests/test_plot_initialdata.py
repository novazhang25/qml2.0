"""Small synthetic checks of standalone initialdata plotting and validation."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qml-initialdata-test-matplotlib"))
from plot_initialdata import load_plot_data, plot_panel, validate_document, write_plots


def sample_document():
    summary = {"molecule": "LiH", "basis": "sto-3g", "r_e_A": 1.5, "E_RHF_re_Ha": -7.9,
               "k_Ha_per_Bohr2": 0.168, "k_Ha_per_A2": 0.6, "mu_eff_amu": 0.875,
               "omega_rad_per_s": 1e14, "hbar_omega_Ha": 0.008, "E_n1_Ha": 0.012,
               "E_total_n1_Ha": -7.888, "n1_min_A": 1.3, "n1_max_A": 1.7}
    scan = []
    for index, distance in enumerate((1.3, 1.5, 1.7), 1):
        displacement = distance - 1.5
        rhf = -7.9 + 0.3 * displacement ** 2
        fci = rhf - 0.04 - 0.005 * displacement
        scan.append({"point_index": index, "bond_length_A": distance, "q_A": displacement,
                     "E_RHF_Ha": rhf, "E_FCI_Ha": fci, "E_corr_Ha": fci - rhf,
                     "E_corr_mHa": 1000 * (fci - rhf), "status": "PASS",
                     "cartesian_A": [[0, 0, 0], [0, 0, distance]]})
    return {"schema_version": "initialdata-v1", "status": "PASS",
            "requested_molecules": ["LiH"],
            "settings": {"basis": "sto-3g", "scan_points": 3, "fci_convention": "frozen-core"},
            "results": [{"molecule": "LiH", "status": "PASS", "summary": summary,
                         "equilibrium_energies": {"E_RHF_Ha": -7.9, "E_FCI_Ha": -7.94,
                                                  "E_corr_Ha": -0.04}, "scan": scan}]}


class InitialDataPlotTests(unittest.TestCase):
    def test_invalid_saved_data_cannot_be_silently_plotted(self):
        base = sample_document()
        validate_document(base)
        changes = (
            lambda d: d.update(status="RUNNING"),
            lambda d: d["results"].append(copy.deepcopy(d["results"][0])),
            lambda d: d["results"][0]["scan"].pop(),
            lambda d: d["results"][0]["scan"][0].update(point_index=2),
            lambda d: d["results"][0]["scan"][0].update(bond_length_A=1.31),
            lambda d: d["results"][0]["scan"][0].update(E_corr_Ha=0.04),
            lambda d: d["results"][0]["scan"][0].update(E_FCI_Ha=float("nan")),
            lambda d: d["results"][0]["scan"][0].update(status="FAIL"),
            lambda d: d["results"][0]["equilibrium_energies"].update(E_corr_Ha=0.04),
            lambda d: d.update(requested_molecules=["LiH", "HF"]),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                document = copy.deepcopy(base)
                change(document)
                with self.assertRaises(ValueError):
                    validate_document(document)

    def test_plots_use_saved_bond_distances_and_rhf_equilibrium_reference(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = sample_document()["results"][0]
        for absolute in (False, True):
            with self.subTest(absolute=absolute):
                figure, ax = plt.subplots()
                try:
                    plot_panel(ax, result, absolute=absolute)
                    reference = 0 if absolute else result["summary"]["E_RHF_re_Ha"]
                    factor = 1 if absolute else 1000
                    for line, field in zip(ax.lines[:2], ("E_RHF_Ha", "E_FCI_Ha")):
                        curve = sorted([(p["bond_length_A"], p[field]) for p in result["scan"]]
                                       + [(1.5, result["equilibrium_energies"][field])])
                        self.assertEqual(list(line.get_xdata()), [x for x, energy in curve])
                        self.assertEqual(list(line.get_ydata()),
                                         [factor * (energy - reference) for x, energy in curve])
                    self.assertEqual(ax.lines[1].get_label(), "CASCI")
                    self.assertEqual(ax.get_xlabel(), "Li-H distance (Å)")
                    self.assertEqual([text.get_text() for text in ax.texts], ["LiH"])
                    self.assertEqual(len(ax.collections), 0)
                finally:
                    plt.close(figure)

    def test_json_loading_and_png_svg_table_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initialdata.json"
            path.write_text(json.dumps(sample_document()), encoding="utf-8")
            original = path.read_bytes()
            files = write_plots(load_plot_data(path), Path(directory) / "plots")
            self.assertEqual({file.name for file in files},
                             {"LiH.png", "LiH.svg", "overview.png", "overview.svg", "plotted_points.csv"})
            self.assertTrue(all(file.stat().st_size > 50 for file in files))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
