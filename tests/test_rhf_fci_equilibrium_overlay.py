"""Check actual plotted data and scientific reference conventions without calculations."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'qml-rhf-overlay-matplotlib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'codes'))
from plots.plot_rhf_fci_equilibrium_overlay import (
    PROJECT, load_plot_data, load_data, equilibrium_points, load_references, combine)
from plots.plot_rhf_bond_scan_30 import local_panel


class EquilibriumOverlayTests(unittest.TestCase):
    def setUp(self):
        self.results = load_plot_data(PROJECT/'results/rhf_30_point_scans/rhf_scan_30.json',
                                      PROJECT/'results/bond_length_part3/vibrational_levels.json')
        self.groups = load_data(PROJECT/'results/rhf_reference_correlation')
        self.points = equilibrium_points(self.results, PROJECT/'json/new_rhf_harmonic_bond_ranges.json')
        self.cache = PROJECT/'results/rhf_reference_correlation/fci_at_rhf_equilibrium.json'
        self.refs = load_references(self.cache, self.points)

    def test_all_plotted_values_and_reference_zeros(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from matplotlib.collections import LineCollection
        original = copy.deepcopy((self.results, self.groups, self.refs))
        combined, table = combine(self.results, self.groups, self.refs)
        self.assertEqual((self.results, self.groups, self.refs), original)
        self.assertEqual(len(table), 9)
        for result, saved in zip(combined, self.results):
            self.assertEqual(result['summary'], saved['summary'])
            s = result['summary']; name = s['molecule']
            fig, ax = plt.subplots()
            try:
                local_panel(ax, result, MaxNLocator, relative=True)
                rhf, fci = ax.lines[:2]
                self.assertNotIn("Local RHF harmonic model", ax.get_legend_handles_labels()[1])
                self.assertNotIn("Common", ax.get_xlabel())
                self.assertEqual(ax.get_title(), "")
                for line, field, ref in ((rhf, 'E_RHF_Ha', s['E_re_Ha']),
                    (fci, 'E_FCI_frozen_core_Ha', s['E_re_Ha'])):
                    at_re = 0.0 if field == 'E_RHF_Ha' else 1000*(self.refs[name]['E_FCI_frozen_core_Ha']-s['E_re_Ha'])
                    expected = sorted([(p['bond_length_A'], 1000*(p[field]-ref)) for p in result['points']]
                                      + [(s['r_e_A'], at_re)])
                    self.assertEqual(list(zip(line.get_xdata(), line.get_ydata())), expected)
                    self.assertEqual(len(expected), 31)
                    self.assertEqual(dict(expected)[s['r_e_A']], at_re)
                levels = [c for c in ax.collections if isinstance(c, LineCollection)]
                self.assertEqual(len(levels), 0)
                self.assertNotIn("n=0", [text.get_text() for text in ax.texts])
                self.assertNotIn("n=1", [text.get_text() for text in ax.texts])
                self.assertFalse(any("[" in text.get_text() for text in ax.texts))
                for n, level in enumerate(levels):
                    self.assertEqual(list(level.get_segments()[0][:,1]), [1000*s[f'E_n{n}_Ha']]*2)
                self.assertLess(min(fci.get_ydata()), 0.0)
                for point, row in zip(result['points'], self.groups[name]):
                    self.assertEqual(point['E_RHF_Ha'], row['E_RHF_Ha'])
                    self.assertEqual(point['E_FCI_frozen_core_Ha'], row['E_FCI_frozen_core_Ha'])
                    self.assertEqual(point['bond_length_A'], row['r'])
            finally:
                plt.close(fig)

    def test_rejects_nearby_geometry_and_wrong_frozen_core(self):
        for key in ('cartesian_A', 'n_frozen_orbitals'):
            data = json.loads(self.cache.read_text())
            if key == 'cartesian_A':
                data['records'][0][key][0][0] += 1e-8
            else:
                data['records'][0][key] += 1
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'cache.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_references(path, self.points)

    def test_boundary_minimum_is_flagged_without_changing_reference(self):
        groups = copy.deepcopy(self.groups)
        groups['LiH'][0]['E_FCI_frozen_core_Ha'] -= 1.0
        _, table = combine(self.results, groups, self.refs)
        row = next(r for r in table if r['molecule'] == 'LiH')
        self.assertEqual(row['FCI_min_location'], 'left_boundary')
        self.assertEqual(row['E_FCI_at_Re_Ha'], self.refs['LiH']['E_FCI_frozen_core_Ha'])
        self.assertLess(row['FCI_sampled_min_relative_to_RHF_Re_mHa'], 0)


if __name__ == '__main__':
    unittest.main()
