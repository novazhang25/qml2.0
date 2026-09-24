"""Exact, solver-free H2S loader regression against saved Ne-core descriptors."""
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import circuit
import run1

LABELS = ('0 S 3s', '0 S 3px', '0 S 3py', '0 S 3pz', '1 H 1s', '2 H 1s')
RETAINED = (2, 6, 7, 8, 9, 10)
CORE = (0, 1, 3, 4, 5)
REFERENCE = ROOT / 'results/audits/fe_all_molecules/fe_eigenvalues.csv'


class H2SValenceLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = sorted((ROOT / 'results/descriptor/H2S').glob('*.npz'))
        cls.hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*cls.paths, REFERENCE]}
        cls.rows = []
        for path in cls.paths:
            with np.load(path, allow_pickle=False) as saved:
                raw = {k: saved[k] for k in saved.files}
            sample, metadata = run1.load_record(path, 'H2S', path.stem)
            cls.rows.append((path, raw, sample, metadata))
        cls.saved_fe = {}
        with REFERENCE.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['molecule'] == 'H2S' and row['view'] == 'saved_6AO_diagnostic':
                    key = (row['geometry_id'], int(row['column_index']))
                    if key in cls.saved_fe:
                        raise AssertionError(f'Duplicate saved FE entry: {key}')
                    cls.saved_fe[key] = float(row['eigenvalue'])

    def test_all_thirty_saved_blocks_and_labels_are_reproduced_exactly(self):
        self.assertEqual([p.stem for p in self.paths], list(run1.GEOMETRY_IDS))
        self.assertEqual(circuit.feature_width('H2S', 'FE'), 6)
        for path, raw, sample, metadata in self.rows:
            with self.subTest(geometry=path.stem):
                labels = tuple(' '.join(str(label).split()) for label in raw['ao_labels'])
                indices_from_labels = tuple(labels.index(label) for label in LABELS)
                self.assertEqual(indices_from_labels, RETAINED)
                np.testing.assert_array_equal(raw['valence_indices'], indices_from_labels)
                np.testing.assert_array_equal(raw['core_indices'], CORE)
                self.assertEqual(sample.active_ao_labels, LABELS)
                self.assertEqual(metadata['source_selected_ao_indices'], list(RETAINED))
                self.assertEqual(raw['P_L'].shape, (11, 11))
                self.assertEqual(sample.pij.shape, (6, 6))
                self.assertEqual(sample.fij.shape, (6, 6))
                block = raw['P_L'][np.ix_(indices_from_labels, indices_from_labels)]
                np.testing.assert_array_equal(sample.pij, block)
                np.testing.assert_array_equal(sample.pii, raw['P_mu'])
                np.testing.assert_array_equal(sample.fii, raw['F_mu'])
                self.assertTrue(np.isrealobj(sample.pij) and np.isfinite(sample.pij).all())

    def test_actual_fe_path_matches_all_thirty_preexisting_saved_fingerprints_exactly(self):
        expected_keys = {(geometry_id, column) for geometry_id in run1.GEOMETRY_IDS for column in range(6)}
        self.assertEqual(set(self.saved_fe), expected_keys)
        with contextlib.redirect_stdout(io.StringIO()):
            spectra = run1.raw_spectra([row[2] for row in self.rows])
        for path, raw, sample, _ in self.rows:
            with self.subTest(geometry=path.stem):
                expected = np.array([self.saved_fe[path.stem, column] for column in range(6)])
                actual = spectra[sample.sample_id]['FE']
                self.assertEqual(actual.shape, (6,))
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual, np.linalg.eigvalsh(sample.pij))
                self.assertTrue(np.isrealobj(actual) and np.isfinite(actual).all())

    def test_full_ao_lowdin_validation_precedes_h2s_selection(self):
        events = []
        original_check, original_select = run1.check_record_array, run1.ao_selection

        def check(actual, expected, path, key, *args, **kwargs):
            events.append(key)
            if key.startswith('full-AO P_L') or key.startswith('full-AO F_L'):
                self.assertEqual(actual.shape, (11, 11))
                self.assertEqual(expected.shape, (11, 11))
            if key.startswith('descriptor.transform_rank4:'):
                self.assertEqual(actual.shape, (11,) * 4)
                self.assertEqual(expected.shape, (11,) * 4)
            return original_check(actual, expected, path, key, *args, **kwargs)

        def select(*args, **kwargs):
            events.append('SELECT')
            return original_select(*args, **kwargs)

        with patch.object(run1, 'check_record_array', side_effect=check), \
             patch.object(run1, 'ao_selection', side_effect=select):
            run1.load_record(self.paths[0], 'H2S', '001')
        for key in ('full-AO P_L = S_half @ P_AO @ S_half',
                    'full-AO F_L = S_minus_half @ F_AO @ S_minus_half',
                    'descriptor.transform_rank4: full-AO Lowdin Lambda transformation'):
            self.assertLess(events.index(key), events.index('SELECT'))

    def test_ten_ao_selection_is_not_silently_accepted_as_saved_ne_core_space(self):
        path, raw, _, _ = self.rows[0]
        altered = dict(raw, valence_indices=np.arange(1, 11), core_indices=np.array([0]),
                       valence_labels=raw['ao_labels'][1:])
        with self.assertRaisesRegex(ValueError, 'saved selection does not match'):
            run1.ao_selection(altered, 'H2S', path)

    @classmethod
    def tearDownClass(cls):
        for path, before in cls.hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise AssertionError(f'Saved input was modified: {path}')


if __name__ == '__main__':
    unittest.main()
