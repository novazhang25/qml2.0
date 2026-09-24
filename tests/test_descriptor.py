"""Tests for descriptor calculation; saved-data repair has its own test suite."""
import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'codes'))
import descriptor as production
from test_methods41_consistent import independent_orders

ROOT = Path(__file__).resolve().parents[1]


class DescriptorTests(unittest.TestCase):
    def test_methods_t_mapping_preserves_exact_values_and_selection(self):
        # Nonsymmetric signed entries distinguish the Methods mapping from
        # the old raw-axis diagonal, even without physical tensor symmetries.
        density = np.arange(25.0).reshape(5, 5) - 20
        fock = -density / 7
        raw = np.arange(625.0).reshape((5,) * 4) - 500
        original = raw.copy()
        valence = [0, 2, 3, 4]
        actual = production.extract_descriptors(density, fock, raw, valence)
        expected = np.empty((4, 4, 4), dtype=raw.dtype)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    expected[i, j, k] = raw[valence[i], valence[k], valence[j], valence[k]]
        pairs = np.asarray(list(itertools.combinations(range(4), 2)))
        triples = np.asarray(list(itertools.combinations(range(4), 3)))
        np.testing.assert_array_equal(actual['T_full'], expected)
        np.testing.assert_array_equal(actual['T_munulambda'], [expected[i, j, k] for i, j, k in triples])
        np.testing.assert_array_equal(actual['pair_indices'], pairs)
        np.testing.assert_array_equal(actual['triple_indices'], triples)
        np.testing.assert_array_equal(actual['P_mu'], np.diag(density)[valence])
        np.testing.assert_array_equal(actual['F_mu'], np.diag(fock)[valence])
        np.testing.assert_array_equal(actual['P_munu'], [density[valence[i], valence[j]] for i, j in pairs])
        np.testing.assert_array_equal(raw, original)
        old = np.diagonal(raw[np.ix_(valence, valence, valence, valence)], axis1=2, axis2=3)
        self.assertFalse(np.array_equal(actual['T_full'], old))
        self.assertTrue(np.any(actual['T_munulambda'] < 0))

    def test_second_order_rdms_match_independent_fermionic_calculation(self):
        t = np.arange(16, dtype=float).reshape(2,2,2,2)/200
        t = (t+t.transpose(1,0,3,2))/2
        actual = production.build_consistent_rdms(t, 4, 4)
        expected, _, _, _ = independent_orders(t, 4, 4, 0)
        for order in range(3):
            for kind in ('P','Gamma'):
                np.testing.assert_allclose(actual[f'{kind}_MO_order{order}'], expected[f'{kind}_MO_order{order}'], atol=1e-13)

    def test_h2o_saved_fullspace_non_t_values_unchanged_without_solver_calls(self):
        record = production.load_records(ROOT/'results/initialdata', ['H2O'])[0]
        with np.load(ROOT/'tests/results/methods41_fullspace/H2O/001.npz', allow_pickle=False) as data:
            saved = {k:data[k] for k in data.files}
        mf = Mock()
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = saved['mo_coeff'], saved['mo_occ'], saved['mo_energy']
        mf.e_tot, mf.converged = float(saved['E_RHF_check']), False
        mf.canonicalize.return_value = mf.mo_energy, mf.mo_coeff
        mf.get_ovlp.return_value = saved['S_AO']
        mf.make_rdm1.return_value = saved['P_AO']
        mf.get_fock.return_value = saved['F_AO']
        mf.get_hcore.return_value = saved['hcore_AO']
        pt = Mock()
        pt.kernel.return_value = float(saved['E_MP2_correlation']), saved['mp2_t2']
        pt.make_rdm1.side_effect = lambda ao_repr=False: saved['P_MP2_AO_pyscf' if ao_repr else 'P_MP2_MO_pyscf']
        pt.make_rdm2.side_effect = lambda ao_repr=False: saved['Gamma_MP2_AO_pyscf' if ao_repr else 'Gamma_MP2_MO_pyscf']
        with patch('pyscf.scf.hf.RHF', return_value=mf), patch('pyscf.mp.mp2.RMP2', return_value=pt) as factory:
            actual = production.calculate_record(record)
        factory.assert_called_once_with(mf, frozen=0)
        for key in ('P_MP2_AO_consistent','Gamma_MP2_AO_consistent','Lambda_HFref_AO',
                    'Lambda_conventional_AO','S_half','S_minus_half','P_L','F_L','Lambda_L',
                    'P_mu','F_mu','P_munu','pair_indices','triple_indices'):
            np.testing.assert_allclose(actual[key], saved[key], rtol=0, atol=1e-12, err_msg=key)
        # Index-only extraction must be exact for this physical geometry.
        # Saved T is the known incorrect legacy mapping, not an oracle.
        v = saved['valence_indices']
        n = len(v)
        expected = np.empty((n, n, n), dtype=actual['Lambda_L'].dtype)
        saved_expected = np.empty_like(expected)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    expected[i, j, k] = actual['Lambda_L'][v[i], v[k], v[j], v[k]]
                    saved_expected[i, j, k] = saved['Lambda_L'][v[i], v[k], v[j], v[k]]
        triples = np.asarray(list(itertools.combinations(range(n), 3)))
        np.testing.assert_array_equal(actual['triple_indices'], triples)
        np.testing.assert_array_equal(actual['T_full'], expected)
        np.testing.assert_array_equal(actual['T_munulambda'], [expected[i, j, k] for i, j, k in triples])
        np.testing.assert_allclose(actual['T_full'], saved_expected, rtol=0, atol=1e-12)
        self.assertGreater(np.max(np.abs(actual['T_full'] - saved['T_full'])), 1e-3)
        self.assertGreater(np.max(np.abs(actual['T_munulambda'] - saved['T_munulambda'])), 1e-3)
        self.assertFalse(bool(actual['rhf_converged']))  # Recorded, not a validation gate.
        self.assertNotIn('status', actual)
        self.assertFalse(bool(actual['validation_performed']))
        self.assertNotIn('validation_json', actual)
        self.assertNotIn('energy_audit_json', actual)

    def test_discovery_reads_thirty_points_per_molecule(self):
        records = production.load_records(ROOT/'results/initialdata', production.MOLECULES)
        self.assertEqual(len(records), 270)
        for name in production.MOLECULES:
            self.assertEqual([r['geometry_id'] for r in records if r['molecule']==name], [f'{i:03d}' for i in range(1,31)])

    def test_dry_run_lists_without_calculations_or_files(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'output'
            with patch.object(production,'calculate_record', side_effect=AssertionError('No calculation')), contextlib.redirect_stdout(io.StringIO()):
                code = production.main(['--molecules','H2O','--dry-run','--output-dir',str(output)])
            self.assertEqual(code,0)
            self.assertFalse(output.exists())

    def test_resume_skips_npz_without_loading_or_revalidation(self):
        record = production.load_records(ROOT/'results/initialdata',['H2O'])[0]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            base = output/'H2O'/'001'
            base.parent.mkdir()
            base.with_suffix('.npz').write_bytes(b'not opened during resume')
            (output/'manifest.json').write_text(json.dumps({
                't_descriptor_version': 2, 't_axis_mapping': production.T_AXIS_MAPPING,
                'ao_selection_rule': production.AO_SELECTION_RULE,
                'records': [{'molecule': 'H2O', 'geometry_id': '001',
                             'q_A': record['q_A'], 'record': str(base.with_suffix('.npz')), 'error': ''}],
            }))
            with patch.object(production,'calculate_record', side_effect=AssertionError('No calculation')), \
                 patch.object(production.np,'load', side_effect=AssertionError('No validation')), contextlib.redirect_stdout(io.StringIO()):
                code = production.main(['--molecules','H2O','--geometry-id','001','--resume','--output-dir',str(output)])
            self.assertEqual(code,0)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['completed_geometry_count'],1)
            self.assertFalse(manifest['validation_performed'])
            self.assertEqual(manifest['t_descriptor_version'], 2)
            self.assertEqual(manifest['t_axis_mapping'], production.T_AXIS_MAPPING)
            self.assertEqual(manifest['ao_selection_rule'], production.AO_SELECTION_RULE)

    def test_legacy_or_unknown_t_mapping_cannot_resume_or_relabel_existing_data(self):
        mapping_cases = [
            {},
            {'t_descriptor_version': 1, 't_axis_mapping': production.T_AXIS_MAPPING},
            {'t_descriptor_version': 2},
            {'t_descriptor_version': 2, 't_axis_mapping': 'T[i,j,k] = raw[i,j,k,k]'},
        ]
        for mapping in mapping_cases:
            for mode in ('--resume', '--rerun-failed'):
                with self.subTest(mapping=mapping, mode=mode), tempfile.TemporaryDirectory() as folder:
                    output = Path(folder)
                    base = output/'H2O'/'001'
                    base.parent.mkdir()
                    base.with_suffix('.npz').write_bytes(b'legacy scientific arrays must remain unchanged')
                    base.with_suffix('.json').write_text(json.dumps({'molecule': 'H2O', 'geometry_id': '001'}))
                    (base.parent/'summary.csv').write_text('molecule,geometry_id\nH2O,001\n')
                    manifest = {'records': [{'molecule': 'H2O', 'geometry_id': '001',
                                              'record': str(base.with_suffix('.npz')), 'error': ''}],
                                **mapping}
                    (output/'manifest.json').write_text(json.dumps(manifest))
                    before = {path.relative_to(output): path.read_bytes()
                              for path in output.rglob('*') if path.is_file()}
                    error = io.StringIO()
                    with patch.object(production, 'calculate_record') as calculate, \
                         contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                        code = production.main(['--molecules', 'H2O', '--geometry-id', '001',
                                                mode, '--output-dir', str(output)])
                    self.assertEqual(code, 1)
                    self.assertIn('older or unknown T mapping', error.getvalue())
                    calculate.assert_not_called()
                    after = {path.relative_to(output): path.read_bytes()
                             for path in output.rglob('*') if path.is_file()}
                    self.assertEqual(after, before)

    def test_selected_batch_has_no_h2o_gate_and_failed_retry_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            args=['--molecules','LiH','--geometry-id','001','--output-dir',folder]
            with patch.object(production,'calculate_record', side_effect=RuntimeError('Synthetic failure')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(production.main(args),1)
            with patch.object(production,'calculate_record', return_value={'P_mu':np.arange(5)}) as calculate, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(production.main(args+['--rerun-failed']),0)
                self.assertEqual(calculate.call_count,1)
                self.assertEqual(calculate.call_args.args[0]['molecule'],'LiH')
            self.assertTrue((Path(folder)/'LiH/001.attempt1.json').exists())
            self.assertNotIn('status', json.loads((Path(folder)/'LiH/001.json').read_text()))


if __name__=='__main__':
    unittest.main()
