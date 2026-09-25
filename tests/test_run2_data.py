"""Saved-data and Run 1 protocol checks for the new pair encoders.

These tests only read the existing descriptor dataset; no electronic-structure
calculation or production training is performed.
"""
import contextlib
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import run1
import run2

INPUTS = ROOT / 'results/descriptor'
REFERENCE = ROOT / 'results/run1/20260924_115423_466018'


def raw_record(molecule, geometry_id='001'):
    path = INPUTS / molecule / f'{geometry_id}.npz'
    with np.load(path, allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}
    return path, raw


def prepare_quietly(module, samples, manifest, metadata, methods):
    with contextlib.redirect_stdout(io.StringIO()):
        return module.prepare_runs(samples, manifest, metadata, methods=methods)


class SavedPairDescriptorTests(unittest.TestCase):
    def test_raw_aabb_matches_methods_abab_in_current_qubit_order(self):
        for molecule in ('CO', 'BeH2'):
            for geometry_id in ('001', '015', '030'):
                with self.subTest(molecule=molecule, geometry=geometry_id):
                    path, raw = raw_record(molecule, geometry_id)
                    sample, metadata = run1.load_record(path, molecule, geometry_id)
                    selected = np.asarray(metadata['source_selected_ao_indices'])
                    actual = run2.extract_lambda_pairs(raw, selected, path)
                    block = raw['Lambda_L'][np.ix_(selected, selected, selected, selected)]
                    methods = block.transpose(0, 2, 1, 3)
                    i, j = np.triu_indices(len(selected), 1)
                    np.testing.assert_array_equal(actual, methods[i, j, i, j])
                    np.testing.assert_array_equal(actual, block[i, i, j, j])
                    self.assertGreater(np.max(np.abs(actual - block[i, j, i, j])), 1e-3)
                    np.testing.assert_array_equal(sample.pij[i, j], raw['P_L'][selected[i], selected[j]])
                    if molecule == 'BeH2':
                        self.assertEqual(selected.tolist(), [5, 1, 2, 3, 4, 6])
                        self.assertFalse(np.array_equal(selected, raw['valence_indices']))

    def test_unknown_axis_metadata_and_different_cumulant_are_rejected(self):
        path, raw = raw_record('CO')
        _, metadata = run1.load_record(path, 'CO', '001')
        selected = np.asarray(metadata['source_selected_ao_indices'])
        altered_cases = []
        changed = dict(raw, rdm_convention=np.asarray('Methods-order tensor, unknown storage'))
        altered_cases.append(changed)
        altered_cases.append(dict(raw, rdm_convention=np.asarray([raw['rdm_convention'].item()])))
        altered_cases.append(dict(raw, rdm_metadata_json=np.asarray([raw['rdm_metadata_json'].item()])))
        md = json.loads(raw['rdm_metadata_json'].item())
        md['formal_cumulant'] = 'Gamma_MP2_AO_consistent-Gamma0[P_MP2]'
        altered_cases.append(dict(raw, rdm_metadata_json=np.asarray(json.dumps(md))))
        md = json.loads(raw['rdm_metadata_json'].item())
        md['index_ordering_AO'] = 'Gamma[p,q,r,s]=sum_spin <p^dagger q^dagger s r>'
        altered_cases.append(dict(raw, rdm_metadata_json=np.asarray(json.dumps(md))))
        changed = dict(raw, Lambda_AO=raw['Lambda_conventional_AO'])
        altered_cases.append(changed)
        changed = dict(raw, Gamma0_AO=raw['Gamma0_AO'].copy())
        changed['Gamma0_AO'][0, 0, 0, 0] += 0.1
        altered_cases.append(changed)
        # Preserve the subtraction identity to independently catch substitution
        # of a different disconnected reference into the saved cumulant.
        altered_cases.append(dict(raw, Gamma0_AO=raw['Gamma0_correlated_AO'],
                                  Lambda_AO=raw['Lambda_conventional_AO'],
                                  Lambda_HFref_AO=raw['Lambda_conventional_AO']))
        for index, altered in enumerate(altered_cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                run2.extract_lambda_pairs(altered, selected, path)

    def test_pair_extraction_does_not_use_or_require_three_index_t(self):
        path, raw = raw_record('CO')
        _, metadata = run1.load_record(path, 'CO', '001')
        selected = np.asarray(metadata['source_selected_ao_indices'])
        expected = run2.extract_lambda_pairs(raw, selected, path)
        without_t = {key: value for key, value in raw.items()
                     if key not in ('T_full', 'T_munulambda', 'triple_indices')}
        np.testing.assert_array_equal(run2.extract_lambda_pairs(without_t, selected, path), expected)

    def test_current_loader_rejects_wrong_rank_four_shape_before_selection(self):
        path, raw = raw_record('CO')
        raw['Lambda_L'] = raw['Lambda_L'][:-1]
        saved = MagicMock()
        saved.__enter__.return_value = saved
        saved.files = list(raw)
        saved.__getitem__.side_effect = raw.__getitem__
        with patch.object(run1.np, 'load', return_value=saved), \
             patch.object(run1, 'ao_selection') as select:
            with self.assertRaisesRegex(ValueError, 'Lambda_L.*full-AO shape'):
                run1.load_record(path, 'CO', '001')
        select.assert_not_called()


class Run1ProtocolParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.manifest, cls.metadata = run2.load_population(
            INPUTS, molecules=('CO', 'BeH2'), split_protocol='15-15')
        cls.prepared, cls.audits = prepare_quietly(
            run2, cls.samples, cls.manifest, cls.metadata, run2.METHODS)

    def test_saved_benchmark_is_co_32_seeds_with_actual_15_15_ids(self):
        config = json.loads((REFERENCE / 'configuration.json').read_text())
        splits = json.loads((REFERENCE / 'split_manifest.json').read_text())
        self.assertEqual(config['molecules'], ['CO'])
        self.assertEqual(config['seeds'], list(run1.SEEDS))
        self.assertEqual(config['epochs'], run1.EPOCHS)
        self.assertEqual(config['batch_size'], run1.BATCH_SIZE)
        self.assertEqual(config['learning_rate'], run1.LEARNING_RATE)
        self.assertEqual(config['target'], run1.TARGET.formula)
        self.assertEqual(config['split_protocol'], '15-15')
        self.assertEqual(config['checkpoint_selection'],
                         'strictly improving train MAE; earliest epoch on ties')
        self.assertEqual(splits['CO']['train'], list(run1.GEOMETRY_IDS[::2]))
        self.assertEqual(splits['CO']['test'], list(run1.GEOMETRY_IDS[1::2]))

    def test_both_models_preserve_run1_populations_targets_and_scalers(self):
        old_samples, old_manifest, old_metadata = run1.load_population(
            INPUTS, molecules=('CO', 'BeH2'), split_protocol='15-15')
        old, _ = prepare_quietly(run1, old_samples, old_manifest, old_metadata, ('MB-1',))
        for molecule in ('CO', 'BeH2'):
            baseline, *base_groups = old[molecule, 'MB-1']
            for method in run2.METHODS:
                with self.subTest(molecule=molecule, method=method):
                    model, *groups = self.prepared[molecule, method]
                    self.assertEqual([len(group) for group in groups], [15, 0, 15])
                    before = asdict(baseline.constants)
                    after = asdict(model.constants)
                    before.pop('model_label')
                    after.pop('model_label')
                    self.assertEqual(before, after)
                    for original, enriched in zip(base_groups, groups, strict=True):
                        self.assertEqual(tuple(s.sample_id for s in original),
                                         tuple(s.sample_id for s in enriched))
                        for a, b in zip(original, enriched, strict=True):
                            self.assertEqual(a.target_definition, b.target_definition)
                            self.assertEqual(a.target_energy, b.target_energy)
                            self.assertEqual(b.target_energy,
                                self.metadata[b.sample_id]['E_FCI'] - self.metadata[b.sample_id]['E_RHF'])
                            for field in ('pii', 'fii', 'pij', 'fij', 'active_ao_indices'):
                                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
                            if method == 'MB2_prime':
                                self.assertIsNone(b.lambda_pairs)

    def test_heldout_descriptors_cannot_change_fitted_mb1_scalers(self):
        changed = tuple(replace(sample, pii=sample.pii * 900,
                        fii=sample.fii * 700, pij=sample.pij * 500)
                        if sample.geometry_index % 2 == 0 else sample
                        for sample in self.samples)
        after, _ = prepare_quietly(run2, changed, self.manifest,
                                  copy.deepcopy(self.metadata), run2.METHODS)
        for key, (model, training, _, _) in self.prepared.items():
            self.assertEqual(model.constants, after[key][0].constants)
            self.assertEqual(model.constants.training_sample_ids,
                             tuple(s.sample_id for s in training))

    def test_prime_only_never_extracts_cumulant_pair_inputs(self):
        with patch.object(run2, 'extract_lambda_pairs',
                          side_effect=AssertionError('Prime must not extract cumulant pairs')) as extract:
            samples, _, _ = run2.load_population(INPUTS, molecules=('CO',),
                split_protocol='15-15', methods=('MB2_prime',))
        extract.assert_not_called()
        self.assertEqual(len(samples), 30)
        self.assertTrue(all(sample.lambda_pairs is None for sample in samples))

    def test_explicit_validation_manifest_preserves_its_actual_ids(self):
        # This exercises the other current Run 1 protocol without claiming that
        # the artificial three-way memberships are the production benchmark.
        split_ids = {'CO': {split: list(run1.GEOMETRY_IDS[offset::3])
                           for offset, split in enumerate(run1.SPLITS)}}
        with tempfile.TemporaryDirectory(prefix='run2-manifest-test-') as folder:
            path = Path(folder) / 'split.json'
            path.write_text(json.dumps(split_ids))
            samples, manifest, metadata = run2.load_population(
                INPUTS, molecules=('CO',), split_manifest=path,
                split_protocol='manifest')
            prepared, _ = prepare_quietly(run2, samples, manifest, metadata, run2.METHODS)
        for method in run2.METHODS:
            model, *groups = prepared['CO', method]
            for split, group in zip(run1.SPLITS, groups, strict=True):
                self.assertEqual([f'{sample.geometry_index:03d}' for sample in group],
                                 split_ids['CO'][split])
            self.assertEqual(model.constants.training_sample_ids,
                             tuple(f'CO/{i}' for i in split_ids['CO']['train']))

    def test_prediction_reload_restores_pair_weights_and_never_refits_scalers(self):
        prepared = {key: value for key, value in self.prepared.items() if key[0] == 'CO'}
        population = [dict(self.metadata[row.sample_id], split=row.split,
                           retained_position=row.retained_position)
                      for row in self.manifest.rows if row.molecule == 'CO']
        with tempfile.TemporaryDirectory(prefix='run2-prediction-reload-') as folder:
            output = Path(folder)
            run1.write_json(output / 'configuration.json', dict(
                molecules=['CO'], methods=list(run2.METHODS), seeds=[0],
                input_dir=str(INPUTS), split_protocol='15-15'))
            run1.write_json(output / 'population.json', population)
            run1.write_json(output / 'split_manifest.json', {
                'CO': {'train': list(run1.GEOMETRY_IDS[::2]),
                       'test': list(run1.GEOMETRY_IDS[1::2])}})
            run1.write_json(output / 'preprocessing.json', {
                f'{molecule}/{method}': asdict(model.constants)
                for (molecule, method), (model, *_) in prepared.items()})
            for method in run2.METHODS:
                model, training, _, testing = prepared['CO', method]
                theta = run2.initialize_parameters(model, training, seed=0)
                with torch.no_grad():
                    for name in model.layout.pair_names:
                        theta[model.layout.names.index(name)] = .71 if name == 'a2' else -.43
                directory = output / 'runs' / 'CO' / method / 'seed_00'
                directory.mkdir(parents=True)
                np.save(directory / 'best_theta.npy', theta.detach().numpy(), allow_pickle=False)
                run1.write_json(directory / 'parameters.json', dict(
                    model=method, seed=0, parameter_names=model.layout.names,
                    parameter_count=model.num_parameters))
                with patch.object(run1, 'compute_encoding_constants',
                                  side_effect=AssertionError('Prediction must not refit scalers')):
                    restored_model, restored_theta, groups = run2.load_prediction_run(
                        output, 'CO', method, 0)
                self.assertTrue(torch.equal(theta, restored_theta))
                self.assertEqual(restored_model.constants, model.constants)
                for split, expected in (('train', training), ('test', testing)):
                    self.assertEqual(tuple(s.sample_id for s in groups[split]),
                                     tuple(s.sample_id for s in expected))
                    self.assertTrue(torch.equal(
                        model.predict_batch(expected[:2], theta),
                        restored_model.predict_batch(groups[split][:2], restored_theta)))
                if method == 'MB2_prime':
                    self.assertTrue(all(s.lambda_pairs is None for g in groups.values() for s in g))


if __name__ == '__main__':
    unittest.main()
