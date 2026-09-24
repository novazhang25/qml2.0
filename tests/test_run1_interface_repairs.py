"""Synthetic interface tests only: these IDs are NOT manuscript split evidence."""
import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import circuit
import run1


def synthetic_fixture(molecule='H2O'):
    """Build artificial descriptors/targets and a test-only three-way partition."""
    spec = circuit.MOLECULES[molecule]
    n = spec.expected_active_ao_count
    ids = list(run1.GEOMETRY_IDS)
    splits = {molecule: dict(train=ids[:10], validation=ids[10:20], test=ids[20:])}
    samples, metadata = [], {}
    for index, geometry_id in enumerate(ids, 1):
        sample_id = f'{molecule}/{geometry_id}'
        p = np.diag(np.linspace(.2, 1.7, n) + index / 100) + .01 * np.ones((n, n))
        target = -.05 - index / 10000
        samples.append(circuit.ProcessedSample(
            sample_id=sample_id, dataset_id=run1.DATASET_ID, molecule=molecule,
            geometry_index=index, scan_coordinate=float(index), scan_coordinate_name='synthetic',
            scan_coordinate_unit='angstrom', basis='sto-3g',
            active_ao_indices=np.asarray(spec.expected_active_ao_indices),
            active_ao_labels=spec.expected_active_ao_labels, active_ao_order_source='synthetic',
            pii=np.diag(p), fii=np.linspace(-2, 1, n) + index / 200,
            pij=p, fij=np.diag(np.linspace(-2, 1, n) + index / 200) + .03 * (np.ones((n, n)) - np.eye(n)), t_lowdin=None, uhf_total_energy=None, fci_total_energy=target,
            fci_corr_target=target, target_energy=target, target_definition=run1.TARGET,
            ump2_total_energy=None, ump2_corr_energy=None, source_file=Path('synthetic.npz'),
            source_file_id=f'{sample_id}.npz', source_sha256='', adapter_profile='synthetic',
            field_provenance=()))
        coords = np.array([[0., 0., 0.], [1 + index / 100, 0., 0.],
                           [-.25, 1 + index / 200, 0.]])
        if molecule == 'H2O2':
            coords = np.array([[float(index), 0., 0.], [0., 0., 1.5],
                               [1., 0., 0.], [0., 1., 1.5]])
        metadata[sample_id] = dict(geometry_id=geometry_id, q_A=float(index),
            bond_length_A=1 + index / 100, fg_coordinates=coords.tolist())
    manifest = run1.population_manifest(samples, splits, source='SYNTHETIC UNIT TEST ONLY')
    return tuple(samples), manifest, metadata, splits


def prepare_quietly(samples, manifest, metadata):
    with contextlib.redirect_stdout(io.StringIO()):
        return run1.prepare_runs(samples, manifest, metadata)


class SplitManifestTests(unittest.TestCase):
    def setUp(self):
        self.samples, self.manifest, self.metadata, self.splits = synthetic_fixture()
        self.inventory = {'H2O': run1.GEOMETRY_IDS}

    def test_explicit_three_way_split_covers_each_of_thirty_ids_once(self):
        actual = run1.validate_split_manifest(self.splits, self.inventory)
        groups = [set(actual['H2O'][split]) for split in run1.SPLITS]
        self.assertTrue(all(groups))
        self.assertEqual(set.union(*groups), set(run1.GEOMETRY_IDS))
        self.assertEqual(sum(map(len, groups)), 30)
        self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]))
        self.assertEqual({r.sample_id for r in self.manifest.rows if r.split == 'validation'},
                         {f'H2O/{i}' for i in self.splits['H2O']['validation']})

    def test_duplicate_within_group_and_across_groups_fail(self):
        for split in ('train', 'validation'):
            with self.subTest(split=split):
                bad = copy.deepcopy(self.splits)
                bad['H2O'][split].append('001')
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    run1.validate_split_manifest(bad, self.inventory)

    def test_missing_unknown_empty_or_malformed_groups_fail(self):
        for change, message in (
            (lambda b: b['H2O']['test'].pop(), 'missing geometry'),
            (lambda b: b['H2O']['test'].append('999'), 'unknown geometry'),
            (lambda b: b['H2O'].update(validation=[]), 'nonempty'),
            (lambda b: b['H2O'].pop('validation'), 'exactly train'),
            (lambda b: b['H2O'].update(validation=[11]), 'string geometry'),
            (lambda b: b['H2O'].update(extra=['001']), 'exactly train'),
        ):
            bad = copy.deepcopy(self.splits)
            change(bad)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                run1.validate_split_manifest(bad, self.inventory)

    def test_unknown_and_missing_molecules_fail(self):
        for bad in ({'FAKE': self.splits['H2O']}, {}, {'H2O': self.splits['H2O'], 'FAKE': {}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run1.validate_split_manifest(bad, self.inventory)

    def test_inventory_must_have_exactly_thirty_unique_geometries(self):
        for inventory in (run1.GEOMETRY_IDS[:-1], run1.GEOMETRY_IDS + ('031',),
                          run1.GEOMETRY_IDS[:-1] + ('001',)):
            with self.subTest(inventory=inventory), self.assertRaisesRegex(ValueError, '30 unique'):
                run1.validate_split_manifest(self.splits, {'H2O': inventory})

    def test_duplicate_json_keys_fail_instead_of_silently_overwriting(self):
        for content in ('{"H2O": {}, "H2O": {}}', '{"H2O": {"train": [], "train": []}}'):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'synthetic.json'
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, 'Duplicate split manifest key'):
                    run1.read_split_manifest(path)

    def test_absent_manifest_blocks_production_before_loading_or_writing(self):
        with patch.object(run1, 'load_population') as loader, \
             patch.object(run1, 'train_one') as trainer, \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                run1.main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('32-seed benchmark is blocked', stderr.getvalue())
        loader.assert_not_called()
        trainer.assert_not_called()
        with self.assertRaisesRegex(ValueError, '--split-manifest is required'):
            run1.load_population(Path('unused'))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.samples, self.manifest, self.metadata, self.splits = synthetic_fixture()

    def test_scaler_functions_receive_only_train_ids_and_four_tuple_interface(self):
        with patch.object(run1, 'fit_constants', wraps=run1.fit_constants) as fingerprint_fit, \
             patch.object(run1, 'compute_encoding_constants', wraps=run1.compute_encoding_constants) as mb_fit:
            prepared, _ = prepare_quietly(self.samples, self.manifest, self.metadata)
        expected = tuple(f'H2O/{i}' for i in self.splits['H2O']['train'])
        self.assertEqual(fingerprint_fit.call_count, 2)
        self.assertEqual(mb_fit.call_count, 1)
        for call in fingerprint_fit.call_args_list + mb_fit.call_args_list:
            self.assertEqual(tuple(s.sample_id for s in call.args[0]), expected)
            self.assertTrue(all(r.split == 'train' for r in call.kwargs['manifest'].rows))
        for method in run1.METHODS:
            model, training, validation, testing = prepared['H2O', method]
            self.assertEqual(model.constants.training_sample_ids, expected)
            for split, population in zip(run1.SPLITS, (training, validation, testing)):
                self.assertEqual(tuple(s.sample_id for s in population),
                                 tuple(f'H2O/{i}' for i in self.splits['H2O'][split]))

    def test_heldout_descriptor_changes_do_not_change_any_fitted_statistics(self):
        before, _ = prepare_quietly(self.samples, self.manifest, self.metadata)
        modified_metadata = copy.deepcopy(self.metadata)
        modified_samples = []
        for sample in self.samples:
            if sample.geometry_index > 10:
                sample = replace(sample, pii=sample.pii * 900, fii=sample.fii * 700, pij=sample.pij * 500)
                modified_metadata[sample.sample_id]['fg_coordinates'] = (
                    np.asarray(modified_metadata[sample.sample_id]['fg_coordinates']) * 100).tolist()
            modified_samples.append(sample)
        after, _ = prepare_quietly(modified_samples, self.manifest, modified_metadata)
        for method in run1.METHODS:
            self.assertEqual(before['H2O', method][0].constants, after['H2O', method][0].constants)

    def test_fe_remains_eigvalsh_of_complete_valence_density_block(self):
        prepared, _ = prepare_quietly(self.samples, self.manifest, self.metadata)
        by_id = {s.sample_id: s for s in self.samples}
        for population in prepared['H2O', 'FE'][1:]:
            for sample in population:
                np.testing.assert_array_equal(sample.fingerprint, np.linalg.eigvalsh(by_id[sample.sample_id].pij))
                self.assertFalse(np.array_equal(sample.fingerprint, np.sort(by_id[sample.sample_id].pii)))
                self.assertTrue(np.isrealobj(sample.fingerprint) and np.isfinite(sample.fingerprint).all())

    def test_saved_record_validates_full_ao_before_selecting_valence(self):
        path = ROOT / 'results/descriptor/H2O/001.npz'
        events = []
        original_check, original_select = run1.check_record_array, run1.ao_selection

        def check(actual, expected, path, key, *args, **kwargs):
            events.append(key)
            return original_check(actual, expected, path, key, *args, **kwargs)

        def select(*args, **kwargs):
            events.append('SELECT_VALENCE')
            return original_select(*args, **kwargs)

        with patch.object(run1, 'check_record_array', side_effect=check), \
             patch.object(run1, 'ao_selection', side_effect=select):
            sample, metadata = run1.load_record(path, 'H2O', '001')
        for key in ('full-AO P_L = S_half @ P_AO @ S_half', 'full-AO F_L = S_minus_half @ F_AO @ S_minus_half',
                    'descriptor.transform_rank4: full-AO Lowdin Lambda transformation'):
            self.assertLess(events.index(key), events.index('SELECT_VALENCE'))
        with np.load(path, allow_pickle=False) as raw:
            selected = metadata['source_selected_ao_indices']
            block = (raw['S_half'] @ raw['P_AO'] @ raw['S_half'])[np.ix_(selected, selected)]
            np.testing.assert_allclose(sample.pij, block, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(sample.pii, raw['P_mu'])
            np.testing.assert_array_equal(sample.fii, raw['F_mu'])
            self.assertEqual(sample.target_energy, float(raw['E_FCI']) - float(raw['E_RHF_input']))

    def test_h2o2_existing_torsion_unwrap_moves_with_feature_name_only(self):
        samples, manifest, metadata, _ = synthetic_fixture('H2O2')
        width = circuit.feature_width('H2O2', 'FG')
        torsion = circuit.fg_feature_names('H2O2').index('dihedral_H1_O1_O2_H2')
        raw = np.tile(np.arange(1., width + 1), (30, 1))
        raw[:, torsion] = (np.linspace(3., 3.4, 30) + np.pi) % (2 * np.pi) - np.pi
        with patch.object(run1, 'fg_features', side_effect=lambda molecule, coords: raw[int(coords[0, 0]) - 1].copy()):
            prepared, _ = prepare_quietly(samples, manifest, metadata)
        actual = np.vstack([s.fingerprint for group in prepared['H2O2', 'FG'][1:] for s in group])
        expected = raw.copy()
        expected[:, torsion] = np.unwrap(raw[:, torsion])
        np.testing.assert_array_equal(actual, expected)


class SyntheticSmokeTests(unittest.TestCase):
    def test_synthetic_cli_smoke_restores_then_evaluates_test_exactly_once_per_model(self):
        samples, _, metadata, splits = synthetic_fixture()
        by_id = {s.sample_id: s for s in samples}
        with tempfile.TemporaryDirectory(prefix='run1-synthetic-only-') as folder:
            directory = Path(folder)
            inputs = directory / 'synthetic-input'
            inputs.mkdir()
            (inputs / 'manifest.json').write_text(json.dumps(dict(schema_version=run1.SCHEMA,
                records=[dict(molecule='H2O', geometry_id=i) for i in run1.GEOMETRY_IDS])))
            manifest_path = directory / 'SYNTHETIC-NOT-MANUSCRIPT.json'
            manifest_path.write_text(json.dumps(splits))
            output = directory / 'output'
            calls = []
            original_evaluate = run1.evaluate

            def observed_evaluate(model, population, **kwargs):
                calls.append((model.constants.model_label, tuple(s.sample_id for s in population),
                              kwargs['theta'].detach().numpy().copy()))
                return original_evaluate(model, population, **kwargs)

            def synthetic_record(path, molecule, geometry_id):
                sid = f'{molecule}/{geometry_id}'
                return by_id[sid], metadata[sid]

            with patch.object(run1, 'load_record', side_effect=synthetic_record), \
                 patch.object(run1, 'evaluate', side_effect=observed_evaluate), \
                 patch.object(run1, 'train', wraps=run1.train) as trainer, \
                 contextlib.redirect_stdout(io.StringIO()):
                run1.main(['--input-dir', str(inputs), '--split-manifest', str(manifest_path),
                           '--molecules', 'H2O', '--smoke-only', '--epochs', '2', '--output-dir', str(output)])
            self.assertEqual(trainer.call_count, 3)
            for call in trainer.call_args_list:
                self.assertEqual(tuple(s.sample_id for s in call.args[2]),
                                 tuple(f'H2O/{i}' for i in splits['H2O']['validation']))
            self.assertEqual(len(calls), 9)
            for method in run1.METHODS:
                model_calls = [c for c in calls if c[0] == method]
                self.assertEqual([c[1] for c in model_calls],
                                 [tuple(f'H2O/{i}' for i in splits['H2O'][split]) for split in run1.SPLITS])
                run_dir = output / 'runs/H2O' / method / 'seed_00'
                result = json.loads((run_dir / 'result.json').read_text())
                self.assertEqual(result['final_test_evaluation_call_count'], 1)
                history = pd.read_csv(run_dir / 'training_history.csv')
                self.assertEqual(len(history), 2)
                self.assertFalse(any('test' in name for name in history.columns))
                self.assertEqual(result['best_epoch'], int(history.loc[history.validation_mae_ha.idxmin(), 'epoch']))
                self.assertAlmostEqual(result['best_validation_mae_ha'], result['restored_validation_metrics']['mae_hartree'])
                self.assertAlmostEqual(result['checkpoint_train_mae_ha'], result['restored_train_metrics']['mae_hartree'])
                restored = np.load(run_dir / 'best_theta.npy', allow_pickle=False)
                for _, _, theta in model_calls:
                    np.testing.assert_array_equal(theta, restored)
                for filename in ('metrics.csv', 'predictions.csv', 'training_history.csv'):
                    numeric = pd.read_csv(run_dir / filename).select_dtypes(include=[np.number]).to_numpy()
                    self.assertTrue(np.isrealobj(numeric) and np.isfinite(numeric).all())
            config = json.loads((output / 'configuration.json').read_text())
            self.assertEqual(config['seeds'], [0])
            self.assertIn('manuscript provenance is not established', config['split_ids_source'])
            self.assertFalse((output / 'overall_summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
