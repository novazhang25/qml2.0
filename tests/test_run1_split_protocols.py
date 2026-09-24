"""Synthetic regression coverage for the optional historical 15/15 protocol."""
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
sys.path.insert(0, str(ROOT / 'tests'))
import run1
from test_run1_interface_repairs import prepare_quietly, synthetic_fixture


def two_way_fixture():
    samples, _, metadata, _ = synthetic_fixture()
    splits = {'H2O': dict(train=list(run1.GEOMETRY_IDS[::2]),
                         test=list(run1.GEOMETRY_IDS[1::2]))}
    manifest = run1.population_manifest(samples, splits,
        source='SYNTHETIC UNIT TEST ONLY', split_protocol='15-15')
    return samples, manifest, metadata, splits


class ProtocolSelectionTests(unittest.TestCase):
    def test_invalid_cli_combinations_fail_before_loading_or_writing(self):
        cases = (
            ([], '--split-manifest is required'),
            (['--split-protocol', 'manifest'], '--split-manifest is required'),
            (['--split-protocol', '15-15', '--split-manifest', 'unused.json'],
             'cannot be combined'),
            (['--split-protocol', 'unknown'], 'invalid choice'),
            (['--split-protocol', '15-15', '--epochs', '0'], 'must be positive'),
        )
        for args, message in cases:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'not-created'
                with patch.object(run1, 'load_population') as loader, \
                     patch.object(run1, 'write_json') as writer, \
                     patch.object(run1, 'train_one') as trainer, \
                     contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        run1.main(args + ['--output-dir', str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                loader.assert_not_called()
                writer.assert_not_called()
                trainer.assert_not_called()
                self.assertFalse(output.exists())

    def test_15_15_membership_is_odd_even_ids_independent_of_scan_order(self):
        samples, _, metadata, splits = two_way_fixture()
        samples = tuple(replace(sample, scan_coordinate=-sample.scan_coordinate)
                        for sample in reversed(samples))
        manifest = run1.population_manifest(samples, splits,
            source='SYNTHETIC UNIT TEST ONLY', split_protocol='15-15')
        prepared, audit = prepare_quietly(samples, manifest, metadata)
        self.assertEqual(manifest.split_protocol, '15-15')
        self.assertEqual((audit[0]['train'], audit[0]['validation'], audit[0]['test']),
                         (15, 0, 15))
        for method in run1.METHODS:
            model, training, validation, testing = prepared['H2O', method]
            self.assertEqual(validation, ())
            self.assertEqual({s.geometry_index for s in training}, set(range(1, 31, 2)))
            self.assertEqual({s.geometry_index for s in testing}, set(range(2, 31, 2)))
            self.assertEqual(model.constants.training_sample_ids,
                             tuple(s.sample_id for s in training))

    def test_scaler_fit_receives_only_train_and_heldout_changes_cannot_affect_it(self):
        samples, manifest, metadata, splits = two_way_fixture()
        with patch.object(run1, 'fit_constants', wraps=run1.fit_constants) as fg_fe_fit, \
             patch.object(run1, 'compute_encoding_constants', wraps=run1.compute_encoding_constants) as mb_fit:
            before, _ = prepare_quietly(samples, manifest, metadata)
        expected = tuple(f'H2O/{i}' for i in splits['H2O']['train'])
        self.assertEqual((fg_fe_fit.call_count, mb_fit.call_count), (2, 1))
        for call in fg_fe_fit.call_args_list + mb_fit.call_args_list:
            self.assertEqual(tuple(s.sample_id for s in call.args[0]), expected)
            self.assertEqual(tuple(r.sample_id for r in call.kwargs['manifest'].rows), expected)
            self.assertTrue(all(r.split == 'train' for r in call.kwargs['manifest'].rows))
        changed_metadata = copy.deepcopy(metadata)
        changed_samples = []
        for sample in samples:
            if sample.geometry_index % 2 == 0:
                sample = replace(sample, pii=sample.pii * 900, fii=sample.fii * 700,
                                 pij=sample.pij * 500)
                changed_metadata[sample.sample_id]['fg_coordinates'] = (
                    np.asarray(changed_metadata[sample.sample_id]['fg_coordinates']) * 100).tolist()
            changed_samples.append(sample)
        after, _ = prepare_quietly(changed_samples, manifest, changed_metadata)
        for method in run1.METHODS:
            self.assertEqual(before['H2O', method][0].constants, after['H2O', method][0].constants)


class TwoWaySyntheticSmokeTests(unittest.TestCase):
    def test_cli_uses_best_train_checkpoint_and_evaluates_each_test_once(self):
        samples, _, metadata, splits = two_way_fixture()
        by_id = {s.sample_id: s for s in samples}
        with tempfile.TemporaryDirectory(prefix='run1-15-15-synthetic-only-') as folder:
            directory = Path(folder)
            inputs = directory / 'synthetic-input'
            inputs.mkdir()
            (inputs / 'manifest.json').write_text(json.dumps(dict(schema_version=run1.SCHEMA,
                records=[dict(molecule='H2O', geometry_id=i) for i in run1.GEOMETRY_IDS])))
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
                run1.main(['--input-dir', str(inputs), '--split-protocol', '15-15',
                           '--molecules', 'H2O', '--smoke-only', '--epochs', '500',
                           '--output-dir', str(output)])
            self.assertEqual(trainer.call_count, 3)
            for call in trainer.call_args_list:
                self.assertEqual(tuple(s.sample_id for s in call.args[1]),
                                 tuple(f'H2O/{i}' for i in splits['H2O']['train']))
                self.assertEqual(call.args[2], ())
                self.assertEqual(call.kwargs['checkpoint_selection'], 'train')
                self.assertEqual(call.kwargs['num_epochs'], 2)
            self.assertEqual(len(calls), 6)
            for method in run1.METHODS:
                model_calls = [c for c in calls if c[0] == method]
                self.assertEqual([c[1] for c in model_calls],
                    [tuple(f'H2O/{i}' for i in splits['H2O'][split]) for split in ('train', 'test')])
                run_dir = output / 'runs/H2O' / method / 'seed_00'
                result = json.loads((run_dir / 'result.json').read_text())
                self.assertEqual(result['checkpoint_selection'], 'train')
                self.assertEqual(result['final_test_evaluation_call_count'], 1)
                self.assertFalse(any('validation' in key for key in result))
                history = pd.read_csv(run_dir / 'training_history.csv')
                self.assertEqual(len(history), 2)
                self.assertFalse(any('test' in name or 'validation' in name for name in history.columns))
                self.assertEqual(result['best_epoch'], int(history.loc[history.train_mae_ha.idxmin(), 'epoch']))
                self.assertAlmostEqual(result['best_train_mae_ha'], result['restored_train_metrics']['mae_hartree'])
                self.assertAlmostEqual(result['checkpoint_train_mae_ha'], result['restored_train_metrics']['mae_hartree'])
                restored = np.load(run_dir / 'best_theta.npy', allow_pickle=False)
                for _, _, theta in model_calls:
                    np.testing.assert_array_equal(theta, restored)
                for filename in ('metrics.csv', 'predictions.csv'):
                    frame = pd.read_csv(run_dir / filename)
                    self.assertEqual(set(frame.split), {'train', 'test'})
                    self.assertFalse(any('validation' in name for name in frame.columns))
                    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
                    self.assertTrue(np.isrealobj(numeric) and np.isfinite(numeric).all())
                predictions = pd.read_csv(run_dir / 'predictions.csv')
                self.assertEqual(predictions.groupby('split').size().to_dict(), {'test': 15, 'train': 15})
            config = json.loads((output / 'configuration.json').read_text())
            self.assertEqual(config['split_protocol'], '15-15')
            self.assertIsNone(config['split_manifest'])
            self.assertEqual(config['seeds'], [0])
            self.assertEqual(config['epochs'], 2)
            self.assertIn('train MAE', config['checkpoint_selection'])
            self.assertEqual(json.loads((output / 'split_manifest.json').read_text()), splits)
            self.assertEqual({r['split'] for r in json.loads((output / 'population.json').read_text())},
                             {'train', 'test'})
            self.assertFalse((output / 'overall_summary.csv').exists())


class TwoWaySummaryTests(unittest.TestCase):
    def setUp(self):
        self.molecules = ('H2O', 'CO')
        self.records = [dict(molecule=molecule, model=method, seed=seed, split=split,
                             mae_mHa=seed + molecule_index * 2)
                        for split in ('train', 'test')
                        for molecule_index, molecule in enumerate(self.molecules)
                        for method in run1.METHODS for seed in range(32)]

    def test_summaries_include_all_32_seeds_and_only_train_test(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            run1.summarize(output, self.records, self.molecules, splits=('train', 'test'))
            summaries = pd.read_csv(output / 'per_molecule_summary.csv')
            self.assertEqual(len(summaries), 12)
            self.assertEqual(set(summaries.split), {'train', 'test'})
            self.assertEqual(set(summaries.n), {32})
            overall = pd.read_csv(output / 'overall_summary.csv')
            self.assertEqual(len(overall), 6)
            self.assertEqual(set(overall.split), {'train', 'test'})
            self.assertEqual(set(overall.n), {32})
            np.testing.assert_allclose(overall['mean'], np.mean(np.arange(32) + 1))
            np.testing.assert_allclose(overall['sd'], np.std(np.arange(32), ddof=1))
            np.testing.assert_allclose(overall['se'], np.std(np.arange(32), ddof=1) / np.sqrt(32))
            per_seed = pd.read_csv(output / 'overall_per_seed.csv')
            self.assertEqual(len(per_seed), 32 * 3 * 2)
            self.assertEqual(set(per_seed.split), {'train', 'test'})

    def test_missing_duplicate_or_unexpected_populations_fail_before_output(self):
        cases = (self.records[1:], self.records + [self.records[0]],
                 self.records + [dict(self.records[0], split='validation')])
        for records in cases:
            with self.subTest(size=len(records)), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                with self.assertRaisesRegex(ValueError, 'incomplete/duplicate seed|populations'):
                    run1.summarize(output, records, self.molecules, splits=('train', 'test'))
                self.assertEqual(list(output.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
