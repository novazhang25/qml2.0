"""Focused FE/FE_prime comparison coverage using saved real descriptor records.

Three-way memberships created here are test fixtures, not manuscript splits.
No production training sweep is launched by these tests.
"""
import contextlib
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
import circuit
import run1
import train
from test_run1_interface_repairs import synthetic_fixture


COMPARISON = ('FE', 'FE_prime')
LABELS = {'FE': 'FE  density eigenvalues:', 'FE_prime': "FE' Fock eigenvalues:"}


def printed_arrays(output):
    """Read the full-precision JSON arrays below the two public output labels."""
    lines = output.splitlines()
    return {
        method: [np.asarray(json.loads(lines[i + 1]))
                 for i, line in enumerate(lines) if line.strip() == label]
        for method, label in LABELS.items()
    }


class RealGeometrySpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.manifest, cls.metadata = run1.load_population(
            ROOT / 'results/descriptor', molecules=('CO',), split_protocol='15-15')

    def prepare(self, samples=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return run1.prepare_runs(
                self.samples if samples is None else samples,
                self.manifest, self.metadata, methods=COMPARISON)[0]

    def test_real_geometry_formulas_ascending_order_and_unchanged_fe(self):
        for molecule, spec in circuit.MOLECULES.items():
            with self.subTest(molecule=molecule):
                path = ROOT / 'results/descriptor' / molecule / '001.npz'
                sample, metadata = run1.load_record(path, molecule, '001')
                with np.load(path, allow_pickle=False) as raw:
                    selected = np.ix_(metadata['source_selected_ao_indices'],
                                      metadata['source_selected_ao_indices'])
                    p_val = (raw['S_half'] @ raw['P_AO'] @ raw['S_half'])[selected]
                    f_val = (raw['S_minus_half'] @ raw['F_AO'] @ raw['S_minus_half'])[selected]
                    old_fe = np.linalg.eigvalsh(raw['P_L'][selected])
                with contextlib.redirect_stdout(io.StringIO()):
                    features = run1.raw_spectra((sample,))[sample.sample_id]
                np.testing.assert_allclose(sample.pij, p_val, rtol=0, atol=1e-12)
                np.testing.assert_allclose(sample.fij, f_val, rtol=0, atol=1e-12)
                np.testing.assert_allclose(features['FE'], np.linalg.eigvalsh(p_val),
                                           rtol=0, atol=1e-12)
                np.testing.assert_allclose(features['FE_prime'], np.linalg.eigvalsh(f_val),
                                           rtol=0, atol=1e-12)
                # Bitwise parity with the previous FE expression, including ordering.
                np.testing.assert_array_equal(features['FE'], old_fe)
                np.testing.assert_array_equal(features['FE'], np.linalg.eigvalsh(sample.pij))
                np.testing.assert_array_equal(features['FE_prime'], np.linalg.eigvalsh(sample.fij))
                for method in COMPARISON:
                    self.assertEqual(features[method].shape, (spec.expected_active_ao_count,))
                    self.assertEqual(circuit.feature_width(molecule, method), spec.expected_active_ao_count)
                    self.assertTrue(np.all(np.diff(features[method]) >= 0))
                self.assertFalse(sample.fij.flags.writeable)
                if molecule == 'CO':
                    self.assertGreater(np.max(np.abs(sample.fij - np.diag(sample.fii))), 0.1)
                    self.assertFalse(np.allclose(features['FE_prime'], np.sort(sample.fii)))

    def test_same_feature_width_and_population_for_every_canonical_molecule(self):
        for molecule, spec in circuit.MOLECULES.items():
            with self.subTest(molecule=molecule):
                samples, manifest, metadata, _ = synthetic_fixture(molecule)
                with contextlib.redirect_stdout(io.StringIO()):
                    prepared, _ = run1.prepare_runs(samples, manifest, metadata, methods=COMPARISON)
                first, second = (prepared[molecule, method] for method in COMPARISON)
                self.assertEqual(first[0].constants.n_active, second[0].constants.n_active)
                self.assertEqual(first[0].constants.n_active, spec.expected_active_ao_count)
                for left, right in zip(first[1:], second[1:], strict=True):
                    self.assertEqual([s.sample_id for s in left], [s.sample_id for s in right])
                    for fe, prime in zip(left, right, strict=True):
                        self.assertEqual(fe.fingerprint.shape, prime.fingerprint.shape)
                        self.assertEqual(fe.fingerprint.shape, (spec.expected_active_ao_count,))

    def test_h2s_ne_core_loader_matches_saved_six_ao_record(self):
        path = ROOT / 'results/descriptor/H2S/001.npz'
        with np.load(path, allow_pickle=False) as raw:
            self.assertEqual(len(raw['valence_indices']), 6)
        self.assertEqual(circuit.MOLECULES['H2S'].expected_active_ao_count, 6)
        sample, metadata = run1.load_record(path, 'H2S', '001')
        self.assertEqual(sample.pij.shape, (6, 6))
        self.assertEqual(metadata['source_selected_ao_indices'], [2, 6, 7, 8, 9, 10])

    def test_every_printed_raw_spectrum_is_the_exact_input_to_preprocessing(self):
        output = io.StringIO()
        scaler_inputs = {}
        original_fit = run1.fit_constants

        def observed_fit(samples, **kwargs):
            # All geometries must already have been printed before the first scaler.
            shown = printed_arrays(output.getvalue())
            for method in COMPARISON:
                self.assertEqual(len(shown[method]), len(self.samples))
            scaler_inputs[kwargs['model_label']] = tuple(samples)
            self.assertTrue(all(row.split == 'train' for row in kwargs['manifest'].rows))
            return original_fit(samples, **kwargs)

        with patch.object(run1, 'fit_constants', side_effect=observed_fit), \
             patch.object(run1, 'compute_encoding_constants',
                          side_effect=AssertionError('FE comparison must not prepare MB')), \
             contextlib.redirect_stdout(output):
            prepared, _ = run1.prepare_runs(self.samples, self.manifest, self.metadata,
                                            methods=COMPARISON)
        shown = printed_arrays(output.getvalue())
        lines = output.getvalue().splitlines()
        table_headers = [i for i, line in enumerate(lines)
                         if line.split() == ['index', 'eig(P_L_valence)', 'eig(F_L_valence)']]
        self.assertEqual(len(table_headers), len(self.samples))
        for geometry_index, original in enumerate(self.samples):
            table_rows = lines[table_headers[geometry_index] + 1:
                               table_headers[geometry_index] + 1 + len(original.pii)]
            table = np.asarray([[float(value) for value in line.split()] for line in table_rows])
            np.testing.assert_array_equal(table[:, 0], np.arange(len(original.pii)))
            for method, column in (('FE', 1), ('FE_prime', 2)):
                np.testing.assert_array_equal(table[:, column], shown[method][geometry_index])
        for method in COMPARISON:
            model, training, validation, testing = prepared['CO', method]
            by_id = {sample.sample_id: sample for sample in training + validation + testing}
            printed_by_id = {original.sample_id: values
                             for original, values in zip(self.samples, shown[method], strict=True)}
            for sample_id, sample in by_id.items():
                np.testing.assert_array_equal(sample.fingerprint, printed_by_id[sample_id])
            for sample in scaler_inputs[method]:
                np.testing.assert_array_equal(sample.fingerprint, printed_by_id[sample.sample_id])
            matrix = np.vstack([sample.fingerprint for sample in training])
            np.testing.assert_array_equal(model.constants.mu, matrix.mean(axis=0))
            np.testing.assert_array_equal(model.constants.sigma, matrix.std(axis=0, ddof=0))

    def test_training_population_only_scalers_and_matched_splits_targets_and_initialization(self):
        prepared = self.prepare()
        changed = tuple(replace(sample, pij=sample.pij * 300,
                                fij=sample.fij * 500 + np.eye(len(sample.pii)))
                        if sample.geometry_index % 2 == 0 else sample for sample in self.samples)
        after = self.prepare(changed)
        first, second = (prepared['CO', method] for method in COMPARISON)
        self.assertIs(type(first[0]), circuit.FingerprintModel)
        self.assertIs(type(first[0]), type(second[0]))
        self.assertEqual(first[0].layout, second[0].layout)
        self.assertEqual(first[0].num_hea_layers, second[0].num_hea_layers)
        for group_a, group_b in zip(first[1:], second[1:], strict=True):
            self.assertEqual([(s.sample_id, s.target_energy) for s in group_a],
                             [(s.sample_id, s.target_energy) for s in group_b])
        for method in COMPARISON:
            model, training, _, _ = prepared['CO', method]
            self.assertEqual(model.constants, after['CO', method][0].constants)
            self.assertEqual(model.constants.training_sample_ids,
                             tuple(f'CO/{i:03d}' for i in range(1, 31, 2)))
            sample = training[0]
            constants = model.constants
            expected_angles = np.pi / 2 * (sample.fingerprint - constants.mu) / constants.sigma_safe
            expected_angles[np.asarray(constants.constant_mask)] = 0
            np.testing.assert_array_equal(circuit.encode(sample.fingerprint, constants),
                                           np.clip(expected_angles, -np.pi, np.pi))
        for seed in run1.SEEDS:
            theta_a = train.initialize_parameters(first[0], first[1], seed=seed)
            theta_b = train.initialize_parameters(second[0], second[1], seed=seed)
            np.testing.assert_array_equal(theta_a.detach().numpy(), theta_b.detach().numpy())

    def test_method_label_alone_cannot_change_readout_optimizer_or_training(self):
        prepared = self.prepare()
        fe_model, training, _, _ = prepared['CO', 'FE']
        # Holding descriptors fixed isolates any accidental method-specific behavior.
        prime_model = circuit.construct_model(replace(fe_model.constants, model_label='FE_prime'), 2)
        theta_fe = train.initialize_parameters(fe_model, training, seed=7)
        theta_prime = train.initialize_parameters(prime_model, training, seed=7)
        np.testing.assert_array_equal(fe_model.raw_readout_features(training[0], theta_fe).detach(),
                                      prime_model.raw_readout_features(training[0], theta_prime).detach())
        optimizers = [train.make_optimizer(theta, learning_rate=run1.LEARNING_RATE, model=model)
                      for model, theta in ((fe_model, theta_fe), (prime_model, theta_prime))]
        self.assertIs(type(optimizers[0]), type(optimizers[1]))
        self.assertEqual(optimizers[0].defaults, optimizers[1].defaults)
        self.assertEqual([tuple(p.shape) for p in optimizers[0].param_groups[0]['params']],
                         [tuple(p.shape) for p in optimizers[1].param_groups[0]['params']])
        # One complete, bounded test epoch verifies the common optimizer and readout.
        results = [train.train(model, training[:2], (), seed=7, num_epochs=1,
                               batch_size=run1.BATCH_SIZE, learning_rate=run1.LEARNING_RATE,
                               update_limit=1, checkpoint_selection='train')
                   for model in (fe_model, prime_model)]
        np.testing.assert_array_equal(results[0].theta.detach(), results[1].theta.detach())
        self.assertEqual(results[0].history, results[1].history)
        self.assertEqual(results[0].update_count, results[1].update_count)
        self.assertEqual(train.evaluate(fe_model, training[:2], theta=results[0].theta).metrics,
                         train.evaluate(prime_model, training[:2], theta=results[1].theta).metrics)

    def test_cli_comparison_validate_only_supports_both_split_protocols(self):
        synthetic_splits = {'CO': dict(train=list(run1.GEOMETRY_IDS[:10]),
                                      validation=list(run1.GEOMETRY_IDS[10:20]),
                                      test=list(run1.GEOMETRY_IDS[20:]))}
        with tempfile.TemporaryDirectory(prefix='fe-prime-test-only-') as folder:
            root = Path(folder)
            split_path = root / 'SYNTHETIC-NOT-MANUSCRIPT.json'
            split_path.write_text(json.dumps(synthetic_splits))
            for protocol in ('15-15', 'manifest'):
                with self.subTest(protocol=protocol):
                    output = root / protocol
                    args = ['--input-dir', str(ROOT / 'results/descriptor'),
                            '--molecules', 'CO', '--methods', *COMPARISON,
                            '--split-protocol', protocol, '--validate-only',
                            '--output-dir', str(output)]
                    if protocol == 'manifest':
                        args += ['--split-manifest', str(split_path)]
                    with patch.object(run1, 'train_one') as trainer, \
                         patch.object(run1, 'compute_encoding_constants',
                                      side_effect=AssertionError('MB model unexpectedly prepared')), \
                         contextlib.redirect_stdout(io.StringIO()):
                        run1.main(args)
                    trainer.assert_not_called()
                    config = json.loads((output / 'configuration.json').read_text())
                    self.assertEqual(config['methods'], list(COMPARISON))
                    self.assertEqual(config['seeds'], list(range(32)))
                    self.assertEqual((config['epochs'], config['batch_size'], config['learning_rate']),
                                     (500, 8, 0.02))
                    preprocess = json.loads((output / 'preprocessing.json').read_text())
                    self.assertEqual(set(preprocess), {'CO/FE', 'CO/FE_prime'})
                    self.assertEqual(preprocess['CO/FE']['training_sample_ids'],
                                     preprocess['CO/FE_prime']['training_sample_ids'])
                    counts = pd.DataFrame(json.loads((output / 'population.json').read_text())).split.value_counts().to_dict()
                    self.assertEqual(counts, {'train': 15, 'test': 15} if protocol == '15-15'
                                     else {'train': 10, 'validation': 10, 'test': 10})

    def test_comparison_aggregation_requires_the_same_32_seeds_for_each_method(self):
        records = [dict(molecule='CO', model=method, seed=seed, split=split,
                        mae_mHa=float(seed))
                   for method in COMPARISON for seed in run1.SEEDS for split in ('train', 'test')]
        with tempfile.TemporaryDirectory(prefix='fe-prime-summary-test-') as folder:
            output = Path(folder)
            run1.summarize(output, records, ('CO',), splits=('train', 'test'), methods=COMPARISON)
            summary = pd.read_csv(output / 'per_molecule_summary.csv')
            self.assertEqual(set(summary.model), set(COMPARISON))
            self.assertEqual(set(summary.n), {32})
            np.testing.assert_allclose(summary['mean'], np.mean(np.arange(32)))
            np.testing.assert_allclose(summary['sd'], np.std(np.arange(32), ddof=1))
            np.testing.assert_allclose(summary['se'], np.std(np.arange(32), ddof=1) / np.sqrt(32))
        with tempfile.TemporaryDirectory(prefix='fe-prime-summary-test-') as folder:
            with self.assertRaisesRegex(ValueError, 'incomplete/duplicate seed'):
                run1.summarize(Path(folder), records[:-1], ('CO',), splits=('train', 'test'),
                               methods=COMPARISON)

    def test_synthetic_cli_smoke_uses_both_methods_and_evaluates_test_once_each(self):
        samples, _, metadata, _ = synthetic_fixture()
        by_id = {s.sample_id: s for s in samples}

        def synthetic_record(path, molecule, geometry_id):
            sample_id = f'{molecule}/{geometry_id}'
            return by_id[sample_id], metadata[sample_id]

        with tempfile.TemporaryDirectory(prefix='fe-prime-synthetic-smoke-only-') as folder:
            directory = Path(folder)
            inputs = directory / 'synthetic-input'
            inputs.mkdir()
            (inputs / 'manifest.json').write_text(json.dumps(dict(
                schema_version=run1.SCHEMA,
                records=[dict(molecule='H2O', geometry_id=i) for i in run1.GEOMETRY_IDS])))
            output = directory / 'output'
            with patch.object(run1, 'load_record', side_effect=synthetic_record), \
                 patch.object(run1, 'train', wraps=run1.train) as trainer, \
                 patch.object(run1, 'evaluate', wraps=run1.evaluate) as evaluator, \
                 contextlib.redirect_stdout(io.StringIO()):
                run1.main(['--input-dir', str(inputs), '--molecules', 'H2O',
                           '--methods', *COMPARISON, '--split-protocol', '15-15',
                           '--smoke-only', '--epochs', '1', '--output-dir', str(output)])
            self.assertEqual(trainer.call_count, 2)
            self.assertEqual(evaluator.call_count, 4)
            for method, call in zip(COMPARISON, trainer.call_args_list, strict=True):
                self.assertEqual(call.args[0].constants.model_label, method)
                self.assertEqual(call.args[2], ())
                self.assertEqual({key: call.kwargs[key] for key in
                                  ('seed', 'num_epochs', 'batch_size', 'learning_rate', 'checkpoint_selection')},
                                 dict(seed=0, num_epochs=1, batch_size=8, learning_rate=.02,
                                      checkpoint_selection='train'))
                method_evaluations = [c for c in evaluator.call_args_list
                                      if c.args[0].constants.model_label == method]
                self.assertEqual([[s.geometry_index for s in c.args[1]] for c in method_evaluations],
                                 [list(range(1, 31, 2)), list(range(2, 31, 2))])
                run_dir = output / 'runs/H2O' / method / 'seed_00'
                result = json.loads((run_dir / 'result.json').read_text())
                self.assertEqual(result['final_test_evaluation_call_count'], 1)
                saved_theta = np.load(run_dir / 'best_theta.npy', allow_pickle=False)
                for evaluation in method_evaluations:
                    np.testing.assert_array_equal(evaluation.kwargs['theta'].detach(), saved_theta)
                predictions = pd.read_csv(run_dir / 'predictions.csv')
                self.assertEqual(predictions.groupby('split').size().to_dict(), {'train': 15, 'test': 15})
                self.assertEqual(set(predictions.model), {method})
                self.assertTrue(np.isfinite(predictions[['target_Ha', 'prediction_Ha', 'error_Ha']]).all().all())
            smoke = json.loads((output / 'smoke_result.json').read_text())
            self.assertEqual(smoke['status'], 'PASS')
            self.assertEqual(smoke['methods'], list(COMPARISON))
            self.assertEqual(smoke['seed'], 0)
            self.assertFalse((output / 'overall_summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
