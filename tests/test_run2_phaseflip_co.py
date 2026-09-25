"""Focused CO phase-flip checks against current Run2 and all 30 saved geometries.

No production training. The CLI's separate --smoke-only mode exercises actual
one-epoch training, checkpoint restoration and output for all six models.
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
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'codes'))
import run2_phaseflip_co as phase


class COPhaseFlipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.manifest, cls.metadata = phase.run2.load_population(
            PROJECT / 'results/descriptor', ('CO',), split_protocol='15-15', methods=('MB2',))
        cls.hashes = phase.descriptor_hashes(cls.samples)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.prepared, _ = phase.prepare_runs(cls.samples, cls.manifest, cls.metadata)
        cls.model = cls.prepared['CO', 'MB2'][0]

    def test_actual_loader_uses_all_eight_valence_aos_in_existing_order(self):
        mapping = phase.ao_mapping(self.samples, self.metadata)
        self.assertEqual(mapping['P_valence_shape'], [8, 8])
        self.assertEqual(mapping['full_ao_indices'], [1, 2, 3, 4, 6, 7, 8, 9])
        self.assertEqual(mapping['ao_labels'], [
            '0 C 2s', '0 C 2px', '0 C 2py', '0 C 2pz',
            '1 O 2s', '1 O 2px', '1 O 2py', '1 O 2pz'])
        self.assertEqual(len(self.samples), 30)
        self.assertEqual(phase.PAIRS, phase.run2.pair_addresses(8))
        self.assertEqual(len(phase.PAIRS), 28)

    def test_all_geometries_invariants_MB1_lambda_source_and_both_identity_signs(self):
        report = phase.validate_experiment(self.samples, self.prepared)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['geometries'], 30)
        self.assertEqual(report['parameter_count'], 19)
        self.assertLessEqual(report['max_identity_prediction_error_Ha'], phase.ATOL)
        self.assertEqual(phase.descriptor_hashes(self.samples), self.hashes)

    def test_exact_five_vectors_and_no_trained_all_minus_or_prime(self):
        self.assertEqual(phase.METHODS, ('MB2', 'MB2_flip_0', 'MB2_flip_01', 'MB2_flip_012',
                                        'MB2_flip_0123', 'MB2_flip_01234'))
        for k, method in enumerate(phase.METHODS):
            self.assertEqual(phase.PHASES[method], (-1,) * k + (1,) * (8 - k))
            self.assertEqual(phase.PHASES[method][5:], (1, 1, 1))
        self.assertNotIn((-1,) * 8, phase.PHASES.values())
        self.assertEqual(set(self.prepared), {('CO', method) for method in phase.METHODS})

    def test_cumulative_variants_each_start_from_original_not_previous_flip(self):
        # Dense nonzero pairs expose any accidental cancellation of earlier flips.
        p = np.add.outer(np.arange(8, dtype=float), np.arange(8, dtype=float)) + 1
        np.fill_diagonal(p, self.samples[0].pii)
        original = replace(self.samples[0], pij=p)
        baseline = {('CO', 'MB2'): (self.model, (original,), (), ())}
        with patch.object(phase.run2, 'prepare_runs', return_value=(baseline, [])):
            prepared, _ = phase.prepare_runs((original,), self.manifest, self.metadata)
        for k, method in enumerate(phase.METHODS):
            changed = prepared['CO', method][1][0]
            d = np.ones(8)
            d[:k] = -1
            np.testing.assert_array_equal(changed.pij, d[:, None] * p * d[None, :])
            np.testing.assert_array_equal(changed.pij[5:, 5:], p[5:, 5:])
            np.testing.assert_array_equal(changed.lambda_pairs, original.lambda_pairs)
        np.testing.assert_array_equal(original.pij, p)

    def test_bad_phase_and_non_8_by_8_inputs_are_rejected(self):
        for bad in ((-1,) * 5, (0,) * 8, (1.5,) * 8, (float('nan'),) * 8):
            with self.subTest(phase=bad), self.assertRaises(ValueError):
                phase.phase_sample(self.samples[0], bad)
        with self.assertRaises(ValueError):
            phase.phase_sample(replace(self.samples[0], pij=np.eye(5)), (1,) * 8)

    def test_exact_original_lambda_MB1_HEA_readout_and_pair_angle_factors(self):
        theta = torch.linspace(-.37, .43, self.model.num_parameters, dtype=torch.float64)
        sample = self.samples[0]
        base_args = self.model.pair_quantum_arguments(sample, theta)
        base_ops = self.model.qnode.construct(base_args, {}).operations
        zz_start, zz_end = 8, 8 + len(phase.PAIRS)
        for method in phase.METHODS:
            model = self.prepared['CO', method][0]
            changed = phase.phase_sample(sample, phase.PHASES[method])
            args = model.pair_quantum_arguments(changed, theta)
            ops = model.qnode.construct(args, {}).operations
            self.assertEqual(model.layout.names, self.model.layout.names)
            self.assertEqual(model.layout.pair_names, ('a2', 'b2'))
            self.assertNotIn('c2', model.layout.names)
            self.assertEqual(model.num_parameters, 19)
            for actual, original in zip(ops[:zz_start] + ops[zz_end:],
                                        base_ops[:zz_start] + base_ops[zz_end:], strict=True):
                self.assertEqual(actual.name, original.name)
                self.assertEqual(tuple(actual.wires), tuple(original.wires))
                for a, b in zip(actual.data, original.data, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for k, (i, j) in enumerate(phase.PAIRS):
                op = ops[zz_start + k]
                expected = (np.pi / 2) * torch.tanh(
                    theta[-2] * changed.pij[i, j] + theta[-1] * sample.lambda_pairs[k])
                self.assertEqual(op.name, 'IsingZZ')
                self.assertEqual(tuple(op.wires), (i, j))
                torch.testing.assert_close(op.data[0], expected, rtol=0, atol=0)
                zz = torch.tensor([1., -1., -1., 1.], dtype=torch.complex128)
                torch.testing.assert_close(phase.qml.matrix(op),
                    torch.diag(torch.exp(-.5j * expected * zz)), rtol=1e-14, atol=1e-14)
            z = torch.linspace(-.6, .8, 8, dtype=torch.float64)
            torch.testing.assert_close(model.readout_features(z), self.model.readout_features(z), rtol=0, atol=0)
            torch.testing.assert_close(model.apply_readout(model.readout_features(z), theta),
                                      self.model.apply_readout(self.model.readout_features(z), theta), rtol=0, atol=0)

    def test_exact_split_original_preprocessing_and_saved_Run2_reference(self):
        population = [dict(self.metadata[r.sample_id], split=r.split, retained_position=r.retained_position)
                      for r in self.manifest.rows]
        config = phase.verify_reference(phase.REFERENCE_RUN2, population, self.prepared,
                                        PROJECT / 'results/descriptor')
        self.assertEqual(config['epochs'], 500)
        for method in phase.METHODS:
            model, training, validation, testing = self.prepared['CO', method]
            self.assertEqual(model.constants, self.model.constants)
            self.assertEqual([s.geometry_index for s in training], list(range(1, 31, 2)))
            self.assertEqual([s.geometry_index for s in testing], list(range(2, 31, 2)))
            self.assertEqual(validation, ())

    def test_all_seed_initializations_and_full_epoch_schedules_match(self):
        _, baseline_training, _, _ = self.prepared['CO', 'MB2']
        for seed in phase.run2.SEEDS:
            initial = phase.training_api.initialize_parameters(self.model, baseline_training, seed=seed)
            expected = phase.batch_schedule(baseline_training, seed, phase.run2.EPOCHS)
            for method in phase.METHODS:
                model, training, _, _ = self.prepared['CO', method]
                torch.testing.assert_close(phase.training_api.initialize_parameters(model, training, seed=seed),
                                           initial, rtol=0, atol=0)
                self.assertEqual(phase.batch_schedule(training, seed, phase.run2.EPOCHS), expected)
            self.assertEqual([len(batch) for batch in expected[0]], [8, 7])

    def test_output_rejects_existing_directory_and_escapes(self):
        for path in (phase.OUTPUT_ROOT, PROJECT / 'results/run2/new_phase_test',
                     phase.OUTPUT_ROOT / '../escape'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                phase.output_path(path)
        phase.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=phase.OUTPUT_ROOT) as folder:
            with self.assertRaises(FileExistsError):
                phase.output_path(folder)
            escape = Path(folder) / 'escape'
            escape.symlink_to(PROJECT / 'results/run2', target_is_directory=True)
            with self.assertRaises(ValueError):
                phase.output_path(escape / 'new_run')

    def test_comparison_uses_mHa_paired_baseline_and_sample_SD(self):
        phase.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        records = [dict(molecule='CO', model=method, seed=seed, split=split,
                        mae_mHa=10 + index + 2 * seed + (split == 'test'))
                   for index, method in enumerate(phase.METHODS)
                   for seed in (0, 1) for split in ('train', 'test')]
        with tempfile.TemporaryDirectory(dir=phase.OUTPUT_ROOT) as folder:
            output = Path(folder)
            phase.summarize(output, records, (0, 1))
            frame = phase.pd.read_csv(output / 'comparison.csv')
            summary = phase.pd.read_csv(output / 'comparison_summary.csv').set_index('model')
            self.assertEqual(frame.columns.tolist(), ['model', 'seed', 'train_MAE', 'test_MAE',
                                                      'test_MAE_minus_matched_baseline'])
            for index, method in enumerate(phase.METHODS):
                np.testing.assert_array_equal(frame[frame.model == method].test_MAE_minus_matched_baseline,
                                               [index, index])
                self.assertAlmostEqual(summary.loc[method, 'test_MAE_sample_SD'], np.sqrt(2))
                self.assertEqual(summary.loc[method, 'energy_unit'], 'mHa')
            phase.summarize(output, [r for r in records if r['seed'] == 0], (0,))
            single = phase.pd.read_csv(output / 'comparison_summary.csv')
            self.assertTrue(single.test_MAE_sample_SD.isna().all())

    def test_cli_requires_explicit_execution_mode(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            phase.main([])


if __name__ == '__main__':
    unittest.main()
