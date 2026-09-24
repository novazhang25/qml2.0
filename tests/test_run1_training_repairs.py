"""Synthetic checks for both checkpoint protocols and test-isolated training."""
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
import train as training


class ScriptedModel:
    """Expose different train/validation minima at known updated parameters."""

    def __init__(self, train_errors, validation_errors):
        self.errors = {'train': train_errors, 'validation': validation_errors}
        self.seen = []
        self.initialization_ids = None

    def initialize_theta(self, samples, seed):
        self.initialization_ids = tuple(s.sample_id for s in samples)
        return torch.zeros(3, dtype=torch.float64, requires_grad=True)

    def predict_batch(self, samples, theta):
        epoch = int(theta[0].item())
        predictions = []
        for sample in samples:
            if sample.sample_id not in ('train-0', 'train-1', 'validation-0'):
                raise AssertionError('Training accessed a forbidden population')
            population = sample.sample_id.rsplit('-', 1)[0]
            self.seen.append((epoch, sample.sample_id))
            predictions.append(self.errors[population][epoch - 1])
        return torch.tensor(predictions, dtype=torch.float64)


def sample(sample_id):
    return SimpleNamespace(sample_id=sample_id, target_energy=0.0)


def scripted_update(model, batch, theta, optimizer, *, epoch):
    with torch.no_grad():
        theta[0] = epoch
    return torch.zeros((), dtype=torch.float64)


class ValidationCheckpointTests(unittest.TestCase):
    def run_script(self, train_errors, validation_errors):
        model = ScriptedModel(train_errors, validation_errors)
        train_samples = [sample('train-0'), sample('train-1')]
        validation_samples = [sample('validation-0')]
        with patch.object(training, 'optimizer_step', side_effect=scripted_update):
            result = training.train(
                model, train_samples, validation_samples,
                seed=9, num_epochs=len(train_errors), batch_size=2,
            )
        return model, result

    def test_restores_validation_winner_instead_of_train_winner_or_last_epoch(self):
        _, result = self.run_script([0.1, 0.4, 0.3], [0.6, 0.2, 0.5])
        self.assertEqual(result.checkpoint.epoch, 2)
        self.assertEqual(result.checkpoint.train_mae_ha, 0.4)
        self.assertEqual(result.checkpoint.validation_mae_ha, 0.2)
        self.assertEqual(result.theta[0].item(), 2)
        self.assertEqual(result.update_count, 3)
        self.assertEqual(min(result.history, key=lambda row: row['train_mae_ha'])['epoch'], 1)
        self.assertEqual(result.history[-1]['epoch'], 3)

    def test_validation_tie_keeps_earliest_epoch_despite_better_training_mae(self):
        _, result = self.run_script([0.8, 0.2, 0.1], [0.2, 0.2, 0.4])
        self.assertEqual(result.checkpoint.epoch, 1)
        self.assertEqual(result.theta[0].item(), 1)
        self.assertEqual(result.checkpoint.train_mae_ha, 0.8)

    def test_checkpoint_owns_parameter_and_optimizer_copies(self):
        theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        optimizer_state = {'state': {'value': torch.tensor([3.0])}}
        optimizer = SimpleNamespace(state_dict=lambda: optimizer_state)
        best = training.BestCheckpoint()
        self.assertTrue(best.consider(epoch=1, train_mae=0.6, validation_mae=0.2,
                                      theta=theta, optimizer=optimizer))
        with torch.no_grad():
            theta.fill_(9)
        optimizer_state['state']['value'].fill_(7)
        checkpoint = best.restore(theta)
        self.assertEqual(theta.item(), 1)
        self.assertEqual(checkpoint.optimizer_state['state']['value'].item(), 3)
        self.assertEqual(best.best_validation_mae, 0.2)
        self.assertFalse(hasattr(best, 'best_train_mae'))

    def test_each_epoch_evaluates_only_train_and_validation_after_updates(self):
        model, result = self.run_script([0.3, 0.2, 0.1], [0.4, 0.1, 0.2])
        self.assertEqual(model.initialization_ids, ('train-0', 'train-1'))
        self.assertEqual(model.seen, [
            (epoch, geometry_id)
            for epoch in (1, 2, 3)
            for geometry_id in ('train-0', 'train-1', 'validation-0')
        ])
        expected_keys = {'epoch', 'train_mae_ha', 'train_mae_mha',
                         'validation_mae_ha', 'validation_mae_mha'}
        for row in result.history:
            self.assertEqual(set(row), expected_keys)
            self.assertTrue(np.isrealobj(list(row.values())))
            self.assertTrue(np.all(np.isfinite(list(row.values()))))

    def test_training_interface_cannot_accept_test_population(self):
        signature = inspect.signature(training.train)
        self.assertIn('validation_samples', signature.parameters)
        self.assertNotIn('test_samples', signature.parameters)
        with self.assertRaisesRegex(TypeError, 'test_samples'):
            training.train(None, [sample('train-0')], [sample('validation-0')],
                           test_samples=object(), seed=0, num_epochs=1)

    def test_empty_validation_and_invalid_epochs_fail_before_initialization(self):
        model = SimpleNamespace(initialize_theta=lambda *_: self.fail('Unexpected initialization'))
        with self.assertRaisesRegex(ValueError, 'Validation set is empty'):
            training.train(model, [sample('train-0')], [], seed=0, num_epochs=1)
        for epochs in (0, -1, 1.5, True):
            with self.subTest(epochs=epochs), self.assertRaisesRegex(ValueError, 'positive integer'):
                training.train(model, [sample('train-0')], [sample('validation-0')],
                               seed=0, num_epochs=epochs)

    def test_nonfinite_validation_metric_cannot_create_checkpoint(self):
        theta = torch.zeros(1, dtype=torch.float64)
        optimizer = SimpleNamespace(state_dict=lambda: {})
        for value in (np.nan, np.inf, -np.inf, 1 + 1j):
            with self.subTest(value=value), self.assertRaises((ValueError, FloatingPointError)):
                training.BestCheckpoint().consider(
                    epoch=1, train_mae=0.1, validation_mae=value,
                    theta=theta, optimizer=optimizer,
                )


class TrainingCheckpointTests(unittest.TestCase):
    def run_script(self, train_errors):
        # Any attempted validation prediction fails because its error list is empty.
        model = ScriptedModel(train_errors, [])
        updates = []
        with patch.object(training, 'optimizer_step', side_effect=scripted_update):
            result = training.train(
                model, [sample('train-0'), sample('train-1')], (),
                seed=9, num_epochs=len(train_errors), batch_size=2,
                checkpoint_selection='train',
                on_update=lambda epoch, batch, *_: updates.append(
                    (epoch, tuple(s.sample_id for s in batch))),
            )
        return model, result, updates

    def test_restores_training_winner_instead_of_last_epoch(self):
        _, result, _ = self.run_script([0.3, 0.1, 0.2])
        self.assertEqual(result.checkpoint.epoch, 2)
        self.assertEqual(result.checkpoint.train_mae_ha, 0.1)
        self.assertIsNone(result.checkpoint.validation_mae_ha)
        self.assertEqual(result.theta[0].item(), 2)
        self.assertEqual(result.history[-1]['epoch'], 3)
        self.assertEqual(result.update_count, 3)

    def test_training_tie_keeps_earliest_epoch(self):
        _, result, _ = self.run_script([0.4, 0.1, 0.1])
        self.assertEqual(result.checkpoint.epoch, 2)
        self.assertEqual(result.theta[0].item(), 2)

    def test_only_training_population_is_accessed_and_no_validation_metrics_exist(self):
        model, result, updates = self.run_script([0.3, 0.1, 0.2])
        self.assertEqual(model.initialization_ids, ('train-0', 'train-1'))
        self.assertEqual(model.seen, [
            (epoch, geometry_id)
            for epoch in (1, 2, 3)
            for geometry_id in ('train-0', 'train-1')
        ])
        self.assertEqual(len(updates), 3)
        for epoch, sample_ids in updates:
            self.assertIn(epoch, (1, 2, 3))
            self.assertEqual(set(sample_ids), {'train-0', 'train-1'})
        for row in result.history:
            self.assertEqual(set(row), {'epoch', 'train_mae_ha', 'train_mae_mha'})
            self.assertTrue(np.all(np.isfinite(list(row.values()))))

    def test_invalid_mode_or_nonempty_validation_fails_before_initialization(self):
        model = SimpleNamespace(initialize_theta=lambda *_: self.fail('Unexpected initialization'))
        with self.assertRaisesRegex(ValueError, 'empty validation set'):
            training.train(model, [sample('train-0')], [sample('validation-0')],
                           seed=0, num_epochs=1, checkpoint_selection='train')
        for selection in ('test', 'auto', '', None):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'checkpoint_selection'):
                training.train(model, [sample('train-0')], (),
                               seed=0, num_epochs=1, checkpoint_selection=selection)
        with self.assertRaisesRegex(TypeError, 'test_samples'):
            training.train(model, [sample('train-0')], (), test_samples=object(),
                           seed=0, num_epochs=1, checkpoint_selection='train')

    def test_nonfinite_training_metric_cannot_create_checkpoint(self):
        theta = torch.zeros(1, dtype=torch.float64)
        optimizer = SimpleNamespace(state_dict=lambda: {})
        for value in (np.nan, np.inf, -np.inf, 1 + 1j):
            with self.subTest(value=value), self.assertRaises((ValueError, FloatingPointError)):
                training.BestCheckpoint(checkpoint_selection='train').consider(
                    epoch=1, train_mae=value, theta=theta, optimizer=optimizer)

    def test_checkpoint_rejects_metrics_from_the_wrong_protocol(self):
        kwargs = dict(epoch=1, train_mae=0.1, theta=torch.zeros(1, dtype=torch.float64),
                      optimizer=SimpleNamespace(state_dict=lambda: {}))
        with self.assertRaisesRegex(ValueError, 'Validation MAE is required'):
            training.BestCheckpoint().consider(**kwargs)
        with self.assertRaisesRegex(ValueError, 'validation_mae=None'):
            training.BestCheckpoint(checkpoint_selection='train').consider(
                validation_mae=0.2, **kwargs)


class FiniteRealMetricTests(unittest.TestCase):
    def test_prediction_rejects_nonfinite_or_complex_values(self):
        for value in (np.nan, np.inf, 1 + 1j, 1 + 0j):
            model = SimpleNamespace(predict_batch=lambda *_: torch.as_tensor([value]))
            with self.subTest(value=value), self.assertRaises((ValueError, FloatingPointError)):
                training.predict_many(model, [sample('train-0')], torch.zeros(1), batch_size=1)

    def test_mae_and_evaluation_reject_nonreal_nonfinite_or_empty_inputs(self):
        for metric in (training.mae, training.error_metrics):
            for values in ([np.nan], [np.inf], [1 + 1j], []):
                with self.subTest(metric=metric.__name__, values=values):
                    with self.assertRaises((ValueError, FloatingPointError)):
                        metric(values, [0.0] * len(values))

    def test_finite_inputs_cannot_silently_emit_overflowed_metrics(self):
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaises(FloatingPointError):
                training.mae(np.array([-1e308]), np.array([1e308]))
            with self.assertRaises(FloatingPointError):
                training.error_metrics(np.array([0.0]), np.array([1e308]))

    def test_metric_values_and_evaluation_outputs_are_finite_real(self):
        metrics = training.error_metrics(np.array([0.0, 0.5]), np.array([0.2, 0.3]))
        self.assertAlmostEqual(metrics['mae_hartree'], 0.2)
        self.assertAlmostEqual(metrics['mae_mHa'], 200)
        self.assertTrue(np.isrealobj(list(metrics.values())))
        self.assertTrue(np.all(np.isfinite(list(metrics.values()))))
        model = SimpleNamespace(predict_batch=lambda samples, _: torch.full(
            (len(samples),), 0.2, dtype=torch.float64))
        evaluation = training.evaluate(model, [sample('synthetic')], torch.zeros(1), batch_size=1)
        self.assertTrue(np.isrealobj(evaluation.predictions))
        self.assertTrue(np.all(np.isfinite(evaluation.predictions)))
        self.assertTrue(np.isrealobj(evaluation.targets))
        self.assertTrue(np.all(np.isfinite(evaluation.targets)))

    def test_evaluation_does_not_discard_complex_target(self):
        model = SimpleNamespace(predict_batch=lambda *_: torch.zeros(1, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'real values'):
            training.evaluate(model, [SimpleNamespace(sample_id='invalid', target_energy=1j)],
                              torch.zeros(1), batch_size=1)


if __name__ == '__main__':
    unittest.main()
