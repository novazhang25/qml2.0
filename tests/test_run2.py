"""Run2 circuit, matched initialization, and training state regressions.

These synthetic tests exercise the existing Run1 optimizer/checkpoint workflow;
they never launch production training or regenerate electronic descriptors.
"""
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pennylane as qml
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'codes'))
sys.path.insert(0, str(PROJECT / 'tests'))
import circuit
import run1
import run2_model
import train
from test_run1_circuit_repairs import mb_constants, physical_sample, tape


def pair_sample(molecule='LiH'):
    """Use unequal, nonzero raw pair descriptors to expose all pair weights."""
    sample = physical_sample(molecule)
    n = len(sample.pii)
    pairs = circuit.pair_addresses(n)
    density = np.diag(sample.pii)
    for index, (mu, nu) in enumerate(pairs):
        density[mu, nu] = density[nu, mu] = .07 + .013 * index
    sample = replace(sample, pij=density, t_lowdin=None)
    return run2_model.PairSample(
        **vars(sample), lambda_pairs=np.linspace(.18, .41, len(pairs)))


def pair_model(method='MB2', molecule='LiH'):
    n = circuit.MOLECULES[molecule].expected_active_ao_count
    constants = replace(mb_constants(), molecule=molecule, n_active=n,
                        training_sample_ids=(f'{molecule}/synthetic',))
    return run2_model.PairEnergyModel(constants=constants, method=method)


def operations(model, sample, theta):
    return tape(model.qnode, *model.pair_quantum_arguments(sample, theta)).operations


class PairCircuitTests(unittest.TestCase):
    def assert_same_operations(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected, strict=True):
            self.assertEqual(left.name, right.name)
            self.assertEqual(tuple(left.wires), tuple(right.wires))
            self.assertEqual(len(left.data), len(right.data))
            for left_angle, right_angle in zip(left.data, right.data, strict=True):
                self.assertTrue(torch.equal(torch.as_tensor(left_angle),
                                            torch.as_tensor(right_angle)))

    def test_original_mb1_encoder_and_two_layer_hea_are_exactly_preserved(self):
        sample = pair_sample()
        original = circuit.CorrelationEnergyModel(mb_constants())
        n = original.constants.n_active
        for method in run2_model.METHODS:
            with self.subTest(method=method):
                model = pair_model(method)
                theta = torch.linspace(.12, .42, model.num_parameters, dtype=torch.float64)
                base_theta = theta[:original.num_parameters]
                p, f, pairs, triple, abc, phi = original._quantum_arguments(sample, base_theta)
                base_ops = tape(original.circuit_bank.get_qnode(original.constants),
                                p, f, pairs, abc, phi).operations
                actual = operations(model, sample, theta)
                pair_count = n * (n - 1) // 2
                self.assertEqual([op.name for op in actual],
                                 ['RY'] * n + ['IsingZZ'] * pair_count
                                 + (['RY'] * n + ['CNOT'] * n) * 2)
                self.assert_same_operations(actual[:n], base_ops[:n])
                self.assert_same_operations(actual[n + pair_count:], base_ops[n:])
                expected_gamma = (math.pi / 2) * torch.tanh(
                    abc[0] * p + abc[1] * f + abc[2])
                self.assertTrue(torch.equal(torch.stack([op.data[0] for op in actual[:n]]),
                                            expected_gamma))
                for left, right in zip(model.split_parameters(theta),
                                       original.split_parameters(base_theta), strict=True):
                    self.assertTrue(torch.equal(left, right))
                # The original padded Z vector, invariant moments and scalar
                # readout must remain the same for any supplied Z features.
                z = torch.linspace(-.7, .8, n, dtype=torch.float64)
                features = original.readout_features(z)
                self.assertTrue(torch.equal(model.readout_features(z), features))
                self.assertTrue(torch.equal(model.apply_readout(features, theta),
                                            original.apply_readout(features, base_theta)))

    def test_pair_angles_use_raw_entries_and_gate_has_exact_half_angle_exponent(self):
        sample = pair_sample()
        n = len(sample.pii)
        pairs = circuit.pair_addresses(n)
        zz_eigenvalues = torch.tensor([1., -1., -1., 1.], dtype=torch.complex128)
        for method in run2_model.METHODS:
            with self.subTest(method=method):
                model = pair_model(method)
                theta = torch.linspace(-.37, .43, model.num_parameters, dtype=torch.float64)
                a2 = theta[model.layout.names.index('a2')]
                b2 = theta[model.layout.names.index('b2')] if method == 'MB2' else 0.
                pair_ops = operations(model, sample, theta)[n:n + len(pairs)]
                self.assertEqual([tuple(op.wires) for op in pair_ops], list(pairs))
                for index, ((mu, nu), operation) in enumerate(zip(pairs, pair_ops, strict=True)):
                    expected_angle = (math.pi / 2) * torch.tanh(
                        a2 * sample.pij[mu, nu] + b2 * sample.lambda_pairs[index])
                    torch.testing.assert_close(operation.data[0], expected_angle, rtol=0, atol=0)
                    # ZZ is diagonal, so this is its exact spectral exponential
                    # without a numerical matrix-exponential approximation.
                    expected_matrix = torch.diag(torch.exp(-.5j * expected_angle * zz_eigenvalues))
                    torch.testing.assert_close(qml.matrix(operation), expected_matrix,
                                               rtol=1e-14, atol=1e-14)

    def test_zero_pair_inputs_produce_identity_and_recover_mb1_predictions(self):
        sample = pair_sample()
        sample = replace(sample, pij=np.zeros_like(sample.pij),
                         lambda_pairs=np.zeros_like(sample.lambda_pairs))
        original = circuit.CorrelationEnergyModel(mb_constants())
        n = len(sample.pii)
        pair_count = n * (n - 1) // 2
        for method in run2_model.METHODS:
            with self.subTest(method=method):
                model = pair_model(method)
                theta = torch.linspace(.14, .39, model.num_parameters, dtype=torch.float64)
                for operation in operations(model, sample, theta)[n:n + pair_count]:
                    self.assertEqual(operation.data[0].item(), 0.)
                    torch.testing.assert_close(qml.matrix(operation),
                                               torch.eye(4, dtype=torch.complex128), rtol=0, atol=0)
                torch.testing.assert_close(model(sample, theta),
                                           original(sample, theta[:original.num_parameters]),
                                           rtol=0, atol=2e-15)

    def test_b2_zero_recovers_prime_for_identical_common_parameters(self):
        full, prime = pair_model(), pair_model('MB2_prime')
        sample = pair_sample()
        theta = torch.linspace(-.31, .44, full.num_parameters, dtype=torch.float64)
        theta[-1] = 0.
        prime_theta = theta[:-1].clone()
        self.assert_same_operations(operations(full, sample, theta),
                                    operations(prime, sample, prime_theta))
        self.assertTrue(torch.equal(full.quantum_features(sample, theta),
                                    prime.quantum_features(sample, prime_theta)))
        self.assertTrue(torch.equal(full.predict_batch([sample], theta),
                                    prime.predict_batch([sample], prime_theta)))

    def test_prime_has_no_b2_and_never_needs_cumulant_or_triple_inputs(self):
        model = pair_model('MB2_prime')
        sample = pair_sample()
        theta = train.initialize_parameters(model, [sample], seed=7)
        self.assertNotIn('b2', model.layout.names)
        self.assertNotIn('c2', model.layout.names)
        expected = model(sample, theta)
        for changed in (replace(sample, lambda_pairs=None),
                        replace(sample, lambda_pairs=sample.lambda_pairs * -173 + 9),
                        replace(sample, t_lowdin=np.full((5, 5, 5), np.nan))):
            with self.subTest(cumulant=changed.lambda_pairs is not None):
                self.assertTrue(torch.equal(model(changed, theta), expected))


class PairParameterTests(unittest.TestCase):
    def test_counts_and_original_parameter_order_for_every_molecule(self):
        for molecule in circuit.MOLECULES:
            n = circuit.MOLECULES[molecule].expected_active_ao_count
            base = circuit.ParameterLayout(n, 2)
            for method, extra in (('MB2', ('a2', 'b2')), ('MB2_prime', ('a2',))):
                with self.subTest(molecule=molecule, method=method):
                    model = pair_model(method, molecule)
                    self.assertEqual(model.num_parameters, base.num_parameters + len(extra))
                    self.assertEqual(model.layout.names, base.names + extra)
                    self.assertEqual(model.layout.bias_index, base.bias_index)

    def test_all_run1_seeds_preserve_common_draws_including_a2(self):
        for molecule in circuit.MOLECULES:
            sample = pair_sample(molecule)
            full = pair_model('MB2', molecule)
            prime = pair_model('MB2_prime', molecule)
            original = circuit.CorrelationEnergyModel(full.constants)
            for seed in run1.SEEDS:
                with self.subTest(molecule=molecule, seed=seed):
                    base_theta = train.initialize_parameters(original, [sample], seed=seed)
                    full_theta = train.initialize_parameters(full, [sample], seed=seed)
                    prime_theta = train.initialize_parameters(prime, [sample], seed=seed)
                    self.assertTrue(torch.equal(full_theta[:original.num_parameters], base_theta))
                    self.assertTrue(torch.equal(prime_theta[:original.num_parameters], base_theta))
                    self.assertTrue(torch.equal(full_theta[:-1], prime_theta))
                    self.assertEqual(full_theta[full.layout.bias_index].item(), sample.target_energy)

    def test_pair_weights_receive_finite_gradients_and_adam_updates(self):
        sample = pair_sample()
        for method in run2_model.METHODS:
            with self.subTest(method=method):
                model = pair_model(method)
                theta = torch.linspace(.1, .4, model.num_parameters,
                                       dtype=torch.float64, requires_grad=True)
                original = theta.detach().clone()
                optimizer = train.make_optimizer(theta, model=model)
                self.assertIs(optimizer.param_groups[0]['params'][0], theta)
                loss = train.optimizer_step(model, [sample], theta, optimizer, epoch=1)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(theta.grad).all())
                for name in ('a2', 'b2') if method == 'MB2' else ('a2',):
                    index = model.layout.names.index(name)
                    self.assertNotEqual(theta.grad[index].item(), 0.)
                    self.assertNotEqual(theta[index].item(), original[index].item())
                    state = optimizer.state[theta]
                    self.assertEqual(state['exp_avg'].shape, theta.shape)
                    self.assertNotEqual(state['exp_avg'][index].item(), 0.)

    def test_checkpoint_restores_all_weights_and_reloaded_predictions(self):
        sample = pair_sample()
        for method in run2_model.METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                model = pair_model(method)
                theta = train.initialize_parameters(model, [sample], seed=11)
                optimizer = train.make_optimizer(theta, model=model)
                train.optimizer_step(model, [sample], theta, optimizer, epoch=1)
                saved = theta.detach().clone()
                prediction = model(sample, theta).detach().clone()
                best = train.BestCheckpoint(checkpoint_selection='train')
                best.consider(epoch=1, train_mae=.1, theta=theta, optimizer=optimizer)
                saved_moment = optimizer.state[theta]['exp_avg'].clone()
                train.optimizer_step(model, [sample], theta, optimizer, epoch=2)
                best.consider(epoch=2, train_mae=.2, theta=theta, optimizer=optimizer)
                with torch.no_grad():
                    theta.add_(17)
                checkpoint = best.restore(theta)
                self.assertEqual(checkpoint.epoch, 1)
                self.assertTrue(torch.equal(theta, saved))
                self.assertTrue(torch.equal(model(sample, theta), prediction))
                optimizer.load_state_dict(checkpoint.optimizer_state)
                self.assertTrue(torch.equal(optimizer.state[theta]['exp_avg'], saved_moment))
                path = Path(folder) / 'best_theta.npy'
                np.save(path, checkpoint.theta.numpy(), allow_pickle=False)
                reloaded = pair_model(method).load_theta(path)
                self.assertTrue(torch.equal(reloaded, saved))
                self.assertTrue(torch.equal(model(sample, reloaded), prediction))

    def test_training_uses_matched_minibatches_despite_extra_b2(self):
        original = pair_sample()
        samples = [replace(original, sample_id=f'LiH/synthetic-{i}',
                           target_energy=-.1 - .01 * i) for i in range(5)]
        orders = {}
        for method in run2_model.METHODS:
            events = []
            result = train.train(
                pair_model(method), samples, (), seed=13, num_epochs=2,
                batch_size=3, learning_rate=run1.LEARNING_RATE,
                checkpoint_selection='train',
                on_update=lambda epoch, batch, *_: events.append(
                    (epoch, tuple(sample.sample_id for sample in batch))))
            self.assertEqual(result.update_count, 4)
            self.assertTrue(torch.isfinite(result.theta).all())
            self.assertTrue(torch.equal(result.theta, result.checkpoint.theta))
            orders[method] = events
        self.assertEqual(orders['MB2'], orders['MB2_prime'])
        expected_order = train.SampleOrder(13)
        expected = [(epoch, tuple(sample.sample_id for sample in batch))
                    for epoch in (1, 2)
                    for batch in train.iter_batches(samples, 3, expected_order.permutation(5))]
        self.assertEqual(orders['MB2'], expected)


if __name__ == '__main__':
    unittest.main()
