"""Matched MB-1' control tests. No production training is performed."""
import contextlib
import copy
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tests'))
import circuit
import plot_run1_results as plots
import run1
import train
from test_run1_circuit_repairs import mb_constants, physical_sample
from test_run1_interface_repairs import synthetic_fixture

PRIME = circuit.MB1_PRIME
BASELINE = ROOT / 'results/run1/20260924_115423_466018'


def encoder_angles(model, p, f, coefficients, phi):
    node = model.circuit_bank.get_qnode(model.constants)
    tape = node.construct((p, f, None, coefficients, phi), {})
    return torch.stack([op.data[0] for op in tape.operations[:model.constants.n_active]])


class MB1PrimeEncoderTests(unittest.TestCase):
    def setUp(self):
        self.mb = circuit.construct_model(mb_constants(), 2)
        self.prime = circuit.construct_model(mb_constants(PRIME), 2)
        self.sample = physical_sample()
        self.p = torch.linspace(-1.1, .8, 5, dtype=torch.float64)
        self.f = torch.linspace(.9, -.6, 5, dtype=torch.float64)
        self.ac = torch.tensor([.37, -.12], dtype=torch.float64, requires_grad=True)
        self.phi = torch.tensor([.17, -.29], dtype=torch.float64)

    def test_exact_requested_encoder_formula_and_gradients(self):
        actual = encoder_angles(self.prime, self.p, self.f, self.ac, self.phi)
        expected = (math.pi / 2) * torch.tanh(self.ac[0] * self.p + self.ac[1])
        self.assertTrue(torch.equal(actual, expected))
        actual_grad = torch.autograd.grad(actual.sum(), self.ac)[0]
        expected_grad = torch.autograd.grad(expected.sum(), self.ac)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-15, atol=1e-15)

    def test_changing_f_tilde_cannot_change_angles_or_quantum_output(self):
        bank = self.prime.circuit_bank
        expected = encoder_angles(self.prime, self.p, self.f, self.ac, self.phi)
        def output(f):
            return bank(self.prime.constants, p_diag_std=self.p, f_diag_std=f,
                        p_matrix=None, t_tensor=None, abc=self.ac, phi=self.phi)
        expected_z = output(self.f)
        for f in (self.f * -900 + 72, torch.full_like(self.f, float('nan')), None):
            with self.subTest(f=f):
                self.assertTrue(torch.equal(expected, encoder_angles(self.prime, self.p, f, self.ac, self.phi)))
                self.assertTrue(torch.equal(expected_z, output(f)))
        theta = train.initialize_parameters(self.prime, [self.sample], seed=7)
        poisoned = replace(self.sample, fii=None, fij=None)
        self.assertTrue(torch.equal(self.prime(self.sample, theta), self.prime(poisoned, theta)))
        changed_scaler = replace(self.prime, constants=replace(
            self.prime.constants, Fii_mean=300., Fii_std=700.))
        self.assertTrue(torch.equal(self.prime(self.sample, theta), changed_scaler(self.sample, theta)))

    def test_changing_p_tilde_can_change_encoder_output(self):
        original = encoder_angles(self.prime, self.p, self.f, self.ac, self.phi)
        changed = encoder_angles(self.prime, self.p + .4, self.f, self.ac, self.phi)
        self.assertTrue(torch.all(original != changed))

    def test_mb1_original_formula_circuit_readout_and_gradients_are_unchanged(self):
        theta = torch.linspace(.1, .4, self.mb.num_parameters, dtype=torch.float64, requires_grad=True)
        p, f, pairs, triple, abc, phi = self.mb._quantum_arguments(self.sample, theta)
        expected_angles = (math.pi / 2) * torch.tanh(abc[0] * p + abc[1] * f + abc[2])
        self.assertTrue(torch.equal(encoder_angles(self.mb, p, f, abc, phi), expected_angles))
        self.assertFalse(torch.equal(encoder_angles(self.mb, p, f, abc, phi),
                                     encoder_angles(self.mb, p, f + .5, abc, phi)))
        self.assertIsNone(pairs)
        self.assertIsNone(triple)
        # Independent frozen expression for the original MB-1 circuit/readout.
        @qml.qnode(qml.device('default.qubit', wires=5, shots=None, seed=0),
                   interface='torch', diff_method='backprop')
        def original_node():
            for i in range(5):
                qml.RY(.5 * math.pi * qml.math.tanh(abc[0] * p[i] + abc[1] * f[i] + abc[2]), wires=i)
            for layer in range(2):
                for i in range(5):
                    qml.RY(phi[layer], wires=i)
                for i in range(5):
                    qml.CNOT(wires=[i, (i + 1) % 5])
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(5))
        z = torch.stack(original_node())
        features = torch.cat((z, z.new_zeros(3), torch.stack((z.mean(), (z ** 2).mean(), (z ** 3).mean()))))
        expected = theta[5] + features @ theta[6:]
        actual = self.mb(self.sample, theta)
        self.assertTrue(torch.equal(actual, expected))
        # The independent reference indexes abc per gate; the model unbinds it
        # once. Backward accumulation can differ at float64 rounding precision.
        torch.testing.assert_close(torch.autograd.grad(actual, theta)[0],
                                   torch.autograd.grad(expected, theta)[0],
                                   rtol=1e-14, atol=1e-15)

    def test_b_is_absent_and_exactly_one_encoder_parameter_is_removed(self):
        for spec in circuit.MOLECULES.values():
            n = spec.expected_active_ao_count
            mb = circuit.ParameterLayout(n, 2)
            prime = circuit.ParameterLayout(n, 2, include_fock=False)
            self.assertEqual(mb.encoder_names, ('a', 'b', 'c'))
            self.assertEqual(prime.encoder_names, ('a', 'c'))
            self.assertEqual(prime.names, tuple(name for name in mb.names if name != 'b'))
            self.assertEqual(mb.num_parameters, max(8, n) + 9)
            self.assertEqual(prime.num_parameters, mb.num_parameters - 1)
        theta = train.initialize_parameters(self.prime, [self.sample], seed=7)
        optimizer = train.make_optimizer(theta, model=self.prime)
        self.assertEqual(sum(p.numel() for g in optimizer.param_groups for p in g['params']), 16)
        self.assertEqual(self.prime.split_parameters(theta)[0].numel(), 2)

    def test_full_prediction_matches_mb1_with_b_zero_and_has_finite_gradients(self):
        mb = train.initialize_parameters(self.mb, [self.sample], seed=13)
        with torch.no_grad():
            mb[1] = 0
        prime = torch.cat((mb[:1], mb[2:])).detach().requires_grad_()
        angles, features, weights = [], [], []
        for model, theta in ((self.mb, mb), (self.prime, prime)):
            p, f, _, _, ac, phi = model._quantum_arguments(self.sample, theta)
            angles.append(encoder_angles(model, p, f, ac, phi))
            features.append(model.raw_readout_features(self.sample, theta))
            weights.append(model.split_parameters(theta)[2])
        # Keep the scientific inputs and circuit outputs bitwise checks.
        for a, b in (angles, features, weights):
            self.assertTrue(torch.equal(a, b))
        original = self.mb(self.sample, mb)
        control = self.prime(self.sample, prime)
        # Removing b shifts the readout slice in storage. Dot-product reduction
        # can round differently even with bitwise-identical features/weights.
        torch.testing.assert_close(original, control, rtol=1e-14, atol=1e-15)
        mb_grad = torch.autograd.grad(original, mb)[0]
        prime_grad = torch.autograd.grad(control, prime)[0]
        self.assertTrue(torch.isfinite(prime_grad).all())
        torch.testing.assert_close(prime_grad, torch.cat((mb_grad[:1], mb_grad[2:])), rtol=1e-14, atol=1e-14)

    def test_shared_initialization_optimizer_and_ansatz_match_every_seed(self):
        for seed in run1.SEEDS:
            mb = train.initialize_parameters(self.mb, [self.sample], seed=seed)
            prime = train.initialize_parameters(self.prime, [self.sample], seed=seed)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            old = .05 * torch.randn(self.mb.num_parameters, dtype=torch.float64, generator=generator)
            old[5] = self.sample.target_energy
            self.assertTrue(torch.equal(mb, old))
            self.assertTrue(torch.equal(prime, torch.cat((mb[:1], mb[2:]))))
            self.assertTrue(prime.is_leaf and prime.requires_grad)
        mb_opt = train.make_optimizer(mb, model=self.mb)
        prime_opt = train.make_optimizer(prime, model=self.prime)
        self.assertIs(type(mb_opt), type(prime_opt))
        self.assertEqual(mb_opt.defaults, prime_opt.defaults)
        tapes = []
        for model, theta in ((self.mb, mb), (self.prime, prime)):
            p, f, pairs, _, ac, phi = model._quantum_arguments(self.sample, theta)
            tapes.append(model.circuit_bank.get_qnode(model.constants).construct((p, f, pairs, ac, phi), {}))
        for a, b in zip(tapes[0].operations[5:], tapes[1].operations[5:], strict=True):
            self.assertEqual((a.name, a.wires), (b.name, b.wires))
            for x, y in zip(a.data, b.data, strict=True):
                self.assertTrue(torch.equal(x, y))
        self.assertEqual([str(m) for m in tapes[0].measurements], [str(m) for m in tapes[1].measurements])


class MatchedPreparationTests(unittest.TestCase):
    def test_every_molecule_keeps_splits_targets_p_scaling_and_parameter_counts(self):
        for molecule in circuit.MOLECULES:
            with self.subTest(molecule=molecule), contextlib.redirect_stdout(io.StringIO()):
                samples, manifest, metadata, _ = synthetic_fixture(molecule)
                prepared, _ = run1.prepare_runs(samples, manifest, metadata, methods=circuit.ONE_BODY_MODELS)
                mb, prime = (prepared[molecule, m] for m in circuit.ONE_BODY_MODELS)
                self.assertIs(type(mb[0]), type(prime[0]))
                self.assertEqual(asdict(mb[0].constants), dict(asdict(prime[0].constants), model_label='MB-1'))
                for first, second in zip(mb[1:], prime[1:], strict=True):
                    self.assertTrue(all(a is b for a, b in zip(first, second, strict=True)))
                self.assertEqual(mb[0].num_parameters, prime[0].num_parameters + 1)
                train_ids = set(prime[0].constants.training_sample_ids)
                changed = tuple(replace(s, pii=s.pii * 500, pij=s.pij * 500,
                                        fii=s.fii * 300, fij=s.fij * 300)
                                if s.sample_id not in train_ids else s for s in samples)
                after, _ = run1.prepare_runs(changed, manifest, metadata, methods=(PRIME,))
                self.assertEqual(prime[0].constants, after[molecule, PRIME][0].constants)
                matrix = np.concatenate([s.pii for s in prime[1]])
                self.assertEqual(prime[0].constants.Pii_mean, matrix.mean())
                self.assertEqual(prime[0].constants.Pii_std, matrix.std(ddof=0))


class MatchedBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.manifest, cls.metadata = run1.load_population(
            ROOT / 'results/descriptor', molecules=('CO',), split_protocol='15-15')
        with contextlib.redirect_stdout(io.StringIO()):
            cls.prepared, cls.audits = run1.prepare_runs(cls.samples, cls.manifest, cls.metadata, methods=(PRIME,))
        cls.population = [dict(cls.metadata[r.sample_id], split=r.split, retained_position=r.retained_position)
                          for r in cls.manifest.rows]

    def args(self, reference=BASELINE):
        return SimpleNamespace(match_mb1_run=reference, molecules=['CO'], epochs=500,
                               input_dir=ROOT / 'results/descriptor', split_protocol='15-15')

    def test_current_reference_has_the_exact_population_and_scalers(self):
        reference = run1.matched_mb1_reference(self.args(), self.prepared, self.population)
        self.assertEqual(sorted(reference[reference.split == 'test'].seed.tolist()), list(range(32)))
        self.assertEqual(self.prepared['CO', PRIME][0].num_parameters, 16)
        self.assertEqual(run1.METHODS, ('FG', 'FE', 'MB-1'))
        self.assertEqual(plots.format_run_label(PRIME, 'run1'), "MB1'")

    def test_mismatched_config_population_scaler_or_seeds_is_rejected(self):
        for key, value in (('epochs', 499), ('learning_rate', .03), ('seeds', [0])):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = json.loads((BASELINE / 'configuration.json').read_text())
                config[key] = value
                run1.write_json(root / 'configuration.json', config)
                with self.assertRaisesRegex(ValueError, key):
                    run1.matched_mb1_reference(self.args(root), self.prepared, self.population)
        population = copy.deepcopy(self.population)
        population[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'population'):
            run1.matched_mb1_reference(self.args(), self.prepared, population)
        prepared = dict(self.prepared)
        model, *groups = prepared['CO', PRIME]
        prepared['CO', PRIME] = (replace(model, constants=replace(model.constants, Pii_mean=99.)), *groups)
        with self.assertRaisesRegex(ValueError, 'preprocessing'):
            run1.matched_mb1_reference(self.args(), prepared, self.population)
        for name in ('missing_seed', 'duplicate_seed'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for filename in ('configuration.json', 'population.json', 'preprocessing.json'):
                    (root / filename).write_bytes((BASELINE / filename).read_bytes())
                frame = pd.read_csv(BASELINE / 'per_seed_metrics.csv')
                selected = (frame.model == 'MB-1') & (frame.seed == 31) & (frame.split == 'test')
                if name == 'missing_seed':
                    frame = frame[~selected]
                else:
                    frame.loc[selected, 'seed'] = 30
                frame.to_csv(root / 'per_seed_metrics.csv', index=False)
                with self.assertRaisesRegex(ValueError, 'seeds'):
                    run1.matched_mb1_reference(self.args(root), self.prepared, self.population)

    def test_launcher_only_dispatches_prime_and_writes_correct_mean_sd_delta(self):
        reference = run1.matched_mb1_reference(self.args(), self.prepared, self.population)
        # Artificial +2 mHa results are confined to a temporary unit-test output.
        def fake_train(output, molecule, method, seed, *args, **kwargs):
            self.assertEqual((molecule, method), ('CO', PRIME))
            self.assertEqual(kwargs, dict(num_epochs=500, checkpoint_selection='train'))
            return reference[reference.seed == seed].assign(model=PRIME, mae_mHa=lambda f: f.mae_mHa + 2).to_dict('records')
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder) / 'control'
            argv = ['--methods', PRIME, '--molecules', 'CO', '--split-protocol', '15-15',
                    '--input-dir', str(ROOT / 'results/descriptor'), '--epochs', '500',
                    '--match-mb1-run', str(BASELINE), '--output-dir', str(output)]
            with patch.object(run1, 'load_population', return_value=(self.samples, self.manifest, self.metadata)), \
                 patch.object(run1, 'prepare_runs', return_value=(self.prepared, self.audits)) as prepare, \
                 patch.object(run1, 'train_one', side_effect=fake_train) as trainer:
                run1.main(argv)
                self.assertEqual(prepare.call_args.kwargs['methods'], (PRIME,))
                self.assertEqual([c.args[3] for c in trainer.call_args_list], list(range(32)))
                table = pd.read_csv(output / 'mb1_prime_comparison.csv').iloc[0]
                scores = reference[reference.split == 'test'].mae_mHa
                self.assertAlmostEqual(table['MB-1 MAE mean'], scores.mean())
                self.assertAlmostEqual(table['MB-1 MAE SD'], scores.std(ddof=1))
                self.assertAlmostEqual(table["MB-1' MAE mean"], scores.mean() + 2)
                self.assertAlmostEqual(table["MB-1' MAE SD"], scores.std(ddof=1))
                self.assertAlmostEqual(table['ΔMAE'], 2)
                self.assertIn('mHa', (output / 'mb1_prime_comparison.md').read_text())
                trainer.reset_mock()
                with self.assertRaises(FileExistsError):
                    run1.main(argv)
                trainer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
