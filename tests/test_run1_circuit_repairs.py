"""Focused tape and FG regressions for the seven direct Run 1 repairs.

The 30 stored Cartesian geometries are read only to check descriptor ordering;
these tests do not select an experimental split or regenerate physical data.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pennylane as qml
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'codes'))
import circuit


def physical_sample(molecule='LiH'):
    n = circuit.MOLECULES[molecule].expected_active_ao_count
    return circuit.ProcessedSample(
        sample_id=f'{molecule}/synthetic', dataset_id='synthetic', molecule=molecule,
        geometry_index=1, scan_coordinate=1.0, scan_coordinate_name='synthetic distance',
        scan_coordinate_unit='Angstrom', basis='sto-3g',
        active_ao_indices=np.array(circuit.MOLECULES[molecule].expected_active_ao_indices),
        active_ao_labels=circuit.MOLECULES[molecule].expected_active_ao_labels,
        active_ao_order_source='synthetic', pii=np.linspace(.2, 1.0, n),
        fii=np.linspace(-.8, -.1, n), pij=np.full((n, n), .07),
        t_lowdin=np.full((n, n, n), .03), uhf_total_energy=-1.0,
        fci_total_energy=-1.1, fci_corr_target=-.1, target_energy=-.1,
        target_definition=circuit.TargetDefinition(), ump2_total_energy=-1.05,
        ump2_corr_energy=-.05, source_file=Path('synthetic.npz'),
        source_file_id='synthetic', source_sha256='synthetic',
        adapter_profile='synthetic', field_provenance=(),
    )


def mb_constants(label='MB-1'):
    return circuit.EncodingConstants(
        manifest_id='synthetic', molecule='LiH', model_label=label,
        training_sample_ids=('LiH/synthetic',), n_active=5,
        Pii_mean=.3, Pii_std=.7, Fii_mean=-.2, Fii_std=.9,
        Pij_abs95=.07, alpha=2.0,
        T_abs95=.03 if label == 'MB-3' else None,
        beta=3.0 if label == 'MB-3' else None,
        triple_contract=circuit.TRIPLE_CONTRACT if label == 'MB-3' else None,
    )


def fingerprint_constants(method, molecule='LiH'):
    n = circuit.feature_width(molecule, method)
    return circuit.FingerprintConstants(
        manifest_id='synthetic', molecule=molecule, model_label=method,
        training_sample_ids=(f'{molecule}/synthetic',), n_active=n,
        mu=(0.,) * n, sigma=(1.,) * n, sigma_safe=(1.,) * n,
        constant_mask=(False,) * n,
    )


def tape(node, *args):
    return qml.tape.make_qscript(node.func)(*args)


class CircuitTapeRepairTests(unittest.TestCase):
    def assert_ansatz(self, operations, n, phi):
        """Pin the original shared angles, ring direction, and boundary handling."""
        expected_count = 2 * n * (2 if n > 1 else 1)
        self.assertEqual(len(operations), expected_count)
        offset = 0
        for layer in range(2):
            for wire in range(n):
                operation = operations[offset]
                self.assertEqual(operation.name, 'RY')
                self.assertEqual(tuple(operation.wires), (wire,))
                self.assertEqual(float(operation.data[0]), float(phi[layer]))
                offset += 1
            if n > 1:
                for wire in range(n):
                    operation = operations[offset]
                    self.assertEqual(operation.name, 'CNOT')
                    self.assertEqual(tuple(operation.wires), (wire, (wire + 1) % n))
                    offset += 1

    def test_fg_fe_each_feature_has_one_data_ry_and_no_encoder_cnots(self):
        for method in ('FG', 'FE'):
            with self.subTest(method=method):
                constants = fingerprint_constants(method, 'H2O')
                model = circuit.FingerprintModel(constants)
                n = constants.n_active
                data_angles = torch.linspace(.13, .79, n, dtype=torch.float64)
                phi = torch.tensor([1.17, -1.39], dtype=torch.float64)
                with patch.object(circuit, 'shared_ansatz', wraps=circuit.shared_ansatz) as helper:
                    z = model.circuit_bank(n, data_angles, phi)
                    helper.assert_called_once_with(n, phi, num_layers=2)
                operations = tape(model.circuit_bank._nodes[n], data_angles, phi).operations
                self.assertEqual([op.name for op in operations].count('RY'), 3 * n)
                self.assertEqual([op.name for op in operations].count('CNOT'), 2 * n)
                for wire, operation in enumerate(operations[:n]):
                    self.assertEqual(operation.name, 'RY')
                    self.assertEqual(tuple(operation.wires), (wire,))
                    self.assertEqual(float(operation.data[0]), float(data_angles[wire]))
                self.assert_ansatz(operations[n:], n, phi)
                self.assertTrue(torch.isfinite(z).all())
                self.assertFalse(z.is_complex())

    def test_shared_ansatz_one_wire_has_no_self_cnot(self):
        bank = circuit.FingerprintCircuit()
        angles = torch.tensor([.3], dtype=torch.float64)
        phi = torch.tensor([.7, -.9], dtype=torch.float64)
        self.assertTrue(torch.isfinite(bank(1, angles, phi)).all())
        operations = tape(bank._nodes[1], angles, phi).operations
        self.assertEqual([op.name for op in operations], ['RY'] * 3)
        self.assert_ansatz(operations[1:], 1, phi)

    def test_mb1_omits_pairs_and_triples_and_uses_same_ansatz(self):
        model = circuit.CorrelationEnergyModel(mb_constants())
        sample = physical_sample()
        theta = torch.linspace(.1, .4, model.num_parameters, dtype=torch.float64)
        p, f, pairs, triple, abc, phi = model._quantum_arguments(sample, theta)
        self.assertIsNone(pairs)
        self.assertIsNone(triple)
        np.testing.assert_array_equal(sample.pij, np.full((5, 5), .07))
        np.testing.assert_allclose(p, (sample.pii - .3) / .7, rtol=0, atol=0)
        np.testing.assert_allclose(f, (sample.fii + .2) / .9, rtol=0, atol=0)
        node = model.circuit_bank.get_qnode(model.constants)
        with patch.object(circuit, 'shared_ansatz', wraps=circuit.shared_ansatz) as helper:
            operations = tape(node, p, f, pairs, abc, phi).operations
            helper.assert_called_once_with(5, phi, num_layers=2)
        self.assertNotIn('IsingZZ', [op.name for op in operations])
        self.assertTrue(all(len(op.wires) < 3 for op in operations))
        expected_gamma = .5 * math.pi * torch.tanh(abc[0] * p + abc[1] * f + abc[2])
        for i, operation in enumerate(operations[:5]):
            self.assertEqual(operation.name, 'RY')
            self.assertEqual(float(operation.data[0]), float(expected_gamma[i]))
        self.assert_ansatz(operations[5:], 5, phi)
        with self.assertRaisesRegex(ValueError, 'exclude pair descriptors'):
            model.circuit_bank(model.constants, p_diag_std=p, f_diag_std=f,
                               p_matrix=torch.zeros((5, 5)), t_tensor=None, abc=abc, phi=phi)

    def test_mb2_mb3_interactions_are_preserved(self):
        for label in ('MB-2', 'MB-3'):
            with self.subTest(label=label):
                model = circuit.CorrelationEnergyModel(mb_constants(label))
                theta = torch.full((model.num_parameters,), .1, dtype=torch.float64)
                p, f, pairs, triple, abc, phi = model._quantum_arguments(physical_sample(), theta)
                args = (p, f, pairs, triple, abc, phi) if label == 'MB-3' else (p, f, pairs, abc, phi)
                operations = tape(model.circuit_bank.get_qnode(model.constants), *args).operations
                names = [op.name for op in operations]
                self.assertEqual(names.count('IsingZZ'), 10)
                self.assertEqual(names.count('MultiRZ'), 10 if label == 'MB-3' else 0)
                self.assert_ansatz(operations[-20:], 5, phi)

    def test_model_features_predictions_and_gradients_are_finite_real(self):
        for method in ('FG', 'FE', 'MB-1'):
            with self.subTest(method=method):
                sample = physical_sample()
                if method == 'MB-1':
                    model = circuit.CorrelationEnergyModel(mb_constants())
                else:
                    constants = fingerprint_constants(method)
                    model = circuit.FingerprintModel(constants)
                    sample = circuit.FingerprintSample(**asdict(sample), fingerprint=np.linspace(.1, .8, constants.n_active))
                theta = torch.full((model.num_parameters,), .1, dtype=torch.float64, requires_grad=True)
                features = model.raw_readout_features(sample, theta)
                prediction = model.forward(sample, theta)
                self.assertEqual(prediction.shape, ())
                self.assertTrue(torch.isfinite(features).all() and torch.isfinite(prediction))
                self.assertFalse(features.is_complex() or prediction.is_complex())
                prediction.backward()
                self.assertTrue(torch.isfinite(theta.grad).all())
                self.assertFalse(theta.grad.is_complex())

    def test_fingerprint_angle_mapping_still_scales_clips_and_zeros_constant_columns(self):
        constants = circuit.FingerprintConstants(
            manifest_id='synthetic', molecule='LiH', model_label='FE',
            training_sample_ids=('synthetic',), n_active=5,
            mu=(1., 3., 2., 0., -1.), sigma=(1., 0., 3., 2., 4.),
            sigma_safe=(1., 1., 3., 2., 4.), constant_mask=(False, True, False, False, False),
        )
        features = np.array([100., 80., 5., 1., -100.])
        expected = np.array([math.pi, 0., math.pi / 2, math.pi / 4, -math.pi])
        np.testing.assert_array_equal(circuit.encode(features, constants), expected)


def old_angle(a, center, b):
    first, second = a - center, b - center
    return np.arccos(np.clip(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)), -1., 1.))


def old_dihedral(p0, p1, p2, p3):
    """Freeze the pre-repair signed dihedral formula independently of circuit.py."""
    b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2
    unit = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, unit) * unit
    w = b2 - np.dot(b2, unit) * unit
    return np.arctan2(np.dot(np.cross(unit, v), w), np.dot(v, w))


class FGOriginalFeatureTests(unittest.TestCase):
    def expected_existing_features(self, molecule, coordinates):
        kind = circuit.MOLECULES[molecule].fg_feature_kind
        d = lambda a, b: np.linalg.norm(coordinates[a] - coordinates[b])
        if kind == 'diatomic_2':
            return np.array([d(0, 1)]), np.array([])
        if kind == 'xh2_6':
            return np.array([d(0, 1), d(0, 2), d(1, 2)]), np.array([old_angle(coordinates[1], coordinates[0], coordinates[2])])
        if kind == 'nh3_7':
            return np.array([d(0, 1), d(0, 2), d(0, 3)]), np.array([old_angle(coordinates[1], coordinates[0], coordinates[2])])
        self.assertEqual(kind, 'h2o2_10')
        o1, o2, h1, h2 = coordinates
        return np.array([d(0, 1), d(0, 2), d(1, 3), d(0, 3), d(1, 2), d(2, 3)]), np.array([
            old_angle(o2, o1, h1), old_angle(o1, o2, h2), old_dihedral(h1, o1, o2, h2),
        ])

    def test_all_30_geometries_match_original_features_and_widths(self):
        for molecule, spec in circuit.MOLECULES.items():
            paths = sorted((PROJECT / 'results' / 'initialdata' / molecule / 'scan').glob('*/point.json'))
            self.assertEqual(len(paths), 30, molecule)
            fixed_names = circuit.fg_feature_names(molecule)
            self.assertEqual(len(fixed_names), len(set(fixed_names)))
            for path in paths:
                with self.subTest(molecule=molecule, geometry=path.parent.name):
                    record = json.loads(path.read_text())
                    # Reproduce the existing stored-atom-to-FG mapping without any geometry edits.
                    available = list(enumerate(record['symbols']))
                    canonical = []
                    for symbol in spec.electronic_atom_order:
                        atom = next(entry for entry in available if entry[1] == symbol)
                        canonical.append(atom[0])
                        available.remove(atom)
                    order = [canonical[i] for i in circuit.geometry_contract(molecule)['source_to_fg']]
                    coordinates = np.array(record['cartesian_A'])[order]
                    distances, other = self.expected_existing_features(molecule, coordinates)
                    actual = circuit.fg_features(molecule, coordinates)
                    n = len(distances)
                    np.testing.assert_array_equal(actual[:n], distances)
                    indices = {'diatomic_2': [0], 'xh2_6': [0, 1], 'nh3_7': [0, 1, 2], 'h2o2_10': [5]}[spec.fg_feature_kind]
                    end = n + len(indices)
                    np.testing.assert_array_equal(actual[n:end], 1.0 / distances[indices])
                    np.testing.assert_allclose(actual[end:], other, rtol=0, atol=1e-15)
                    names = circuit.fg_feature_names(molecule)
                    self.assertEqual(names, fixed_names)
                    self.assertEqual(names[n:end], tuple(f'1/{names[i]}' for i in indices))
                    self.assertEqual(len(names), {'diatomic_2': 2, 'xh2_6': 6, 'nh3_7': 7, 'h2o2_10': 10}[spec.fg_feature_kind])
                    self.assertEqual(actual.shape, (len(names),))
                    self.assertEqual(len(names), circuit.feature_width(molecule, 'FG'))
                    self.assertTrue(np.isrealobj(actual) and np.isfinite(actual).all())
                    self.assertFalse(actual.flags.writeable)

    def test_peroxide_keeps_two_original_angles_and_signed_dihedral(self):
        names = circuit.fg_feature_names('H2O2')
        self.assertEqual(names[7:], ('angle_O2_O1_H1', 'angle_O1_O2_H2', 'dihedral_H1_O1_O2_H2'))
        self.assertEqual(len(names), 10)

    def test_nonpositive_nonfinite_and_overflowing_distances_are_rejected(self):
        for invalid in (0., -1., float('inf'), float('nan'), np.nextafter(0., 1.)):
            with self.subTest(distance=invalid):
                with self.assertRaisesRegex(ValueError, 'finite|strictly positive'):
                    circuit._distances_with_reciprocals([1., invalid])
        for molecule, spec in circuit.MOLECULES.items():
            with self.subTest(molecule=molecule):
                with self.assertRaises(ValueError):
                    circuit.fg_features(molecule, np.zeros((len(spec.electronic_atom_order), 3)))


if __name__ == '__main__':
    unittest.main()
