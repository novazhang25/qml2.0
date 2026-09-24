"""Active descriptor selection regressions; no electronic-structure solvers run.

The archived reference uniformly retains each atom's highest principal shell:
H 1s, second-period 2s/2p, and S 3s/3p. The revised descriptor-only rule
excludes heavy-atom 1s and retains S 2s/2p as well. H2S alone changes size;
neither descriptor selection rule freezes occupied MP2 molecular orbitals.
These tests compare both producer selectors. The Run 1 loader separately uses
the saved Ne-core-excluded six-AO H2S records, without changing either producer.
"""
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pyscf import gto

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import circuit
import descriptor
import run1
from methods41_reference import select_valence_aos as legacy_select_valence_aos


# Zero-based indices in the stored full-AO order, before Run 1's atom permutation.
RETAINED = {
    'LiH': (1, 2, 3, 4, 5),
    'HF': (0, 2, 3, 4, 5),
    'BeH2': (1, 2, 3, 4, 5, 6),
    'H2O': (1, 2, 3, 4, 5, 6),
    'H2S': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    'NH3': (1, 2, 3, 4, 5, 6, 7),
    'N2': (1, 2, 3, 4, 6, 7, 8, 9),
    'CO': (1, 2, 3, 4, 6, 7, 8, 9),
    'H2O2': (1, 2, 3, 4, 6, 7, 8, 9, 10, 11),
}
EXCLUDED = {
    'LiH': (0,), 'HF': (1,), 'BeH2': (0,), 'H2O': (0,), 'H2S': (0,),
    'NH3': (0,), 'N2': (0, 5), 'CO': (0, 5), 'H2O2': (0, 5),
}
LEGACY_RETAINED = {**RETAINED, 'H2S': (2, 6, 7, 8, 9, 10)}


def molecule_for(record):
    return gto.M(atom=list(zip(record['symbols'], record['cartesian_A'])),
                 basis=record['basis'], charge=record['charge'], spin=record['spin'],
                 unit='Angstrom', cart=False, symmetry=False, verbose=0)


class DescriptorAOSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = descriptor.load_records(ROOT / 'results/initialdata', descriptor.MOLECULES)

    def test_all_nine_molecules_all_thirty_geometries_have_exact_retained_indices(self):
        self.assertEqual(set(RETAINED), set(descriptor.MOLECULES))
        self.assertEqual(len(self.records), 270)
        counts, first_labels = {}, {}
        for record in self.records:
            name = record['molecule']
            with self.subTest(molecule=name, geometry=record['geometry_id']):
                mol = molecule_for(record)
                actual = descriptor.select_valence_aos(mol)
                v, c = actual['valence_indices'], actual['core_indices']
                np.testing.assert_array_equal(v, RETAINED[name])
                np.testing.assert_array_equal(c, EXCLUDED[name])
                self.assertEqual(sorted([*v, *c]), list(range(mol.nao_nr())))
                self.assertFalse(set(v) & set(c))
                self.assertEqual(actual['ao_selection_rule'].item(), descriptor.AO_SELECTION_RULE)
                for i, (atom, _, orbital, _) in enumerate(mol.ao_labels(fmt=False)):
                    excluded_1s = mol.atom_charge(atom) > 2 and orbital == '1s'
                    self.assertEqual(i in c, excluded_1s)
                    self.assertEqual(i in v, not excluded_1s)
                np.testing.assert_array_equal(actual['valence_labels'], actual['ao_labels'][v])
                np.testing.assert_array_equal(actual['core_labels'], actual['ao_labels'][c])
                labels = tuple(actual['valence_labels'])
                self.assertEqual(labels, first_labels.setdefault(name, labels))
                counts[name] = counts.get(name, 0) + 1
        self.assertEqual(counts, {name: 30 for name in RETAINED})

    def test_hydrogen_helium_and_higher_principal_shell_boundary(self):
        # Artificial AO metadata isolates the exact Z > 2 / principal == 1 rule,
        # including higher shells that are not present in this STO-3G dataset.
        entries = [(0, 'H', '1s', ''), (1, 'He', '1s', ''),
                   (2, 'Li', '1s', ''), (2, 'Li', '2s', ''),
                   (3, 'S', '1s', ''), (3, 'S', '2s', ''), (3, 'S', '2p', 'x'),
                   (3, 'S', '3s', ''), (3, 'S', '3p', 'x'),
                   (3, 'S', '4s', ''), (3, 'S', '10s', '')]
        mol = SimpleNamespace(
            ao_labels=lambda fmt=True: [f'{a} {s} {o}{m}' for a, s, o, m in entries] if fmt else entries,
            ao_loc_nr=lambda: np.arange(len(entries) + 1), nbas=len(entries),
            bas_atom=lambda shell: entries[shell][0], atom_charge=lambda atom: (1, 2, 3, 16)[atom],
            bas_angular=lambda shell: int(entries[shell][2].endswith('p')),
            bas_nctr=lambda shell: 1, bas_nprim=lambda shell: 3)
        selection = descriptor.select_valence_aos(mol)
        np.testing.assert_array_equal(selection['core_indices'], [2, 4])
        np.testing.assert_array_equal(selection['valence_indices'], [0, 1, 3, 5, 6, 7, 8, 9, 10])

    def test_uniform_legacy_rule_changes_only_h2s_in_actual_stored_ao_order(self):
        changed = set()
        for record in self.records:
            name = record['molecule']
            with self.subTest(molecule=name, geometry=record['geometry_id']):
                mol = molecule_for(record)
                with np.load(ROOT / 'results/descriptor' / name / f"{record['geometry_id']}.npz",
                             allow_pickle=False) as saved:
                    labels = tuple(' '.join(str(label).split()) for label in saved['ao_labels'])
                self.assertEqual(labels, tuple(' '.join(label.split()) for label in mol.ao_labels()))
                # Derive the uniform highest-shell selection from the stored AO
                # order, independently of hard-coded indices or sulfur branches.
                parsed = [label.split() for label in labels]
                highest = {}
                for atom, _, orbital in parsed:
                    principal = int(orbital[0])  # The actual STO-3G labels use shells 1--3.
                    highest[atom] = max(highest.get(atom, 0), principal)
                expected_legacy = tuple(i for i, (atom, _, orbital) in enumerate(parsed)
                                        if int(orbital[0]) == highest[atom])
                legacy = legacy_select_valence_aos(mol)['valence_indices']
                revised = descriptor.select_valence_aos(mol)['valence_indices']
                np.testing.assert_array_equal(legacy, expected_legacy)
                np.testing.assert_array_equal(legacy, LEGACY_RETAINED[name])
                np.testing.assert_array_equal(revised, RETAINED[name])
                if not np.array_equal(legacy, revised):
                    changed.add(name)
                    self.assertEqual((len(legacy), len(revised)), (6, 10))
                else:
                    self.assertNotEqual(name, 'H2S')
        self.assertEqual(changed, {'H2S'})

    def test_h2s_retains_stored_sulfur_2s_and_all_three_2p_aos(self):
        for record in self.records:
            if record['molecule'] != 'H2S':
                continue
            with self.subTest(geometry=record['geometry_id']):
                with np.load(ROOT / 'results/descriptor/H2S' / f"{record['geometry_id']}.npz",
                             allow_pickle=False) as saved:
                    parsed = [str(label).split() for label in saved['ao_labels']]
                mol = molecule_for(record)
                current = descriptor.select_valence_aos(mol)
                legacy = legacy_select_valence_aos(mol)
                v, c = current['valence_indices'], current['core_indices']
                sulfur_2sp = {i for i, (_, element, orbital) in enumerate(parsed)
                              if element == 'S' and orbital in ('2s', '2px', '2py', '2pz')}
                self.assertEqual(len(sulfur_2sp), 4)
                self.assertEqual({parsed[i][2] for i in sulfur_2sp}, {'2s', '2px', '2py', '2pz'})
                self.assertTrue(sulfur_2sp.issubset(v))
                self.assertEqual(set(v) - set(legacy['valence_indices']), sulfur_2sp)
                self.assertEqual([parsed[i][1:] for i in c], [['S', '1s']])
                self.assertEqual({parsed[i][2] for i in legacy['core_indices']},
                                 {'1s', '2s', '2px', '2py', '2pz'})
                self.assertEqual(len(v), 10)

    def test_all_stored_hydrogen_1s_aos_are_retained(self):
        for record in self.records:
            with self.subTest(molecule=record['molecule'], geometry=record['geometry_id']):
                mol = molecule_for(record)
                current = descriptor.select_valence_aos(mol)
                hydrogens = {i for i, (_, element, orbital, _) in enumerate(mol.ao_labels(fmt=False))
                             if element == 'H' and orbital == '1s'}
                self.assertEqual(len(hydrogens), record['symbols'].count('H'))
                self.assertTrue(hydrogens.issubset(current['valence_indices']))
                self.assertFalse(hydrogens.intersection(current['core_indices']))

    def test_all_nine_descriptor_dimensions_and_existing_t_axis_mapping(self):
        for name, retained in RETAINED.items():
            with self.subTest(molecule=name):
                full_n, n = len(retained) + len(EXCLUDED[name]), len(retained)
                p = np.arange(full_n ** 2, dtype=float).reshape(full_n, full_n)
                f = -p / 3
                raw = np.arange(full_n ** 4, dtype=float).reshape((full_n,) * 4) - 1000
                before = raw.copy()
                actual = descriptor.extract_descriptors(p, f, raw, retained)
                pairs = np.asarray(list(itertools.combinations(range(n), 2)))
                triples = np.asarray(list(itertools.combinations(range(n), 3)))
                self.assertEqual(actual['P_mu'].shape, (n,))
                self.assertEqual(actual['F_mu'].shape, (n,))
                self.assertEqual(actual['P_munu'].shape, (n * (n - 1) // 2,))
                self.assertEqual(actual['T_full'].shape, (n, n, n))
                self.assertEqual(actual['T_munulambda'].shape, (n * (n - 1) * (n - 2) // 6,))
                np.testing.assert_array_equal(actual['P_mu'], np.diag(p)[list(retained)])
                np.testing.assert_array_equal(actual['F_mu'], np.diag(f)[list(retained)])
                np.testing.assert_array_equal(actual['pair_indices'], pairs)
                np.testing.assert_array_equal(actual['triple_indices'], triples)
                expected = np.empty((n, n, n))
                for i, j, k in np.ndindex(expected.shape):
                    expected[i, j, k] = raw[retained[i], retained[k], retained[j], retained[k]]
                np.testing.assert_array_equal(actual['T_full'], expected)
                np.testing.assert_array_equal(actual['T_munulambda'], expected[tuple(triples.T)])
                np.testing.assert_array_equal(raw, before)
                for value in actual.values():
                    self.assertTrue(np.isrealobj(value) and np.isfinite(value).all())

    def test_run1_adapter_and_fe_mb_dimensions_match_saved_nine_molecule_subspaces(self):
        for record in self.records:
            if record['geometry_id'] != '001':
                continue
            name = record['molecule']
            with self.subTest(molecule=name):
                mol = molecule_for(record)
                with np.load(ROOT / 'results/descriptor' / name / '001.npz', allow_pickle=False) as saved:
                    raw = {key: saved[key] for key in
                           ('ao_labels', 'valence_indices', 'core_indices', 'valence_labels', 'symbols')}
                selected, valence, labels, _ = run1.ao_selection(raw, name, Path('synthetic.npz'))
                expected_order = (5, 1, 2, 3, 4, 6) if name == 'BeH2' else LEGACY_RETAINED[name]
                np.testing.assert_array_equal(selected, expected_order)
                np.testing.assert_array_equal(valence, LEGACY_RETAINED[name])
                spec = circuit.MOLECULES[name]
                self.assertEqual(spec.expected_active_ao_count, len(LEGACY_RETAINED[name]))
                self.assertEqual(spec.expected_core_count, mol.nao_nr() - len(LEGACY_RETAINED[name]))
                self.assertEqual(spec.expected_active_ao_count + spec.expected_core_count, mol.nao_nr())
                self.assertEqual(labels, spec.expected_active_ao_labels)
                self.assertEqual(circuit.feature_width(name, 'FE'), len(LEGACY_RETAINED[name]))

    def test_calculation_transforms_full_h2s_space_before_selection_without_solver_calls(self):
        record = next(r for r in self.records if r['molecule'] == 'H2S' and r['geometry_id'] == '001')
        with np.load(ROOT / 'results/descriptor/H2S/001.npz', allow_pickle=False) as saved:
            source = {key: saved[key] for key in saved.files}
        mol = molecule_for(record)
        mf, pt = Mock(), Mock()
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = source['mo_coeff'], source['mo_occ'], source['mo_energy']
        mf.e_tot, mf.converged = float(source['E_RHF_calculated']), True
        mf.canonicalize.return_value = mf.mo_energy, mf.mo_coeff
        mf.get_ovlp.return_value = source['S_AO']
        mf.make_rdm1.return_value = source['P_AO']
        mf.get_fock.return_value = source['F_AO']
        mf.get_hcore.return_value = source['hcore_AO']
        pt.kernel.return_value = float(source['E_MP2_correlation']), source['mp2_t2']
        pt.make_rdm1.side_effect = lambda ao_repr=False: source['P_MP2_AO_pyscf' if ao_repr else 'P_MP2_MO_pyscf']
        pt.make_rdm2.side_effect = lambda ao_repr=False: source['Gamma_MP2_AO_pyscf' if ao_repr else 'Gamma_MP2_MO_pyscf']
        rdm_keys = ['P_MP2_MO_consistent', 'Gamma_MP2_MO_consistent'] + [
            f'{kind}_MO_order{order}' for kind in ('P', 'Gamma') for order in range(3)]
        events = []
        original_factors, original_transform = descriptor.lowdin_factors, descriptor.transform_rank4
        original_select, original_extract = descriptor.select_valence_aos, descriptor.extract_descriptors

        def factors(s):
            self.assertEqual(s.shape, (11, 11))
            events.append('full_lowdin_factors')
            return original_factors(s)

        def transform(tensor, a):
            self.assertEqual(tensor.shape, (11,) * 4)
            self.assertEqual(a.shape, (11, 11))
            events.append('full_rank4_transform')
            return original_transform(tensor, a)

        def select(molecule):
            self.assertEqual(events[-2:], ['full_lowdin_factors', 'full_rank4_transform'])
            events.append('select')
            return original_select(molecule)

        def extract(p, f, raw, retained):
            self.assertEqual(events[-1], 'select')
            self.assertEqual(p.shape, (11, 11))
            self.assertEqual(f.shape, (11, 11))
            self.assertEqual(raw.shape, (11,) * 4)
            np.testing.assert_array_equal(retained, RETAINED['H2S'])
            events.append('extract')
            return original_extract(p, f, raw, retained)

        with patch('pyscf.gto.M', return_value=mol), \
             patch.object(mol, 'intor', return_value=source['eri_AO']), \
             patch('pyscf.scf.hf.RHF', return_value=mf), \
             patch('pyscf.mp.mp2.RMP2', return_value=pt) as mp2_factory, \
             patch.object(descriptor, 'build_consistent_rdms', return_value={k: source[k] for k in rdm_keys}) as rdms, \
             patch.object(descriptor, 'lowdin_factors', side_effect=factors), \
             patch.object(descriptor, 'transform_rank4', side_effect=transform), \
             patch.object(descriptor, 'select_valence_aos', side_effect=select), \
             patch.object(descriptor, 'extract_descriptors', side_effect=extract):
            actual = descriptor.calculate_record(record)
        mp2_factory.assert_called_once_with(mf, frozen=0)
        rdms.assert_called_once_with(source['mp2_t2'], 11, mol.nelectron)
        self.assertEqual(events[-2:], ['select', 'extract'])
        for key in ('S_half', 'S_minus_half', 'P_L', 'F_L', 'Lambda_L',
                    'P_MP2_AO_consistent', 'Gamma_MP2_AO_consistent', 'Lambda_HFref_AO'):
            np.testing.assert_allclose(actual[key], source[key], atol=1e-12, rtol=0, err_msg=key)
        for key in ('E_MP2_correlation', 'E_FCI', 'E_RHF_input', 'input_fci_frozen_core',
                    'mp2_t2', 'frozen_core', 'mp2_frozen_core', 'rdm_frozen_core'):
            np.testing.assert_array_equal(actual[key], source[key], err_msg=key)
        self.assertEqual(actual['P_mu'].shape, (10,))
        self.assertEqual(actual['T_full'].shape, (10, 10, 10))

    def test_resume_cannot_relabel_legacy_ao_selection_as_1s_only(self):
        record = self.records[0]
        for selection_rule in (None, 'highest-principal-shell'):
            with self.subTest(rule=selection_rule), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                manifest = dict(t_descriptor_version=2, t_axis_mapping=descriptor.T_AXIS_MAPPING,
                                records=[dict(molecule=record['molecule'], geometry_id=record['geometry_id'],
                                              record='legacy.npz', error='')])
                if selection_rule is not None:
                    manifest['ao_selection_rule'] = selection_rule
                path = output / 'manifest.json'
                path.write_text(json.dumps(manifest))
                before = path.read_bytes()
                error = io.StringIO()
                with patch.object(descriptor, 'calculate_record') as calculate, \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                    code = descriptor.main(['--molecules', record['molecule'], '--geometry-id', record['geometry_id'],
                                            '--resume', '--output-dir', str(output)])
                self.assertEqual(code, 1)
                self.assertIn('older or unknown post-Lowdin AO selection', error.getvalue())
                self.assertEqual(path.read_bytes(), before)
                calculate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
