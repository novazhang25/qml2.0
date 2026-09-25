#!/usr/bin/env python3
"""Run 1 entry point: FG, density-spectrum FE, Fock-spectrum FE_prime, and MB-1.

This file loads and validates full-AO records, selects final descriptors,
prepares matched populations, and writes results. circuit.py contains copied
feature/model code; train.py contains copied optimization/evaluation code.
No canonical_pipeline imports or electronic-structure calculations are used.

Data flow: full-AO Lowdin validation -> final valence selection -> train-only
scalers -> one-molecule seed-0 smoke gate -> remaining requested seed runs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from circuit import (
    MOLECULES, ProcessedSample, TargetDefinition, FingerprintSample, ParameterLayout,
    MB1_PRIME, ONE_BODY_MODELS,
    compute_encoding_constants, standardize_diagonal, fg_features, geometry_contract,
    feature_width, fg_feature_names, fit_constants, construct_model,
)
from train import compute_statistics, evaluate, initialize_parameters, train

PROJECT = Path(__file__).resolve().parents[1]

# Data loading: validate full-AO transformations before final valence selection

SPLIT_RULE = 'explicit per-molecule train/validation/test geometry IDs from supplied split manifest'
SPLITS = ('train', 'validation', 'test')
SPLIT_PROTOCOLS = {'manifest': SPLITS, '15-15': ('train', 'test')}
LEGACY_SPLIT_RULE = 'all 30 geometries: odd geometry IDs train (15), even geometry IDs test (15); no validation set'

def validate_split_manifest(value, geometry_ids_by_molecule, *, molecules=None, split_protocol='manifest'):
    """Validate explicit geometry IDs without choosing a split or changing its membership."""
    split_names = SPLIT_PROTOCOLS[split_protocol]
    if not isinstance(value, dict) or not value:
        raise ValueError('Split manifest must be a nonempty object keyed by molecule')
    selected = tuple(geometry_ids_by_molecule if molecules is None else molecules)
    missing_molecules = set(selected) - set(value)
    unknown_molecules = set(value) - set(geometry_ids_by_molecule)
    if missing_molecules or unknown_molecules:
        raise ValueError(f'Split manifest molecule mismatch: missing={sorted(missing_molecules)}, unknown={sorted(unknown_molecules)}')
    validated = {}
    for molecule, groups in value.items():
        inventory = tuple(geometry_ids_by_molecule[molecule])
        if len(inventory) != 30 or len(set(inventory)) != 30:
            raise ValueError(f'{molecule}: exactly 30 unique geometry IDs are required')
        if not isinstance(groups, dict) or set(groups) != set(split_names):
            raise ValueError(f'{molecule}: exactly {", ".join(split_names)} groups are required')
        seen, partitions = (set(), {})
        for split in split_names:
            ids = groups[split]
            if not isinstance(ids, (list, tuple)) or not ids or not all(isinstance(i, str) and i for i in ids):
                raise ValueError(f'{molecule}/{split}: a nonempty list of string geometry IDs is required')
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError(f'{molecule}/{split}: duplicate geometry ID within or across splits')
            unknown = set(ids) - set(inventory)
            if unknown:
                raise ValueError(f'{molecule}/{split}: unknown geometry IDs {sorted(unknown)}')
            seen.update(ids)
            partitions[split] = tuple(ids)
        if seen != set(inventory):
            raise ValueError(f'{molecule}: missing geometry IDs {sorted(set(inventory) - seen)}')
        validated[molecule] = partitions
    return validated

def read_split_manifest(path):
    """Read user-supplied IDs and reject duplicate JSON keys instead of overwriting them."""
    if path is None:
        raise ValueError('An explicit --split-manifest is required. Exact manuscript split IDs are unavailable; the 32-seed benchmark is blocked without a real split manifest. No split will be generated.')

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate split manifest key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)

def population_manifest(samples, splits, *, source, split_protocol='manifest'):
    """Attach validated memberships to the existing scan order; never infer memberships."""
    inventory = {}
    for sample in samples:
        inventory.setdefault(sample.molecule, []).append(f'{sample.geometry_index:03d}')
    selected = {m: splits[m] for m in inventory}
    validated = validate_split_manifest(selected, inventory, split_protocol=split_protocol)
    assignments = {f'{m}/{i}': split for m, groups in validated.items()
                   for split, ids in groups.items() for i in ids}
    if len({s.sample_id for s in samples}) != len(samples) or set(assignments) != {s.sample_id for s in samples}:
        raise ValueError('Sample identities must match the explicit split geometry IDs exactly once')
    positions = retained_positions(samples, set(assignments))
    rows = tuple(SimpleNamespace(sample_id=s.sample_id, molecule=s.molecule,
        geometry_index=s.geometry_index, scan_coordinate=s.scan_coordinate,
        scan_coordinate_unit=s.scan_coordinate_unit, source_file_id=s.source_file_id,
        source_sha256=s.source_sha256, included=True,
        retained_position=positions[s.sample_id], split=assignments[s.sample_id]) for s in samples)
    return SimpleNamespace(manifest_id=DATASET_ID, dataset_id=DATASET_ID,
        target_definition=TARGET, rows=rows,
        split_rule=LEGACY_SPLIT_RULE if split_protocol == '15-15' else SPLIT_RULE,
        split_protocol=split_protocol, split_manifest_source=str(source))

def retained_positions(samples, included_ids):
    """Preserve scan ordering independently of explicit split memberships."""
    positions = {}
    for molecule in sorted({s.molecule for s in samples}):
        selected = sorted((s for s in samples if s.molecule == molecule and s.sample_id in included_ids), key=lambda s: (s.scan_coordinate, s.geometry_index))
        positions.update({s.sample_id: i for i, s in enumerate(selected)})
    return positions

def transform_rank4(tensor, A):
    """Verify the saved cumulant transformation with all four indices in full AO space."""
    return np.einsum('ap,bq,cr,ds,pqrs->abcd', A, A, A, A, tensor, optimize=True)
DATASET_ID = 'run1-methods41-fullspace'
SCHEMA = 'descriptor-v1-calculation-only'
SCHEMAS = (SCHEMA, 'methods41-v3-fullspace')
TARGET = TargetDefinition(reference='RHF', formula='E_FCI - E_RHF_input')
GEOMETRY_IDS = tuple((f'{i:03d}' for i in range(1, 31)))

def require(condition, path, key, message):
    """Stop input loading with the source path and offending quantity when a condition fails."""
    if not condition:
        raise ValueError(f'{path}: {key}: {message}')

def check_record_array(actual, expected, path, key, tolerance=1e-08):
    """Compare saved and derived full-space arrays using the stated numerical tolerance."""
    actual, expected = (np.asarray(actual), np.asarray(expected))
    require(actual.shape == expected.shape, path, key, f'shape {actual.shape} != expected {expected.shape}')
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    require(np.isfinite(error) and error <= tolerance, path, key, f'maximum absolute mismatch {error:.17g}; tolerance {tolerance:g}')

def ao_selection(raw, molecule, path):
    """Map final descriptors by atom occurrence and AO labels, never tensors."""
    spec = MOLECULES[molecule]
    symbols = tuple(raw['symbols'].tolist())
    labels = tuple((' '.join(str(s).split()) for s in raw['ao_labels']))
    canonical_atoms, used = ([], set())
    for symbol in spec.electronic_atom_order:
        candidates = [i for i, s in enumerate(symbols) if s == symbol and i not in used]
        require(bool(candidates), path, 'symbols', 'canonical atom occurrence is missing')
        canonical_atoms.append(candidates[0])
        used.add(candidates[0])
    require(len(used) == len(symbols), path, 'symbols', 'unexpected source atoms')
    parsed = [label.split() for label in labels]
    require(all((len(p) == 3 and p[0].isdigit() for p in parsed)), path, 'ao_labels', 'expected atom-index, symbol, and orbital labels')
    require(all((int(p[0]) < len(symbols) and symbols[int(p[0])] == p[1] for p in parsed)), path, 'ao_labels', 'AO atom labels disagree with symbols')
    full_order = [i for atom in canonical_atoms for i, p in enumerate(parsed) if int(p[0]) == atom]
    require(len(full_order) == len(labels), path, 'ao_labels', 'invalid full-AO membership')
    selected = np.asarray([full_order[i] for i in spec.expected_active_ao_indices], dtype=int)
    mapped_labels = tuple((f'{canonical_atoms.index(int(parsed[i][0]))} {parsed[i][1]} {parsed[i][2]}' for i in selected))
    expected = spec.expected_active_ao_labels
    require(mapped_labels == expected, path, 'ao_labels', f'canonical labels {mapped_labels} != {expected}')
    valence = np.asarray(raw['valence_indices'])
    core = np.asarray(raw['core_indices'])
    require(valence.dtype.kind in 'iu' and core.dtype.kind in 'iu', path, 'valence_indices/core_indices', 'integer indices required')
    require(np.array_equal(valence, np.sort(selected)), path, 'valence_indices', 'saved selection does not match canonical valence AO labels')
    require(np.array_equal(core, [i for i in range(len(labels)) if i not in selected]), path, 'core_indices', 'core and valence must partition the full AO basis')
    require(tuple((' '.join(str(s).split()) for s in raw['valence_labels'])) == tuple((labels[i] for i in valence)), path, 'valence_labels', 'labels disagree with saved full-AO indices')
    if molecule == 'BeH2':
        require(selected.tolist() == [5, 1, 2, 3, 4, 6], path, 'selected_AOs', 'expected source Be,Hminus,Hplus to canonical Hminus,Be,Hplus mapping')
    else:
        require(selected.tolist() == list(spec.expected_active_ao_indices), path, 'selected_AOs', 'unexpected nonidentity AO mapping')
    fg_order = [canonical_atoms[i] for i in geometry_contract(molecule)['source_to_fg']]
    return (selected, valence, mapped_labels, fg_order)

def load_record(path, molecule, geometry_id):
    """Load one saved full-AO record, validate it, then select final valence descriptors."""
    with np.load(path, allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}

    def scalar(key):
        """Read a required scalar from the current NPZ record."""
        require(key in raw and raw[key].shape == (), path, key, 'required scalar is missing')
        return raw[key].item()
    schema = scalar('schema_version')
    require(schema in SCHEMAS, path, 'schema_version', f'expected one of {SCHEMAS}')
    if 'status' in raw:
        require(scalar('status') in ('PASS', 'COMPLETED'), path, 'status', 'record failed')
    require(scalar('molecule') == molecule and scalar('geometry_id') == geometry_id and (scalar('point_index') == int(geometry_id)), path, 'identity', 'molecule/geometry mismatch')
    for key in ('frozen_core', 'mp2_frozen_core', 'rdm_frozen_core', 'charge', 'spin'):
        require(scalar(key) == 0, path, key, 'expected zero')
    require(scalar('basis') == 'sto-3g' and scalar('units') == 'Angstrom' and (scalar('energy_units') == 'Hartree'), path, 'units/basis', 'unsupported convention')
    require(scalar('rhf_converged') is True, path, 'rhf_converged', 'RHF must have converged')
    if 'validation_json' in raw:
        validation = json.loads(scalar('validation_json'))
        require(validation.get('passed') is True and validation.get('failed_checks') == [], path, 'validation_json', 'saved formal validation failed')
        require(bool(validation.get('checks')) and all((c.get('passed') is True for c in validation['checks'].values())), path, 'validation_json.checks', 'saved formal checks failed')
    spec = MOLECULES[molecule]
    n = spec.expected_active_ao_count + spec.expected_core_count
    matrices = ('S_AO', 'P_AO', 'F_AO', 'P_MP2_AO_consistent', 'S_half', 'S_minus_half', 'P_L', 'F_L')
    tensors = ('Gamma_MP2_AO', 'Gamma_MP2_AO_consistent', 'Gamma0_AO', 'Lambda_AO', 'Lambda_HFref_AO', 'Lambda_L')
    for key in matrices + tensors:
        shape = (n, n) if key in matrices else (n,) * 4
        require(key in raw and raw[key].shape == shape and np.isrealobj(raw[key]) and np.isfinite(raw[key]).all(), path, key, f'finite real full-AO shape {shape} required')
    require(len(raw['ao_labels']) == n, path, 'ao_labels', 'full-AO dimension mismatch')
    S, P, F = (raw[key] for key in ('S_AO', 'P_AO', 'F_AO'))
    half, inverse = (raw['S_half'], raw['S_minus_half'])
    for key in matrices:
        check_record_array(raw[key], raw[key].T, path, f'{key} symmetry')
    require(float(np.linalg.eigvalsh(S).min()) > 1e-08, path, 'S_AO', 'overlap not positive definite')
    check_record_array(half @ half, S, path, 'S_half @ S_half = S_AO')
    check_record_array(inverse @ S @ inverse, np.eye(n), path, 'full-AO Lowdin metric')
    check_record_array(half @ inverse, np.eye(n), path, 'S_half @ S_minus_half')
    check_record_array(raw['P_L'], half @ P @ half, path, 'full-AO P_L = S_half @ P_AO @ S_half')
    check_record_array(raw['F_L'], inverse @ F @ inverse, path, 'full-AO F_L = S_minus_half @ F_AO @ S_minus_half')
    check_record_array(raw['Gamma_MP2_AO'], raw['Gamma_MP2_AO_consistent'], path, 'formal Gamma source')
    check_record_array(raw['Lambda_AO'], raw['Lambda_HFref_AO'], path, 'formal Lambda source')
    check_record_array(raw['Lambda_L'], transform_rank4(raw['Lambda_AO'], half), path, 'descriptor.transform_rank4: full-AO Lowdin Lambda transformation')
    selected, valence, labels, fg_order = ao_selection(raw, molecule, path)
    p_source = raw['P_L'][np.ix_(valence, valence)]
    f_source = raw['F_L'][np.ix_(valence, valence)]
    pairs = np.asarray([(i, j) for i in range(len(valence)) for j in range(i + 1, len(valence))])
    check_record_array(raw['pair_indices'], pairs, path, 'pair_indices', tolerance=0)
    check_record_array(raw['P_mu'], np.diag(p_source), path, 'P_mu = final diag(P_L)[valence]')
    check_record_array(raw['F_mu'], np.diag(f_source), path, 'F_mu = final diag(F_L)[valence]')
    check_record_array(raw['P_munu'], p_source[pairs[:, 0], pairs[:, 1]], path, 'P_munu = final P_L[valence pairs]')
    p_final = raw['P_L'][np.ix_(selected, selected)]
    f_final = raw['F_L'][np.ix_(selected, selected)]
    rhf, fci = (float(scalar('E_RHF_input')), float(scalar('E_FCI')))
    target = fci - rhf
    for key in ('E_corr_input', 'E_FCI_minus_E_RHF_input'):
        check_record_array(scalar(key), target, path, key, tolerance=1e-10)
    rhf_key = 'E_RHF_calculated' if 'E_RHF_calculated' in raw else 'E_RHF_check'
    check_record_array(scalar(rhf_key), rhf, path, rhf_key, tolerance=1e-10)
    q, bond = (float(scalar('q_A')), float(scalar('bond_length_A')))
    check_record_array(scalar('scan_coordinate'), q, path, 'scan_coordinate = q_A', tolerance=0)
    require(np.isfinite([q, bond, rhf, fci, target]).all() and bond > 0, path, 'energies/coordinates', 'nonfinite or invalid scalar')
    coords = raw['cartesian_A']
    require(coords.shape == (len(raw['symbols']), 3) and np.isfinite(coords).all(), path, 'cartesian_A', 'finite full-precision Cartesian coordinates required')
    file_id = f'{molecule}/{geometry_id}.npz'
    sample_id = f'{molecule}/{geometry_id}'
    sample = ProcessedSample(sample_id=sample_id, dataset_id=DATASET_ID, molecule=molecule, geometry_index=int(geometry_id), scan_coordinate=q, scan_coordinate_name='bond-coordinate displacement from RHF equilibrium', scan_coordinate_unit='angstrom', basis='sto-3g', active_ao_indices=np.asarray(spec.expected_active_ao_indices), active_ao_labels=labels, active_ao_order_source='final descriptor selection by atom occurrence and AO label', pii=np.diag(p_final), fii=np.diag(f_final), pij=p_final, fij=f_final, t_lowdin=None, uhf_total_energy=None, fci_total_energy=fci, fci_corr_target=target, target_energy=target, target_definition=TARGET, ump2_total_energy=None, ump2_corr_energy=None, source_file=path, source_file_id=file_id, source_sha256='', adapter_profile=schema, field_provenance=(('pii/fii/pij/fij', 'final selection from saved full-AO P_L/F_L'),))
    metadata = dict(molecule=molecule, geometry_id=geometry_id, q_A=q, bond_length_A=bond, symbols=raw['symbols'].tolist(), cartesian_A=coords.tolist(), fg_coordinates=coords[fg_order].tolist(), source_selected_ao_indices=selected.tolist(), E_RHF=rhf, E_FCI=fci, E_corr=target, full_ao_count=n, input_fci_frozen_core=int(scalar('input_fci_frozen_core')), source_file=str(path))
    return (sample, metadata)

def load_population(input_dir, molecules=tuple(MOLECULES), *, split_manifest=None, split_protocol='manifest'):
    """Load 30 samples using a supplied three-way split or the explicit 15/15 protocol."""
    if split_protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f'Unknown split protocol: {split_protocol}')
    if split_protocol == '15-15' and split_manifest is not None:
        raise ValueError('--split-manifest cannot be combined with --split-protocol 15-15')
    split_value = read_split_manifest(split_manifest) if split_protocol == 'manifest' else None
    molecules = tuple(dict.fromkeys(molecules))
    if not molecules or any((m not in MOLECULES for m in molecules)):
        raise ValueError('Select at least one canonical molecule')
    if split_protocol == '15-15':
        split_value = {molecule: dict(train=GEOMETRY_IDS[::2], test=GEOMETRY_IDS[1::2]) for molecule in molecules}
    input_dir = Path(input_dir).resolve()
    path = input_dir / 'manifest.json'
    saved = json.loads(path.read_text())
    require(saved.get('schema_version') in SCHEMAS, path, 'schema_version', 'full-space descriptor dataset required')
    if 'dataset_status' in saved:
        require(saved['dataset_status'] == 'PASS', path, 'dataset_status', 'saved dataset validation failed')
    expected = {(molecule, geometry_id) for molecule in molecules for geometry_id in GEOMETRY_IDS}
    rows = [r for r in saved.get('records', []) if r['molecule'] in molecules]
    require(len(rows) == len(expected) and {(r['molecule'], r['geometry_id']) for r in rows} == expected, path, 'records', 'each selected molecule must contain exactly IDs 001..030')
    require(all((not r.get('error') and r.get('status', 'COMPLETED') in ('PASS', 'COMPLETED') for r in rows)), path, 'records.error/status', 'every geometry must have completed without an error')
    inventory = {}
    for record in saved.get('records', []):
        inventory.setdefault(record['molecule'], []).append(record['geometry_id'])
    splits = validate_split_manifest(split_value, inventory, molecules=molecules, split_protocol=split_protocol)
    samples, metadata = ([], {})
    for molecule in molecules:
        for geometry_id in GEOMETRY_IDS:
            source = input_dir / molecule / f'{geometry_id}.npz'
            try:
                sample, details = load_record(source, molecule, geometry_id)
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError(f'{source}: invalid/missing record field: {exc}') from exc
            samples.append(sample)
            metadata[sample.sample_id] = details
    source = Path(split_manifest).resolve() if split_manifest is not None else 'built-in 15-15 protocol'
    manifest = population_manifest(samples, splits, source=source, split_protocol=split_protocol)
    return (tuple(samples), manifest, metadata)

# Run preparation, smoke gate, seed sweep, and outputs

METHODS = ('FG', 'FE', 'MB-1')
AVAILABLE_METHODS = ('FG', 'FE', 'FE_prime', 'MB-1', MB1_PRIME)
SEEDS = tuple(range(32))
EPOCHS, BATCH_SIZE, LEARNING_RATE = (500, 8, 0.02)

def write_json(path, value):
    """Write a readable result record and reject nonfinite JSON numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def close(actual, expected, quantity):
    """Check model-preparation parity and report the exact mismatching quantity."""
    actual, expected = (np.asarray(actual), np.asarray(expected))
    if actual.shape != expected.shape:
        raise ValueError(f'{quantity}: shape {actual.shape} != {expected.shape}')
    error = float(np.max(np.abs(actual - expected), initial=0.0))
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError(f'{quantity}: max absolute mismatch {error:.17g}; tolerance 1e-12')
    return error

def raw_spectra(samples):
    """Compute and print the exact raw FE/FE_prime arrays before any preprocessing."""
    spectra = {}
    for sample in samples:
        n = MOLECULES[sample.molecule].expected_active_ao_count
        for label, matrix in (('P_L_valence', sample.pij), ('F_L_valence', sample.fij)):
            if matrix is None or matrix.shape != (n, n):
                raise ValueError(f'{sample.sample_id}: full {n}x{n} {label} matrix required for spectrum comparison')
            if not np.isrealobj(matrix) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-8):
                raise ValueError(f'{sample.sample_id}: finite real symmetric {label} required')
        # eigvalsh supplies ascending order. Do not sort, clip, or scale either spectrum here.
        fe = np.linalg.eigvalsh(sample.pij)
        fe_prime = np.linalg.eigvalsh(sample.fij)
        spectra[sample.sample_id] = {'FE': fe, 'FE_prime': fe_prime}
        print(f'{sample.molecule} geom_{sample.geometry_index:03d}', flush=True)
        print('FE  density eigenvalues:', flush=True)
        print(json.dumps(fe.tolist(), allow_nan=False), flush=True)
        print("FE' Fock eigenvalues:", flush=True)
        print(json.dumps(fe_prime.tolist(), allow_nan=False), flush=True)
        print('index    eig(P_L_valence)         eig(F_L_valence)', flush=True)
        for index, (density, fock) in enumerate(zip(fe, fe_prime, strict=True)):
            print(f'{index:<8d} {density: .17g}    {fock: .17g}', flush=True)
        print(flush=True)
    return spectra


def prepare_runs(samples, manifest, metadata, *, methods=METHODS):
    """Prepare selected methods with identical splits and shared fingerprint machinery."""
    methods = tuple(dict.fromkeys(methods))
    if not methods or any(method not in AVAILABLE_METHODS for method in methods):
        raise ValueError('Select at least one supported Run 1 method')
    spectra = raw_spectra(samples)
    prepared, audits = ({}, [])
    for molecule in dict.fromkeys((s.molecule for s in samples)):
        rows = sorted((r for r in manifest.rows if r.molecule == molecule), key=lambda r: r.retained_position)
        by_id = {s.sample_id: s for s in samples}
        physical = tuple((by_id[r.sample_id] for r in rows))
        train_ids = tuple((r.sample_id for r in rows if r.split == 'train'))
        validation_ids = tuple((r.sample_id for r in rows if r.split == 'validation'))
        test_ids = tuple((r.sample_id for r in rows if r.split == 'test'))
        fitting_manifest = SimpleNamespace(**dict(vars(manifest), rows=tuple(r for r in rows if r.split == 'train')))
        training_physical = tuple(by_id[i] for i in train_ids)
        for method in methods:
            if method in ONE_BODY_MODELS:
                selected = physical
                constants = compute_encoding_constants(training_physical, manifest=fitting_manifest, molecule=molecule, model_label=method)
            else:
                raw = np.vstack([fg_features(molecule, np.asarray(metadata[s.sample_id]['fg_coordinates'], dtype=float)) if method == 'FG' else spectra[s.sample_id][method] for s in physical])
                if method == 'FG' and molecule == 'H2O2':
                    torsion_index = fg_feature_names(molecule).index('dihedral_H1_O1_O2_H2')
                    wrapped = raw[:, torsion_index].copy()
                    if np.any(np.abs(np.diff(wrapped)) > np.pi):
                        raw[:, torsion_index] = np.unwrap(wrapped)
                selected = tuple((FingerprintSample(**vars(s), fingerprint=x) for s, x in zip(physical, raw, strict=True)))
                fitting_samples = tuple(s for s in selected if s.sample_id in train_ids)
                constants = fit_constants(fitting_samples, manifest=fitting_manifest, molecule=molecule, model_label=method)
                reference = fg_features(molecule, np.asarray(metadata[physical[0].sample_id]['fg_coordinates'], dtype=float)) if method == 'FG' else spectra[physical[0].sample_id][method]
                close(selected[0].fingerprint, reference, f'{physical[0].sample_id}/{method}: ' + ('geometry.fg_features' if method == 'FG' else f'raw spectrum {method}'))
                if raw.shape != (len(rows), feature_width(molecule, method)):
                    raise ValueError(f'{molecule}/{method}: canonical feature width mismatch')
            selected_by_id = {s.sample_id: s for s in selected}
            training = tuple((selected_by_id[i] for i in train_ids))
            validation = tuple((selected_by_id[i] for i in validation_ids))
            testing = tuple((selected_by_id[i] for i in test_ids))
            if constants.training_sample_ids != train_ids:
                raise ValueError(f'{molecule}/{method}: scaler training population mismatch')
            prepared[molecule, method] = (construct_model(constants, 2), training, validation, testing)
        n = MOLECULES[molecule].expected_active_ao_count
        expected_parameters = ParameterLayout(n, 2).num_parameters
        if 'MB-1' in methods:
            model, training, _, _ = prepared[molecule, 'MB-1']
            theta = initialize_parameters(model, training, seed=0)
            p, f, pairs, triple, abc, phi = model._quantum_arguments(physical[0], theta)
            c = model.constants
            close(p.detach().numpy(), standardize_diagonal(physical[0].pii, c.Pii_mean, c.Pii_std), f'{molecule}: model._quantum_arguments/Pii vs constants.standardize_diagonal')
            close(f.detach().numpy(), standardize_diagonal(physical[0].fii, c.Fii_mean, c.Fii_std), f'{molecule}: model._quantum_arguments/Fii vs constants.standardize_diagonal')
            n = c.n_active
            expected_parameters = ParameterLayout(n, 2).num_parameters
            if pairs is not None or triple is not None or theta.numel() != expected_parameters or (model.num_parameters != max(8, n) + 9):
                raise ValueError(f'{molecule}: canonical MB-1 preparation/parameter count mismatch')
            node = model.circuit_bank.get_qnode(c, num_hea_layers=2)
            tape = node.construct((p, f, pairs, abc, phi), {})
            operations = [op.name for op in tape.operations]
            expected_ops = ['RY'] * n
            expected_ops += (['RY'] * n + ['CNOT'] * n) * 2
            if operations != expected_ops or len(tape.measurements) != n:
                raise ValueError(f'{molecule}: circuit.CachedCircuitBank MB-1 tape mismatch: {operations}')
        targets = [s.target_energy for s in physical]
        audit = dict(molecule=molecule, retained=len(rows), train=len(train_ids), validation=len(validation_ids), test=len(test_ids), geometry_ids=[metadata[r.sample_id]['geometry_id'] for r in rows], train_geometry_ids=[metadata[i]['geometry_id'] for i in train_ids], validation_geometry_ids=[metadata[i]['geometry_id'] for i in validation_ids], test_geometry_ids=[metadata[i]['geometry_id'] for i in test_ids], target_min_Ha=min(targets), target_max_Ha=max(targets), FG_dimension=feature_width(molecule, 'FG'), FG_feature_names=fg_feature_names(molecule), FE_dimension=n, FE_prime_dimension=n, methods=list(methods), MB1_input_dimension=2 * n, MB1_qubits=n, MB1_parameters=expected_parameters)
        if MB1_PRIME in methods:
            model, training, _, _ = prepared[molecule, MB1_PRIME]
            theta = initialize_parameters(model, training, seed=0)
            p, f, pairs, triple, ac, phi = model._quantum_arguments(physical[0], theta)
            close(p.detach().numpy(), standardize_diagonal(physical[0].pii, model.constants.Pii_mean, model.constants.Pii_std), f'{molecule}: MB-1\' P scaling')
            if f is not None or pairs is not None or triple is not None or theta.numel() != expected_parameters - 1:
                raise ValueError(f'{molecule}: MB-1\' inputs/parameter count mismatch')
            tape = model.circuit_bank.get_qnode(model.constants, num_hea_layers=2).construct((p, f, pairs, ac, phi), {})
            if [op.name for op in tape.operations] != ['RY'] * n + (['RY'] * n + ['CNOT'] * n) * 2 or len(tape.measurements) != n:
                raise ValueError(f'{molecule}: MB-1\' circuit tape mismatch')
            audit.update(MB1_PRIME_display="MB-1'", MB1_PRIME_input_dimension=n,
                         MB1_PRIME_qubits=n, MB1_PRIME_encoder_parameters=2,
                         MB1_encoder_parameters=3, MB1_PRIME_parameters=model.num_parameters)
        audits.append(audit)
        print(json.dumps(audit), flush=True)
        for method in methods:
            _, tr, va, te = prepared[molecule, method]
            if tuple((s.sample_id for s in tr)) != train_ids or tuple((s.sample_id for s in va)) != validation_ids or tuple((s.sample_id for s in te)) != test_ids:
                raise ValueError(f'{molecule}/{method}: unmatched geometry populations')
    return (prepared, audits)

def train_one(output, molecule, method, seed, prepared, metadata, *, num_epochs=EPOCHS, checkpoint_selection='validation'):
    """Train and evaluate one molecule/model/seed and save its predictions, metrics, and parameters."""
    model, training, validation, testing = prepared[molecule, method]
    directory = output / 'runs' / molecule / method / f'seed_{seed:02d}'
    directory.mkdir(parents=True, exist_ok=False)
    print(f'Training {molecule} / {method} / seed {seed}: {num_epochs} epochs', flush=True)

    def progress(epoch, batch, loss, theta, optimizer):
        """Print periodic progress without modifying training state."""
        if epoch % 100 == 0 and batch is not None:
            if progress.last_epoch != epoch:
                print(f'  {molecule}/{method}/{seed}: epoch {epoch}', flush=True)
                progress.last_epoch = epoch
    progress.last_epoch = 0
    result = train(model, training, validation, seed=seed, num_epochs=num_epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, on_update=progress, checkpoint_selection=checkpoint_selection)
    predictions, metrics = ([], [])
    checkpoint_metrics = dict(checkpoint_selection=checkpoint_selection)
    if checkpoint_selection == 'validation':
        checkpoint_metrics['best_validation_mae_ha'] = result.checkpoint.validation_mae_ha
    else:
        checkpoint_metrics['best_train_mae_ha'] = result.checkpoint.train_mae_ha
    test_evaluation_call_count = 0
    for split, population in (('train', training), ('validation', validation), ('test', testing)):
        if split == 'validation' and checkpoint_selection == 'train':
            continue
        ev = evaluate(model, population, theta=result.theta, batch_size=BATCH_SIZE)
        if split == 'test':
            test_evaluation_call_count += 1
        if not np.isfinite(list(ev.metrics.values())).all():
            raise ValueError(f'{molecule}/{method}/{seed}/{split}: nonfinite evaluation metrics')
        metrics.append(dict(molecule=molecule, model=method, seed=seed, split=split, count=len(population), best_epoch=result.checkpoint.epoch, **checkpoint_metrics, **ev.metrics))
        for sample, prediction in zip(population, ev.predictions, strict=True):
            meta = metadata[sample.sample_id]
            predictions.append(dict(molecule=molecule, model=method, seed=seed, split=split, geometry_id=meta['geometry_id'], q_A=meta['q_A'], bond_length_A=meta['bond_length_A'], target_Ha=sample.target_energy, prediction_Ha=float(prediction), error_Ha=float(prediction - sample.target_energy)))
    if test_evaluation_call_count != 1:
        raise RuntimeError('Final test evaluation must occur exactly once')
    for metric in metrics:
        metric['final_test_evaluation_call_count'] = test_evaluation_call_count
    metrics_by_split = {metric['split']: metric for metric in metrics}
    saved_result = dict(best_epoch=result.checkpoint.epoch, **checkpoint_metrics,
        checkpoint_train_mae_ha=result.checkpoint.train_mae_ha,
        restored_train_metrics=metrics_by_split['train'],
        final_test_metrics=metrics_by_split['test'], final_test_evaluation_call_count=test_evaluation_call_count)
    if 'validation' in metrics_by_split:
        saved_result['restored_validation_metrics'] = metrics_by_split['validation']
    write_json(directory / 'result.json', saved_result)
    pd.DataFrame(predictions).to_csv(directory / 'predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(directory / 'metrics.csv', index=False)
    pd.DataFrame(result.history).to_csv(directory / 'training_history.csv', index=False)
    np.save(directory / 'best_theta.npy', result.theta.detach().numpy(), allow_pickle=False)
    print('  Finished: ' + ', '.join(f"{metric['split']} MAE={metric['mae_mHa']:.8g}" for metric in metrics) + ' mHa', flush=True)
    return metrics

def summarize(output, records, molecules, *, splits=SPLITS, methods=METHODS):
    """Require complete seed coverage and write molecule and overall train/test summaries."""
    frame = pd.DataFrame(records)
    summaries, overall, overall_runs = ([], [], [])
    if set(frame.split) != set(splits):
        raise ValueError('Summary populations do not match the selected split protocol')
    if set(frame.model) != set(methods):
        raise ValueError('Summary methods do not match the selected comparison')
    for split in splits:
        part = frame[frame.split == split]
        for molecule in molecules:
            for method in methods:
                seeds = sorted(part[(part.molecule == molecule) & (part.model == method)].seed.tolist())
                if seeds != list(SEEDS):
                    raise ValueError(f'{molecule}/{method}/{split}: incomplete/duplicate seed set {seeds}')
        renamed = part.rename(columns={'model': 'model_internal', 'mae_mHa': 'test_MAE_mHa'})
        summary = compute_statistics(renamed).reset_index().rename(columns={'model_internal': 'model'})
        summaries.append(summary.assign(split=split))
        per_seed = renamed.groupby(['model_internal', 'seed'], as_index=False).test_MAE_mHa.mean()
        per_seed['molecule'] = 'ALL'
        overall_runs.append(per_seed.rename(columns={'model_internal': 'model', 'test_MAE_mHa': 'mae_mHa'}).assign(split=split))
        overall.append(compute_statistics(per_seed).reset_index().rename(columns={'model_internal': 'model'}).assign(split=split))
    frame.to_csv(output / 'per_seed_metrics.csv', index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / 'per_molecule_summary.csv', index=False)
    pd.concat(overall_runs, ignore_index=True).to_csv(output / 'overall_per_seed.csv', index=False)
    pd.concat(overall, ignore_index=True).to_csv(output / 'overall_summary.csv', index=False)

def matched_mb1_reference(args, prepared, population):
    """Verify an existing completed MB-1 benchmark before creating a control run."""
    root = args.match_mb1_run.resolve()
    config = json.loads((root / 'configuration.json').read_text())
    expected = dict(molecules=list(dict.fromkeys(args.molecules)), seeds=list(SEEDS),
        epochs=args.epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        target=TARGET.formula, input_dir=str(args.input_dir.resolve()),
        split_protocol=args.split_protocol, full_AO_until_final_descriptor_selection=True,
        implementation='local codes/run1.py + circuit.py + train.py',
        checkpoint_selection=f"strictly improving {'train' if args.split_protocol == '15-15' else 'validation'} MAE; earliest epoch on ties")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'MB-1 reference mismatch for {key}: {config.get(key)!r} != {value!r}')
    if 'MB-1' not in config.get('methods', []):
        raise ValueError('Reference must contain MB-1')
    if json.loads((root / 'population.json').read_text()) != population:
        raise ValueError('MB-1 reference population, geometry, target or split mismatch')
    scalers = json.loads((root / 'preprocessing.json').read_text())
    for molecule in expected['molecules']:
        constants = asdict(prepared[molecule, MB1_PRIME][0].constants)
        constants['model_label'] = 'MB-1'
        if json.loads(json.dumps(constants)) != scalers.get(f'{molecule}/MB-1'):
            raise ValueError(f'{molecule}: MB-1 reference preprocessing mismatch')
    frame = pd.read_csv(root / 'per_seed_metrics.csv')
    frame = frame[frame.model == 'MB-1'].copy()
    if set(frame.molecule) != set(expected['molecules']) or set(frame.split) != set(SPLIT_PROTOCOLS[args.split_protocol]):
        raise ValueError('MB-1 reference metric populations mismatch')
    for molecule in expected['molecules']:
        for split in SPLIT_PROTOCOLS[args.split_protocol]:
            part = frame[(frame.molecule == molecule) & (frame.split == split)]
            count = sum(row['molecule'] == molecule and row['split'] == split for row in population)
            if sorted(part.seed.tolist()) != list(SEEDS) or not (part['count'] == count).all():
                raise ValueError(f'{molecule}/{split}: incomplete MB-1 reference seeds or population')
    if not np.isfinite(frame.mae_mHa).all() or (frame.mae_mHa < 0).any():
        raise ValueError('MB-1 reference MAEs must be finite and nonnegative')
    return frame


def write_mb1_comparison(output, reference, records, molecules):
    """Use Run 1's seed mean and sample SD in mHa; never retrain the reference."""
    frame = pd.concat((reference, pd.DataFrame(records)), ignore_index=True)
    frame = frame[frame.split == 'test']
    for molecule in molecules:
        for method in ONE_BODY_MODELS:
            part = frame[(frame.molecule == molecule) & (frame.model == method)]
            if sorted(part.seed.tolist()) != list(SEEDS):
                raise ValueError(f'{molecule}/{method}: comparison requires all 32 seeds exactly once')
    if not np.isfinite(frame.mae_mHa).all() or (frame.mae_mHa < 0).any():
        raise ValueError('Comparison MAEs must be finite and nonnegative')
    stats = compute_statistics(frame.rename(columns={'model': 'model_internal', 'mae_mHa': 'test_MAE_mHa'}))
    rows = []
    for molecule in molecules:
        mb, prime = (stats.loc[(molecule, method)] for method in ONE_BODY_MODELS)
        rows.append({'Molecule': molecule, 'MB-1 MAE mean': mb['mean'], 'MB-1 MAE SD': mb['sd'],
                     "MB-1' MAE mean": prime['mean'], "MB-1' MAE SD": prime['sd'],
                     'ΔMAE': prime['mean'] - mb['mean']})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'mb1_prime_comparison.csv', index=False)
    lines = ["MB-1 versus MB-1' test MAE (mHa). Mean and sample SD (ddof=1) over seeds 0–31.",
             '', "ΔMAE = MAE(MB-1') − MAE(MB-1). Positive values mean MB-1' has higher MAE.", '',
             '| ' + ' | '.join(comparison.columns) + ' |',
             '| ' + ' | '.join(['---'] * len(comparison.columns)) + ' |']
    for row in comparison.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join([row[0]] + [f'{v:.12g}' for v in row[1:]]) + ' |')
    (output / 'mb1_prime_comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main(argv=None):
    """Parse the run selection, audit inputs, gate the seed sweep on smoke success, and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'results/descriptor')
    parser.add_argument('--split-protocol', choices=tuple(SPLIT_PROTOCOLS), default='manifest', help='manifest (default): explicit train/validation/test IDs; 15-15: odd IDs train, even IDs test, best-training-MAE checkpoint.')
    parser.add_argument('--split-manifest', type=Path, help='Required for the default manifest protocol: explicit per-molecule train/validation/test geometry IDs.')
    parser.add_argument('--molecules', nargs='+', choices=tuple(MOLECULES), default=list(MOLECULES), help='Molecules to run; default all nine.')
    parser.add_argument('--methods', nargs='+', choices=AVAILABLE_METHODS, default=list(METHODS), help='Methods to compare; default FG FE MB-1. Use FE FE_prime for density versus Fock spectra.')
    parser.add_argument('--output-dir', type=Path, help='New directory; default results/run1/<timestamp>. Existing paths are rejected.')
    parser.add_argument('--match-mb1-run', type=Path, help="Completed MB-1 Run 1 directory. Require an exactly matched MB1_PRIME-only run and write the comparison table.")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs; default 500. Smoke-only runs use at most two epochs.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--validate-only', action='store_true', help='Load, prepare and audit; do not train.')
    modes.add_argument('--smoke-only', action='store_true', help='Audit and run seed-0 for H2O if selected, otherwise the first molecule.')
    args = parser.parse_args(argv)
    methods = tuple(dict.fromkeys(args.methods))
    if args.match_mb1_run is not None and (methods != (MB1_PRIME,) or args.smoke_only):
        parser.error('--match-mb1-run requires --methods MB1_PRIME and a full run or --validate-only')
    if args.split_protocol == 'manifest' and args.split_manifest is None:
        parser.error('--split-manifest is required for the default manifest protocol. Exact manuscript split IDs are unavailable; the 32-seed benchmark is blocked without a real split manifest. Use --split-protocol 15-15 to explicitly select the separate train/test protocol.')
    if args.split_protocol == '15-15' and args.split_manifest is not None:
        parser.error('--split-manifest cannot be combined with --split-protocol 15-15')
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    epochs = min(args.epochs, 2) if args.smoke_only else args.epochs
    output = args.output_dir or PROJECT / 'results/run1' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    samples, manifest, metadata = load_population(args.input_dir.resolve(), molecules=args.molecules, split_manifest=args.split_manifest, split_protocol=args.split_protocol)
    prepared, audits = prepare_runs(samples, manifest, metadata, methods=methods)
    molecules = [a['molecule'] for a in audits]
    smoke_molecule = 'H2O' if 'H2O' in molecules else molecules[0]
    checkpoint_selection = 'train' if args.split_protocol == '15-15' else 'validation'
    population = [dict(metadata[r.sample_id], split=r.split, retained_position=r.retained_position) for r in manifest.rows]
    reference = matched_mb1_reference(args, prepared, population) if args.match_mb1_run is not None else None
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'population.json', population)
    write_json(output / 'audit.json', audits)
    write_json(output / 'configuration.json', dict(molecules=molecules, methods=methods, seeds=(0,) if args.smoke_only else SEEDS, epochs=epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, target='E_FCI - E_RHF_input', input_dir=str(args.input_dir.resolve()), implementation='local codes/run1.py + circuit.py + train.py', population=manifest.split_rule, split_protocol=args.split_protocol, split_manifest=str(args.split_manifest.resolve()) if args.split_manifest is not None else None, split_ids_source='user-supplied; manuscript provenance is not established by this interface' if args.split_protocol == 'manifest' else LEGACY_SPLIT_RULE, full_AO_until_final_descriptor_selection=True, FE='ascending eigenvalues of the valence block of full-AO Lowdin RHF density P_L', FE_prime='ascending eigenvalues of the valence block of full-AO Lowdin RHF Fock F_L', MB1='P_mu/F_mu; physical Pij alpha fit retained, no pair or triple gates', checkpoint_selection=f'strictly improving {checkpoint_selection} MAE; earliest epoch on ties', overall='equal-weight molecule MAE per seed, then canonical seed mean/SD/SE; all summary units mHa'))
    write_json(output / 'split_manifest.json', {molecule: {split: [metadata[r.sample_id]['geometry_id'] for r in manifest.rows if r.molecule == molecule and r.split == split] for split in SPLIT_PROTOCOLS[args.split_protocol]} for molecule in molecules})
    write_json(output / 'preprocessing.json', {f'{molecule}/{method}': asdict(model.constants) for (molecule, method), (model, _, _, _) in prepared.items()})
    if MB1_PRIME in methods:
        write_json(output / 'mb1_prime_control.json', dict(
            model=MB1_PRIME, display_label="MB-1'",
            encoder='gamma_mu = (pi/2) * tanh(a1 * P_tilde_mu + c1)',
            trainable_encoder_parameters=['a1', 'c1'],
            initialization='same seeded MB-1 normal vector with b1 removed; same training-target-mean bias',
            reference_run=str(args.match_mb1_run.resolve()) if args.match_mb1_run is not None else None,
            MAE_units='mHa', seed_SD_ddof=1))
    print(f'All populations and canonical preparation checks passed. Output: {output}', flush=True)
    if args.validate_only:
        return
    metrics, completed = ([], set())
    for method in methods:
        metrics.extend(train_one(output, smoke_molecule, method, 0, prepared, metadata, num_epochs=epochs, checkpoint_selection=checkpoint_selection))
        completed.add((smoke_molecule, method, 0))
    write_json(output / 'smoke_result.json', dict(status='PASS', molecule=smoke_molecule, seed=0, epochs=epochs, methods=methods, finite_predictions_and_metrics=True, metrics=metrics))
    print(f"Smoke passed: {smoke_molecule}/seed-0, {'/'.join(methods)}, finite evaluation.", flush=True)
    if args.smoke_only:
        return
    for molecule in molecules:
        for method in methods:
            for seed in SEEDS:
                if (molecule, method, seed) not in completed:
                    metrics.extend(train_one(output, molecule, method, seed, prepared, metadata, num_epochs=epochs, checkpoint_selection=checkpoint_selection))
    summarize(output, metrics, molecules, splits=SPLIT_PROTOCOLS[args.split_protocol], methods=methods)
    if reference is not None:
        write_mb1_comparison(output, reference, metrics, molecules)
    print(f'Run 1 complete: {len(molecules)} molecules x {len(methods)} models x 32 seeds. Results: {output}', flush=True)
if __name__ == '__main__':
    main()
