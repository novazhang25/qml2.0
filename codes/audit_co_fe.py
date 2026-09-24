#!/usr/bin/env python3
"""Audit all saved CO FE spectra and their production pipeline without solvers/training."""
from __future__ import annotations
import argparse
import contextlib
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import circuit
import run1

PROJECT = Path(__file__).resolve().parents[1]


def maximum(values):
    return float(np.max(np.abs(values)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'results/descriptor')
    parser.add_argument('--reference-run', type=Path,
                        default=PROJECT / 'results/run1/20260924_115423_466018')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'results/audits/co_fe')
    parser.add_argument('--tolerance', type=float, default=1e-15)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('--tolerance must be finite and positive')
    source_paths = sorted((args.input_dir / 'CO').glob('*.npz'))
    if [p.stem for p in source_paths] != list(run1.GEOMETRY_IDS):
        raise ValueError('Expected exactly CO geometry files 001..030')
    hashes = {str(path.resolve()): digest(path) for path in source_paths}
    samples, manifest, metadata = run1.load_population(args.input_dir, ('CO',), split_protocol='15-15')
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        prepared, _ = run1.prepare_runs(samples, manifest, metadata, methods=('FE',))
    model, training, validation, testing = prepared['CO', 'FE']
    features = {s.sample_id: s.fingerprint for s in training + validation + testing}
    rows, entries = [], []
    for sample, path in zip(samples, source_paths, strict=True):
        with np.load(path, allow_pickle=False) as saved:
            d = {key: saved[key] for key in saved.files}
        S, P, C, occ = (d[k] for k in ('S_AO', 'P_AO', 'mo_coeff', 'mo_occ'))
        v, core = d['valence_indices'], d['core_indices']
        assert np.array_equal(v, [1, 2, 3, 4, 6, 7, 8, 9])
        assert np.array_equal(core, [0, 5])
        assert np.array_equal(occ, [2.] * 7 + [0.] * 3)
        assert int(d['electron_count']) == 14
        eig_s, U = np.linalg.eigh(S)
        fresh_half = (U * np.sqrt(eig_s)) @ U.T
        from_mos = (C * occ) @ C.T
        transformed = d['S_half'] @ P @ d['S_half']
        fresh_transformed = fresh_half @ from_mos @ fresh_half
        block = d['P_L'][np.ix_(v, v)]
        coupling = d['P_L'][np.ix_(v, core)]
        eig = np.linalg.eigvalsh(block)
        fresh_eig = np.linalg.eigvalsh(fresh_transformed[np.ix_(v, v)])
        np.testing.assert_array_equal(eig, features[sample.sample_id])
        np.testing.assert_array_equal(block, sample.pij)
        zero = np.abs(eig) <= args.tolerance
        two = np.abs(eig - 2.) <= args.tolerance
        checks = {
            'P_AO_minus_C_occ_CT': maximum(P - from_mos),
            'C_T_S_C_minus_I': maximum(C.T @ S @ C - np.eye(10)),
            'S_half_squared_minus_S': maximum(d['S_half'] @ d['S_half'] - S),
            'S_half_minus_independent_eigh_root': maximum(d['S_half'] - fresh_half),
            'saved_P_L_minus_full_transform': maximum(d['P_L'] - transformed),
            'P_S_P_minus_2P': maximum(P @ S @ P - 2 * P),
            'P_L_squared_minus_2P_L': maximum(d['P_L'] @ d['P_L'] - 2 * d['P_L']),
            'trace_P_S_minus_14': abs(float(np.trace(P @ S)) - 14.),
            'trace_P_L_minus_14': abs(float(np.trace(d['P_L'])) - 14.),
            'valence_projector_compression_identity': maximum(block @ block - 2 * block + coupling @ coupling.T),
            'production_FE_minus_independent_MO_overlap_reconstruction': maximum(eig - fresh_eig),
        }
        for name, error in checks.items():
            if error > 1e-10:
                raise ValueError(f'{path}: {name} residual {error} exceeds 1e-10')
        initial_path = Path(str(d['input_state_path']))
        with np.load(initial_path, allow_pickle=False) as initial:
            initial_occ = initial['mo_occ']
            np.testing.assert_array_equal(initial_occ, occ)
            initial_p_error = maximum(initial['rhf_density_ao'] -
                                      (initial['mo_coeff'] * initial_occ) @ initial['mo_coeff'].T)
            initial_to_descriptor = maximum(initial['rhf_density_ao'] - P)
            initial_s_error = maximum(initial['overlap_ao'] - S)
        entry = dict(geometry_id=path.stem, bond_length_A=float(d['bond_length_A']),
                     eigenvalues=eig.tolist(), distances_to_zero=np.abs(eig).tolist(),
                     distances_to_two=np.abs(eig - 2).tolist(),
                     near_zero_indices=np.flatnonzero(zero).tolist(),
                     near_two_indices=np.flatnonzero(two).tolist(),
                     full_P_L_eigenvalues=np.linalg.eigvalsh(d['P_L']).tolist(),
                     valence_core_coupling_singular_values=np.linalg.svd(coupling, compute_uv=False).tolist(),
                     independent_reconstruction_eigenvalues=fresh_eig.tolist(),
                     near_two_indices_independent=np.flatnonzero(abs(fresh_eig - 2) <= args.tolerance).tolist(),
                     checks=checks, upstream_checkpoint=str(initial_path),
                     upstream_density_from_MO_residual=initial_p_error,
                     upstream_overlap_residual=initial_s_error,
                     upstream_to_reconverged_descriptor_density_difference=initial_to_descriptor,
                     RHF_vs_MP2_density_difference=maximum(P - d['P_MP2_AO_consistent']),
                     saved_RHF_energy_difference=abs(float(d['E_RHF_calculated'] - d['E_RHF_input'])),
                     scf_commutator_residual=maximum(d['F_AO'] @ P @ S - S @ P @ d['F_AO']))
        entries.append(entry)
        for i, value in enumerate(eig):
            rows.append(dict(geometry_id=path.stem, index=i, eigenvalue=format(value, '.17g'),
                             distance_to_zero=format(abs(value), '.17g'),
                             distance_to_two=format(abs(value - 2), '.17g'),
                             near_zero=bool(zero[i]), near_two=bool(two[i])))
    X = np.vstack([features[s.sample_id] for s in samples])
    printed_lines = captured.getvalue().splitlines()
    printed_fe = [json.loads(printed_lines[i + 1]) for i, line in enumerate(printed_lines)
                  if line == 'FE  density eigenvalues:']
    np.testing.assert_array_equal(np.asarray(printed_fe), X)
    angles = np.vstack([circuit.encode(row, model.constants) for row in X])
    prior = json.loads((args.reference_run / 'preprocessing.json').read_text())['CO/FE']
    current = asdict(model.constants)
    for key in ('mu', 'sigma', 'sigma_safe', 'constant_mask', 'training_sample_ids'):
        np.testing.assert_array_equal(current[key], prior[key])
    for path in source_paths:
        assert digest(path) == hashes[str(path.resolve())], 'Input descriptor changed during audit'
    zero_ids = [e['geometry_id'] for e in entries if e['near_zero_indices']]
    two_ids = [e['geometry_id'] for e in entries if e['near_two_indices']]
    inactive = np.flatnonzero(model.constants.constant_mask).tolist()
    check_maxima = {key: max(e['checks'][key] for e in entries) for key in entries[0]['checks']}
    upstream_maxima = {key: max(e[key] for e in entries) for key in (
        'upstream_density_from_MO_residual', 'upstream_overlap_residual',
        'upstream_to_reconverged_descriptor_density_difference',
        'saved_RHF_energy_difference', 'scf_commutator_residual')}
    coupling_ranges = [(min(e['valence_core_coupling_singular_values'][i] for e in entries),
                        max(e['valence_core_coupling_singular_values'][i] for e in entries))
                       for i in range(2)]
    result = dict(tolerance=args.tolerance, criterion='abs(lambda)<=tol OR abs(lambda-2)<=tol; no relative tolerance',
                  input_dir=str(args.input_dir.resolve()), reference_run=str(args.reference_run.resolve()),
                  geometry_count=30, near_zero_geometry_ids=zero_ids, near_two_geometry_ids=two_ids,
                  near_zero_value_count=sum(len(e['near_zero_indices']) for e in entries),
                  near_two_value_count=sum(len(e['near_two_indices']) for e in entries),
                  eigenvalue_min=X.min(axis=0).tolist(), eigenvalue_max=X.max(axis=0).tolist(),
                  training_standard_deviations=list(model.constants.sigma),
                  zeroed_encoding_indices=inactive,
                  encoded_column_ranges=np.ptp(angles, axis=0).tolist(),
                  saved_run_preprocessing_exact_match=True, input_sha256=hashes,
                  pipeline_residual_maxima=check_maxima, upstream_residual_maxima=upstream_maxima, geometries=entries,
                  versions=dict(numpy=np.__version__),
                  scope='Read-only saved-data and current-code audit; no solvers, training, or scientific changes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output_dir / 'eigenvalues.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / 'production_raw_spectra.txt').write_text(captured.getvalue())
    report = [
        '# CO FE density-eigenvalue audit', '',
        f'All 30 geometries audited. Absolute criterion: |lambda| <= {args.tolerance:g} or |lambda-2| <= {args.tolerance:g}; rtol=0.',
        f'Near-zero geometries: {len(zero_ids)}/30; values: {result["near_zero_value_count"]}.',
        f'Near-two geometries: {len(two_ids)}/30; values: {result["near_two_value_count"]}.',
        f'Near-two geometry IDs: {", ".join(two_ids)}.', '',
        '## Pipeline audit', '',
        '1. initialdata.rhf_arrays saves the closed-shell spin-summed RHF density and MO occupations.',
        '2. descriptor.calculate_record uses that density as an SCF initial guess, reconverges RHF, canonicalizes MOs, then sets P_AO=mf.make_rdm1(). MP2 densities are separate arrays.',
        '3. descriptor.lowdin_factors computes the full 10x10 positive square root of S; P_L=S_half @ P_AO @ S_half.',
        '4. The full transformation precedes AO selection. C1s/O1s indices [0,5] are excluded; valence indices are [1,2,3,4,6,7,8,9].',
        '5. run1.load_record validates the stored transforms and selects the full 8x8 block as sample.pij. raw_spectra applies np.linalg.eigvalsh directly, without rounding, clipping, diagonal-only extraction, or post-sort.',
        '6. prepare_runs fingerprints were checked exactly against the independently extracted stored block spectra for all 30 geometries.',
        '7. circuit.fit_constants uses only odd-ID training geometries. Statistics match the completed CO run exactly.', '',
        'Maximum absolute residuals across all geometries (audit acceptance: 1e-10, separate from the requested 1e-15 classification):', '',
        *[f'- {key}: {value:.17g}' for key, value in check_maxima.items()], '',
        'Upstream checkpoint / reconverged RHF diagnostics (the upstream density is an initial guess, not required to be identical to the reconverged density):', '',
        *[f'- {key}: {value:.17g}' for key, value in upstream_maxima.items()], '',
        'Code inspected: codes/initialdata.py (rhf_arrays), codes/descriptor.py (calculate_record, lowdin_factors, select_valence_aos), codes/run1.py (load_record, raw_spectra, prepare_runs), codes/circuit.py (fit_constants, encode). The installed PySCF scf.hf.make_rdm1 builds the spin-summed density as (C_occ * occupations) @ C_occ.conj().T.', '',
        '## Interpretation and preprocessing consequence', '',
        'CO has 14 electrons, 10 spatial AOs and occupations [2,2,2,2,2,2,2,0,0,0]. The full orthonormal RHF density is twice an occupied-space projector: P_L^2=2 P_L. Its full spectrum is therefore seven 2s and three 0s, up to numerical error.',
        'For the 8-dimensional valence compression, dimension counting forces at least five eigenvalues at 2 and at least one at 0 in exact arithmetic. Cropping does not generally preserve idempotency: the audit verifies A^2-2A=-B B^T for valence block A and valence/core coupling B.',
        'The actual CO spectra contain two near-zero entries, one varying fractional entry and five near-two entries. The strict 1e-15 test does not include every near-two entry: matrix arithmetic/eigensolver roundoff can exceed that cutoff.',
        f'Valence/core coupling singular-value ranges: {coupling_ranges}. For this scan there is only one significant singular value, consistent with one eigenvalue departing materially from the endpoints. The valence trace is 10 plus the fractional eigenvalue: removing core AOs is not the same operation as removing exactly occupied core MOs.',
        f'The production scaler masks zero-based columns {inactive} because their training SD is below 1e-12. Their encoding angles are exactly zero for all 30 geometries. Only column 2 (the third eigenvalue) supplies varying input angles.',
        f'Third eigenvalue range: {X[:, 2].min():.17g} to {X[:, 2].max():.17g}.',
        f'Training SD by column: {list(model.constants.sigma)}.',
        'This is a low-information RHF-spectrum descriptor for this CO scan, not evidence that FE accidentally uses Fock, MP2, diagonal density elements, or manually rounded occupations. The audit does not independently rerun electronic-structure solvers.', '',
        '## All matching geometries: actual raw FE eigenvalues', '',
        'Indices below are zero-based. Arrays retain float64 round-trip precision.', '',
    ]
    for e in entries:
        if not (e['near_zero_indices'] or e['near_two_indices']):
            continue
        report += [f'### CO geom_{e["geometry_id"]}',
                   f'Near 0 indices: {e["near_zero_indices"]}; near 2 indices: {e["near_two_indices"]}.',
                   '```text', '[' + ', '.join(format(x, '.17g') for x in e['eigenvalues']) + ']', '```', '']
    report += ['Inputs were hash-checked unchanged. No production code or descriptor was modified.', '']
    (args.output_dir / 'report.md').write_text('\n'.join(report))
    print('\n'.join(report))
    print(f'Saved audit to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
