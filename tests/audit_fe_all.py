#!/usr/bin/env python3
"""Read-only full-space RHF and FE information audit; no solvers or training."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import circuit
import run1

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/descriptor'
OUTPUT = ROOT / 'results/audits/fe_all_molecules'
TOL = 1e-10
SCALER_THRESHOLD = 1e-12


def maximum(a):
    return float(np.max(np.abs(a)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUTPUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    full_rows, column_rows, spectrum_rows, summaries = [], [], [], []
    hashes, loader_errors, fitted = {}, {}, {}
    molecules = tuple(circuit.MOLECULES)
    for molecule in molecules:
        paths = sorted((INPUT / molecule).glob('*.npz'))
        if [p.stem for p in paths] != list(run1.GEOMETRY_IDS):
            raise ValueError(f'{molecule}: expected exactly 001..030')
        samples, q_by_id, views = [], {}, {}
        for path in paths:
            hashes[str(path.resolve())] = sha(path)
            with np.load(path, allow_pickle=False) as f:
                d = {key: f[key] for key in f.files}
            S, P, H = d['S_AO'], d['P_AO'], d['S_half']
            PL = H @ P @ H
            ev = np.linalg.eigvalsh(PL)
            distances = np.minimum(np.abs(ev), np.abs(ev - 2))
            n, electrons = len(P), int(d['electron_count'])
            svals, U = np.linalg.eigh(S)
            root = (U * np.sqrt(svals)) @ U.T
            checks = dict(
                projector_residual_max=maximum(PL @ PL - 2 * PL),
                eigenvalue_distance_min=float(distances.min()),
                eigenvalue_distance_max=float(distances.max()),
                density_symmetry_residual=maximum(PL - PL.T),
                root_reconstruction_residual=maximum(H - root),
                saved_transform_residual=maximum(PL - d['P_L']),
                electron_trace_residual=abs(float(np.trace(PL)) - electrons),
                MO_density_residual=maximum(P - (d['mo_coeff'] * d['mo_occ']) @ d['mo_coeff'].T),
            )
            occ = d['mo_occ']
            passed = (np.isfinite(PL).all() and np.all(svals > 0)
                      and bool(d['rhf_converged']) and np.all(np.isin(occ, [0., 2.]))
                      and np.sum(occ) == electrons and all(value <= TOL for value in checks.values())
                      and np.sum(ev > 1) == electrons // 2)
            full_rows.append(dict(molecule=molecule, geometry_id=path.stem, full_AO_width=n,
                                  electrons=electrons, **checks, status='PASS' if passed else 'FAIL',
                                  eigenvalues=json.dumps(ev.tolist())))
            q_by_id[path.stem] = float(d['q_A'])
            try:
                sample, _ = run1.load_record(path, molecule, path.stem)
            except ValueError as exc:
                if molecule != 'H2S' or 'saved selection does not match canonical valence AO labels' not in str(exc):
                    raise
                loader_errors[f'{molecule}/{path.stem}'] = str(exc)
                # Diagnostic views only: do not change saved indices or bypass the loader for training.
                saved = d['valence_indices']
                intended = np.asarray(circuit.MOLECULES[molecule].expected_active_ao_indices)
                for name, indices in (('saved_6AO_diagnostic', saved), ('intended_10AO_diagnostic', intended)):
                    views.setdefault(name, []).append((path.stem, np.linalg.eigvalsh(d['P_L'][np.ix_(indices, indices)])))
            else:
                samples.append(sample)
                views.setdefault('Run1_exact', []).append((path.stem, np.linalg.eigvalsh(sample.pij)))
        for view, pairs in views.items():
            by_id = dict(pairs)
            ids = sorted(by_id, key=lambda i: (q_by_id[i], int(i)))
            train_ids = [i for i in ids if int(i) % 2 == 1]
            X = np.vstack([by_id[i] for i in ids])
            Xtrain = np.vstack([by_id[i] for i in train_ids])
            mu, std = Xtrain.mean(axis=0), Xtrain.std(axis=0, ddof=0)
            actual_mask, requested_mask = std < SCALER_THRESHOLD, std <= SCALER_THRESHOLD
            active = np.flatnonzero(std > SCALER_THRESHOLD)
            if view == 'Run1_exact':
                split = {molecule: dict(train=run1.GEOMETRY_IDS[::2], test=run1.GEOMETRY_IDS[1::2])}
                manifest = run1.population_manifest(samples, split, source='built-in 15-15 protocol', split_protocol='15-15')
                rows = tuple(r for r in manifest.rows if r.split == 'train')
                fitting_manifest = SimpleNamespace(**dict(vars(manifest), rows=rows))
                fingerprints = [circuit.FingerprintSample(**vars(sample), fingerprint=by_id[f'{sample.geometry_index:03d}'])
                                for sample in samples if sample.geometry_index % 2 == 1]
                constants = circuit.fit_constants(fingerprints, manifest=fitting_manifest, molecule=molecule, model_label='FE')
                np.testing.assert_array_equal(constants.mu, mu)
                np.testing.assert_array_equal(constants.sigma, std)
                np.testing.assert_array_equal(constants.constant_mask, actual_mask)
                fitted[molecule] = asdict(constants)
            for j in range(X.shape[1]):
                column_rows.append(dict(molecule=molecule, view=view, FE_width=X.shape[1], column_index=j,
                    train_mean=float(mu[j]), train_std_ddof0=float(std[j]),
                    actual_constant_mask_lt_1e_minus12=bool(actual_mask[j]),
                    requested_constant_mask_le_1e_minus12=bool(requested_mask[j]), active_std_gt_1e_minus12=bool(std[j] > SCALER_THRESHOLD),
                    all30_min=float(X[:, j].min()), all30_max=float(X[:, j].max()),
                    train_geometry_ids=' '.join(train_ids)))
            for geometry_id, eig in pairs:
                for j, value in enumerate(eig):
                    spectrum_rows.append(dict(molecule=molecule, view=view, geometry_id=geometry_id,
                                              column_index=j, eigenvalue=float(value)))
            local = [r for r in full_rows if r['molecule'] == molecule]
            summaries.append(dict(molecule=molecule, view=view, full_space_pass_count=sum(r['status']=='PASS' for r in local),
                geometry_count=len(local), full_projector_residual_max=max(r['projector_residual_max'] for r in local),
                full_eigenvalue_distance_min=min(r['eigenvalue_distance_min'] for r in local),
                full_eigenvalue_distance_max=max(r['eigenvalue_distance_max'] for r in local),
                FE_width=X.shape[1], constant_dimensions=int(requested_mask.sum()), active_dimensions=len(active),
                active_column_indices=' '.join(str(i) for i in active),
                threshold_boundary_count=int(np.sum(std == SCALER_THRESHOLD)),
                loader_status='PASS' if view == 'Run1_exact' else 'BLOCKED: saved 6 AOs; loader expects 10'))
    for path, before in hashes.items():
        if sha(Path(path)) != before:
            raise RuntimeError(f'Source changed during audit: {path}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv('full_space_checks.csv', full_rows)
    write_csv('fe_column_statistics.csv', column_rows)
    write_csv('fe_eigenvalues.csv', spectrum_rows)
    main_summary = []
    for molecule in molecules:
        matching = [r for r in summaries if r['molecule'] == molecule]
        row = dict(matching[0])
        row['saved_FE_width'] = row['FE_width']
        if molecule == 'H2S':
            row.update(view='BLOCKED', FE_width=circuit.feature_width(molecule, 'FE'),
                       constant_dimensions='', active_dimensions='', active_column_indices='')
        main_summary.append(row)
    write_csv('summary.csv', main_summary)
    write_csv('h2s_diagnostics.csv', [r for r in summaries if r['molecule'] == 'H2S'])
    (OUTPUT / 'provenance.json').write_text(json.dumps(dict(
        input_sha256=hashes, input_dir=str(INPUT), numpy_version=np.__version__, projector_tolerance=TOL,
        scaler_threshold=SCALER_THRESHOLD, split='15 odd geometry IDs train; 15 even IDs test; training order follows Run 1 retained positions',
        actual_mask_rule='std < 1e-12', requested_mask_rule='std <= 1e-12', activity_rule='std > 1e-12',
        loader_errors=loader_errors, actual_Run1_FE_constants=fitted), indent=2) + '\n')
    if any(r['threshold_boundary_count'] for r in summaries):
        raise ValueError('Boundary-valued column detected: strict and inclusive masks require separate reporting')
    lines = ['# FE audit: all nine molecules, all 30 geometries', '',
        'Read-only audit of saved data. No RHF/MP2/FCI calculation, training, or scientific code/data change.',
        'Full-space PASS requires max-norm projector residual, eigenvalue endpoint distances, transform/root/symmetry/MO-density/trace residuals <= 1e-10, positive overlap, converged closed-shell occupations, and the correct occupied rank.',
        'Full spectra use reconstructed P_L = S_half @ P_AO @ S_half. FE uses the exact stored block selected by the Run 1 loader for every compatible molecule.',
        'Training uses Run 1 15/15: odd IDs 001,003,...,029; std uses ddof=0. Actual source mask is std < 1e-12; requested audit mask is std <= 1e-12. Both masks are exported. No column lies exactly on the threshold, so their classifications agree here. Activity is std > 1e-12, never the 1e-15 endpoint test.', '',
        '| Molecule | Full-space PASS | Max projector residual | Max endpoint distance | FE width | Active dimensions | Active columns (0-based) | Loader |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for molecule in molecules:
        rows = [r for r in summaries if r['molecule'] == molecule]
        r = rows[0]
        if molecule == 'H2S':
            lines.append(f'| H2S | {r["full_space_pass_count"]}/30 | {r["full_projector_residual_max"]:.3e} | {r["full_eigenvalue_distance_max"]:.3e} | 10 expected / 6 saved | BLOCKED | See diagnostics below | MISMATCH |')
        else:
            lines.append(f'| {molecule} | {r["full_space_pass_count"]}/30 | {r["full_projector_residual_max"]:.3e} | {r["full_eigenvalue_distance_max"]:.3e} | {r["FE_width"]} | {r["active_dimensions"]} | {r["active_column_indices"]} | PASS |')
    lines += ['', 'H2S: all 30 full-space matrices can be audited. All 30 Run 1 loads reject the saved valence selection. The six-AO saved-block and ten-AO intended-block spectra below are separate, explicitly labeled diagnostics; neither represents a successful H2S Run 1 load. No indices or files were repaired.', '',
              '## Full-space geometry checks', '',
              '| Molecule | Geometry | max abs(P_L^2-2P_L) | Min eig distance to {0,2} | Max eig distance to {0,2} | Status |',
              '|---|---|---:|---:|---:|---|']
    for r in full_rows:
        lines.append(f'| {r["molecule"]} | {r["geometry_id"]} | {r["projector_residual_max"]:.17g} | {r["eigenvalue_distance_min"]:.17g} | {r["eigenvalue_distance_max"]:.17g} | {r["status"]} |')
    lines += ['', '## Per-column FE information content', '',
              'CSV files preserve numerical precision. Constant columns below use the requested <= threshold; the actual strict-< mask is identical for these data.']
    for r in summaries:
        molecule, view = r['molecule'], r['view']
        lines += ['', f'### {molecule} — {view}', '',
                  f'Width {r["FE_width"]}; active dimensions {r["active_dimensions"]}; loader: {r["loader_status"]}.', '',
                  '| Column | Training mean | Training std | Constant | Active | All-30 min | All-30 max |',
                  '|---:|---:|---:|---|---|---:|---:|']
        for c in column_rows:
            if c['molecule'] == molecule and c['view'] == view:
                lines.append(f'| {c["column_index"]} | {c["train_mean"]:.17g} | {c["train_std_ddof0"]:.17g} | {c["requested_constant_mask_le_1e_minus12"]} | {c["active_std_gt_1e_minus12"]} | {c["all30_min"]:.17g} | {c["all30_max"]:.17g} |')
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:22]))
    print('H2S diagnostics:', [r for r in summaries if r['molecule']=='H2S'])
    print(f'Full-space checks: {len(full_rows)}; input files hash-verified unchanged: {len(hashes)}')
    print(f'Output: {OUTPUT}')


if __name__ == '__main__':
    main()
