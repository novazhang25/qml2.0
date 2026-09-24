#!/usr/bin/env python3
"""Audit the H2S Ne-core loader against saved descriptors and raw FE spectra.

This reads existing full-AO data only. It never runs electronic-structure
calculations, prepares a model, fits a scaler, or modifies a reference file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import run1


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / 'results/descriptor'
REFERENCE_CSV = ROOT / 'results/audits/fe_all_molecules/fe_eigenvalues.csv'
REFERENCE_VIEW = 'saved_6AO_diagnostic'
GEOMETRY_IDS = tuple(f'{i:03d}' for i in range(1, 31))
SELECTED_LABELS = ('0 S 3s', '0 S 3px', '0 S 3py', '0 S 3pz',
                   '1 H 1s', '2 H 1s')
CORE_LABELS = ('0 S 1s', '0 S 2s', '0 S 2px', '0 S 2py', '0 S 2pz')
TOLERANCE = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _max_abs(value):
    _require(np.isrealobj(value) and np.isfinite(value).all(),
             'Audit encountered nonfinite or nonreal values')
    return float(np.max(np.abs(value), initial=0.))


def _reference_spectra(path):
    values = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['molecule'] != 'H2S' or row['view'] != REFERENCE_VIEW:
                continue
            key = (row['geometry_id'], int(row['column_index']))
            _require(key not in values, f'{path}: duplicate H2S FE value {key}')
            values[key] = float(row['eigenvalue'])
    expected = {(geometry_id, column) for geometry_id in GEOMETRY_IDS
                for column in range(6)}
    _require(set(values) == expected,
             f'{path}: expected all 30 H2S IDs, with exactly six FE columns each')
    spectra = {geometry_id: np.asarray([values[geometry_id, j] for j in range(6)])
               for geometry_id in GEOMETRY_IDS}
    _require(all(np.isfinite(value).all() for value in spectra.values()),
             f'{path}: FE reference must contain finite real values')
    return spectra


def audit_h2s(input_dir=INPUT_DIR, reference_csv=REFERENCE_CSV):
    """Return a JSON-serializable audit; fail on any saved-block/FE mismatch."""
    input_dir, reference_csv = Path(input_dir).resolve(), Path(reference_csv).resolve()
    paths = sorted((input_dir / 'H2S').glob('*.npz'))
    _require(tuple(path.stem for path in paths) == GEOMETRY_IDS,
             f'{input_dir}: expected exactly H2S geometry files 001..030')
    source_hashes_before = {str(path): _sha256(path) for path in paths}
    reference_hash_before = _sha256(reference_csv)
    references = _reference_spectra(reference_csv)
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            raw = {key: saved[key] for key in saved.files}
        labels = tuple(' '.join(str(label).split()) for label in raw['ao_labels'])
        _require(len(labels) == 11 and len(set(labels)) == 11,
                 f'{path}: expected eleven unique full-AO labels')
        selected = np.asarray([i for i, label in enumerate(labels)
                               if label in SELECTED_LABELS], dtype=int)
        core = np.asarray([i for i, label in enumerate(labels)
                           if label in CORE_LABELS], dtype=int)
        selected_labels = tuple(labels[i] for i in selected)
        core_labels = tuple(labels[i] for i in core)
        _require(selected_labels == SELECTED_LABELS and core_labels == CORE_LABELS,
                 f'{path}: unexpected H2S AO labels or ordering')
        _require(np.array_equal(raw['valence_indices'], selected)
                 and np.array_equal(raw['core_indices'], core),
                 f'{path}: saved AO selection is not the sulfur Ne-core convention')

        # Transform all eleven AOs before taking either index of the six-AO block.
        for key in ('P_L', 'P_AO', 'S_half'):
            _require(raw[key].shape == (11, 11), f'{path}: {key} is not full AO')
            _max_abs(raw[key])
        full_transform = raw['S_half'] @ raw['P_AO'] @ raw['S_half']
        saved_block = raw['P_L'][np.ix_(selected, selected)]
        transformed_block = full_transform[np.ix_(selected, selected)]
        sample, metadata = run1.load_record(path, 'H2S', path.stem)
        _require(sample.pij.shape == (6, 6), f'{path}: loader must return a 6x6 block')
        _require(metadata['source_selected_ao_indices'] == selected.tolist(),
                 f'{path}: loader selected different source AO indices')

        pairs = raw['pair_indices']
        reconstructed = np.diag(raw['P_mu']).copy()
        reconstructed[pairs[:, 0], pairs[:, 1]] = raw['P_munu']
        reconstructed[pairs[:, 1], pairs[:, 0]] = raw['P_munu']
        fe = np.linalg.eigvalsh(sample.pij)
        residuals = {
            'loader_block_vs_saved_P_L_block': _max_abs(sample.pij - saved_block),
            'loader_FE_vs_saved_raw_FE': _max_abs(fe - references[path.stem]),
            'saved_full_P_L_vs_full_AO_Lowdin_transform':
                _max_abs(raw['P_L'] - full_transform),
            'loader_block_vs_full_transform_then_selection':
                _max_abs(sample.pij - transformed_block),
            'loader_block_vs_saved_P_mu_P_munu_reconstruction':
                _max_abs(sample.pij - reconstructed),
        }
        _require(np.array_equal(sample.pij, saved_block),
                 f'{path}: loader does not reproduce the saved P_L block exactly')
        _require(np.array_equal(fe, references[path.stem]),
                 f'{path}: loader FE does not reproduce the saved raw FE exactly')
        _require(all(value <= TOLERANCE for value in residuals.values()),
                 f'{path}: full-space/descriptor check exceeds {TOLERANCE}: {residuals}')
        rows.append(dict(geometry_id=path.stem, full_AO_dimension=11,
                         valence_shape=[6, 6], descriptor_qubit_dimension=6,
                         selected_ao_indices=selected.tolist(),
                         selected_ao_labels=list(selected_labels),
                         core_ao_indices=core.tolist(), core_ao_labels=list(core_labels),
                         raw_FE=fe.tolist(), saved_raw_FE=references[path.stem].tolist(),
                         max_abs_residuals=residuals, status='PASS'))

    source_hashes_after = {str(path): _sha256(path) for path in paths}
    reference_hash_after = _sha256(reference_csv)
    _require(source_hashes_before == source_hashes_after,
             'Source descriptor data changed during the audit')
    _require(reference_hash_before == reference_hash_after,
             'Saved FE reference changed during the audit')
    report = dict(
        status='PASS', molecule='H2S', geometry_count=len(rows),
        full_AO_dimension=11, valence_shape=[6, 6], descriptor_qubit_dimension=6,
        selected_ao_indices=rows[0]['selected_ao_indices'],
        selected_ao_labels=rows[0]['selected_ao_labels'],
        core_ao_indices=rows[0]['core_ao_indices'], core_ao_labels=rows[0]['core_ao_labels'],
        max_abs_residuals={key: max(row['max_abs_residuals'][key] for row in rows)
                           for key in rows[0]['max_abs_residuals']},
        reference_csv=str(reference_csv), reference_view=REFERENCE_VIEW,
        reference_kind='Previously saved raw eigvalsh of the saved six-AO P_L block; unscaled',
        input_dir=str(input_dir), numpy_version=np.__version__,
        nonexact_check_tolerance=TOLERANCE, exact_block_match=True, exact_FE_match=True,
        source_hashes_before=source_hashes_before, source_hashes_after=source_hashes_after,
        reference_sha256_before=reference_hash_before, reference_sha256_after=reference_hash_after,
        inputs_unchanged=True, per_geometry=rows)
    # Validate the same strict JSON representation used by the CLI before return.
    json.dumps(report, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR)
    parser.add_argument('--reference-csv', type=Path, default=REFERENCE_CSV)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'results/audits/h2s_ne_core_loader')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('--output-dir must be new; existing audit results are never overwritten')
    report = audit_h2s(args.input_dir, args.reference_csv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'audit.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    flat_rows = []
    for row in report['per_geometry']:
        flat = {key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items() if key != 'max_abs_residuals'}
        flat.update(row['max_abs_residuals'])
        flat_rows.append(flat)
    with (args.output_dir / 'per_geometry.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = [
        '# H2S Ne-core valence-loader audit', '',
        'PASS: all 30 saved H2S geometries load as six-AO blocks.',
        'Retained labels: S 3s, 3px, 3py, 3pz, and both H 1s AOs.',
        f"Stored full-AO indices: {report['selected_ao_indices']}.",
        'The full eleven-AO Lowdin transformation precedes subspace selection.', '',
        'The loader block and raw FE match their saved references exactly.',
        'The saved raw FE reference is the historical saved_6AO_diagnostic view in',
        f'`{report["reference_csv"]}`; it contains unscaled eigenvalues.', '',
        '| Comparison | Maximum absolute residual over 30 geometries |',
        '|---|---:|',
    ]
    lines.extend(f'| {key} | {value:.17g} |'
                 for key, value in report['max_abs_residuals'].items())
    lines += ['', 'Descriptor and reference SHA-256 hashes were unchanged.',
              'No electronic-structure calculations, training, or data regeneration were run.']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({key: report[key] for key in
                      ('status', 'geometry_count', 'valence_shape', 'max_abs_residuals')},
                     indent=2, allow_nan=False))
    print(f'Audit output: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
