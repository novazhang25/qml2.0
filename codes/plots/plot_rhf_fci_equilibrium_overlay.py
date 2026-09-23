#!/usr/bin/env python3
"""Overlay saved RHF/FCI scans, both referenced to the RHF equilibrium total energy.

Only --compute-missing-references permits new single-point reference calculations.
The original scans, harmonic data, and original FCI cache are never written.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

# Allow direct execution as well as imports through the plots package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from current_range_fci import (MOLECULES, IDENTITY_FIELDS, ACTIVE_SPACES,
                               compute_missing_fci, validate_fci_record)
from plots.plot_rhf_bond_scan_30 import PROJECT, load_plot_data, write_plots
from plots.plot_rhf_reference_correlation import load_data
from rhf_bond_scan_30 import require, sha256


def equilibrium_points(results, harmonic_path):
    harmonic = json.loads(harmonic_path.read_text())
    by_name = {r['summary']['molecule']: (i, r) for i, r in enumerate(harmonic['results'])}
    points = {}
    for result in results:
        s = result['summary']; name = s['molecule']
        i, h = by_name[name]
        require(s['r_e_A'] == h['summary']['s_eq_A'], f'{name}: reference coordinate mismatch')
        require(s['E_re_Ha'] == float(h['input']['metadata_row']['energy_hartree']),
                f'{name}: reference RHF energy mismatch')
        points[name] = dict(molecule=name, geom_index=-1, source_point_index=0,
            basis='sto-3g', symbols=h['symbols'], cartesian_A=h['equilibrium_cartesian_A'],
            bond_length_angstrom=s['r_e_A'], q_A=0.0, E_RHF_Ha=s['E_re_Ha'], charge=0, spin=0,
            rhf_source_file=str(harmonic_path.resolve()),
            rhf_source_key=f'results[{i}].input.metadata_row.energy_hartree')
    return points


def load_references(path, points):
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    require(document['schema_version'] == 'rhf-equilibrium-fci-references-v1', 'Invalid reference cache')
    records = {}
    for record in document['records']:
        name = record['molecule']
        require(name not in records and name in points, 'Duplicate/unexpected reference')
        validate_fci_record(record, points[name])
        records[name] = record
    return records


def combine(results, groups, references):
    combined = copy.deepcopy(results)
    table = []
    for result in combined:
        s = result['summary']; name = s['molecule']; ref = references[name]
        result['fci_reference'] = ref
        for point, row in zip(result['points'], groups[name]):
            require(point['point_index'] == row['geom_index'] + 1 and
                    point['bond_length_A'] == row['r'] and point['E_RHF_Ha'] == row['E_RHF_Ha'],
                    f'{name}: scan identity mismatch')
            point['E_FCI_frozen_core_Ha'] = row['E_FCI_frozen_core_Ha']
            point['DeltaE_RHF_mHa'] = 1000 * (point['E_RHF_Ha'] - s['E_re_Ha'])
            point['DeltaE_FCI_mHa'] = 1000 * (row['E_FCI_frozen_core_Ha'] - s['E_re_Ha'])
        # Reference points are plotted explicitly, independently of the 30 scan points.
        require(abs(1000 * (ref['E_RHF_Ha'] - s['E_re_Ha'])) < 1e-9, f'{name}: RHF zero failed')
        fci_at_re = 1000 * (ref['E_FCI_frozen_core_Ha'] - s['E_re_Ha'])
        minimum = min(p['E_FCI_frozen_core_Ha'] for p in result['points'])
        for i, point in enumerate(result['points']):
            if point['E_FCI_frozen_core_Ha'] != minimum:
                continue
            location = 'left_boundary' if i == 0 else 'right_boundary' if i == 29 else 'interior'
            table.append(dict(molecule=name, R_e_RHF_A=s['r_e_A'], E_RHF_at_Re_Ha=s['E_re_Ha'],
                E_FCI_at_Re_Ha=ref['E_FCI_frozen_core_Ha'],
                E_corr_at_Re_mHa=fci_at_re,
                FCI_sampled_min_coordinate_A=point['bond_length_A'],
                FCI_sampled_min_relative_to_RHF_Re_mHa=point['DeltaE_FCI_mHa'], FCI_min_location=location))
    return combined, table


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'results/plots/rhf_fci_equilibrium_overlay')
    parser.add_argument('--reference-cache', type=Path, default=PROJECT/'results/rhf_reference_correlation/fci_at_rhf_equilibrium.json')
    parser.add_argument('--compute-missing-references', action='store_true')
    parser.add_argument('--references-only', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    source = PROJECT/'results/rhf_reference_correlation'
    provenance = json.loads((source/'provenance.json').read_text())
    inputs = {key: Path(provenance['inputs'][key]) for key in ('ranges_json', 'rhf_scan_json', 'harmonic_json', 'fci_cache')}
    protected = list(inputs.values()) + [source/'per_geometry.csv', source/'provenance.json']
    before = {str(p): sha256(p) for p in protected}
    results = load_plot_data(inputs['rhf_scan_json'], inputs['ranges_json'])
    groups = load_data(source)
    require(len(results) == 9 and sum(len(r['points']) for r in results) == 270, 'Expected all nine scans')
    points = equilibrium_points(results, inputs['harmonic_json'])
    original_cache = json.loads(inputs['fci_cache'].read_text())['records']
    cached = {(r['molecule'], r['geom_index']): r for r in original_cache}
    # Check the exact frozen-core convention of every original FCI record.
    geometry = {(r['molecule'], r['geom_index']): r for r in provenance['per_geometry_provenance']}
    for name, rows in groups.items():
        for row in rows:
            key = name, row['geom_index']
            validate_fci_record(cached[key], geometry[key])
            require(cached[key]['E_FCI_frozen_core_Ha'] == row['E_FCI_frozen_core_Ha'], 'FCI cache mismatch')
    refs = load_references(args.reference_cache, points)
    reused = list(refs); computed = []
    for name in MOLECULES:
        if name in refs:
            continue
        point = points[name]
        # Match full nuclei/basis/charge/spin, not merely a nearby scalar distance.
        matches = [r for r in original_cache if all(r[k] == point[k] for k in
                   ('molecule', 'symbols', 'cartesian_A', 'basis', 'charge', 'spin'))]
        if matches:
            record = dict(matches[0])
            record.update({k: point[k] for k in IDENTITY_FIELDS})
            validate_fci_record(record, point)
            reused.append(name)
        else:
            print(f'{name}: exact frozen-core FCI at RHF R_e is missing from saved results.', flush=True)
            require(args.compute_missing_references,
                    'Use --compute-missing-references to calculate only missing exact-equilibrium references.')
            print(f'{name}: calculating only the reference at R_e; reconstructing RHF orbitals.', flush=True)
            record = compute_missing_fci(point, threads=args.threads, allow_rhf_rebuild=True)
            computed.append(name)
        refs[name] = record
        args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.reference_cache.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(schema_version='rhf-equilibrium-fci-references-v1',
            records=list(refs.values())), indent=2, allow_nan=False)+'\n')
        temporary.replace(args.reference_cache)
    require(before == {str(p): sha256(p) for p in protected}, 'Original input changed')
    if args.references_only:
        print(f'References saved: {args.reference_cache}; computed={computed}; reused={reused}')
        return
    combined, table = combine(results, groups, refs)
    filenames = write_plots(combined, args.output, relative=True)
    write_csv(args.output/'reference_energies_and_fci_minima.csv', table)
    write_csv(args.output/'plotted_points.csv', [p for result in combined for p in result['points']])
    writer = csv.DictWriter(sys.stdout, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
    for row in table:
        if row['FCI_min_location'] != 'interior':
            print(row['molecule'] + ': FCI equilibrium minimum is not established within the current scan.')
    require(before == {str(p): sha256(p) for p in protected}, 'Original input changed')
    audit = dict(inputs_sha256=before, reference_cache=str(args.reference_cache.resolve()),
        reference_cache_sha256=sha256(args.reference_cache), computed_this_run=computed,
        reused_this_run=reused, original_inputs_unchanged=True, scan_points=270,
        active_spaces=ACTIVE_SPACES, reference_geometry='optimized RHF equilibrium nuclei',
        energy_reference='1000 * (E_method(R) - E_RHF(R_e_RHF))',
        reference_zero_tolerance_mHa=1e-9, RHF_reference_zero_checks_passed=True, FCI_reference_is_correlation_energy=True)
    (args.output/'verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    notes = ['# RHF and frozen-core FCI at the same RHF reference geometry', '',
        'R_e is the optimized RHF geometry, not an FCI equilibrium geometry. Both methods subtract the single saved RHF total energy at these exact nuclei.',
        'All 30 scan points are retained. The RHF reference point is zero; the FCI reference point is the signed frozen-core correlation energy at R_e. Negative FCI relative energies and displaced minima are preserved.',
        'The local RHF harmonic curve and both n=0/n=1 lines and labels are hidden. The compact panel order is LiH/HF/BeH2, NH3/H2O/H2S, N2/CO/H2O2; molecule labels are inside the top-right corners and the legend is inside LiH. Selected-range shading remains; bracketed bond-range text stays hidden. Saved vibrational levels, selected range, force constants and frequencies are unchanged. No scan energies were recalculated.',
        'FCI uses the existing canonical-RHF frozen-core CASCI convention (active-space FCI), with core and nuclear energy included.',
        'Exact FCI equilibrium references were absent from the original scan dataset and were calculated separately; subsequent plotting reuses the reference cache.',
        'Reference calculations rebuild RHF orbitals at fixed nuclei and verify the stored RHF total within 1e-9 Ha; they never replace the original RHF total.', '',
        '## Inputs', *[f'- {p} (SHA256 {value})' for p, value in before.items()],
        f'- {args.reference_cache.resolve()} (SHA256 {sha256(args.reference_cache)})', '',
        '## Outputs', *[f'- {name}' for name in filenames],
        '- reference_energies_and_fci_minima.csv', '- plotted_points.csv', '- verification.json']
    (args.output/'README.md').write_text('\n'.join(notes)+'\n')
    print(f'PASS: source hashes unchanged; RHF reference zeros verified; FCI uses the same RHF energy reference. Output: {args.output.resolve()}')


if __name__ == '__main__':
    main()
