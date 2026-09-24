#!/usr/bin/env python3
"""Collect saved exact-RHF-equilibrium RHF/FCI energies without any calculations."""
import csv
import json
from pathlib import Path
from current_range_fci import MOLECULES, validate_fci_record
from plots.plot_rhf_fci_equilibrium_overlay import equilibrium_points, load_references
from plots.plot_rhf_bond_scan_30 import PROJECT, load_plot_data
from rhf_bond_scan_30 import sha256


def main():
    harmonic_path = PROJECT/'json/new_rhf_harmonic_bond_ranges.json'
    levels_path = PROJECT/'results/bond_length_part3/vibrational_levels.json'
    scan_path = PROJECT/'results/rhf_30_point_scans/rhf_scan_30.json'
    reference_path = PROJECT/'results/rhf_reference_correlation/fci_at_rhf_equilibrium.json'
    cache_path = PROJECT/'results/rhf_reference_correlation/fci_current_range_cache.json'
    source_paths = [harmonic_path, levels_path, scan_path, reference_path, cache_path,
                    PROJECT/'results/rhf_reference_correlation/per_geometry.csv']
    before = {str(p): sha256(p) for p in source_paths}
    results = load_plot_data(scan_path, levels_path)
    points = equilibrium_points(results, harmonic_path)
    refs = load_references(reference_path, points)
    assert set(refs) == set(MOLECULES), 'Exact equilibrium reference missing; no approximation allowed'
    harmonic = json.loads(harmonic_path.read_text())
    references = json.loads(reference_path.read_text())
    indices = {r['molecule']: i for i, r in enumerate(references['records'])}
    scan_cache = json.loads(cache_path.read_text())['records']
    rows, checks = [], []
    for name in MOLECULES:
        p, fci = points[name], refs[name]
        validate_fci_record(fci, p)
        h = next(r for r in harmonic['results'] if r['summary']['molecule'] == name)
        assert fci['cartesian_A'] == h['equilibrium_cartesian_A'] == p['cartesian_A']
        assert fci['symbols'] == h['symbols'] == p['symbols']
        assert fci['E_RHF_Ha'] == p['E_RHF_Ha']
        convention_keys = ('basis', 'charge', 'spin', 'n_frozen_orbitals', 'n_frozen_electrons',
                           'n_active_orbitals', 'n_active_electrons', 'canonical_rhf_orbitals')
        for scan_record in (r for r in scan_cache if r['molecule'] == name):
            assert all(fci[k] == scan_record[k] for k in convention_keys)
        rhf_e, fci_e = p['E_RHF_Ha'], fci['E_FCI_frozen_core_Ha']
        corr = fci_e - rhf_e
        rows.append(dict(molecule=name, Re_A=p['bond_length_angstrom'],
            E_RHF_Re_Ha=rhf_e, E_FCI_Re_Ha=fci_e, E_corr_Re_Ha=corr, E_corr_Re_mHa=1000*corr,
            RHF_source=f"{harmonic_path}#{p['rhf_source_key']}",
            FCI_source=f"{reference_path}#records[{indices[name]}].E_FCI_frozen_core_Ha",
            FCI_newly_calculated='false'))
        checks.append(dict(molecule=name, exact_cartesian_geometry_identical=True,
                           symbols_identical=True, frozen_core_convention_matches_scan=True,
                           cartesian_A=p['cartesian_A'], symbols=p['symbols']))
    assert before == {str(p): sha256(p) for p in source_paths}, 'Source file changed'
    output = PROJECT/'results/rhf_reference_correlation/equilibrium_energies.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output.with_suffix('.verification.json').write_text(json.dumps(dict(
        inputs_sha256=before, original_inputs_unchanged=True, geometry_checks=checks,
        FCI_already_available=list(MOLECULES), FCI_newly_calculated=[],
        note='All references were saved before this collection run. No electronic-structure calculations or plotting performed.'
    ), indent=2)+'\n')
    print('molecule,Re_A,E_RHF_Re_Ha,E_FCI_Re_Ha,E_corr_Re_Ha,E_corr_Re_mHa,FCI_newly_calculated')
    for r in rows:
        print(','.join(str(r[k]) for k in ('molecule','Re_A','E_RHF_Re_Ha','E_FCI_Re_Ha','E_corr_Re_Ha','E_corr_Re_mHa','FCI_newly_calculated')))
    print(f'PASS: exact Cartesian geometry and frozen-core conventions match for all nine molecules. CSV: {output}')


if __name__ == '__main__':
    main()
