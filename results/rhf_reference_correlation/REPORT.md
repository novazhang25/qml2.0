# RHF-reference correlation: current Part 3 ranges

Status: **PASS**; evaluated **270/270**; all 270 successfully validated: **True**.

The sole geometry population is the new Part 3 selected range: 30 points per molecule. No external cutoff or Protocol-7 population is read. Source RHF point indices 1–30 map to geom_index 0–29. E_corr = E_FCI_frozen_core_total - saved_E_RHF_total. All totals are absolute. Min is most negative, max least negative; span = max - min. No absolute-value transformation is applied. Ties use the smallest geometry index.

| Molecule | N | min E_corr (mHa) | geom | max E_corr (mHa) | geom | span (mHa) |
| --- | --- | --- | --- | --- | --- | --- |
| LiH | 30 | -23.3045184237 | 29 | -16.7609113772 | 0 | 6.54360704646 |
| BeH2 | 30 | -39.6241601776 | 29 | -28.4517873578 | 0 | 11.1723728198 |
| H2O | 30 | -71.8870056284 | 29 | -40.7567345409 | 0 | 31.1302710875 |
| NH3 | 30 | -85.1479834963 | 29 | -55.7430162321 | 0 | 29.4049672642 |
| N2 | 30 | -191.82993769 | 29 | -145.763923007 | 0 | 46.0660146829 |
| CO | 30 | -159.831879175 | 29 | -125.870415569 | 0 | 33.9614636063 |
| HF | 30 | -43.9417102816 | 29 | -18.1631129889 | 0 | 25.7785972926 |
| H2S | 30 | -53.2040355437 | 29 | -33.1678959996 | 0 | 20.0361395441 |
| H2O2 | 30 | -118.225286367 | 29 | -96.5324125696 | 0 | 21.6928737971 |

CSV min/max/span also appear in Hartree; machine-readable values retain full binary64 round-trip precision. Incomplete populations are explicit below.

## Actual bond-length ranges

| Molecule | Selected n | Lower (Angstrom) | Upper (Angstrom) | Frozen orbitals | Frozen electrons |
| --- | --- | --- | --- | --- | --- |
| LiH | 1 | 1.26277286421 | 1.75885084185 | 1 | 2 |
| BeH2 | 1 | 1.14860752107 | 1.43246932093 | 1 | 2 |
| H2O | 1 | 0.876681981646 | 1.10213637112 | 1 | 2 |
| NH3 | 1 | 0.937645634804 | 1.12739979233 | 1 | 2 |
| N2 | 1 | 1.06029712024 | 1.20740489474 | 2 | 4 |
| CO | 1 | 1.06809819769 | 1.22286315422 | 2 | 4 |
| HF | 1 | 0.801790953348 | 1.10913444387 | 1 | 2 |
| H2S | 1 | 1.20307019456 | 1.45424154312 | 5 | 10 |
| H2O2 | 1 | 1.30662569949 | 1.48587145877 | 2 | 4 |

## Sources, geometry and total-energy convention

- Range: `/Users/novaz/Desktop/qml2.0/results/bond_length_part3/vibrational_levels.json` → `results[].summary.selected_range_min_A/selected_range_max_A`.
- RHF: `/Users/novaz/Desktop/qml2.0/results/rhf_30_point_scans/rhf_scan_30.json` → `results[].points[].E_RHF_Ha`; nuclei: `results[].symbols` and `results[].points[].cartesian_A`; same-record basis.
- Harmonic geometry: `/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json`; used to validate the new path.
- FCI: `/Users/novaz/Desktop/qml2.0/results/rhf_reference_correlation/fci_current_range_cache.json` → `records[].E_FCI_frozen_core_Ha`. Reuse requires matching molecule/index, basis, exact nuclear coordinates, reference and core convention.
- Per-geometry source fields, nuclei and FCI components are recorded in `provenance.json`.

Saved RHF energies remain unchanged. Their orbitals were not serialized. Only explicit missing-FCI/orbital-rebuild options reconstruct them; the strict local RHF solver must reproduce the saved energy within 1e-9 Ha. The rebuilt value never replaces E_RHF. Canonicalization identifies the lowest occupied RHF core orbitals before CASCI.

Frozen-core counts are recorded in `/Users/novaz/Desktop/qml2.0/codes/current_range_fci.py` → `ACTIVE_SPACES`. They retain the previously inspected paper convention in `encoding_qml/canonical_pipeline/canonical_pipeline/molecules.py` → `MOLECULES[molecule].expected_core_count`; this analysis does not read that project's grid or cutoff. Basis STO-3G, charge 0, spin 0. ncas = n_AO - ncore; nelecas = N_electrons - 2*ncore; all non-core orbitals are active.

PySCF CASCI kernel()[0] is the molecular total. The cache also records mc.e_cas and mc.get_h1eff()[1], verifying total = active + core. The core scalar includes nuclear repulsion. Neither core nor nuclear energy is added twice. UHF and shifted PES energies are not used.

FCI totals computed this run: 270; reused: 0. Successful RHF orbital reconstructions for missing FCI: 270.

## Comparison with 1.6 mHa

Within threshold means -0.0016 <= E_corr_Ha <= 0.0016. E_corr_over_1p6_mHa stays signed. span_over_1p6_mHa = span_Ha/0.0016; span_within_1p6_mHa tests span <= 0.0016 Ha. Equality passes without extra tolerance. Span measures the largest RHF-versus-FCI discrepancy in an energy difference between sampled geometries; a small span does not imply small absolute RHF errors. The reference is frozen-core STO-3G, not experiment or exact all-electron energy. Missing points are excluded from counts and extrema.

| Molecule | N | Within 1.6 mHa | Outside 1.6 mHa | span / 1.6 mHa | span <= 1.6 mHa |
| --- | --- | --- | --- | --- | --- |
| LiH | 30 | 0 | 30 | 4.08975440404 | False |
| BeH2 | 30 | 0 | 30 | 6.98273301238 | False |
| H2O | 30 | 0 | 30 | 19.4564194297 | False |
| NH3 | 30 | 0 | 30 | 18.3781045401 | False |
| N2 | 30 | 0 | 30 | 28.7912591768 | False |
| CO | 30 | 0 | 30 | 21.2259147539 | False |
| HF | 30 | 0 | 30 | 16.1116233079 | False |
| H2S | 30 | 0 | 30 | 12.522587215 | False |
| H2O2 | 30 | 0 | 30 | 13.5580461232 | False |

## Validation and missing geometries

FCI <= RHF tolerance: 1e-08 Ha; violations: 0. Chemical accuracy does not alter this test.

- LiH: 30/30; unevaluated indices [].
- BeH2: 30/30; unevaluated indices [].
- H2O: 30/30; unevaluated indices [].
- NH3: 30/30; unevaluated indices [].
- N2: 30/30; unevaluated indices [].
- CO: 30/30; unevaluated indices [].
- HF: 30/30; unevaluated indices [].
- H2S: 30/30; unevaluated indices [].
- H2O2: 30/30; unevaluated indices [].

No missing, duplicate, geometry-consistency or variational violations detected.
