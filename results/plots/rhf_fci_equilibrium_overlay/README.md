# RHF and frozen-core FCI at the same RHF reference geometry

R_e is the optimized RHF geometry, not an FCI equilibrium geometry. Both methods subtract the single saved RHF total energy at these exact nuclei.
All 30 scan points are retained. The RHF reference point is zero; the FCI reference point is the signed frozen-core correlation energy at R_e. Negative FCI relative energies and displaced minima are preserved.
The local RHF harmonic curve and both n=0/n=1 lines and labels are hidden. The compact panel order is LiH/HF/BeH2, NH3/H2O/H2S, N2/CO/H2O2; molecule labels are inside the top-right corners and the legend is inside LiH. Selected-range shading remains; bracketed bond-range text stays hidden. Saved vibrational levels, selected range, force constants and frequencies are unchanged. No scan energies were recalculated.
FCI uses the existing canonical-RHF frozen-core CASCI convention (active-space FCI), with core and nuclear energy included.
Exact FCI equilibrium references were absent from the original scan dataset and were calculated separately; subsequent plotting reuses the reference cache.
Reference calculations rebuild RHF orbitals at fixed nuclei and verify the stored RHF total within 1e-9 Ha; they never replace the original RHF total.

## Inputs
- /Users/novaz/Desktop/qml2.0/results/bond_length_part3/vibrational_levels.json (SHA256 dbb2b85f3b594de3879d6aff03e5b94ccc6eddf4414f4b3e40691efc0b85f4bc)
- /Users/novaz/Desktop/qml2.0/results/rhf_30_point_scans/rhf_scan_30.json (SHA256 261a03ce80c9eac0c3166579483bd4138148faf2349138b605973f9077a4c2c5)
- /Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json (SHA256 d698dde31c8252930dcd39aeea1721c5f4bde42efc81d3571a566a88cbfb5999)
- /Users/novaz/Desktop/qml2.0/results/rhf_reference_correlation/fci_current_range_cache.json (SHA256 ecfb8ace519b992bc59e53b175d338ad6c4dcee427bb4b3371ba8636fa32e290)
- /Users/novaz/Desktop/qml2.0/results/rhf_reference_correlation/per_geometry.csv (SHA256 f5c217dcd0d8f8569bce5b1b39726a34e20498dcbb2f127249a6d4e42c6f002e)
- /Users/novaz/Desktop/qml2.0/results/rhf_reference_correlation/provenance.json (SHA256 216b8c39ebb9f4bd385fd1ef5eab2f9a6e09ee00e1acc0ae2fa5eadbb82c7052)
- /Users/novaz/Desktop/qml2.0/results/rhf_reference_correlation/fci_at_rhf_equilibrium.json (SHA256 b75c3f05f76956476185e35b534791711d2e5f8c4258ea063147945ba6529642)

## Outputs
- overview.png
- overview.svg
- LiH.png
- LiH.svg
- HF.png
- HF.svg
- BeH2.png
- BeH2.svg
- NH3.png
- NH3.svg
- H2O.png
- H2O.svg
- H2S.png
- H2S.svg
- N2.png
- N2.svg
- CO.png
- CO.svg
- H2O2.png
- H2O2.svg
- reference_energies_and_fci_minima.csv
- plotted_points.csv
- verification.json
