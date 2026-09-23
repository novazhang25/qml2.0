# RHF scan over the current bond ranges

`codes/rhf_bond_scan_30.py` calculates 30 fresh RHF single-point energies per molecule using the selected ranges in `results/bond_length_part3/vibrational_levels.json`, schema `part3-zero-energy-v2`. All nine currently select n=1. The default run requests **270 data points**.

The saved scan is complete: **270/270 accepted points, overall status PASS**. Its Part 3 metadata has been migrated to the current zero-energy rule, and the overlay plots have been regenerated. This update preserved all existing RHF energies, coordinates and solver diagnostics; it performed no new SCF calculations. All 13 focused scanner and overlay tests passed.

To run a new production scan:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py
```

## Part 3 decision and metadata-only updates

Part 3 reuses the saved absolute equilibrium RHF energy `E_re`, force constant, effective mass and angular frequency. It converts the saved frequency using `hbar_omega_Ha = hbar_J_s * omega_rad_per_s / Hartree_J`, then calculates

```text
E_n0_Ha = 0.5 * hbar_omega_Ha
E_n1_Ha = 1.5 * hbar_omega_Ha
E_total_n0_Ha = E_re_Ha + E_n0_Ha
E_total_n1_Ha = E_re_Ha + E_n1_Ha
n0_pass = E_total_n0_Ha < 0
n1_pass = E_total_n1_Ha < 0
selected_n = 1 if n1_pass else 0 if n0_pass else None
```

Both signs are negative for every molecule, so all nine select n=1 and retain their existing ranges. Part 3 uses no dissociation-energy criterion. **The supplied `E_re` is an absolute RHF total energy including nuclear repulsion, and the workflow does not establish zero as the physical allowed-state threshold.** The numerical sign results therefore carry `energy_reference_status=UNVALIDATED_ZERO_REFERENCE`; they do not establish physical vibrational binding. No energy shift is introduced (`reference_shift_Ha=0`).

To update existing scan metadata after regenerating Part 3, without executing SCF:

```bash
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py --rebind-part3
```

This command requires a completed PASS scan and refuses changes to the numeric grid, selected level/range, basis, coordinate, equilibrium energy, force constant, mass, frequency, harmonic input or geometry source. It replaces only the Part 3 summary metadata and records the new input hash. A fingerprint verifies exact preservation of every saved atom, point, energy, geometry check and solver diagnostic. Original numerical-run source hashes remain recorded separately from metadata-migration provenance. Point CSVs and XYZ files are left untouched. Repeating a rebind against the same current inputs makes no changes.

## Sampling and geometry

For each saved interval, the 30 bond coordinates are

```text
r_i = r_min + i * (r_max - r_min) / 29, i = 0,...,29
q_i = r_i - r_e
```

Both endpoints are included exactly. The 30-point symmetric grid does not include the equilibrium midpoint; no extra equilibrium calculation is added. Every scan energy is a fresh RHF solve at its own geometry. No old scan energies or harmonic approximations are substituted. The overlay includes the saved optimized equilibrium in the orange RHF curve without changing these 30 scan samples.

| Molecule | Current minimum [Angstrom] | Current maximum [Angstrom] | Coordinate |
| --- | --- | --- | --- |
| LiH | 1.262772864212423 | 1.7588508418515771 | Li-H |
| BeH2 | 1.1486075210698299 | 1.4324693209261700 | Both Be-H bonds together |
| H2O | 0.8766819816463665 | 1.1021363711241332 | Both O-H bonds together |
| NH3 | 0.9376456348042379 | 1.1273997923280357 | All three N-H bonds together |
| N2 | 1.0602971202446738 | 1.2074048947433262 | N-N |
| CO | 1.0680981976887212 | 1.2228631542192787 | C-O |
| HF | 0.8017909533483780 | 1.1091344438716220 | H-F |
| H2S | 1.2030701945616575 | 1.4542415431205170 | Both S-H bonds together |
| H2O2 | 1.3066256994935583 | 1.4858714587716013 | O-O, carrying each OH group rigidly |

These values document the current input, not constants embedded in the scanner. The program reads the full-precision saved endpoints on each run.

The existing `StretchPath` preserves equilibrium angles, fixed bond lengths and torsions. For polyatomic molecules the tabulated coordinate is the common participating bond coordinate. H2O2 retains both OH lengths, both OOH angles and the HOOH torsion. All 270 geometries are validated before the first RHF calculation.

## Method and output

The basis comes from the existing matching RHF equilibrium/harmonic records: currently **STO-3G** for all molecules. The solver remains conventional all-electron, restricted closed-shell RHF without density fitting. Existing convergence, orbital consistency and internal RHF stability checks are reused. Previous accepted orbitals seed the next nearby point, with a fresh-guess fallback in the existing solver. Equilibrium geometry and force constants are not recomputed. The scanner consumes and validates the current saved Part 3 selection.

`E_RHF_Ha` is the absolute RHF total energy at the geometry, including nuclear repulsion. `delta_E_RHF_Ha = E_RHF_Ha - E_re_Ha` is the change from the saved equilibrium energy. No vibrational energy is added to these RHF data points.

Outputs go to `results/rhf_30_point_scans/`:

- `rhf_scan_30.csv`: combined point data; 270 rows after a complete default run.
- `rhf_scan_30.json`: all point geometries, diagnostics, source hashes, current range metadata and overall status.
- `summary.csv`: requested, accepted and failed point counts per molecule.
- `molecules/<molecule>.csv`: 30 rows per completed molecule.
- `geometries/<molecule>.xyz`: a 30-frame XYZ trajectory per completed molecule, in grid order.

Outputs are checkpointed after every point. An unsuccessful point has blank energy columns and an explicit failure, with its solver diagnostics retained in JSON. The program still attempts the other grid points and exits unsuccessfully if any molecule lacks 30 accepted points. `status=RUNNING` or `PENDING` identifies incomplete calculations. Internal convergence retries can perform additional SCF iterations at a grid point; they do not create extra data points.

Running the command again calculates all requested points afresh and replaces these output files. Use `--output-dir` for a separate run directory, `--molecules LiH` to choose a subset, or `--threads 2` to change the default one PySCF thread. The fixed number of points is 30.

The passing tests in `tests/test_rhf_bond_scan_30.py` and `tests/test_plot_rhf_bond_scan_30.py` cover inclusive sampling, preservation of all nine saved ranges, input mismatch rejection, fake-solver calls and failed-point handling, metadata migration and idempotence, unchanged plotted energies, and invalid zero-test flags or energy shifts. These tests perform no RHF calculations.

## Overlay on plot 1

The latest plotting-code update uses absolute `E (Hartree)` and adds the saved optimized equilibrium point to every panel. Saved production figures have not been regenerated for these presentation updates; run the command below to replace them.

To overlay the completed 30-point RHF scans on the local harmonic-range panel, using the same 3x3 layout and colors as `tests/plot.py`:

```bash
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/plots/plot_rhf_bond_scan_30.py
```

The plotting command reads saved results only and performs no RHF calculations. It requires 30 accepted points per molecule and verifies that the scan metadata is bound to the current Part 3 range file, with matching ranges, energies, zero-test flags and energy-reference warning.

Outputs go to `results/plots/rhf_30_overlay/`: `overview.png`, `overview.svg`, individual molecule PNG/SVG files, `plotted_points.csv`, and figure notes. Orange points and connecting lines show the 30 actual RHF samples over the blue dashed harmonic curve; the n=0/n=1 segments and green selected-range shading are retained.

The saved optimized equilibrium `(r_e_A, E_re_Ha)` is included in coordinate order in the orange RHF curve, using the same circular marker and shared RHF legend entry. The displayed curve contains 30 scan points plus the saved equilibrium. The equilibrium point has no separate numerical annotation box; its saved values are retained unchanged in the curve. “Exact” here means those saved optimized values, not an inferred curve-fit minimum or the lowest of the 30 samples. The marker reuses the existing equilibrium calculation; it is not a new scan point or a new RHF calculation, and the 30-point grid and scan records remain unchanged.

The x axis is the bond coordinate in Angstrom. The y axis is absolute `E (Hartree)`: `E_RHF_Ha` for the RHF samples, `E_re + k*q^2/2` for the harmonic potential, and `E_total_n0`/`E_total_n1` for the vibrational levels. The harmonic minimum is therefore at the saved absolute equilibrium energy `E_re`. No equilibrium energy or sampled minimum is subtracted for display. Part 3 continues to compare the unshifted totals `E_re + E_n` with zero. Generated figure notes retain the unvalidated physical zero-reference explanation; the figures have no bottom text footer.
