# Part 3: zero-energy criterion and reference audit

The current implementation is `codes/bond_length_part3.py`. The pipeline calls the same `compute_levels` arithmetic from `codes/bond_length.py:evaluate_part3`. Only Part 3 and its downstream presentation/metadata have changed. No new RHF calculation, geometry optimization, nuclear Hessian or harmonic-frequency calculation was performed for this revision.

## Exact input energy and zero-reference finding

**UNVALIDATED_ZERO_REFERENCE:** the supplied `E_re` is an unshifted absolute RHF total molecular (Born–Oppenheimer) energy, including nuclear repulsion. It is not a well energy already referenced to an allowed-vibration threshold. The existing workflow does not define zero Hartree as that physical threshold. The requested `< 0` calculation is reported literally, with no invented reference transformation; its pass flags do not establish physical vibrational binding.

The trace is:

1. `tests/molecules_rhf.py:optimize.energy_gradient` calls `energy = mf.kernel()` and returns that converged RHF energy to BFGS.
2. Its `main` writes the optimized `result.fun` to `energy_hartree` in `results/rhf_geometries/rhf_equilibrium_summary.csv`.
3. The current pipeline's unchanged `optimize_equilibrium` likewise returns `float(mf.e_tot)` and stores it as `E_eq_Eh`/`energy_hartree`; `reuse_equilibrium` reads that saved value directly.
4. Installed PySCF `scf.hf.energy_tot` returns `mf.energy_elec(...)[0] + mf.energy_nuc()`. Thus “electronic energy” here means the molecular RHF energy at fixed nuclei, including the internuclear term, not just the electron-only expectation value.
5. `bond_length_part3.analyze_molecule` reads `float(metadata["energy_hartree"])`. The saved Part 2 stationary energy is a consistency check, never a replacement for Part 1's optimized value.

Every result records `reference_shift_Ha = 0`, `energy_reference_status = UNVALIDATED_ZERO_REFERENCE`, and `physical_vibrational_binding_established = false`.

## Audit of the removed implementation

These equations are historical audit evidence only:

```text
Old standalone criterion: E_re + E_n < E_diss
Equivalent old relative criterion: E_n < D_e, where D_e = E_diss - E_re
Legacy selector: E_n_rel = -D_e + (n + 0.5)*hbar_omega; E_n_rel < 0
```

They test a large-separation threshold rather than the newly requested literal zero on the unshifted input scale. The earlier signed gap was algebraically consistent with its threshold convention; it was not the absolute vibrational-level energy. The audit found no direct comparison of a positive excitation with absolute `E_diss`, no level defined as `E_diss - E_n`, and no double subtraction of `E_re`. The reference convention, fields and decision have now been explicitly replaced.

Every former Part 3 entry point or consumer found in the audit:

| File / function before this revision | How the old threshold entered | Current treatment |
| --- | --- | --- |
| `codes/bond_length_part3.py:compute_levels` | Accepted the absolute plateau, formed depth, classified totals and relative excitations, and emitted gaps/bound flags | Accepts only `E_re_Ha` and `hbar_omega_Ha`; strict zero test |
| Same file: `validate_plateau`, `analyze_molecule`, CLI and report | Loaded old selection JSON, validated the plateau, supplied its energy, and printed depth/gap tables | Plateau dependency and CLI input removed; reads only Parts 1/2 |
| `tests/rhf_bound_level_selection.py:select_level`, `scan_one` | Scanned the asymptote, formed depth and `-D_e + E_n`; old q=0 reevaluation also supplied equilibrium energy | Selector/Part 3 scan/report removed; launcher delegates to the new code. Raw RHF/geometry helpers remain for independent scanning |
| `codes/bond_length.py:selection_cache_matches`, `cached_records`, `process_molecule` stage 3 | Validated/reused a plateau cache or called the plateau scanner/selector | Selection cache and scan settings removed; stage 3 calls `evaluate_part3` from saved inputs |
| Same file: summary and report output | Exported plateau/depth, dissociation-relative levels and bound flags | Exports excitation energies, absolute totals, zero pass flags and reference warning |
| `codes/rhf_bond_scan_30.py:validate_range_record` and saved metadata | Required plateau/depth evidence and embedded old Part 3 summaries | Requires v2 zero-sign results; metadata migrated after unchanged-grid/input checks |
| `codes/plots/bond_length_part3_plots.py` | Drew threshold/depth comparisons and used old fields | Draws absolute levels and literal zero with the reference warning |
| `codes/plots/plot_rhf_bond_scan_30.py` and `tests/plot.py` | Consumed old fields/labels; legacy plot helper contained threshold logic | Overlay validates new fields and displays absolute RHF/harmonic/level energies; legacy launcher delegates to it |
| `tests/hartree.py` | Entered the old selector | Delegates to the new Part 3 |

Old criterion values were present in CSV/JSON tables, Markdown reports and PNG/SVG figures. They are replaced in current outputs or explicitly archived. No LaTeX source was found. The raw plateau scan CSVs remain historical numerical observations, not inputs to current Part 3. No active Part 3 decision, table schema or plot calculation uses `E_diss` or `D_e`.

## New equations and strict decision

```text
Q = hbar_J_s * saved_omega_rad_per_s / Hartree_J       [Ha]
E_n0 = 0.5 * Q                                       [Ha]
E_n1 = 1.5 * Q                                       [Ha]
E_total_n0 = saved_E_re + E_n0                        [Ha]
E_total_n1 = saved_E_re + E_n1                        [Ha]
n0_pass = E_total_n0 < 0
n1_pass = E_total_n1 < 0
selected_n = 1 if n1_pass else 0 if n0_pass else None
```

Equality at zero fails. No tolerance modifies the sign decision. Only n=0 and n=1 are tested; selecting 1 makes no claim about n>=2. Saved masses are reused as the effective masses of the existing collective coordinate, rather than replaced with a diatomic approximation for polyatomic molecules.

## Unit checks

The saved Part 2 angular frequency in rad/s is used directly. Independent checks, which never overwrite that frequency, verify:

```text
k_SI = k_Ha_per_Bohr2 * Hartree_J / Bohr_m**2           [N/m]
mu_kg = mu_eff_amu * amu_kg                            [kg]
omega_check = sqrt(k_SI / mu_kg)                       [rad/s]
Q_check = 2*pi*hbar_J_s*c_m_per_s*100*wavenumber_cm1 / Hartree_J
```

All nine saved frequencies match the k/mass check exactly in floating-point arithmetic (maximum relative discrepancy 0). The largest quantum conversion residual is `3.469446951953614e-18 Ha`. Angstrom curvature, SI curvature, mass conversions, the existing harmonic turning points and the Part 1/2 energy agreement are also checked. All energy additions and zero comparisons use Hartree.

## Corrected molecule-by-molecule table

| Molecule | E_re [Ha] | k [Ha/Bohr^2] | mu_eff [amu] | omega [rad/s] | hbar_omega [Ha] | E_n0 [Ha] | E_n1 [Ha] | E_total_n0 [Ha] | E_total_n1 [Ha] | n0_pass | n1_pass | selected_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | -7.86338212892 | 0.116210397501 | 0.880161046804 | 3.51840856678e+14 | 0.00851062334631 | 0.00425531167315 | 0.0127659350195 | -7.85912681725 | -7.8506161939 | YES | YES | 1 |
| BeH2 | -15.5613528078 | 0.473247780631 | 2.016 | 4.69141527176e+14 | 0.0113479908832 | 0.00567399544158 | 0.0170219863247 | -15.5556788123 | -15.5443308215 | YES | YES | 1 |
| H2O | -74.9659011923 | 1.24689394745 | 1.92283768262 | 7.7973727198e+14 | 0.0188609426818 | 0.00943047134088 | 0.0282914140226 | -74.956470721 | -74.9376097783 | YES | YES | 1 |
| NH3 | -55.4554197788 | 1.62924775325 | 2.93261189058 | 7.21723951054e+14 | 0.017457667553 | 0.00872883377648 | 0.0261865013294 | -55.446690945 | -55.4292332775 | YES | YES | 1 |
| N2 | -107.500654263 | 1.8886404537 | 7.0035 | 5.02830862315e+14 | 0.0121628969315 | 0.00608144846575 | 0.0182443453972 | -107.494572814 | -107.482409917 | YES | YES | 1 |
| CO | -111.225449514 | 1.57382886196 | 6.86054941092 | 4.63771602568e+14 | 0.0112180986183 | 0.00560904930916 | 0.0168271479275 | -111.219840465 | -111.208622366 | YES | YES | 1 |
| HF | -98.5728473472 | 0.725274201861 | 0.957213059852 | 8.42852647653e+14 | 0.0203876306133 | 0.0101938153067 | 0.03058144592 | -98.5626535319 | -98.5422659012 | YES | YES | 1 |
| H2S | -394.311630059 | 0.794499972345 | 1.95899192757 | 6.1664536031e+14 | 0.014915937988 | 0.007457968994 | 0.022373906982 | -394.30417209 | -394.289256152 | YES | YES | 1 |
| H2O2 | -148.764996621 | 0.709124310059 | 8.46231114857 | 2.80298994797e+14 | 0.00678010846038 | 0.00339005423019 | 0.0101701626906 | -148.761606567 | -148.754826459 | YES | YES | 1 |

Energy columns are Hartree; k is Ha/Bohr^2, mass amu and omega rad/s. CSV/JSON retain full precision.

All nine molecules have `n0_pass=YES`, `n1_pass=YES`, and `selected_n=1`. **No molecule fails both tests.** These are sign-test outcomes subject to the explicit energy-reference finding above.

## Changed files and artifacts

Code:

- `codes/bond_length_part3.py`: current analysis, units/reference checks, table and report.
- `codes/bond_length.py`: Part 3 integration and output/cache changes only; equilibrium and harmonic calculation functions unchanged.
- `codes/plots/bond_length_part3_plots.py`: absolute energy / zero-comparison plots.
- `codes/rhf_bond_scan_30.py`: new Part 3 schema validation and metadata-only migration; RHF numerical solver and grid retained.
- `codes/plots/plot_rhf_bond_scan_30.py`: absolute-energy overlay, inputs and zero/reference validation.
- `tests/rhf_bound_level_selection.py`, `tests/hartree.py`, `tests/plot.py`: remove obsolete Part 3 entry paths and delegate to the current implementation; retain unrelated raw numerical helpers.

Tests:

- `tests/test_bond_length_part3.py`
- `tests/test_bond_length_part3_plots.py`
- `tests/test_bond_length.py`
- `tests/test_rhf_bound_level_selection.py`
- `tests/test_rhf_bond_scan_30.py`
- `tests/test_plot_rhf_bond_scan_30.py`

Documentation:

- This `md/PART3_CORRECTION.md`, `md/BOND_LENGTH_PIPELINE.md`, `md/RHF_30_POINT_SCAN.md`, and `md/RHF_BOUND_LEVEL_SELECTION_REPORT.md`.
- Explicit archive notices under `results/part3_history/pre_zero_criterion/` and the pre-existing `results/plots/v1/` archive.

Current output:

- `results/bond_length_part3/vibrational_levels.csv`, `.json`, `PART3_REPORT.md`, and all nine PNG/SVG plot pairs.
- `results/bond_length/bond_length.csv`, `.json`, and `BOND_LENGTH_REPORT.md` regenerated in reuse-only mode.
- `results/rhf_30_point_scans/rhf_scan_30.json`: updated Part 3 summaries and separate metadata-migration provenance. Its point records are unchanged. `summary.csv` remains byte-identical.
- `results/plots/rhf_30_overlay/`: regenerated overview, molecule plots and notes. `plotted_points.csv` remains byte-identical.
- `json/rhf_bound_level_selection.json` and `results/rhf_bound_level_selection.csv` are compatibility symlinks to the current v2 Part 3 outputs, preventing stale parallel tables. Their old contents are preserved in the archive. Existing `tests/` symlinks resolve through these current aliases.

Historical results and reports were copied before replacement to `results/part3_history/pre_zero_criterion/`. They are not cache inputs. The pre-existing `results/plots/v1/` figures remain explicitly historical. No unrelated user changes were reverted.

## Current overlay energy axis and equilibrium point

The overlay on plot 1 uses absolute `E (Hartree)` on the y axis. RHF points show the saved `E_RHF_Ha`; the harmonic curve shows `E_re + k*q^2/2`; the n=0/n=1 segments show `E_total_n0` and `E_total_n1`. Its minimum is at the saved `E_re`, and neither `E_re` nor the sampled minimum is subtracted for display. This presentation change preserves the saved RHF data, harmonic quantities, bond ranges and Part 3 sign decisions. The physical zero-reference warning remains applicable.

Every panel includes the saved optimized equilibrium `(r_e_A, E_re_Ha)` in the orange RHF curve, sorted by bond coordinate, with the same circular marker and shared RHF legend entry. The equilibrium point has no separate numerical annotation box. The curve displays 30 scan points plus the saved equilibrium. Their scan grid and stored records remain unchanged; no new RHF calculation or fitted minimum is used.

The absolute-axis update passed the two focused overlay tests, including absolute plotted coordinates and levels for all nine molecules. Production figures have not been regenerated for the latest axis/equilibrium presentation updates; use the plotting command below to replace the previously saved figures.

## Verification and scope preservation

**60 targeted tests passed:** 9 arithmetic/input/report tests, 5 Part 3 plot tests, 14 pipeline tests, 19 retained numerical-helper/legacy-entry tests, and 13 scanner/overlay tests. Tests cover strict equality/near-zero boundaries, all three outcomes, invalid inputs, no hidden energy shift, exact saved-input reuse, changed units, obsolete threshold inputs, delegation, pipeline cache behavior, and metadata migration without solver calls.

SHA-256 checks confirm no changes to `tests/molecules_rhf.py`, `tests/new_rhf_harmonic_bond_ranges.py`, the Part 1 equilibrium CSV, or the Part 2 harmonic JSON. The pipeline's `starting_geometries`, `optimize_equilibrium`, `reuse_equilibrium`, and `harmonic_cache_matches` functions are byte-identical to their prior versions. All 270 saved RHF point dictionaries, atom identities, geometries, solver diagnostics, all scan CSV/XYZ files, selected ranges and plotted numerical samples remain unchanged. Metadata migration stores the same before/after numerical payload hash. The figure overview and a molecule's absolute-level/zero plot were visually inspected.

Part 3, the reuse-only joined report, scan metadata and plots have been regenerated from saved inputs. No production SCF, optimization or Hessian was run for this revision.

## Reproduce the saved-input analysis and overlay

```bash
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py --rebind-part3
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/plots/plot_rhf_bond_scan_30.py
```

The first command regenerates current Part 3 tables/reports/absolute plots; the second updates saved scan metadata only after verifying unchanged physical inputs and ranges; the third overlays the existing 30 RHF points on plot 1. No command above invokes a new RHF solve. To refresh the joined pipeline report from saved Parts 1/2:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/bond_length.py --reuse-only
```
