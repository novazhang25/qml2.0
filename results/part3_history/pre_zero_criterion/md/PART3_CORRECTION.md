# Corrected Part 3: audit, equations and saved-data table

The replacement is `codes/bond_length_part3.py`. It redoes only the n=0/n=1 analysis from existing Parts 1–2 and the validated RHF plateau. No optimization, Hessian, SCF, or new dissociation scan is performed.

**Execution status:** the new program, tests and plot renderer have not been run, as requested. The numerical table below was independently assembled from the existing saved inputs using arithmetic; it is not claimed as output from an executed replacement program. The provided command generates the full-precision CSV/JSON, the runtime consistency report, and PNG/SVG plots.

## What the audit found

- `tests/rhf_bound_level_selection.py:60` calculates `-D_e + (n+1/2)*hbar*omega`; line 79 tests whether that value is negative. Lines 456–458 define `D_e = E_diss - E_eq`. This is algebraically correct when the result is understood as a signed gap to dissociation.
- `codes/bond_length.py:375–377` reuses that selector, and lines 433–440 describe its dissociation-zero convention. It omits explicit absolute `E_total_0` and `E_total_1` and independent checks between the two references.
- `tests/plot.py:82–91` and `:110–134` use a consistent `E-E_eq` axis. Lines 197–198 label dissociation-relative gaps as `E0/E1`, which is ambiguous under the requested notation. Its current main emits only the local overview without a threshold comparison.
- No audited statement directly compares positive `E_n` with absolute `E_diss`, defines a level as `E_diss-E_n`, or subtracts `E_re` twice. Calling the old signed gap an absolute energy would be wrong, but the old classifications are not demonstrated to be wrong.
- The old scan assigns its q=0 RHF reevaluation to `E_eq` at `tests/rhf_bound_level_selection.py:394`. The replacement reads the optimized `energy_hartree` directly from Part 1's metadata and cross-checks both saved reevaluations within 1e-9 Ha. Their present differences are around 1e-14–1e-13 Ha.

## Correct equations

`E_re` is the absolute optimized RHF equilibrium electronic energy. `E_diss` is the absolute validated large-separation RHF plateau. Only `D_e` denotes the relative well depth.

```text
k_SI = k[Ha/Bohr^2] * Hartree_J / Bohr_m^2
mu_kg = mu_eff[amu] * amu_kg
omega = sqrt(k_SI / mu_kg)                          [rad/s]

E_0 = (1/2) * hbar_J_s * omega / Hartree_J           [Ha]
E_1 = (3/2) * hbar_J_s * omega / Hartree_J           [Ha]
E_total_0 = E_re + E_0                              [Ha]
E_total_1 = E_re + E_1                              [Ha]
D_e = E_diss - E_re                                 [Ha]

n=0 below dissociation iff E_total_0 < E_diss iff E_0 < D_e
n=1 below dissociation iff E_total_1 < E_diss iff E_1 < D_e

(E_re + E_n) - E_diss = E_n - D_e
abs((E_re + E_n) - E_diss) = abs(E_n - D_e)
```

The signed gaps have the **same sign**. A binding margin defined in the reverse direction, `E_diss-E_total(n)`, has the opposite sign. Equality at the threshold is **NO**; the warning window never changes the strict classification.

The saved mass is checked against `sum_A m_A |dR_A/dq|^2` on the existing collective coordinate. It equals the ordinary reduced mass for diatomics. The polyatomic paths retain their saved generalized masses, including H2O2's O–O coordinate with the attached OH groups.

## Complete corrected table from saved inputs

Sources: `results/rhf_geometries/rhf_equilibrium_summary.csv` (optimized E_re), `json/new_rhf_harmonic_bond_ranges.json` (k, mass, physical constants), and `json/rhf_bound_level_selection.json` (absolute validated E_diss). Display rounding does not determine classification.

| Molecule | E_re [Ha] | E_diss [Ha] | D_e [Ha] | k [Ha/Bohr^2] | mu_eff [amu] | omega [rad/s] | E_0 [Ha] | E_1 [Ha] | E_re + E_0 [Ha] | E_re + E_1 [Ha] | n=0 below? | n=1 below? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | -7.863382129 | -7.519403313 | 0.343978816 | 0.116210398 | 0.880161047 | 3.51840857e+14 | 0.004255312 | 0.012765935 | -7.859126817 | -7.850616194 | YES | YES |
| BeH2 | -15.561352808 | -14.897741284 | 0.663611524 | 0.473247781 | 2.016000000 | 4.69141527e+14 | 0.005673995 | 0.017021986 | -15.555678812 | -15.544330821 | YES | YES |
| H2O | -74.965901192 | -74.207677735 | 0.758223457 | 1.246893947 | 1.922837683 | 7.79737272e+14 | 0.009430471 | 0.028291414 | -74.956470721 | -74.937609778 | YES | YES |
| NH3 | -55.455419779 | -54.183259474 | 1.272160305 | 1.629247753 | 2.932611891 | 7.21723951e+14 | 0.008728834 | 0.026186501 | -55.446690945 | -55.429233277 | YES | YES |
| N2 | -107.500654263 | -106.727669991 | 0.772984272 | 1.888640454 | 7.003500000 | 5.02830862e+14 | 0.006081448 | 0.018244345 | -107.494572814 | -107.482409917 | YES | YES |
| CO | -111.225449514 | -110.751403537 | 0.474045977 | 1.573828862 | 6.860549411 | 4.63771603e+14 | 0.005609049 | 0.016827148 | -111.219840465 | -111.208622366 | YES | YES |
| HF | -98.572847347 | -98.011266032 | 0.561581315 | 0.725274202 | 0.957213060 | 8.42852648e+14 | 0.010193815 | 0.030581446 | -98.562653532 | -98.542265901 | YES | YES |
| H2S | -394.311630059 | -393.578532578 | 0.733097482 | 0.794499972 | 1.958991928 | 6.16645360e+14 | 0.007457969 | 0.022373907 | -394.304172090 | -394.289256152 | YES | YES |
| H2O2 | -148.764996621 | -148.287014342 | 0.477982280 | 0.709124310 | 8.462311149 | 2.80298995e+14 | 0.003390054 | 0.010170163 | -148.761606567 | -148.754826459 | YES | YES |

Both levels are below dissociation for **LiH, BeH2, H2O, NH3, N2, CO, HF, H2S and H2O2** in the saved constrained RHF harmonic model. The highest selected level among the two tested levels remains n=1 for every molecule.

## Numerical gap checks from saved inputs

The signed gaps and their absolute magnitudes are displayed separately. All 18 independent Boolean comparisons agree exactly. The largest arithmetic gap residual in this saved-data preview is approximately 2.0e-14 Ha. The program independently recalculates these checks when you run it and rejects inconsistent classifications.

| Molecule | n | (E_re+E_n)-E_diss [Ha] | E_n-D_e [Ha] | abs absolute gap [Ha] | abs relative gap [Ha] | Difference [Ha] | Classifications agree? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | 0 | -0.339723503956 | -0.339723503956 | 0.339723503956 | 0.339723503956 | 0.000e+0 | YES |
| LiH | 1 | -0.331212880609 | -0.331212880609 | 0.331212880609 | 0.331212880609 | 0.000e+0 | YES |
| BeH2 | 0 | -0.657937528758 | -0.657937528758 | 0.657937528758 | 0.657937528758 | 7.772e-16 | YES |
| BeH2 | 1 | -0.646589537875 | -0.646589537875 | 0.646589537875 | 0.646589537875 | 4.441e-16 | YES |
| H2O | 0 | -0.748792985932 | -0.748792985932 | 0.748792985932 | 0.748792985932 | 9.992e-16 | YES |
| H2O | 1 | -0.729932043250 | -0.729932043250 | 0.729932043250 | 0.729932043250 | 2.887e-15 | YES |
| NH3 | 0 | -1.263431471499 | -1.263431471499 | 1.263431471499 | 1.263431471499 | 2.665e-15 | YES |
| NH3 | 1 | -1.245973803946 | -1.245973803946 | 1.245973803946 | 1.245973803946 | 8.882e-16 | YES |
| N2 | 0 | -0.766902823091 | -0.766902823091 | 0.766902823091 | 0.766902823091 | 7.772e-16 | YES |
| N2 | 1 | -0.754739926159 | -0.754739926159 | 0.754739926159 | 0.754739926159 | 2.331e-15 | YES |
| CO | 0 | -0.468436927491 | -0.468436927491 | 0.468436927491 | 0.468436927491 | 3.331e-15 | YES |
| CO | 1 | -0.457218828873 | -0.457218828873 | 0.457218828873 | 0.457218828873 | 4.219e-15 | YES |
| HF | 0 | -0.551387499705 | -0.551387499705 | 0.551387499705 | 0.551387499705 | 2.109e-15 | YES |
| HF | 1 | -0.530999869092 | -0.530999869092 | 0.530999869092 | 0.530999869092 | 6.439e-15 | YES |
| H2S | 0 | -0.725639512754 | -0.725639512754 | 0.725639512754 | 0.725639512754 | 6.661e-15 | YES |
| H2S | 1 | -0.710723574766 | -0.710723574766 | 0.710723574766 | 0.710723574766 | 1.998e-14 | YES |
| H2O2 | 0 | -0.474592225433 | -0.474592225433 | 0.474592225433 | 0.474592225433 | 6.994e-15 | YES |
| H2O2 | 1 | -0.467812116972 | -0.467812116972 | 0.467812116972 | 0.467812116972 | 7.383e-15 | YES |

The runtime residual tolerance is 16 floating-point ULPs at the energy scale, solely for checking arithmetic consistency. If rounding changes a Boolean comparison, the program stops before reporting a classification. The saved plateau convergence tolerance is separate and still used to flag near-threshold cases.

## Files added or updated

- `codes/bond_length_part3.py`: replacement analysis, input checks, full table, JSON and report generation.
- `codes/bond_length_part3_plots.py`: per-molecule PNG/SVG plots of absolute local harmonic levels, absolute RHF scan and threshold, and explicitly relative E_n versus D_e.
- `tests/test_bond_length_part3.py`: regression tests for references, unit conversion, masses, threshold equality, numerical agreement and saved data.
- `md/PART3_CORRECTION.md`: this audit, equations, complete saved-data table and run instructions.
- `md/BOND_LENGTH_PIPELINE.md`: points Part 3 users to the replacement.
- `md/RHF_BOUND_LEVEL_SELECTION_REPORT.md`: marks the previous relative-energy report as historical.

Parts 1 and 2, their existing source modules and saved outputs, the legacy Part 3 source, and the original plateau evidence are preserved. Keeping the old source files intact also prevents source-hash changes from triggering fresh calculations. New outputs go to a separate directory.

## Run

The system Python has an existing Homebrew matplotlib installation; no PySCF import or additional installation is needed by the replacement.

```bash
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py
```

Default outputs in `/Users/novaz/Desktop/qml2.0/results/bond_length_part3/`:

- `vibrational_levels.csv`: complete table, strict YES/NO flags, both signed gaps, both absolute gaps, residuals and agreement flags.
- `vibrational_levels.json`: full-precision results, input hashes, constants and saved plateau evidence.
- `PART3_REPORT.md`: audit, formulas, complete tables, per-molecule outcomes and links to plots.
- `plots/<molecule>_vibrational_levels.png` and `.svg`: three panels per molecule with explicit energy references.

For the numerical report without plotting dependencies, append `--no-plots`. Plots and tests remain unexecuted and their rendering has not been visually verified.

The threshold retains the interpretation of the saved constrained RHF path; these harmonic comparisons do not establish experimental dissociation energies or exact anharmonic bound states.
