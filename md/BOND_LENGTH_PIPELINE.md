# RHF bond-length pipeline

The current entry point is [`codes/bond_length.py`](../codes/bond_length.py). Part 3 now evaluates the requested strict zero-energy rule using the unchanged Part 1 equilibrium energy and Part 2 harmonic angular frequency. It performs no RHF scan and reads no dissociation threshold or old selection cache. Parts 1 and 2 retain their existing optimization, curvature, mass and frequency calculations.

The pipeline imports the validated numerical utilities from `tests/new_rhf_harmonic_bond_ranges.py` and `tests/rhf_bound_level_selection.py`, plus the shared Part 3 calculation from `codes/bond_length_part3.py`. PySCF, NumPy and SciPy are installed in `.venv-rhf`.

## Three stages

1. **RHF equilibrium:** reuse a matching converged geometry or optimize all Cartesian coordinates with analytic RHF gradients. The requested basis is used throughout. Explicit new optimization uses rough starting geometries, with multiple torsional starts for H2O2.
2. **Force constant and frequency:** define the existing collective path through that equilibrium, calculate `k = t^T H t` without normalizing the physical tangent, and calculate `mu_eff = sum_A m_A |t_A|^2`. The harmonic calculation supplies `omega`, the n=0 and n=1 extents, and their bond ranges. Its stationarity, internal-coordinate, Hessian and finite-difference checks are unchanged.
3. **Level selection:** read `E_eq_Eh` from Part 1 as `E_re_Ha`, without shifting it. Read `omega_rad_per_s` from Part 2, convert its energy quantum to Hartree, and apply the following rule:

```python
hbar_omega_Ha = hbar_J_s * omega_rad_per_s / Hartree_J
E_n0_Ha = 0.5 * hbar_omega_Ha
E_n1_Ha = 1.5 * hbar_omega_Ha
E_total_n0_Ha = E_re_Ha + E_n0_Ha
E_total_n1_Ha = E_re_Ha + E_n1_Ha
n0_pass = E_total_n0_Ha < 0
n1_pass = E_total_n1_Ha < 0

if n1_pass:
    selected_n = 1
elif n0_pass:
    selected_n = 0
else:
    selected_n = None
```

Equality to zero fails the test. A selected level uses its already calculated Part 2 interval; if neither level passes, the selected range is `None`. Part 3 retains the saved `k`, `mu_eff`, and `omega` values. It independently checks that the saved wavenumber and angular frequency produce the same Hartree energy quantum, and verifies that the Part 1 equilibrium energy agrees with Part 2's saved equilibrium energy within `1e-9 Ha`.

The H2O2 coordinate is O–O separation. For BeH2, H2O, NH3 and H2S it is the common participating X–H bond-length displacement. Coordinate ranges are Angstrom; `k` is reported in Hartree/Bohr² and Hartree/Angstrom², `mu_eff` in amu, and `omega` in rad/s. `E_re_Ha`, both excitation energies and both total energies are in Hartree before comparison.

## Energy-reference audit

Part 1's `optimize_equilibrium()` stores `float(mf.e_tot)` in `E_eq_Eh` and the equilibrium metadata's `energy_hartree`. Reuse reads this saved number unchanged. In PySCF, this is the absolute RHF molecular Born–Oppenheimer energy: the electronic contribution plus nuclear repulsion. It is not an energy already shifted to a vibrational threshold.

The workflow does not define absolute `0 Ha` as the intended physical threshold for allowed vibrational states. Therefore every Part 3 result explicitly carries `energy_reference_status = UNVALIDATED_ZERO_REFERENCE` and `reference_shift_Ha = 0.0`, with a warning in the CSV, JSON, report and console output. The pass flags describe the requested numerical sign test; they do not establish physical vibrational binding. `validation_status = PASS` means the numerical workflow passed its checks, not that the zero reference was physically validated.

## Run

Regenerate all nine results using saved Parts 1 and 2, without SCF, optimization or Hessian calculations:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/bond_length.py --reuse-only
```

The default command reuses matching validated inputs and computes missing Parts 1 or 2:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/bond_length.py
```

`--reoptimize` explicitly requests new equilibria and subsequent harmonic calculations. `--recompute` keeps matching equilibria but recomputes the harmonic calculation. Neither can be combined with `--reuse-only`. Part 3 always recomputes its additions and comparisons from the supplied Parts 1 and 2 outputs.

Other supported options are `--basis`, `--molecules`, `--metadata`, `--harmonic-json`, `--output-dir`, `--gtol`, `--maxiter`, `--threads`, and `--help`. The former selection-cache, scan-distance, plateau-tolerance and threshold-warning options have been removed. A failed equilibrium or harmonic stage blocks selection for that molecule and reports an explicit status.

For Part 3 alone, use the standalone postprocessor:

```bash
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py
```

Its separate CSV, JSON, report and plots are stored under `results/bond_length_part3/`. See [the Part 3 audit and full numerical table](PART3_CORRECTION.md).

## Output

The pipeline's default output directory is `results/bond_length/`:

- [`bond_length.csv`](../results/bond_length/bond_length.csv): one joined row per molecule, including unchanged equilibrium and harmonic fields; `E_re_Ha`, `omega_rad_per_s`, `hbar_omega_Ha`, `E_n0_Ha`, `E_n1_Ha`, `E_total_n0_Ha`, `E_total_n1_Ha`, `n0_pass`, `n1_pass`, `selected_n`, the selected range, and explicit reference-status fields.
- [`bond_length.json`](../results/bond_length/bond_length.json): Parts 1 and 2 evidence, corrected Part 3 values, unit checks, stage provenance, and schema `bond-length-zero-energy-v2`.
- [`BOND_LENGTH_REPORT.md`](../results/bond_length/BOND_LENGTH_REPORT.md): equations, numerical table, stage statuses and energy-reference warnings.
- `equilibria/`: newly optimized geometries and metadata, only when optimization is requested or required.

Current Part 3 exports contain no scan arrays or dissociation-based selection fields. The writer no longer creates scan CSVs. Results are checkpointed after each molecule. Validated Parts 1 and 2 may be reused after source-hash, constants, basis, geometry, equilibrium-energy and harmonic checks; old Part 3 caches are not inputs.

## Completed validation and regeneration

The all-nine-molecule pipeline was successfully regenerated with `--reuse-only`. Every saved result records `equilibrium: reused`, `harmonic: reused`, and `selection: postprocessed`. No new SCF, equilibrium optimization, Hessian or scan was run for this regeneration.

For LiH, BeH2, H2O, NH3, N2, CO, HF, H2S and H2O2, both total energies are negative and `selected_n = 1`. No molecule has neither level passing. All nine retain `UNVALIDATED_ZERO_REFERENCE`.

Fourteen focused pipeline tests passed, including all-nine reuse-only output generation with numerical engines prohibited, repeated cache reuse, strict zero boundaries, invalid input rejection, saved-frequency conversion agreement, and propagation of equilibrium/harmonic failures. The Part 1 optimization and reuse functions and the Part 2 validation functions were also checked to be unchanged.
