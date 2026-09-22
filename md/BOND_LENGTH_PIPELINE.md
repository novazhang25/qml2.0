# RHF bond-length pipeline

Entry point: `codes/bond_length.py`. The equivalent `codes/bond-length.py` launcher avoids underscores when copying a terminal command.

The entry point imports the existing validated harmonic and bound-level engines from the same `codes` directory. Keep `new_rhf_harmonic_bond_ranges.py` and `rhf_bound_level_selection.py` alongside it. It requires PySCF, NumPy and SciPy, already installed in `.venv-rhf`.

## Three stages

1. **RHF equilibrium:** reuse a matching converged geometry or optimize all Cartesian coordinates with analytic RHF gradients. The requested basis is used throughout. An explicit new optimization uses rough starting guesses, with multiple torsional starts for H2O2.
2. **Force constant:** define the validated collective path through that equilibrium; calculate `k = t^T H t` without normalizing the physical tangent, calculate `mu_eff = sum_A m_A |t_A|^2`, and obtain the harmonic n=0 and n=1 ranges. Existing stationarity, internal-coordinate, Hessian and finite-difference checks remain required.
3. **Level selection:** establish a confirmed large-distance plateau on the same all-electron RHF path; calculate `D_e = E_diss - E_eq`, `E0_rel = -D_e + 0.5*hbar*omega` and `E1_rel = -D_e + 1.5*hbar*omega`. Choose n=1 if its relative energy is negative, otherwise n=0 if its relative energy is negative, otherwise NONE.

The H2O2 coordinate is O-O separation. For BeH2, H2O, NH3 and H2S it is the common participating X-H bond-length displacement.

Energy outputs are Hartree. Force constants are Eh/Bohr^2 and Eh/Angstrom^2, effective masses are amu, and coordinate ranges are Angstrom. A selected level is bound only with respect to the constrained RHF-path threshold; it is not a physical dissociation-energy determination.

## Run

Default: reuse matching validated results and compute missing stages.

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/bond-length.py
```

Explicitly optimize new equilibria and recompute all subsequent stages:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/bond-length.py --reoptimize
```

Only collect validated existing results, with no SCF or optimization:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/bond-length.py --reuse-only
```

Use `--recompute` to retain matching equilibria but recompute k and dissociation. Use `--basis`, `--molecules`, `--output-dir`, and `--help` for configuration. A different basis invalidates incompatible cached results. A failed stage blocks the later stages for that molecule and produces an explicit status rather than a guessed range.

## Output

The default output directory is `/Users/novaz/Desktop/qml2.0/results/bond_length/`:

- `bond_length.csv`: one joined row per molecule, including r_e, k, effective mass, harmonic energy quantum, D_e, E0/E1, selected n and range.
- `bond_length.json`: full evidence from all three stages, including geometry, gradients, Hessian, normal-mode diagnostics, finite differences, and dissociation scans.
- `BOND_LENGTH_REPORT.md`: formulas, numerical summary, stage statuses and RHF warnings.
- `equilibria/`: newly optimized geometries and metadata, when optimization is requested or required.
- `scans/`: separate RHF energy scan CSV files.

Existing workflow files remain unchanged. Results are checkpointed after each molecule. Source hashes, constants, geometry, basis, equilibrium energy, harmonic ranges and plateau evidence determine whether a prior result can be reused.

## Validation performed during implementation

Fourteen tests passed, including the full LiH optimization-to-selection pipeline in a temporary directory, reuse of all nine existing results with numerical calculations disabled, changed-input cache invalidation, correct Hartree conversion, and propagation of optimization/harmonic failures. A fresh production run for all nine molecules was not launched.
