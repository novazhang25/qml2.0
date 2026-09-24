# Initial data calculation

`codes/initialdata.py` is a standalone calculation file. It requires NumPy, SciPy and PySCF, and imports no other project scripts or previous result files. Plotting is separate in `codes/plot_initialdata.py` and requires Matplotlib.

The production calculation covers LiH, BeH2, H2O, NH3, N2, CO, HF, H2S and H2O2 using neutral, closed-shell RHF/STO-3G:

1. Optimize each full Cartesian geometry with RHF analytic gradients. Save all optimization starts and iterations.
2. Calculate the RHF nuclear Hessian and project onto the mass-aligned collective stretch: `k = t^T H t`, `mu_eff = sum_A m_A |t_A|^2`. Validate the Hessian, stationary geometry, finite-difference curvature and normal modes.
3. Calculate `omega = sqrt(k_SI/mu_kg)`, `E_n1 = 1.5*hbar*omega`, and the n=1 classical turning points `r_e +/- sqrt(3*hbar/(mu_eff*omega))`. The scan always uses n=1. The sign of `E_RHF(re)+E_n1` is recorded; it is not used to establish physical vibrational binding or select a different range.
4. Calculate both RHF and frozen-core FCI at 30 evenly spaced geometries, including both n=1 endpoints. An additional equilibrium RHF/FCI point is stored separately because an even grid of 30 points does not contain its midpoint.
5. Save `E_corr = E_FCI - E_RHF` in Ha and mHa for every matching geometry and the equilibrium point.

For the star-shaped molecules, all X-H distances stretch together while the optimized angles stay fixed. H2O2 stretches O-O while retaining its optimized OH groups and torsion. Diatomics stretch their single bond.

## FCI convention

FCI retains the project's existing frozen-core convention. It is exact diagonalization over all noncore STO-3G orbitals using canonical RHF orbitals; the orbitals are not optimized by CASSCF. All reported RHF/FCI totals include nuclear repulsion. The core energy already includes nuclear repulsion and must not have it added again.

| Molecule | Frozen spatial orbitals | Active spatial orbitals | Active electrons |
|---|---:|---:|---:|
| LiH | 1 | 5 | 2 |
| BeH2 | 1 | 6 | 4 |
| H2O | 1 | 6 | 8 |
| NH3 | 1 | 7 | 8 |
| N2 | 2 | 8 | 10 |
| CO | 2 | 8 | 10 |
| HF | 1 | 5 | 8 |
| H2S | 5 | 6 | 8 |
| H2O2 | 2 | 10 | 14 |

## Run

Production, from any working directory:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/initialdata.py --resume
```

`--resume` creates the output folder if absent and otherwise validates and reuses completed stages. It requires the same molecule selection, numerical settings, source code and dependency versions. Missing or changed wavefunction/geometry files are reported as failures. Choose a new `--output-dir` after changing calculation settings or code. Without `--resume`, an existing run is never overwritten. Interrupted or failed runs retain their completed calculations.

Plot saved production results with the system Python installation, which has Matplotlib:

```bash
/opt/homebrew/bin/python3 /Users/novaz/Desktop/qml2.0/codes/plot_initialdata.py
```

The plotter reads `results/initialdata/initialdata.json` and writes individual and overview PNG/SVG plots to `results/plots/initialdata`. It follows the compact reference overlay: a 3×3 grid ordered LiH/HF/BeH2, NH3/H2O/H2S, N2/CO/H2O2, orange RHF circles, magenta CASCI squares, green n=1 range shading, inside molecule labels and one legend in LiH. There are no additional correlation panels or harmonic/level curves. Both methods use the RHF equilibrium total as their energy zero; `--absolute` displays unshifted totals. Exact equilibrium points join the curves. Correlation remains in the saved results and exported CSV. Plotting performs no molecular calculations.

## Saved files

The default calculation output is `results/initialdata/`. Every selected molecule uses 30 scan points.

- `initialdata.json`: complete run, settings, physical constants, dependency versions, formulas, full optimization history, Hessian, harmonic and path diagnostics, RHF/FCI point records, status and failures. Updated after each stage and scan point.
- `summary.csv`: equilibrium, force constant, effective mass, frequency, n=1 energy/range, equilibrium FCI/correlation energy and completion status.
- `scan.csv`: all scan-point RHF, FCI and correlation energies, core/active/nuclear components and status.
- `equilibrium_energies.csv`: the additional equilibrium RHF/FCI/correlation point for each molecule.
- `source_initialdata.py`: exact calculation source snapshot used for this run.
- `<molecule>/result.json`: complete record for that molecule.
- `<molecule>/equilibrium/`: optimized XYZ and optimization metadata CSV.
- `<molecule>/equilibrium_energies/` and `<molecule>/scan/001/` through `030/`: exact geometry XYZ, `point.json`, and compressed `electronic_state.npz`. The NPZ stores canonical MO coefficients, orbital energies/occupations, RHF density, overlap, one-electron and Fock matrices, active-space Hamiltonian integrals, CI vector, core energy, RHF/FCI/correlation totals, symbols, Cartesian coordinates and PySCF molecule specification. Read with `numpy.load(..., allow_pickle=False)`.

Energies are Hartree unless a field explicitly says mHa, lengths are Angstrom, masses are amu, and the Hessian and force constant carry their stated units. Completed XYZ/NPZ files have SHA-256 checksums in the point records.
