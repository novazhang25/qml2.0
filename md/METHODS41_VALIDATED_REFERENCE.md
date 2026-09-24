# Methods 4.1 descriptor pipeline

`codes/methods41.py` implements RHF/RMP2, a consistent wavefunction-based
density, the RHF-reference cumulant, full-AO symmetric Löwdin transformation,
valence-AO selection, descriptors, validation, and NPZ output.
`codes/methods41_data.py` independently validates the existing input records.
No input file or target FCI energy is modified, and no checksum machinery is added.

The formal descriptor source in schema `methods41-v3-fullspace` is exactly:
**normalized truncated MP2 wavefunction, expectation values through second order**.
Standard PySCF MP2 densities are retained as a separate audit; their known
contraction failures cannot veto otherwise valid formal descriptors.
Production calculations use the full orbital space: all-AO RHF, RMP2 with
`frozen=0`, full-MO/full-AO RDMs, and full-AO Löwdin transformation. Core AOs
are removed only when selecting the final descriptors.

**Current status:** the full-space H2O/001 smoke test passed, followed by all
270 geometries: **270 successful, 0 failed, 0 not run**. All 42 focused tests
passed. The maximum formal AO 2-RDM contraction residual across the dataset
was `9.237e-14`; the maximum RHF/input energy difference was `5.684e-13 Ha`.

For H2O/001, the full-space MP2 amplitude shape is `(5, 5, 2, 2)`, the AO
matrices are `(7, 7)`, and rank-4 AO tensors are `(7, 7, 7, 7)`. The formal
AO contraction residual was `7.105e-15`; the independent standard PySCF
audit retained its expected defect of approximately `0.018660079`, without
vetoing formal output. Its descriptors have 6 one-body entries, 15 pairs,
20 triples, and `T_full` shape `(6, 6, 6)`.

See the [validation report](../results/methods41_fullspace/VALIDATION_REPORT.md)
for complete norm, energy, contraction, and descriptor results; the
[validation summary](../results/methods41_fullspace/validation_summary.json)
provides the same run's machine-readable metrics. The
[dataset manifest](../results/methods41_fullspace/manifest.json) and per-geometry
validation sidecars record the current full-space run.

The earlier standard-density H2O diagnostic remains untouched at
`results/methods41/H2O/001.json`. It measured a 2-RDM contraction error of
`0.0186531915517` and pair-number trace `90.0399800685154`, instead of `90`.
Those results motivated the explicitly selected density definition below.
The subsequent `results/methods41_consistent/` run completed 270 geometries
using the now-superseded frozen-core MP2 policy. It is preserved as historical
output and is not the current deliverable. New outputs use
`results/methods41_fullspace/`; neither previous run is overwritten.

## Actual input conventions

Inputs are `results/initialdata/initialdata.json`, `scan.csv`, the source snapshot
`source_initialdata.py`, `<molecule>/result.json`, and
`<molecule>/scan/001/` through `030/`. Each scan directory contains
`geometry.xyz`, `point.json`, and `electronic_state.npz`. The separately stored
equilibrium point has index 0 and is excluded.

The JSON `cartesian_A` coordinates retain full precision. Discovery verifies
them against the XYZ, NPZ, and duplicated JSON records; XYZ rounding is allowed
only within `1e-14 Å`. Geometry IDs are explicitly sorted numerically. The scan
coordinate `q_A` is the bond-coordinate displacement from RHF equilibrium;
`bond_length_A` is the bond coordinate `r_e + q_A`. Diatomics stretch their bond; BeH2, H2O,
H2S, and NH3 stretch all central-atom–H bonds together; H2O2 stretches O–O with
its OH groups and torsion fixed. Each scan has 30 increasing, evenly spaced
points, including the documented endpoints.

All inputs are neutral, closed-shell singlets (`charge=0`, PySCF `spin=0`), with
all-electron spherical STO-3G, Angstrom coordinates, and Hartree energies.
Stored total energies include nuclear repulsion. FCI uses canonical RHF
orbitals and freezes the lowest occupied spatial MOs:

| Molecule | Electrons | Full AOs | Input FCI frozen MOs | Production MP2/RDM frozen MOs | Valence AOs / P_mu / F_mu | Pairs | Triples |
|---|---:|---:|---:|---:|---:|---:|---:|
| LiH | 4 | 6 | 1 | 0 | 5 | 10 | 10 |
| HF | 10 | 6 | 1 | 0 | 5 | 10 | 10 |
| BeH2 | 6 | 7 | 1 | 0 | 6 | 15 | 20 |
| H2O | 10 | 7 | 1 | 0 | 6 | 15 | 20 |
| H2S | 18 | 11 | 5 | 0 | 6 | 15 | 20 |
| NH3 | 10 | 8 | 1 | 0 | 7 | 21 | 35 |
| N2 | 14 | 10 | 2 | 0 | 8 | 28 | 56 |
| CO | 14 | 10 | 2 | 0 | 8 | 28 | 56 |
| H2O2 | 18 | 12 | 2 | 0 | 10 | 45 | 120 |

`input_fci_frozen_core` records the original target convention. It does not
control the new calculations: production MP2 and both density constructions
freeze zero orbitals and correlate the full occupied space. The preserved
`E_FCI` and `E_corr_input = E_FCI - E_RHF_input` therefore describe frozen-core
targets, while the new MP2 energies and RDMs describe full-space correlation.
This correlation-space mismatch is explicitly acknowledged and follows the
user's current preference; FCI targets are not recomputed or relabeled as
full-space values.

Removing core AOs is restricted to final descriptor selection, after all full
space densities and the full-AO Löwdin transformation are complete. Selection
uses atom, AO principal-shell labels, and basis-shell metadata. In particular,
H2S descriptor core AO indices are `[0, 1, 3, 4, 5]`, not the first five
positions. These descriptor exclusions do not freeze any MP2 electrons.

The audit found 30 matched rows for every molecule, no duplicate/mismatched
records, exact agreement of JSON/CSV/NPZ energies, and zero residual in
`E_corr = E_FCI - E_RHF`. Maximum XYZ rounding was `5.55e-16 Å`.
The saved audit is `results/methods41_input_audit.json`; this is an input audit,
not evidence that new RHF/MP2 calculations passed.

## Formal density and independent PySCF audit

The installed PySCF version is 2.14.0. Its implementation and
[official source documentation](https://pyscf.org/_modules/pyscf/mp/mp2.html#make_rdm2)
specify the spatial, spin-summed index convention

```text
Gamma[p,q,r,s] = sum_(sigma,tau) <p_sigma† r_tau† s_tau q_sigma>
D[p,q] = sum_sigma <q_sigma† p_sigma>
```

The pipeline retains that ordering and the full MO/AO representation, with
no frozen orbitals. With full-space canonical first-order MP2 doubles defining
`chi`, the wavefunction family is

```text
Psi(eta) = (Phi_HF + eta*chi) / sqrt(1 + eta**2*w)
w = <chi|chi>,    <Phi_HF|chi> = 0
```

For either the one- or two-body density operator, expanding its normalized
expectation through order `eta**2` and then setting `eta=1` gives

```text
R_consistent = R0 + R1 + R_chichi - w*R0
R0 = <Phi_HF|R|Phi_HF>
R1 = <Phi_HF|R|chi> + <chi|R|Phi_HF>
R_chichi = <chi|R|chi>
```

The subtraction `-w*R0` is the second-order normalization term. This is not
the unexpanded rational expectation at `eta=1`, nor a correlated-density
product used to repair a tensor. PySCF `make_rdm12` contracts explicitly
constructed determinant coefficients to obtain the density contributions;
it does not run an FCI solver or recalculate target FCI energies. All occupied
MOs participate in the MP2 excitation space; core electrons are not held
doubly occupied in the correlated wavefunction.

The formal one- and two-body densities obey the nonorthogonal AO identity

```text
sum_(r,s) Gamma[p,q,r,s] * S[r,s] = (N - 1) * D[q,p]
```

Here `D` is the consistent correlated 1-RDM. The official Methods 4.1
disconnected reference uses the **RHF** spin-summed density `P_AO`:

```text
Gamma0[p,q,r,s] = P[p,q]*P[r,s] - 0.5*P[p,s]*P[r,q]
Lambda_HF = Gamma_consistent - Gamma0(P_HF)
```

The required HF-reference cumulant contraction is `(N-1)*(D-P_HF).T`,
generally not zero. Only this cumulant supplies the official `Lambda_AO`,
`Lambda_L`, `T_full`, and `T_munulambda` descriptors.

A conventional correlated-reference cumulant is saved separately for diagnosis:

```text
Lambda_conventional = Gamma_consistent - Gamma0(D_consistent)
sum_(r,s) Lambda_conventional[p,q,r,s]*S[r,s] = (0.5*D @ S @ D - D)[p,q]
```

The last expression assumes real symmetric `D` with `Tr(D @ S)=N`; it is
generally nonzero because a correlated density is not RHF-idempotent. The
pipeline checks both cumulant identities against their own references, and
checks the zero-amplitude RHF limit separately.

PySCF's truncated, unrelaxed MP2 density builder does not generally satisfy
the exact contraction above. For real canonical MOs, write `D = 2*O + delta`,
where `O` projects onto all occupied MOs. Direct
contraction of the implemented construction gives the defect

```text
sum_r Gamma[p,q,r,r] - (N-1)*D[q,p] = delta - O @ delta - delta @ O
```

The standard PySCF defect is generally nonzero at second order in MP2
amplitudes. Both its measured defect and the residual against this analytic
identity belong to the **standard-density audit**. They do not weaken the
formal consistent-density contraction tolerances and cannot mark a valid
formal record as failed. Conversely, a failure in a formal density, energy,
geometry, or descriptor validation remains a failure and stops the full-run
H2O gate.

The wavefunction norm audit records `w` and the normalization correction;
the actual unnormalized vector has norm squared `1+w`. This is distinct from
checking `Tr(D*S)=N` and the 2-RDM pair-number trace `N*(N-1)`.
Energy audits also retain distinct quantities: the canonical RMP2 correlation
energy, contractions of the standard PySCF density, and contractions of the
consistent second-order expectation density. Contracting the latter with the
full Hamiltonian can retain terms different from the canonical second-order
MP2 energy. Such differences are reported with their definition, rather than
silently replacing the canonical MP2 energy or an existing FCI target.

## Commands

Use the repository environment. The default output directory is
`results/methods41_fullspace/`.

Read-only data and AO audit; prints a report and writes no output files:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/methods41.py --dry-run
```

H2O geometry 001 end-to-end smoke test, including an independent explicit
rank-4 transformation check:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/methods41.py --smoke-test --resume
```

Equivalent single-geometry selection:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/methods41.py --molecules H2O --geometry-id 001 --resume
```

Full 9 × 30 generation is conditional on passing the consistent-density
smoke test. This chain stops if that test fails:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/methods41.py --smoke-test --resume && \
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/methods41.py --resume
```

`--resume` revalidates and skips successful records. Failed or invalid records
are not treated as successes. After resolving a failure, retry it explicitly:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python /Users/novaz/Desktop/qml2.0/codes/methods41.py --smoke-test --resume --rerun-failed
```

For all failed records, omit `--smoke-test`; the full-run smoke gate still
applies. `--input-dir`, `--output-dir`, and `--threads` are supported. Input
and output directories must be separate and non-nested. Use a new output
directory if changing scientific conventions or validation tolerances.
OpenMP support is detected from the library's thread-setting result. Its known
no-OpenMP warning is suppressed and the limitation is recorded in the manifest.

Run focused tests without molecular electronic calculations:

```bash
/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python -m unittest discover -s /Users/novaz/Desktop/qml2.0/tests -p 'test_methods41*.py'
```

## Outputs and validation reports

Default outputs are `results/methods41_fullspace/<molecule>/<ID>.npz`, a matching
`.json` validation sidecar, `<molecule>/summary.csv`, and
`results/methods41_fullspace/manifest.json`. The schema is `methods41-v3-fullspace`.
NPZ records load with `numpy.load(path, allow_pickle=False)` and contain:

- Identity, source paths, coordinates, scan coordinate, charge/spin/basis,
  electron count, `input_fci_frozen_core`, the separate production MP2/RDM
  setting `frozen=0`, the acknowledged correlation-space mismatch, and
  input/check RHF, unchanged input FCI, and correlation energies.
- Full `S_AO`, `P_AO`, `F_AO`, the consistent correlated 1-RDM and 2-RDM,
  `Gamma0_AO`, `Lambda_AO`, `S_half`, `S_minus_half`, `P_L`, `F_L`, and
  `Lambda_L`. Official descriptor intermediates use the consistent density
  and the HF-reference cumulant.
- Separate standard PySCF density tensors and validation audit, the conventional
  correlated-reference cumulant, wavefunction norm contributions, and energy audits.
- All AO labels, core/valence labels and full-AO indices, basis-shell metadata,
  canonical orbitals/occupations/energies, MP2 amplitudes and correlation energy.
- `P_mu`, `F_mu`, `pair_indices`, `P_munu`, `T_full`, `triple_indices`, and
  `T_munulambda`; pairs and triples use zero-based **valence-local** indices.
- Status, RHF/MP2 convergence information, library/RDM convention, and
  `validation_json` with residuals, tolerances, tensor shapes, and failed checks.

The formal tensors are named `P_MP2_AO_consistent` and
`Gamma_MP2_AO_consistent`, with corresponding `_MO_consistent` tensors.
Standard-library tensors use the `_pyscf` suffix. `Lambda_HFref_AO` feeds the
descriptor transformation; `Lambda_conventional_AO` is diagnostic only.
`rdm_metadata_json` specifies the source, formulas, expansion and normalization;
`energy_audit_json` holds energy comparisons. Coefficients at each retained
order are saved as `P_MO_order0/1/2` and `Gamma_MO_order0/1/2` and their AO forms.
Explicit failed retries retain previous NPZ/JSON evidence as `.attemptN` files.

All rank-4 intermediates retain full-AO dimensions. `T_full` has shape
`(N_valence, N_valence, N_valence)` and selects
`Lambda_L[valence[mu], valence[lambda], valence[nu], valence[lambda]]` without
summing over lambda. The saved rank-4 tensor remains in PySCF ordering,
`Lambda_raw[p,q,r,s] = Lambda_methods[p,r,q,s]`; this extraction implements
Methods `T[mu,nu,lambda] = Lambda_methods[mu,nu,lambda,lambda]`.
Historical datasets retain the earlier incorrect raw last-two-axis diagonal;
the corrected 270-record copy is `results/descriptor-v2-methods-t`.
Pairs/triples are lexicographically ordered unique
combinations. No absolute values, permutation averaging, or extra
symmetrization are applied.

Validation includes finite/real values, fixed AO and descriptor ordering,
dimensions, energy consistency, overlap eigenvalues, Löwdin metric, electron
count, RHF idempotency, exact consistent 2-RDM/particle/cumulant contractions, the
zero-amplitude limit, and explicit index/transformation checks. Default RHF
energy agreement is `1e-10 Ha`; matrix/RDM residual limits are `1e-8`;
overlap eigenvalues at or below `1e-8` stop the calculation. The manifest
separately records successful, failed, and requested-but-not-run counts;
it also records dataset completion (`PASS`, `PARTIAL`, or `FAIL`),
`dataset_not_run_geometry_count`, and input/AO exclusions. Summary rows show
`PASS`, `FAIL`, `INPUT_FAIL`, or `NOT_RUN`.
