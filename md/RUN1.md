# Run 1: FG / FE / FE′ / MB-1

**Current H2S loader:** the saved Ne-core-excluded convention is restored:
S 3s, 3px, 3py, 3pz and both H 1s AOs, stored indices `[2,6,7,8,9,10]`.
H2S therefore uses a 6-by-6 valence block and 6 FE/MB qubits. Selection still
follows full-AO Löwdin validation. The earlier 10-AO proposal below is history
for the loader; this repair changes no MP2/FCI calculation, FE definition,
training protocol, or saved descriptor data.

**AO-selection history:** the H2S FE/MB dimensions in this document reflect
the legacy highest-principal-shell/chemical-core descriptor rule: H retained
1s, Li–F retained 2s/2p, and S retained 3s/3p. The revised rule excludes only
heavy-atom 1s after the full-AO Löwdin transformation; H2S alone changes from
6 to 10 retained AOs, FE entries, and MB qubits. See
[current AO selection](DESCRIPTOR.md) for verified indices and dimensions.
Historical tables and other protocol details below are preserved by this
AO-selection-only documentation update; no training, MP2, or FCI setting is
changed here.

Run 1 has three code files, with function docstrings explaining their roles:

- `codes/circuit.py`: sample/AO definitions, geometric features, training-only
  descriptor scaling, quantum circuits, and linear energy models.
- `codes/train.py`: initialization, Adam updates, minibatches, checkpoint
  selection, evaluation, and statistics across seeds.
- `codes/run1.py`: entry point, full-AO data loading, FE/FE′ eigenvalue selection,
  population/split audit, smoke gate, seed sweep, and output writing.

`run1.py` reads `results/descriptor`, produced by `codes/descriptor.py`.
No RHF, MP2, or FCI calculation is launched. The target is the saved
`E_FCI - E_RHF_input`; frozen-core FCI labels are retained unchanged.

Runtime checks stop with the record, quantity/code path, and mismatch instead
of changing a scientific convention. Interface and training tests use bounded
synthetic runs; they do not establish numerical parity with historical results.

## Run

The existing `encoding_qml` environment contains Torch and PennyLane. Two
split protocols are supported. The default `manifest` mode requires a JSON
file containing explicit, disjoint `train`, `validation`, and `test` ID lists
for each selected molecule. Every ID `001`–`030` must occur exactly once, and
each group must be nonempty. Replace the placeholder path below with the real
split file; no manuscript split is inferred or generated:

```bash
/Users/novaz/Desktop/encoding_qml/.venv/bin/python -B /Users/novaz/Desktop/qml2.0/codes/run1.py --split-manifest /absolute/path/to/split_manifest.json
```

The command audits all records and model inputs, runs H2O/seed 0 for all three
models for the full 500 epochs, verifies finite predictions and metrics, then
runs the remaining seeds. Those three smoke runs also count as production
runs. Failure stops execution before the remaining seed sweep.

Optional `--validate-only` stops after input/model preparation checks;
`--smoke-only` runs at most two epochs for seed 0 of the smoke molecule.
`--input-dir` and `--output-dir` accept
explicit paths. Output defaults to a new `results/run1/<timestamp>/` directory;
existing output directories are rejected.

To run only CO with the previous **15 training / 15 test** protocol, all 32
seeds, and FG, FE, and MB-1:

```bash
/Users/novaz/Desktop/encoding_qml/.venv/bin/python -B /Users/novaz/Desktop/qml2.0/codes/run1.py --molecules CO --split-protocol 15-15
```

Only selected molecules are loaded and trained. The smoke stage uses H2O when
selected, otherwise the first selected molecule (CO here). Its seed-0 runs
count toward the 32 seeds. Summaries include only the selected molecules.

The `15-15` mode assigns odd geometry IDs to training and even IDs to test,
with no validation set. It restores the checkpoint with the lowest training
MAE. The default manifest mode restores the checkpoint with the lowest
validation MAE. Both keep the earliest epoch on ties and evaluate the test set
once after restoration; test data never enters training or checkpoint
selection. `--split-manifest` cannot be combined with `--split-protocol 15-15`.
This option restores the old split and checkpoint criterion while using the
current model implementation; it does not promise historical numerical parity.

## FE versus FE′ comparison

The existing FE density descriptor is unchanged. FE′ (`FE_prime` in arguments
and result directories) uses the Fock spectrum from the same valence Löwdin
representation:

```python
P_L = S_half @ P_RHF_AO @ S_half
F_L = S_minus_half @ F_RHF_AO @ S_minus_half
x_FE = np.linalg.eigvalsh(P_L[np.ix_(valence_idx, valence_idx)])
x_FE_prime = np.linalg.eigvalsh(F_L[np.ix_(valence_idx, valence_idx)])
```

Both transformations use the full AO matrices before the existing valence
selection. `load_record` retains the complete selected Fock block as `fij`;
FE′ uses its off-diagonal entries as well as its diagonal. Both spectra have
the same valence AO count, with the ascending order returned by `eigvalsh`
preserved directly. No clipping, reordering, or standardization is applied to
the raw spectra.

`raw_spectra` prints both raw arrays and an indexed side-by-side table for
every loaded geometry before any scaler is fitted. `prepare_runs` passes those
same values into the fingerprint preprocessing. Each method fits its own
per-feature statistics using the identical training IDs, then uses the same
standardization, angle conversion, clipping, circuit, initialization,
optimizer, target, training settings, readout, seeds, and aggregation. Thus the
descriptor matrix is the only scientific difference. The existing 15/15 and
three-way manifest split protocols both support the comparison.

Run the CO comparison with the previously selected 15/15 protocol:

```bash
/Users/novaz/Desktop/encoding_qml/.venv/bin/python -B /Users/novaz/Desktop/qml2.0/codes/run1.py --molecules CO --methods FE FE_prime --split-protocol 15-15
```

This prints all 30 pairs of spectra and trains both methods with seeds 0–31
for 500 epochs. Add `--validate-only` to print spectra and check preparation
without training. To use a real three-way split, replace `--split-protocol
15-15` with `--split-manifest /absolute/path/to/split_manifest.json`.
The default method list stays `FG FE MB-1`; `--methods` explicitly selects
the comparison, or can select `FG FE FE_prime MB-1` together. Results are
saved separately under `runs/CO/FE/` and `runs/CO/FE_prime/` in a new timestamp
directory. Descriptor generation, electronic-structure calculations,
valence selection, and all MB encoders are unchanged by this addition.

The current default H2S descriptor files retain six valence AOs, while the
existing loader expects ten. The loader rejects that pre-existing mismatch;
this comparison does not change or repair those inputs. CO has eight valence
AOs for both FE and FE′ and passes the preparation checks.

## Scientific source map

The functions below were copied from the scientific reference files into the
three local files. There are **no imports from `canonical_pipeline`**, no path
injection, and no runtime dependency on its source tree. The historical
benchmark entry point is not used. The command uses the existing Python
environment only for installed NumPy, pandas, Torch, and PennyLane libraries.

| Component | Local implementation and original scientific source |
|---|---|
| FG raw features | `circuit.fg_features`, copied from `geometry.fg_features`; `circuit.geometry_contract` retains the stored-coordinate atom-order mapping |
| FG peroxide branch | Small FG unwrap branch copied from `preparation.prepare` |
| FE raw vector | `run1.prepare_runs`: `np.linalg.eigvalsh(sample.pij)`, where `pij` is the complete valence block of full-AO `P_L` |
| FG/FE scaling and model | `circuit.fit_constants`, `FingerprintSample`, `FingerprintModel`, `FingerprintCircuit`, `encode`, copied from `fingerprint.py`; dispatch copied from `preparation.construct_model` |
| MB-1 scaling | `circuit.compute_encoding_constants`, copied from `constants.py` |
| MB-1 model and circuit | `circuit.CorrelationEnergyModel`, `ParameterLayout`, `CachedCircuitBank`, copied from `model.py` and `circuit.py` |
| Split | `run1.load_population`: explicit three-way manifest by default, or odd IDs train/even IDs test with `--split-protocol 15-15` |
| Training and evaluation | `train.train`, `train.initialize_parameters`, `train.evaluate`, copied from `train.py` and `evaluate.py` |
| Seed aggregation | `train.compute_statistics`, copied from `compat.benchmark.py` |

FG uses stored full-precision Cartesian coordinates in Angstrom, reordered
into the canonical FG atom convention. Angles are in radians. Diatomics use
`[d, 1/d]`; XH2 uses `[XH1, XH2, HH, 1/XH1, 1/XH2, angle]`; NH3 uses three
NH distances, their reciprocals, then H1-N-H2 angle. H2O2 uses
`[OO, O1H1, O2H2, O1H2, O2H1, HH, 1/HH, O2-O1-H1 angle, O1-O2-H2 angle, H1-O1-O2-H2 dihedral]`.
Its last column is unwrapped on the ordered retained scan only if an adjacent
jump exceeds pi. The old fixed-geometry builder is not appropriate for the new
optimized geometries or for `q_A`, which is a displacement.

FE is **the ascending eigenvalues of the valence block of the full-AO Löwdin
RHF density matrix**:

```python
P_L = S_half @ P_RHF_AO @ S_half
P_val_L = P_L[np.ix_(valence_idx, valence_idx)]
x_FE = np.linalg.eigvalsh(P_val_L)
```

The saved `P_L` is checked against this full-AO transformation before selecting
the valence block. The adapter's legacy `sample.pij` field stores that complete
block, including its diagonal. Its canonical AO permutation leaves the
eigenvalues unchanged. FE uses no diagonal-only vector or Fock quantity, and
its dimension remains the valence AO count. This explicitly replaces the old
canonical FE raw descriptor while retaining its scaler and model.

Both fingerprints fit each column's mean and population
standard deviation on training rows only. Standard deviations below `1e-12`
encode to zero; other entries are scaled by pi/2 and clipped to [-pi, pi].

S, P, F, Gamma, Lambda and all Löwdin operations retain full-AO dimensions.
The adapter only selects entries from the already transformed full matrices.
New BeH2 uses Be,Hminus,Hplus storage while canonical electronic order is
Hminus,Be,Hplus: its final selected source AO indices are `[5,1,2,3,4,6]`.
The corresponding canonical addresses are `[0,2,3,4,5,6]`. Upstream tensors
are neither reordered nor reduced. Actual source indices are saved in the
population output.

MB-1 encodes P/F diagonals, with separate scalar mean/std fitted over all
training geometry/AO entries. Canonical physical-Pij p95/alpha fitting is
retained; the canonical model then zeros the pair encoder and includes its
IsingZZ(0) gates. No triple descriptor is passed to MB-1. Its reported input
width is `2 * number_of_valence_AOs`, excluding the inactive pair-fit dependency.

## Population and configuration

The all-30 population follows the accepted full-range Protocol-7 configuration
in `encoding_qml/results/protocol7_fresh_cf_30geom_fci_target/protocol7_config.json`.
The coordinates themselves come from the newly generated scan. Later cutoff
runs use different populations; their old integer geometry IDs are not applied
to the new grid.

Every molecule uses IDs **001–030**. In `15-15` mode, odd numbered IDs train
(15) and even numbered IDs test (15). In manifest mode, the supplied file sets
the three memberships and counts. All three models share the exact records,
targets and split. At runtime the audit prints every ID, target range, count
and input width before training. The table below describes `15-15` mode.

| Molecule | Retained/train/test | FG | FE | MB-1 input | MB-1 parameters |
|---|---:|---:|---:|---:|---:|
| LiH | 30/15/15 | 2 | 5 | 10 | 17 |
| BeH2 | 30/15/15 | 6 | 6 | 12 | 17 |
| H2O | 30/15/15 | 6 | 6 | 12 | 17 |
| NH3 | 30/15/15 | 7 | 7 | 14 | 17 |
| N2 | 30/15/15 | 2 | 8 | 16 | 17 |
| CO | 30/15/15 | 2 | 8 | 16 | 17 |
| HF | 30/15/15 | 2 | 5 | 10 | 17 |
| H2S | 30/15/15 | 6 | 6 | 12 | 17 |
| H2O2 | 30/15/15 | 10 | 10 | 20 | 19 |

Seeds 0–31, 500 epochs, batch size 8, Adam learning rate 0.02, float64 CPU,
analytic default.qubit, two HEA layers, and the padded-Z-plus-three-moments
linear readout are unchanged. Canonical model-specific initialization,
parameter block shapes, shuffle streams, and MSE updates are retained.
Per-epoch evaluation uses training and, when supplied, validation data.
Checkpoint restoration follows the selected protocol described above.
MB-1 has five circuit parameters (a,b,c,phi0,phi1); totals above include bias
and linear readout weights.

## Saved results

`population.json` includes exact coordinates, IDs, target energies, AO selection
and split. `split_manifest.json` records the exact memberships used: two groups
for `15-15`, three groups for `manifest`. `configuration.json` records the
protocol and checkpoint criterion. `audit.json` and `preprocessing.json` record
the audit, settings, and fitted training statistics. Each molecule/model/seed
directory contains predictions, metrics for the active splits, training history, and best
parameters. `smoke_result.json` is written only after all smoke runs pass.

`per_seed_metrics.csv`, `per_molecule_summary.csv`, `overall_per_seed.csv`, and
`overall_summary.csv` are written after complete seed coverage. Summary columns
mean, sample SD, median, and SE are in mHa; SE = SD/sqrt(32). Train and test
statistics are separate, with validation statistics also present in manifest
mode. Canonical code does not define one cross-molecule
scalar: the overall summary explicitly averages the nine molecule MAEs within
each seed, then applies canonical mean/SD/SE across seeds. Since all molecules
have equal split sizes, that mean also equals pooled split MAE.
