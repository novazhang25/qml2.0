# Run2: MB2 and MB2′

`codes/run2.py` uses the current `run1.py` loader, MB1 preparation, training
writer and full-run summary function, together with the current `train.py`.
Run2 skips the optional FE spectrum audit during preparation, so it neither
calculates nor prints unused density/Fock eigenvalues. Run1's default spectrum
audit remains enabled.
`codes/run2_model.py` adds the two pair encoders. The only Run1 circuit change
is extracting the original MB1 RY encoder into a shared function with the
same arithmetic, operations, parameter order and gradients.

## Architecture and parameters

The circuit is original MB1 → one ZZ layer → original two-layer HEA → original
scalar energy readout. All unique pairs `mu < nu` are applied once, before
the HEA. There are no triple gates, re-uploading, pair intercepts or new
pair scaling constants.

The shared original encoder uses
`gamma_mu = (pi/2) * tanh(a1*P_tilde_mu + b1*F_tilde_mu + c1)`.
Its P/F diagonal means and population standard deviations are fitted on
training samples only through the existing MB1 preparation.

- MB2: `gamma_munu = (pi/2) * tanh(a2*P_munu + b2*L_munu)`.
- MB2′ (`MB2_prime` in paths and arguments):
  `gamma_munu = (pi/2) * tanh(a2*P_munu)`.
- Both use `qml.IsingZZ(gamma_munu) = exp(-1j*gamma_munu*ZZ/2)`.

MB2′ has no `b2` entry in its trainable vector and receives no cumulant pair
inputs. In a joint run, its samples have `lambda_pairs=None`. A prime-only
run skips cumulant pair extraction. Both still use the current Run1 record
validation, including its existing full-AO integrity checks.

The original flat MB1 vector is retained in full, then `a2` and, for MB2,
`b2` are appended. Original names `a`, `b`, `c` in saved parameter maps denote
`a1`, `b1`, `c1`. For K = max(8, number of qubits):

| Model | Total trainable parameters | CO |
|---|---:|---:|
| MB1 | K + 9 | 17 |
| MB2 | K + 11 | 19 |
| MB2′ | K + 10 | 18 |

Initialization draws the original MB1 vector with its unchanged shape first,
then draws each new scalar from `0.05*N(0,1)` with the same local seed stream.
The bias uses the original training-target mean. Thus the original parameters
are bitwise equal to MB1 and all common parameters, including `a2`, are
bitwise equal between MB2 and MB2′ for matched seeds. Independent
`train.SampleOrder(seed)` streams keep minibatch ordering matched.

## Saved descriptor convention

The current dataset stores full-AO rank-four tensors as
`G[p,q,r,s] = Lambda_Methods[p,r,q,s]`. This convention is documented in
`codes/descriptor.py` in `extract_descriptors` and `T_REPAIR_MAPPING`, and is
consistent with saved `rdm_convention` and `rdm_metadata_json` fields.
Run2 checks those fields, the raw-axis HF disconnected product, and the saved
Methods subtraction identity. Run1 validates the full-AO Löwdin transform.
These are consistency checks; no upstream data is regenerated.

For the final Run1 source AO indices `i` and `j`, the pair entry is read
directly as `Lambda_L[i,i,j,j]`. Neither `Lambda_L[i,j,i,j]`, T, nor the saved
alternative correlated-density cumulant is used. Raw `P_munu` comes from
the same selected `P_L` block. No centering or normalization is added.
The retained MB1 `alpha` metadata is unused by both new pair layers.

Source selection comes directly from Run1, including BeH2's canonical
source order `[5,1,2,3,4,6]` and H2S's existing six-AO valence space.

## Current matched experiment

The completed original MB1 reference is
`results/run1/20260924_115423_466018`. It contains **CO only** with seeds
**0–31**. The launch command below matches this actual molecule and seed
set, and verifies its saved configuration, population and fitted MB1 scalers.

- Input: `results/descriptor`, geometry IDs 001–030.
- Training: 001, 003, …, 029; test: 002, 004, …, 030; no validation set.
- Target: `E_FCI - E_RHF_input`, unscaled Hartree.
- MSE loss; Adam; learning rate 0.02; batch size 8; 500 epochs.
- Float64 CPU; analytic `default.qubit`; Torch backpropagation.
- Checkpoint: strictly improving training MAE; earliest epoch on ties.
- Test evaluation occurs once after restoration and is never used to select
  a checkpoint.

The CLI also supports Run1's explicit three-way `manifest` protocol, which
requires `--split-manifest` and selects checkpoints by validation MAE.
It never silently generates a three-way split. The explicit `15-15` option
below selects the current completed Run1 benchmark protocol.

## Verification performed

90 tests passed: 20 new Run2 model/data tests, 66 current Run1 tests, and
4 current H2S loader tests. These cover exact original MB1 encoding,
gate order, the ZZ exponential, zero-input identity, `b2=0` equivalence,
finite nonzero pair gradients and Adam updates, checkpoint/optimizer
restoration, saved prediction reload, parameter counts, all 32 matched
initializations, minibatch ordering, axes, AO ordering, and train-only fitting.

A two-epoch CO/seed-0 smoke run of both models passed at
`results/run2/smoke_co_20260925`. Checkpoint reload reproduces all saved
predictions within `9.72e-17` Ha (CSV round-trip precision). These short-run
MAEs verify execution and are not full experiment results:

| Model | Train MAE (mHa) | Test MAE (mHa) |
|---|---:|---:|
| MB2 | 20.9803990105 | 21.9463883247 |
| MB2′ | 20.9801129083 | 21.9461437489 |

No production training was run. Existing Run1 results and upstream descriptor
files were not changed.

## Launch the full experiment

Run this command from a terminal. The existing `encoding_qml` environment
supplies installed libraries only; its trainer and split rules are not imported.

```bash
cd /Users/novaz/Desktop/qml2.0 && \
MPLCONFIGDIR=/private/tmp/qml-run2-mpl \
/Users/novaz/Desktop/encoding_qml/.venv/bin/python -B codes/run2.py \
  --input-dir /Users/novaz/Desktop/qml2.0/results/descriptor \
  --molecules CO \
  --methods MB2 MB2_prime \
  --split-protocol 15-15 \
  --epochs 500 \
  --match-run1 /Users/novaz/Desktop/qml2.0/results/run1/20260924_115423_466018
```

The output is a new timestamped directory under `results/run2/`.
`--output-dir` may name a new directory under that root; existing paths are
rejected. `--smoke-only` caps execution at two epochs, one molecule and seed 0.
`--validate-only` performs preparation and reference checks without training.

The output saves `configuration.json`, `population.json`, `split_manifest.json`,
`preprocessing.json`, `parameter_counts.json` and `audit.json`. Separate
`runs/<molecule>/<model>/seed_<NN>/` directories contain the same Run1
predictions, metrics, history, result and best-theta files, plus
`initial_theta.npy` and a named `parameters.json`.

MAE CSVs retain Run1's format: `per_seed_metrics.csv`,
`per_molecule_summary.csv`, `overall_per_seed.csv` and `overall_summary.csv`.
Mean, sample SD, median and SE are in mHa. In a one-seed smoke run SD and SE
are empty, because a sample SD cannot be estimated from one seed.

For prediction loading, `run2.load_prediction_run(output, molecule, method,
seed)` returns the model, all saved best weights, and samples grouped by
saved split. It restores fitted constants and verifies saved population and
parameter identities without fitting scalers again.
