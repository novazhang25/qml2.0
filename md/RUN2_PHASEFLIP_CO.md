# CO MB2 phase-flip experiment

`codes/run2_phaseflip_co.py` imports the current repository's Run2 loader,
MB2 circuit and training loop. No historical `encoding_qml` implementation is
imported. Its Python environment can supply installed dependencies.

The loader selects `P_L[np.ix_(selected, selected)]` as `sample.pij`. For CO,
this is an **8 x 8 valence-AO matrix**, in full-AO index order
`[1, 2, 3, 4, 6, 7, 8, 9]` (zero based):
`C 2s, C 2px, C 2py, C 2pz, O 2s, O 2px, O 2py, O 2pz`.
Every geometry's actual source mapping and matrix shape are checked at launch.
The experiment does not select or rephase occupied MO coefficients.

The six independently trained models are original `MB2`, `MB2_flip_0`,
`MB2_flip_01`, `MB2_flip_012`, `MB2_flip_0123`, and `MB2_flip_01234`.
Each name lists the cumulative AO/qubit prefix whose signs are negative.
AO/qubit indices 5, 6 and 7 remain positive in every trained variant.
Each matrix is constructed directly from the **original** P, never from the
previous flipped matrix. These fixed vectors apply to every geometry, split and seed.
Only a newly allocated pair density `d[:, None] * P * d[None, :]` changes.
Original MB1 inputs, fitted constants, Lambda pair values and axis convention,
pair order, ZZ angle factors, HEA, readout and 19-parameter layout are reused.
All-minus is an identity sanity check only, never a trained seventh model.

The runner checks its population, target, AO mapping, split, preprocessing
and production settings against the read-only saved current CO Run2 reference
`results/run2/20260925_113307_117058`. It always creates a fresh matched MB2
baseline in its own output directory. Each seed uses the unchanged MB2
initialization and a fresh Adam optimizer for every model. The actual training
minibatches are checked against the same saved `train.SampleOrder(seed)` schedule.

Production retains seeds 0–31, 500 epochs, batch size 8, Adam learning rate
0.02, unscaled Hartree MSE, float64 CPU, and analytic `default.qubit` backprop.
Geometries 001, 003, ..., 029 are training; 002, 004, ..., 030 are test.
Checkpoint selection uses strictly improving training MAE with earliest ties.
Test evaluation occurs once after restoring the selected checkpoint.

## Commands

Full experiment (explicit execution mode required):

```bash
cd /Users/novaz/Desktop/qml2.0 && \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/Users/novaz/Desktop/qml2.0/results/run2_phaseflip_co/.mplconfig \
XDG_CACHE_HOME=/Users/novaz/Desktop/qml2.0/results/run2_phaseflip_co/.cache \
/Users/novaz/Desktop/encoding_qml/.venv/bin/python -B \
  codes/run2_phaseflip_co.py --production
```

Focused tests, without production training:

```bash
cd /Users/novaz/Desktop/qml2.0 && \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/Users/novaz/Desktop/qml2.0/results/run2_phaseflip_co/.mplconfig \
XDG_CACHE_HOME=/Users/novaz/Desktop/qml2.0/results/run2_phaseflip_co/.cache \
/Users/novaz/Desktop/encoding_qml/.venv/bin/python -B -m unittest discover \
  -s tests -p 'test_run2_phaseflip_co.py' -v
```

Use `--smoke-only` in place of `--production` for exactly one epoch, seed 0,
baseline plus all five variants. Use `--validate-only` for descriptor,
invariant, fixed-parameter prediction and initialization checks without training.

## Outputs

All new experiment results are written to a fresh timestamped child of
`results/run2_phaseflip_co/`. `--output-dir` can select a new child there;
existing directories, other roots and symlink escapes are rejected.

- Root and per-model configuration: phase vectors, full-AO indices and labels,
  zero-based pair order, training configuration, energy units and conventions.
- Population, split manifest, MB1 preprocessing, descriptor SHA256 hashes,
  parameter counts and invariant check results.
- `matched_protocol/seed_NN/`: common initial parameters and epoch/batch IDs.
- `runs/CO/MODEL/seed_NN/`: initial/best parameters, named weights, actual batch
  orders, predictions in Hartree, history, metrics and checkpoint metadata.
- Current Run2 metric tables plus `comparison.csv` with `model`, `seed`,
  `train_MAE`, `test_MAE`, `test_MAE_minus_matched_baseline`, all MAEs in **mHa**.
- `comparison_summary.csv`: means and sample SDs (`ddof=1`) across seeds for
  train MAE, test MAE and the paired baseline difference. One-seed SD is blank.
- `smoke_result.json` or `completion.json` only after successful completion.

The focused tests cover every geometry's symmetry, diagonal and eigenvalue
invariance; exact pair extraction, Lambda/MB1 preservation and immutable
source arrays; both uniform-sign feature/prediction identities; original gate
angles and unchanged architecture; all 32 initializations and 500-epoch batch
schedules; exact production reference matching; output isolation and summaries.

## Verification results

All 11 focused tests passed. The one-epoch seed-0 smoke run passed for all six
models in `results/run2_phaseflip_co/20260925_145336_529379/`.
Saved initial parameters and actual minibatch orders match across all six
models. Reloaded checkpoints reproduce every saved prediction exactly, and the
baseline initialization and first-epoch training MAE match the original Run2.
Descriptor SHA256 hashes remain unchanged. See `verification_report.json`,
`validation.json`, and `comparison.csv` in that output directory.
These are smoke results; full production training has not been launched.
