# MB-1' matched Run 1 control

Implementation and tests have been compiled. Following the reported test
failures, the 11 focused control tests and 9 existing circuit regression tests
pass. Production training has not been launched; MB-1' MAE values remain pending.

## Encoder and parameter counts

Internal identifier: `MB1_PRIME`. Table and plot label: `MB-1'`.

\[
\gamma_\mu=\frac{\pi}{2}\tanh(a_1\widetilde P_\mu+c_1).
\]

`a1` and `c1` are shared across the qubits. There is no `b1` parameter in
the trainable tensor or optimizer. The encoder never reads or standardizes F.
P retains MB-1's original definition and training-only pooled scalar
population mean and standard deviation. The shared preparation code still
validates the source F descriptors and records the original F statistics;
these do not enter the MB-1' encoder, circuit, or readout.

For n active AOs and the existing two shared ansatz layers, let K = max(8, n).

| Model | Trainable encoder parameters | Total trainable parameters | CO total |
| --- | --- | --- | --- |
| MB-1 | 3: a1, b1, c1 | K + 9 | 17 |
| MB-1' | 2: a1, c1 | K + 8 | 16 |

The remaining parameters are the same two shared RY ansatz angles, one bias,
and K + 3 readout weights. The directed CNOT rings, Z expectations, padding,
three Z moments, linear readout, and depth remain shared with MB-1.

Initialization uses the original MB-1 normal-vector draw for each seed and
discards its b1 entry before creating the trainable tensor. Thus all surviving
parameters have exactly the same initial values as MB-1 for that seed. The
bias is overwritten with the same training target mean. The discarded random
draw is not a trainable or frozen model parameter.

## Reference and population

Reference directory:
`/Users/novaz/Desktop/qml2.0/results/run1/20260924_115423_466018`.
This is the completed benchmark using the current local Run 1 implementation.
It contains CO only; the command therefore selects CO only.

- Dataset: `/Users/novaz/Desktop/qml2.0/results/descriptor`.
- Retained CO geometries: 001–030, with the original AO mapping (8 qubits).
- Training: odd geometry IDs (15). Test: even geometry IDs (15). No validation set.
- Target: E_FCI − E_RHF_input, unscaled Hartree.
- Seeds: 0–31, with the existing independent minibatch shuffle generator.
- Optimizer: the existing single-vector Adam; learning rate 0.02.
- Epochs: 500. Batch size: 8.
- Checkpoint: strictly improving training MAE, earliest epoch on ties.
- Evaluation and aggregation: existing Run 1 functions, test MAE in mHa,
  seed arithmetic mean and sample SD (ddof=1). Overall results keep the
  existing equal-weight molecule aggregation.

The optional `--match-mb1-run` guard verifies the saved reference configuration,
full population metadata and memberships, fitted constants, and all 32 seeds
before creating the new result directory or training. It requires an
MB1_PRIME-only run. Existing output paths are rejected. The reference is only read.

## Per-molecule comparison

All entries are test MAE in mHa. Reference values are from the saved Run 1
`per_molecule_summary.csv`; no MB-1 retraining is needed.

| Molecule | MB-1 MAE mean | MB-1 MAE SD | MB-1' MAE mean | MB-1' MAE SD | ΔMAE |
| --- | --- | --- | --- | --- | --- |
| CO | 1.046504628012002 | 1.2727316399663917 | Pending execution | Pending execution | Pending execution |

ΔMAE = MAE(MB-1') − MAE(MB-1).
After execution, the command writes the completed table to
`mb1_prime_comparison.csv` and `mb1_prime_comparison.md` inside the new result
directory and prints it to the terminal. These use the existing seed-statistics
function and the stored MB-1 per-seed metrics.

## Modified files

- `codes/circuit.py`: optional two-parameter encoder in the shared MB model;
  F is omitted from MB-1' quantum inputs. Existing MB-1 formula is retained.
- `codes/train.py`: preserve seeded MB-1 parameter draws when removing b1.
  Optimizer, loss, batching, checkpointing, evaluation and aggregation are unchanged.
- `codes/run1.py`: method selection, preparation audit, reference matching,
  and comparison reporting through the existing launcher.
- `codes/plot_run1_results.py`: accept `MB1_PRIME` and label it `MB-1'`.
  Existing default model selection is unchanged.
- `tests/test_run1_mb1_prime.py` (new): requested five properties, original
  MB-1 circuit/readout regression, seeded initialization and optimizer parity,
  matched preparation, reference guards, comparison statistics and output protection.
- `md/MB1_PRIME_CONTROL.md` (new): this implementation and execution report.

FG, FE and MB-1 retain their scientific definitions and default behavior.
No descriptor data or existing benchmark results have been edited.

## Verification and execution

Python bytecode compilation and whitespace checks pass. All 11 tests in
`test_run1_mb1_prime.py` and all 9 tests in `test_run1_circuit_repairs.py` pass.
No production training or benchmark evaluation was launched.

The two initial failures were overly strict bitwise comparisons across
different float64 computation paths. With identical encoder angles, quantum
outputs, readout features and weights, the final prediction differed by
2.7755575615628914e-17 Ha. The independently constructed reference gradient
differed by at most 1.1102230246251565e-16. Only these two assertions now use
`rtol=1e-14, atol=1e-15`; model code is unchanged by this repair. Encoder formula,
F-invariance and matching shared parameters retain bitwise checks, and the
full-prediction test additionally checks identical angles, features and weights.

The command below runs the focused tests first, then launches only MB-1'.
The shared Python environment already contains Torch, PennyLane and pandas.

```bash
cd /Users/novaz/Desktop/qml2.0 && \
/Users/novaz/Desktop/encoding_qml/.venv/bin/python -m unittest discover -s tests -p 'test_run1_mb1_prime.py' && \
/Users/novaz/Desktop/encoding_qml/.venv/bin/python codes/run1.py \
  --input-dir /Users/novaz/Desktop/qml2.0/results/descriptor \
  --molecules CO \
  --methods MB1_PRIME \
  --split-protocol 15-15 \
  --epochs 500 \
  --match-mb1-run /Users/novaz/Desktop/qml2.0/results/run1/20260924_115423_466018 \
  --output-dir /Users/novaz/Desktop/qml2.0/results/run1/mb1_prime_matched_20260925
```

Output directory:
`/Users/novaz/Desktop/qml2.0/results/run1/mb1_prime_matched_20260925`.
It has not been created. Choose a new output path if this command has already
completed or started a previous run; the launcher never overwrites it.
