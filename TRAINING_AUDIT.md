# Current training and evaluation audit

Audit date: 2026-09-24. Authoritative scope: the working-tree implementation, not Git HEAD or historical reports. This audit creates only this requested report; no source, results, manifests, or data were modified. No training or production entry point was run.

## 1. Executive summary

1. **The checkout is not unconditionally train/test-only.** `main` defaults to `manifest`, requiring nonempty train/validation/test groups and selecting by validation MAE. The explicitly selected `15-15` path uses 15 training and 15 test geometries, no validation, and selects by training MAE. Describing the default as train/test-only is a **CURRENT CODE CONFLICT** with the intended protocol. [R:35–37, 244–255, 450–473; T:183–233]
2. In `15-15`, **odd one-based geometry IDs train; even IDs test**. Equivalently, even zero-based positions in the ID list train, odd positions test. Every model uses the same deterministic membership. [R:126, 254–255, 362–365]
3. FG/FE use training-only, per-column population mean/std and clipped angle encoding. MB-1 uses separate scalar P/F population statistics pooled across training geometries and selected orbitals, followed by a trainable tanh encoder. Targets remain unscaled correlation energies in Hartree. [C:380–398, 462–465, 740–762; R:225–240]
4. Standard runs use 32 seeds (0–31), 500 epochs, batch size 8, Adam at 0.02, and minibatch mean squared error. The selected end-of-epoch MAE checkpoint is restored, with earliest exact ties retained. Test prediction metrics are evaluated once per run after restoration. [R:286–288, 382–417; T:145–173, 206–233]
5. **The current default dataset blocks an all-nine-molecule run even with `15-15`.** Loading `results/descriptor/H2S/001.npz` fails the valence-index check: saved `[2,6,7,8,9,10]`, required `[1,2,3,4,5,6,7,8,9,10]`. Thus the 32-seed, nine-molecule protocol is implemented but cannot presently execute against the default inputs. No repair was attempted. [R:164, 469–474; C:124–128]
6. No test labels determine the best epoch. However, labels are loaded and audited before training, and smoke-run test finiteness gates continuation. H2O2 FG also has a full-population torsion-unwrapping branch: an unqualified claim that *all preprocessing* is independent of test inputs is too strong. That branch is inactive on the inspected default H2O2 data. [R:225–241, 322–329, 358–359, 393–397, 483–495]

## 2. Evidence and verification scope

Source-reference shorthand below gives file, function context, and inclusive line spans:

- **R** = [codes/run1.py](/Users/novaz/Desktop/qml2.0/codes/run1.py:1).
- **T** = [codes/train.py](/Users/novaz/Desktop/qml2.0/codes/train.py:1).
- **C** = [codes/circuit.py](/Users/novaz/Desktop/qml2.0/codes/circuit.py:1).
- **D** = [md/RUN1.md](/Users/novaz/Desktop/qml2.0/md/RUN1.md:1), used only as a comparison description, never to establish current behavior.

Read-only runtime checks imported the current modules using Python `-B`, called `load_population` and `prepare_runs` directly, and inspected installed library signatures. These calls load/validate inputs, fit scalers in memory, initialize preparation-check parameters, and construct the MB-1 gate tape; they do not train or write run outputs. Third-party imports created temporary font-cache files outside the repository.

All eight other molecules passed independent loading and preparation for all three models, with 15/0/15 counts. H2S failed as described above. The all-nine loading attempt fails before preparation/training and before creation of the output directory. This is not a completed benchmark or a convergence test.

Runtime inspected: Python 3.13.2, NumPy 2.5.1, pandas 3.0.5, Torch 2.13.0, PennyLane 0.45.1. These are the audit environment's installed versions, not versions pinned by Run 1.

Source SHA-256 at inspection:

| File | SHA-256 |
|---|---|
| codes/run1.py | `aa5e94a7a7ac8141181bc4b39fe209bc02d2b26cbf48a6a34c346a8fb8b394ee` |
| codes/train.py | `9e2ab522c78f2fd9cd86dd58a162bab63ab6c279311f0c43d2c35c0e60601fc7` |
| codes/circuit.py | `f4e14733f9fa327485d0d583cbd5b64fa1633d211b9f723dcfd6dd5651d36c8b` |
| results/descriptor/manifest.json | `a49ec891ea44c4d51bc12286f91efda55700be430519002b2c19a5126ec68d3b` |

The existing `results/run1/20260924_115423_466018/configuration.json` declares CO-only, `15-15`, seeds 0–31, 500 epochs. This identifies that saved configuration, not proof of its completion, source equivalence, or an all-nine benchmark. Current behavior below comes from source and direct preparation checks.

## 3. Dataset, split, and loading

### Exact split

`GEOMETRY_IDS` is `001,...,030`. `load_population` sets training to `GEOMETRY_IDS[::2]` and testing to `GEOMETRY_IDS[1::2]` only when `split_protocol == '15-15'`. [R:126; R:244–255]

- Training: `001,003,005,007,009,011,013,015,017,019,021,023,025,027,029`.
- Test: `002,004,006,008,010,012,014,016,018,020,022,024,026,028,030`.
- Validation: empty.

This is an **alternating train--test split by geometry ID**. Do not write “even-indexed geometries train” without specifying zero-based indexing: `sample.geometry_index` is actually `int(geometry_id)`, so odd values of that field train. [R:240]

`retained_positions` sorts by `(scan_coordinate, geometry_index)` within molecule; preparation orders each population by that retained position. Sorting changes presentation/training input order, not membership. No seed participates in assigning memberships. [R:112–118, 308–315]

| Molecule | Required total | Train | Validation | Test | Default-data preparation |
|---|---:|---:|---:|---:|---|
| LiH | 30 | 15 | 0 | 15 | PASS |
| BeH2 | 30 | 15 | 0 | 15 | PASS |
| H2O | 30 | 15 | 0 | 15 | PASS |
| NH3 | 30 | 15 | 0 | 15 | PASS |
| N2 | 30 | 15 | 0 | 15 | PASS |
| CO | 30 | 15 | 0 | 15 | PASS |
| HF | 30 | 15 | 0 | 15 | PASS |
| H2S | 30 | 15 | 0 | 15 | BLOCKED; counts are protocol requirements |
| H2O2 | 30 | 15 | 0 | 15 | PASS |

FG, FE, and MB-1 receive identical ordered train/validation/test sample IDs, enforced explicitly. [R:334–365]

### Default manifest branch and filtering

The default `manifest` branch requires an externally supplied JSON with exactly three nonempty, disjoint groups whose union is all 30 IDs for each selected molecule. Counts are determined by that manifest and are **not fixed in code**; alternation is not enforced. A fixed manifest is deterministic. No manifest was supplied with this audit request, so exact three-way manuscript counts are not verifiable. Running with no protocol/manifest arguments fails argument parsing. [R:39–88, 450–462]

There is **no geometry cutoff, target threshold, or geometry-dropping filter before or after splitting** in the current Run 1 path. It selects requested molecules, requires all 30 IDs, checks record status and physical/numerical consistency, and aborts on invalid input rather than dropping it. Every resulting manifest row has `included=True`. Any upstream choice of scan range is already embodied in the saved input files; Run 1 does not call the scan generator. [R:90–118, 176–241, 256–281]

### Physical preprocessing shared by the models

`load_record` reads saved full-AO arrays, checks basis/units, RHF convergence, matrix symmetry, overlap positivity, full-AO Löwdin transforms, cumulant consistency, and saved descriptor consistency. It uses

\[
P_L=S^{1/2}P_{AO}S^{1/2},\qquad F_L=S^{-1/2}F_{AO}S^{-1/2}.
\]

These relations are checked against saved arrays before final AO selection; no fresh electronic-structure calculation is called. Canonical selection follows atom occurrence and AO labels. BeH2 has an explicit source-to-canonical reordering. “Valence” here means the code's selected AO list, which for H2S excludes only sulfur 1s and retains ten AOs, not a conventional six-orbital outer-shell space. [R:141–241; C:104–132]

The target is checked as `E_FCI - E_RHF_input`, in Hartree, with no target centering, standardization, or division by molecule size. The bias initialization described below does not transform labels. `TargetDefinition`'s class defaults for UHF do not override Run 1's explicit RHF target. [R:125, 225–240; C:21–29]

## 4. Model-specific preprocessing and scaling

### FG and FE

FG derives distances in Å, reciprocals in Å⁻¹, and angles in radians from full-precision stored Cartesian coordinates in the FG atom order. Raw columns are:

| Molecules | Ordered FG features |
|---|---|
| LiH, N2, CO, HF | bond distance, reciprocal distance |
| BeH2, H2O, H2S | X–H1, X–H2, H1–H2 distances; reciprocals of the first two; H1–X–H2 angle |
| NH3 | three N–H distances; their three reciprocals; H1–N–H2 angle |
| H2O2 | O1–O2, O1–H1, O2–H2, O1–H2, O2–H1, H1–H2 distances; reciprocal H1–H2; two O–O–H angles; H1–O1–O2–H2 torsion |

Distance/angle validity checks reject degenerate inputs; the arccos argument is clipped to [-1,1] for geometric evaluation. [C:137–230; R:321]

FE is the ascending eigenvalue vector `np.linalg.eigvalsh(P_val_L)`, using the entire selected block including off-diagonals. It is not the vector of density diagonal entries, and it uses no Fock features. Scaling is by eigenvalue rank/feature column, not by AO label. [R:223–224, 240, 321]

For each molecule and method independently, with training size N and feature column j:

\[
\mu_j=N^{-1}\sum_{i\in\mathrm{train}}x_{ij},\quad
\sigma_j=\sqrt{N^{-1}\sum_{i\in\mathrm{train}}(x_{ij}-\mu_j)^2}.
\]
\[
\theta_{ij}=\begin{cases}
0,&\sigma_j<10^{-12},\\
\operatorname{clip}\left[\frac{\pi}{2}\frac{x_{ij}-\mu_j}{\sigma_j},-\pi,\pi\right],&\text{otherwise}.
\end{cases}
\]

`ddof=0` is explicit. Near-constant columns have `sigma_safe=1`, then their encoded values are explicitly set to zero for **all** splits, including a test value different from the training mean. Exactly `sigma=1e-12` is not masked. There is no tanh for FG/FE. Scalars are not pooled over features or molecules. Train-only manifest rows and samples are passed to the fitter; training IDs are verified after fitting. [R:314–329, 338–339; C:740–762]

**Full-population preprocessing caveat:** H2O2 torsion is examined across all retained geometries, and `np.unwrap` is applied if any adjacent jump exceeds π, before selecting training fingerprints for fitting. Thus test covariates could indirectly change training torsion values/statistics for another accepted input scan; test targets never enter this operation. On the actual default H2O2 files, the maximum adjacent torsion difference was `1.3322676295501878e-15` radians, so the branch did not execute. The accurate claim is “scaler statistics are fitted on training rows only,” with this sequence-processing qualification. [R:322–329]

### MB-1

For each molecule, flatten all selected orbital diagonal entries across training geometries separately for P and F. With n selected orbitals:

\[
\mu_P=(Nn)^{-1}\sum_{i\in\mathrm{train}}\sum_{a=1}^{n}P_{ia},\quad
\sigma_P^2=(Nn)^{-1}\sum_{i,a}(P_{ia}-\mu_P)^2,
\quad \widetilde P_{ia}=(P_{ia}-\mu_P)/\sigma_P,
\]

and identically for F. This is **pooled across orbitals**, not per-orbital scaling. Both standard deviations explicitly use `ddof=0`. There is no epsilon floor or constant-orbital mask: a nonpositive pooled scale aborts; arbitrarily small positive scales are accepted. A constant individual orbital remains encoded using the pooled statistics. [C:263–276, 308–327, 365–398, 609–631]

The trainable shared encoder is

\[
\theta_{ia}=\frac{\pi}{2}\tanh(a\widetilde P_{ia}+b\widetilde F_{ia}+c).
\]

There is no separate hard clipping of standardized P/F. The tanh bounds the encoded angles mathematically to (-π/2,π/2). [C:460–465]

Training-only off-diagonal selected-density entries for `i<j` also fit

\[
q_{95}=\operatorname{percentile}_{95}(|P_{ij}|),\qquad
\alpha=\operatorname{atanh}(0.9)/q_{95},
\]

using NumPy's explicit `method='linear'`. A nonpositive q95 is rejected. This is a real preparation dependency but **does not affect MB-1 predictions**: MB-1 passes no pair matrix and queues no pair gates. No T/triple scaling or gate is used. Do not describe alpha as a trainable MB-1 parameter or report IsingZZ(0) gates. [C:278–286, 380–391, 466–480, 623–631; R:341–357]

## 5. Parameters, readout, and initialization

Let n denote model qubits (FG feature count or selected AO count), and K=max(8,n). Both model families measure each real qubit's Z expectation and build

\[
f(z)=(z_1,\ldots,z_n,\underbrace{0,\ldots,0}_{K-n},
 n^{-1}\sum z_i,n^{-1}\sum z_i^2,n^{-1}\sum z_i^3),\qquad
\widehat E=b_E+w^Tf(z).
\]

Padding does not create additional qubits. Moments are calculated before padding, over n real qubits. Each of two ansatz layers applies a shared RY angle to every qubit followed by the ordered directed CNOT ring `q -> (q+1) mod n`. There are two shared angles total, not two angles per qubit. [C:415–423, 567–582, 640–659, 805–819, 879–899]

| Model | Optimized parameter blocks | Total parameter-vector entries |
|---|---|---:|
| FG | 2 shared ansatz angles; 1 energy bias; K+3 readout weights | K+6 |
| FE | 2 shared ansatz angles; 1 energy bias; K+3 readout weights | K+6 |
| MB-1 | 3 shared encoder coefficients a,b,c; 2 shared ansatz angles; 1 energy bias; K+3 readout weights | K+9 |

FG/FE Adam receives three tensors in a single optimizer parameter group, sharing storage with the flat parameter vector, with gradients bound back to those blocks. MB-1 Adam receives one flat tensor in one group. These are separate parameter blocks, not different learning-rate groups. [C:521–556, 764–792, 842–872; T:54–64]

| Molecule | FG qubits | FE/MB-1 qubits | MB-1 raw P/F width | FG parameters | FE parameters | MB-1 parameters |
|---|---:|---:|---:|---:|---:|---:|
| LiH | 2 | 5 | 10 | 14 | 14 | 17 |
| BeH2 | 6 | 6 | 12 | 14 | 14 | 17 |
| H2O | 6 | 6 | 12 | 14 | 14 | 17 |
| NH3 | 7 | 7 | 14 | 14 | 14 | 17 |
| N2 | 2 | 8 | 16 | 14 | 14 | 17 |
| CO | 2 | 8 | 16 | 14 | 14 | 17 |
| HF | 2 | 5 | 10 | 14 | 14 | 17 |
| H2S | 6 | 10 | 20 | 14 | 16 | 19 |
| H2O2 | 10 | 10 | 20 | 16 | 16 | 19 |

Counts follow C:104–132, 203–214, 731–738 and the layouts. H2S counts describe current code, not successful execution on its incompatible default files.

All non-bias entries are initially drawn from a zero-mean normal distribution with standard deviation 0.05 (variance 0.0025). The energy bias is overwritten with the ordered training target mean, in Hartree, and is identical across seeds for a fixed molecule/split. Bias is subsequently optimized. [T:18–32; C:842–847]

There is no explicitly frozen subset of the parameter vector. However, K−n weights multiply exact padded zeros and have zero data gradient; with the current zero weight decay they stay at their random initial values. They are included in the counts above but are functionally inactive. The readout mean-Z feature is also linearly redundant with individual Z features; counts are allocated parameter entries, not independent identifiable degrees of freedom. Scaler constants, raw FG/FE encoding, and unused alpha are fixed, not optimizer parameters.

### Exact RNG behavior

Each run receives the integer seed directly from `range(32)`. Initialization and shuffling create **separate CPU Torch generators initialized to the same seed**; neither consumes the other's state. Shuffle state persists across epochs within a run. [R:287, 493–495; T:28–29, 34–43, 206–208; C:844–845]

- MB-1 draws one vector of `model.num_parameters` normal values, then overwrites the bias at index 5. The overwritten draw is retained in RNG consumption.
- FG/FE draw three successive blocks with shapes `(2,)`, `()`, and `(K+3,)`, then overwrite the scalar bias. Replacing those draws with one flat draw is not guaranteed to preserve values.
- Run seed identity affects both initialization and minibatch order. Equal seeds give matched permutation streams for equal-sized ordered training populations. Equal seeds do **not** imply the same parameter values between MB-1 and fingerprints because shapes and draw ordering differ.
- Preparation's separate MB-1 seed-0 initialization does not advance later run-local generators. Cached QNodes do not contain learned parameter state. No global Torch seeding, NumPy shuffle, or global deterministic-algorithm setting is called in this path.

## 6. Optimization and checkpoint selection

| Setting | Exact current behavior | Source |
|---|---|---|
| Optimizer | `torch.optim.Adam`; lr is the only optimizer keyword explicitly supplied | T:54–64; C:849–855 |
| Learning rate | 0.02 | R:288, 382 |
| Epochs | 500 by default; positive `--epochs` override; smoke-only at most 2 | R:454–465 |
| Minibatches | 8; retain last partial batch, hence 8+7 for 15 training geometries | T:45–52 |
| Loss | mean squared target-prediction error per minibatch, in Ha² | T:66–88 |
| Shuffle | fresh `torch.randperm(N, generator=order.generator).tolist()` every epoch | T:34–43, 211–214 |
| Weight decay | not supplied; installed Adam default 0 | T:64; C:854; runtime signature |
| Betas/epsilon | not supplied; installed defaults (0.9,0.999), 1e-8 | same |
| Other Adam options | not supplied; installed defaults include amsgrad=False, maximize=False, foreach=None, fused=None, decoupled_weight_decay=False | same |
| Clipping/scheduler | no gradient clipping, no learning-rate scheduler | T:75–91, 206–233 |
| Duration | all requested epochs; no early stopping | T:211–233 |

For minibatch B, loss is `|B|^{-1} sum (prediction-target)^2`. Each batch causes one Adam step; the seven-sample final batch is not rescaled to eight or dropped. At the default duration this gives 1,000 updates per 15-15 run. End-of-epoch training MAE is recomputed on the full training set; it is not the average of minibatch losses. Training history contains MAE, not an epoch MSE series. [T:70–73, 203, 211–231]

### Exact selection rule

For `15-15`, `main` passes `checkpoint_selection='train'`; `train` requires an empty validation sequence. After every completed epoch, it computes full-training MAE, saves a checkpoint only if the MAE is **strictly less** than the best so far, and runs all epochs. The initial untrained vector (epoch 0) is not a candidate. Exact ties retain the earlier epoch, with no tolerance-based tie rule. [R:473; T:145–165, 194–198, 211–232]

At the end, the best parameter vector is restored. Therefore the returned/saved model is not necessarily the final epoch. A checkpoint stores a deep copy of Adam state in memory, but restoration copies only parameters; the returned final optimizer state corresponds to the last epoch. Only the selected theta is saved by Run 1, not a resumable optimizer checkpoint. [T:163, 167–173, 232–233; R:417]

**CURRENT CODE CONFLICT:** validation selection remains the default in `main`, `train_one`, `train`, and `BestCheckpoint`; manifest mode still evaluates validation MAE every epoch and selects by it. This is active supported behavior, not merely old comments. The explicit `15-15` branch is internally consistent and does not accidentally call validation evaluation. [R:368, 450, 473; T:135, 183, 224–230]

## 7. Test evaluation and information exposure

`train` never receives the testing sequence. `train_one` evaluates the restored theta on training and test populations, skipping validation in training-selection mode. Test `evaluate` is called once per molecule/model/seed, after restoration, with an explicit count assertion. It evaluates all test samples through ordered minibatches; “once” means one complete test pass, not one QNode execution. No test MAE is calculated per epoch. [R:382–405; T:93–102, 183–233, 256–266]

For signed errors e_i=prediction_i−target_i, saved metrics are

\[
\mathrm{MAE}=N^{-1}\sum|e_i|,\quad
\mathrm{RMSE}=\sqrt{N^{-1}\sum e_i^2},\quad
\mathrm{maxAE}=\max|e_i|,\quad
\mathrm{amplitude}=\max e_i-\min e_i.
\]

Each is emitted in Hartree and mHa (factor 1,000). MSE optimization loss is Ha²; final metric output does not include a separate MSE field. [T:244–254]

Per run, `predictions.csv` saves geometry IDs, scan/bond coordinates, targets, predictions, and **signed per-geometry errors** in Hartree for each active split. Absolute errors can be obtained from the signed-error column but are not separately saved. `metrics.csv`, `result.json`, `training_history.csv`, and `best_theta.npy` save evaluation summaries, selected epoch/MAE, history, and restored parameters. [R:398–417]

“No test-label checkpoint selection” is justified. “Test labels are never inspected until final evaluation” is not: all labels undergo consistency checks at load time, whole-population target min/max are printed during preparation, and targets are written into population metadata. Those checks do not choose an epoch or fit scalers. Smoke-run test predictions/metrics must be finite before the remaining sweep proceeds, so there is a test-dependent numerical-success gate, but no threshold on test accuracy or search over test-selected models. [R:225–241, 358–359, 396–397, 475, 483–495]

## 8. Seeds, independence, and execution

The full sweep trains each selected molecule/model for every seed 0 through 31, 32 runs per pair, matched across FG/FE/MB-1. Each call creates fresh theta, Adam state, and shuffle generator; there is no warm start across seeds or transfer between molecules/models. Fitted scalers and stateless model/circuit caches are reused because the split is fixed. [R:304–340, 483–496; T:206–208]

With all nine valid molecules there would be 9×3×32=864 runs. The H2O seed-0 runs (or first selected molecule if H2O is absent) serve as the smoke gate and are counted in the sweep, not retrained or additional seeds. In a normal run these are full-duration runs; `--smoke-only` instead uses just seed 0 and at most two epochs and returns before aggregation. `--validate-only` returns without training. [R:465, 472, 481–496]

Both model families optimize CPU float64 real parameters and use float64 real features/readouts. No CUDA/MPS transfer is made. Quantum execution is PennyLane `default.qubit` with Torch interface and `diff_method='backprop'`. MB-1 explicitly supplies `shots=None`, backend, and fixed device seed 0 via `CircuitIdentity`; FG/FE omit shots and device seed in the constructor. The installed FG/FE device reports `Shots(total=None)`, so it is also analytic, without shot noise. Do not claim the run seed is passed to the FG/FE device. [C:402–412, 458, 491, 558–573, 809–819, 842–845, 883; T:28–32, 60–64]

No explicit thread-count, deterministic-algorithm, GPU determinism, or precision-default global setting is imposed. Local generators and analytic simulation make the intended trajectories reproducible within a compatible runtime, but the code does not guarantee bitwise equality across versions/platforms. Circuit state storage is complex internally; “float64” here states the explicit real training/data precision, not a claim that a statevector has real dtype.

## 9. Result aggregation

For molecule m, method k, seed s, first compute geometry-averaged split MAE in mHa:

\[
A_{mks}=\frac{1000}{N_m}\sum_{i\in\mathrm{split}(m)}|\widehat E_{mksi}-E_{mi}|.
\]

For each molecule/model, summaries calculate mean, sample SD, n, median, and SE over the 32 seed MAEs:

\[
\bar A_{mk}=\frac1{32}\sum_s A_{mks},\quad
SD_{mk}=\sqrt{\frac1{31}\sum_s(A_{mks}-\bar A_{mk})^2},\quad
SE_{mk}=SD_{mk}/\sqrt{32}.
\]

`compute_statistics` uses pandas groupby `std`, whose installed default is **ddof=1** (not explicitly passed). Median is computed; no confidence interval or bootstrap is computed. Summary statistics aggregate MAE only, although per-seed outputs contain the other evaluation metrics. [T:268–272; R:434–440]

For the overall statistic, **average molecule MAEs within the same seed first**, equally weighting the M selected molecules:

\[
O_{ks}=\frac1M\sum_{m=1}^M A_{mks},\quad
\bar O_k=\frac1{32}\sum_s O_{ks},\quad
SD_{O,k}=\sqrt{\frac1{31}\sum_s(O_{ks}-\bar O_k)^2},\quad
SE_{O,k}=SD_{O,k}/\sqrt{32}.
\]

Overall median is the median of these 32 O values. Uncertainty is seed variation of the per-seed molecule average, not SD across molecules, not pooled geometry-error SD, and not an average of molecule SEs. Matched-seed cross-molecule covariance is therefore retained. Equal geometry counts in `15-15` make the within-seed average numerically equivalent to a pooled geometry MAE; arbitrary manifest counts need not. A molecule subset changes M; “ALL” means all selected molecules, not automatically nine. [R:421–444]

Before writing summaries the code requires exactly seeds 0–31 once for every molecule/model/active split. It emits `per_seed_metrics.csv`, `per_molecule_summary.csv`, `overall_per_seed.csv`, and `overall_summary.csv`. Train/test summaries are separate; manifest mode also summarizes validation. Renaming `mae_mHa` to `test_MAE_mHa` is an internal reuse of the statistics function, not mixing split values. [R:425–444, 496]

## 10. Current conflicts and limitations

- **CURRENT CODE CONFLICT — default protocol:** the requested train/test-only narrative applies only to explicit `15-15`, not the default command or low-level function defaults.
- **CURRENT CODE CONFLICT — default H2S data:** current code expects ten selected AOs, but the first default H2S record supplies the old six. The all-nine protocol stops before training. The other eight independently passed preparation. This report does not certify a completed nine-molecule benchmark.
- **Preprocessing qualification:** H2O2's full-scan unwrap can use held-out covariates, although inactive on the inspected data; all scaler fit reductions themselves are restricted to training rows.
- **Test-access qualification:** labels are loaded/audited before training, and test finiteness gates the smoke stage, but no test metric selects a checkpoint.
- **Documentation mismatch:** D:132–134 describes MB-1 IsingZZ(0) gates, while current MB-1 has no pair gates; D's historical H2S table reports the old dimensions, with a warning at its top. Current H2S FE/MB-1 dimensions are ten and parameter counts 16/19.
- **Not verified:** exact three-way manuscript IDs, provenance of any earlier manuscript draft, completion or reproducibility of existing saved training trajectories, and convergence over 32 seeds. No historical report was used to infer executable behavior.

## 11. Paper-ready LaTeX subsection

**Applicability:** the text below describes the explicit `15-15` protocol at 500 epochs. It must not be presented as evidence that the presently blocked all-nine default dataset has completed that experiment. The H2O2 full-scan unwrap branch was inactive on the inspected dataset. If another dataset activates it, qualify the preprocessing description accordingly.

```latex
\subsection{Training and Evaluation}
For each molecule, the 30 geometries were assigned deterministically using
an alternating train--test split: odd geometry IDs (1,3,\ldots,29) formed
the 15-sample training set and even IDs (2,4,\ldots,30) formed the 15-sample
test set. FG, FE, and MB-1 used identical memberships, with no validation
set. The target was the unscaled correlation energy
$E_{\mathrm{FCI}}-E_{\mathrm{RHF}}$, expressed in Hartree.

FG geometric features and FE ascending eigenvalues of the selected
L\"owdin density block were standardized independently by feature column,
using training-set means and population standard deviations. Standardized
values were multiplied by $\pi/2$ and clipped to $[-\pi,\pi]$; columns with
standard deviation below $10^{-12}$ were encoded as zero. MB-1 density
and Fock diagonal entries were standardized separately using statistics
pooled over training geometries and selected orbitals, and encoded as
$(\pi/2)\tanh(a\widetilde P_i+b\widetilde F_i+c)$.

Each model used two shared-angle RY ansatz layers with CNOT rings and a
linear energy readout of the padded single-qubit Z expectations and their
first three raw moments. Non-bias parameters were initialized from
$\mathcal{N}(0,0.05^2)$; the readout bias was initialized to the mean training
target. Parameters were optimized for 500 epochs using Adam with learning
rate 0.02, default $(\beta_1,\beta_2)=(0.9,0.999)$ and $\epsilon=10^{-8}$,
and zero weight decay. The objective was the minibatch mean squared energy
error, with batch size eight and the final incomplete batch retained.
Training samples were reshuffled every epoch using a persistent seed-local
random generator, separate from the initialization generator.

The checkpoint with the lowest full-training-set MAE over completed epochs
was restored, retaining the earliest epoch on exact ties. Test prediction
metrics were then evaluated once per run; test MAE was not used for
checkpoint selection. Each molecule and model was trained independently
with matched seeds 0--31. Simulations used analytic PennyLane
\texttt{default.qubit} execution, the Torch interface and backpropagation,
with CPU float64 training parameters. Per-molecule test MAEs were summarized
by their mean and sample standard deviation across seeds, with standard
error $\mathrm{SD}/\sqrt{32}$. Overall results first averaged molecule MAEs
equally within each seed and then computed the same statistics across the
32 seed-level averages. MAEs were reported in mHa.
```

## 12. Conflict table

“Previous/old description” below identifies the inspected repository description or the user's stated intended change. It does not assert access to an unseen historical manuscript. MATCH means agreement with that named comparator; CHANGED means a demonstrable difference from the named description, not a reconstructed chronology.

| Topic | Current code | Previous/old description | Status |
|---|---|---|---|
| Train/validation/test vs train/test only | Default remains three-way manifest; explicit 15-15 is two-way [R:450–473] | User states current workflow changed to train/test only | CURRENT CODE CONFLICT |
| Validation-based checkpointing | Active default; training MAE only in 15-15 [T:183–233] | User requests removal of old validation-based behavior from current narrative | CURRENT CODE CONFLICT |
| Split rule | Odd one-based IDs train, even test in 15-15; arbitrary explicit groups otherwise [R:254–255] | D:65–70 and 144–149 describe the same two branches | MATCH |
| Scaler fitting | Training rows only; per-column FG/FE versus pooled MB-1 [C:380–391,740–762] | D:119–121,132–135 describe train-only fitting | MATCH |
| Test leakage: epoch selection | No test metric chooses epoch [T:183–233] | D:68–70 says test data do not enter checkpoint selection | MATCH |
| Test independence: all preprocessing | Full-scan H2O2 unwrap may involve test inputs; inactive on default data [R:322–329] | Any blanket interpretation of D:69–70 as total test-data independence is too broad | CURRENT CODE CONFLICT |
| Seed count | 32, IDs 0–31; smoke-only exception [R:287,477,489–495] | D:163 specifies seeds 0–31 | MATCH |
| Epochs | 500 default; override and smoke-only exception [R:454–465] | D:43–49,163 specifies 500 and bounded smoke-only mode | MATCH |
| Batch size | 8, including partial batch [R:288; T:45–52] | D:163 specifies 8 | MATCH |
| Learning rate | Adam 0.02 [R:288,382] | D:163 specifies 0.02 | MATCH |
| Initialization | Normal SD 0.05, training-mean bias, model-specific draw blocks [T:18–32; C:842–847] | D:165–166 says canonical initialization retained but supplies no exact old distribution/draw specification | NOT VERIFIABLE |
| Aggregation | Equal-molecule MAE per seed, then mean/sample SD/SE and median across 32 seeds [R:421–444; T:268–272] | D:182–189 describes these summary conventions | MATCH |
| MB-1 pair gates | No pair gates; alpha fit remains unused [C:466–472] | D:132–134 says zero-angle IsingZZ gates remain | CHANGED |
| H2S dimensions | 10 FE/MB qubits, 20 MB inputs, 16 FE / 19 MB parameters [C:124–128, layouts] | D:160 historical table says 6 FE/MB qubits, 12 inputs, 17 MB parameters; D:3–10 warns table is historical | CHANGED |
| H2S default input compatibility | Loader rejects saved six-index selection [R:164] | Current model requires ten selected AOs, default dataset still contains six | CURRENT CODE CONFLICT |
| Exact old manuscript split/counts | No supplied three-way manuscript manifest; no inferred split | Exact old manuscript IDs/counts were not provided | NOT VERIFIABLE |
