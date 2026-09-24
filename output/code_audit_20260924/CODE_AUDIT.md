# Audit of qml2.0/codes

Scope: current executable code, using the supplied table only as a checklist of computational stages. This is a code audit, not a comparison with that document. No code or datasets were changed. No production calculations, training, or repairs were run.

## Findings

**1. Default input cannot support the default nine-molecule run — confirmed failure.** `run1.py:485,488` selects `results/descriptor` and all nine molecules. Its H2S records retain six AOs, but [circuit.py:125–129](/Users/novaz/Desktop/qml2.0/codes/circuit.py:125) and `run1.ao_selection:140–172` require ten. Loading H2S/001 was reproduced and raises `valence_indices: saved selection does not match canonical valence AO labels` at line164. Thus even with a valid split selection, the default full run stops. The checked `descriptor-1s-only` H2S record passes. Also, descriptor production defaults to `results/descriptor-v2-methods-t` ([descriptor.py:569](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:569)), so producer and consumer defaults do not connect automatically. These are configuration/data-contract findings, not reasons to change the scientific AO selection.

**2. FE_prime is executable but unavailable in the plotting CLI — confirmed interface gap.** Current `run1.py:287,489` accepts FE_prime, and `raw_spectra:305–328` computes ascending eigenvalues of the selected Löwdin Fock block. However, `plot_run1_results.py:52,71` restricts `--runs` to FG, FE and MB-1. That plotting entry point cannot admit an FE_prime result. This affects output coverage, not the computed spectrum or trained energy.

**3. Normal descriptor generation is calculation-only, not scientific acceptance — validation gap.** `descriptor.calculate_record:149–157` proceeds from RHF.kernel through canonicalization into RMP2 without rejecting `mf.converged=False` or performing the initial-data stability checks. `lowdin_factors:51–53` applies square roots/inverse square roots without an eigenvalue threshold. `save_record:232–239` records successful return without scientific validation. The producer explicitly stores `validation_performed=False`. The downstream loader rejects unconverged RHF and validates overlap/transforms (`run1.py:191,198–216`), but standalone descriptor completion does not establish scientific validity. No failing-convergence sample was observed in the numerical checks; this is a source-established acceptance gap, not a claimed observed bad result.

**Important verified dataflow, not defects:** T is calculated and saved but not consumed by current Run 1; MB-1 contains no pair or triple gates. Its pair scale is nevertheless fitted and must be positive. FCI freezes core MOs, while descriptor RMP2 freezes none. AO selection happens after full-space Löwdin transformations. These are actual implementation choices; the audit does not propose changing them.

## Computational-stage audit

Source abbreviations: I=`initialdata.py`; D=`descriptor.py`; R=`run1.py`; C=`circuit.py`; T=`train.py`; P=`plot_run1_results.py`, all under `/Users/novaz/Desktop/qml2.0/codes`.

M denotes full AO/spatial MO count; v_i maps local retained AO i to a full source AO; n denotes the selected descriptor count, or the feature/wire count for geometric fingerprints. Energies are Hartree (Ha), geometry Å. Formulas use ordinary transpose for the implemented real arrays. “Library” marks behavior delegated outside repository code, not an independently audited internal algorithm.

| Layer | Exact current source | Verified computation, representation and dataflow |
|---|---|---|
| 1 Geometry | I `optimize_equilibrium`771–824; `StretchPath`194–261; `calculate_harmonic`431–579; `scan_grid`858–865 | Cartesian RHF optimization; k=Σt_ax H_abxy t_by, μ=Σm_a t_ax²; harmonic n=1 endpoints s_e±Δ, Δ=√(3ℏ/(μω)) with SI→Å conversion. Inclusive30-point linspace; saved IDs1..30. Raw internal stretch undergoes mass-weighted proper alignment. Output X(N_atom,3); descriptor consumes JSON coordinates. |
| 2 AO overlap | I `rhf_arrays`887–893; D `calculate_record`144–161 | Molecule geometry/basis/charge/spin and spherical AOs define S(M,M); `mf.get_ovlp()` delegates integrals to PySCF. Dimensionless overlap; full space retained. |
| 3 RHF reference | I `configured_rhf`607–613, `polish_orbitals`616–647, `solve_point`660–743; D139–159 | Initial-data RHF has tight convergence/internal-stability checks. D separately recomputes RHF using saved AO density as its initial guess. P=C diag(occupations) Cᵀ; F=h+J[P]−K[P]/2. Spin-summed P(M,M), F(M,M) Ha, C(M,M). SCF/integral internals library-bound. |
| 4 Core/space definitions | I `ACTIVE_SPACES`37–41; D `select_valence_aos`91–119 | FCI frozen MOs and excluded descriptor AOs are independent. MP2/RDM frozen=0. Descriptor exclusion is Z>2 and principal shell1, determined from AO/shell metadata; H2S retains10 AOs. |
| 5 MP2 amplitudes | D `calculate_record`156–169 | `RMP2(mf,frozen=0)`; t2 shape(n_occ,n_occ,n_virt,n_virt), dimensionless, all occupied/virtual spatial MOs. Restricted amplitude/energy construction delegated to PySCF; saved-array amplitude checks agree to roundoff. |
| 6 Used RDM construction | D `build_consistent_rdms`60–88 | Determinant Φ and χ from zero singles and restricted t2; w=vdot(χ,χ). For either RDM, R1=R[Φ+χ]−R[Φ]−R[χ], R2=R[χ]−wR[Φ], R=R0+R1+R2. Full MO 1-RDM(M,M),2-RDM(M,M,M,M). PySCF determinant/RDM APIs are boundaries. Raw MP2 RDMs and exactly normalized state are stored diagnostics. |
| 7 Tensor ordering | D `transform_rank4`56–57; `extract_descriptors`129–136 | Full rank-four arrays retain raw PySCF ordering. Only T extraction applies transpose(0,2,1,3). No explicit four-spin-block pipeline. Raw and permuted indices must remain distinct. |
| 8 MO→AO | D `calculate_record`171–175; `transform_rank4`56–57 | D_AO=C D_MO Cᵀ; G_abcd=Σ_pqrs C_ap C_bq C_cr C_ds G_pqrs. Full M-sized axes, no conjugation or AO restriction. |
| 9 AO restriction | D122–136,182–186; R `ao_selection`140–172 | Restrict P_L,F_L,Λ_L only after full-space transforms. D stores source-ascending v; R selects canonical atom/AO order. BeH2 model source indices=[5,1,2,3,4,6], not sorted. |
| 10 Disconnected reference | D `disconnected_rhf`47–48;176 | G0_pqrs=P_pq P_rs−0.5 P_ps P_rq, using full RHF spin-summed P. No summed p,q,r,s in these outer products. |
| 11 Reference subtraction | D176–181 | Λ=G_consistent−G0[P_RHF], full AO(M⁴). The conventional subtraction using correlated density is separately stored; it does not feed descriptor extraction. |
| 12 Extra diagnostic contraction | D160–210 | No separate all-ones fourth-axis-sum stage is executed. Stored raw RDMs, order components and alternate cumulant are diagnostics; their existence does not imply model use. |
| 13 FCI and target | I `frozen_core_fci`896–943; `calculate_point`946–990; R227–232 | Canonical RHF CASCI, active MOs[ncore,M); ncas=M−ncore, Ncas=N_e−2ncore. E_FCI=E_active+E_core including nuclear/core terms. y=E_FCI−E_RHF,input. CI shape(choose(ncas,Ncas/2),choose(ncas,Ncas/2)); active h1 and pair-packed h2 saved, not used by D. Solver library-bound. |
| 14 Overlap powers | D `lowdin_factors`51–53; R208–211 | Full S=U diag(s) Uᵀ; H=U diag(√s)Uᵀ, J=U diag(1/√s)Uᵀ. Producer no clipping/threshold; loader requires min(s)>1e−8 and matrix identities. |
| 15 One-body Löwdin | D182–184; R212–213 | P_L=H P_RHF H; F_L=J F_AO J, both full(M,M). No explicit averaging with transpose here. Correlated density does not supply P_L. |
| 16 Rank-four Löwdin | D56–57,182–184; R216 | Λ_L,abcd=Σ_pqrs H_ap H_bq H_cr H_ds Λ_AO,pqrs. All four full-space axes transformed before selection. |
| 17 T extraction | D `extract_descriptors`122–136 | T_ijk=Λ_L[v_i,v_k,v_j,v_k], **no sum over k**. T_full(n,n,n), selected triples(choose(n,3),). R constructs T=None; current launched models do not consume T. |
| 18 P/F features | D133–136; R218–241 | Saved P_mu/F_mu are selected diagonals(n); P_munu strict pairs(choose(n,2),). R validates source order, then rebuilds canonical pii/fii/pij and full selected fij. P dimensionless, F Ha. FE uses eigvalsh(pij); optional FE_prime uses eigvalsh(fij), ascending, classical spectral features. |
| 19 Population | R `load_population`244–282; `retained_positions`111–117 | Exactly IDs001..030 per molecule, no cutoff filtering. Reject incomplete/error records. Sort by(q_A,geometry_index); zero-based traversal positions. |
| 20 Split | R `validate_split_manifest`41–73;244–269;331–366 | Default explicit disjoint train/validation/test sets cover all30. Explicit15-15 uses odd IDs train/even IDs test; membership is independent of sort order. |
| 21 Diagonal scaling | C `compute_encoding_constants`345–392; `standardize_diagonal`394–399 | MB scalar mean/std pooled over N_train×n entries, separately P/F; ddof0. Standardize(x−μ)/σ, require positive finite σ; no clipping/epsilon floor. Training rows only. |
| 22 Pair/triple indexing | C `pair_addresses`241–244; `triple_addresses`245–248;249–262 | Local i<j row-major and i<j<k nested lexicographic; output lengths choose(n,2),choose(n,3). Pairs feed scale fitting. Triple helper unused in Run1. |
| 23 Pair/triple scales | C `_p95_abs`279–287;381–392 | Linear95th percentile of pooled absolute entries, zeros included; α=atanh(.9)/q95. β only MB-3 helper. Empty/nonpositive inputs reject, no flatten fallback. MB-1 fits α but does not use it in gates. |
| 24 Local encoding | C `CachedCircuitBank`461–466; `encode`755–763; `FingerprintCircuit`800–820 | MB θ_i=(π/2)tanh(a Pstd_i+b Fstd_i+c), shared trainable a,b,c. Fingerprints: per-column train mean/std; σ<1e−12 masked, safe σ=1; scaleπ/2, zero masks, then clip[−π,π]. RY half-angle convention delegated to PennyLane. |
| 25 Pair encoding | C467–473,610–640 | IsingZZ((π/2)tanh(αP_ij)) exists for helper models. MB-1 passes no pair input and queues no pair gates. Not an active Run1 stage. |
| 26 Triple encoding | C474–481 | MultiRZ((π/2)tanh(βT_ijk)) helper for i<j<k. No current Run1 method supplies triple inputs; runtime triple results not verified. |
| 27 HEA | C `shared_ansatz`416–424 | Each layer: same RY(phi_l) on all wires, then sequential CNOT q→(q+1) mod n, ascending q, wrap last. L=2, single encoding. n=2 executes both directed edges. |
| 28 Measurement | C482–483,818–820 | Ordered Pauli-Z expectation on every physical wire, z(n), dimensionless. Simulation/autodiff library-bound. Independent statevector forward predictions checked; direct runtime z residual not separately obtained. |
| 29 Readout features | C `pad_z_to_length`568–574; `invariant_moments`576–583;641–660 | m=max(8,n); zero-pad z on right; M_k=mean(z^k),k1..3 using **unpadded** z. f=[z_pad,M1,M2,M3], shape(m+3). No extra qubits. |
| 30 Prediction | C `CorrelationEnergyModel`655–681; `FingerprintModel`880–910 | y_hat=b+w·f, scalar Ha, no target standardization. Fingerprint n is feature width, not necessarily AO count. |
| 31 Initialization | T `initialize_parameters`18–32; C843–848 | Local seeded CPU Torch generator; MB full .05*randn vector, fingerprint separate phi/bias/weight draws; bias overwritten with training target mean. MB D=L+m+7, fingerprints D=L+m+4. |
| 32 Batches/loss | T `SampleOrder`34–43; `iter_batches`45–52; `mse_loss`70–73;183–233 | Separate persistent same-seed shuffle; fresh permutation each epoch; final partial batch retained. L=mean((prediction−target)²) over actual batch, Ha². |
| 33 Adam | T54–91; C850–873 | Zero grads to None, backward, model-specific gradient binding, Adam step. MB one vector group; fingerprints shared-storage blocks. Installed Torch defaults and two diagnostic steps checked; complete training trajectory not replayed. |
| 34 Checkpoint | T `BestCheckpoint`132–173;211–233; R402–453 | Strict improvement; earliest tie. Validation MAE for manifest, training MAE for15-15. Restore theta only, not optimizer; final test evaluation after restoration. |
| 35 Metrics | T `error_metrics`244–254; `evaluate`256–266 | e=prediction−target; MAE=mean(abs(e)); RMSE=√mean(e²); max(abs(e)); max(e)−min(e). Scalar Ha and ×1000 mHa after reduction; each split separately. |
| 36 Seed summaries | T `compute_statistics`268–272; R `summarize`455–480; P131–149 | Require32 seeds in full launcher. Mean/sampleSD(ddof1)/count/median/SE. Overall: equal molecule mean within seed, then seed statistics. Core singleton SD/SE NaN; plot helper SD0,SE NaN. |
| 37 Plot assembly | P `validate_seed_runs`152–168; `load_prediction`346–375;377–397 | Exact configured seed coverage and equality of sorted(geometry ID,split,bond_length_A,target). Bars test mean±SE; curves per-geometry signed-error mean±sampleSD. FE_prime CLI gap noted above. Initial-data plot is a separate branch: relative1000(E−E_RHF(re)) or absolute Ha; curve31 points including equilibrium, CSV30 scans. |

## Cross-stage checks

- The initial RHF/CASCI reference and descriptor RHF recomputation are separate calculations; the loader checks their RHF energies within1e−10 Ha. The target remains the input FCI−RHF difference.
- The determinant RDM is truncated through second order; `ci_normalized_eta1` and `ci_second_order_normalization` are stored but do not replace the implemented RDM algebra.
- Full-space matrix powers precede restriction. Stored source-local indices, canonical full-AO indices and circuit wire positions are different index systems.
- FG has explicit molecule-dependent distance/reciprocal/angle/torsion order. H2O2 torsion can be unwrapped across the complete sorted scan before train-only fitting; this is a preprocessing dependency on the full scan, not label usage.
- Current FE_prime uses the Fock spectrum in Ha before per-column scaling. FE uses the density spectrum. Both become dimensionless rotation inputs; neither changes the target definition.
- Historical fixed-angle fields in the molecule registry are unused by the current geometry generator. Historical UHF/UMP2-named sample fields receive None; names do not establish computations.

## Numerical verification already completed

| Check | Exact shapes / coverage | Maximum absolute residual |
|---|---|---:|
| Independent aligned geometry reconstruction | 27 samples across9 molecules; (2,3)/(3,3)/(4,3) | 5.10702591327572e−15 Å |
| Harmonic grid reconstruction | 9×(30,) | 2.220446049250313e−16 Å |
| FCI−RHF target arithmetic | LiH/H2S/H2O2 IDs001/015/030, scalars | 0 Ha |
| Independent LiH active Hamiltonian | H(25,25),CI(5,5);3 samples | Rayleigh8.881784197001252e−16 Ha; eigenvector6.661338147750939e−16 Ha |
| Full rank-four transforms | LiH/H2S/H2O2 ID001;6⁴/11⁴/12⁴ arrays | 1.5987211554602254e−14 |
| Literal T indexing against saved current-selection descriptors | (5,5,5)/(10,10,10)/(10,10,10); selected(10)/(120)/(120) | 0 |
| Independent NumPy statevector→saved prediction | FG/FE/MB-1, seeds0/31, CO IDs001/016/030; states(4)/(256)/(256), readout(11) | 3.3306690738754696e−16 Ha |
| Saved error/metric recomputation | 96 runs; residuals(2880,), metrics(192,8) | 0 |
| Earliest training-MAE checkpoint minimum | 96 histories×500 epochs; outputs(96,) | 0 |
| Independent seed summary | (6,5) | 4.440892098500626e−16 |

These samples support the checked arithmetic, not universal correctness. PySCF's compiled integral/SCF/determinant engines, PennyLane derivatives, and a full training replay were not independently audited. The saved learning run is CO-only and15-15; validation-selected training, multi-molecule aggregation at runtime, FE_prime training and MB-2/3 runtime remain **NOT_VERIFIABLE from these numerical samples**. Their described code paths were inspected.

Two files, `run1.py` and `circuit.py`, changed externally during this session to add FE_prime. Final source references above use the refreshed files; earlier numerical runtime checks are identified as pre-addition evidence, not a blanket certification of the new branch. Independent saved-array/statevector arithmetic remains valid for the named saved samples. `evidence.json` records earlier and final source hashes and only code/numerical evidence; no document-comparison results are included.
