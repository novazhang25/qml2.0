# Run2 Lambda_ijij end-to-end audit

Audit date: 2026-09-25. All indices in this report are **zero-based**. Scope: the current CO dataset and the production Run2 experiment `20260925_113307_117058`.

**Final answer: YES. Run2 MB2 encodes the intended physicists-order, spin-summed RHF-reference quantity `Lambda_Methods[i,j,i,j]`.** Its actual input is the full-AO saved raw tensor entry `Lambda_L[I,I,J,J]`, with `(I,J)` obtained from the same AO mapping used by MB1. Every one of the 30 geometries and 28 unique qubit pairs was checked (840 inputs).

| Audit component | Verdict | Basis |
|---|---|---|
| Ordering | **PASS** | The generating operator definition, internal PySCF normal ordering, independent fermionic contractions, and saved tensors agree. The raw-to-Methods relationship is established physically, not assumed from a transpose. |
| RHF-reference subtraction | **PASS** | The direct term and negative one-half exchange term follow from the four spin channels. The saved density is spin-summed RHF; independent reconstruction agrees with all inputs. |
| Full-AO Lowdin transformation and AO mapping | **PASS** | All four full-AO axes are transformed before selection; 10 full AOs map to the same 8 ordered qubits as MB1. |
| Pair extraction and MB2 consumption | **PASS** | Raw `[I,I,J,J]` equals converted Methods `[I,J,I,J]`; 28 distinct pairs occur once each and reach the actual angle calculation unchanged. |

No descriptor mismatch was found. No production correction is proposed or applied.

## 1. Actual experiment and audit boundaries

The experiment's [configuration:68](/Users/novaz/Desktop/qml2.0/results/run2/20260925_113307_117058/configuration.json:68) points to `/Users/novaz/Desktop/qml2.0/results/descriptor`, with CO geometries 001–030, odd IDs training and even IDs testing, and no validation subset (lines 68–73). The [saved population:35](/Users/novaz/Desktop/qml2.0/results/run2/20260925_113307_117058/population.json:35) records full-AO selection `[1,2,3,4,6,7,8,9]`, full dimension 10, and the actual NPZ source paths (first record lines 35–50). Every population entry was matched against the loader's source path and AO mapping. The separate smoke run uses this same CO dataset; no additional geometry is introduced by it.

All 30 NPZs contain `mp2_t2`, MO coefficients/occupancies, `ci_reference`, `ci_first_order`, MO/AO Gamma tensors, RHF density, overlap, and full-AO `Lambda_L`. Their schema is `descriptor-v1-calculation-only`; their saved PySCF version is 2.14.0. None of the required physical source tensors is missing. Metadata and previous audit summaries were treated as claims to check.

Execution used `/Users/novaz/Desktop/encoding_qml/.venv/bin/python -B`, with PySCF 2.14.0, NumPy 2.5.1, Torch 2.13.0, and PennyLane 0.45.1. The inspected PySCF sources in this project's `.venv` are byte-identical to the executed environment's five source files listed below.

Only in-memory determinant/RDM reconstruction, loader calls, circuit-tape construction, and fixed-checkpoint forward evaluation were performed. There was no SCF/MP2 energy solve, FCI diagonalization, optimizer step, training, descriptor write, or training-setting edit. SHA-256 checks before and after covered **501 existing code, input, and result files**; every digest was unchanged. The new durable workspace artifact is this report. The executed audit harness and numerical scratch output were placed under `/tmp`.

Historical launch-time byte identity is **NOT_VERIFIABLE** from the saved run metadata: it does not preserve immutable source/descriptor hashes. This is distinct from the four current-path verdicts above. As a numerical connection to the existing experiment, all **960 saved MB2 predictions** (32 checkpoints × 30 geometries) were reproduced using the current loader, saved preprocessing, and unchanged checkpoints; maximum absolute difference was **5.5511151231257827e-17 Hartree**, at **CO/016, seed 26**. No original training-time runtime trace is claimed.

## 2. Physical convention established from the generating implementation

### 2.1 Which Gamma actually supplies Lambda

[descriptor.calculate_record:139](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:139) loads the RHF starting density (147–148), constructs `RHF` (149), canonicalizes its MOs (155), and obtains all-electron `RMP2(mf, frozen=0)` amplitudes (156–157). `P = mf.make_rdm1()` is obtained from that **RHF object** at line 158.

The conventional library `pt.make_rdm1/2` outputs are saved under `_pyscf` names at lines 166–169. They are **not** the source selected for the formal descriptor. Line 170 calls [descriptor.build_consistent_rdms:60](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:60). This constructor:

1. Converts a unit RHF reference and first-order doubles `t2` to determinant vectors using `cisd.amplitudes_to_cisdvec` and `cisd.to_fcivec(..., frozen=0)` (69–75).
2. Calls `direct_spin1.make_rdm12(..., reorder=True)` on `Phi`, `chi`, and `Phi+chi` (77–79).
3. Forms `R0 + R1 + R_chichi - w*R0`, with `w=<chi|chi>` (76, 80–84). Algebraically this is `R(Phi+chi) - w*R(Phi)`. It is the expectation expansion through second order of `(Phi+eta*chi)/sqrt(1+eta^2*w)`, **not** exact division by `1+w` at eta=1.

Thus `Gamma_MP2_MO_consistent` is the implemented perturbative Gamma. The constructor was rerun from saved `mp2_t2` without any solver for all 30 geometries: `ci_reference`, `ci_first_order`, `P_MP2_MO_consistent`, and `Gamma_MP2_MO_consistent` matched the saved fields exactly (maximum residual 0).

### 2.2 Operator definition, independent of repository convention strings

The actual called function [PySCF fci/direct_spin1.py:365](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/direct_spin1.py:365) defines the spin-traced 2-RDM explicitly in its four alpha/beta channels (370–373), invokes `FCIrdm12kern_sf` (382–383), and normal-orders it when `reorder=True` (384–385):

```text
G_Gamma[p,q,r,s] = sum_(sigma,tau) <a†_(p,sigma) a†_(r,tau) a_(s,tau) a_(q,sigma)>
P[p,q]          = sum_sigma <a†_(q,sigma) a_(p,sigma)>
```

This definition is also documented by the [official PySCF generating-function source](https://pyscf.org/_modules/pyscf/fci/direct_spin1.html#make_rdm12); the installed, matching version is the primary evidence here.

The lower-level implementation matters: [PySCF fci/rdm.py:118](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/rdm.py:118) documents the intermediate density-operator product; its normal-order correction is implemented at [PySCF fci/rdm.py:35](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/rdm.py:35). Starting from `<a†_p a_q a†_r a_s>`, anticommutation gives the delta contribution plus `<a†_p a†_r a_s a_q>`. The code subtracts that delta contribution at line 40. This step is **not** the Methods middle-axis permutation.

Define the intended physicists-order tensor by grouping creation indices first:

```text
Gamma_Methods[a,b,c,d] = sum_(sigma,tau) <a†_(a,sigma) a†_(b,tau) a_(d,tau) a_(c,sigma)>
Lambda_Methods       = Gamma_Methods - Gamma_RHF_reference_Methods
```

Matching the actual operators establishes, for both Gamma and its reference-subtracted Lambda:

```text
G_Gamma[p,q,r,s]  = Gamma_Methods[p,r,q,s]
G_Lambda[p,q,r,s] = Lambda_Methods[p,r,q,s]
Lambda_Methods[I,J,I,J] = G_Lambda[I,I,J,J]
```

Independent bitstring ladder-operator contractions of the saved CO wavefunctions confirm the **full** raw Gamma tensor, not just diagonal identities; maximum MO residual is `1.9539925233402755e-14` at CO/021, raw MO index `(4,4,5,5)`. The independent Slater fixture in section 6 also matches the called PySCF constructor with nonzero off-diagonal density. These are independent physical checks beyond a transpose identity.

### 2.3 Transpose and permutation ledger

| Stage | Exact operation | Meaning and evidence |
|---|---|---|
| Restricted doubles → CI | `t2.transpose(0,2,1,3)`; `t2 - t2.transpose(1,0,2,3)` | Opposite-spin amplitude reshaping and same-spin antisymmetrization, [PySCF ci/cisd.py:337](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/ci/cisd.py:337) and line 342. These are amplitude axes `(i,j,a,b)`, not saved RDM axes. No frozen-orbital reorder occurs because `frozen=0` returns at 349–350. |
| C RDM driver | `_transpose_jikl`, equivalent to rank-four `(1,0,2,3)` | Corrects the creation/destruction order inherited from the bra contraction; definition [PySCF lib/mcscf/fci_rdm.c:181](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/lib/mcscf/fci_rdm.c:181), explanation 198–207, actual `symm=1` call at 270. The driver first fills symmetric matrix halves (258–269). |
| Python RDM return | `rdm1.T`; rank-four returned without a Python transpose | [PySCF fci/rdm.py:151](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/rdm.py:151) establishes the one-body orientation. |
| Normal-order correction | `rdm2[:,k,k,:] -= rdm1.T` | Delta correction, not an axis conversion: [PySCF fci/rdm.py:39](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/rdm.py:39). |
| Particle-exchange averaging | Average with flattened matrix transpose, equivalent to `(2,3,0,1)` | [PySCF fci/rdm.py:42](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/rdm.py:42)–45. This symmetry average does not convert to Methods ordering. |
| MO → full AO | `C @ P @ C.T`; four separate factors `C` for Gamma | [descriptor.calculate_record:171](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:171), 171–175, and `transform_rank4` 56–57. Rank-four axes retain their identities. All audited arrays are real. |
| Lowdin factors | `(U*sqrt(eigenvalues)) @ U.T`, corresponding inverse | [descriptor.lowdin_factors:51](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:51). Matrix reconstruction only. |
| Full AO → full Lowdin | `S_half @ P @ S_half`; four `S_half` factors for Lambda | [descriptor.calculate_record:182](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:182), 182–184. No rank-four permutation. |
| Descriptor T branch | Local selected `L.transpose(0,2,1,3)` | [descriptor.extract_descriptors:122](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:122), 126–132. It creates the local Methods view for T, and does not modify `Lambda_L`. Run2 does not consume T. This current branch is not used as proof of the original saved tensor convention. |
| Run1 validation | Matrix `.T` comparisons | [run1.load_record:207](/Users/novaz/Desktop/qml2.0/codes/run1.py:207), 207–208, are symmetry checks only; they do not replace any tensor. |
| Run2 load → pair extraction → angle | **No transpose** | [run2.extract_lambda_pairs:70](/Users/novaz/Desktop/qml2.0/codes/run2.py:70), 70–76, and [PairEnergyModel.pair_quantum_arguments:130](/Users/novaz/Desktop/qml2.0/codes/run2_model.py:130), 130–151. Uses raw `[I,I,J,J]` directly. |
| Explicit Methods view, audit only | `raw.transpose(0,2,1,3)[I,J,I,J]` | Compared exactly with the raw selection and live MB2 input for all 840 pairs. No such additional conversion is needed or performed on the production path. |

## 3. RHF-reference subtraction derived from spin summation

Let `d[p,q]=<a†_(q,alpha) a_(p,alpha)>=<a†_(q,beta) a_(p,beta)>` for a closed-shell determinant; the spin-summed density is `P=2d`. For a fixed spin pair, evaluating the normal-ordered four-operator expectation gives

```text
<a†_(p,sigma) a†_(r,tau) a_(s,tau) a_(q,sigma)>
  = d[q,p] d[s,r] - delta_(sigma,tau) d[s,p] d[q,r].
```

The minus sign is fermionic exchange. Summing the four `(sigma,tau)` channels yields four direct products but only two same-spin exchange products:

```text
G_RHF[p,q,r,s] = 4 d[q,p] d[s,r] - 2 d[s,p] d[q,r]
                = P[q,p] P[s,r] - (1/2) P[s,p] P[q,r].
```

The audited real RHF density is symmetric, so this equals the actual production expression `P[p,q]*P[r,s] - 0.5*P[p,s]*P[r,q]` at [descriptor.disconnected_rhf:47](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:47). The real-symmetric condition is explicit; no unrestricted-spin or complex-density generalization is assumed.

`calculate_record` lines 175–181 form `Gamma - Gamma0(P_RHF)` and save it as both `Lambda_HFref_AO` and `Lambda_AO`. Gamma itself is not Lambda. Applying the same full linear AO transformation to all four axes carries both direct/exchange terms into the Lowdin basis. Hence the required pair identity is

```text
Lambda_Methods[I,J,I,J]
 = Gamma_MP2_Methods[I,J,I,J]
   - P_RHF_L[I,I] * P_RHF_L[J,J]
   + 0.5 * P_RHF_L[I,J] * P_RHF_L[J,I].
```

The plus sign on exchange in Lambda follows from subtracting the reference's negative exchange term. The factor one-half follows from the spin count; it is not a fitting coefficient.

The density identity was also verified numerically for every geometry. [PySCF scf/hf.py:855](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/scf/hf.py:855) constructs `C_occ diag(mo_occ) C_occ†` (866–868), and [PySCF scf/hf.py:1167](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/scf/hf.py:1167) assigns occupancy 2 (1172). Every CO record has occupancies `[2,2,2,2,2,2,2,0,0,0]`, 14 electrons, seven occupied and three virtual MOs, and zero MP2/RDM frozen orbitals. Reconstructed `P_AO` matches exactly; `Tr(P_AO S)` differs from 14 by at most `2.3092638912203256e-14`; `P_AO S P_AO - 2 P_AO` has maximum residual `1.8207657603852567e-14`. A spin-block density would have trace 7. The target FCI's saved frozen-core count 2 does not freeze MOs in this descriptor construction.

The **RHF-reference cumulant** is distinct from the separately stored diagnostic `Lambda_conventional_AO = Gamma - Gamma0(P_MP2_AO_consistent)` ([descriptor.calculate_record:180](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:180), 180–181). Run2 checks the RHF-reference alias and consumes `Lambda_L`; it never substitutes the correlated-density version. Replacing RHF P by MP2 D changes a checked pair by as much as **0.08170180176565674**, at CO/030, qubits `(2,6)`: actual RHF-reference value `0.09130355000905473`, correlated-reference value `0.00960174824339799`. The two definitions are numerically distinguishable in this dataset.

## 4. Full-AO transformation, saved field, loader, and actual angle input

The explicit rank-four transform at [descriptor.transform_rank4:56](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:56) is

```text
out[a,b,c,d] = sum_(p,q,r,s) A[a,p] A[b,q] A[c,r] A[d,s] tensor[p,q,r,s].
```

`calculate_record` computes full-AO `Lambda_L` at line 184, only then selects valence AO labels/indices at 185 and extracts descriptors at 186. [descriptor.save_record:221](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:221) saves all arrays unchanged into NPZ (235–238), including this full `(10,10,10,10)` raw `Lambda_L`. It does not save a Methods-converted Lambda field in its place.

[run1.load_record:175](/Users/novaz/Desktop/qml2.0/codes/run1.py:175) opens the actual NPZ and validates full dimensions and real finite entries (177–203), density/metric transforms (205–214), the formal Lambda alias (216), and the full-AO rank-four transformation (217), **before** AO selection (218). It obtains MB1 P/F from full-AO Lowdin tensors using `selected` at 226–227. The full positive square root was recomputed from `S_AO` independently in this audit; it matches the saved `S_half` exactly.

[run1.ao_selection:141](/Users/novaz/Desktop/qml2.0/codes/run1.py:141) maps canonical atom occurrences and labels to **source full-AO indices** (146–159), validates the labels and valence/core partition (160–171), and returns `selected`. CO's canonical contract is defined at [circuit.MOLECULES:119](/Users/novaz/Desktop/qml2.0/codes/circuit.py:119)–120:

| Qubit/local valence index | Full-AO index | AO label |
|---|---|---|
| 0 | 1 | C 2s |
| 1 | 2 | C 2px |
| 2 | 3 | C 2py |
| 3 | 4 | C 2pz |
| 4 | 6 | O 2s |
| 5 | 7 | O 2px |
| 6 | 8 | O 2py |
| 7 | 9 | O 2pz |

Full-AO indices 0 and 5 (C/O 1s) are excluded only after transformation. If `V = raw[np.ix_(selected,selected,selected,selected)]` is already selected, the correct raw access is `V[i,i,j,j]`, using local 0–7 indices; for an already converted selected Methods tensor it is `V_Methods[i,j,i,j]`. Applying full-AO indices to that 8-dimensional tensor would be a different, erroneous operation. The production path uses the full 10-dimensional tensor and the full-AO map exactly once.

[run2.load_population:79](/Users/novaz/Desktop/qml2.0/codes/run2.py:79) calls Run1's loader (84–85), reopens each `sample.source_file` (90), passes `source_selected_ao_indices` to the extractor (91–92), and stores the returned values in `PairSample.lambda_pairs` (93). The extractor's actual line is [run2.extract_lambda_pairs:73](/Users/novaz/Desktop/qml2.0/codes/run2.py:73):

```python
values = raw['Lambda_L'][i, i, j, j]
```

Here `i,j` are the mapped full-AO arrays created at 71–72, not qubit indices. Its convention strings and disconnected-product check at 53–69 alone would not prove physical ordering; the source/operator and determinant checks above provide the independent evidence.

[circuit.pair_addresses:243](/Users/novaz/Desktop/qml2.0/codes/circuit.py:243)–245 enumerates `(i,j)` lexicographically with `i<j`. [PairSample.__post_init__:34](/Users/novaz/Desktop/qml2.0/codes/run2_model.py:34) enforces `K*(K-1)/2` entries. [run2.prepare_runs:97](/Users/novaz/Desktop/qml2.0/codes/run2.py:97) preserves these samples in the MB2 populations (105–121); only MB2_prime removes Lambda (109–112).

[PairEnergyModel.pair_quantum_arguments:130](/Users/novaz/Desktop/qml2.0/codes/run2_model.py:130) takes that exact vector, validates it (142–148), and makes a float64 Torch tensor without normalization or reordering (149). The QNode invokes `pair_encoder` at [PairEnergyModel circuit:106](/Users/novaz/Desktop/qml2.0/codes/run2_model.py:106)–109; the actual consumer is [pair_encoder:75](/Users/novaz/Desktop/qml2.0/codes/run2_model.py:75)–85:

```text
argument = a2 * p_matrix[mu,nu] + b2 * lambda_pairs[position]
gamma    = (pi/2) * tanh(argument)
IsingZZ(gamma, wires=[mu,nu])
```

For every geometry the audit traced Python execution immediately before line 84, captured the actual Lambda operand and argument, and checked the queued IsingZZ angle and wires. All 28 pairs occurred once, in the expected order; all 840 captured Lambda operands matched the loader vector exactly. A separate in-memory coefficient probe `a2=0, b2=1` confirmed the angles respond directly to Lambda. Saved coefficients/settings were not edited. The real saved seed-0 coefficients were used for the main angle check, and all 32 saved checkpoints for prediction replay. This tests the live consumer, not an unused intermediate.

## 5. Numerical evidence and one concrete pair

All comparisons use float64 maximum absolute residuals, **zero relative tolerance**. General independent tensor/pair checks use `1e-10`; exact selections and copies use `0`; the Slater fixture uses `1e-12`; gate-angle checks use `1e-14`. Run1's production loader is looser (`1e-8` at [run1.check_record_array:134](/Users/novaz/Desktop/qml2.0/codes/run1.py:134)); the audit does not rely on that threshold alone. Every comparison passed. A zero maximum means all entries tie; the first entry is shown in the detailed table.

| Check | Maximum absolute residual | Worst geometry / indices | Tolerance |
|---|---:|---|---:|
| Rebuild actual Gamma constructor from saved t2 | 0 | CO/001; (0, 0, 0, 0) | 1e-10 |
| Independent operators vs full saved MO Gamma | 1.9539925233402755e-14 | CO/021; (4, 4, 5, 5) | 1e-10 |
| Independent operators vs full saved AO Gamma | 1.3322676295501878e-14 | CO/026; (0, 0, 6, 6) | 1e-10 |
| Independent RHF determinant vs saved Gamma0_AO | 2.6645352591003757e-15 | CO/027; (0, 0, 5, 5) | 1e-10 |
| Independent Gamma minus independent reference vs Lambda_AO | 1.3322676295501878e-14 | CO/026; (0, 0, 6, 6) | 1e-10 |
| Independent full Lowdin Lambda vs saved Lambda_L | 1.2603145523731873e-14 | CO/013; (0, 0, 0, 0) | 1e-10 |
| Saved AO Lambda transformed independently vs saved Lambda_L | 0 | CO/001; (0, 0, 0, 0) | 1e-10 |
| RHF occupied-orbital P vs saved P_AO | 0 | CO/001; (0, 0) | 1e-10 |
| RHF trace minus 14 | 2.3092638912203256e-14 | CO/001 | 1e-10 |
| RHF PSP minus 2P | 1.8207657603852567e-14 | CO/001; (6, 6) | 1e-10 |
| MB1 matrix vs mapped full Lowdin P | 0 | CO/001; (0, 0) | 1e-10 |
| Saved Gamma → independent pair subtraction vs MB2 | 1.4918621893400541e-15 | CO/016; qubits (0, 6) | 1e-10 |
| Independent operators → pair subtraction vs MB2 | 5.8147930914742574e-15 | CO/014; qubits (0, 6) | 1e-10 |
| Raw [I,I,J,J] vs MB2 tensor | 0 | CO/001; qubits (0, 1) | 0 |
| Converted Methods [I,J,I,J] vs MB2 tensor | 0 | CO/001; qubits (0, 1) | 0 |
| Selected tensor with local indices vs MB2 tensor | 0 | CO/001; qubits (0, 1) | 0 |
| Actual line-84 Lambda operand vs loaded vector | 0 | CO/001; qubits (0, 1) | 0 |
| Actual pre-tanh argument vs independent formula | 0 | CO/001; qubits (0, 1) | 1e-14 |
| Actual IsingZZ angle vs independent formula | 2.2204460492503131e-16 | CO/015; qubits (3, 7) | 1e-14 |
| Lambda-only coefficient probe | 5.5511151231257827e-17 | CO/005; qubits (5, 6) | 1e-14 |
| Independent Slater determinant RHF residual | 4.4408920985006262e-16 | fixture; (0, 0, 1, 1) | 1e-12 |
| Existing checkpoint predictions vs saved results (Hartree) | 5.5511151231257827e-17 | CO/016/seed_26 | 1e-10 |

The worst saved-Gamma pair reconstruction is **CO/016, qubits (0,6), full AOs (1,8)**: `1.4918621893400541e-15`. The worst fully independent operator pair reconstruction is **CO/014, qubits (0,6), full AOs (1,8)**: `5.814793091474257e-15`.

The following concrete pair has substantial off-diagonal density and distinguishes the two raw accesses. It is **CO/030** (test), bond length **1.2228631538780776 Å**, qubits **(2,6)** = **(C 2py, O 2py)**, full-AO indices **(3,8)**. It is pair position 16 in zero-based lexicographic enumeration.

| Quantity | Value / address |
|---|---|
| Raw saved Lambda address | `Lambda_L[3,3,8,8]` |
| Full Methods Lambda address | `Lambda_Methods[3,8,3,8]` |
| Selected Methods Lambda address | `Lambda_Methods_valence[2,6,2,6]` |
| Gamma in full Lowdin basis | `Gamma_MP2_Methods[3,8,3,8] = G_Gamma_L[3,3,8,8] = 0.4918059567795105` |
| `P_RHF_L[3,3]` | `0.5539112347759835` |
| `P_RHF_L[8,8]` | `1.4460887652240153` |
| `P_RHF_L[3,8]` | `0.8949887225775036` |
| `P_RHF_L[8,3]` | `0.8949887225775035` |
| Direct term, subtracted | `P[3,3]*P[8,8] = 0.8010048135409116` |
| Exchange magnitude, added in Lambda | `0.5*P[3,8]*P[8,3] = 0.4005024067704558` |
| Reconstructed Lambda | `0.4918059567795105 - 0.8010048135409116 + 0.4005024067704558 = 0.09130355000905471` |
| Actual Run2 Lambda input | `0.09130355000905473` |
| Absolute residual | `2.7755575615628914e-17` |
| Incorrect raw access | `Lambda_L[3,8,3,8] = -0.1751308207472711` |
| Error from that incorrect access | `0.2664343707563258` |

Gamma above was transformed directly from saved `Gamma_MP2_AO_consistent`; it was not recovered by adding a reference back to saved Lambda. Raw `[I,J,I,J]` is Methods `[I,I,J,J]`, a different component. The incorrect-index discrepancy shown here is the largest across all 840 pairs. The correct production line 73 requires **no change**.

### Coverage of all Run2 geometries

Each row covers every one of its 28 pairs; “worst pair” refers to the saved-Gamma reconstruction column. Both residual columns compare against the actual tensor supplied to MB2's angle calculation.

| Geometry | Split | Bond length (Å) | Pairs | Saved-Gamma max residual | Independent-operator max residual | Worst saved-Gamma qubit pair |
|---|---|---:|---:|---:|---:|---|
| CO/001 | train | 1.068098197421302 | 28 | 1.193489751e-15 | 3.413935801e-15 | (4, 7) |
| CO/002 | test | 1.073434920057743 | 28 | 8.465450563e-16 | 1.734723476e-15 | (0, 7) |
| CO/003 | train | 1.078771642694183 | 28 | 7.494005416e-16 | 3.711440877e-15 | (5, 6) |
| CO/004 | test | 1.084108365330624 | 28 | 8.049116929e-16 | 2.806782584e-15 | (0, 6) |
| CO/005 | train | 1.089445087967065 | 28 | 6.852157730e-16 | 3.743533261e-15 | (4, 7) |
| CO/006 | test | 1.094781810603505 | 28 | 1.236857838e-15 | 3.139849492e-15 | (0, 4) |
| CO/007 | train | 1.100118533239945 | 28 | 1.009609063e-15 | 3.233524559e-15 | (4, 5) |
| CO/008 | test | 1.105455255876386 | 28 | 6.938893904e-16 | 4.690692279e-15 | (0, 6) |
| CO/009 | train | 1.110791978512827 | 28 | 5.967448757e-16 | 3.705369345e-15 | (0, 6) |
| CO/010 | test | 1.116128701149267 | 28 | 1.113692472e-15 | 2.026157020e-15 | (0, 7) |
| CO/011 | train | 1.121465423785708 | 28 | 7.025630078e-16 | 3.101685575e-15 | (0, 7) |
| CO/012 | test | 1.126802146422148 | 28 | 5.412337245e-16 | 2.123301535e-15 | (0, 6) |
| CO/013 | train | 1.132138869058589 | 28 | 1.054711873e-15 | 3.164135620e-15 | (0, 5) |
| CO/014 | test | 1.137475591695029 | 28 | 5.273559367e-16 | 5.814793091e-15 | (1, 4) |
| CO/015 | train | 1.14281231433147 | 28 | 4.996003611e-16 | 3.157196726e-15 | (0, 6) |
| CO/016 | test | 1.14814903696791 | 28 | 1.491862189e-15 | 2.380040609e-15 | (0, 6) |
| CO/017 | train | 1.153485759604351 | 28 | 6.869504965e-16 | 1.679212325e-15 | (0, 5) |
| CO/018 | test | 1.158822482240791 | 28 | 5.828670879e-16 | 3.580469254e-15 | (0, 5) |
| CO/019 | train | 1.164159204877232 | 28 | 1.370431546e-15 | 3.405262183e-15 | (0, 7) |
| CO/020 | test | 1.169495927513672 | 28 | 6.869504965e-16 | 2.749536709e-15 | (0, 7) |
| CO/021 | train | 1.174832650150113 | 28 | 9.254749744e-16 | 4.912736884e-15 | (0, 4) |
| CO/022 | test | 1.180169372786553 | 28 | 8.049116929e-16 | 5.136516212e-15 | (0, 7) |
| CO/023 | train | 1.185506095422994 | 28 | 4.857225733e-16 | 2.974183400e-15 | (0, 5) |
| CO/024 | test | 1.190842818059435 | 28 | 6.036837696e-16 | 2.872702076e-15 | (0, 5) |
| CO/025 | train | 1.196179540695875 | 28 | 9.645062526e-16 | 4.185887748e-15 | (0, 5) |
| CO/026 | test | 1.201516263332316 | 28 | 8.673617380e-16 | 5.784435431e-15 | (0, 6) |
| CO/027 | train | 1.206852985968756 | 28 | 4.787836794e-16 | 5.111362722e-15 | (0, 6) |
| CO/028 | test | 1.212189708605197 | 28 | 8.881784197e-16 | 3.951700078e-15 | (4, 7) |
| CO/029 | train | 1.217526431241637 | 28 | 1.047772979e-15 | 5.266620473e-15 | (0, 5) |
| CO/030 | test | 1.222863153878078 | 28 | 7.147060721e-16 | 1.804112415e-15 | (4, 6) |

## 6. Independent validation construction

### 6.1 CO wavefunctions and independent tensor algebra

The independent path does not call the production subtraction helper or PySCF's RDM routines to build its CO Gamma/reference tensors. Determinant bitstrings were enumerated independently in ascending integer order for seven alpha and seven beta electrons in ten spatial MOs. Alpha modes occupy bits 0–9 and beta modes 10–19; all saved coefficients, including arbitrarily small nonzero coefficients, were retained.

For each occupied mode `m`, the annihilator removes the bit and multiplies the amplitude by `(-1)**popcount(bits below m)`. For every ordered mode pair, two successive annihilations build `a_b a_a |psi>`. Their Gram matrix gives

```text
H[(p,sigma;r,tau),(q,sigma;s,tau)]
 = <a_(r,tau) a_(p,sigma) psi | a_(s,tau) a_(q,sigma) psi>
 = <psi | a†_(p,sigma) a†_(r,tau) a_(s,tau) a_(q,sigma) | psi>.
```

Summing all four spin channels constructs raw Gamma from the operator definition. One-annihilation Gram matrices construct P. The independent perturbative tensor is `G(Phi+chi)-<chi|chi>*G(Phi)`; its equality to the saved tensor was tested over every component for every geometry. The same independent determinant calculation supplies the RHF reference `G(Phi)`.

An independently implemented transform contracts one tensor axis at a time using `tensordot`, restoring that axis position with `moveaxis`. It does not use `descriptor.transform_rank4`. Two pair checks were made:

1. Transform saved AO Gamma with the independently recomputed full `S_half`; subtract the independently derived direct/exchange terms from RHF P; compare every pair with the live MB2 Lambda tensor.
2. Construct Gamma independently from the saved determinant amplitudes; transform it with `S_half @ C`; subtract the RHF terms; compare every pair again.

The full independently reference-subtracted AO and Lowdin tensors were also compared, so validation is not limited to pair entries. Rebuilding the production constructor from saved t2 is a separate provenance consistency check, not the basis of the independent operator validation.

### 6.2 Closed-shell Slater determinant with off-diagonal P

Use three orthonormal spatial basis functions, four electrons, and two occupied orbitals

```text
u = (1, 1, 0) / sqrt(2)
v = (1,-1, 2) / sqrt(6).
```

The fixture state was built from vacuum by applying the four occupied-orbital creation operators for `u_alpha`, `v_alpha`, `u_beta`, and `v_beta`, with bit-parity signs and superposition coefficients. It was **not** constructed by inserting a density into a disconnected 2-RDM formula. The ladder-operator Gram calculation above then produced its full 1- and 2-RDMs independently.

The resulting spin-summed density is

```text
P = [[4/3,  2/3,  2/3],
     [2/3,  4/3, -2/3],
     [2/3, -2/3,  4/3]].
```

After Gamma was independently formed, applying the production subtraction to it gave a maximum residual of **4.440892098500626e-16** over all 81 components (tolerance `1e-12`). The operator-built Gamma also matched `direct_spin1.make_rdm12(..., reorder=True)` within **2.7755575615628914e-16**, establishing that the called library returns the physical convention used in this audit.

For spatial pair `(0,1)`:

```text
G_Gamma[0,0,1,1] = 14/9 = 1.555555555555556
G_Gamma[0,1,0,1] =  2/9 = 0.22222222222222227
direct            = 16/9 = 1.7777777777777788
exchange magnitude=  2/9 = 0.22222222222222235
correct residual  = 14/9 - 16/9 + 2/9 = 0 (roundoff only)
wrong-index test  =  2/9 - 16/9 + 2/9 = -4/3
```

The measured wrong-index residual was `-1.3333333333333341`. A closed-shell two-electron rank-one example can accidentally make these raw components equal; this four-electron rank-two reference avoids that degeneracy. Its zero Lambda alone is not treated as an index discriminator: the unequal independently evaluated Gamma components and the wrong-index subtraction test supply that discrimination. The actual correlated CO example in section 5 separately distinguishes the saved Lambda components themselves.

## 7. Evidence fingerprints and reproducibility

The hashes below identify the actual files read during this audit. The existing uncommitted workspace changes were left intact. Library citations refer to the executed environment, whose source contents were verified identical to the locally inspected version.

| Source file | SHA-256 |
|---|---|
| [descriptor.py:1](/Users/novaz/Desktop/qml2.0/codes/descriptor.py:1) | `ee9fb65bbca840db594294289c0cf1895dbbe933e3a2077ab6bb838a354c90b8` |
| [run1.py:1](/Users/novaz/Desktop/qml2.0/codes/run1.py:1) | `96d4f483816ab42fb234f6e183263269eb6e23a6312788b5f2af72d0505a33eb` |
| [run2.py:1](/Users/novaz/Desktop/qml2.0/codes/run2.py:1) | `301b5efe2dbcb249a280e5ad0f35aa1a78b51aa6dedecdcd91e9a1c20a6ee9e0` |
| [run2_model.py:1](/Users/novaz/Desktop/qml2.0/codes/run2_model.py:1) | `04d20dc27bb0450e60a3c582ff3a044d64eb9855c4139f78768bd8271a4bb565` |
| [circuit.py:1](/Users/novaz/Desktop/qml2.0/codes/circuit.py:1) | `1c4018734cc703ab5c9935933116556db55c974b0fc8b48db00ee34ab140fc20` |
| [PySCF fci/direct_spin1.py:1](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/direct_spin1.py:1) | `df00b56a8a4f996bbe4bddbb697db08b64c6f3110d68765ad5ac6798dff87d2d` |
| [PySCF fci/rdm.py:1](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/fci/rdm.py:1) | `d66cd9412a6a44fb937a4b6d35e1393798a73e2809713637d985328591fd4a9b` |
| [PySCF lib/mcscf/fci_rdm.c:1](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/lib/mcscf/fci_rdm.c:1) | `f0eceef459a267e5800fa8209f42640f52e84a4ae79e764abb2e635cdcf78985` |
| [PySCF scf/hf.py:1](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/scf/hf.py:1) | `34ea1a9057da3a2d5aa37fe51d4e7305580bcb8e0118a5f17ba95ba5bb36c22f` |
| [PySCF ci/cisd.py:1](/Users/novaz/Desktop/encoding_qml/.venv/lib/python3.13/site-packages/pyscf/ci/cisd.py:1) | `3db0537aa822bdcacc524f041fb454506f0376340c676b9c567599a9c2e70ddf` |

| Actual descriptor NPZ | SHA-256 |
|---|---|
| [CO/001.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/001.npz) | `1ef76560f3f1db279db7e6babfcd8df014daacbb12c7103a06b30d929759c33b` |
| [CO/002.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/002.npz) | `bdc3e66daf5861321b56ec031aa4ac6b003b4b6ee956b2694753b08a1fc8a6f2` |
| [CO/003.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/003.npz) | `968f1a51e44b191a62f8f5de3c368b361e8628ae139e5580d5873795950887e3` |
| [CO/004.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/004.npz) | `59e98f843cb9527b1b3ff5864712e74955d43fdca4584b2ea7684bd990987dee` |
| [CO/005.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/005.npz) | `f1cc8a29022c483229b02eafb4620c91195ecfbcc8d8c7b4e7c08a9a0c323bd2` |
| [CO/006.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/006.npz) | `592abd13b7c2a8f201f8630f5e0d0cbc30e4ab6d91a505f684466f590a5ef19b` |
| [CO/007.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/007.npz) | `a3c3e4425dee27353935839c053a2f79975322c4112f67013e5f1d8c405b8bdd` |
| [CO/008.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/008.npz) | `aa119bb547ca6cd8ce175cc471b96ff2e5f111be4fb29e03b06069ae024779c0` |
| [CO/009.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/009.npz) | `f1b2fdd8a422371f454010e0e20c2b0b06ea83fdbe71701482b1671475c19090` |
| [CO/010.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/010.npz) | `52588fd42a46d5410b7c1f051167140e624b8b8f2b78120557674b1884d9a582` |
| [CO/011.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/011.npz) | `f95ec47f8a4ca078444eadbf2862b282e46edbfc801a0afbe42f3953006f97e8` |
| [CO/012.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/012.npz) | `ce595e8e64ad862784602dabab04ef5f0acfc15f3ab9403a95dea21e854d0473` |
| [CO/013.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/013.npz) | `b9195a12a44b6a1c61fd2a60d794ba41aa2cd47f0bc62da02844542ce047be44` |
| [CO/014.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/014.npz) | `1d9ffb29c6974b07000bb0a6631569922ab21c14c62d27c2c4fb481986ed8792` |
| [CO/015.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/015.npz) | `7e4a87105333f0cfece5f3f6eeccdea178ab422d2441d62d5cf2d5b093f03949` |
| [CO/016.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/016.npz) | `962e6fdd7b788eb67125c33d826b966726e63d5838265438cf711eb74916db30` |
| [CO/017.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/017.npz) | `0b6046b0c4f599730521e8459c8f77e738cb3c9e58bacfc88678dd0c0ee70292` |
| [CO/018.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/018.npz) | `409ae2b76192a9f811f8e863a808e4535889eab425c8890e4d7e3d2bd6ee3715` |
| [CO/019.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/019.npz) | `3beb9790a4dde9fd5e6fa20f0c57cc5fae23ee3e38d212c45bdebb5f218ae3fe` |
| [CO/020.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/020.npz) | `f0f69c0d6d160249f2f1103b5f7de159243b6cb405d9881596460dbf0be2c99b` |
| [CO/021.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/021.npz) | `b5a6c10a5eedede8a8821327d7c27ebcd5ccdd126f92982f9377914744c6a6af` |
| [CO/022.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/022.npz) | `f905f95c7091c28e4aabacd888d84214ff806715c935b6ccd3c755cd195005bd` |
| [CO/023.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/023.npz) | `009f5e725d67ac550b31e250e40160a70a6d51ed7252c7c83e71c162686bf438` |
| [CO/024.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/024.npz) | `718cf1cab21eb8c5556aa72ba792a7b31311195e7f755ad28365e89abf1816f3` |
| [CO/025.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/025.npz) | `24e84abb49bc244ba94239b71c1dfcd8a7b530c6e8575293bdbf3cbcbb054a00` |
| [CO/026.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/026.npz) | `c62f7fda50f3fd23d4f69338f1543cbb8ac315427bc940765790159842b6aa58` |
| [CO/027.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/027.npz) | `7a677f5cbb0b087419c8e1fe5ac9725c17c774982e01e179c5c4ecaa750ca63a` |
| [CO/028.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/028.npz) | `70f0619aab6b15a9ddb888924d6223a0e0bfd59bf2f51cec8a1f20d863382447` |
| [CO/029.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/029.npz) | `357e9f28b01a848668cec3064aada82f88c3140173e0b09839e9dd7648107992` |
| [CO/030.npz](/Users/novaz/Desktop/qml2.0/results/descriptor/CO/030.npz) | `960ed08683ed11eacb1ab9aff5d7e3b290e05002c713f90293a29fa79639cdab` |

The executed harness is reproduced below so the independent fixture, all-geometry checks, actual-consumer trace, tolerances, and read-only prediction replay can be inspected without depending on temporary files. It imports production code to exercise the real consumer, but its `operator_rdms` and `transform` functions implement the independent physical check. It writes only a scratch JSON under `/tmp`; it never calls a production main function, training routine, or result writer. Harness SHA-256: `985f6de60fc3e7e4e7aaef28823df2e0ccfd8bf5a77b170ea24431c8c6d7d15e`.

<details>
<summary>Executed read-only audit harness</summary>

```python
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, hashlib, itertools, math, csv, time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path('/Users/novaz/Desktop/qml2.0')
sys.path.insert(0, str(ROOT / 'codes'))
import pyscf
from pyscf import lib
lib.num_threads(1)
from pyscf.ci import cisd
from pyscf.fci import direct_spin1
import torch, pennylane as qml
import run1, run2, descriptor
from circuit import EncodingConstants
from run2_model import PairEnergyModel

RUN = ROOT / 'results/run2/20260925_113307_117058'
config = json.loads((RUN/'configuration.json').read_text())
population = json.loads((RUN/'population.json').read_text())
source_paths = [Path(r['source_file']) for r in population]
protected = sorted(set(source_paths + list((ROOT/'codes').glob('*.py')) +
    list(RUN.rglob('*.json')) + list(RUN.rglob('*.npy')) + list(RUN.rglob('*.csv')) +
    [Path(config['input_dir'])/'manifest.json']))
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p):digest(p) for p in protected}
stats = {}
def check(name, actual, expected, geometry, tolerance=1e-10, labels=None):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    delta = np.abs(a-b)
    assert np.isfinite(delta).all(), name
    flat = int(delta.argmax()) if delta.size else 0
    value = float(delta.flat[flat]) if delta.size else 0.0
    index = list(np.unravel_index(flat, delta.shape)) if delta.size else []
    entry = dict(max_abs=value, geometry=geometry, index=[int(x) for x in index], tolerance=tolerance)
    if labels is not None and delta.size:
        entry['pair'] = list(labels[flat])
    if name not in stats or value > stats[name]['max_abs']:
        stats[name] = entry
    assert value <= tolerance, (name, entry)
    return value

def bits(n, k):
    return sorted(sum(1 << i for i in c) for c in itertools.combinations(range(n), k))

def ci_state(ci, n, nelec):
    strings = bits(n, nelec//2)
    assert ci.shape == (len(strings),)*2
    return {strings[i] | (strings[j]<<n): float(ci[i,j])
            for i,j in zip(*np.nonzero(ci))}

def annihilate(bit, mode):
    if not (bit >> mode) & 1:
        return None
    return bit ^ (1 << mode), (-1.0 if (bit & ((1 << mode)-1)).bit_count() % 2 else 1.0)

def create_orbital(state, coeff):
    out = {}
    for bit, amplitude in state.items():
        for mode, c in enumerate(coeff):
            if c == 0 or (bit >> mode) & 1:
                continue
            sign = -1.0 if (bit & ((1 << mode)-1)).bit_count()%2 else 1.0
            target = bit | (1 << mode)
            out[target] = out.get(target, 0.0) + amplitude*c*sign
    return {b:a for b,a in out.items() if a != 0}

def operator_rdms(state, n):
    # Independent fermionic ladder operations, with no PySCF RDM or Wick formula.
    modes = 2*n
    mode_pairs = list(itertools.combinations(range(modes), 2))
    pair_index = {p:i for i,p in enumerate(mode_pairs)}
    row1, col1, data1, row2, col2, data2 = [], [], [], [], [], []
    for bit, amp in state.items():
        occupied = [m for m in range(modes) if (bit >> m) & 1]
        for a in occupied:
            target, sign = annihilate(bit, a)
            row1.append(a); col1.append(target); data1.append(amp*sign)
        for a,b in itertools.combinations(occupied,2):
            target, sa = annihilate(bit,a)
            target, sb = annihilate(target,b)
            row2.append(pair_index[a,b]); col2.append(target); data2.append(amp*sa*sb)
    one = coo_matrix((data1,(row1,col1)),shape=(modes,1<<modes)).tocsr()
    two = coo_matrix((data2,(row2,col2)),shape=(len(mode_pairs),1<<modes)).tocsr()
    h1 = (one.conj()@one.T).toarray()
    h2 = (two.conj()@two.T).toarray()
    p = (h1[:n,:n]+h1[n:,n:]).T
    idx = np.zeros((modes,modes),dtype=int)
    sign = np.zeros((modes,modes))
    for k,(a,b) in enumerate(mode_pairs):
        idx[a,b] = idx[b,a] = k
        sign[a,b], sign[b,a] = 1., -1.
    g = np.zeros((n,)*4)
    for sigma,tau in itertools.product(range(2),repeat=2):
        ix = idx[sigma*n:(sigma+1)*n,tau*n:(tau+1)*n]
        sg = sign[sigma*n:(sigma+1)*n,tau*n:(tau+1)*n]
        g += h2[ix[:,None,:,None],ix[None,:,None,:]]*sg[:,None,:,None]*sg[None,:,None,:]
    return p,g

def transform(g, a):
    # Independent sequential axis contractions; preserves all four axis identities.
    out = g
    for axis in range(4):
        out = np.moveaxis(np.tensordot(a,out,axes=(1,axis)),0,axis)
    return out

def fixture():
    n=3
    u=np.array([1.,1.,0.])/np.sqrt(2)
    v=np.array([1.,-1.,2.])/np.sqrt(6)
    state={0:1.}
    for spin in range(2):
        for orbital in (u,v):
            coeff=np.zeros(6); coeff[spin*3:(spin+1)*3]=orbital
            state=create_orbital(state,coeff)
    p,g=operator_rdms(state,n)
    strings=bits(3,2)
    ci=np.array([[state.get(a|(b<<3),0.) for b in strings] for a in strings])
    pp,gp=direct_spin1.make_rdm12(ci,3,4,reorder=True)
    check('fixture_P_operator_vs_PySCF',p,pp,'fixture',1e-12)
    check('fixture_Gamma_operator_vs_PySCF',g,gp,'fixture',1e-12)
    check('fixture_P_vs_occupied_orbitals',p,2*(np.outer(u,u)+np.outer(v,v)),'fixture',1e-12)
    # Apply the production subtraction only AFTER independently constructing Gamma.
    residual=g-descriptor.disconnected_rhf(p)
    check('fixture_RHF_reference_residual',residual,np.zeros_like(g),'fixture',1e-12)
    return dict(orbitals=[u.tolist(),v.tolist()], norm=sum(x*x for x in state.values()),
        P=p.tolist(), gamma_raw_0011=float(g[0,0,1,1]),gamma_raw_0101=float(g[0,1,0,1]),
        direct=float(p[0,0]*p[1,1]),exchange=float(.5*p[0,1]*p[1,0]),
        residual_max=float(abs(residual).max()),
        wrong_index_residual=float(g[0,1,0,1]-p[0,0]*p[1,1]+.5*p[0,1]*p[1,0]))

fixture_result=fixture()
print('Fixture complete', fixture_result, flush=True)
samples,manifest,metadata=run2.load_population(Path(config['input_dir']),tuple(config['molecules']),
    split_protocol=config['split_protocol'],methods=('MB2',))
constants_data=json.loads((RUN/'preprocessing.json').read_text())['CO/MB2']
model=PairEnergyModel(EncodingConstants(**constants_data),method='MB2')
theta=torch.tensor(np.load(RUN/'runs/CO/MB2/seed_00/best_theta.npy'),dtype=torch.float64)
expected_pairs=list(itertools.combinations(range(8),2))
assert len(samples)==30 and len(expected_pairs)==28
details=[]
wrong_max={'difference':-1}
correlated_max={'difference':-1}
angle_observations=[]
def trace_angle(frame,event,arg):
    if event=='line' and frame.f_code.co_name=='pair_encoder' and frame.f_lineno==84:
        d=frame.f_locals
        angle_observations.append((d['position'],d['mu'],d['nu'],float(d['lambda_pairs'][d['position']]),float(d['argument'])))
    return trace_angle

for sample in samples:
    gid=sample.sample_id
    raw=dict(np.load(sample.source_file,allow_pickle=False))
    pop=next(r for r in population if r['geometry_id']==f'{sample.geometry_index:03}')
    selected=np.array(metadata[gid]['source_selected_ao_indices'])
    assert selected.tolist()==pop['source_selected_ao_indices']==[1,2,3,4,6,7,8,9]
    assert str(sample.source_file)==pop['source_file']
    assert np.array_equal(raw['pair_indices'],np.array(expected_pairs))
    n=len(raw['mo_occ']); nelec=int(raw['electron_count'])
    assert n==10 and nelec==14 and raw['mo_occ'].tolist()==[2.]*7+[0.]*3
    assert str(raw['pyscf_version'])==pyscf.__version__=='2.14.0'
    # Rebuild the exact generating constructor from saved MP2 amplitudes; no solver.
    rebuilt=descriptor.build_consistent_rdms(raw['mp2_t2'],n,nelec)
    for key in ('ci_reference','ci_first_order','Gamma_MP2_MO_consistent','P_MP2_MO_consistent'):
        check('constructor_'+key,rebuilt[key],raw[key],gid)
    ref=ci_state(raw['ci_reference'],n,nelec)
    chi=ci_state(raw['ci_first_order'],n,nelec)
    total=ref.copy()
    for b,a in chi.items(): total[b]=total.get(b,0.)+a
    p0,g0=operator_rdms(ref,n)
    pt,gt=operator_rdms(total,n)
    w=sum(a*a for a in chi.values())
    p_ind=pt-w*p0; g_ind=gt-w*g0
    check('operator_Gamma_MO',g_ind,raw['Gamma_MP2_MO_consistent'],gid)
    check('operator_P_MP2_MO',p_ind,raw['P_MP2_MO_consistent'],gid)
    C=raw['mo_coeff']; S=raw['S_AO']
    vals,vecs=np.linalg.eigh(S)
    half=(vecs*np.sqrt(vals))@vecs.T
    check('S_half_positive_root',half,raw['S_half'],gid)
    check('MO_orthonormal',C.T@S@C,np.eye(n),gid)
    p_rhf=(C*raw['mo_occ'])@C.T
    check('P_AO_is_RHF_spin_sum',p_rhf,raw['P_AO'],gid)
    check('RHF_electron_trace',np.trace(p_rhf@S),14.,gid)
    check('RHF_idempotency',p_rhf@S@p_rhf,2*p_rhf,gid)
    pL=half@p_rhf@half
    check('P_L_full_transform',pL,raw['P_L'],gid)
    check('MB1_P_mapping',sample.pij,pL[np.ix_(selected,selected)],gid)
    gAO=transform(g_ind,C)
    check('operator_Gamma_AO',gAO,raw['Gamma_MP2_AO_consistent'],gid)
    g0AO=transform(g0,C)
    check('operator_RHF_Gamma0_AO',g0AO,raw['Gamma0_AO'],gid)
    check('Lambda_AO_independent',gAO-g0AO,raw['Lambda_AO'],gid)
    for key,other in (('Gamma_MP2_AO','Gamma_MP2_AO_consistent'),('Gamma0_HF_AO','Gamma0_AO'),('Lambda_HFref_AO','Lambda_AO')):
        check('alias_'+key,raw[key],raw[other],gid,0.)
    lambdaL_ind=transform(gAO-g0AO,half)
    check('Lambda_L_independent_full',lambdaL_ind,raw['Lambda_L'],gid)
    check('Lambda_L_saved_full_transform',transform(raw['Lambda_AO'],half),raw['Lambda_L'],gid)
    gL_source=transform(raw['Gamma_MP2_AO_consistent'],half)
    gL_ind=transform(g_ind,half@C)
    i=np.array([selected[a] for a,b in expected_pairs]); j=np.array([selected[b] for a,b in expected_pairs])
    direct=pL[i,i]*pL[j,j]; exchange=.5*pL[i,j]*pL[j,i]
    reconstructed=gL_source[i,i,j,j]-direct+exchange
    independently_reconstructed=gL_ind[i,i,j,j]-direct+exchange
    args=model.pair_quantum_arguments(sample,theta)
    actual=args[3].numpy()
    e_saved=check('pair_saved_Gamma_reconstruction',reconstructed,actual,gid,labels=expected_pairs)
    e_operator=check('pair_operator_reconstruction',independently_reconstructed,actual,gid,labels=expected_pairs)
    check('loader_to_angle_tensor',sample.lambda_pairs,actual,gid,0.,expected_pairs)
    check('raw_I_I_J_J',raw['Lambda_L'][i,i,j,j],actual,gid,0.,expected_pairs)
    methods=raw['Lambda_L'].transpose(0,2,1,3)
    check('Methods_I_J_I_J',methods[i,j,i,j],actual,gid,0.,expected_pairs)
    selected_raw=raw['Lambda_L'][np.ix_(selected,selected,selected,selected)]
    a=np.array([a for a,b in expected_pairs]); b=np.array([b for a,b in expected_pairs])
    check('selected_tensor_local_indices',selected_raw[a,a,b,b],actual,gid,0.,expected_pairs)
    wrong=raw['Lambda_L'][i,j,i,j]
    k=int(np.argmax(abs(wrong-actual)))
    if abs(wrong[k]-actual[k])>wrong_max['difference']:
        wrong_max=dict(difference=float(abs(wrong[k]-actual[k])),geometry=gid,pair=expected_pairs[k],
            full_AO_pair=[int(i[k]),int(j[k])],correct=float(actual[k]),wrong=float(wrong[k]))
    dL=half@raw['P_MP2_AO_consistent']@half
    correlated=gL_source[i,i,j,j]-dL[i,i]*dL[j,j]+.5*dL[i,j]*dL[j,i]
    k=int(np.argmax(abs(correlated-actual)))
    if abs(correlated[k]-actual[k])>correlated_max['difference']:
        correlated_max=dict(difference=float(abs(correlated[k]-actual[k])),geometry=gid,pair=expected_pairs[k],
            actual=float(actual[k]),correlated_reference=float(correlated[k]))
    angle_observations.clear()
    sys.settrace(trace_angle)
    try: tape=model.qnode.construct(args,{})
    finally: sys.settrace(None)
    gates=[op for op in tape.operations if op.name=='IsingZZ']
    assert [tuple(op.wires) for op in gates]==expected_pairs
    assert [(mu,nu) for pos,mu,nu,value,argument in angle_observations]==expected_pairs
    check('angle_line84_Lambda',np.array([v[3] for v in angle_observations]),actual,gid,0.,expected_pairs)
    arguments=np.array([v[4] for v in angle_observations])
    coeff=args[-1].numpy()
    check('angle_argument',arguments,coeff[0]*sample.pij[a,b]+coeff[1]*actual,gid,1e-14,expected_pairs)
    check('IsingZZ_angles',np.array([float(op.data[0]) for op in gates]),.5*np.pi*np.tanh(arguments),gid,1e-14,expected_pairs)
    # Non-training sensitivity fixture on the same QNode: isolate Lambda with a2=0,b2=1.
    probe=theta.clone(); probe[-2]=0.; probe[-1]=1.
    probe_args=model.pair_quantum_arguments(sample,probe)
    probe_tape=model.qnode.construct(probe_args,{})
    probe_gates=[op for op in probe_tape.operations if op.name=='IsingZZ']
    check('Lambda_only_sensitivity_angles',np.array([float(op.data[0]) for op in probe_gates]),.5*np.pi*np.tanh(actual),gid,1e-14,expected_pairs)
    k=int(np.argmax(abs(reconstructed-actual)))
    details.append(dict(geometry=gid,bond_length_A=float(raw['bond_length_A']),split=pop['split'],pairs=len(gates),
        saved_pair_residual=e_saved,operator_pair_residual=e_operator,worst_pair=expected_pairs[k],
        rows=[dict(pair=pair,full_AO=[int(i[t]),int(j[t])],gamma=float(gL_source[i[t],i[t],j[t],j[t]]),
            direct=float(direct[t]),exchange=float(exchange[t]),lambda_reconstructed=float(reconstructed[t]),
            run2_input=float(actual[t]),wrong_raw_ijij=float(wrong[t]),
            P_II=float(pL[i[t],i[t]]),P_JJ=float(pL[j[t],j[t]]),P_IJ=float(pL[i[t],j[t]]),P_JI=float(pL[j[t],i[t]]))
            for t,pair in enumerate(expected_pairs)]))
    print(gid,'pairs',len(gates),'saved residual',e_saved,'operator residual',e_operator,flush=True)

# Forward evaluation only, using existing checkpoints, links current data path to saved results.
replay_count=0
with torch.no_grad():
    for seed in config['seeds']:
        seed_dir=RUN/f'runs/CO/MB2/seed_{seed:02}'
        saved_rows=list(csv.DictReader((seed_dir/'predictions.csv').open()))
        saved_by_id={r['geometry_id']:r for r in saved_rows}
        assert len(saved_by_id)==len(saved_rows)==30
        t=torch.tensor(np.load(seed_dir/'best_theta.npy'),dtype=torch.float64)
        for sample in samples:
            expected=float(saved_by_id[f'{sample.geometry_index:03}']['prediction_Ha'])
            actual=float(model(sample,t))
            check('saved_prediction_replay',actual,expected,f'{sample.sample_id}/seed_{seed:02}',1e-10)
            replay_count+=1
        print('Forward replay seed',seed,'complete',flush=True)

after={str(p):digest(p) for p in protected}
assert before==after, 'Protected input/source/result files changed during audit'
out=dict(run=str(RUN),config=config,source_hashes={str(p):before[str(p)] for p in source_paths},
    code_hashes={str(p):before[str(p)] for p in (ROOT/'codes').glob('*.py')},stats=stats,
    fixture=fixture_result,details=details,wrong_index=wrong_max,correlated_reference=correlated_max,
    protected_count=len(protected),protected_unchanged=True,predictions_replayed=replay_count,
    versions=dict(pyscf=pyscf.__version__,numpy=np.__version__,torch=torch.__version__,pennylane=qml.__version__))
Path('/tmp/run2_lambda_audit_results.json').write_text(json.dumps(out,indent=2)+'\n')
print('AUDIT COMPLETE',json.dumps(stats,indent=2),flush=True)
```

</details>

**Conclusion: ordering PASS; subtraction PASS; AO mapping PASS; extraction/consumption PASS. Across the entire current Run2 CO population, MB2 receives and encodes the intended `Lambda_Methods[i,j,i,j]` RHF-reference descriptor.**
