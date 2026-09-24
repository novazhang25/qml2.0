# Methods 4.1 full-space validation report

Current result: **270 successful, 0 failed, 0 not run**. All production MP2/RDM calculations use **zero frozen orbitals**. Full AO tensors are retained through the Lowdin transformation; valence AOs are selected only for descriptors.

The original FCI labels retain their frozen-core convention and their exact saved energies. They were not recomputed. Standard PySCF density failures are audit results only.

42 focused tests passed. The H2O smoke test passed before the remaining geometries were generated.

| Molecule | Geometries | Full AOs | Occupied MOs correlated | Virtual MOs | One-body | Pair | Triple |
|---|---:|---:|---:|---:|---:|---:|---:|
| LiH | 30 | 6 | 2 | 4 | 5 | 10 | 10 |
| HF | 30 | 6 | 5 | 1 | 5 | 10 | 10 |
| BeH2 | 30 | 7 | 3 | 4 | 6 | 15 | 20 |
| H2O | 30 | 7 | 5 | 2 | 6 | 15 | 20 |
| H2S | 30 | 11 | 9 | 2 | 6 | 15 | 20 |
| NH3 | 30 | 8 | 5 | 3 | 7 | 21 | 35 |
| N2 | 30 | 10 | 7 | 3 | 8 | 28 | 56 |
| CO | 30 | 10 | 7 | 3 | 8 | 28 | 56 |
| H2O2 | 30 | 12 | 9 | 3 | 10 | 45 | 120 |

## Maximum formal residuals

| Check | Maximum residual |
|---|---:|
| rhf_energy_error_Ha | 5.684342e-13 |
| consistent_AO_contraction | 9.237056e-14 |
| consistent_MO_contraction | 9.947598e-14 |
| consistent_AO_contraction_loop_vs_einsum | 3.552714e-14 |
| consistent_AO_particle_number | 6.394885e-14 |
| consistent_AO_pair_number | 1.136868e-12 |
| hf_reference_cumulant_contraction | 8.038015e-14 |
| conventional_cumulant_contraction | 1.437739e-13 |
| lowdin_metric_max_error | 4.773959e-15 |
| rhf_idempotency_lowdin_max_error | 1.794120e-13 |
| rank4_explicit_loop_max_error | 2.775558e-17 |
| consistent_second_order_energy_error_Ha | 5.282857e-13 |

## H2O smoke comparison

Both tracks use P shape `(7,7)` and Gamma shape `(7,7,7,7)`. Full-space t2 shape is `(5,5,2,2)`.

| Track/basis | Electron number | Pair number | Contraction maximum |
|---|---:|---:|---:|
| consistent/MO | 10 | 90 | 7.105427e-15 |
| consistent/AO | 9.99999999999999 | 89.9999999999999 | 7.105427e-15 |
| pyscf_audit/MO | 10 | 90.0399959958546 | 1.070563e-02 |
| pyscf_audit/AO | 10 | 90.0399959958546 | 1.866008e-02 |

HF-reference cumulant contraction residual: 8.812395e-15. Its expected contraction is (N-1)(D-P_HF), not zero.
Conventional diagnostic cumulant contraction residual: 1.261491e-14; expected contraction is 0.5 D S D-D.

Standard PySCF audit reports exact-contraction failures for 270 geometries; none invalidates the consistent descriptor track.

The complete residuals, normalization, perturbation order, energy comparisons and index conventions are in each geometry JSON/NPZ and `validation_summary.json`.
