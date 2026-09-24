# CO FE density-eigenvalue audit

All 30 geometries audited. Absolute criterion: |lambda| <= 1e-15 or |lambda-2| <= 1e-15; rtol=0.
Near-zero geometries: 30/30; values: 60.
Near-two geometries: 23/30; values: 32.
Near-two geometry IDs: 001, 002, 003, 004, 008, 009, 010, 011, 012, 013, 014, 016, 017, 018, 019, 020, 022, 024, 025, 026, 027, 028, 029.

## Pipeline audit

1. initialdata.rhf_arrays saves the closed-shell spin-summed RHF density and MO occupations.
2. descriptor.calculate_record uses that density as an SCF initial guess, reconverges RHF, canonicalizes MOs, then sets P_AO=mf.make_rdm1(). MP2 densities are separate arrays.
3. descriptor.lowdin_factors computes the full 10x10 positive square root of S; P_L=S_half @ P_AO @ S_half.
4. The full transformation precedes AO selection. C1s/O1s indices [0,5] are excluded; valence indices are [1,2,3,4,6,7,8,9].
5. run1.load_record validates the stored transforms and selects the full 8x8 block as sample.pij. raw_spectra applies np.linalg.eigvalsh directly, without rounding, clipping, diagonal-only extraction, or post-sort.
6. prepare_runs fingerprints were checked exactly against the independently extracted stored block spectra for all 30 geometries.
7. circuit.fit_constants uses only odd-ID training geometries. Statistics match the completed CO run exactly.

Maximum absolute residuals across all geometries (audit acceptance: 1e-10, separate from the requested 1e-15 classification):

- P_AO_minus_C_occ_CT: 0
- C_T_S_C_minus_I: 1.7763568394002505e-14
- S_half_squared_minus_S: 4.6629367034256575e-15
- S_half_minus_independent_eigh_root: 0
- saved_P_L_minus_full_transform: 0
- P_S_P_minus_2P: 1.8207657603852567e-14
- P_L_squared_minus_2P_L: 1.7319479184152442e-14
- trace_P_S_minus_14: 2.3092638912203256e-14
- trace_P_L_minus_14: 3.1974423109204508e-14
- valence_projector_compression_identity: 1.4840559336981585e-14
- production_FE_minus_independent_MO_overlap_reconstruction: 0

Upstream checkpoint / reconverged RHF diagnostics (the upstream density is an initial guess, not required to be identical to the reconverged density):

- upstream_density_from_MO_residual: 0
- upstream_overlap_residual: 0
- upstream_to_reconverged_descriptor_density_difference: 8.1978868138321559e-13
- saved_RHF_energy_difference: 1.1368683772161603e-13
- scf_commutator_residual: 3.8447023342769171e-13

Code inspected: codes/initialdata.py (rhf_arrays), codes/descriptor.py (calculate_record, lowdin_factors, select_valence_aos), codes/run1.py (load_record, raw_spectra, prepare_runs), codes/circuit.py (fit_constants, encode). The installed PySCF scf.hf.make_rdm1 builds the spin-summed density as (C_occ * occupations) @ C_occ.conj().T.

## Interpretation and preprocessing consequence

CO has 14 electrons, 10 spatial AOs and occupations [2,2,2,2,2,2,2,0,0,0]. The full orthonormal RHF density is twice an occupied-space projector: P_L^2=2 P_L. Its full spectrum is therefore seven 2s and three 0s, up to numerical error.
For the 8-dimensional valence compression, dimension counting forces at least five eigenvalues at 2 and at least one at 0 in exact arithmetic. Cropping does not generally preserve idempotency: the audit verifies A^2-2A=-B B^T for valence block A and valence/core coupling B.
The actual CO spectra contain two near-zero entries, one varying fractional entry and five near-two entries. The strict 1e-15 test does not include every near-two entry: matrix arithmetic/eigensolver roundoff can exceed that cutoff.
Valence/core coupling singular-value ranges: [(0.10179938942177644, 0.11268979948149252), (7.511513450082079e-16, 6.3499487918986304e-15)]. For this scan there is only one significant singular value, consistent with one eigenvalue departing materially from the endpoints. The valence trace is 10 plus the fractional eigenvalue: removing core AOs is not the same operation as removing exactly occupied core MOs.
The production scaler masks zero-based columns [0, 1, 3, 4, 5, 6, 7] because their training SD is below 1e-12. Their encoding angles are exactly zero for all 30 geometries. Only column 2 (the third eigenvalue) supplies varying input angles.
Third eigenvalue range: 0.0051950521266224126 to 0.0063697825182544854.
Training SD by column: [2.1956223049242207e-16, 2.0258761022431528e-16, 0.0003676539989454187, 2.1413206232596582e-15, 1.500513412242058e-15, 1.3322676295501878e-15, 1.897149936107019e-15, 2.2535066179978845e-15].
This is a low-information RHF-spectrum descriptor for this CO scan, not evidence that FE accidentally uses Fock, MP2, diagonal density elements, or manually rounded occupations. The audit does not independently rerun electronic-structure solvers.

## All matching geometries: actual raw FE eigenvalues

Indices below are zero-based. Arrays retain float64 round-trip precision.

### CO geom_001
Near 0 indices: [0, 1]; near 2 indices: [6].
```text
[-3.7759955008108545e-17, 4.8184916485817117e-16, 0.0051950521266224126, 1.9999999999999898, 1.9999999999999936, 1.9999999999999984, 1.9999999999999998, 2.0000000000000013]
```

### CO geom_002
Near 0 indices: [0, 1]; near 2 indices: [6].
```text
[-2.741597229261468e-16, 5.2115118001115522e-17, 0.0052692990386140615, 1.9999999999999942, 1.9999999999999984, 1.9999999999999987, 2, 2.0000000000000031]
```

### CO geom_003
Near 0 indices: [0, 1]; near 2 indices: [7].
```text
[-3.025509174258873e-16, 8.0506312500856012e-17, 0.0053411923698809538, 1.999999999999992, 1.999999999999996, 1.9999999999999987, 1.9999999999999989, 2.0000000000000009]
```

### CO geom_004
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-4.0968882432673009e-17, 4.0968882432673009e-17, 0.0054107027495375881, 1.9999999999999927, 1.9999999999999962, 2, 2.0000000000000036, 2.0000000000000036]
```

### CO geom_005
Near 0 indices: [0, 1]; near 2 indices: [].
```text
[-6.782644242895087e-17, 2.8987104735398217e-16, 0.0054778050943116163, 1.9999999999999951, 1.9999999999999984, 1.9999999999999987, 2.0000000000000036, 2.0000000000000058]
```

### CO geom_006
Near 0 indices: [0, 1]; near 2 indices: [].
```text
[-2.926416439814838e-16, 7.0597039056452525e-17, 0.0055424784607707167, 1.999999999999996, 1.999999999999998, 2.0000000000000013, 2.0000000000000022, 2.000000000000004]
```

### CO geom_007
Near 0 indices: [0, 1]; near 2 indices: [].
```text
[-4.4977684551365017e-16, 4.4977684551365017e-16, 0.0056047058966917085, 1.99999999999999, 1.9999999999999944, 1.9999999999999984, 2.0000000000000013, 2.0000000000000027]
```

### CO geom_008
Near 0 indices: [0, 1]; near 2 indices: [4, 5].
```text
[-9.0793716535035841e-16, 1.975874565023319e-17, 0.0056644742920171076, 1.9999999999999902, 1.9999999999999998, 2.0000000000000009, 2.0000000000000031, 2.0000000000000062]
```

### CO geom_009
Near 0 indices: [0, 1]; near 2 indices: [6, 7].
```text
[-2.7971453342643721e-17, 2.7971453342643721e-17, 0.0057217742298103502, 1.9999999999999909, 1.9999999999999942, 1.9999999999999964, 1.9999999999999996, 1.9999999999999996]
```

### CO geom_010
Near 0 indices: [0, 1]; near 2 indices: [5, 6, 7].
```text
[-1.4987248933898924e-16, 1.4987248933898924e-16, 0.0057765998375756133, 1.9999999999999907, 1.9999999999999927, 1.9999999999999993, 2.0000000000000009, 2.0000000000000009]
```

### CO geom_011
Near 0 indices: [0, 1]; near 2 indices: [6, 7].
```text
[-2.3725216698459049e-16, 1.5207562059559185e-17, 0.0058289486393126033, 1.9999999999999911, 1.999999999999998, 1.9999999999999987, 2, 2.0000000000000009]
```

### CO geom_012
Near 0 indices: [0, 1]; near 2 indices: [6, 7].
```text
[-2.854239668519544e-16, 5.0746857177698571e-16, 0.005878821408611179, 1.9999999999999916, 1.9999999999999958, 1.9999999999999982, 2.0000000000000009, 2.0000000000000009]
```

### CO geom_013
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-4.7870599531733866e-17, 4.9195980938179648e-16, 0.0059262220230948959, 1.9999999999999931, 1.9999999999999971, 2.0000000000000009, 2.0000000000000018, 2.0000000000000049]
```

### CO geom_014
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-1.979291834325409e-19, 4.442871390334952e-16, 0.005971157320479259, 1.999999999999996, 1.9999999999999982, 2, 2.0000000000000013, 2.0000000000000053]
```

### CO geom_015
Near 0 indices: [0, 1]; near 2 indices: [].
```text
[-4.5784268459049232e-16, 4.5784268459049232e-16, 0.0060136369564878223, 1.9999999999999949, 1.999999999999996, 2.0000000000000013, 2.0000000000000044, 2.0000000000000067]
```

### CO geom_016
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-2.8058052088184754e-17, 4.7214726193824731e-16, 0.0060536732648537317, 1.9999999999999949, 1.999999999999998, 1.9999999999999996, 2.0000000000000013, 2.0000000000000036]
```

### CO geom_017
Near 0 indices: [0, 1]; near 2 indices: [5, 6].
```text
[-1.7292207664634688e-17, 1.7292207664634688e-17, 0.0060912811195985416, 1.9999999999999967, 1.9999999999999984, 1.9999999999999993, 2.0000000000000004, 2.0000000000000022]
```

### CO geom_018
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-1.2938503506741559e-17, 4.5702771335680418e-16, 0.0061264777997704911, 1.9999999999999962, 1.999999999999998, 1.9999999999999996, 2.0000000000000013, 2.0000000000000027]
```

### CO geom_019
Near 0 indices: [0, 1]; near 2 indices: [7].
```text
[2.1886810278126736e-16, 2.2522110706879526e-16, 0.0061592828567942348, 1.9999999999999944, 1.9999999999999949, 1.9999999999999964, 1.9999999999999967, 1.9999999999999993]
```

### CO geom_020
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-5.5818094172142555e-17, 5.5818094172142555e-17, 0.0061897179845737949, 1.9999999999999925, 1.9999999999999989, 1.9999999999999991, 2.0000000000000022, 2.000000000000004]
```

### CO geom_021
Near 0 indices: [0, 1]; near 2 indices: [].
```text
[-1.5144798107602992e-17, 4.5923400795766558e-16, 0.0062178068924572116, 1.9999999999999938, 1.9999999999999978, 1.9999999999999987, 2.0000000000000018, 2.0000000000000058]
```

### CO geom_022
Near 0 indices: [0, 1]; near 2 indices: [4].
```text
[-4.2831267451307934e-17, 4.2831267451307934e-17, 0.006243575181166916, 1.9999999999999962, 2, 2.0000000000000027, 2.0000000000000036, 2.0000000000000058]
```

### CO geom_023
Near 0 indices: [0, 1]; near 2 indices: [].
```text
[-4.449091839953961e-16, 8.1997414533348512e-19, 0.0062670502217927471, 1.9999999999999951, 1.9999999999999969, 1.9999999999999982, 1.9999999999999987, 2.0000000000000013]
```

### CO geom_024
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-4.5026333128345257e-16, 6.1741214333899091e-18, 0.0062882610378911519, 1.9999999999999942, 1.9999999999999978, 1.9999999999999991, 2.0000000000000013, 2.0000000000000044]
```

### CO geom_025
Near 0 indices: [0, 1]; near 2 indices: [5].
```text
[-4.4960424317478361e-16, 5.5150333247210589e-18, 0.0063072381907669506, 1.9999999999999918, 1.9999999999999967, 2.0000000000000004, 2.0000000000000022, 2.0000000000000044]
```

### CO geom_026
Near 0 indices: [0, 1]; near 2 indices: [6].
```text
[-1.7535905910244373e-17, 4.6162511576030696e-16, 0.0063240136679676429, 1.9999999999999967, 1.9999999999999969, 1.9999999999999989, 1.9999999999999996, 2.0000000000000036]
```

### CO geom_027
Near 0 indices: [0, 1]; near 2 indices: [6].
```text
[-4.9477251014455885e-16, 5.0683300294496273e-17, 0.0063386207750231183, 1.999999999999996, 1.9999999999999973, 1.9999999999999989, 2.0000000000000009, 2.0000000000000013]
```

### CO geom_028
Near 0 indices: [0, 1]; near 2 indices: [5, 6].
```text
[-5.3850108639289286e-16, 9.4411876542830256e-17, 0.0063510940304447594, 1.9999999999999973, 1.9999999999999989, 1.9999999999999991, 1.9999999999999996, 2.0000000000000027]
```

### CO geom_029
Near 0 indices: [0, 1]; near 2 indices: [5, 6].
```text
[-1.2237125330601407e-17, 1.2237125330601407e-17, 0.0063614690640014029, 1.9999999999999944, 1.9999999999999962, 1.9999999999999991, 2.0000000000000009, 2.0000000000000031]
```

### CO geom_030
Near 0 indices: [0, 1]; near 2 indices: [].
```text
[-2.6514727777035627e-16, 4.8719188269538752e-16, 0.0063697825182544854, 1.9999999999999956, 1.9999999999999976, 1.9999999999999989, 2.0000000000000031, 2.000000000000004]
```

Inputs were hash-checked unchanged. No production code or descriptor was modified.
