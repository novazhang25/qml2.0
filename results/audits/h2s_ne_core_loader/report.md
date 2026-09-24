# H2S Ne-core valence-loader audit

PASS: all 30 saved H2S geometries load as six-AO blocks.
Retained labels: S 3s, 3px, 3py, 3pz, and both H 1s AOs.
Stored full-AO indices: [2, 6, 7, 8, 9, 10].
The full eleven-AO Lowdin transformation precedes subspace selection.

The loader block and raw FE match their saved references exactly.
The saved raw FE reference is the historical saved_6AO_diagnostic view in
`/Users/novaz/Desktop/qml2.0/results/audits/fe_all_molecules/fe_eigenvalues.csv`; it contains unscaled eigenvalues.

| Comparison | Maximum absolute residual over 30 geometries |
|---|---:|
| loader_block_vs_saved_P_L_block | 0 |
| loader_FE_vs_saved_raw_FE | 0 |
| saved_full_P_L_vs_full_AO_Lowdin_transform | 0 |
| loader_block_vs_full_transform_then_selection | 0 |
| loader_block_vs_saved_P_mu_P_munu_reconstruction | 3.3306690738754696e-16 |

Descriptor and reference SHA-256 hashes were unchanged.
No electronic-structure calculations, training, or data regeneration were run.
