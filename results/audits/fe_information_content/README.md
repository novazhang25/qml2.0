# Run 1 FE information content

All 30 geometries per molecule; H2S excluded. Raw, unstandardized eigenvalues. Columns are zero-based. Training std uses the 15 odd IDs and ddof=0; test uses the 15 even IDs. Activity: training std > 1e-12. Min/max/span and Pearson correlations use all 30 points. Geometry correlation uses saved q_A in Å (index correlation is also exported). Energy is the saved Run 1 target E_FCI − E_RHF_input in Hartree.

No RHF, FCI, descriptor generation, or training was run. Existing Run1_exact eigenvalues were read from the prior FE audit; all 240 descriptor hashes match that audit. Lines follow geometry order, not eigenvalue sorting. Endpoint annotations identify geometry IDs.

| Molecule | Column | Min | Max | Span | Training std | r(q, λ) | r(Ecorr, λ) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LiH | 3 | 0.02477415965 | 0.03020492195 | 0.005430762302 | 0.001706354686 | 0.9991886 | -0.9831490 |
| BeH2 | 3 | 0.009826946621 | 0.01818082484 | 0.008353878222 | 0.002532014037 | 0.9958505 | -0.9830231 |
| H2O | 1 | 0.003162070653 | 0.004017510702 | 0.0008554400496 | 0.0002599922078 | -0.9899251 | 0.9968709 |
| NH3 | 2 | 0.007096535596 | 0.007411335702 | 0.0003148001059 | 8.749703006e-05 | -0.3492279 | 0.3995444 |
| N2 | 2 | 0.004713738862 | 0.00575905318 | 0.001045314317 | 0.0003272186131 | 0.9740416 | -0.9679549 |
| CO | 2 | 0.005195052127 | 0.006369782518 | 0.001174730392 | 0.0003676539989 | 0.9746451 | -0.9703145 |
| HF | 0 | 0.001119880052 | 0.0018802488 | 0.0007603687484 | 0.0002308717278 | -0.9994128 | 0.9979950 |
| H2O2 | 1 | 0.001979216835 | 0.002005635466 | 2.641863123e-05 | 7.926397629e-06 | 0.9900922 | -0.9808310 |
| H2O2 | 2 | 0.004276623764 | 0.005651878324 | 0.00137525456 | 0.000413014313 | -0.9996961 | 0.9967980 |

## Geometry folds and single-valuedness

Visual inspection: all nine active eigenvalue curves appear smooth at the sampled resolution. LiH, BeH2, H2O, N2, CO, HF, and both H2O2 columns are monotonic and support approximately single-valued energy relations over this scan. NH3 column 2 is smooth but nonmonotonic: its energy relation folds into two branches, with up to 26.23 mHa interpolated separation at the same eigenvalue. Smoothness therefore does not imply an informative one-to-one descriptor. These conclusions are restricted to the sampled scan.

- **LiH, column 3:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.
- **BeH2, column 3:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.
- **H2O, column 1:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.
- **NH3, column 2:** 1 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 26.227 mHa.
- **N2, column 2:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.
- **CO, column 2:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.
- **HF, column 0:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.
- **H2O2, column 1:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.
- **H2O2, column 2:** 0 sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = 0 mHa.

Branch separation measures the largest vertical energy difference where two nonadjacent geometry-order line segments overlap in eigenvalue. It is a sampled, piecewise-linear diagnostic, not a fitted physical model or a proof between geometries. Zero separation for a monotonic eigenvalue supports a single-valued relation on this scan. Pearson correlation alone does not establish smoothness or single-valuedness.

## Plots

### LiH, column 3

![Geometry](LiH_lambda_3_geometry.png)

![Correlation energy](LiH_lambda_3_ecorr.png)

### BeH2, column 3

![Geometry](BeH2_lambda_3_geometry.png)

![Correlation energy](BeH2_lambda_3_ecorr.png)

### H2O, column 1

![Geometry](H2O_lambda_1_geometry.png)

![Correlation energy](H2O_lambda_1_ecorr.png)

### NH3, column 2

![Geometry](NH3_lambda_2_geometry.png)

![Correlation energy](NH3_lambda_2_ecorr.png)

### N2, column 2

![Geometry](N2_lambda_2_geometry.png)

![Correlation energy](N2_lambda_2_ecorr.png)

### CO, column 2

![Geometry](CO_lambda_2_geometry.png)

![Correlation energy](CO_lambda_2_ecorr.png)

### HF, column 0

![Geometry](HF_lambda_0_geometry.png)

![Correlation energy](HF_lambda_0_ecorr.png)

### H2O2, column 1

![Geometry](H2O2_lambda_1_geometry.png)

![Correlation energy](H2O2_lambda_1_ecorr.png)

### H2O2, column 2

![Geometry](H2O2_lambda_2_geometry.png)

![Correlation energy](H2O2_lambda_2_ecorr.png)

