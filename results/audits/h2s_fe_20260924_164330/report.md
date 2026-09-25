# H2S FE audit rerun

Current Run 1 loader: 30/30 PASS. Both saved and current expected valence selections contain six AOs: [2,6,7,8,9,10]. The previous six-versus-ten loader mismatch is no longer present in the current code.

Full-space projector: 30/30 PASS at tolerance 1e-10.
Max projector residual: 1.9984014443252818e-14.
Full-spectrum distance from {0,2}: 0 to 1.1990408665951691e-14.

FE width: 6; active dimensions (std > 1e-12): 2; active columns (zero-based): [0, 1].
Run 1 15/15 split: odd IDs train; even IDs test. Means/std/mask verified through actual circuit.fit_constants via run1.prepare_runs. Std uses ddof=0. Strict < and requested <= masks agree.

| Column | Training mean | Training std | Constant | All-30 min | All-30 max |
|---:|---:|---:|---|---:|---:|
| 0 | 0.03098780829172549 | 0.0015688420035938112 | False | 0.028062287803305663 | 0.033238250373682776 |
| 1 | 0.037516680645774424 | 0.00079546572501408704 | False | 0.035962139238866486 | 0.038563541971926796 |
| 2 | 1.9999999999999964 | 2.7916461737648472e-15 | True | 1.9999999999999891 | 2.0000000000000018 |
| 3 | 1.9999999999999984 | 3.0075912366101663e-15 | True | 1.9999999999999909 | 2.000000000000004 |
| 4 | 2.0000000000000013 | 2.031839577982629e-15 | True | 1.9999999999999973 | 2.0000000000000049 |
| 5 | 2.0000000000000031 | 2.4937629136492327e-15 | True | 2.0000000000000004 | 2.0000000000000093 |

All 30 descriptor files were hash-verified unchanged in this rerun. No scientific code/data changes, electronic-structure solvers, or training.
