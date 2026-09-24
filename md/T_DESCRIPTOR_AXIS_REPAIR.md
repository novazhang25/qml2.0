# Methods 4.1 T axis repair — PASS

Completed 2026-09-24. The corrected dataset is
`/Users/novaz/Desktop/qml2.0/results/descriptor-v2-methods-t`.
The original `results/descriptor` dataset remains unchanged.
The numerical evidence, per-record shapes, residuals, and source/output hashes
are in `results/descriptor-v2-methods-t/t_repair_audit.json`.

## Scientific change

The saved rank-four tensor remains in PySCF/raw ordering:

```text
G[p,q,r,s] = Lambda_Methods[p,r,q,s]
T[i,j,k] = Lambda_Methods[i,j,k,k] = G[i,k,j,k]
```

Old extraction:

```python
T = np.diagonal(L, axis1=2, axis2=3).copy()
```

Corrected extraction:

```python
lambda_methods = L.transpose(0, 2, 1, 3)
T = np.diagonal(lambda_methods, axis1=2, axis2=3).copy()
```

Here `L` is the valence block of the saved full-AO `Lambda_L`. The transpose
is a temporary view used for extraction; stored rank-four tensors are unchanged.
The third index is retained, not summed. Signed values are retained.
`T_munulambda` still selects lexicographic valence-local `i < j < k`.

## Changed files and functions

- `codes/descriptor.py`: `extract_descriptors` corrects T indexing; `parse_args`
  defaults to the new versioned output; `run` records the T mapping and refuses
  to resume/relabel a legacy manifest with the wrong or unknown mapping.
- `codes/descriptor.py` repair mode (merged from the former standalone repair script): `audit_record` validates the saved full
  tensors and exact T indexing; `patch_archive` replaces only the two T array
  payloads; `repair_dataset` performs complete preflight, safe versioned output,
  and source/non-T preservation checks. It calls no solvers or RDM builders.
- `tests/methods41_reference.py`: `extract_descriptors` and the independent
  explicit-index check in `validate_record` now use the Methods mapping. The
  indexing-only check has zero tolerance.
- `tests/test_descriptor.py`: nonsymmetric exact-mapping regression, real H2O
  direct-loop check, unchanged non-T assertions, corrected fixture path, and
  legacy-manifest rejection tests.
- `tests/test_methods41.py`: reference extraction expectations use the correct
  raw axes and assert that the old mapping differs.
- `tests/test_methods41_production.py`: the obsolete entry point for a removed
  producer reuses the current authoritative descriptor tests.
- `tests/test_repair_t_descriptors.py` (new): seven focused migration tests.
- `md/DESCRIPTOR.md`, `md/METHODS41.md`, and
  `md/METHODS41_VALIDATED_REFERENCE.md`: corrected only the T mapping and related
  dataset/versioning instructions. This report is `md/T_DESCRIPTOR_AXIS_REPAIR.md`.

No changes were made to RHF, MP2 amplitudes, Option-3 RDM construction, raw RDM
ordering, correlation space, Lowdin transformation, valence selection, circuit
architecture, training, targets, populations, cutoff/readout definitions, or
non-T scaling rules. Run 1 implementation files were not edited.

## Validation and regeneration

| Check | Result |
|---|---:|
| Geometry records audited | 270 (9 molecules, 30 each) |
| Geometry records repaired in new output | 270 |
| Missing usable full rank-four cumulants | 0 |
| Failed records | 0 |
| Explicit loop versus vectorized T, maximum absolute error | 0, exact for all 270 |
| Packed T selection versus direct-loop selection | 0, exact for all 270 |
| Non-T NPY payloads verified byte-for-byte unchanged | 25,380 (94 per record) |
| Original source files hash-verified unchanged | 550 |
| Production solver calls during migration | 0 |
| Production RDM regenerations during migration | 0 |

Every source record passed the existing strict absolute RDM tolerance `1e-8`.
The full consistent MO/AO RDM dimensions, real/finite values, particle and pair
counts, contraction identities, symmetries, MO-to-AO transformation, RHF
disconnected term, cumulant subtraction, saved aliases, and full-AO Lowdin
transformation were checked. Maximum MO/AO contraction residuals were
`9.947598300641403e-14` / `9.237055564881302e-14`. Maximum pair-number residual
was `1.1368683772161603e-12`. Saved cumulant subtraction and Lowdin transformation
reproduced exactly.

For every geometry, including H2O/001, the independent comparison used:

```python
T_loop = np.empty((n, n, n), dtype=lambda_raw.dtype)
for i in range(n):
    for j in range(n):
        for k in range(n):
            T_loop[i, j, k] = lambda_raw[i, k, j, k]
np.testing.assert_array_equal(T_vectorized, T_loop)
```

After writing, an independent read-back repeated the raw NPY payload checks,
source hashes, and literal loops, and compared selection against the actual
`circuit.triple_addresses` and `circuit.strict_upper_triple_values` functions.
All checks passed exactly. Only `T_full.npy` and `T_munulambda.npy` changed in
the NPZ archives. The raw tensor, dtype, shapes, ordering, `P_mu`, `F_mu`,
`P_munu`, energies, geometry, and all other NPZ fields retain their original
byte representation. New manifest/sidecar provenance identifies the T repair;
historical NPZ `validation_performed` metadata was intentionally preserved.

| Molecule | T_full shape | T_munulambda shape |
|---|---|---|
| LiH | (5,5,5) | (10,) |
| HF | (5,5,5) | (10,) |
| BeH2 | (6,6,6) | (20,) |
| H2O | (6,6,6) | (20,) |
| H2S | (6,6,6) | (20,) |
| NH3 | (7,7,7) | (35,) |
| N2 | (8,8,8) | (56,) |
| CO | (8,8,8) | (56,) |
| H2O2 | (10,10,10) | (120,) |

## Old versus corrected T

Mean absolute difference is over all scalar entries in the named field across
the molecule's 30 geometries. No geometry is unchanged, either exactly or at
absolute tolerance `1e-8`, for either field. The comparison did not influence
the scientific definition.

| Molecule | T_full max abs | T_full mean abs | Packed T max abs | Packed T mean abs |
|---|---:|---:|---:|---:|
| LiH | 0.1322956503 | 0.0077102030 | 0.0815984745 | 0.0071906523 |
| HF | 0.2137961627 | 0.0079012382 | 0.0431702021 | 0.0058910135 |
| BeH2 | 0.0843818306 | 0.0080341201 | 0.0657819212 | 0.0066783760 |
| H2O | 0.1252949381 | 0.0116834199 | 0.0621789682 | 0.0106601197 |
| H2S | 0.1074185892 | 0.0105895939 | 0.0701340039 | 0.0106628359 |
| NH3 | 0.1257483796 | 0.0104620492 | 0.0414096067 | 0.0127858367 |
| N2 | 0.3336982220 | 0.0150264879 | 0.1316995723 | 0.0088248254 |
| CO | 0.2664343708 | 0.0134574104 | 0.0690860889 | 0.0077834848 |
| H2O2 | 0.2387977042 | 0.0075133416 | 0.0575400109 | 0.0067864974 |

## Tests and reproduction

The repair implementation has since been merged into `codes/descriptor.py`
under `--repair-t`; the standalone repair script was removed. After the merge,
all 60 focused tests pass, including three additional tests for repair-mode
dispatch, defaults/audit-only behavior, and incompatible option rejection.
The merge did not regenerate or modify either scientific dataset.

57 distinct focused tests passed: 8 current descriptor tests, 30 Methods
reference tests, 12 consistent-RDM tests, and 7 migration tests. The legacy
test entry point reuses the current descriptor suite and is not counted again.
An in-memory restoration of the old mapping made both production/reference
regression tests fail. The migration suite also explicitly rejects that mapping.
Existing numerical assertions were not weakened; exact indexing is checked at
zero tolerance. Tests exercise small synthetic/fixture density algebra and do
not run production electronic-structure calculations.

Run the focused suite from any directory:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/novaz/Desktop/qml2.0/tests /Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python -B -m unittest test_descriptor test_methods41 test_methods41_consistent test_repair_t_descriptors -q
```

The original standalone migration was run successfully; its equivalent command after merging into `descriptor.py` is below. Repeating it refuses
to overwrite the now-existing corrected dataset. Use an explicitly new
`--output-dir` only when another copy is needed.

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python -B /Users/novaz/Desktop/qml2.0/codes/descriptor.py --repair-t --source-dir /Users/novaz/Desktop/qml2.0/results/descriptor --output-dir /Users/novaz/Desktop/qml2.0/results/descriptor-v2-methods-t
```

## Downstream reruns and remaining blockers

The repair itself is complete. No 32-seed production training was launched.
MB-3, Random-T, Shuffled-T, any other T-dependent controls, their fitted T scales,
and resulting summaries must use the repaired data. FG, FE, MB-1, MB-2, and
rank-four-only pair controls do not require reruns solely because of this bug.

There are currently no compatible executable rerun commands for the requested
T models in this repository:

| Requested rerun | Current repository support | Exact runnable command |
|---|---|---|
| MB-3 | Circuit and scaler exist, but Run 1 does not load T or launch MB-3 | Unavailable |
| Random-T | No compatible current implementation/launcher | Unavailable |
| Shuffled-T | No compatible current implementation/launcher | Unavailable |
| Other T-dependent controls | None located | Unavailable |

`codes/run1.py` restricts `METHODS` to `FG`, `FE`, and `MB-1`, sets
`ProcessedSample.t_lowdin=None`, and has no model/control selection flags.
It also requires an explicit supplied train/validation/test split manifest;
none was located. The legacy controls in sibling `encoding_qml` use UHF data
and target conventions and are not compatible drop-in commands. No such
command has been invented or executed.

The required configuration for a future compatible launcher is the exact new
dataset path above, the unchanged original split IDs/seeds/protocol, and the
existing train-only T scaling rule: linear-interpolation `p95(abs(T))` followed
by `beta=atanh(0.9)/p95`. Refit that scale from corrected training T values;
do not reuse an old fitted T scale. No control randomization/shuffling policy
was chosen by this repair.

Producer triples remain in saved valence-local order. A future MB-3 loader must
map all three T_full axes to its existing model AO order before selecting
triples. In particular, BeH2 saved valence indices are `[1,2,3,4,5,6]`, while
Run 1's model order is `[5,1,2,3,4,6]`; the packed vector cannot be copied
blindly. This preserves the existing downstream addressing convention and
does not change producer valence selection.
