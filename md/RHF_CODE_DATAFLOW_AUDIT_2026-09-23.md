# RHF code-level data-flow audit

Audit date: 2026-09-23. Workspace: `/Users/novaz/Desktop/qml2.0`.

This report was derived from the current Python source, the installed PySCF source, and saved numerical arrays. Existing reports were not used as evidence for the implementation. Only this new report is an output of the audit. No workflow entry point, RHF solve, geometry optimization, nuclear Hessian calculation, production plot regeneration, or result-file update was executed. Read-only arithmetic re-evaluated saved contractions and scalar expressions; geometry-only calls reconstructed finite-difference paths without invoking an electronic solver.

**Finding:** the saved tangent/Hessian contractions, masses, frequencies, quantum energies and n=0/n=1 decisions are internally consistent with the inspected operations. The implementation is a constrained one-dimensional harmonic model. Its finite-difference tangent, residual path-curvature term, fixed effective mass, and literal zero-energy decision must be distinguished from exact full-dimensional quantum vibrational dynamics. The absolute-zero sign test remains explicitly physically unvalidated by the saved energy-reference definition.

**Correction to the prior explanation:** the current plot source and saved SVG show **30 scan points plus the separately saved equilibrium point**, hence 31 orange data markers per molecule. The numerical scan still contains exactly 30 records per molecule and the plotted-points CSV still has 270 rows overall. Section K identifies the source statements and saved evidence.

## Reading conventions and actual execution order

- `N` is the number of nuclei. Cartesian arrays have rows in the saved `symbols` order and columns `(x,y,z)`. LiH has `N=2`; H₂O has `N=3`.
- A scalar Python number is labelled **scalar**; a NumPy scalar has conceptual shape `()`. JSON arrays retain the listed dimensions as nested lists. No NPZ handoff is used in this workflow.
- `H[a,b,x,y]` has shape `(N,N,3,3)`, **not** `(N,3,N,3)`. A matrix indexed by `(a,x),(b,y)` requires `H.transpose(0,2,1,3).reshape(3*N,3*N)`. This matrix conversion is used only for explanation in this audit, not by the production projection.
- \(r_e\) or source `s_eq` is a scalar equilibrium bond coordinate in Å. \(\mathbf R_e\) or `reference` is the `(N,3)` equilibrium Cartesian geometry. \(E_{\rm re}\) is a scalar absolute molecular RHF energy in Ha. These are three different objects.
- **Definition** means a mathematical quantity; **producer** means executed source arithmetic that originally constructed it; **saved** means the persisted result; **check** means later arithmetic that can reject inconsistent inputs but does not replace the saved value.
- Numerical examples use saved full-precision values. Displayed matrices may be rounded as stated; matrix arithmetic checks use the unrounded saved arrays.

The conceptual path requested in the question is useful, but source execution is specifically:

```text
Part1 RHF optimization -> equilibrium CSV energy and XYZ coordinates
Part2 saved XYZ -> R(q) -> centered finite-difference t
Part2 equilibrium RHF reevaluation -> analytic nuclear H
Part2 t,H -> k; t,masses -> mu_eff
Part2 k,mu_eff -> omega -> oscillator length -> A0,A1 -> both bond ranges
Part3 saved original E_re + saved omega -> Q -> E_n -> absolute total energies
Part3 reads saved A/ranges, checks (k A_n²)/2 = E_n, selects0/1/None
Scanner reads selected range, reconstructs R(q), solves30 fixed geometries
Plot reads saved30 RHF points + saved E_re, evaluates harmonic drawing formula
```

In particular, **Part2 constructs A/ranges before Part3 constructs E_n**. Section H proves the equivalence between the actual oscillator-length expression and `sqrt(2*E_n/k)`; it does not pretend the latter is the source expression.

The numerical constants used by the saved harmonic producer and standalone Part3 are the following values, read from `runtime.constants` in the harmonic JSON. Source assignment is from installed `pyscf.data.nist` at harmonic-source lines43–48; serialization is at768–769.

| Saved key / source constant | Actual numerical value | Physical role |
|---|---:|---|
| `Bohr_A` / `BOHR_A` | 0.52917721092 | Å per Bohr |
| `Bohr_m` / `BOHR_M` | 5.2917721092e-11 | m per Bohr |
| `Hartree_J` / `EH_J` | 4.359744644911914e-18 | J per Ha |
| `amu_kg` / `AMU_KG` | 1.660539040427164e-27 | kg per amu |
| `hbar_J_s` / `HBAR` | 1.0545718001391127e-34 | J·s |
| `c_m_per_s` / `C_MS` | 299792458 | m/s |

These values are reported as actually used; the audit does not update constants. The stored numerical environment is Python 3.13.2, PySCF 2.14.0, NumPy 2.5.3, SciPy 1.18.1, one PySCF thread, RHF/STO-3G for the current nine molecules.

Primary saved inputs:

- [Equilibrium CSV](/Users/novaz/Desktop/qml2.0/results/rhf_geometries/rhf_equilibrium_summary.csv)
- [Harmonic JSON, including geometries, masses, tangent and full Hessian](/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json)
- [Part3 JSON](/Users/novaz/Desktop/qml2.0/results/bond_length_part3/vibrational_levels.json)
- [30-point scan JSON](/Users/novaz/Desktop/qml2.0/results/rhf_30_point_scans/rhf_scan_30.json)

Every lettered section below follows the requested eight-part structure. A–I contain the primary code/equation derivation; J–K trace the downstream sample and figure consumers.

## A. R(q): how the geometry path is constructed

### 1. PHYSICAL / MATHEMATICAL QUANTITY

The actual path is an `(N,3)` Cartesian array in Å:

\[
R(q)=\operatorname{Align}_{m,R_e}(R_{\rm raw}(q)).
\]

Here q is a scalar displacement in Å. For a participating central-atom–H bond, the source defines \(\ell_h=\|R_{e,h}-R_{e,c}\|\), \(u_h=(R_{e,h}-R_{e,c})/\ell_h\), and \(R_{{\rm raw},h}(q)=R_{e,c}+(\ell_h+q)u_h\). Every selected bond receives the same **additive** q, while its equilibrium ray stays fixed. This is a prescribed collective stretch, not a normal-mode displacement or a geometry optimization at every q.

### 2. EXACT SOURCE IMPLEMENTATION

File: [new_rhf_harmonic_bond_ranges.py](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:199). `StretchPath.__init__`: lines 199–234; `raw`: 236–254; `__call__`: 256–257; `center`/`align`: 175–187.

[tests/new_rhf_harmonic_bond_ranges.py, lines 231–257](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:231)

```python
    self.lengths = np.array([bond(self.ref, *pair) for pair in self.stretches])
    if np.any(self.lengths < 1e-8):
        raise ValueError("A participating equilibrium bond has zero length.")
    self.s_eq = float(np.mean(self.lengths))

def raw(self, q):
    if np.any(self.lengths + q <= 0):
        raise ValueError("A stretch displacement would produce a nonpositive bond length.")
    coords = self.ref.copy()
    if self.name in DIATOMICS:
        axis = (self.ref[1] - self.ref[0]) / self.lengths[0]
        coords[1] = coords[0] + (self.lengths[0] + q) * axis
    elif self.name in STARS:
        for (heavy, h), length in zip(self.stretches, self.lengths):
            ray = (self.ref[h] - self.ref[heavy]) / length
            coords[h] = coords[heavy] + (length + q) * ray
    else:
        o1, o2 = self.oxygens
        h1, h2 = self.peroxide_h
        axis = (self.ref[o2] - self.ref[o1]) / self.lengths[0]
        coords[o2] = coords[o1] + (self.lengths[0] + q) * axis
        coords[h1] = coords[o1] + (self.ref[h1] - self.ref[o1])
        coords[h2] = coords[o2] + (self.ref[h2] - self.ref[o2])
    return coords

def __call__(self, q):
    return align(self.raw(q), self.ref, self.masses)[0]
```

[tests/new_rhf_harmonic_bond_ranges.py, lines 175–187](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:175)

```python
def center(coords, masses):
    return np.sum(masses[:, None] * coords, axis=0) / masses.sum()


def align(coords, reference, masses):
    """Mass-weighted, proper (det=+1) row-vector Kabsch/Eckart alignment."""
    ref_com = center(reference, masses)
    x, y = coords - center(coords, masses), reference - ref_com
    u, _, vt = np.linalg.svd(x.T @ (masses[:, None] * y))
    proper = np.eye(3)
    proper[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ proper @ vt
    return x @ rotation + ref_com, rotation
```

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Symbol | Shape | Units |
|---|---|---|---|
| `self.ref`, `reference` | \(R_e\) | `(N,3)` | Å |
| `self.stretches` | selected atom pairs | list of P integer pairs | atom indices |
| `self.lengths` | \(\ell_i\) | `(P,)` | Å |
| `self.s_eq` | \(s_e=P^{-1}\sum_i\ell_i\) | scalar | Å |
| `q` | q | scalar | Å |
| `axis`, `ray` | \(u_i\) | `(3,)` | dimensionless |
| `coords` | \(R_{raw}(q)\) | `(N,3)` | Å |
| `masses` | \(m_A\) | `(N,)` | amu |
| `masses[:,None]` | broadcast mass column | `(N,1)` | amu |
| conceptual atom-space mass matrix, not allocated | \(W=\operatorname{diag}(m_1,\ldots,m_N)\) | `(N,N)` | amu |
| `ref_com`, `center(coords,masses)` | \(C_e,C(q)\) | `(3,)` | Å |
| `x`,`y` | X,Y | `(N,3)` | Å |
| `x.T @ (masses[:,None]*y)` | \(X^TWY\) | `(3,3)` | amu Å² |
| `u`,`vt`,`proper`,`rotation` | \(U,V^T,D,O\) | `(3,3)` | dimensionless |
| discarded SVD values `_` | singular values | `(3,)` | amu Å² |
| returned aligned array | R(q) | `(N,3)` | Å |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

* `bond(R,i,j)=norm(R[j]-R[i])` (152–153): \(\ell_{ij}=\sqrt{\sum_\alpha(R_{j\alpha}-R_{i\alpha})^2}\).
* `(ref[h]-ref[c])/length`: \(u_{h\alpha}=(R_{e,h\alpha}-R_{e,c\alpha})/\ell_h\).
* `coords[c]+(length+q)*ray`: \(R_{raw,h\alpha}=R_{e,c\alpha}+(\ell_h+q)u_{h\alpha}=R_{e,h\alpha}+qu_{h\alpha}\).
* `np.mean(lengths)`: \(s_e=P^{-1}\sum_i\ell_i\), the mean, not an arbitrarily selected bond length.
* `sum(m[:,None]*coords,axis=0)/m.sum()`: \(C_\alpha=\sum_A m_AR_{A\alpha}/\sum_A m_A\).
* `x.T @ (m[:,None]*y)`: \(S_{\alpha\beta}=\sum_A X_{A\alpha}m_AY_{A\beta}\). SVD produces \(S=U\Sigma V^T\).
* `proper[-1,-1]=...`: \(D=\operatorname{diag}(1,1,d)\), d=1 if \(\det(UV^T)\ge0\), otherwise −1; this removes a possible reflection.
* `rotation=u @ proper @ vt`: \(O_{\alpha\beta}=\sum_{ij}U_{\alpha i}D_{ij}V^T_{j\beta}\).
* `x @ rotation + ref_com`: \(R_{A\beta}(q)=\sum_\alpha X_{A\alpha}O_{\alpha\beta}+C_{e,\beta}\). These are **row-vector** coordinates. Alignment preserves internal distances, returns the equilibrium COM, and removes a best-fit rigid rotation.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

[tests/molecules_rhf.py:76](/Users/novaz/Desktop/qml2.0/tests/molecules_rhf.py:76) `optimize` converts the initial geometry from Å to Bohr (78), subtracts its **unweighted** coordinate mean (79), and flattens `(N,3)` to `(3N,)`. `energy_gradient` reshapes to `(N,3)` (84), builds an RHF molecule with `unit="Bohr"` (85–94), and returns `float(energy), gradient.ravel()` (104). BFGS consumes the pair (106–107). `result.x.reshape(-1,3)*BOHR` returns Å (110). Lines 188–193 write XYZ coordinates to **12 decimal places**. Thus the subsequent path uses the rounded saved XYZ, not the original in-memory optimizer array. CSV `energy_hartree` is written from `result.fun` (199).

The original equilibrium producer is quoted directly here:

[tests/molecules_rhf.py, lines 76–97](/Users/novaz/Desktop/qml2.0/tests/molecules_rhf.py:76)

```python
def optimize(atoms, args, label):
    symbols = [atom[0] for atom in atoms]
    initial = np.array([atom[1] for atom in atoms], dtype=float) / BOHR
    initial -= initial.mean(axis=0)
    evaluations = 0

    def energy_gradient(flat):
        nonlocal evaluations
        coords = flat.reshape(-1, 3)
        mol = gto.M(atom=list(zip(symbols, coords)), basis=args.basis,
                    unit="Bohr", charge=0, spin=0, symmetry=False,
                    verbose=0, max_memory=args.memory_mb)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        mf.diis_space = 12
        mf.direct_scf_tol = 1e-14
        energy = mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF SCF did not converge")
        gradient = mf.nuc_grad_method().kernel()
```

[tests/molecules_rhf.py, lines 104–112](/Users/novaz/Desktop/qml2.0/tests/molecules_rhf.py:104)

```python
    return float(energy), gradient.ravel()

result = minimize(energy_gradient, initial.ravel(), jac=True, method="BFGS",
                  options={"gtol": args.gtol, "maxiter": args.maxiter})
max_gradient = float(np.max(np.abs(result.jac)))
return {"result": result, "symbols": symbols,
        "coords": result.x.reshape(-1, 3) * BOHR,
        "gradient": max_gradient, "evaluations": evaluations,
        "converged": bool(max_gradient <= args.gtol)}
```

`load_equilibrium` (harmonic source 118–148) reads atom order and XYZ into `(N,3)`, checks method/basis/status/composition and labels units Å. `calculate:432–440` supplies the geometry and isotope-average masses to `StretchPath`. Harmonic code does not optimize it again.

Raw movement by molecule:

| Branch | Atoms/components moved before alignment |
|---|---|
| LiH/N₂/CO/HF | atom 0 fixed; atom 1 moves along equilibrium bond axis |
| BeH₂/H₂O/NH₃/H₂S | heavy atom fixed; each H moves along its own heavy–H ray; every length increment is q and all angles stay fixed |
| H₂O₂ | first O and attached H fixed; second O and attached H translate together along O–O; OH lengths, OOH angles and torsion fixed |

Peroxide chooses the one-to-one O–H assignment with minimum summed O–H distances (`__init__:212–227`); a tie within `1e-8 Å` raises an error. **Alignment can move atoms that were fixed in raw geometry**, and in general orientation all Cartesian components can change. There is no clipping: nonpositive stretched lengths raise an error (237–238); equilibrium selected-length spread `>1e-6 Å` produces a validation failure (275–276), without forcing lengths equal.

### 6. UNITS TRACE

```text
ref[h]-ref[c]             : Å
norm(vector)             : sqrt(sum(Å²)) = Å
ray=vector/length        : Å/Å = 1
(length+q)*ray           : (Å+Å)*1 = Å
coords[c]+(...)          : Å+Å = Å
m[:,None]*coords         : amu Å
sum(...)/sum(m)          : (amu Å)/amu = Å
x=coords-C, y=ref-Ce      : Å
X.T @ (W*Y)              : Å*(amu Å) = amu Å²
SVD U,V; det correction  : dimensionless
x @ rotation + ref_com   : Å*1+Å = Å
```

### 7. SAVED DATA / HANDOFF

```text
result.x*BOHR -> <molecule>_converged.xyz (12-decimal Å)
 -> load_equilibrium -> reference -> StretchPath.ref
reference -> harmonic JSON results[i].equilibrium_cartesian_A [N,3]
symbols -> results[i].symbols [N strings]
path.definition -> results[i].summary.coordinate_definition
path.s_eq -> results[i].summary.s_eq_A
XYZ/CSV hashes -> results[i].input.xyz_sha256 / metadata_sha256
```

Harmonic `write_outputs:726–735` serializes NumPy arrays through `jsonable:92–103`. Current saved file: [new_rhf_harmonic_bond_ranges.json](/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json). Part 3 copies the scalar coordinate and definition to `r_e_A`/`coordinate` (`codes/bond_length_part3.py:116–118`); it does not construct R(q). Scanner `prepare_plans:151–165` invokes `tests/rhf_bound_level_selection.py:90–110` `prepare_path`, reconstructs the geometry path from the saved arrays, validates it, and calls `path(q)`.

### 8. NUMERICAL WALKTHROUGH

The numbers below came from saved JSON and a geometry-only arithmetic re-evaluation: no SCF, Hessian, or optimization was run. Harmonic source SHA-256 exactly equals saved `runtime.script_sha256`: `e6687127ab5752b7a203cab27f177572774e85e7a71acec9f4b617e699b17bd9`.

LiH, atom order `[Li,H]`:

```text
R_e [Å] = [[0,0,-0.755405926516], [0,0,0.755405926516]]
r_e = norm([0,0,1.510811853032]) = 1.510811853032 Å
      [r_e=|R_H-R_Li|]
axis = [0,0,1.510811853032]/1.510811853032 = [0,0,1]
       [u=(R_H-R_Li)/r_e]
raw_H,z(+0.0002) = -0.755405926516+(1.510811853032+0.0002)*1
                 = 0.755605926516 Å [R_raw,H=R_Li+(r_e+q)u]

raw(-0.0002) = [[0,0,-0.755405926516], [0,0,0.755205926516]]
raw(+0.0002) = [[0,0,-0.755405926516], [0,0,0.755605926516]]
R(-0.0002)   = [[0,0,-0.7553805616443342], [0,0,0.7552312913876659]]
R(+0.0002)   = [[0,0,-0.7554312913876660], [0,0,0.7555805616443340]]
```

Rotations are identity. `COM_z=(6.94*(-0.755405926516)+1.008*0.755405926516)/7.948=-0.5637981826991586 Å`, implementing \(C_z=\sum mR_z/\sum m\). Positive-q raw COM is `-0.5637728178274927 Å`; alignment subtracts approximately `0.0000253648716659 Å` from each raw z coordinate: \(R=R_{raw}-C(q)+C_e\).

H₂O, atom order `[O,H,H]`:

```text
R_e [Å] = [[0,0,-0.423868772177],
           [ 0.758079720250,0,0.211934386089],
           [-0.758079720250,0,0.211934386089]]
length = norm([0.758079720250,0,0.635803158266])
       = 0.9894091763852498 Å [ell=|R_H-R_O|]
ray_H1 = [0.758079720250,0,0.635803158266]/0.9894091763852498
       = [0.7661943494597464,0,0.6426089159480718] [u_H=(R_H-R_O)/ell]
ray_H2 = [-0.7661943494597464,0,0.6426089159480718]

raw(-0.0002) = [[0,0,-0.4238687721770000],
                [ 0.7579264813801081,0,0.2118058643058103],
                [-0.7579264813801081,0,0.2118058643058103]]
raw(+0.0002) = [[0,0,-0.4238687721770000],
                [ 0.7582329591198919,0,0.2120629078721896],
                [-0.7582329591198919,0,0.2120629078721896]]
R(-0.0002)   = [[0,0,-0.4238543897226613],
                [ 0.7579264813801081,0,0.2118202467601490],
                [-0.7579264813801081,0,0.2118202467601490]]
R(+0.0002)   = [[0,0,-0.4238831546313386],
                [ 0.7582329591198919,0,0.2120485254178509],
                [-0.7582329591198919,0,0.2120485254178509]]
```

Each raw H row is `R_e,H+q*ray_H`, while O stays fixed. Displayed zeros suppress SVD roundoff; B retains the actual O x derivative. The alignment translation derivative is `2*1.008*0.6426089159480718/18.015 ≈ 0.07191227169315`, implementing \(C'_z=\sum_A m_Au_{A,z}/\sum_A m_A\). Hence aligned O moves toward negative z. Equilibrium H–O–H angle is `100.02672794095223°` and stays fixed.

## B. t=dR/dq: how the tangent is actually obtained

### 1. PHYSICAL / MATHEMATICAL QUANTITY

Definition: \(t_{A\alpha}=\partial R_{A\alpha}/\partial q|_0\). Source approximation:

\[
t^{(\delta)}_{A\alpha}=\frac{R_{A\alpha}(+\delta)-R_{A\alpha}(-\delta)}{2\delta},\qquad t=t^{(0.0002\,Å)}.
\]

It is a central finite difference of the **aligned** path, not an analytic derivative, autodifferentiation, or a normal-mode eigenvector. It ordinarily has second-order truncation error in δ; these symmetric affine examples are limited primarily by roundoff. No Euclidean normalization is applied.

### 2. EXACT SOURCE IMPLEMENTATION

File: [new_rhf_harmonic_bond_ranges.py](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:272), `TANGENT_STEPS:49`, `StretchPath.validate:272–323`, `calculate:438–440`.

[tests/new_rhf_harmonic_bond_ranges.py, lines 49–49](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:49)

```python
TANGENT_STEPS = (1e-3, 5e-4, 2e-4)
```

[tests/new_rhf_harmonic_bond_ranges.py, lines 277–283](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:277)

```python
for delta in TANGENT_STEPS:
    plus, minus = self(delta), self(-delta)
    t = (plus - minus) / (2 * delta)
    tangents.append(t)
    residual_com = np.linalg.norm(center(t, self.masses))
    residual_rot = np.linalg.norm(np.sum(
        self.masses[:, None] * np.cross(self.ref - center(self.ref, self.masses), t), axis=0))
```

[tests/new_rhf_harmonic_bond_ranges.py, lines 319–323](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:319)

```python
convergence = [float(np.max(np.abs(t - tangents[-1]))) for t in tangents]
if max(convergence) > TOL["tangent_convergence_max"]:
    failures.append("The Cartesian tangent did not converge with displacement step.")
return tangents[-1], {"equilibrium_internals": equilibrium, "samples": samples,
                      "tangent_max_differences_from_finest": convergence, "failures": failures}
```

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Symbol | Shape | Units |
|---|---|---|---|
| `delta` | δ | scalar | Å |
| `plus`,`minus` | \(R(±δ)\) | `(N,3)` | Å |
| `t`,`tangent` | \(t^{(δ)},t\) | `(N,3)` | Å/Å=1 |
| `tangents` | candidate tangents | list of 3 `(N,3)` arrays | dimensionless |
| `center(t,masses)` | \(C'\) | `(3,)` | dimensionless |
| `residual_com` | \(\|C'\|\) | scalar | dimensionless |
| `cross(ref-Ce,t)` | \((R_{e,A}-C_e)×t_A\) | `(N,3)` | Å |
| `residual_rot` | rotational residual norm | scalar | amu Å |
| `convergence` | max entrywise differences | list of 3 scalars | dimensionless |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

* `(plus-minus)/(2*delta)` is \(t^{(δ)}_{A\alpha}=[R_{A\alpha}(δ)-R_{A\alpha}(-δ)]/(2δ)\), element by element. No flattening or norm division occurs.
* `center(t,m)` is \(C'_\alpha=\sum_A m_At_{A\alpha}/\sum_A m_A\); its norm checks translation removal.
* `sum(m[:,None]*cross(ref-Ce,t),axis=0)` is \(L_\gamma=\sum_{A\alpha\beta}m_A\epsilon_{\gamma\alpha\beta}(R_{e,A\alpha}-C_{e,\alpha})t_{A\beta}\). Its norm checks infinitesimal rotation removal.
* `max(abs(t-tangents[-1]))` is \(\max_{A\alpha}|t^{(δ)}_{A\alpha}-t^{(δ_{min})}_{A\alpha}|\). It is a convergence diagnostic, not extrapolation or averaging.
* `diff(p[key]['value'],m[key]['value'])/(2*delta)` (284–300) central-differences each internal coordinate. Stretched bond derivatives must be one; fixed-coordinate derivatives must be zero within tolerance. For torsions, `angular_difference(a,b)=(a-b+180)%360-180` (171–172) wraps degree differences locally.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

`self(±delta)` calls raw geometry plus SVD alignment from A, so t includes the derivative of alignment. `validate` checks all three step sizes, internal-coordinate derivatives, pair-distance preservation and proper rotation. Tolerances: tangent convergence `2e-7`; COM derivative `2e-9`; rotation residual `2e-7 amu Å`; pair-distance change `2e-12 Å`; determinant 1 with absolute tolerance `1e-12`. The finest tangent is returned directly. A path failure returns `FAIL_PATH` before RHF/Hessian calculation (`calculate:443–445`).

### 6. UNITS TRACE

```text
plus-minus                  : Å
2*delta                     : Å
(plus-minus)/(2*delta)       : Å/Å = 1
center(t,m)                 : (amu*1)/amu = 1
(ref-Ce) × t                : Å*1 = Å
sum(m*((ref-Ce) × t))        : amu Å
bond finite difference      : Å/Å = 1
angle finite difference     : degree/Å
t(delta)-t(finest)           : dimensionless
```

The same numerical t represents \(dR_{Bohr}/dq_{Bohr}\), because the common length conversion cancels: \((dR_Å/a_0)/(dq_Å/a_0)=dR_Å/dq_Å\). No Bohr conversion factor multiplies t before the Hessian projection.

### 7. SAVED DATA / HANDOFF

```text
path.validate -> tangent -> results[i].tangent_dR_dq [N,3]
 -> calculate's Hessian and mass contractions
samples[j] -> results[i].path_validation.samples[j]
 (delta_A, tangent, internal_derivatives_per_A, residuals)
convergence -> results[i].path_validation.tangent_max_differences_from_finest
```

Assignments: 438–440; diagnostic dictionaries: 315–323. Part 3 neither reads nor projects t. Scanner `prepare_path` recomputes geometry-only t to verify the same path (`tests/rhf_bound_level_selection.py:104–109`), with `np.allclose(...,atol=2e-8,rtol=0)`; it does not replace k or mass.

### 8. NUMERICAL WALKTHROUGH

LiH, using A's aligned coordinates:

```text
t_Li,z = (-0.7554312913876660 - (-0.7553805616443342))/0.0004
       = -0.12682435832939154
t_H,z  = (0.7555805616443340 - 0.7552312913876659)/0.0004
       = 0.8731756416702208
         [each is t_Az=(R_Az(+δ)-R_Az(-δ))/(2δ)]
t = [[0,0,-0.12682435832939154], [0,0,0.8731756416702208]]
```

Continuous diatomic expressions would be `-1.008/(6.94+1.008)` and `6.94/(6.94+1.008)` along z, but those analytic expressions are **not** substituted into the source. Actual step differences are `[4.440892098500626e-13,6.661338147750939e-13,0]`. Finest COM residual is `3.013023482783672e-13`, rotation residual zero. Saved and re-evaluated t agree bit for bit.

H₂O:

```text
t_O,z = (-0.4238831546313386 - (-0.42385438972266126))/0.0004
      = -0.07191227169328629
t_H1,x = (0.7582329591198919-0.7579264813801081)/0.0004
       = 0.7661943494594037
t_H1,z = (0.2120485254178509-0.21182024676014899)/0.0004
       = 0.5706966442547978
         [each is t_Aα=(R_Aα(+δ)-R_Aα(-δ))/(2δ)]
t = [[-2.7103567011263198e-15,0,-0.07191227169328629],
     [ 0.7661943494594037,    0, 0.5706966442547978],
     [-0.7661943494594037,    0, 0.5706966442547978]]
```

O x is SVD/finite-difference roundoff; y is identically zero. Raw H z derivative was `0.6426089159480841`; alignment changes it as shown. Both O–H derivatives are `0.9999999999996123`; angle derivative `-3.552713678800501e-11 degree/Å`; finest COM residual `1.7847776226386743e-13`; rotation residual `3.085305911646558e-15 amu Å`. Step differences `[3.3306690738754696e-13,1.6653345369377348e-13,0]`. Saved and re-evaluated t agree bit for bit.

## C. H: how the Cartesian Hessian is represented

### 1. PHYSICAL / MATHEMATICAL QUANTITY

The object actually projected is the **unweighted Cartesian Hessian of total RHF molecular energy**:

\[
H_{abxy}=\frac{\partial^2 E_{\rm RHF}}{\partial R_{ax}\partial R_{by}},\quad
a,b=0,\ldots,N-1,\quad x,y=0,1,2.
\]

Here `x=0,1,2` means Cartesian x,y,z; `a,b` preserve XYZ atom order. The raw shape is `(N,N,3,3)`, not `(3N,3N)`. Nuclear repulsion is included.

### 2. EXACT SOURCE IMPLEMENTATION

File: [tests/new_rhf_harmonic_bond_ranges.py](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:462), `calculate`, lines 462–470:

```python
hessobj = mf.Hessian()
hessobj.max_cycle = 100
# The installed RHF Hessian solver uses mf.conv_tol_cpscf for CPHF.
mf.conv_tol_cpscf = 1e-12
hessian = hessobj.kernel()
if hessian.shape != (mol.natm, mol.natm, 3, 3) or not np.isfinite(hessian).all():
    raise RuntimeError("Analytic Hessian has an invalid shape or nonfinite values.")
symmetry_error = float(np.max(np.abs(hessian - hessian.transpose(1, 0, 3, 2))))
translation_error = float(np.max(np.abs(hessian.sum(axis=1))))
```

`calculate`, lines 473–484 and 495–498:

```python
probe = np.sin(np.arange(1, mol.natm * 3 + 1)).reshape(mol.natm, 3)
probe /= np.linalg.norm(probe)
epsilon_bohr = 1e-4
probe_gradients = []
for sign in (1, -1):
    displaced = reference + sign * epsilon_bohr * BOHR_A * probe
    check_mf = run_rhf(symbols, displaced, basis, mf.make_rdm1())
    probe_gradients.append(check_mf.nuc_grad_method().kernel())
numerical_response = (probe_gradients[0] - probe_gradients[1]) / (2 * epsilon_bohr)
analytic_response = np.einsum("abxy,by->ax", hessian, probe)
probe_error = float(np.max(np.abs(numerical_response - analytic_response)))
probe_scale = float(np.max(np.abs(analytic_response)))
```

The installed implementation is [pyscf/hessian/rhf.py](/Users/novaz/Desktop/qml2.0/.venv-rhf/lib/python3.13/site-packages/pyscf/hessian/rhf.py:556), `HessianBase.kernel`, lines 568–572:

```python
de = self.hess_elec(mo_energy, mo_coeff, mo_occ, atmlst=atmlst)
self.de = de + self.hess_nuc(self.mol, atmlst=atmlst)
if self.base.do_disp():
    self.de += self.get_dispersion()
return self.de
```

No dispersion is configured by this workflow. Installed `hess_elec`, lines 52–60, forms partial electronic derivatives plus the first-order orbital response (`solve_mo1`); lines 79–83 add the response terms. At lines 85–86 PySCF constructs the opposite triangle as `de2[j0,i0] = de2[i0,j0].T`. `hess_nuc`, lines 362–383, allocates `(N,N,3,3)` and differentiates nuclear Coulomb repulsion using `mol.atom_coord(i)`.

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Symbol | Shape | Units |
|---|---|---|---|
| `reference` | \(R_e\) | `(N,3)` | Å |
| `hessian` | \(H_{abxy}\) | `(N,N,3,3)` | Ha/Bohr² |
| `hessian.transpose(1,0,3,2)` | \(H_{bayx}\) | `(N,N,3,3)` | Ha/Bohr² |
| `hessian.sum(axis=1)` | \(\sum_bH_{abxy}\) | `(N,3,3)` | Ha/Bohr² |
| `probe` | \(v_{by}\) | `(N,3)` | dimensionless |
| `epsilon_bohr` | \(\epsilon\) | scalar | Bohr |
| each `probe_gradients[i]` | \(g(R_e\pm\epsilon v)\) | `(N,3)` | Ha/Bohr |
| `analytic_response`, `numerical_response` | \(Hv\), directional derivative of \(g\) | `(N,3)` | Ha/Bohr² |
| `symmetry_error`, `translation_error`, `probe_error`, `probe_scale` | corresponding maximum absolute entries | scalar | Ha/Bohr² |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

The source deliberately keeps the tensor ordering:

```text
Python: hessian.transpose(1,0,3,2)
Index:  output[a,b,x,y] = hessian[b,a,y,x]
Math:   exchange the two complete nuclear-coordinate indices.

Python: max(abs(hessian - exchanged_hessian))
Index:  max_abxy |H_abxy - H_bayx|
Math:   test H = H^T; do not replace H by (H+H^T)/2.

Python: hessian.sum(axis=1)
Index:  sum_b H_abxy
Math:   response to uniform Cartesian translation in direction y.

Python: np.sin(np.arange(1,3*N+1)).reshape(N,3); probe /= norm(probe)
Index:  v_ax = sin(3*a+x+1)/sqrt(sum_b,y sin(3*b+y+1)^2)
Math:   a unit-length generic Cartesian test direction, unrelated to the physical tangent.

Python: displaced = reference + sign * epsilon_bohr * BOHR_A * probe
Index:  R_ax^±[Å] = R_e,ax[Å] ± epsilon[Bohr] * a0[Å/Bohr] * v_ax
Math:   R^± = R_e ± epsilon v, after converting the displacement to Å.

Python: (probe_gradients[0]-probe_gradients[1])/(2*epsilon_bohr)
Index:  [g_ax(R^+) - g_ax(R^-)]/(2*epsilon)
Math:   centered finite-difference approximation to (Hv)_ax, error O(epsilon²).

Python: np.einsum("abxy,by->ax", hessian, probe)
Index:  response_ax = sum_b,y H_abxy v_by
Math:   H v.
```

Acceptance uses `symmetry_error <= 1e-8`, `translation_error <= 1e-7`, and `probe_error <= 2e-7 + 2e-5*probe_scale`, all absolute quantities in Ha/Bohr². Constants are in the literal `TOL` dictionary, lines 63–66. The reported relative probe error divides by `max(probe_scale,1e-30)` (line 491); this denominator floor prevents division by zero and does not change H or the acceptance criterion.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

`calculate` creates `mol` from saved equilibrium XYZ coordinates (lines 432–435), then recomputes a tightly converged RHF solution **at that supplied geometry**, not a new optimized geometry (`run_rhf`, lines 334–354; call line 447). Nuclear gradients are generated at line 448. The Hessian is calculated from that RHF state.

`make_molecule`, lines 326–331, explicitly supplies `unit="Angstrom"`, neutral charge, spin 0, no symmetry, spherical basis functions (`cart=False`), and rejects ECP or non-all-electron input. PySCF stores coordinates in Bohr: installed [gto/mole.py](/Users/novaz/Desktop/qml2.0/.venv-rhf/lib/python3.13/site-packages/pyscf/gto/mole.py:2174) lines 2174–2183 perform the length conversion; `atom_coord`/`atom_coords`, lines 3239–3264, default to Bohr. Thus the derivatives returned by the atomic-unit electronic and nuclear Hessian routines are Ha/Bohr². The independent gradient probe verifies the convention numerically.

**No workflow reshaping, symmetrization, mass weighting, translation removal, or unit conversion is applied to H before projection.** PySCF's internal triangle-copy operation should not be confused with a workflow symmetry repair. The workflow's symmetry/translation tests report failures rather than modifying H.

To display H as an ordinary matrix, the valid audit-only transformation is `hessian.transpose(0,2,1,3).reshape(3*N,3*N)`, giving row/column order `(atom0-x, atom0-y, atom0-z, atom1-x,...)`. `hessian.reshape(3*N,3*N)` by itself would be the wrong order. The production curvature projection uses the raw tensor directly.

Mass weighting and this correct flattening occur separately **after** the scalar curvature/ranges calculation: `normal_modes` lines 396–424, called at line 555, invokes installed `thermo.harmonic_analysis`. In installed [hessian/thermo.py](/Users/novaz/Desktop/qml2.0/.venv-rhf/lib/python3.13/site-packages/pyscf/hessian/thermo.py:55), lines 55–56 create `mass_hess = einsum('pqxy,p,q->pqxy',hess,mass**-.5,mass**-.5)` and then reshape. That is a normal-mode diagnostic; it does not replace the raw H or chosen coordinate.

The workflow sets a CPHF base tolerance `mf.conv_tol_cpscf=1e-12` and `hessobj.max_cycle=100`; installed `solve_mo1`, [rhf.py:330](/Users/novaz/Desktop/qml2.0/.venv-rhf/lib/python3.13/site-packages/pyscf/hessian/rhf.py:330), actually calls CPHF with `tol=mf.conv_tol_cpscf*(ia1-ia0)`, scaled by atoms in the solve block. Reporting every solver call as exactly `1e-12` would omit that detail.

### 6. UNITS TRACE

```text
reference                                           : Å
PySCF internal coordinates                          : Bohr
E_RHF electronic + nuclear repulsion                : Ha
gradient = dE/dR                                    : Ha/Bohr
hessian = d(gradient)/dR                            : Ha/Bohr²
hessian transpose, subtraction, sum, abs/max         : Ha/Bohr²
sin(integer), normalized probe                      : dimensionless
epsilon_bohr * BOHR_A * probe                        : Bohr*(Å/Bohr)*1 = Å
probe-gradient subtraction                          : Ha/Bohr
gradient difference/(2*epsilon_bohr)                 : (Ha/Bohr)/Bohr = Ha/Bohr²
einsum(H,probe)                                     : (Ha/Bohr²)*1 = Ha/Bohr²
probe_error/max(probe_scale,1e-30)                    : dimensionless
```

The explicit finite-difference probe and finite SCF/CPHF tolerances make the comparison numerical; the Hessian itself is an analytic RHF derivative, not an energy finite-difference Hessian.

### 7. SAVED DATA / HANDOFF

```text
hessian -> results[i].cartesian_hessian_Eh_per_Bohr2
symmetry/translation/probe diagnostics -> results[i].hessian_validation
gradient -> results[i].stationarity.gradient_Eh_per_Bohr
```

These assignments occur at lines 452–458 and 485–494. `write_outputs` lines 726–735 converts NumPy arrays recursively to lists (`jsonable`, lines 92–103) and writes JSON. The current saved file is [json/new_rhf_harmonic_bond_ranges.json](/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json); `tests/new_rhf_harmonic_bond_ranges.json` is the compatibility path. H's actual direct consumer is `calculate` line 500 and later `normal_modes` line 555.

**Part 3 does not recalculate or project H.** It reads summary k/mass/frequency values (`codes/bond_length_part3.py`, lines 99–109). The full raw Hessian is retained as provenance and audit evidence.

The installed PySCF version is 2.14.0. Read-only `inspect.getsource` SHA-256 checks match all three saved source hashes: RHF Hessian module, `thermo.harmonic_analysis`, and `gto.mole.atom_mass_list`. The Hessian module SHA is `2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5`.

### 8. NUMERICAL WALKTHROUGH

For LiH, `symbols=['Li','H']`; H has shape `(2,2,3,3)`. The matrix below is the stored data reordered only for this report by `H.transpose(0,2,1,3).reshape(6,6)`. Rows/columns are `(Li-x,Li-y,Li-z,H-x,H-y,H-z)`; every entry is Ha/Bohr²:

```text
[[ 4.989818216949438e-08 -7.415240012289409e-18  1.270519832239949e-17 -4.989812818489980e-08  1.805967509577463e-17 -1.265654139126831e-17]
 [-1.805967509577473e-17  4.989818208622765e-08  2.552891822360946e-17  7.415240012289395e-18 -4.989812810163308e-08 -2.544908766307777e-17]
 [ 1.265654139126849e-17  2.544908766307792e-17  1.162103975013398e-01 -1.270519832239944e-17 -2.552891822360915e-17 -1.162103975012971e-01]
 [-4.989812818489980e-08  7.415240012289395e-18 -1.270519832239944e-17  4.989812904532265e-08 -1.179468133202010e-17  1.270519832239932e-17]
 [ 1.805967509577463e-17 -4.989812810163308e-08 -2.552891822360915e-17 -1.302573140197056e-17  4.989812896205592e-08  2.552891822360916e-17]
 [-1.265654139126831e-17 -2.544908766307777e-17 -1.162103975012971e-01  1.265654139126833e-17  2.544908766307774e-17  1.162103975012967e-01]]
```

Actual read-only arithmetic on the saved tensor gives:

| Numerical Python operation | Formula | Actual result |
|---|---|---|
| `max(abs(H-H.transpose(1,0,3,2)))` | \(\max\lvert H_{abxy}-H_{bayx}\rvert\) | `1.0644435083485322e-17 Ha/Bohr²` |
| `max(abs(H.sum(axis=1)))` | \(\max\lvert\sum_bH_{abxy}\rvert\) | `5.3984594572398237e-14 Ha/Bohr²` |
| `einsum('abxy,by->ax',H,v)` | \(\sum_{by}H_{abxy}v_{by}\) | rows `(4.511270296027083e-8,5.273223135807755e-8,0.027644638189610875)` and `(-4.511267763562678e-8,-5.273220405476588e-8,-0.0276446381896074)` Ha/Bohr² |
| `max(abs(saved_numerical_response-Hv))` | \(\max\lvert\partial_v g-Hv\rvert\) | `1.0172651430939705e-7 Ha/Bohr²` |
| `2e-7+2e-5*max(abs(Hv))` | allowed probe error | `7.528927637922176e-7 Ha/Bohr²`; passes |

The actual probe is `[[0.47599500946345563,0.5143624023896641,0.0798273746625673],[-0.4281005731864941,-0.5424348283348374,-0.15805700387850827]]`. These checks reuse stored gradient responses; no RHF, gradients, or Hessians were rerun for this audit.

## D. k: exact Hessian projection

### 1. PHYSICAL / MATHEMATICAL QUANTITY

The **implemented** scalar force constant is

\[
\boxed{k_q=\sum_{a,x,b,y}t_{ax}H_{abxy}t_{by}=t^THt.}
\]

The exact chain-rule curvature along a curved geometry path at a nonstationary point is instead

\[
\frac{d^2E(R(q))}{dq^2}=t^THt+\sum_{a,x}g_{ax}\frac{d^2R_{ax}}{dq^2}.
\]

These are not asserted identical in the source: the residual-gradient term is separately estimated and saved, but is **not added to k**. The assumption behind using k as harmonic curvature is local stationarity; a nonzero residual gradient is checked against tolerances.

### 2. EXACT SOURCE IMPLEMENTATION

File: [tests/new_rhf_harmonic_bond_ranges.py](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:500), `calculate`, lines 500–503:

```python
k_bohr = float(np.einsum("ax,abxy,by->", tangent, hessian, tangent))
mu_amu = float(np.einsum("a,ax,ax->", masses, tangent, tangent))
summary.update(k_q_Eh_per_Bohr2=k_bohr, k_q_Eh_per_A2=k_bohr / BOHR_A**2,
               k_q_N_per_m=k_bohr * EH_J / BOHR_M**2, mu_eff_amu=mu_amu, mu_eff_kg=mu_amu * AMU_KG)
```

`calculate`, lines 517–539:

```python
# At imperfect stationarity d2E/dq2 = t H t + gradient . R''.
acceleration_step = TANGENT_STEPS[-1]
acceleration = (path(acceleration_step) - 2 * reference + path(-acceleration_step)) / acceleration_step**2
gradient_correction = float(np.sum(gradient * acceleration) * BOHR_A)
result["path_acceleration"] = {"R_second_derivative_per_A": acceleration,
                               "gradient_correction_Eh_per_Bohr2": gradient_correction}
fd_rows = []
for h in ENERGY_STEPS:
    energies = [run_rhf(symbols, path(sign * h), basis, mf.make_rdm1()).e_tot for sign in (1, -1)]
    second = math.fsum([float(energies[0]), float(energies[1]), -2 * float(mf.e_tot)])
    fd_a = second / h**2
    fd_bohr = second / (h / BOHR_A)**2
    fd_rows.append({"h_A": h, "E_plus_Eh": float(energies[0]), "E_zero_Eh": float(mf.e_tot),
                    "E_minus_Eh": float(energies[1]), "k_FD_Eh_per_A2": fd_a,
                    "k_FD_Eh_per_Bohr2": fd_bohr,
                    "relative_difference": abs(fd_bohr - k_bohr) / max(abs(k_bohr), 1e-30),
                    "difference_from_acceleration_corrected_H_Eh_per_Bohr2": fd_bohr - k_bohr - gradient_correction})
# Predetermine the two smallest steps; never select a favorable step after the fact.
fine, coarse = fd_rows[-1], fd_rows[-2]
richardson = (4 * fine["k_FD_Eh_per_Bohr2"] - coarse["k_FD_Eh_per_Bohr2"]) / 3
allowed_fd = TOL["fd_absolute_Eh_per_Bohr2"] + TOL["fd_relative"] * abs(k_bohr)
fd_ok = all(abs(r["k_FD_Eh_per_Bohr2"] - k_bohr) <= allowed_fd for r in (coarse, fine))
fd_ok = fd_ok and abs(fine["k_FD_Eh_per_Bohr2"] - coarse["k_FD_Eh_per_Bohr2"]) <= allowed_fd
```

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Mathematical symbol | Shape | Units |
|---|---|---|---|
| `tangent` | \(t_{ax}\) | `(N,3)` | Å/Å = dimensionless |
| `hessian` | \(H_{abxy}\) | `(N,N,3,3)` | Ha/Bohr² |
| `k_bohr` | \(k_q\) in atomic length units | scalar | Ha/Bohr² |
| `gradient` | \(g_{ax}\) | `(N,3)` | Ha/Bohr |
| `acceleration_step` | \(\delta\) | scalar | Å |
| `acceleration` | \(R''_{ax}(q)\) | `(N,3)` | Å⁻¹ |
| `gradient_correction` | \(g\cdot R''\) converted to Bohr-coordinate curvature | scalar | Ha/Bohr² |
| `h` | energy-probe displacement | scalar | Å |
| `energies` | \([E(+h),E(-h)]\) | length-2 Python list | each Ha |
| `second` | \(E(+h)+E(-h)-2E(0)\) | scalar | Ha |
| `fd_a`, `fd_bohr` | central-difference \(d^2E/dq^2\) | scalar | Ha/Å², Ha/Bohr² |
| `richardson`, `allowed_fd` | diagnostic extrapolation, acceptance tolerance | scalar | Ha/Bohr² |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

```text
Python: np.einsum("ax,abxy,by->", tangent, hessian, tangent)
Index:  sum_a=0^(N-1) sum_x=0^2 sum_b=0^(N-1) sum_y=0^2 t[a,x]*H[a,b,x,y]*t[b,y]
Math:   t^T H t, where flattening t would use atom-major x,y,z order.
```

The empty right side of `->` requests a scalar; `float(...)` changes NumPy scalar representation to Python `float`, not units. The tangent is **not** normalized to Euclidean length one, and no denominator `t·t` is inserted. This is curvature with respect to the actual q defined by bond displacements, not a unit Cartesian displacement coordinate.

```text
Python: (path(delta)-2*reference+path(-delta))/delta**2
Index:  a_ax = [R_ax(delta)-2 R_e,ax+R_ax(-delta)]/delta²
Math:   central second derivative R''(0)+O(delta²).

Python: sum(gradient*acceleration)*BOHR_A
Index:  a0_A * sum_a,x g_ax a_ax
Math:   residual chain-rule term, converted to Ha/Bohr².

Python: math.fsum([Eplus,Eminus,-2*Ezero])
Math:   E(+h)+E(-h)-2E(0), with accurate floating-point summation.

Python: second/h**2; second/(h/BOHR_A)**2
Math:   central curvature in Å coordinates; central curvature in Bohr coordinates.

Python: (4*fine-coarse)/3
Math:   [4*k_FD(h)-k_FD(2h)]/3, canceling the leading O(h²) term.
```

`math.fsum` reduces summation error, but cannot eliminate errors already present in the converged SCF energies. The Richardson value is diagnostic only; neither it nor any FD curvature replaces `k_bohr`.

The FD acceptance threshold is

\[
\tau_k=2\times10^{-6}\ {\rm Ha/Bohr^2}+2\times10^{-4}|k_q|.
\]

Both predefined finest steps, 0.002 and 0.001 Å, must individually agree with the **uncorrected** `k_bohr` within \(\tau_k\), and must agree with one another within \(\tau_k\). The additional saved `fd_bohr-k_bohr-gradient_correction` is not the value used by that acceptance test.

For diatomics only, lines 507–515 independently compute `axis @ hessian[1,1] @ axis`: \(\sum_{xy}e_x H_{11xy}e_y\), the curvature when moving atom 1 along the bond. Translational invariance makes it agree with the COM-preserving two-atom projection. The allowed absolute discrepancy is `1e-7 Ha/Bohr²`.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

The tangent comes from `path.validate()` at line 439; it is the finest centered finite-difference tangent at `delta=2e-4 Å` from the aligned path, not an analytic vector. H comes from section C and retains its raw `(N,N,3,3)` ordering. The reference is the saved equilibrium geometry, and the gradient is freshly evaluated by Part 2 at that geometry (line 448).

`TANGENT_STEPS=(1e-3,5e-4,2e-4)` and `ENERGY_STEPS=(0.01,0.005,0.002,0.001)` are declared at lines 49–50. The path acceleration uses the same finest tangent step. The energy FD calculations reuse the equilibrium density matrix as an initial SCF guess, but obtain separately converged total energies at each displaced geometry.

Stationarity checks at lines 459–460 require maximum component ≤`2e-6 Ha/Bohr` and RMS ≤`1e-6 Ha/Bohr`. Lines 549–550 reject nonpositive curvature/mass; there is no absolute-value replacement or clipping to make a negative curvature positive. `max(abs(k_bohr),1e-30)` only floors denominators of diagnostic relative errors. Failed checks are accumulated in `failures`; final validation is `PASS` only if that list is empty (line 565).

### 6. UNITS TRACE

Both R and q were measured in Å when forming t, so

\[
t=\frac{dR_{\AA}}{dq_{\AA}}=
\frac{d(R_{\AA}/a_{0,\AA})}{d(q_{\AA}/a_{0,\AA})}
=\frac{dR_{\rm Bohr}}{dq_{\rm Bohr}}.
\]

The same dimensionless t can therefore project the Bohr-based Hessian:

```text
t * H * t                                  : 1*(Ha/Bohr²)*1 = Ha/Bohr²
k_bohr / BOHR_A**2                         : (Ha/Bohr²)/(Å/Bohr)² = Ha/Å²
k_bohr * EH_J / BOHR_M**2                  : (Ha/Bohr²)*(J/Ha)/(m/Bohr)² = J/m² = N/m
path(delta)-2*reference+path(-delta)        : Å
... / delta²                              : Å/Å² = 1/Å
gradient*acceleration                     : Ha/(Bohr*Å)
sum(...)*BOHR_A                           : Ha/(Bohr*Å)*(Å/Bohr) = Ha/Bohr²
fsum([Eplus,Eminus,-2*Ezero])              : Ha
second/h²                                 : Ha/Å²
h/BOHR_A                                  : Å/(Å/Bohr) = Bohr
second/(h/BOHR_A)²                         : Ha/Bohr²
(4*fine-coarse)/3                          : Ha/Bohr²
```

Constants are taken from `pyscf.data.nist` at lines 43–48 and written verbatim under `runtime.constants` at lines 768–769. The saved/current constants relevant here are `BOHR_A=0.52917721092`, `BOHR_M=5.2917721092e-11`, `EH_J=4.359744644911914e-18`.

### 7. SAVED DATA / HANDOFF

```text
k_bohr -> results[i].summary.k_q_Eh_per_Bohr2
k_bohr/BOHR_A² -> results[i].summary.k_q_Eh_per_A2
k_bohr*EH_J/BOHR_M² -> results[i].summary.k_q_N_per_m
acceleration/correction -> results[i].path_acceleration
all E(±h), E(0), finite curvatures -> results[i].finite_differences.rows
fine relative difference -> results[i].summary.Hessian_FD_relative_difference
Richardson diagnostic -> results[i].finite_differences.richardson_Eh_per_Bohr2
```

The direct scientific handoff is `oscillator_ranges(k_bohr,mu_amu,path.s_eq)` at line 552. It computes frequency and extents from the projected k, **not from a fitted RHF curve or normal-mode frequency**.

Part 3 `analyze_molecule`, [codes/bond_length_part3.py:99](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:99), reads `hs['k_q_Eh_per_Bohr2']` and validates the saved Å and SI versions at lines 107–108. It does not calculate H, t, their contraction, gradient corrections, or FD curvature. It exports the same value as `summary.k_Ha_per_Bohr2`, the saved converted values as `summary.k_Ha_per_A2` and `summary.k_N_per_m` (lines 116–118). The distinct spelling `Eh` versus `Ha` is a field-name convention; both mean Hartree.

### 8. NUMERICAL WALKTHROUGH

The LiH tangent is exactly the saved array

```text
t = [[0,0,-0.12682435832939154],
     [0,0, 0.8731756416702208]]
```

Since x/y components are zero, only the four zz terms contribute. The stored zz block and actual arithmetic are:

| Numerical Python multiplication | Corresponding term | Value [Ha/Bohr²] |
|---|---|---:|
| `(-0.12682435832939154)*0.1162103975013398*(-0.12682435832939154)` | \(t_{0z}H_{00zz}t_{0z}\) | 0.0018691765937462214 |
| `(-0.12682435832939154)*(-0.11621039750129714)*0.8731756416702208` | \(t_{0z}H_{01zz}t_{1z}\) | 0.012869132500554286 |
| `0.8731756416702208*(-0.11621039750129714)*(-0.12682435832939154)` | \(t_{1z}H_{10zz}t_{0z}\) | 0.012869132500554286 |
| `0.8731756416702208*0.11621039750129669*0.8731756416702208` | \(t_{1z}H_{11zz}t_{1z}\) | 0.08860295590635259 |
| `einsum('ax,abxy,by->',t,H,t)` | \(\sum_{abxy}t_{ax}H_{abxy}t_{by}\) | **0.11621039750120737** |
| `0.11621039750120737/0.52917721092**2` | \(k_{\AA}=k_{\rm Bohr}/a_{0,\AA}^2\) | **0.414994862916199 Ha/Å²** |
| `0.11621039750120737*4.359744644911914e-18/(5.2917721092e-11)**2` | \(k_{\rm SI}=k_{\rm Bohr}E_h/a_{0,m}^2\) | **180.92716312648525 N/m** |

The stored acceleration's z-components are `(-2.7755575615628914e-9,-5.551115123125783e-9) Å⁻¹`. The corresponding saved gradient z-components are `(-1.4247488588914337e-7,1.4247488611118797e-7) Ha/Bohr`.

```text
Python: ((-1.4247488588914337e-7)*(-2.7755575615628914e-9)
       +( 1.4247488611118797e-7)*(-5.551115123125783e-9))*0.52917721092
Formula: a0_A * sum_a,x g_ax R''_ax
Result: -2.0926167181291045e-16 Ha/Bohr² (saved diagnostic; not added to k).
```

LiH is a straight diatomic path; these tiny acceleration entries reflect numerical differentiation/alignment arithmetic. Across the saved molecules, the largest absolute residual correction is H2O2's `1.3095160978237728e-9 Ha/Bohr²`.

At the finest energy-FD step:

```text
Python: math.fsum([-7.863381921354176,-7.863381921492881,-2*(-7.86338212892111)])/(0.001/0.52917721092)**2
Formula: [E(+h)+E(-h)-2E(0)]/(h/a0_A)²
Result: 0.11621048169195046 Ha/Bohr².
```

All stored probes are:

| h [Å] | E(+h) [Ha] | E(0) [Ha] | E(-h) [Ha] | k_FD [Ha/Bohr²] |
|---:|---:|---:|---:|---:|
| .01 | -7.863361574957727 | -7.86338212892111 | -7.863361180509771 | .11621848592945257 |
| .005 | -7.863376965037416 | -7.86338212892111 | -7.863376917752705 | .11621241960189234 |
| .002 | -7.863381299989811 | -7.86338212892111 | -7.863381297868337 | .11621072095605835 |
| .001 | -7.863381921354176 | -7.86338212892111 | -7.863381921492881 | .11621048169195046 |

```text
Python: (4*0.11621048169195046-0.11621072095605835)/3
Formula: [4*k_FD(h)-k_FD(2h)]/3
Result: 0.11621040193724784 Ha/Bohr² (diagnostic only).

Python: 2e-6+2e-4*abs(0.11621039750120737)
Formula: tau_k = absolute_tol+relative_tol*|k|
Result: 2.5242079500241476e-5 Ha/Bohr².

Python: abs(0.11621048169195046-0.11621039750120737)/abs(0.11621039750120737)
Formula: |k_FD(h)-k|/|k|
Result: 7.244682480716566e-7; saved finest-step relative discrepancy.
```

Both fine/coarse differences and their mutual difference are below the saved tolerance. The production value passed onward is still `0.11621039750120737 Ha/Bohr²`.

## E. mu_eff: exact mass projection

### 1. PHYSICAL / MATHEMATICAL QUANTITY

\[
T=\tfrac12\sum_{A\alpha}m_A\dot R_{A\alpha}^2
=\tfrac12\mu_{eff}\dot q^2,\qquad
\mu_{eff}=\sum_{A\alpha}m_At_{A\alpha}^2.
\]

The source uses t at equilibrium and holds this projected mass constant in a one-coordinate harmonic model. It does not compute q-dependent mass or a full multidimensional kinetic operator. The ordinary diatomic reduced mass is a special-case check, not the general algorithm.

### 2. EXACT SOURCE IMPLEMENTATION

File: [new_rhf_harmonic_bond_ranges.py](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:435), `calculate:435–440,500–506`.

[tests/new_rhf_harmonic_bond_ranges.py, lines 435–440](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:435)

```python
mol = make_molecule(symbols, reference, basis)
masses = mol.atom_mass_list(isotope_avg=True)
result["isotope_average_masses_amu"] = masses
path = StretchPath(name, symbols, reference, masses)
tangent, diagnostics = path.validate()
result["path_validation"], result["tangent_dR_dq"] = diagnostics, tangent
```

[tests/new_rhf_harmonic_bond_ranges.py, lines 501–506](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:501)

```python
mu_amu = float(np.einsum("a,ax,ax->", masses, tangent, tangent))
summary.update(k_q_Eh_per_Bohr2=k_bohr, k_q_Eh_per_A2=k_bohr / BOHR_A**2,
               k_q_N_per_m=k_bohr * EH_J / BOHR_M**2, mu_eff_amu=mu_amu, mu_eff_kg=mu_amu * AMU_KG)
if name in DIATOMICS:
    reduced_mass = float(np.prod(masses) / np.sum(masses))
    error = abs(mu_amu / reduced_mass - 1)
```

Installed [PySCF atom_mass_list](/Users/novaz/Desktop/qml2.0/.venv-rhf/lib/python3.13/site-packages/pyscf/gto/mole.py:1998), lines 1998–2033: `isotope_avg=True` chooses `elements.MASSES` (2005–2009), indexes by atomic charge (2027–2031), and returns `numpy.array(mass)` (2033). Workflow `make_molecule:326–330` supplies no nuclear mass overrides.

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Symbol | Shape | Units |
|---|---|---|---|
| `masses` | \(m_A\) | `(N,)` | amu |
| `tangent` | \(t_{A\alpha}\) | `(N,3)` | dimensionless |
| `mu_amu` | \(\mu_{eff}\) | scalar | amu |
| conceptual repeated mass matrix, not allocated for einsum | \(M=\mathrm{diag}(m_1,m_1,m_1,\ldots,m_N,m_N,m_N)\) | `(3N,3N)` | amu |
| `AMU_KG` | atomic mass-unit conversion | scalar | kg/amu |
| `summary['mu_eff_kg']` | SI \(\mu_{eff}\) | scalar | kg |
| `reduced_mass` | \(m_1m_2/(m_1+m_2)\) | scalar | amu |
| `error` | relative diagnostic | scalar | dimensionless |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

```text
Python:
  mu_amu = float(np.einsum("a,ax,ax->", masses, tangent, tangent))
Index notation:
  mu_eff = sum(a=0..N-1) sum(x=0..2) masses[a]*tangent[a,x]*tangent[a,x]
Mathematical notation:
  mu_eff = Σ_A m_A |t_A|² = vec(t)^T M vec(t)
```

The empty einsum output sums over atom and Cartesian indices. `float` only changes the scalar container. No mass square roots, vector normalization, division by atom count, or Cartesian averaging occurs. `mu_amu*AMU_KG` converts amu to kg. `prod(masses)/sum(masses)` equals the two-body reduced mass only because `DIATOMICS` guarantees N=2. It is stored under `diatomic_checks.ordinary_reduced_mass_amu`, does not replace projected mass, and its relative discrepancy `abs(mu_amu/reduced_mass-1)` must be ≤`2e-8` (72,514).

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

Atomic order is unchanged from XYZ. The source requests isotope-average tabulated masses, not an explicitly selected isotopologue, and performs no electron-mass subtraction. LiH masses are `[6.94,1.008] amu`; H₂O `[15.999,1.008,1.008] amu`, isotope-average values from installed PySCF. t is the unnormalized aligned finite-difference array from B. The central atom contributes to polyatomic mass because alignment makes it recoil. No ordinary O–H reduced mass is substituted.

### 6. UNITS TRACE

```text
m_A                            : amu
t_Ax                           : Å/Å = 1
m_A*t_Ax*t_Ax                  : amu*1*1 = amu
sum over x and A               : amu
AMU_KG                         : 1.660539040427164e-27 kg/amu
mu_amu*AMU_KG                  : amu*(kg/amu) = kg
prod(masses)/sum(masses), N=2   : amu²/amu = amu
mu_amu/reduced_mass-1          : amu/amu-1 = 1
```

With q in meters, kinetic energy has units `kg*(m/s)^2=J`; converting R and q together preserves t numerically. No coordinate length conversion is needed within this mass contraction.

### 7. SAVED DATA / HANDOFF

```text
atom_mass_list -> results[i].isotope_average_masses_amu [N]
einsum -> mu_amu -> results[i].summary.mu_eff_amu
mu_amu*AMU_KG -> results[i].summary.mu_eff_kg
mu_amu -> oscillator_ranges(k_bohr,mu_amu,path.s_eq) -> omega
saved summary.mu_eff_amu/kg
 -> codes/bond_length_part3.py analyze_molecule:100,109,118
 -> Part 3 summary.mu_eff_amu/kg (same values)
 -> scanner validate_range_record:86–91 (exact equality with harmonic fields)
```

Part 3 recomputes `mu*amu_kg` only as a conversion check (109), not to replace mass or repeat its tangent contraction. Scanner `prepare_path` compares current isotope-average masses against the saved array before reconstructing geometry and tangent.

### 8. NUMERICAL WALKTHROUGH

LiH, with the actual finite-difference tangent:

```text
6.94*(-0.12682435832939154)**2 = 0.11162585998769364 amu [m_Li|t_Li|²]
1.008*0.8731756416702208**2   = 0.7685351868158513 amu [m_H|t_H|²]
0.11162585998769364+0.7685351868158513
 = 0.880161046803545 amu [mu_eff=Σ_A m_A|t_A|²]
0.880161046803545*1.660539040427164e-27
 = 1.461541780080527e-27 kg [mu_kg=mu_amu*c_amu_to_kg]
```

H₂O:

```text
15.999*((-2.7103567011263198e-15)**2+0**2+(-0.07191227169328629)**2)
 = 0.0827368257466043 amu [m_O|t_O|²]
1.008*(0.7661943494594037**2+0**2+0.5706966442547978**2)
 = 0.9200504284344637 amu [m_H1|t_H1|²]
1.008*((-0.7661943494594037)**2+0**2+0.5706966442547978**2)
 = 0.9200504284344637 amu [m_H2|t_H2|²]
0.0827368257466043+0.9200504284344637+0.9200504284344637
 = 1.9228376826155318 amu [mu_eff=Σ_A m_A|t_A|²]
1.9228376826155318*1.660539040427164e-27
 = 3.192947040387587e-27 kg [mu_kg=mu_amu*c_amu_to_kg]
```

Both re-evaluated masses equal their saved `summary.mu_eff_amu` bit for bit. The H₂O number includes both H motion and O recoil, so an ordinary O–H reduced-mass description would be incorrect.

## F. omega: unit conversion and calculation

### 1. PHYSICAL / MATHEMATICAL QUANTITY

For the chosen coordinate, the scalar angular frequency is

\[
\omega=\sqrt{k_{\rm SI}/\mu_{\rm kg}},\qquad
\widetilde\nu=\omega/(2\pi c\,100).
\]

Here the second expression produces a wavenumber in cm⁻¹ when `c` is in m/s. This is the frequency of the selected one-dimensional coordinate, not automatically a Cartesian normal-mode eigenfrequency. Section E defines its effective mass; that mass must accompany the curvature from section D.

### 2. EXACT SOURCE IMPLEMENTATION

Producer: `tests/new_rhf_harmonic_bond_ranges.py`, `oscillator_ranges`, lines 366–379. The constants are assigned from `pyscf.data.nist` at lines 43–48.

[tests/new_rhf_harmonic_bond_ranges.py, lines 43–48](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:43)

```python
BOHR_A = nist.BOHR
BOHR_M = BOHR_A * 1e-10
EH_J = nist.HARTREE2J
AMU_KG = nist.ATOMIC_MASS
HBAR = nist.HBAR
C_MS = nist.LIGHT_SPEED_SI
```

[tests/new_rhf_harmonic_bond_ranges.py, lines 366–379](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:366)

```python
def oscillator_ranges(k_bohr, mu_amu, s_eq):
    k_si, mu_kg = k_bohr * EH_J / BOHR_M**2, mu_amu * AMU_KG
    if k_si <= 0 or mu_kg <= 0 or not np.isfinite([k_si, mu_kg]).all():
        raise ValueError("A positive finite curvature and generalized mass are required.")
    omega = math.sqrt(k_si / mu_kg)
    length = math.sqrt(HBAR / (mu_kg * omega)) / 1e-10
    n1 = math.sqrt(3) * length
    summary = {
        "k_q_Eh_per_Bohr2": k_bohr, "k_q_Eh_per_A2": k_bohr / BOHR_A**2,
        "k_q_N_per_m": k_si, "mu_eff_amu": mu_amu, "mu_eff_kg": mu_kg,
        "omega_rad_per_s": omega, "harmonic_frequency_cm1": omega / (2 * math.pi * C_MS * 100),
        "Delta_q_n0_A": length, "Delta_q_n1_A": n1,
        "n0_min_A": s_eq - length, "n0_max_A": s_eq + length,
        "n1_min_A": s_eq - n1, "n1_max_A": s_eq + n1,
```

Later consumer/checker: `codes/bond_length_part3.py`, `analyze_molecule`, lines 99–110:

[codes/bond_length_part3.py, lines 99–110](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:99)

```python
k = finite(hs["k_q_Eh_per_Bohr2"], "saved k", True)
mu = finite(hs["mu_eff_amu"], "saved effective mass", True)
omega = finite(hs["omega_rad_per_s"], "saved angular frequency", True)
c = {key: finite(constants[key], key, True)
     for key in ("Bohr_A", "Bohr_m", "Hartree_J", "amu_kg", "hbar_J_s", "c_m_per_s")}
close(c["Bohr_m"], c["Bohr_A"] * 1e-10, "Bohr conversion", atol=0)
omega_check = math.sqrt((k * c["Hartree_J"] / c["Bohr_m"]**2) / (mu * c["amu_kg"]))
close(omega, omega_check, "omega versus saved k/mass", rtol=1e-10, atol=0)
close(hs["k_q_Eh_per_A2"], k / c["Bohr_A"]**2, "curvature Angstrom conversion", atol=0)
close(hs["k_q_N_per_m"], k * c["Hartree_J"] / c["Bohr_m"]**2, "curvature SI conversion", atol=0)
close(hs["mu_eff_kg"], mu * c["amu_kg"], "effective mass conversion", atol=0)
# The saved omega is used directly, not replaced by omega_check.
```

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Mathematical symbol | Shape | Units |
|---|---|---|---|
| `k_bohr`; Part 3 `k` | \(k_B\) | scalar | Ha/Bohr² |
| `EH_J`; `c['Hartree_J']` | \(C_E\) | scalar | J/Ha, numerical conversion factor |
| `BOHR_M`; `c['Bohr_m']` | \(C_B\) | scalar | m/Bohr |
| `mu_amu`; Part 3 `mu` | \(\mu_u\) | scalar | amu |
| `AMU_KG`; `c['amu_kg']` | \(C_m\) | scalar | kg/amu |
| `k_si` | \(k_{\rm SI}\) | scalar | N/m = J/m² |
| `mu_kg` | \(\mu_{\rm kg}\) | scalar | kg |
| `omega` | \(\omega\) | scalar | rad/s; radian is dimensionless |
| `C_MS`; `c['c_m_per_s']` | \(c\) | scalar | m/s |
| `harmonic_frequency_cm1` | \(\widetilde\nu\) | scalar JSON number | cm⁻¹ |
| `omega_check` | \(\omega_{\rm check}\) | scalar | rad/s |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

```text
Python:       k_si = k_bohr * EH_J / BOHR_M**2
Scalar math:  k_SI = k_B C_E / C_B²

Python:       mu_kg = mu_amu * AMU_KG
Scalar math:  mu_kg = mu_u C_m

Python:       omega = math.sqrt(k_si / mu_kg)
Scalar math:  omega = (k_SI / mu_kg)^(1/2)

Python:       omega / (2 * math.pi * C_MS * 100)
Scalar math:  nu_tilde_cm^-1 = omega / (2 pi c_m/s · 100)
```

There is no array contraction in this step. The division and square root are Python scalar floating-point operations. Positive finite curvature/mass are required; negative curvature is not converted with `abs`, clipped to zero, or forced to yield a real frequency.

In Part 3, `omega = finite(hs['omega_rad_per_s'], ...)` first **loads** the saved frequency. `omega_check` independently reconstructs the expression from the saved k/mass. `close` checks agreement but does not assign `omega_check` back into `omega`. Its comparison here has `rtol=1e-10, atol=0`.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

`calculate` passes the actual `k_bohr`, `mu_amu`, and `path.s_eq` into `oscillator_ranges` at harmonic-source line 552. Thus the frequency producer consumes the contractions in sections D/E, not a tabulated experimental mass/frequency. The same source calls `normal_modes` later at line 555. Its `dominant_mode_frequency_cm1` is diagnostic and is not substituted for `omega`.

Part 3 gets `hs = harmonic['summary']`, where `harmonic` is the molecule record loaded from the saved harmonic JSON. Its constants come from that JSON's `runtime.constants` (Part 3 lines 212–214), not a fresh call to an online constants table. The numerical constants used by both saved computation and this trace are listed in the report preamble.

### 6. UNITS TRACE

```text
k_bohr * EH_J   : (Ha/Bohr²) · (J/Ha) = J/Bohr²
BOHR_M**2       : (m/Bohr)²
k_si           : (J/Bohr²)/(m²/Bohr²) = J/m² = N/m = kg/s²
mu_amu*AMU_KG  : amu · kg/amu = kg
k_si / mu_kg   : (kg/s²)/kg = s⁻²
sqrt(...)      : s⁻¹, reported as rad/s for angular frequency
omega/(2*pi)   : cycles/s
... / C_MS     : m⁻¹
... / 100      : cm⁻¹
```

There is no additional factor of \(2\pi\) in \(\sqrt{k/\mu}\). That factor appears only when converting angular frequency to ordinary frequency/wavenumber.

### 7. SAVED DATA / HANDOFF

```text
oscillator_ranges.omega
 -> returned summary['omega_rad_per_s']
 -> calculate.summary.update(ranges) [harmonic source 552–553]
 -> harmonic JSON results[i].summary.omega_rad_per_s
 -> Part3 analyze_molecule.hs['omega_rad_per_s'] [101]
 -> Part3 summary.omega_rad_per_s [118]
 -> scanner summary -> plot input validation

k_si, mu_kg
 -> harmonic summary.k_q_N_per_m / mu_eff_kg
 -> Part3 checks at108–109 and copied summary fields at118

omega_check
 -> Part3 input_validation.omega_from_k_mu_check_rad_per_s
 -> accompanying omega_relative_error; CHECK ONLY
```

The standalone Part 3 uses saved constants. The integrated pipeline's `evaluate_part3` at `codes/bond_length.py:227–258` uses its `constants()` wrapper, which reads constants from the imported harmonic module; current values agree. That wrapper independently checks the saved wavenumber but does not itself repeat the k/mass consistency check present in standalone `analyze_molecule`.

### 8. NUMERICAL WALKTHROUGH

LiH, evaluated from the saved k and mass using the source's scalar arithmetic:

| Numerical Python operation and result | Corresponding formula |
|---|---|
| `0.11621039750120737 * 4.359744644911914e-18 / (5.2917721092e-11)**2 = 180.92716312648525` N/m | \(k_{SI}=k_B C_E/C_B^2\) |
| `0.880161046803545 * 1.660539040427164e-27 = 1.461541780080527e-27` kg | \(\mu_{kg}=\mu_u C_m\) |
| `180.92716312648525 / 1.461541780080527e-27 = 1.2379198842780714e29` s⁻² | \(\omega^2=k_{SI}/\mu_{kg}\) |
| `sqrt(1.2379198842780714e29) = 351840856677855.3` rad/s | \(\omega=\sqrt{k_{SI}/\mu_{kg}}\) |
| `351840856677855.3 / (2*pi*299792458*100) = 1867.8659194944714` cm⁻¹ | \(\widetilde\nu=\omega/(2\pi c\,100)\) |
| `abs(351840856677855.3 - 351840856677855.3) / 351840856677855.3 = 0.0` | Saved/check relative discrepancy |

H₂O provides a useful distinction: the saved collective-coordinate wavenumber is `4139.498437570796` cm⁻¹; its dominant normal-mode wavenumber is `4139.638283101215` cm⁻¹. They are close but not identical. Current code uses the former via its saved angular frequency, not the latter.

## G. hbar*omega and E_n

### 1. PHYSICAL / MATHEMATICAL QUANTITY

\[
Q_{\rm Ha}=\frac{\hbar_{\rm J\,s}\omega}{C_E},\qquad
E_{n0}=\frac12Q_{\rm Ha},\qquad E_{n1}=\frac32Q_{\rm Ha}.
\]

These scalar energies are measured above the RHF equilibrium minimum. `E_n0_Ha` includes zero-point energy. The gap between the two levels is \(Q\). The model is one selected one-dimensional harmonic coordinate; no sum over all polyatomic vibrational modes is computed here.

### 2. EXACT SOURCE IMPLEMENTATION

`codes/bond_length_part3.py`, `analyze_molecule`, lines 110–115:

[codes/bond_length_part3.py, lines 110–115](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:110)

```python
# The saved omega is used directly, not replaced by omega_check.
quantum = c["hbar_J_s"] * omega / c["Hartree_J"]
from_wavenumber = (2 * math.pi * c["hbar_J_s"] * c["c_m_per_s"] * 100
                   * finite(hs["harmonic_frequency_cm1"], "saved wavenumber", True) / c["Hartree_J"])
close(quantum, from_wavenumber, "Hartree quantum versus wavenumber", atol=1e-15)
values = compute_levels(re, quantum)
```

Same file, `compute_levels`, lines 55–63:

[codes/bond_length_part3.py, lines 55–63](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:55)

```python
def compute_levels(E_re_Ha, hbar_omega_Ha):
    """Apply strict <0 to the supplied Hartree values, without shifts/tolerances."""
    re = finite(E_re_Ha, "E_re_Ha")
    quantum = finite(hbar_omega_Ha, "hbar_omega_Ha", True)
    en0 = finite(0.5 * quantum, "E_n0_Ha", True)
    en1 = finite(1.5 * quantum, "E_n1_Ha", True)
    e0, e1 = finite(re + en0, "E_total_n0_Ha"), finite(re + en1, "E_total_n1_Ha")
    n0_pass, n1_pass = e0 < 0.0, e1 < 0.0
    selected = 1 if n1_pass else 0 if n0_pass else None
```

Same file, scalar validation helpers, lines 43–52:

[codes/bond_length_part3.py, lines 43–52](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:43)

```python
def finite(value, name, positive=False):
    number = float(value)
    require(math.isfinite(number) and (not positive or number > 0),
            f"{name} must be finite" + (" and positive." if positive else "."))
    return number


def close(actual, expected, name, *, rtol=1e-10, atol=1e-13):
    require(math.isclose(finite(actual, name), finite(expected, name), rel_tol=rtol, abs_tol=atol),
            f"Inconsistent {name}: {actual!r} versus {expected!r}.")
```

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Mathematical symbol | Shape | Units |
|---|---|---|---|
| `c['hbar_J_s']` | \(\hbar\) | scalar | J·s |
| `omega` | \(\omega\) | scalar | rad/s |
| `c['Hartree_J']` | \(C_E\) | scalar | J/Ha |
| `quantum`; `hbar_omega_Ha` | \(Q\) | scalar | Ha |
| `hs['harmonic_frequency_cm1']` | \(\widetilde\nu\) | scalar | cm⁻¹ |
| `from_wavenumber` | \(Q_{\rm check}\) | scalar | Ha |
| `en0`; `E_n0_Ha` | \(E_{n0}\) | scalar | Ha |
| `en1`; `E_n1_Ha` | \(E_{n1}\) | scalar | Ha |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

```text
Python:       quantum = c['hbar_J_s'] * omega / c['Hartree_J']
Scalar math:  Q = hbar · omega / C_E

Python:       from_wavenumber = 2*pi*hbar*c*100*nu_tilde / C_E
Scalar math:  Q_check = hbar · (2*pi*c*100*nu_tilde) / C_E

Python:       en0 = finite(0.5 * quantum, ..., True)
Scalar math:  E_n0 = (0 + 1/2) Q

Python:       en1 = finite(1.5 * quantum, ..., True)
Scalar math:  E_n1 = (1 + 1/2) Q
```

`finite` first casts to `float`, then rejects nonfinite numbers and, for these quantities, nonpositive values. It does not clip, round to an energy tolerance, or change the reference. Multiplication by `0.5` and `1.5` implements exactly the two requested n values; no general eigenvalue solver or numerical nuclear Schrödinger equation is run.

`close(quantum, from_wavenumber, ..., atol=1e-15)` uses the helper default `rtol=1e-10`. Its acceptance condition is the standard scalar `math.isclose` condition, \(|a-b|\le\max(10^{-10}\max(|a|,|b|),10^{-15})\). This is a conversion check, not a tolerance in the later sign decision.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

`analyze_molecule` loads the already validated positive `omega` from Part 2, reads the constants dictionary from saved runtime metadata, and builds `quantum` with one multiplication followed by one division. It passes only `re` and `quantum` to `compute_levels`.

`compute_levels` receives no Hessian, tangent, dissociation energy, well depth or RHF scan as a decision input. `analyze_molecule` receives the full harmonic record, including saved arrays, but does not use those arrays to reconstruct the decision inputs. `from_wavenumber` is a check reconstructed from a second saved representation of the same frequency; it is not an independent physical measurement.

### 6. UNITS TRACE

```text
hbar * omega           : (J·s) · s⁻¹ = J
(hbar * omega)/Hartree_J: J/(J/Ha) = Ha
0.5 * quantum          : dimensionless · Ha = Ha
1.5 * quantum          : dimensionless · Ha = Ha
100 * wavenumber_cm1   : converts cm⁻¹ -> m⁻¹
c * (100*wavenumber)   : (m/s) · m⁻¹ = s⁻¹
2*pi * preceding      : angular frequency, rad/s
hbar * preceding / C_E : Ha
```

The constants are consistent with the saved Part 2 calculation. The implementation does not substitute a newer numerical constant partway through the standalone calculation.

### 7. SAVED DATA / HANDOFF

```text
Part2 summary.omega_rad_per_s [READ]
 -> analyze_molecule.quantum [NEW SCALAR ARITHMETIC]
 -> compute_levels.quantum/en0/en1 [NEW SCALAR ARITHMETIC]
 -> Part3 results[i].summary.hbar_omega_Ha / E_n0_Ha / E_n1_Ha
 -> vibrational_levels.json and vibrational_levels.csv [Part3 lines232–242]
 -> scanner validate_range_record / load_inputs
 -> plot load_plot_data and harmonic-level consistency checks

from_wavenumber
 -> input_validation.hbar_omega_from_wavenumber_Ha
 -> abs(quantum-from_wavenumber)
 -> input_validation.quantum_conversion_residual_Ha
```

JSON and numeric CSV columns preserve floating-point values; Markdown presentation uses `display` with 12 significant digits (Part 3 lines138–143). Report rounding does not feed the sign test.

### 8. NUMERICAL WALKTHROUGH

| Numerical Python operation and result, LiH | Corresponding formula |
|---|---|
| `1.0545718001391127e-34 * 351840856677855.3 = 3.7104144558925345e-20` J | \(\hbar\omega\) in J |
| `3.7104144558925345e-20 / 4.359744644911914e-18 = 0.00851062334630725` Ha | \(Q=\hbar\omega/C_E\) |
| `0.5 * 0.00851062334630725 = 0.004255311673153625` Ha | \(E_{n0}=Q/2\) |
| `1.5 * 0.00851062334630725 = 0.012765935019460876` Ha | \(E_{n1}=3Q/2\) |
| `2*pi*1.0545718001391127e-34*299792458*100*1867.8659194944714/4.359744644911914e-18 = 0.008510623346307249` Ha | \(Q_{check}=2\pi\hbar c\,100\widetilde\nu/C_E\) |
| `abs(0.00851062334630725 - 0.008510623346307249) = 1.734723475976807e-18` Ha | \(\lvert Q-Q_{check}\rvert\) |

Each number above reproduces the current saved Part 3 fields from the saved inputs. This audit performed only scalar arithmetic, not a new RHF calculation.

## H. Turning points A_n and bond ranges

### 1. PHYSICAL / MATHEMATICAL QUANTITY

For the harmonic model,

\[
\frac12kA_n^2=E_n,\qquad
A_n=\sqrt{(2n+1)\frac{\hbar}{\mu\omega}}.
\]

With \(\ell=\sqrt{\hbar/(\mu\omega)}\), \(A_0=\ell\), \(A_1=\sqrt3\ell\), and

\[
r_{n,\min}=r_e-A_n,\qquad r_{n,\max}=r_e+A_n.
\]

These are harmonic classical turning points. They are not the full support of a quantum wavefunction, probability quantiles, or turning points solved from the nonquadratic RHF scan.

### 2. EXACT SOURCE IMPLEMENTATION

`tests/new_rhf_harmonic_bond_ranges.py`, `oscillator_ranges`, lines 370–379:

[tests/new_rhf_harmonic_bond_ranges.py, lines 370–379](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:370)

```python
omega = math.sqrt(k_si / mu_kg)
length = math.sqrt(HBAR / (mu_kg * omega)) / 1e-10
n1 = math.sqrt(3) * length
summary = {
    "k_q_Eh_per_Bohr2": k_bohr, "k_q_Eh_per_A2": k_bohr / BOHR_A**2,
    "k_q_N_per_m": k_si, "mu_eff_amu": mu_amu, "mu_eff_kg": mu_kg,
    "omega_rad_per_s": omega, "harmonic_frequency_cm1": omega / (2 * math.pi * C_MS * 100),
    "Delta_q_n0_A": length, "Delta_q_n1_A": n1,
    "n0_min_A": s_eq - length, "n0_max_A": s_eq + length,
    "n1_min_A": s_eq - n1, "n1_max_A": s_eq + n1,
```

Harmonic `calculate` checks the final range at lines 561–564:

[tests/new_rhf_harmonic_bond_ranges.py, lines 561–564](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:561)

```python
if modes["imaginary_mode_count"]:
    result["failures"].append("FAIL_EQUILIBRIUM_MINIMUM: imaginary vibrational modes were found.")
if not np.isfinite(list(ranges.values())).all() or ranges["n1_min_A"] <= 0:
    result["failures"].append("FAIL_RANGE: nonfinite extents or nonpositive bond interval endpoint.")
```

Part 3 **reads** amplitudes and endpoints and checks an equation, in `analyze_molecule`, lines119–126:

[codes/bond_length_part3.py, lines 119–126](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:119)

```python
for n in (0, 1):
    for key in (f"Delta_q_n{n}_A", f"n{n}_min_A", f"n{n}_max_A"):
        values[key] = finite(hs[key], key, True)
    close(0.5 * values["k_Ha_per_A2"] * values[f"Delta_q_n{n}_A"]**2,
          values[f"E_n{n}_Ha"], f"n={n} saved turning point", rtol=1e-9)
n = values["selected_n"]
values["selected_range_min_A"] = None if n is None else values[f"n{n}_min_A"]
values["selected_range_max_A"] = None if n is None else values[f"n{n}_max_A"]
```

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Mathematical symbol | Shape | Units |
|---|---|---|---|
| `HBAR` | \(\hbar\) | scalar | J·s |
| `mu_kg`, `omega` | \(\mu,\omega\) | scalar each | kg, rad/s |
| `length` | \(A_0=\ell\) | scalar | Å after division by `1e-10` |
| `n1` | \(A_1=\sqrt3\ell\) | scalar | Å; despite its name, this is an amplitude, not the integer n |
| `s_eq` | \(r_e\), mean participating equilibrium length | scalar | Å |
| `Delta_q_n0_A`, `Delta_q_n1_A` | \(A_0,A_1\) | scalar each | Å |
| `n0_min_A`, `n0_max_A` | \(r_{0,\min},r_{0,\max}\) | scalar each | Å |
| `n1_min_A`, `n1_max_A` | \(r_{1,\min},r_{1,\max}\) | scalar each | Å |
| Part3 `k_Ha_per_A2` | \(k_A\) | scalar | Ha/Å² |
| Part3 `n` | selected integer or `None` | scalar/None | dimensionless |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

```text
Python:       length = sqrt(HBAR / (mu_kg * omega)) / 1e-10
Scalar math:  A0_A = sqrt(hbar / (mu_kg omega)) / C_A, C_A=10^-10 m/Å

Python:       n1 = sqrt(3) * length
Scalar math:  A1_A = sqrt(3) A0_A

Python:       s_eq - length, s_eq + length
Scalar math:  r0_min = re - A0; r0_max = re + A0

Python:       s_eq - n1, s_eq + n1
Scalar math:  r1_min = re - A1; r1_max = re + A1

Part3 check:  0.5 * k_Ha_per_A2 * Delta_q_n{n}_A**2
Scalar math:  (1/2) k_A A_n,A², compared with saved-omega-derived E_n,Ha
```

**Actual execution order differs from the conceptual chain in the question.** Part 2 does not explicitly calculate `E_n` and then run `sqrt(2*E_n/k)`. It computes `omega -> length -> n1 -> endpoints` first. Later Part 3 computes `Q -> E_n` and checks that the saved amplitude gives the same harmonic energy. The two expressions are algebraically equivalent because `k=mu*omega**2`, subject to floating-point rounding and the harmonic model.

The stored ranges are symmetric about `s_eq`. For a star molecule `s_eq` is the mean of the participating equilibrium lengths; those lengths are checked to agree within `1e-6 Å`, not asserted identical in exact arithmetic. Each bond still receives the same q relative to its own stored equilibrium length.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

`oscillator_ranges(k_bohr, mu_amu, path.s_eq)` receives the scalar outputs of sections D–F and the geometry-derived `s_eq`. It constructs both ranges whether or not Part 3 will later choose either level.

The same function separately constructs 95%, 99%, and 99.9% probability intervals at lines381–392. It solves `probability_mass(n,a)-probability=0` with `brentq` on `[0,10]`, `xtol=1e-14`, then multiplies `a*length`. The functions are

\[
P_0(a)=\operatorname{erf}(a),\quad
P_1(a)=\operatorname{erf}(a)-2ae^{-a^2}/\sqrt\pi.
\]

Those separate probability intervals are saved for diagnostics. They do not set `n0_min_A`, `n1_min_A`, the selected scan interval, or the plotted level-segment endpoints.

### 6. UNITS TRACE

```text
mu_kg * omega              : kg/s
HBAR/(mu_kg*omega)         : (J·s)/(kg/s) = J·s²/kg = m²
sqrt(HBAR/(mu_kg*omega))   : m
... / (1e-10 m/Å)         : Å
sqrt(3)*length            : Å
s_eq ± amplitude          : Å ± Å = Å
0.5*k_Ha_per_A2*A_A**2    : (Ha/Å²)·Å² = Ha
```

No absolute value or endpoint clipping is applied. Part 2 reports a failure when its `n1_min_A <= 0`; Part 3's `finite(..., True)` also requires positive amplitudes and positive endpoints. Inconsistent values are rejected rather than repaired. The turning-point check uses `rtol=1e-9` and the `close` helper's default `atol=1e-13 Ha`.

### 7. SAVED DATA / HANDOFF

```text
length / n1 / s_eq±length / s_eq±n1
 -> oscillator_ranges.summary.{Delta_q_n0_A,Delta_q_n1_A,n0_min_A,...,n1_max_A}
 -> harmonic JSON results[i].summary [write_outputs726–734]
 -> Part3 analyze_molecule reads hs[key] [119–121]
 -> Part3 summary copies each float unchanged
 -> chosen n selects summary.selected_range_min_A / selected_range_max_A [124–126]
 -> scanner build_grid reads selected_range_* [151–165]
 -> plot local_panel uses n0/n1 endpoints for horizontal segments
```

`harmonic_oscillator.oscillator_length_A` and `q_n0_minus_A`, `q_n0_plus_A`, `q_n1_minus_A`, `q_n1_plus_A` are additional saved representations. Part 3 consumes the `summary` fields instead. It does not recompute endpoints from `r_e ± A_n`; it copies them and checks the energy/amplitude relation. Scanner checks bind copied endpoints back to the saved harmonic record.

### 8. NUMERICAL WALKTHROUGH

| Numerical Python operation and result, LiH | Corresponding formula |
|---|---|
| `1.461541780080527e-27 * 351840856677855.3 = 5.142301119740103e-13` kg/s | \(\mu\omega\) |
| `1.0545718001391127e-34 / 5.142301119740103e-13 = 2.0507779991546117e-22` m² | \(\ell^2=\hbar/(\mu\omega)\) |
| `sqrt(2.0507779991546117e-22)/1e-10 = 0.14320537696450547` Å | \(A_0=\ell/C_A\) |
| `sqrt(3)*0.14320537696450547 = 0.24803898881957717` Å | \(A_1=\sqrt3 A_0\) |
| `1.510811853032 - 0.14320537696450547 = 1.3676064760674946` Å | \(r_{0,\min}=r_e-A_0\) |
| `1.510811853032 + 0.14320537696450547 = 1.6540172299965055` Å | \(r_{0,\max}=r_e+A_0\) |
| `1.510811853032 - 0.24803898881957717 = 1.262772864212423` Å | \(r_{1,\min}=r_e-A_1\) |
| `1.510811853032 + 0.24803898881957717 = 1.7588508418515771` Å | \(r_{1,\max}=r_e+A_1\) |
| `0.5*0.414994862916199*0.14320537696450547**2` ≈ `0.004255311673153624` Ha | \(k_A A_0^2/2=E_{n0}\), roundoff residual about \(-8.67\times10^{-19}\) Ha |
| `0.5*0.414994862916199*0.24803898881957717**2` = `0.012765935019460871` Ha | \(k_A A_1^2/2=E_{n1}\), residual \(-5.20\times10^{-18}\) Ha |

## I. Part-3 n=0/n=1 decision

### 1. PHYSICAL / MATHEMATICAL QUANTITY

The **implemented** total-energy equation and literal sign rule are

\[
E_{\rm total,n}=E_{\rm re}+E_n,
\qquad
n_{\rm selected}=\begin{cases}
1&E_{\rm total,1}<0,\\
0&E_{\rm total,0}<0\ \text{and}\ E_{\rm total,1}\ge0,\\
\mathrm{None}&E_{\rm total,0}\ge0.
\end{cases}
\]

This rule is implemented exactly as requested. It is not established by the current energy-reference definition as a physical bound-state criterion: `E_re` is the unshifted absolute RHF molecular energy including nuclear repulsion. Code explicitly stores that limitation. No dissociation plateau or well depth enters the decision.

### 2. EXACT SOURCE IMPLEMENTATION

`codes/bond_length_part3.py`, `compute_levels`, lines55–69:

[codes/bond_length_part3.py, lines 55–69](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:55)

```python
def compute_levels(E_re_Ha, hbar_omega_Ha):
    """Apply strict <0 to the supplied Hartree values, without shifts/tolerances."""
    re = finite(E_re_Ha, "E_re_Ha")
    quantum = finite(hbar_omega_Ha, "hbar_omega_Ha", True)
    en0 = finite(0.5 * quantum, "E_n0_Ha", True)
    en1 = finite(1.5 * quantum, "E_n1_Ha", True)
    e0, e1 = finite(re + en0, "E_total_n0_Ha"), finite(re + en1, "E_total_n1_Ha")
    n0_pass, n1_pass = e0 < 0.0, e1 < 0.0
    selected = 1 if n1_pass else 0 if n0_pass else None
    return {"E_re_Ha": re, "hbar_omega_Ha": quantum, "E_n0_Ha": en0, "E_n1_Ha": en1,
            "E_total_n0_Ha": e0, "E_total_n1_Ha": e1, "n0_pass": n0_pass, "n1_pass": n1_pass,
            "selected_n": selected, "decision_criterion": "E_total_n < 0 Ha", "reference_shift_Ha": 0.0,
            "energy_reference_status": "UNVALIDATED_ZERO_REFERENCE", "energy_reference_warning": REFERENCE_WARNING,
            "physical_vibrational_binding_established": False,
            "validation_status": "PASS" if selected is not None else "NO_LEVEL_PASSES_ZERO_TEST"}
```

Its `E_re` input is established by `analyze_molecule`, lines88–98:

[codes/bond_length_part3.py, lines 88–98](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py:88)

```python
require(metadata["molecule"] == hs["molecule"] == name, "Molecule mismatch.")
require(metadata["method"] == "RHF" and metadata["status"] == "converged"
        and metadata["optimizer_success"].lower() == "true", "Part 1 equilibrium is not converged RHF.")
require(hs["validation_status"] == "PASS" and not harmonic.get("failures")
        and harmonic["finite_differences"]["convergence_pass"] is True, "Part 2 harmonic validation failed.")
require(metadata["basis"].lower() == hs["RHF_basis"].lower(), "RHF basis mismatch.")
re = finite(metadata["energy_hartree"], "Part 1 energy_hartree")
gradient = finite(metadata["max_gradient_hartree_per_bohr"], "Part 1 nuclear gradient")
require(0 <= gradient <= 1e-6, "Part 1 nuclear gradient exceeds the existing limit.")
close(re, harmonic["input"]["metadata_row"]["energy_hartree"], "saved optimized energy", rtol=0, atol=1e-12)
close(re, harmonic["stationarity"]["E_RHF_Eh"], "Part 2 stationary energy", rtol=0, atol=1e-9)
```

The original equilibrium producer is `tests/molecules_rhf.py:82–110` and its CSV writer is at199–206. The literal saved value is `f'{result.fun:.15f}'`; Part 3 applies `float` through `finite` to that CSV text. It does not use the Part 2 RHF reevaluation as the energy baseline. The integrated caller `codes/bond_length.py:evaluate_part3`, lines227–240, passes `equilibrium['E_eq_Eh']` to the same `compute_levels`.

The energy convention can also be verified in the installed PySCF producer, `pyscf.scf.hf.energy_tot`, rather than inferred from the field name:

[.venv-rhf/lib/python3.13/site-packages/pyscf/scf/hf.py, lines 300–319](/Users/novaz/Desktop/qml2.0/.venv-rhf/lib/python3.13/site-packages/pyscf/scf/hf.py:300)

```python
def energy_tot(mf, dm=None, h1e=None, vhf=None):
    r'''Total Hartree-Fock energy, electronic part plus nuclear repulsion
    See :func:`scf.hf.energy_elec` for the electron part

    Note this function has side effects which cause mf.scf_summary updated.

    '''
    nuc = mf.energy_nuc()
    mf.scf_summary['nuc'] = nuc.real

    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    if mf.do_disp():
        if 'dispersion' in mf.scf_summary:
            e_tot += mf.scf_summary['dispersion']
        else:
            e_disp = mf.get_dispersion()
            mf.scf_summary['dispersion'] = e_disp
            e_tot += e_disp

    return e_tot
```

Here `mf.energy_elec(...)[0] + mf.energy_nuc()` is the electronic expectation plus nuclear repulsion. The SCF kernel calls this function at `hf.py:133,179`; `mf.do_disp()` is false for this configured RHF workflow. No separated-fragment reference or additive normalization is inserted before the CSV write.

[codes/bond_length.py, lines 227–240](/Users/novaz/Desktop/qml2.0/codes/bond_length.py:227)

```python
def evaluate_part3(equilibrium, harmonic_record):
    """Use Parts 1/2 verbatim; apply the requested zero test without an energy shift."""
    hs = harmonic_record["summary"]
    E_re = equilibrium["E_eq_Eh"]
    if not math.isclose(E_re, harmonic_record["stationarity"]["E_RHF_Eh"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("Part 1 equilibrium energy differs from the validated Part 2 input.")
    omega = hs["omega_rad_per_s"]
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError("Part 2 must supply a positive finite angular frequency in rad/s.")
    quantum = constants()["hbar_J_s"] * omega / constants()["Hartree_J"]
    from_wavenumber = hs["harmonic_frequency_cm1"] * selection.conversion_from_constants(constants())
    if not math.isclose(quantum, from_wavenumber, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Saved angular frequency and wavenumber give inconsistent energy quanta.")
    summary = part3.compute_levels(E_re, quantum)
```

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Mathematical symbol | Shape | Units |
|---|---|---|---|
| metadata `energy_hartree` | persisted \(E_{\rm re}\) | scalar CSV string | Ha |
| `re`, `E_re_Ha` | \(E_{\rm re}\) | scalar float | Ha |
| `harmonic['stationarity']['E_RHF_Eh']` | Part 2 reevaluation \(E^{(2)}(0)\) | scalar float | Ha |
| `en0`, `en1` | \(E_{n0}, E_{n1}\) | scalar each | Ha |
| `e0`, `e1` | \(E_{\rm total,0},E_{\rm total,1}\) | scalar each | Ha |
| `n0_pass`, `n1_pass` | truth values of strict inequalities | scalar bool each | dimensionless |
| `selected` / `selected_n` | \(n_{\rm selected}\) | scalar int or None | dimensionless |
| `reference_shift_Ha` | additive shift | scalar float, exactly0 | Ha |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

```text
Python:       re = finite(metadata['energy_hartree'], ...)
Scalar math:  E_re = float(the original equilibrium CSV energy)

Python:       e0 = finite(re + en0, ...)
Scalar math:  E_total,0 = E_re + E_n0

Python:       e1 = finite(re + en1, ...)
Scalar math:  E_total,1 = E_re + E_n1

Python:       n0_pass, n1_pass = e0 < 0.0, e1 < 0.0
Logic:        b0 := [E_total,0 < 0]; b1 := [E_total,1 < 0]

Python:       selected = 1 if n1_pass else 0 if n0_pass else None
Logic:        if b1 select1; otherwise if b0 select0; otherwise no selected level
```

The baseline is added exactly once. There is no subtraction of `E_re`, plateau, or other shift. `0.0` means zero on the input Hartree scale. No `isclose` or tolerance appears in either sign comparison. A total equal to zero fails. The selected value is the highest **among 0 and1**, not a maximum over all harmonic levels.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

Standalone `main` loads the equilibrium CSV using `csv.DictReader`, indexes molecule rows, loads the harmonic JSON, and indexes its `results` by `summary.molecule` (lines210–214). It checks the CSV SHA256 against each harmonic record's `input.metadata_sha256` (215–216).

`analyze_molecule` then requires matching identity/basis, Part1 converged RHF with optimizer success, Part2 `validation_status == PASS`, empty failures, and finite-difference convergence. It checks the original nuclear gradient against `1e-6 Ha/Bohr`. Its equilibrium comparisons have absolute tolerances `1e-12 Ha` against Part2's copied input row and `1e-9 Ha` against Part2's stationary-energy reevaluation. These tolerances validate the inputs, not the `<0` result.

The integrated pipeline follows a separate input/caching wrapper, `process_molecule` and `harmonic_cache_matches`, then calls the same pure arithmetic function. Its successful reuse-only path reads Parts1/2 and sets selection source to `postprocessed` (pipeline lines351–374). Its `energy_view` wrapper (70–84) stores mode-energy quanta in place of wavenumber fields in the joined pipeline JSON and reverses that representation when loading that cache. `omega_rad_per_s`, the Hessian, and tangent are not replaced by mode-energy fields.

### 6. UNITS TRACE

```text
CSV energy_hartree -> float : Ha -> Ha, only representation changes
re + en0                   : Ha + Ha = Ha
re + en1                   : Ha + Ha = Ha
e0 < 0.0 / e1 < 0.0        : Ha compared with 0Ha -> Boolean
selected                   : Boolean control flow -> dimensionless integer/None
reference_shift_Ha         : 0Ha, no reference transformation
```

Finite checks reject NaN, infinity, nonpositive quantum/excitations, and nonfinite totals. Totals themselves can be positive, zero or negative. Their sign is evaluated using the resulting Python floating-point numbers, without pre-rounding. The code does not silently turn a tiny positive total into a passing negative result.

### 7. SAVED DATA / HANDOFF

```text
Part1 optimize.energy_gradient: mf.kernel() -> result.fun
 -> equilibrium CSV row.energy_hartree (15 decimal places)
 -> Part3 analyze_molecule.re
 -> compute_levels(re, quantum)
 -> summary.E_total_n0_Ha / E_total_n1_Ha
 -> summary.n0_pass / n1_pass / selected_n
 -> analyze_molecule chooses already saved n0/n1 endpoint fields
 -> summary.selected_range_min_A / selected_range_max_A
 -> vibrational_levels.json, vibrational_levels.csv [240–241]
 -> scanner validates and consumes chosen interval
 -> plot validates and consumes absolute level energies
```

`selected_n=None` produces `selected_range_min_A=None` and `selected_range_max_A=None`. The downstream scanner requires a selected integer0 or1, so it rejects an unselected range. `validation_status` is `PASS` when a level passes, otherwise `NO_LEVEL_PASSES_ZERO_TEST`; it does not override `energy_reference_status='UNVALIDATED_ZERO_REFERENCE'` or `physical_vibrational_binding_established=False`.

The current legacy JSON/CSV paths are symlinks to these current outputs. They are not a separate source of threshold values. Current decision code contains no `E_diss` or `D_e` parameter or arithmetic.

### 8. NUMERICAL WALKTHROUGH

| Numerical Python operation and result, LiH | Corresponding formula |
|---|---|
| `float('-7.863382128921103') = -7.863382128921103` Ha | \(E_{re}\), read from Part1 |
| `-7.86338212892111 - (-7.863382128921103) = -7.105427357601002e-15` Ha | \(E^{(2)}(0)-E_{re}\), CHECK ONLY |
| `-7.863382128921103 + 0.004255311673153625 = -7.859126817247949` Ha | \(E_{total,0}=E_{re}+Q/2\) |
| `-7.863382128921103 + 0.012765935019460876 = -7.850616193901642` Ha | \(E_{total,1}=E_{re}+3Q/2\) |
| `-7.859126817247949 < 0.0 -> True` | \(E_{total,0}<0\) |
| `-7.850616193901642 < 0.0 -> True` | \(E_{total,1}<0\) |
| `1 if True else 0 if True else None -> 1` | Highest passing n in {0,1} |
| `hs['n1_min_A'], hs['n1_max_A'] -> (1.262772864212423, 1.7588508418515771)` Å | Chosen \([r_e-A_1,r_e+A_1]\), copied |

The current saved records for all nine molecules have both pass flags true and `selected_n=1`. This is an observation about the literal sign rule with their supplied absolute energies; it does not remove the reference warning.

## J. 30-point RHF scan

### 1. PHYSICAL / MATHEMATICAL QUANTITY

The fresh-scan branch samples the **fixed-geometry electronic potential**, not vibrational eigenenergies:

\[
r_i=r_{\min}+\frac{i}{29}(r_{\max}-r_{\min}),\quad
q_i=r_i-r_e,\quad \mathbf R_i=\mathbf R(q_i),\quad
U_i=E_{\rm RHF}(\mathbf R_i),\qquad i=0,\ldots,29.
\]

The saved auxiliary diagnostic is \(\Delta U_i=U_i-E_{\rm re}\). Neither \(E_n\) nor a dissociation energy is added to \(U_i\). The nuclei are not reoptimized at each sampled geometry.

### 2. EXACT SOURCE IMPLEMENTATION

[codes/rhf_bond_scan_30.py](/Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py:58), `build_grid`, lines 58–65:

[codes/rhf_bond_scan_30.py, lines 58–65](/Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py:58)

```python
def build_grid(lower, upper):
    """Exactly 30 ascending, equally spaced coordinates including saved endpoints."""
    lower, upper = finite(lower, "lower endpoint"), finite(upper, "upper endpoint")
    require(0 < lower < upper, "Endpoints must satisfy 0 < lower < upper.")
    grid = [lower + (upper - lower) * index / (POINT_COUNT - 1) for index in range(POINT_COUNT)]
    grid[0], grid[-1] = lower, upper
    require(all(a < b for a, b in zip(grid, grid[1:])), "Range is too narrow for 30 distinct floats.")
    return grid
```

`POINT_COUNT=30` at line 28. `prepare_plans`, lines 151–167:

[codes/rhf_bond_scan_30.py, lines 156–165](/Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py:156)

```python
path, symbols, basis = engine.prepare_path(harmonic)
points = []
for index, length in enumerate(build_grid(summary["selected_range_min_A"],
                                           summary["selected_range_max_A"]), 1):
    q = length - path.s_eq
    coords = path(q)
    geometry = engine.validate_scan_geometry(path, q, coords)
    require(geometry["passed"] is True, f"{summary['molecule']}: invalid geometry at point {index}.")
    points.append({"point_index": index, "bond_length_A": length, "q_A": q,
                   "cartesian_A": coords, "geometry_validation": geometry})
```

`scan_molecule`, lines 170–207, specifically lines 185–197:

[codes/rhf_bond_scan_30.py, lines 185–197](/Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py:185)

```python
mf, state = engine.solve_point(plan["symbols"], point["cartesian_A"], plan["basis"], previous=previous)
row["diagnostics"] = state
for key in ("scf_converged", "internal_stable", "orbital_gradient_norm_Eh"):
    row[key] = state.get(key)
require(mf is not None and state.get("accepted") is True
        and state.get("scf_converged") is True and state.get("internal_stable") is True,
        state.get("failure", "RHF convergence or internal stability failed."))
energy = finite(state["energy_Eh"], "RHF energy")
gradient = finite(state["orbital_gradient_norm_Eh"], "RHF orbital gradient")
require(0 <= gradient <= 1e-10, "RHF orbital gradient exceeds 1e-10 Ha.")
row.update(E_RHF_Ha=energy, delta_E_RHF_Ha=energy - summary["E_re_Ha"], status="PASS")
summary["accepted_points"] += 1
previous = mf
```

[tests/rhf_bound_level_selection.py](/Users/novaz/Desktop/qml2.0/tests/rhf_bound_level_selection.py:90), `prepare_path`, lines 90–110, reconstructs the path and checks its tangent against the saved tangent; `validate_scan_geometry`, lines 113–126, checks the internal coordinates. `configured_rhf`, lines 138–144, sets conventional RHF; `solve_point`, lines 199–282, performs the SCF and consistency checks. Continuation is **transported and reorthonormalized orbitals**, not blindly copied density:

[tests/rhf_bound_level_selection.py, lines 219–225](/Users/novaz/Desktop/qml2.0/tests/rhf_bound_level_selection.py:219)

```python
metric = previous.mo_coeff.T @ mf.get_ovlp() @ previous.mo_coeff
metric_values, metric_vectors = np.linalg.eigh(metric)
if np.min(metric_values) <= 1e-10:
    raise ValueError("Transported orbital overlap is singular.")
inverse_sqrt = (metric_vectors / np.sqrt(metric_values)) @ metric_vectors.T
mf.mo_coeff = previous.mo_coeff @ inverse_sqrt
mf.mo_occ, mf.mo_energy = previous.mo_occ.copy(), previous.mo_energy.copy()
```

`solve_point` lines 228–235 and 237–266 then polish/restart and, when needed, repair an internally unstable RHF state. This internal **orbital** stability Hessian is separate from the nuclear Cartesian Hessian in section C.

### 3. VARIABLE-TO-SYMBOL MAP

Here \(N\) is the number of nuclei, \(B\) the number of atomic-orbital basis functions, \(M\) the number of molecular orbitals, and \(d=N_{\rm occ}N_{\rm virt}\).

| Code variable | Symbol | Shape | Units |
|---|---|---|---|
| `lower`, `upper`, `length`, `path.s_eq` | \(r_{\min},r_{\max},r_i,r_e\) | scalars | Å |
| `grid` | \((r_0,\ldots,r_{29})\) | Python list `(30,)` | Å |
| `q` | \(q_i\) | scalar | Å |
| `coords`, `cartesian_A` | \(R_{A\alpha}(q_i)\) | array/list `(N,3)` | Å |
| `previous.mo_coeff`, `mf.mo_coeff` | \(C_{\rm old},C_{\rm new}\) | `(B,M)` | AO expansion coefficients |
| `mf.get_ovlp()` | \(S\) | `(B,B)` | dimensionless overlaps |
| `metric`, `inverse_sqrt` | \(G,G^{-1/2}\) | `(M,M)` | dimensionless |
| `metric_values`, `metric_vectors` | \(g_j,V_{ij}\) | `(M,)`, `(M,M)` | dimensionless |
| `mf.mo_occ` | \(f_j\) | `(M,)` | electron count; 0 or 2 |
| `mf.make_rdm1()` | \(D\) | `(B,B)` | density-matrix representation |
| orbital `gradient`, `step` | \(g_{ia},\delta\theta_{ia}\) | `(d,)` | Ha, dimensionless rotation |
| orbital `matrix`, `vectors`, `eigenvalues` | \(K,V,\lambda\) | `(d,d)`, `(d,d)`, `(d,)` | Ha, dimensionless, Ha |
| `state["energy_Eh"]`, `E_RHF_Ha` | \(U_i\) | scalar | Ha |
| `delta_E_RHF_Ha` | \(U_i-E_{\rm re}\) | scalar | Ha |
| `result["points"]` | sampled records | list length 30 | mixed fields |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

Grid construction is literal scalar affine interpolation:

```text
Python: lower + (upper-lower)*index/(POINT_COUNT-1)
Index:  r_i = r_min + (r_max-r_min)i/29, i=0,...,29
Math:   a uniform closed interval with Δr=(r_max-r_min)/29
```

The explicit assignments `grid[0], grid[-1]=lower,upper` restore the exact saved floating-point endpoints. There is no endpoint extension, rounding to a display precision, or extra equilibrium sample in the scan. Distinctness is checked using strict `<`.

```text
Python: q = length - path.s_eq; coords = path(q)
Index:  q_i = r_i-r_e; coords[A,alpha] = R_{A,alpha}(q_i)
Math:   U_i = E_RHF(R(r_i-r_e))
```

For geometry validation, the expected internal coordinate is `original+q` for a stretch and `original` for a fixed coordinate. Bond errors are scalar differences; fixed torsion errors use `angular_difference` to account for periodic angles. Tolerances are scaled upward for large coordinates:

\[
\epsilon_\ell=\max(5\times10^{-10},256\epsilon_{\rm float}\max(1,\max_{A\alpha}|R_{A\alpha}|))\ \text{Å},
\]

\[
\epsilon_\theta=\max\left(2\times10^{-7},\frac{180}{\pi}\frac{8\epsilon_\ell}{\ell_{\rm fixed,min}}\right)\ \text{degrees}.
\]

If there are no fixed bonds the implementation uses `1.` as the fallback length scale. These tolerances assess geometry; they do not alter it.

Orbital continuation constructs

\[
G_{ij}=\sum_{\mu\nu}C^{\rm old}_{\mu i}S^{\rm new}_{\mu\nu}C^{\rm old}_{\nu j},\quad
G=V\operatorname{diag}(g_j)V^{\mathsf T},
\]

\[
(G^{-1/2})_{ij}=\sum_k V_{ik}g_k^{-1/2}V_{jk},\qquad
C^{\rm new}=C^{\rm old}G^{-1/2}.
\]

NumPy's `metric_vectors / np.sqrt(metric_values)` divides column \(j\) by \(\sqrt{g_j}\), which implements the expression above. The singularity test rejects \(g_{\min}\le10^{-10}\). Orbitals and occupations are then refined at the new geometry; this continuation does not substitute a previous energy.

The optional polishing Newton step (`polish_orbitals`, lines 170–175) uses the truncated inverse of the **orbital** Hessian:

\[
\delta\theta=-\sum_{j:|\lambda_j|>2\times10^{-10}}v_j\frac{v_j^{\mathsf T}g}{\lambda_j},\qquad
\delta\theta\leftarrow\delta\theta\min\left(1,\frac{0.1}{\max(\|\delta\theta\|,10^{-30})}\right).
\]

This clips the rotation-step norm to at most 0.1; it does not clip a bond displacement or energy. `expm(unpack_uniq_var(step,mo_occ))` rotates the occupied/virtual orbital blocks. `orbital_stability`, lines 129–135, constructs columns `2*operator(e_j)` and diagonalizes `(matrix+matrix.T)/2`, explicitly symmetrizing this **orbital** matrix. An internally stable result has minimum eigenvalue \(\ge-10^{-7}\) Ha. Up to eight repairs try rotations \(\pm0.35v_{\min}\), choose the lower-energy trial, and restart Newton refinement. Thus this is a numerical stationarity/stability test with a finite tolerance, not a mathematical proof of the global RHF minimum.

The electronic acceptance checks map to

\[
\mathrm{Tr}(DS)=\sum_{\mu\nu}D_{\mu\nu}S_{\nu\mu}=N_e,
\qquad C^{\mathsf T}SC=I,
\]

with electron-count error `<1e-8`, maximum orthogonality error `<1e-9`, occupations in `{0,2}`, finite energy, no density fitting, and orbital gradient norm `<=1e-10` Ha. Polishing additionally requires gradient `<=1e-11` Ha and consecutive energy change `<=1e-13` Ha (lines 157–165). Failed points are retained with an error record, and `previous=None` prevents them from seeding the next calculation.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

`load_inputs`, lines 127–148, reads the Part 3 JSON and harmonic JSON. It requires the current schema, matching harmonic-file hash, matching k/mass/frequency fields, and recomputes \(\hbar\omega/E_h\) solely as a consistency check. `validate_range_record`, lines 68–111, requires the already saved `selected_n` and endpoints; a `None` result is rejected rather than assigned an invented scan range.

`engine.prepare_path` loads `symbols`, `equilibrium_cartesian_A`, `isotope_average_masses_amu`, and the basis from the harmonic record, constructs `StretchPath`, validates it, and checks the reconstructed `(N,3)` tangent against saved `tangent_dR_dq` with `atol=2e-8, rtol=0`. It verifies `s_eq` within `1e-10` Å. The original saved coordinate, not a diatomic substitute, therefore supplies every geometry.

`make_molecule` in [tests/new_rhf_harmonic_bond_ranges.py:326](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py:326) passes `unit="Angstrom", charge=0, spin=0, symmetry=False, cart=False` to PySCF. The stored coordinates enter as Å; PySCF's molecular object performs its internal Bohr conversion.

The scanner invokes `solve_point` with `previous`, not `density`. The first point uses `fresh_minao`; subsequent successful points first try transported orbitals and retain a fresh-MINAO fallback. Raw RHF calculations are performed only in the normal branch of `main` (lines 376–426). The `--rebind-part3` branch returns before importing the numerical engine (lines 371–375).

### 6. UNITS TRACE

```text
upper-lower                         Å - Å = Å
index/(30-1)                        dimensionless
lower+(upper-lower)*index/29         Å
q=length-path.s_eq                   Å - Å = Å
path(q)                             Cartesian coordinates in Å
PySCF molecule(unit='Angstrom')      converts to its internal atomic units
mf.e_tot -> state['energy_Eh']       absolute molecular RHF energy in Ha
energy-summary['E_re_Ha']            Ha - Ha = Ha (diagnostic only)
C_old.T @ S_new @ C_old              dimensionless metric
inverse_sqrt                        dimensionless
orbital Hessian @ rotation          Ha × dimensionless = Ha gradient
orbital Newton step                 Ha/Ha = dimensionless angle parameter
```

No Hartree-to-joule conversion occurs in the scanner. The `1e-10` gradient threshold here is an **orbital-rotation energy gradient** in Ha, not a nuclear gradient in Ha/Bohr.

### 7. SAVED DATA / HANDOFF

```text
Part3 summary.selected_range_min_A/max_A
  -> build_grid -> point.bond_length_A
Part3 summary.r_e_A + harmonic saved geometry
  -> prepare_path / path(q) -> point.q_A, point.cartesian_A
solve_point state.energy_Eh
  -> point.E_RHF_Ha
  -> results/rhf_30_point_scans/rhf_scan_30.json results[*].points[*]
  -> combined rhf_scan_30.csv, molecules/<molecule>.csv
  -> plot_rhf_bond_scan_30.load_plot_data -> absolute plotted point
point.cartesian_A
  -> JSON and geometries/<molecule>.xyz
state convergence/stability diagnostics
  -> point.diagnostics and status fields -> plotting acceptance gate
```

`write_outputs`, lines 225–242, checkpoints JSON/CSVs after each point and writes XYZ coordinates. JSON retains full coordinates and diagnostics; the point CSV has only the fields declared at scanner lines 31–35.

**Provenance audit, not a fresh calculation:** the current saved scan is `PASS` with 270 accepted samples, but its numerical-run scanner and solver hashes differ from today's source hashes. The file deliberately retains these historical hashes. Its geometry-source hash still matches today's harmonic builder. Two `part3_metadata_migrations` records have `SCF_run:false`, `points_modified:false`. Their before/after numerical hash and a hash recomputed from the current saved point payload all equal:

```text
444b5aac1e173d9a50994184c06f18b015ab2ca6c1cf09f562d26bbc2e99c5ea
```

`rebind_saved_scan`, lines 254–349, verifies unchanged ranges, selected level, basis, equilibrium reference, physical inputs, harmonic/source hashes, all 30 coordinates, and diagnostics. It replaces only each summary and provenance metadata and asserts the complete point-payload digest stays unchanged. It does not recalculate or independently certify the saved SCF solutions. The existing saved energies must therefore be identified as **reused historical numerical results**, not results newly produced by the current source during this audit.

### 8. NUMERICAL WALKTHROUGH

LiH uses saved RHF/STO-3G data. These arithmetic evaluations read saved numbers; no electronic calculation was executed:

```text
Python numerical operation:
  step = (1.7588508418515771 - 1.262772864212423)/29
       = 0.017106137159970832 Å
Formula:
  Δr = (r_max-r_min)/29

Python numerical operation:
  r[i] = 1.262772864212423
         + (1.7588508418515771-1.262772864212423)*i/29
  q[i] = r[i]-1.510811853032
Formula:
  r_i = r_min+iΔr; q_i=r_i-r_e
```

CSV/JSON `point_index=i+1`, hence the following rows use 1,15,16,30:

| Saved point | \(i\) | \(r_i\) [Å] | \(q_i\) [Å] | Saved \(U_i\) [Ha] | \(E_{re}+0.5kq_i^2\) [Ha] | \(U_i-V_{harm}\) [Ha] |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1.262772864212423 | -0.2480389888195771 | -7.846905563298792 | -7.850616193901642 | 0.003710630602849818 |
| 15 | 14 | 1.5022587844520146 | -0.008553068579985501 | -7.863366825894265 | -7.863366949450212 | 0.0000001235559468071301 |
| 16 | 15 | 1.5193649216119853 | 0.008553068579985279 | -7.8633670714604085 | -7.863366949450212 | -0.0000001220101966126208 |
| 30 | 29 | 1.7588508418515771 | 0.2480389888195771 | -7.853198586047023 | -7.850616193901642 | -0.0025823921453813625 |

Every harmonic column evaluates the explicit Python arithmetic `-7.863382128921103 + 0.5*0.414994862916199*q[i]**2`, corresponding to \(E_{\rm re}+\tfrac12kq_i^2\). The RHF values are **lookups**, not values derived from this parabola. At point 1, saved Li/H coordinates are `[[0,0,-0.7239485409183487],[0,0,0.5388243232940743]]` Å; at point 30 they are `[[0,0,-0.7868633121136515],[0,0,0.9719875297379256]]` Å. Their z-differences recover the endpoint bond lengths. The first point's orbital gradient is `9.552380327402676e-13` Ha, its lowest internal orbital-stability eigenvalue is `0.7316011552895147` Ha, and its saved electron count is `3.999999999999999`. These are recorded acceptance evidence, not rerun checks.

Because 30 is even and the range is symmetric, neither middle scan row has \(q=0\). The selected harmonic endpoints lie at the harmonic \(n=1\) energy, but the actual RHF endpoint energies are different and asymmetric. The samples are not being used to redetermine \(n\), fit \(k\), solve a vibrational Schrödinger equation, or locate anharmonic turning points.

## K. Plotting

### 1. PHYSICAL / MATHEMATICAL QUANTITY

The current overlay uses one **absolute** Hartree energy coordinate:

\[
(r_i,U_i),\quad(r_e,E_{\rm re}),\quad
V_{\rm harm}(r)=E_{\rm re}+\frac12k(r-r_e)^2,\quad
L_n=E_{\rm re}+(n+\tfrac12)\hbar\omega.
\]

The level segment extends from \(r_e-A_n\) to \(r_e+A_n\). The shaded interval is the saved selected range. The scan supplies 30 points; the plot now adds the already saved equilibrium as the **31st orange marker**. This is an actual source change from the earlier explanation that described only 30 plotted RHF markers.

### 2. EXACT SOURCE IMPLEMENTATION

[codes/plot_rhf_bond_scan_30.py](/Users/novaz/Desktop/qml2.0/codes/plot_rhf_bond_scan_30.py:116), `local_panel`, lines 116–153:

[codes/plot_rhf_bond_scan_30.py, lines 119–134](/Users/novaz/Desktop/qml2.0/codes/plot_rhf_bond_scan_30.py:119)

```python
width = 1.28 * s["Delta_q_n1_A"]
q = [-width + 2 * width * i / 500 for i in range(501)]
model = [s["E_re_Ha"] + 0.5 * s["k_Ha_per_A2"] * displacement**2 for displacement in q]
ax.axvspan(s["selected_range_min_A"], s["selected_range_max_A"],
           color=GREEN, alpha=.12, label="Selected bond range")
ax.plot([s["r_e_A"] + displacement for displacement in q], model,
        color=BLUE, ls="--", lw=1.7, label="Local harmonic model")
rhf_curve = sorted([(p["bond_length_A"], p["E_RHF_Ha"]) for p in points]
                   + [(s["r_e_A"], s["E_re_Ha"])])
ax.plot([r for r, energy in rhf_curve], [energy for r, energy in rhf_curve],
        "o-", color=ORANGE, lw=1.05, ms=3.5, markeredgecolor="white", markeredgewidth=.35,
        zorder=4, label="RHF calculated points")
for n, color in ((0, PURPLE), (1, GREEN)):
    level = s[f"E_total_n{n}_Ha"]
    ax.hlines(level, s[f"n{n}_min_A"], s[f"n{n}_max_A"], color=color, lw=1.8)
    ax.scatter([s[f"n{n}_min_A"], s[f"n{n}_max_A"]], [level, level], s=16, color=color, zorder=5)
```

Lines 138–149 set bounds and labels:

[codes/plot_rhf_bond_scan_30.py, lines 138–149](/Users/novaz/Desktop/qml2.0/codes/plot_rhf_bond_scan_30.py:138)

```python
energies = [s["E_re_Ha"], *model, *(p["E_RHF_Ha"] for p in points),
            s["E_total_n0_Ha"], s["E_total_n1_Ha"]]
low, high = min(energies), max(energies)
span = max(high - low, 1e-12)
ax.set_xlim(s["r_e_A"] - width, s["r_e_A"] + width)
ax.set_ylim(low - .055 * span, high + .30 * span)  # Include RHF compression-side excursions.
ax.set_title(NAMES.get(s["molecule"], s["molecule"]), loc="left", fontweight="semibold")
ax.set_xlabel(COORDINATES[s["molecule"]] + " (Å)")
ax.set_ylabel(r"$E$ (Hartree)")
ax.xaxis.set_major_locator(ticker(5))
ax.yaxis.set_major_locator(ticker(5))
ax.ticklabel_format(axis="y", style="plain", useOffset=False)
```

`load_plot_data`, lines 37–113, validates input summaries, the grid, and all energies before creating plotted records. [tests/plot.py:10](/Users/novaz/Desktop/qml2.0/tests/plot.py:10) is only a compatibility launcher importing this script's `main`; it contains no independent physics or plotting calculations.

### 3. VARIABLE-TO-SYMBOL MAP

| Code variable | Symbol | Shape | Units |
|---|---|---|---|
| `s` | Part 3 summary | mapping; scalar fields | field dependent |
| `points` | RHF scan samples | list of 30 records | mixed |
| `width` | \(w=1.28A_1\) | scalar | Å |
| `q` | display \(q_j\) | list `(501,)` | Å |
| `model` | display \(V_j\) | list `(501,)` | Ha |
| plot x list `r_e+q` | display \(r_j\) | list `(501,)` | Å |
| `rhf_curve` | \((r,U)\) ordered pairs | list `(31,2)` for current data | Å, Ha columns |
| `level` | \(L_n\) | scalar | Ha |
| level endpoints | \((r_{n,-},r_{n,+})\) | two scalars | Å |
| `energies` | values used for limits | list `(534,)` = 1+501+30+2 | Ha |
| `low`, `high`, `span` | \(y_{lo},y_{hi},\Delta y\) | scalars | Ha |
| `E_RHF_minus_E_re_Ha` | \(U_i-E_{\rm re}\) | scalar per CSV row | Ha, diagnostic only |

### 4. HOW THE CODE IMPLEMENTS THE EQUATION

```text
Python: width=1.28*A1; q[j]=-width+2*width*j/500
Index:  q_j=-1.28A1+(2.56A1)j/500, j=0,...,500
Math:   501 display evaluations over [-1.28A1,+1.28A1]

Python: model[j]=E_re+0.5*k_A*q[j]**2
Index:  V_j=E_re+(1/2)k_A q_j q_j
Math:   harmonic potential on an absolute energy reference

Python: plot_x[j]=r_e+q[j]
Index:  r_j=r_e+q_j
Math:   plot displacement model against absolute bond length
```

This is direct evaluation of an analytic parabola, not a fit, spline, interpolation of the RHF energies, or 501 new SCF calculations.

```text
Python: sorted([(r_i,U_i) for points] + [(r_e,E_re)])
Index:  order the 31 pairs by r
Math:   union of 30 sampled potential points and one saved minimum
```

`"o-"` draws circular markers and straight segments between these existing points. These connecting lines guide the eye; they do not provide new computed energy data. The equilibrium point is reused from Part 1/2, not inferred from the even-sized grid.

```text
Python: hlines(L_n,n_min,n_max)
Math:   {(r,L_n) : r ∈ [r_e-A_n,r_e+A_n]}
Python: axvspan(selected_min,selected_max)
Math:   shade the saved selected interval, for every displayed y
```

The y bounds use all modeled energies, all 30 RHF energies, the equilibrium energy, and both levels:

\[
y_{lo}=\min\mathcal E,\quad y_{hi}=\max\mathcal E,\quad
s_y=\max(y_{hi}-y_{lo},10^{-12}\ \mathrm{Ha}),
\]

\[
[y_{\min},y_{\max}]=[y_{lo}-0.055s_y,\ y_{hi}+0.30s_y].
\]

The `1e-12` floor prevents a collapsed vertical window. The 0.055/0.30 padding changes display limits only. No energy values or ranges are clipped or transformed. Axis tick offsets are explicitly disabled; negative absolute ticks are shown directly. `n0_pass`/`n1_pass` are checked in `load_plot_data`, but the plotting routine does not alter the selection.

### 5. HOW THE INPUT OBJECT IS CONSTRUCTED

`load_plot_data` reads `rhf_scan_30.json` and `vibrational_levels.json`. It checks the Part 3 hash against the scan's current `provenance.ranges_sha256` (lines 41–46). It then requires exact agreement between matching summary fields (lines 64–69), unshifted reference and warning fields (71–74), and all 30 successful/stable geometry records (91–106).

The loaded summary supplies curvature, amplitudes, ranges, and totals. They are **reused**. Only consistency identities are recalculated:

[codes/plot_rhf_bond_scan_30.py, lines 84–90](/Users/novaz/Desktop/qml2.0/codes/plot_rhf_bond_scan_30.py:84)

```python
    equal_number(excitation, (n + 0.5) * quantum, f"n={n} excitation")
    equal_number(0.5 * curvature * amplitude**2, excitation, f"n={n} harmonic units")
    equal_number(current[f"E_total_n{n}_Ha"], re + excitation, f"n={n} absolute level")
    require(current[f"n{n}_pass"] is (current[f"E_total_n{n}_Ha"] < 0),
            f"{name}: incorrect n={n} zero-test result.")
selected = 1 if current["n1_pass"] else 0 if current["n0_pass"] else None
require(current["selected_n"] == selected, f"{name}: wrong selected n.")
```

`equal_number` is imported from the scanner and uses a default absolute tolerance `1e-12` in the quantity's units; here the compared values are Hartree. This tolerance does not soften the sign criterion, whose Boolean is rechecked with literal `<0` at plot lines 87–90. No solver is imported or executed by this plotting path.

### 6. UNITS TRACE

```text
width=1.28*Delta_q_n1_A               dimensionless × Å = Å
q_j=-width+2*width*j/500             Å + Å × dimensionless = Å
r_e_A+q_j                          Å + Å = Å
q_j**2                             Å²
k_Ha_per_A2*q_j**2                  (Ha/Å²) × Å² = Ha
E_re_Ha+0.5*k_Ha_per_A2*q_j**2      Ha + Ha = Ha
E_total_n0_Ha / E_total_n1_Ha       saved absolute Ha
point.E_RHF_Ha                     saved absolute Ha
energy-re                          Ha-Ha=Ha, checked CSV diagnostic only
high-low; padded limits            Ha; dimensionless multipliers × Ha
```

There is no hidden subtraction of `E_re`, a sampled minimum, or any dissociation reference from plotted energies. The label is `$E$ (Hartree)`; the physical zero warning in the generated notes still says that zero is not validated as the vibrational threshold.

### 7. SAVED DATA / HANDOFF

```text
scan.results[*].points[*].E_RHF_Ha
  -> load_plot_data -> results[*].points[*].E_RHF_Ha
  -> local_panel rhf_curve (plus saved equilibrium)
Part3 summary.E_re_Ha,k_Ha_per_A2,r_e_A,Delta_q_n1_A
  -> local_panel analytic model
Part3 summary.E_total_n{n}_Ha,n{n}_min_A,n{n}_max_A
  -> horizontal level segments and endpoint markers
write_plots
  -> results/plots/rhf_30_overlay/overview.png/.svg
  -> <molecule>.png/.svg
main
  -> plotted_points.csv, README.md
```

`write_plots`, lines 156–202, renders a 3-column overview and individual panels; `main`, lines 205–246, writes the CSV and notes. `plotted_points.csv` contains exactly the original 270 scan rows, **not** the nine extra equilibrium points displayed by `local_panel`. Its relative-energy column remains only a diagnostic. This distinction is explicit in the current generated README.

Saved-artifact verification during this read-only audit:

- The current LiH SVG contains the absolute ticks `−7.864`, `−7.858`, `−7.852`, `−7.846`, `−7.840` and the `E (Hartree)` label.
- SVG data group `line2d_18` contains **31** orange markers; the separate legend contains one illustrative marker. This establishes that the existing figure includes the saved equilibrium addition.
- The README scan/Part 3 hashes exactly match the current input files: scan `261a03ce80c9eac0c3166579483bd4138148faf2349138b605973f9077a4c2c5`; Part 3 `dbb2b85f3b594de3879d6aff03e5b94ccc6eddf4414f4b3e40691efc0b85f4bc`.
- The plotted CSV has 270 rows; its LiH absolute energies match the 30 saved LiH scan energies exactly.
- The plot notes record input hashes but **do not record the plotting script's source hash**. Therefore the artifact checks verify the present numerical/reference behavior, not an exact cryptographic provenance chain from a specific plotting-source revision. The README's initial phrase “no fitting or extra points” is imprecise in isolation; its next sentences explicitly identify the extra saved equilibrium point. The source itself is unambiguous.

### 8. NUMERICAL WALKTHROUGH

For LiH, the current code arithmetic is:

```text
Python: width = 1.28 * 0.24803898881957717
              = 0.31748990568905877 Å
Formula: w = 1.28 A1

Python: q[0] = -width; q[250] = -width+2*width*250/500; q[500] = width
Result: (-0.31748990568905877, 0.0, 0.31748990568905877) Å
Formula: q_j=-w+2wj/500

Python: x_limits = (1.510811853032-width, 1.510811853032+width)
Result: (1.1933219473429413, 1.8283017587210588) Å
Formula: [r_e-w,r_e+w]

Python: model[0] = -7.863382128921103
                   +0.5*0.414994862916199*(-0.31748990568905877)**2
Result: -7.842466420985218 Ha
Formula: V_harm(-w)=E_re+(1/2)kw²

Python: model[250] = -7.863382128921103+0.5*0.414994862916199*0.0**2
Result: -7.863382128921103 Ha
Formula: V_harm(0)=E_re

Python: level0 = -7.863382128921103+0.004255311673153625
Result: -7.859126817247949 Ha
Formula: L0=E_re+E_n0

Python: level1 = -7.863382128921103+0.012765935019460876
Result: -7.850616193901642 Ha
Formula: L1=E_re+E_n1

Python: low=min(energies); high=max(energies)
Result: low=-7.863382128921103; high=-7.842466420985218 Ha
Python: span=max(high-low,1e-12)=0.020915707935884598 Ha
Python: y_limits=(low-.055*span, high+.30*span)
Result: (-7.864532492857577, -7.836191708604453) Ha
Formula: [min(E)-0.055Δ, max(E)+0.30Δ]
```

The 31st plotted pair is exactly `(1.510811853032, -7.863382128921103)`, reusing saved \((r_e,E_{\rm re})\). Section J's first/middle/last RHF values provide the other examples on the orange curve. At both selected endpoints the harmonic level is `-7.850616193901642` Ha, whereas the sampled RHF values are `-7.846905563298792` and `-7.853198586047023` Ha; the plot deliberately displays that harmonic-versus-RHF difference instead of moving either endpoint or redefining the vibrational levels.


## Verification ledger and preservation check

The following is an audit-only recomputation from persisted arrays and scalars, using NumPy in the saved virtual environment with bytecode writing disabled. The audit did not call `calculate`, `run_rhf`, `mf.kernel`, `mf.Hessian().kernel`, workflow `main`, or any output writer. Geometry-only `StretchPath` operations were permitted to reconstruct the finite differences in A/B. Reproducing an old contraction confirms arithmetic consistency; it does not replace the physical approximations or independently verify the electronic wavefunction.

| Molecule | N | H shape | t shape | Δk [Ha/Bohr²] | Δmu [amu] | relative Δomega | ΔQ [Ha] | ΔE_total,0 [Ha] | ΔE_total,1 [Ha] | selection agrees |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| LiH | 2 | (2,2,3,3) | (2,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| BeH2 | 3 | (3,3,3,3) | (3,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| H2O | 3 | (3,3,3,3) | (3,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| NH3 | 4 | (4,4,3,3) | (4,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| N2 | 2 | (2,2,3,3) | (2,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| CO | 2 | (2,2,3,3) | (2,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| HF | 2 | (2,2,3,3) | (2,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| H2S | 3 | (3,3,3,3) | (3,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |
| H2O2 | 4 | (4,4,3,3) | (4,3) | 0 | 0 | 0 | 0 | 0 | 0 | YES |

Each Δ is the recomputed value minus its corresponding saved value (frequency uses an absolute relative discrepancy). All table zeros are actual floating-point zeros for these operations, not merely rounded small discrepancies. This does not mean the underlying RHF/finite-difference approximation has zero physical or numerical error.

**Source/data identity.** The current harmonic source hash matches `runtime.script_sha256`; the current standalone Part3 source hash matches its `provenance.script_sha256`. Installed `inspect.getsource` hashes match the harmonic runtime records for the PySCF RHF Hessian module, `thermo.harmonic_analysis`, and `gto.mole.atom_mass_list`. Part3's saved equilibrium CSV and harmonic-JSON hashes match the current files. The scan's original numerical source hashes require the historical qualification documented in J; metadata migrations do not turn historical SCF data into a fresh calculation.

| Audited file | Current SHA256 |
|---|---|
| [tests/molecules_rhf.py](/Users/novaz/Desktop/qml2.0/tests/molecules_rhf.py) | `4319004bcf0798c3cc6989567d749e08c3cdfaccf79f51ee9abd84839fcea15f` |
| [tests/new_rhf_harmonic_bond_ranges.py](/Users/novaz/Desktop/qml2.0/tests/new_rhf_harmonic_bond_ranges.py) | `e6687127ab5752b7a203cab27f177572774e85e7a71acec9f4b617e699b17bd9` |
| [codes/bond_length_part3.py](/Users/novaz/Desktop/qml2.0/codes/bond_length_part3.py) | `65d7bb3dbcec58125bc9754e3fcb25addd65b9b1fed8c3475b0df31198b30cfd` |
| [codes/rhf_bond_scan_30.py](/Users/novaz/Desktop/qml2.0/codes/rhf_bond_scan_30.py) | `70f363601927df03d96972b8828afe62a4818143165c782bbe53cb1e2f9fd3ba` |
| [codes/plot_rhf_bond_scan_30.py](/Users/novaz/Desktop/qml2.0/codes/plot_rhf_bond_scan_30.py) | `63f3d0cc29b6352304bc66dae01b347d4b9c793286b1ebc494faaeb0a45f1097` |
| [tests/rhf_bound_level_selection.py](/Users/novaz/Desktop/qml2.0/tests/rhf_bound_level_selection.py) | `6136f7685523f2fdcd91396a5d3f136709c912bf0be91c70590a614fba5b11a5` |
| [codes/bond_length.py](/Users/novaz/Desktop/qml2.0/codes/bond_length.py) | `511f352d4b7f9866190571ed32416aa2e72216e1d39029669d5ee46847505549` |
| [results/rhf_geometries/rhf_equilibrium_summary.csv](/Users/novaz/Desktop/qml2.0/results/rhf_geometries/rhf_equilibrium_summary.csv) | `6c14f2ec907d5ec5ec3404c380c5dcb504eed0941a2f49891ba403e8644c3396` |
| [json/new_rhf_harmonic_bond_ranges.json](/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json) | `d698dde31c8252930dcd39aeea1721c5f4bde42efc81d3571a566a88cbfb5999` |
| [results/bond_length_part3/vibrational_levels.json](/Users/novaz/Desktop/qml2.0/results/bond_length_part3/vibrational_levels.json) | `dbb2b85f3b594de3879d6aff03e5b94ccc6eddf4414f4b3e40691efc0b85f4bc` |
| [results/rhf_30_point_scans/rhf_scan_30.json](/Users/novaz/Desktop/qml2.0/results/rhf_30_point_scans/rhf_scan_30.json) | `261a03ce80c9eac0c3166579483bd4138148faf2349138b605973f9077a4c2c5` |

**Read-only verification:** 225 of 231 inventoried pre-existing non-Finder files under `codes/`, `tests/`, `json/`, and `results/` have identical before/after SHA256 values. The specifically audited workflow sources, equilibrium CSV, harmonic JSON, Part3 JSON, and scan JSON all match their before-audit hashes. This audit writes only the new Markdown report under `md/`; it did not modify any source or existing scientific result.

The shared workspace was also active elsewhere during this audit. Unrelated correlation/FCI additions, plot artifacts, and Finder metadata appeared or changed. The following pre-existing files also changed concurrently; this audit did not write or revert them. The preservation statement concerns the specifically audited workflow files, not a claim that the entire shared repository was idle:

- `codes/bond_length_part3_plots.py`
- `codes/rhf_reference_correlation_audit.py`
- `results/rhf_reference_correlation/REPORT.md`
- `results/rhf_reference_correlation/per_geometry.csv`
- `results/rhf_reference_correlation/per_molecule_summary.csv`
- `results/rhf_reference_correlation/provenance.json`


## Consolidated LiH handoff trace

A–I give the individual derivations, arrays, exact source excerpts and units. This final ledger collects the requested values in execution order. Each saved lookup is marked separately from arithmetic; the full Hessian and expanded projection are in C/D.

| Item | Numerical Python operation or lookup | Formula and value |
|---|---|---|
| Equilibrium coordinate | `path.s_eq = mean(path.lengths)` | r_e = 1.510811853032 Å |
| Original equilibrium energy | `float(metadata['energy_hartree'])` | E_re = -7.863382128921103 Ha |
| Tangent | `(path(0.0002)-path(-0.0002))/(2*0.0002)` | t ≈ [[0.0, 0.0, -0.12682435832939154], [0.0, 0.0, 0.8731756416702208]] (dimensionless) |
| Hessian | `array(record['cartesian_hessian_Eh_per_Bohr2'])` (saved lookup) | H[a,b,x,y], shape(2,2,3,3), Ha/Bohr²; numerical matrix in C.8 |
| Projected k | `einsum('ax,abxy,by->',t,H,t)` | Σ_axby t_ax H_abxy t_by = 0.11621039750120737 Ha/Bohr² |
| Effective mass | `einsum('a,ax,ax->',m,t,t)` | Σ_ax m_a t_ax² = 0.880161046803545 amu |
| SI curvature | `0.11621039750120737*4.359744644911914e-18/(5.2917721092e-11)**2` | k_SI = 180.92716312648525 N/m |
| SI mass | `0.880161046803545*1.660539040427164e-27` | mu_kg = 1.461541780080527e-27 kg |
| Angular frequency | `sqrt(180.92716312648525/1.461541780080527e-27)` | omega = 351840856677855.3 rad/s |
| A0 (constructed in Part2) | `sqrt(1.0545718001391127e-34/(1.461541780080527e-27*351840856677855.3))/1e-10` | sqrt(hbar/(mu*omega)) = 0.14320537696450547 Å |
| A1 (constructed in Part2) | `sqrt(3)*0.14320537696450547` | sqrt(3)*A0 = 0.24803898881957717 Å |
| Q (constructed in Part3) | `1.0545718001391127e-34*351840856677855.3/4.359744644911914e-18` | hbar*omega/C_E = 0.00851062334630725 Ha |
| E_n0 | `0.5*0.00851062334630725` | Q/2 = 0.004255311673153625 Ha |
| E_n1 | `1.5*0.00851062334630725` | 3Q/2 = 0.012765935019460876 Ha |
| E_total_n0 | `-7.863382128921103+0.004255311673153625` | E_re+E_n0 = -7.859126817247949 Ha |
| E_total_n1 | `-7.863382128921103+0.012765935019460876` | E_re+E_n1 = -7.850616193901642 Ha |
| Final selected_n | `1 if -7.850616193901642<0 else 0 if -7.859126817247949<0 else None` | selected_n =1; both literal sign tests True |
| Selected r_min | `hs['n1_min_A']` (saved lookup; originally `1.510811853032-0.24803898881957717`) | r_e-A1 = 1.262772864212423 Å |
| Selected r_max | `hs['n1_max_A']` (saved lookup; originally `1.510811853032+0.24803898881957717`) | r_e+A1 = 1.7588508418515771 Å |

The geometry/tangent/mass portion is repeated numerically for H₂O in A.8, B.8 and E.8, including the oxygen motion introduced by alignment and the separate mass contribution of each nucleus.
