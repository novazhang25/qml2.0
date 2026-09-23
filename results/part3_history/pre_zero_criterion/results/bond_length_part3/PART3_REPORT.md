# Corrected Part 3: n=0 / n=1 vibrational levels

## Audit

The old selector in tests/rhf_bound_level_selection.py:57-84 used
E_n_rel = -D_e + (n+1/2)*hbar*omega and tested E_n_rel < 0. With
D_e = E_diss - E_eq (lines 456-458), this is algebraically correct: E_n_rel
is the signed gap E_total(n)-E_diss, not an absolute vibrational-level energy.
There is no evidence in the audited implementation of comparing E_n directly
against absolute E_diss, using E_diss-E_n as a level, or subtracting E_re twice.

codes/bond_length.py:375-377 reused that selector; lines 433-440 documented
the dissociation-zero convention. The missing outputs were E_total_0 and
E_total_1 and an independent absolute-versus-relative consistency test.
tests/plot.py:82-91 and 110-134 used a consistent E-E_eq axis with threshold
D_e; lines 197-198 called the dissociation-relative gaps E0/E1, an ambiguous
label under the requested notation. Its current main generated only the local
overview, without a threshold comparison. Those gaps must not be presented
as either E_n above equilibrium or absolute E_total(n).

The replacement uses the optimized energy from Part 1's metadata. The old
selector used a fresh q=0 SCF value (tests/rhf_bound_level_selection.py:394),
which differs from the saved optimized energy only by numerical roundoff in
the present data. Both saved reproductions are checked against Part 1, but
neither replaces it. Parts 1 and 2 and the saved plateau are unchanged.

## Energy definitions and units

E_re is the absolute optimized RHF equilibrium energy. E_diss is the unchanged absolute validated RHF plateau energy. D_e is a relative well depth. E_n is excitation energy above E_re, including zero-point energy. E_total(n) is the absolute level energy.

```text
k_SI = k[Ha/Bohr^2] * Hartree_J / Bohr_m^2
mu_kg = mu_eff[amu] * amu_kg
omega = sqrt(k_SI / mu_kg)  [rad/s]
E_n = (n + 1/2) * hbar_J_s * omega / Hartree_J  [Ha]
E_total(n) = E_re + E_n  [Ha]
D_e = E_diss - E_re  [Ha]
bound(n) iff E_total(n) < E_diss iff E_n < D_e
(E_re + E_n) - E_diss = E_n - D_e
```

The signed gaps have the SAME sign; their absolute values are equal. A positive binding margin E_diss-E_total(n) has the opposite sign. Equality at the threshold is NO. Classification uses strict inequalities without rounding or tolerance offsets.

The original collective mass sum_A m_A |dR_A/dq|^2 is checked and reused. For diatomics it equals the ordinary reduced mass; polyatomic collective paths retain their own generalized masses. No single-bond mass is substituted.

## Complete numerical table

| Molecule | E_re [Ha] | E_diss [Ha] | D_e [Ha] | k [Ha/Bohr^2] | mu_eff [amu] | omega [rad/s] | E_0 [Ha] | E_1 [Ha] | E_re + E_0 [Ha] | E_re + E_1 [Ha] | n=0 below dissociation? | n=1 below dissociation? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | -7.86338212892 | -7.51940331329 | 0.343978815629 | 0.116210397501 | 0.880161046804 | 3.51840856678e+14 | 0.00425531167315 | 0.0127659350195 | -7.85912681725 | -7.8506161939 | YES | YES |
| BeH2 | -15.5613528078 | -14.8977412836 | 0.6636115242 | 0.473247780631 | 2.016 | 4.69141527176e+14 | 0.00567399544158 | 0.0170219863247 | -15.5556788123 | -15.5443308215 | YES | YES |
| H2O | -74.9659011923 | -74.207677735 | 0.758223457273 | 1.24689394745 | 1.92283768262 | 7.7973727198e+14 | 0.00943047134088 | 0.0282914140226 | -74.956470721 | -74.9376097783 | YES | YES |
| NH3 | -55.4554197788 | -54.1832594735 | 1.27216030528 | 1.62924775325 | 2.93261189058 | 7.21723951054e+14 | 0.00872883377648 | 0.0261865013294 | -55.446690945 | -55.4292332775 | YES | YES |
| N2 | -107.500654263 | -106.727669991 | 0.772984271556 | 1.8886404537 | 7.0035 | 5.02830862315e+14 | 0.00608144846575 | 0.0182443453972 | -107.494572814 | -107.482409917 | YES | YES |
| CO | -111.225449514 | -110.751403537 | 0.4740459768 | 1.57382886196 | 6.86054941092 | 4.63771602568e+14 | 0.00560904930916 | 0.0168271479275 | -111.219840465 | -111.208622366 | YES | YES |
| HF | -98.5728473472 | -98.0112660321 | 0.561581315012 | 0.725274201861 | 0.957213059852 | 8.42852647653e+14 | 0.0101938153067 | 0.03058144592 | -98.5626535319 | -98.5422659012 | YES | YES |
| H2S | -394.311630059 | -393.578532578 | 0.733097481748 | 0.794499972345 | 1.95899192757 | 6.1664536031e+14 | 0.007457968994 | 0.022373906982 | -394.30417209 | -394.289256152 | YES | YES |
| H2O2 | -148.764996621 | -148.287014342 | 0.477982279663 | 0.709124310059 | 8.46231114857 | 2.80298994797e+14 | 0.00339005423019 | 0.0101701626906 | -148.761606567 | -148.754826459 | YES | YES |

CSV/JSON retain full float precision; the Markdown display uses 12 significant digits.

## Independent consistency checks

| Molecule | n=0 abs((E_re+E_n)-E_diss) [Ha] | n=0 abs(E_n-D_e) [Ha] | n=0 gap residual [Ha] | n=0 classifications agree? | n=1 abs((E_re+E_n)-E_diss) [Ha] | n=1 abs(E_n-D_e) [Ha] | n=1 gap residual [Ha] | n=1 classifications agree? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | 0.339723503956 | 0.339723503956 | 0 | YES | 0.331212880609 | 0.331212880609 | 0 | YES |
| BeH2 | 0.657937528758 | 0.657937528758 | 7.77156117238e-16 | YES | 0.646589537875 | 0.646589537875 | 4.4408920985e-16 | YES |
| H2O | 0.748792985932 | 0.748792985932 | 9.99200722163e-16 | YES | 0.72993204325 | 0.72993204325 | 2.88657986403e-15 | YES |
| NH3 | 1.2634314715 | 1.2634314715 | 2.6645352591e-15 | YES | 1.24597380395 | 1.24597380395 | 8.881784197e-16 | YES |
| N2 | 0.766902823091 | 0.766902823091 | 7.77156117238e-16 | YES | 0.754739926159 | 0.754739926159 | 2.33146835171e-15 | YES |
| CO | 0.468436927491 | 0.468436927491 | 3.33066907388e-15 | YES | 0.457218828873 | 0.457218828873 | 4.21884749358e-15 | YES |
| HF | 0.551387499705 | 0.551387499705 | 2.10942374679e-15 | YES | 0.530999869092 | 0.530999869092 | 6.43929354283e-15 | YES |
| H2S | 0.725639512754 | 0.725639512754 | 6.66133814775e-15 | YES | 0.710723574766 | 0.710723574766 | 1.99840144433e-14 | YES |
| H2O2 | 0.474592225433 | 0.474592225433 | 6.99440505514e-15 | YES | 0.467812116972 | 0.467812116972 | 7.38298311376e-15 | YES |

Both signed and absolute gap residuals must be within 16 floating-point ULPs of the energy scale. The two independently computed Boolean classifications must agree exactly; otherwise execution fails before writing classifications. This tolerance is unrelated to the plateau convergence tolerance. A level near that plateau tolerance is flagged NEAR_THRESHOLD without changing the strict comparison.

## Molecule-by-molecule outcome

- LiH: n=0 YES; n=1 YES; status PASS; selected n=1.
- BeH2: n=0 YES; n=1 YES; status PASS; selected n=1.
- H2O: n=0 YES; n=1 YES; status PASS; selected n=1.
- NH3: n=0 YES; n=1 YES; status PASS; selected n=1.
- N2: n=0 YES; n=1 YES; status PASS; selected n=1.
- CO: n=0 YES; n=1 YES; status PASS; selected n=1.
- HF: n=0 YES; n=1 YES; status PASS; selected n=1.
- H2S: n=0 YES; n=1 YES; status PASS; selected n=1.
- H2O2: n=0 YES; n=1 YES; status PASS; selected n=1.

These are harmonic levels compared with a constrained RHF-path threshold. The saved RHF model's dissociation limitations still apply.

## Plots

Each molecule has an absolute-energy harmonic zoom, an absolute-energy RHF scan with E_diss and E_total levels, and a clearly labeled relative-to-E_re comparison with D_e. Existing RHF points and harmonic turning points are reused; the harmonic curve is a local model.

- [plots/LiH_vibrational_levels.png](plots/LiH_vibrational_levels.png)
- [plots/LiH_vibrational_levels.svg](plots/LiH_vibrational_levels.svg)
- [plots/BeH2_vibrational_levels.png](plots/BeH2_vibrational_levels.png)
- [plots/BeH2_vibrational_levels.svg](plots/BeH2_vibrational_levels.svg)
- [plots/H2O_vibrational_levels.png](plots/H2O_vibrational_levels.png)
- [plots/H2O_vibrational_levels.svg](plots/H2O_vibrational_levels.svg)
- [plots/NH3_vibrational_levels.png](plots/NH3_vibrational_levels.png)
- [plots/NH3_vibrational_levels.svg](plots/NH3_vibrational_levels.svg)
- [plots/N2_vibrational_levels.png](plots/N2_vibrational_levels.png)
- [plots/N2_vibrational_levels.svg](plots/N2_vibrational_levels.svg)
- [plots/CO_vibrational_levels.png](plots/CO_vibrational_levels.png)
- [plots/CO_vibrational_levels.svg](plots/CO_vibrational_levels.svg)
- [plots/HF_vibrational_levels.png](plots/HF_vibrational_levels.png)
- [plots/HF_vibrational_levels.svg](plots/HF_vibrational_levels.svg)
- [plots/H2S_vibrational_levels.png](plots/H2S_vibrational_levels.png)
- [plots/H2S_vibrational_levels.svg](plots/H2S_vibrational_levels.svg)
- [plots/H2O2_vibrational_levels.png](plots/H2O2_vibrational_levels.png)
- [plots/H2O2_vibrational_levels.svg](plots/H2O2_vibrational_levels.svg)

## Provenance

```json
{
  "source_files_sha256": {
    "/Users/novaz/Desktop/qml2.0/results/rhf_geometries/rhf_equilibrium_summary.csv": "6c14f2ec907d5ec5ec3404c380c5dcb504eed0941a2f49891ba403e8644c3396",
    "/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json": "d698dde31c8252930dcd39aeea1721c5f4bde42efc81d3571a566a88cbfb5999",
    "/Users/novaz/Desktop/qml2.0/json/rhf_bound_level_selection.json": "aaf805e5cbf9a65a7b267fbcbe8d711b34560ffd208e9941b8f33c344b79ca8a"
  },
  "script_sha256": "55b4c3277931c405e5768370794f0b13b17c17aa87c34db41eabb0f77cab364b",
  "constants": {
    "Bohr_A": 0.52917721092,
    "Bohr_m": 5.2917721092e-11,
    "Hartree_J": 4.359744644911914e-18,
    "amu_kg": 1.660539040427164e-27,
    "hbar_J_s": 1.0545718001391127e-34,
    "c_m_per_s": 299792458,
    "source": "installed pyscf.data.nist"
  },
  "equilibrium_reoptimized": false,
  "force_constant_recomputed": false,
  "SCF_or_plateau_recomputed": false,
  "plot_script_sha256": "2ee9aec6103ab451877cad6fbb11388b5668168c1b6069a7f83eb1aab6b87c15"
}
```
