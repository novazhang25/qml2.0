# Complete RHF bond-length pipeline

This pipeline connects (1) RHF equilibrium geometry, (2) the collective stretching force constant, and (3) the requested n=0/n=1 zero-energy test without shifting the supplied equilibrium energy.

All energies are Hartree (Eh). Lengths are Angstrom (A). A force constant has dimensions of energy/length^2; its reported units are Eh/Bohr^2 and Eh/A^2. Atomic masses are PySCF isotope-average masses in amu.

## Summary

| molecule | r_e_A | k_q_Eh_per_Bohr2 | mu_eff_amu | omega_rad_per_s | hbar_omega_Ha | E_re_Ha | E_n0_Ha | E_n1_Ha | E_total_n0_Ha | E_total_n1_Ha | n0_pass | n1_pass | selected_n | selected_range_min_A | selected_range_max_A | validation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | 1.51081185 | 0.116210398 | 0.880161047 | 3.51840857e+14 | 0.00851062335 | -7.86338213 | 0.00425531167 | 0.012765935 | -7.85912682 | -7.85061619 | True | True | 1 | 1.26277286 | 1.75885084 | PASS |
| BeH2 | 1.29053842 | 0.473247781 | 2.016 | 4.69141527e+14 | 0.0113479909 | -15.5613528 | 0.00567399544 | 0.0170219863 | -15.5556788 | -15.5443308 | True | True | 1 | 1.14860752 | 1.43246932 | PASS |
| H2O | 0.989409176 | 1.24689395 | 1.92283768 | 7.79737272e+14 | 0.0188609427 | -74.9659012 | 0.00943047134 | 0.028291414 | -74.9564707 | -74.9376098 | True | True | 1 | 0.876681982 | 1.10213637 | PASS |
| NH3 | 1.03252271 | 1.62924775 | 2.93261189 | 7.21723951e+14 | 0.0174576676 | -55.4554198 | 0.00872883378 | 0.0261865013 | -55.4466909 | -55.4292333 | True | True | 1 | 0.937645635 | 1.12739979 | PASS |
| N2 | 1.13385101 | 1.88864045 | 7.0035 | 5.02830862e+14 | 0.0121628969 | -107.500654 | 0.00608144847 | 0.0182443454 | -107.494573 | -107.48241 | True | True | 1 | 1.06029712 | 1.20740489 | PASS |
| CO | 1.14548068 | 1.57382886 | 6.86054941 | 4.63771603e+14 | 0.0112180986 | -111.22545 | 0.00560904931 | 0.0168271479 | -111.21984 | -111.208622 | True | True | 1 | 1.0680982 | 1.22286315 | PASS |
| HF | 0.955462699 | 0.725274202 | 0.95721306 | 8.42852648e+14 | 0.0203876306 | -98.5728473 | 0.0101938153 | 0.0305814459 | -98.5626535 | -98.5422659 | True | True | 1 | 0.801790953 | 1.10913444 | PASS |
| H2S | 1.32865587 | 0.794499972 | 1.95899193 | 6.1664536e+14 | 0.014915938 | -394.31163 | 0.00745796899 | 0.022373907 | -394.304172 | -394.289256 | True | True | 1 | 1.20307019 | 1.45424154 | PASS |
| H2O2 | 1.39624858 | 0.70912431 | 8.46231115 | 2.80298995e+14 | 0.00678010846 | -148.764997 | 0.00339005423 | 0.0101701627 | -148.761607 | -148.754826 | True | True | 1 | 1.3066257 | 1.48587146 | PASS |

## 1. RHF equilibrium r_e

A converged equilibrium is reused only if its molecule, requested RHF basis and nuclear-gradient tolerance match. Otherwise analytic all-electron RHF gradients drive an unconstrained Cartesian BFGS optimization. The built-in geometries are only optimization seeds. Peroxide uses three torsional starts and the lowest-energy converged candidate is retained. --reoptimize explicitly requests a new optimization.

No RHF equilibrium length, angle or torsion is hard-coded into the stretching path. The resulting Cartesian geometry is the sole geometric input for stages 2 and 3.

## 2. Collective force constant k

At q=0, t=dR/dq. The path increases each participating bond by q, with optimized angular geometry fixed. For H2O2 only the O-O separation increases; the optimized OH groups and dihedral remain fixed. Mass-weighted Eckart/Kabsch alignment removes overall translation and rotation.

    k_q = t^T H_RHF t
    mu_eff = sum_A m_A |t_A|^2
    omega = sqrt(k_SI / mu_kg)
    l_q = sqrt(hbar / (mu_kg * omega))
    Delta_q(n) = sqrt(2*n+1) * l_q

The tangent is not Euclidean-normalized. The RHF nuclear Hessian is analytic, with shape and convention checks and independent energy finite differences. Stationarity, fixed internal coordinates, positive curvature, masses and normal modes must pass the existing harmonic validation. A failed stage cannot yield a selected range.

## 3. n=0 or n=1

    hbar_omega_Ha = hbar_J_s * omega_rad_per_s / Hartree_J
    E_n0_Ha = 0.5 * hbar_omega_Ha
    E_n1_Ha = 1.5 * hbar_omega_Ha
    E_total_n0_Ha = E_re_Ha + E_n0_Ha
    E_total_n1_Ha = E_re_Ha + E_n1_Ha

If E_total_n1_Ha<0, choose n=1. Otherwise, if E_total_n0_Ha<0, choose n=0. Otherwise choose NONE. Equality fails the strict test. Use the corresponding existing Part 2 interval.

Energy-reference audit: Part 1 stores float(mf.e_tot) from the converged all-electron RHF calculation in E_eq_Eh and energy_hartree. This is the absolute RHF Born-Oppenheimer molecular energy, including nuclear repulsion. Part 3 copies it into E_re_Ha without a shift. The workflow does not establish that its absolute zero is the intended physical threshold, so every result carries UNVALIDATED_ZERO_REFERENCE. Passing this requested numerical test does not establish a physically bound state.

Part 2 already supplies k, mu_eff and omega. Only hbar*omega is converted from joules to Hartree; the saved wavenumber conversion is checked independently. E_re, both excitation energies and both total energies are in Hartree before comparison.

## Reuse and output files

The program validates Parts 1/2 inputs before reuse. Part 3 always recomputes its simple additions and comparisons; old selection caches cannot enter the decision. --recompute invalidates harmonic results. --reuse-only forbids SCF, optimization and Hessians. Importing the module launches none of them.

bond_length.csv contains the joined result. bond_length.json preserves the equilibrium and harmonic stages with the corrected Part 3 output. New equilibrium XYZ and metadata files are stored in equilibria/. Partial results are saved after each molecule.

## LiH

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## BeH2

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## H2O

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## NH3

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## N2

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## CO

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## HF

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## H2S

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## H2O2

Status: PASS. Basis: sto-3g.

Stage sources: {'equilibrium': 'reused', 'harmonic': 'reused', 'selection': 'postprocessed'}.

E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.



- E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

## Provenance

```json
{
  "source_hashes": {
    "pipeline": "511f352d4b7f9866190571ed32416aa2e72216e1d39029669d5ee46847505549",
    "harmonic": "e6687127ab5752b7a203cab27f177572774e85e7a71acec9f4b617e699b17bd9",
    "part3": "65d7bb3dbcec58125bc9754e3fcb25addd65b9b1fed8c3475b0df31198b30cfd",
    "scan_utilities": "6136f7685523f2fdcd91396a5d3f136709c912bf0be91c70590a614fba5b11a5"
  },
  "constants": {
    "Bohr_A": 0.52917721092,
    "Hartree_J": 4.359744644911914e-18,
    "hbar_J_s": 1.0545718001391127e-34,
    "c_m_per_s": 299792458,
    "amu_kg": 1.660539040427164e-27
  },
  "python_version": "3.13.2",
  "pyscf_version": "2.14.0",
  "metadata": "/Users/novaz/Desktop/qml2.0/results/rhf_geometries/rhf_equilibrium_summary.csv",
  "harmonic_input": "/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json",
  "part3_schema_version": "part3-zero-energy-v2",
  "cache_notes": []
}
```
