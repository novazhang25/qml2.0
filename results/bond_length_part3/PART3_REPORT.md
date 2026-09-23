# Part 3: explicit zero-energy sign test

## Energy-reference audit

**UNVALIDATED_ZERO_REFERENCE.** E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding.

Part 1 stores the BFGS objective in energy_hartree. That objective is converged PySCF RHF mf.e_tot; installed scf.hf.energy_tot adds energy_elec(...)[0] and energy_nuc(). It is not an already referenced well energy. Part 3 reads the CSV value directly without an offset.

The old Part 3 tested the total level against a large-separation plateau. That selector and its threshold inputs have been removed. Old artifacts exist only as explicitly historical evidence under results/part3_history/pre_zero_criterion; they are not inputs to this analysis.

## Calculation and exact criterion

```text
hbar_omega [Ha] = hbar_J_s * saved_omega_rad_per_s / Hartree_J
E_n0 [Ha] = 0.5 * hbar_omega
E_n1 [Ha] = 1.5 * hbar_omega
E_total_n0 [Ha] = saved_E_re + E_n0
E_total_n1 [Ha] = saved_E_re + E_n1
n0_pass = E_total_n0 < 0
n1_pass = E_total_n1 < 0
selected_n = 1 if n1_pass else 0 if n0_pass else None
```

Equality at zero does not pass. The literal zero refers to the unchanged input energy scale. PASS denotes successful numerical/input validation, not resolution of the reference warning.

## Complete table

| Molecule | E_re [Ha] | k [Ha/Bohr^2] | mu_eff [amu] | omega [rad/s] | hbar_omega [Ha] | E_n0 [Ha] | E_n1 [Ha] | E_total_n0 [Ha] | E_total_n1 [Ha] | n0_pass | n1_pass | selected_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | -7.86338212892 | 0.116210397501 | 0.880161046804 | 3.51840856678e+14 | 0.00851062334631 | 0.00425531167315 | 0.0127659350195 | -7.85912681725 | -7.8506161939 | YES | YES | 1 |
| BeH2 | -15.5613528078 | 0.473247780631 | 2.016 | 4.69141527176e+14 | 0.0113479908832 | 0.00567399544158 | 0.0170219863247 | -15.5556788123 | -15.5443308215 | YES | YES | 1 |
| H2O | -74.9659011923 | 1.24689394745 | 1.92283768262 | 7.7973727198e+14 | 0.0188609426818 | 0.00943047134088 | 0.0282914140226 | -74.956470721 | -74.9376097783 | YES | YES | 1 |
| NH3 | -55.4554197788 | 1.62924775325 | 2.93261189058 | 7.21723951054e+14 | 0.017457667553 | 0.00872883377648 | 0.0261865013294 | -55.446690945 | -55.4292332775 | YES | YES | 1 |
| N2 | -107.500654263 | 1.8886404537 | 7.0035 | 5.02830862315e+14 | 0.0121628969315 | 0.00608144846575 | 0.0182443453972 | -107.494572814 | -107.482409917 | YES | YES | 1 |
| CO | -111.225449514 | 1.57382886196 | 6.86054941092 | 4.63771602568e+14 | 0.0112180986183 | 0.00560904930916 | 0.0168271479275 | -111.219840465 | -111.208622366 | YES | YES | 1 |
| HF | -98.5728473472 | 0.725274201861 | 0.957213059852 | 8.42852647653e+14 | 0.0203876306133 | 0.0101938153067 | 0.03058144592 | -98.5626535319 | -98.5422659012 | YES | YES | 1 |
| H2S | -394.311630059 | 0.794499972345 | 1.95899192757 | 6.1664536031e+14 | 0.014915937988 | 0.007457968994 | 0.022373906982 | -394.30417209 | -394.289256152 | YES | YES | 1 |
| H2O2 | -148.764996621 | 0.709124310059 | 8.46231114857 | 2.80298994797e+14 | 0.00678010846038 | 0.00339005423019 | 0.0101701626906 | -148.761606567 | -148.754826459 | YES | YES | 1 |

Energy columns are Hartree; k is Ha/Bohr^2, mass amu and omega rad/s. CSV/JSON retain full precision.

## Unit checks

Existing k, mu_eff and omega are reused unchanged. Independent checks compare saved omega with sqrt((k*Hartree_J/Bohr_m^2)/(mu_eff*amu_kg)), and compare the quantum with 2*pi*hbar*c*100*wavenumber/Hartree_J. Checked values never replace the saved inputs.

| Molecule | omega relative error | quantum conversion residual [Ha] |
| --- | --- | --- |
| LiH | 0 | 1.73472347598e-18 |
| BeH2 | 0 | 0 |
| H2O | 0 | 0 |
| NH3 | 0 | 0 |
| N2 | 0 | 1.73472347598e-18 |
| CO | 0 | 1.73472347598e-18 |
| HF | 0 | 3.46944695195e-18 |
| H2S | 0 | 3.46944695195e-18 |
| H2O2 | 0 | 0 |

## Results

Neither level passes: none of the listed molecules.

- LiH: n0_pass=YES, n1_pass=YES, selected_n=1.
- BeH2: n0_pass=YES, n1_pass=YES, selected_n=1.
- H2O: n0_pass=YES, n1_pass=YES, selected_n=1.
- NH3: n0_pass=YES, n1_pass=YES, selected_n=1.
- N2: n0_pass=YES, n1_pass=YES, selected_n=1.
- CO: n0_pass=YES, n1_pass=YES, selected_n=1.
- HF: n0_pass=YES, n1_pass=YES, selected_n=1.
- H2S: n0_pass=YES, n1_pass=YES, selected_n=1.
- H2O2: n0_pass=YES, n1_pass=YES, selected_n=1.

## Plots

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
    "/Users/novaz/Desktop/qml2.0/json/new_rhf_harmonic_bond_ranges.json": "d698dde31c8252930dcd39aeea1721c5f4bde42efc81d3571a566a88cbfb5999"
  },
  "script_sha256": "65d7bb3dbcec58125bc9754e3fcb25addd65b9b1fed8c3475b0df31198b30cfd",
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
  "frequency_recomputed": false,
  "SCF_recomputed": false,
  "reference_shift_Ha": 0.0,
  "plot_script_sha256": "44834a293a69f2cd2d5a94b65fb5adfc567ea7e171a19bea35a334eb91a7c5fc"
}
```
