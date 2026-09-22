# New RHF harmonic bond/stretch ranges

These are newly derived equilibrium-centered RHF harmonic ranges, independent of any previous QML geometry grids. Only the supplied converged RHF optimization metadata, its XYZ geometries, and its basis labels are used. No geometry is reoptimized, and no previous scan or cutoff is read or compared.

Generated: 2026-09-22T16:40:27.958127+00:00. PySCF 2.14.0; neutral, singlet, all-electron RHF; no density fitting.

## Main result

The default is the **n=1 classical turning-point range**, s_eq ± sqrt(3) l_q, in Angstrom. For collective stretches s is the common participating bond length; for H2O2 it is the O-O separation. Only PASS rows are validated results. Values attached to failed rows are diagnostic.

| Molecule | RHF basis | s_eq [A] | Delta_q(n=1) [A] | n1_min [A] | n1_max [A] | Status |
| --- | --- | --- | --- | --- | --- | --- |
| LiH | sto-3g | 1.51081185 | 0.248038989 | 1.26277286 | 1.75885084 | PASS |
| BeH2 | sto-3g | 1.29053842 | 0.1419309 | 1.14860752 | 1.43246932 | PASS |
| H2O | sto-3g | 0.989409176 | 0.112727195 | 0.876681982 | 1.10213637 | PASS |
| NH3 | sto-3g | 1.03252271 | 0.0948770788 | 0.937645635 | 1.12739979 | PASS |
| N2 | sto-3g | 1.13385101 | 0.0735538872 | 1.06029712 | 1.20740489 | PASS |
| CO | sto-3g | 1.14548068 | 0.0773824783 | 1.0680982 | 1.22286315 | PASS |
| HF | sto-3g | 0.955462699 | 0.153671745 | 0.801790953 | 1.10913444 | PASS |
| H2S | sto-3g | 1.32865587 | 0.125585674 | 1.20307019 | 1.45424154 | PASS |
| H2O2 | sto-3g | 1.39624858 | 0.0896228796 | 1.3066257 | 1.48587146 | PASS |

## Definitions and units

Scientific chain: optimized R_e -> newly defined q -> analytic RHF Hessian H -> t=dR/dq -> k_q=t^T H t -> mu_eff=sum_A m_A |t_A|^2 -> omega -> harmonic extents -> new ranges.

Both Cartesian R and q are measured in Angstrom when constructing t. Consequently t is dimensionless and has the same numerical components for dR_Bohr/dq_Bohr. It is never Euclidean-normalized for curvature or mass. H[a,b,x,y] has units Eh/Bohr^2. Flattening uses H.transpose(0,2,1,3).reshape(3N,3N). Thus k_A=k_Bohr/a0_A^2 and k_SI=k_Bohr*Eh_J/a0_m^2. An energy finite difference uses (h_A/a0_A)^2 in the denominator.

omega=sqrt(k_SI/mu_kg); wavenumber=omega/(2*pi*c_m_s*100); l_q=sqrt(hbar/(mu_kg*omega)). The turning points are ±l_q (n=0) and ±sqrt(3)l_q (n=1). These are not RMS widths or probability cutoffs. The 1D oscillator has infinite tails; the ranges use the requested classical turning-point convention. The harmonic model is local; agreement of small-step curvatures does not establish anharmonic accuracy across the full interval.

All constants are taken consistently from the installed pyscf.data.nist, including its Bohr-to-Angstrom conversion. Masses are explicitly mol.atom_mass_list(isotope_avg=True), also passed to the normal-mode analysis.

For star molecules the optimized bond directions define fixed angular internal coordinates and each radial length is increased by q. For peroxide, changing the optimized O-O radial length translates each rigid OH fragment consistently. A mass-weighted proper Kabsch rotation and center-of-mass alignment then impose the equilibrium Eckart frame. No historical length, angle, torsion, builder, or nominal equilibrium value enters this construction.

Normal-mode overlaps use u=sqrt(M)t/sqrt(mu_eff) and v_j=sqrt(M)L_j, with L_j^T M L_k=delta_jk. The diagnostic squared weights sum to one. Mode phases are arbitrary; absolute overlaps and signed overlaps are both saved. More than one weight above 1e-6 is labeled mixed; a dominant weight below 0.99 indicates substantial mixing. Small admixtures are reported explicitly. The collective path is never replaced by a normal mode.

## Validation policy

SCF thresholds: energy 1e-13 Eh, orbital gradient norm 1e-10, integral screening 1e-14, at most 400 cycles. The relaxed extra SCF convergence cycle is disabled. A second-order RHF solver is used only if needed. CPHF tolerance is 1e-12. Stationarity requires max Cartesian gradient <=2e-6 and RMS <=1e-6 Eh/Bohr; these limits accommodate finite optimizer convergence while the independent curvature tests remain required.

The actual installed Hessian convention is checked by shape, symmetry, translational sum rules, and an independent analytic-gradient finite difference along a generic Cartesian probe (step 1e-4 Bohr). Energy finite differences are evaluated at all four prescribed steps. Both 0.002 and 0.001 A must agree with the projected Hessian and with each other within 2e-6 Eh/Bohr^2 + 2e-4 |k_q|. This allows absolute SCF cancellation noise and O(h^2) truncation without choosing a favorable step after seeing results. The table's relative difference always uses h=0.001 A; Richardson extrapolation is an additional diagnostic.

At a nonexact stationary point, the path energy curvature includes g dot R'' in addition to t^T H t. That correction is measured and reported; a large nuclear gradient is flagged and never silently accepted. Additional checks include internal-coordinate preservation, converged tangent, translation/rotation removal, positive mass/curvature, diatomic reduced masses, normal-mode normalization, spectral curvature reconstruction, absence of imaginary vibrational modes, and finite positive bond intervals.

All numerical tolerances, constants, source hashes, input hashes, Cartesian Hessians, tangents, energies and modes are preserved in the JSON output. Missing geometry, basis, or converged RHF provenance produces NEEDS_INPUT.

## Complete summary

| molecule | RHF_basis | coordinate_definition | s_eq_A | k_q_Eh_per_Bohr2 | k_q_Eh_per_A2 | k_q_N_per_m | mu_eff_amu | mu_eff_kg | omega_rad_per_s | harmonic_frequency_cm1 | Delta_q_n0_A | Delta_q_n1_A | n0_min_A | n0_max_A | n1_min_A | n1_max_A | Hessian_FD_relative_difference | dominant_normal_mode | dominant_mode_frequency_cm1 | mode_overlap | validation_status | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | sto-3g | q = r(Li-H) - r_e(Li-H) | 1.51081185 | 0.116210398 | 0.414994863 | 180.927163 | 0.880161047 | 1.46154178e-27 | 3.51840857e+14 | 1867.86592 | 0.143205377 | 0.248038989 | 1.36760648 | 1.65401723 | 1.26277286 | 1.75885084 | 7.24468248e-07 | 1 | 1867.86592 | 1 | PASS | — |
| BeH2 | sto-3g | r(Be-H_i) = r_e(Be-H_i) + q for every H | 1.29053842 | 0.473247781 | 1.6899985 | 736.796192 | 2.016 | 3.34764671e-27 | 4.69141527e+14 | 2490.59611 | 0.0819438433 | 0.1419309 | 1.20859458 | 1.37248226 | 1.14860752 | 1.43246932 | 1.18592235e-06 | 3 | 2490.59611 | 1 | PASS | — |
| H2O | sto-3g | r(O-H_i) = r_e(O-H_i) + q for every H | 0.989409176 | 1.24689395 | 4.45273912 | 1941.28055 | 1.92283768 | 3.19294704e-27 | 7.79737272e+14 | 4139.49844 | 0.0650830762 | 0.112727195 | 0.9243261 | 1.05449225 | 0.876681982 | 1.10213637 | 2.50264821e-06 | 2 | 4139.63828 | 0.99995342 | PASS | — |
| NH3 | sto-3g | r(N-H_i) = r_e(N-H_i) + q for every H | 1.03252271 | 1.62924775 | 5.81814934 | 2536.56454 | 2.93261189 | 4.86971653e-27 | 7.21723951e+14 | 3831.51515 | 0.054777307 | 0.0948770788 | 0.977745407 | 1.08730002 | 0.937645635 | 1.12739979 | 2.40393231e-06 | 4 | 3832.93311 | 0.999572002 | PASS | — |
| N2 | sto-3g | q = r(N-N) - r_e(N-N) | 1.13385101 | 1.88864045 | 6.74445749 | 2940.41124 | 7.0035 | 1.16295852e-26 | 5.02830862e+14 | 2669.44732 | 0.0424663566 | 0.0735538872 | 1.09138465 | 1.17631736 | 1.06029712 | 1.20740489 | 2.89942407e-06 | 1 | 2669.44732 | 1 | PASS | — |
| CO | sto-3g | q = r(C-O) - r_e(C-O) | 1.14548068 | 1.57382886 | 5.62024489 | 2450.28326 | 6.86054941 | 1.13922101e-26 | 4.63771603e+14 | 2462.08806 | 0.0446767947 | 0.0773824783 | 1.10080388 | 1.19015747 | 1.0680982 | 1.22286315 | 2.79415409e-06 | 1 | 2462.08806 | 1 | PASS | — |
| HF | sto-3g | q = r(H-F) - r_e(H-F) | 0.955462699 | 0.725274202 | 2.59000119 | 1129.17438 | 0.95721306 | 1.58948966e-27 | 8.42852648e+14 | 4474.56771 | 0.0887224235 | 0.153671745 | 0.866740275 | 1.04418512 | 0.801790953 | 1.10913444 | 2.51704546e-06 | 1 | 4474.56771 | 1 | PASS | — |
| H2S | sto-3g | r(S-H_i) = r_e(S-H_i) + q for every H | 1.32865587 | 0.794499972 | 2.83721091 | 1236.95151 | 1.95899193 | 3.25298258e-27 | 6.1664536e+14 | 3273.66999 | 0.0725069229 | 0.125585674 | 1.25614895 | 1.40116279 | 1.20307019 | 1.45424154 | 1.28033649e-06 | 2 | 3274.25522 | 0.999764302 | PASS | — |
| H2O2 | sto-3g | q = r(O-O) - r_e(O-O); both OH lengths, OOH angles and HOOH torsion fixed | 1.39624858 | 0.70912431 | 2.53232888 | 1104.03073 | 8.46231115 | 1.4051998e-26 | 2.80298995e+14 | 1488.0618 | 0.0517437937 | 0.0896228796 | 1.34450479 | 1.44799237 | 1.3066257 | 1.48587146 | 1.58707798e-06 | 2 | 1486.76526 | 0.997888748 | PASS | — |

## LiH — PASS

Basis: sto-3g. Coordinate: q = r(Li-H) - r_e(Li-H).

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/LiH_converged.xyz`. SHA-256: `5bd116d6bfd71636103d062b3ac9e6551e8958c3d40ab0702c3c1a5dc38df336`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| Li1 | -0 | 0 | -0.755405927 | 6.94 |
| H2 | 0 | -0 | 0.755405927 | 1.008 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| Li1-H2 | stretch | 1.51081185 | A | 1 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| Li1 | 0 | 0 | -0.126824358 |
| H2 | 0 | 0 | 0.873175642 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 4.4408921e-13 | 5.10970788e-14 | 0 |
| 0.0005 | 6.66133815e-13 | 2.55345708e-14 | 0 |
| 0.0002 | 0 | 3.01302348e-13 | 0 |

RHF energy: -7.86338212892111 Eh. Maximum/RMS nuclear gradient: 1.42475e-07 / 8.22579e-08 Eh/Bohr. Gradient projection: 2.69239e-07 Eh/A.

Hessian shape: (2, 2, 3, 3); symmetry error 1.06e-17; translation error 5.4e-14 Eh/Bohr^2. Gradient-probe relative error: 3.68e-06.

Analytic k_q: 0.1162103975 Eh/Bohr^2 = 0.4149948629 Eh/A^2 = 180.9271631 N/m. Generalized mass: 0.8801610468 amu = 1.46154178e-27 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -7.86336118050977 | -7.86338212892111 | -7.86336157495773 | 0.116218486 | 6.96015883e-05 |
| 0.005 | -7.86337691775270 | -7.86338212892111 | -7.86337696503742 | 0.11621242 | 1.74003422e-05 |
| 0.002 | -7.86338129786834 | -7.86338212892111 | -7.86338129998981 | 0.116210721 | 2.78335552e-06 |
| 0.001 | -7.86338192149288 | -7.86338212892111 | -7.86338192135418 | 0.116210482 | 7.24468248e-07 |

Richardson curvature: 0.1162104019 Eh/Bohr^2; relative difference 3.82e-08. Residual-gradient path correction: -2.09e-16 Eh/Bohr^2.

Ordinary diatomic reduced mass: 0.8801610468 amu (relative discrepancy 7.75e-13); ordinary d2E/dr2: 0.1162103975 Eh/Bohr^2.

omega = 3.518408567e+14 rad/s; collective frequency = 1867.865919 cm^-1.

n=0: q in [-0.143205377, 0.143205377] A; s in [1.367606476, 1.654017230] A.

n=1: q in [-0.248038989, 0.248038989] A.

**NEW n=1 harmonic range: [1.262772864, 1.758850842] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.19846888 | 0.19846888 | 1.31234297 | 1.70928073 |
| 0 | 0.99 | -0.260832321 | 0.260832321 | 1.24997953 | 1.77164417 |
| 0 | 0.999 | -0.33320365 | 0.33320365 | 1.1776082 | 1.8440155 |
| 1 | 0.95 | -0.283074832 | 0.283074832 | 1.22773702 | 1.79388668 |
| 1 | 0.99 | -0.341070397 | 0.341070397 | 1.16974146 | 1.85188225 |
| 1 | 0.999 | -0.408402004 | 0.408402004 | 1.10240985 | 1.91921386 |

Dominant vibrational mode: 1 at 1867.865919 cm^-1. Mixed coordinate (multiple weights >1e-6): False. Total secondary-mode weight: 0.000000%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 2.1e-16.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 1867.86592 | 1 | 1 |

## BeH2 — PASS

Basis: sto-3g. Coordinate: r(Be-H_i) = r_e(Be-H_i) + q for every H.

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/BeH2_converged.xyz`. SHA-256: `4d581fba6403a5909ca7d13803addc698313d796424e820ad7c146c36d3f7e87`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| Be1 | -0 | 0 | 0 | 9.0121831 |
| H2 | -0 | -0 | -1.29053842 | 1.008 |
| H3 | 0 | -0 | 1.29053842 | 1.008 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| Be1-H2 | stretch | 1.29053842 | A | 1 |
| Be1-H3 | stretch | 1.29053842 | A | 1 |
| H2-Be1-H3 | fixed_angle | 180 | deg | 0 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| Be1 | 0 | 0 | 0 |
| H2 | 0 | 0 | -1 |
| H3 | 0 | 0 | 1 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 0 | 0 | 0 |
| 0.0005 | 0 | 0 | 0 |
| 0.0002 | 0 | 0 | 0 |

RHF energy: -15.56135280778208 Eh. Maximum/RMS nuclear gradient: 6.6242e-09 / 3.12268e-09 Eh/Bohr. Gradient projection: 2.50359e-08 Eh/A.

Hessian shape: (3, 3, 3, 3); symmetry error 3.9e-18; translation error 1.84e-13 Eh/Bohr^2. Gradient-probe relative error: 3.51e-06.

Analytic k_q: 0.4732477806 Eh/Bohr^2 = 1.689998503 Eh/A^2 = 736.7961925 N/m. Generalized mass: 2.016 amu = 3.347646706e-27 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -15.56126726164642 | -15.56135280778208 | -15.56126933463517 | 0.473302196 | 0.000114983731 |
| 0.005 | -15.56133155273282 | -15.56135280778208 | -15.56133181165365 | 0.473261391 | 2.8759882e-05 |
| 0.002 | -15.56134941952625 | -15.56135280778208 | -15.56134943601265 | 0.473249968 | 4.62144157e-06 |
| 0.001 | -15.56135196177023 | -15.56135280778208 | -15.56135196379342 | 0.473248342 | 1.18592235e-06 |

Richardson curvature: 0.4732477999 Eh/Bohr^2; relative difference 4.07e-08. Residual-gradient path correction: 0 Eh/Bohr^2.

omega = 4.691415272e+14 rad/s; collective frequency = 2490.596113 cm^-1.

n=0: q in [-0.081943843, 0.081943843] A; s in [1.208594578, 1.372482264] A.

n=1: q in [-0.141930900, 0.141930900] A.

**NEW n=1 harmonic range: [1.148607521, 1.432469321] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.113566286 | 0.113566286 | 1.17697214 | 1.40410471 |
| 0 | 0.99 | -0.149251399 | 0.149251399 | 1.14128702 | 1.43978982 |
| 0 | 0.999 | -0.190663146 | 0.190663146 | 1.09987528 | 1.48120157 |
| 1 | 0.95 | -0.161978832 | 0.161978832 | 1.12855959 | 1.45251725 |
| 1 | 0.99 | -0.195164593 | 0.195164593 | 1.09537383 | 1.48570301 |
| 1 | 0.999 | -0.23369255 | 0.23369255 | 1.05684587 | 1.52423097 |

Dominant vibrational mode: 3 at 2490.596113 cm^-1. Mixed coordinate (multiple weights >1e-6): False. Total secondary-mode weight: 0.000000%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 1.89e-15.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 827.043326 | 8.38095071e-17 | 7.02403347e-33 |
| 2 | 827.043326 | 3.2784978e-16 | 1.07485478e-31 |
| 3 | 2490.59611 | 1 | 1 |
| 4 | 2775.971 | 5.94339011e-15 | 3.5323886e-29 |

## H2O — PASS

Basis: sto-3g. Coordinate: r(O-H_i) = r_e(O-H_i) + q for every H.

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/H2O_converged.xyz`. SHA-256: `377f1d69faf91f946711398747912b26b5b2373c707c995cfb781b913adfa283`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| O1 | 0 | 0 | -0.423868772 | 15.999 |
| H2 | 0.75807972 | -0 | 0.211934386 | 1.008 |
| H3 | -0.75807972 | -0 | 0.211934386 | 1.008 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| O1-H2 | stretch | 0.989409176 | A | 1 |
| O1-H3 | stretch | 0.989409176 | A | 1 |
| H2-O1-H3 | fixed_angle | 100.026728 | deg | -3.55271368e-11 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| O1 | -2.7103567e-15 | 0 | -0.0719122717 |
| H2 | 0.766194349 | 0 | 0.570696644 |
| H3 | -0.766194349 | 0 | 0.570696644 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 3.33066907e-13 | 5.72577059e-15 | 3.5862149e-16 |
| 0.0005 | 1.66533454e-13 | 1.81432798e-14 | 6.94575751e-17 |
| 0.0002 | 0 | 1.78477762e-13 | 3.08530591e-15 |

RHF energy: -74.96590119229924 Eh. Maximum/RMS nuclear gradient: 4.81833e-07 / 2.6167e-07 Eh/Bohr. Gradient projection: -1.00883e-06 Eh/A.

Hessian shape: (3, 3, 3, 3); symmetry error 4.44e-16; translation error 7.56e-12 Eh/Bohr^2. Gradient-probe relative error: 1.17e-06.

Analytic k_q: 1.246893947 Eh/Bohr^2 = 4.452739117 Eh/A^2 = 1941.280552 N/m. Generalized mass: 1.922837683 amu = 3.19294704e-27 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -74.96567433528531 | -74.96590119229924 | -74.96568266413180 | 1.24720553 | 0.000249890305 |
| 0.005 | -74.96584500531870 | -74.96590119229924 | -74.96584605384852 | 1.24697183 | 6.24633942e-05 |
| 0.002 | -74.96589225148550 | -74.96590119229924 | -74.96589232197846 | 1.24690641 | 9.99705529e-06 |
| 0.001 | -74.96589896076175 | -74.96590119229924 | -74.96589897108647 | 1.24689707 | 2.50264821e-06 |

Richardson curvature: 1.246893953 Eh/Bohr^2; relative difference 4.51e-09. Residual-gradient path correction: 4.81e-16 Eh/Bohr^2.

omega = 7.79737272e+14 rad/s; collective frequency = 4139.498438 cm^-1.

n=0: q in [-0.065083076, 0.065083076] A; s in [0.924326100, 1.054492253] A.

n=1: q in [-0.112727195, 0.112727195] A.

**NEW n=1 harmonic range: [0.876681982, 1.102136371] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.0901988842 | 0.0901988842 | 0.899210292 | 1.07960806 |
| 0 | 0.99 | -0.118541428 | 0.118541428 | 0.870867749 | 1.1079506 |
| 0 | 0.999 | -0.151432293 | 0.151432293 | 0.837976884 | 1.14084147 |
| 1 | 0.95 | -0.128650064 | 0.128650064 | 0.860759113 | 1.11805924 |
| 1 | 0.99 | -0.155007522 | 0.155007522 | 0.834401654 | 1.1444167 |
| 1 | 0.999 | -0.185607966 | 0.185607966 | 0.80380121 | 1.17501714 |

Dominant vibrational mode: 2 at 4139.638283 cm^-1. Mixed coordinate (multiple weights >1e-6): True. Total secondary-mode weight: 0.009316%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 6.85e-16.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 2169.85299 | 0.00965184204 | 9.31580547e-05 |
| 2 | 4139.63828 | 0.99995342 | 0.999906842 |
| 3 | 4390.67449 | 6.45881512e-15 | 4.17162928e-29 |

## NH3 — PASS

Basis: sto-3g. Coordinate: r(N-H_i) = r_e(N-H_i) + q for every H.

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/NH3_converged.xyz`. SHA-256: `0ffc22eea129ddd7700ded6a49e804b37f80dee7200b39b2abc2b216958e36ef`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| N1 | 0 | 0 | -0.319480247 | 14.007 |
| H2 | 0.940558128 | 0 | 0.106493416 | 1.008 |
| H3 | -0.470279064 | 0.814547233 | 0.106493416 | 1.008 |
| H4 | -0.470279064 | -0.814547233 | 0.106493416 | 1.008 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| N1-H2 | stretch | 1.03252271 | A | 1 |
| N1-H3 | stretch | 1.03252271 | A | 1 |
| N1-H4 | stretch | 1.03252271 | A | 1 |
| H2-N1-H3 | fixed_angle | 104.163872 | deg | 3.55271368e-11 |
| H2-N1-H4 | fixed_angle | 104.163872 | deg | 0 |
| H3-N1-H4 | fixed_angle | 104.163872 | deg | -3.55271368e-11 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| N1 | 7.92486026e-15 | -1.57478041e-14 | -0.0732528915 |
| H2 | 0.910932143 | 7.25150367e-14 | 0.339303324 |
| H3 | -0.455466072 | 0.788890377 | 0.339303324 |
| H4 | -0.455466072 | -0.788890377 | 0.339303324 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 8.8817842e-13 | 3.06831512e-14 | 1.24379771e-13 |
| 0.0005 | 1.16573418e-12 | 6.92786238e-14 | 5.48914784e-13 |
| 0.0002 | 0 | 7.04853054e-14 | 1.861067e-13 |

RHF energy: -55.45541977879940 Eh. Maximum/RMS nuclear gradient: 9.88422e-07 / 4.23047e-07 Eh/Bohr. Gradient projection: -1.97016e-06 Eh/A.

Hessian shape: (4, 4, 3, 3); symmetry error 8.19e-16; translation error 2.85e-12 Eh/Bohr^2. Gradient-probe relative error: 4.32e-07.

Analytic k_q: 1.629247753 Eh/Bohr^2 = 5.818149344 Eh/A^2 = 2536.564545 N/m. Generalized mass: 2.932611891 amu = 4.869716535e-27 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -55.45512353116162 | -55.45541977879940 | -55.45513407193047 | 1.6296386 | 0.000239891213 |
| 0.005 | -55.45534638147165 | -55.45541977879940 | -55.45534771367123 | 1.62934545 | 5.99662843e-05 |
| 0.002 | -55.45540809645036 | -55.45541977879940 | -55.45540818832775 | 1.62926339 | 9.59558087e-06 |
| 0.001 | -55.45541686249783 | -55.45541977879940 | -55.45541687693765 | 1.62925167 | 2.40393231e-06 |

Richardson curvature: 1.629247764 Eh/Bohr^2; relative difference 6.72e-09. Residual-gradient path correction: 2.72e-15 Eh/Bohr^2.

omega = 7.217239511e+14 rad/s; collective frequency = 3831.515146 cm^-1.

n=0: q in [-0.054777307, 0.054777307] A; s in [0.977745407, 1.087300021] A.

n=1: q in [-0.094877079, 0.094877079] A.

**NEW n=1 harmonic range: [0.937645635, 1.127399792] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.0759160792 | 0.0759160792 | 0.956606634 | 1.10843879 |
| 0 | 0.99 | -0.0997706402 | 0.0997706402 | 0.932752073 | 1.13229335 |
| 0 | 0.999 | -0.127453305 | 0.127453305 | 0.905069408 | 1.15997602 |
| 1 | 0.95 | -0.108278594 | 0.108278594 | 0.924244119 | 1.14080131 |
| 1 | 0.99 | -0.130462404 | 0.130462404 | 0.902060309 | 1.16298512 |
| 1 | 0.999 | -0.156217332 | 0.156217332 | 0.876305382 | 1.18874005 |

Dominant vibrational mode: 4 at 3832.933110 cm^-1. Mixed coordinate (multiple weights >1e-6): True. Total secondary-mode weight: 0.085581%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 7.99e-16.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 1411.5365 | 0.0292542663 | 0.000855812098 |
| 2 | 2076.12485 | 1.51581039e-14 | 2.29768114e-28 |
| 3 | 2076.12485 | 3.5503432e-13 | 1.26049368e-25 |
| 4 | 3832.93311 | 0.999572002 | 0.999144188 |
| 5 | 4107.84962 | 4.09375494e-12 | 1.67588295e-23 |
| 6 | 4107.84962 | 1.81045343e-14 | 3.27774162e-28 |

## N2 — PASS

Basis: sto-3g. Coordinate: q = r(N-N) - r_e(N-N).

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/N2_converged.xyz`. SHA-256: `48fa582eb091c119802080fb129137868852c1f18a0478cf6d2366248e2722ee`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| N1 | 0 | 0 | -0.566925504 | 14.007 |
| N2 | -0 | -0 | 0.566925504 | 14.007 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| N1-N2 | stretch | 1.13385101 | A | 1 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| N1 | 0 | 0 | -0.5 |
| N2 | 0 | 0 | 0.5 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 0 | 0 | 0 |
| 0.0005 | 0 | 0 | 0 |
| 0.0002 | 0 | 0 | 0 |

RHF energy: -107.50065426275155 Eh. Maximum/RMS nuclear gradient: 5.21325e-07 / 3.00987e-07 Eh/Bohr. Gradient projection: -9.85162e-07 Eh/A.

Hessian shape: (2, 2, 3, 3); symmetry error 1.31e-16; translation error 1.27e-12 Eh/Bohr^2. Gradient-probe relative error: 7.79e-07.

Analytic k_q: 1.888640454 Eh/Bohr^2 = 6.744457493 Eh/A^2 = 2940.411244 N/m. Generalized mass: 7.0035 amu = 1.162958517e-26 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -107.50030981083385 | -107.50065426275155 | -107.50032407298806 | 1.88918912 | 0.000290508013 |
| 0.005 | -107.50056905592530 | -107.50065426275155 | -107.50057084589511 | 1.88877762 | 7.26248127e-05 |
| 0.002 | -107.50064071474739 | -107.50065426275155 | -107.50064083261221 | 1.8886624 | 1.16220509e-05 |
| 0.001 | -107.50065088240760 | -107.50065426275155 | -107.50065089861846 | 1.88864593 | 2.89942407e-06 |

Richardson curvature: 1.888640438 Eh/Bohr^2; relative difference 8.12e-09. Residual-gradient path correction: 0 Eh/Bohr^2.

Ordinary diatomic reduced mass: 7.0035 amu (relative discrepancy 2.2e-13); ordinary d2E/dr2: 1.888640454 Eh/Bohr^2.

omega = 5.028308623e+14 rad/s; collective frequency = 2669.447317 cm^-1.

n=0: q in [-0.042466357, 0.042466357] A; s in [1.091384651, 1.176317364] A.

n=1: q in [-0.073553887, 0.073553887] A.

**NEW n=1 harmonic range: [1.060297120, 1.207404895] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.058854286 | 0.058854286 | 1.07499672 | 1.19270529 |
| 0 | 0.99 | -0.077347643 | 0.077347643 | 1.05650336 | 1.21119865 |
| 0 | 0.999 | -0.0988087551 | 0.0988087551 | 1.03504225 | 1.23265976 |
| 1 | 0.95 | -0.0839434733 | 0.0839434733 | 1.04990753 | 1.21779448 |
| 1 | 0.99 | -0.101141573 | 0.101141573 | 1.03270943 | 1.23499258 |
| 1 | 0.999 | -0.121108198 | 0.121108198 | 1.01274281 | 1.25495921 |

Dominant vibrational mode: 1 at 2669.447317 cm^-1. Mixed coordinate (multiple weights >1e-6): False. Total secondary-mode weight: 0.000000%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 8.23e-16.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 2669.44732 | 1 | 1 |

## CO — PASS

Basis: sto-3g. Coordinate: q = r(C-O) - r_e(C-O).

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/CO_converged.xyz`. SHA-256: `d9829f73018ab1b643e2d9c705e40b129dfbd375cdb0f7d42cc9f2264694f64d`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| C1 | 0 | -0 | -0.572740338 | 12.011 |
| O2 | -0 | 0 | 0.572740338 | 15.999 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| C1-O2 | stretch | 1.14548068 | A | 1 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| C1 | 0 | 0 | -0.571188861 |
| O2 | 0 | 0 | 0.428811139 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 0 | 1.27471519e-14 | 0 |
| 0.0005 | 1.11022302e-13 | 9.82672232e-14 | 0 |
| 0.0002 | 0 | 1.27471519e-14 | 0 |

RHF energy: -111.22544951413570 Eh. Maximum/RMS nuclear gradient: 3.81688e-10 / 2.20365e-10 Eh/Bohr. Gradient projection: -7.21279e-10 Eh/A.

Hessian shape: (2, 2, 3, 3); symmetry error 4.97e-17; translation error 1.66e-12 Eh/Bohr^2. Gradient-probe relative error: 6.72e-07.

Analytic k_q: 1.573828862 Eh/Bohr^2 = 5.620244891 Eh/A^2 = 2450.283257 N/m. Generalized mass: 6.860549411 amu = 1.139221014e-26 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -111.22516255094106 | -111.22544951413570 | -111.22517429257712 | 1.57427765 | 0.000285155074 |
| 0.005 | -111.22537852229624 | -111.22544951413570 | -111.22537998983773 | 1.57394104 | 7.12791957e-05 |
| 0.002 | -111.22543822655686 | -111.22544951413570 | -111.22543832047890 | 1.57384679 | 1.13910937e-05 |
| 0.001 | -111.22544669813482 | -111.22544951413570 | -111.22544670987600 | 1.57383326 | 2.79415409e-06 |

Richardson curvature: 1.573828749 Eh/Bohr^2; relative difference 7.15e-08. Residual-gradient path correction: 0 Eh/Bohr^2.

Ordinary diatomic reduced mass: 6.860549411 amu (relative discrepancy 2.2e-13); ordinary d2E/dr2: 1.573828862 Eh/Bohr^2.

omega = 4.637716026e+14 rad/s; collective frequency = 2462.088056 cm^-1.

n=0: q in [-0.044676795, 0.044676795] A; s in [1.100803881, 1.190157471] A.

n=1: q in [-0.077382478, 0.077382478] A.

**NEW n=1 harmonic range: [1.068098198, 1.222863154] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.0619177406 | 0.0619177406 | 1.08356294 | 1.20739842 |
| 0 | 0.99 | -0.0813737047 | 0.0813737047 | 1.06410697 | 1.22685438 |
| 0 | 0.999 | -0.1039519 | 0.1039519 | 1.04152878 | 1.24943258 |
| 1 | 0.95 | -0.088312858 | 0.088312858 | 1.05716782 | 1.23379353 |
| 1 | 0.99 | -0.106406145 | 0.106406145 | 1.03907453 | 1.25188682 |
| 1 | 0.999 | -0.127412063 | 0.127412063 | 1.01806861 | 1.27289274 |

Dominant vibrational mode: 1 at 2462.088056 cm^-1. Mixed coordinate (multiple weights >1e-6): False. Total secondary-mode weight: 0.000000%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 2.42e-16.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 2462.08806 | 1 | 1 |

## HF — PASS

Basis: sto-3g. Coordinate: q = r(H-F) - r_e(H-F).

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/HF_converged.xyz`. SHA-256: `785ad7b77f9ff32ec72a91d75bfa72dc403fe715234acc178a531cceae1b2aa6`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| H1 | -0 | 0 | -0.477731349 | 1.008 |
| F2 | 0 | -0 | 0.477731349 | 18.9984032 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| H1-F2 | stretch | 0.955462699 | A | 1 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| H1 | 0 | 0 | -0.949616131 |
| F2 | 0 | 0 | 0.0503838692 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 1.66533454e-13 | 4.88341784e-16 | 0 |
| 0.0005 | 1.94289029e-13 | 1.2171919e-13 | 0 |
| 0.0002 | 0 | 7.11702657e-14 | 0 |

RHF energy: -98.57284734715999 Eh. Maximum/RMS nuclear gradient: 5.45312e-10 / 3.14835e-10 Eh/Bohr. Gradient projection: 1.03049e-09 Eh/A.

Hessian shape: (2, 2, 3, 3); symmetry error 7.48e-18; translation error 5.63e-12 Eh/Bohr^2. Gradient-probe relative error: 1.35e-06.

Analytic k_q: 0.7252742019 Eh/Bohr^2 = 2.590001191 Eh/A^2 = 1129.174382 N/m. Generalized mass: 0.9572130599 amu = 1.589489656e-27 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -98.57271528370198 | -98.57284734715999 | -98.57272034423825 | 0.72545975 | 0.000255832382 |
| 0.005 | -98.57281465383294 | -98.57284734715999 | -98.57281528631682 | 0.72532058 | 6.39450005e-05 |
| 0.002 | -98.57284214686757 | -98.57284734715999 | -98.57284218734181 | 0.725281612 | 1.02164172e-05 |
| 0.001 | -98.57284604962730 | -98.57284734715999 | -98.57284605468497 | 0.725276027 | 2.51704546e-06 |

Richardson curvature: 0.725274166 Eh/Bohr^2; relative difference 4.94e-08. Residual-gradient path correction: -4e-19 Eh/Bohr^2.

Ordinary diatomic reduced mass: 0.9572130599 amu (relative discrepancy 4.98e-13); ordinary d2E/dr2: 0.7252742019 Eh/Bohr^2.

omega = 8.428526477e+14 rad/s; collective frequency = 4474.567708 cm^-1.

n=0: q in [-0.088722423, 0.088722423] A; s in [0.866740275, 1.044185122] A.

n=1: q in [-0.153671745, 0.153671745] A.

**NEW n=1 harmonic range: [0.801790953, 1.109134444] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.122960746 | 0.122960746 | 0.832501953 | 1.07842344 |
| 0 | 0.99 | -0.161597813 | 0.161597813 | 0.793864886 | 1.11706051 |
| 0 | 0.999 | -0.206435233 | 0.206435233 | 0.749027466 | 1.16189793 |
| 1 | 0.95 | -0.175378087 | 0.175378087 | 0.780084611 | 1.13084079 |
| 1 | 0.99 | -0.21130905 | 0.21130905 | 0.744153649 | 1.16677175 |
| 1 | 0.999 | -0.253024127 | 0.253024127 | 0.702438571 | 1.20848683 |

Dominant vibrational mode: 1 at 4474.567708 cm^-1. Mixed coordinate (multiple weights >1e-6): False. Total secondary-mode weight: 0.000000%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 1.32e-15.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 4474.56771 | 1 | 1 |

## H2S — PASS

Basis: sto-3g. Coordinate: r(S-H_i) = r_e(S-H_i) + q for every H.

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/H2S_converged.xyz`. SHA-256: `b63ed5c0fe71ba1316abd8a8398f9ae07b2ed13cf4958e86fc8c7fe2cdb3e99c`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| S1 | 0 | 0 | -0.612383114 | 32.06 |
| H2 | 0.959972392 | -0 | 0.306191557 | 1.008 |
| H3 | -0.959972392 | -0 | 0.306191557 | 1.008 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| S1-H2 | stretch | 1.32865587 | A | 1 |
| S1-H3 | stretch | 1.32865587 | A | 1 |
| H2-S1-H3 | fixed_angle | 92.5248565 | deg | 0 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| S1 | 1.30300354e-15 | -6.03346244e-14 | -0.0409019368 |
| H2 | 0.722513944 | 9.59488124e-13 | 0.650454412 |
| H3 | -0.722513944 | 9.59488124e-13 | 0.650454412 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 1.05536472e-12 | 6.27180983e-14 | 1.77548839e-13 |
| 0.0005 | 7.67518285e-13 | 9.21426754e-15 | 3.55498719e-13 |
| 0.0002 | 0 | 4.63135082e-14 | 1.77682621e-12 |

RHF energy: -394.31163005927652 Eh. Maximum/RMS nuclear gradient: 1.96893e-07 / 8.25472e-08 Eh/Bohr. Gradient projection: 1.48422e-07 Eh/A.

Hessian shape: (3, 3, 3, 3); symmetry error 9.7e-10; translation error 1.23e-09 Eh/Bohr^2. Gradient-probe relative error: 1.31e-06.

Analytic k_q: 0.7944999723 Eh/Bohr^2 = 2.837210905 Eh/A^2 = 1236.951505 N/m. Generalized mass: 1.958991928 amu = 3.252982576e-27 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -394.31148607581144 | -394.31163005927652 | -394.31149027777849 | 0.794622828 | 0.000154632761 |
| 0.005 | -394.31159433072111 | -394.31163005927652 | -394.31159485481845 | 0.794530673 | 3.86415377e-05 |
| 0.002 | -394.31162436829811 | -394.31163005927652 | -394.31162440134102 | 0.794504893 | 6.19391145e-06 |
| 0.001 | -394.31162863871532 | -394.31163005927652 | -394.31162864262319 | 0.79450099 | 1.28033649e-06 |

Richardson curvature: 0.7944996883 Eh/Bohr^2; relative difference 3.58e-07. Residual-gradient path correction: -6.95e-16 Eh/Bohr^2.

omega = 6.166453603e+14 rad/s; collective frequency = 3273.669988 cm^-1.

n=0: q in [-0.072506923, 0.072506923] A; s in [1.256148946, 1.401162792] A.

n=1: q in [-0.125585674, 0.125585674] A.

**NEW n=1 harmonic range: [1.203070195, 1.454241543] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.100487622 | 0.100487622 | 1.22816825 | 1.42914349 |
| 0 | 0.99 | -0.132063121 | 0.132063121 | 1.19659275 | 1.46071899 |
| 0 | 0.999 | -0.168705756 | 0.168705756 | 1.15995011 | 1.49736162 |
| 1 | 0.95 | -0.143324821 | 0.143324821 | 1.18533105 | 1.47198069 |
| 1 | 0.99 | -0.172688802 | 0.172688802 | 1.15596707 | 1.50134467 |
| 1 | 0.999 | -0.206779754 | 0.206779754 | 1.12187612 | 1.53543562 |

Dominant vibrational mode: 2 at 3274.255224 cm^-1. Mixed coordinate (multiple weights >1e-6): True. Total secondary-mode weight: 0.047134%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 1.12e-09.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 1609.51582 | 0.0217103711 | 0.000471340213 |
| 2 | 3274.25522 | 0.999764302 | 0.99952866 |
| 3 | 3322.21402 | 4.72432211e-10 | 2.23192194e-19 |

## H2O2 — PASS

Basis: sto-3g. Coordinate: q = r(O-O) - r_e(O-O); both OH lengths, OOH angles and HOOH torsion fixed.

Input: `/Users/novaz/Desktop/qml2.0/rhf_geometries/H2O2_converged.xyz`. SHA-256: `930f11fba73ff6fd2df4debbc0ec38ad5a898ddb739b67c705e7029a7f3b1195`.

| Atom | x [A] | y [A] | z [A] | Average mass [amu] |
| --- | --- | --- | --- | --- |
| O1 | -0.251242334 | -0.0366516384 | -0.688739789 | 15.999 |
| O2 | -0.120371213 | -0.223554968 | 0.688739789 | 15.999 |
| H3 | 0.595305046 | -0.454720839 | -1.02158953 | 1.008 |
| H4 | -0.223691499 | 0.714927445 | 1.02158953 | 1.008 |

Optimized internal coordinates and finest-step derivatives:

| Internal coordinate | Role | Equilibrium value | Unit | Derivative per A |
| --- | --- | --- | --- | --- |
| O1-O2 | stretch | 1.39624858 | A | 1 |
| O1-H3 | fixed_bond | 1.00110603 | A | -1.11022302e-12 |
| O2-H4 | fixed_bond | 1.00110603 | A | -1.66533454e-12 |
| H3-O1-O2 | fixed_angle | 101.119247 | deg | 7.10542736e-11 |
| O1-O2-H4 | fixed_angle | 101.119247 | deg | 1.42108547e-10 |
| H3-O1-O2-H4 | fixed_dihedral | 124.990186 | deg | -7.10542736e-11 |

Unnormalized tangent t (dimensionless):

| Atom | t_x | t_y | t_z |
| --- | --- | --- | --- |
| O1 | -0.0283382239 | 0.040471178 | -0.498629125 |
| O2 | 0.0283382239 | -0.040471178 | 0.498629125 |
| H3 | -0.019384594 | 0.0276840693 | -0.459796082 |
| H4 | 0.019384594 | -0.0276840693 | 0.459796082 |

Tangent convergence and Eckart residuals:

| delta [A] | max |t-t_finest| | COM derivative norm | Rotation residual [amu A] |
| --- | --- | --- | --- |
| 0.001 | 1.48131507e-08 | 1.60750144e-14 | 1.97409686e-12 |
| 0.0005 | 3.2401859e-09 | 1.58044907e-14 | 1.36550856e-11 |
| 0.0002 | 0 | 1.21684268e-13 | 2.22246505e-11 |

RHF energy: -148.76499662127230 Eh. Maximum/RMS nuclear gradient: 7.52455e-07 / 3.24158e-07 Eh/Bohr. Gradient projection: -1.279e-06 Eh/A.

Hessian shape: (4, 4, 3, 3); symmetry error 4.44e-15; translation error 4.79e-12 Eh/Bohr^2. Gradient-probe relative error: 2.75e-06.

Analytic k_q: 0.7091243101 Eh/Bohr^2 = 2.532328881 Eh/A^2 = 1104.030728 N/m. Generalized mass: 8.462311149 amu = 1.405199803e-26 kg.

| h [A] | E(-h) [Eh] | E(0) [Eh] | E(+h) [Eh] | k_FD [Eh/Bohr^2] | Relative difference |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -148.76486791558213 | -148.76499662127230 | -148.76487205427526 | 0.709235759 | 0.000157164167 |
| 0.005 | -148.76496470247434 | -148.76499662127230 | -148.76496522936111 | 0.709152169 | 3.92860708e-05 |
| 0.002 | -148.76499153757396 | -148.76499662127230 | -148.76499157559147 | 0.709128766 | 6.28413194e-06 |
| 0.001 | -148.76499535177052 | -148.76499662127230 | -148.76499535844118 | 0.709125435 | 1.58707798e-06 |

Richardson curvature: 0.7091243252 Eh/Bohr^2; relative difference 2.14e-08. Residual-gradient path correction: 1.31e-09 Eh/Bohr^2.

omega = 2.802989948e+14 rad/s; collective frequency = 1488.061803 cm^-1.

n=0: q in [-0.051743794, 0.051743794] A; s in [1.344504785, 1.447992373] A.

n=1: q in [-0.089622880, 0.089622880] A.

**NEW n=1 harmonic range: [1.306625699, 1.485871459] A.**

Separate symmetric probability intervals (normalized oscillator densities; not turning points):

| n | Probability | q_min [A] | q_max [A] | s_min [A] | s_max [A] |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.95 | -0.0717119216 | 0.0717119216 | 1.32453666 | 1.4679605 |
| 0 | 0.99 | -0.0942454404 | 0.0942454404 | 1.30200314 | 1.49049402 |
| 0 | 0.999 | -0.120395067 | 0.120395067 | 1.27585351 | 1.51664365 |
| 1 | 0.95 | -0.102282233 | 0.102282233 | 1.29396635 | 1.49853081 |
| 1 | 0.99 | -0.123237525 | 0.123237525 | 1.27301105 | 1.5194861 |
| 1 | 0.999 | -0.14756617 | 0.14756617 | 1.24868241 | 1.54381475 |

Dominant vibrational mode: 2 at 1486.765255 cm^-1. Mixed coordinate (multiple weights >1e-6): True. Total secondary-mode weight: 0.421805%; substantial mixing (dominant weight <0.99): False. Weight sum: 1.000000000000. Spectral curvature reconstruction relative error: 4.97e-16.

| Mode (1-based) | Frequency [cm^-1] | Absolute overlap | Squared weight |
| --- | --- | --- | --- |
| 1 | 184.623246 | 0.0357318307 | 0.00127676372 |
| 2 | 1486.76526 | 0.997888748 | 0.995781954 |
| 3 | 1589.49635 | 2.17874263e-12 | 4.74691947e-24 |
| 4 | 1780.8871 | 0.0516592552 | 0.00266867865 |
| 5 | 4140.51722 | 0.0165107087 | 0.000272603501 |
| 6 | 4147.90448 | 4.97112139e-12 | 2.47120479e-23 |

## Reproducibility

```json
{
  "python": "3.13.2",
  "python_executable": "/Users/novaz/Desktop/qml2.0/.venv-rhf/bin/python",
  "pyscf": "2.14.0",
  "numpy": "2.5.3",
  "scipy": "1.18.1",
  "threads": 1,
  "script_sha256": "e6687127ab5752b7a203cab27f177572774e85e7a71acec9f4b617e699b17bd9",
  "source_hashes": {
    "rhf_hessian": "2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5",
    "harmonic_analysis": "bb5045974f9dd05aff0ae0fff33aaaca8cca233860a74ff8caf4c89bc165e0cd",
    "atom_mass_list": "9752bebb5dad67181d6c2c029d76d8c1af1aff38862fcd8f134f811803c6644c"
  },
  "constants": {
    "Bohr_A": 0.52917721092,
    "Bohr_m": 5.2917721092e-11,
    "Hartree_J": 4.359744644911914e-18,
    "amu_kg": 1.660539040427164e-27,
    "hbar_J_s": 1.0545718001391127e-34,
    "c_m_per_s": 299792458,
    "source": "installed pyscf.data.nist"
  },
  "SCF": {
    "conv_tol": 1e-13,
    "conv_tol_grad": 1e-10,
    "direct_scf_tol": 1e-14,
    "max_cycle": 400,
    "density_fitting": false,
    "all_electron": true,
    "conv_tol_cpscf": 1e-12
  }
}
```

Installed implementation was inspected and hashed. Convention references: [PySCF RHF Hessian](https://pyscf.org/_modules/pyscf/hessian/rhf.html), [PySCF harmonic analysis](https://pyscf.org/_modules/pyscf/hessian/thermo.html), [PySCF atomic masses](https://pyscf.org/_modules/pyscf/gto/mole.html#atom_mass_list).
