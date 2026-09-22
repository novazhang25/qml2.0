# RHF bound-level selection

The harmonic level energy measured upward from the potential minimum is

    epsilon_n = hbar * omega * (n + 1/2),

which is always positive. To test whether the level is bound, set the RHF-path dissociation threshold to zero. Since the equilibrium minimum is then at -D_e,

    E_n_rel = -D_e + epsilon_n.

Therefore E_n_rel < 0 means the level lies below dissociation.

All energies and energy thresholds in this report use Hartree (Eh). The harmonic quantum is hbar*omega expressed as an energy. Bond lengths and coordinate ranges retain Angstrom units.

## Numerical results

D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used.

The comparison is restricted to n=0 and n=1. Existing harmonic turning-point ranges are reused exactly. The plateau threshold is an actually sampled energy; no fit or extrapolated asymptote is substituted. A missing, unconverged or branch-ambiguous plateau produces no selection. PASS certifies only the numerical test within this constrained RHF model, not physically correct dissociation or actual bound vibrational levels.

| Molecule | D_e^(RHF-path) [Eh] | epsilon_0 [Eh] | epsilon_1 [Eh] | E0_rel [Eh] | E1_rel [Eh] | Selected n | Range min [A] | Range max [A] | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | 0.343978816 | 0.00425531167 | 0.012765935 | -0.339723504 | -0.331212881 | 1 | 1.26277286 | 1.75885084 | PASS |
| BeH2 | 0.663611524 | 0.00567399544 | 0.0170219863 | -0.657937529 | -0.646589538 | 1 | 1.14860752 | 1.43246932 | PASS |
| H2O | 0.758223457 | 0.00943047134 | 0.028291414 | -0.748792986 | -0.729932043 | 1 | 0.876681982 | 1.10213637 | PASS |
| NH3 | 1.27216031 | 0.00872883378 | 0.0261865013 | -1.26343147 | -1.2459738 | 1 | 0.937645635 | 1.12739979 | PASS |
| N2 | 0.772984272 | 0.00608144847 | 0.0182443454 | -0.766902823 | -0.754739926 | 1 | 1.06029712 | 1.20740489 | PASS |
| CO | 0.474045977 | 0.00560904931 | 0.0168271479 | -0.468436927 | -0.457218829 | 1 | 1.0680982 | 1.22286315 | PASS |
| HF | 0.561581315 | 0.0101938153 | 0.0305814459 | -0.5513875 | -0.530999869 | 1 | 0.801790953 | 1.10913444 | PASS |
| H2S | 0.733097482 | 0.00745796899 | 0.022373907 | -0.725639513 | -0.710723575 | 1 | 1.20307019 | 1.45424154 | PASS |
| H2O2 | 0.47798228 | 0.00339005423 | 0.0101701627 | -0.474592225 | -0.467812117 | 1 | 1.3066257 | 1.48587146 | PASS |

## Method and acceptance criteria

The only structural and harmonic input is the validated current harmonic JSON. Its builder hash, constants, equilibrium coordinate and tangent are checked before scanning. The nuclear Hessian and turning points are not recalculated. All nine molecules retain their RHF basis and existing mass-weighted Eckart/Kabsch path. For H2O2, s is O-O separation with both O-H lengths, O-O-H angles and H-O-O-H dihedral fixed. Other polyatomic s values are their common participating bond length.

Conventional all-electron RHF uses energy tolerance 1e-13 Eh, orbital-gradient acceptance <=1e-10, integral screening 1e-14 and at least 400 permitted SCF cycles. The numerical solver targets a tighter gradient and rebuilds the full Fock matrix. Every accepted point has a directly recomputed energy and orbital residual. All occupations remain 0 or 2. Neighboring paired orbitals are reorthonormalized in the new AO overlap metric and refined before diagonalization can reset nearly degenerate occupations. A second-order solver and restricted orbital rotations may improve convergence; they do not change the RHF energy model.

An internal RHF orbital-stability check rejects stationary electronic saddle points. Any correction rotates paired spatial orbitals within RHF and is recorded. This electronic stability matrix is distinct from the nuclear Hessian used in harmonic analysis. The check does not establish physical open-shell dissociation accuracy. At a candidate asymptote, an independent fresh initial density and a deterministically perturbed restricted orbital guess must reproduce the continuation energy within the plateau tolerance; otherwise the branch is flagged.

Initial q grid [A]: [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]. Afterwards the absolute separation s is doubled. The documented maximum displacement is 10000000 A. Extremely large separations, if needed, only test the numerical constrained-RHF asymptote; Coulombic restricted tails can decay slowly.

A plateau requires four consecutive accepted points: a three-point candidate and a farther confirmation. Each absolute-separation ratio must be >=1.5. All four energies must have full span, maximum difference from the last energy, and maximum adjacent difference strictly below 1e-06 Eh. Failed points are never removed from the consecutive-window test. Scanning stops after three consecutive SCF failures because outward RHF continuation is then unresolved. Reaching the maximum distance alone never certifies a plateau.

Geometry invariants are checked at every scan point with explicitly recorded roundoff-aware tolerances. The complete point list, failed solver attempts, energies, SCF residuals, internal stability and geometry diagnostics are preserved in JSON and per-molecule scan CSV files.

At the input boundary, the saved harmonic frequency is converted to the energy quantum hbar*omega in Hartree using hbar, light speed and the Hartree energy from the original harmonic calculation. All subsequent well-depth and level comparisons are performed directly in Hartree.

The exact decision is E1_rel<0 -> n=1, otherwise E0_rel<0 -> n=0, otherwise NONE. Any |E_n_rel| < 4.55633525805e-05 Eh is marked NEAR_THRESHOLD while retaining the exact sign decision. This warning window is not a silently applied sign tolerance. It covers the numerical plateau tolerance in Hartree; it does not quantify RHF model error. No Morse correction, experimental dissociation energy, UHF energy or historical scan is used.

## Complete summary

| molecule | basis | coordinate | s_eq_A | E_eq_Eh | E_diss_Eh | D_e_Eh | harmonic_quantum_Eh | half_hbar_omega_Eh | three_halves_hbar_omega_Eh | E0_rel_Eh | E1_rel_Eh | n0_bound | n1_bound | selected_n | selected_range_min_A | selected_range_max_A | asymptote_status | RHF_dissociation_warning | validation_status | selected_reason | near_threshold_levels | last_q_A | accepted_scan_points | failed_scan_points | bound_level_test_status | physical_dissociation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiH | sto-3g | q = r(Li-H) - r_e(Li-H) | 1.51081185 | -7.86338213 | -7.51940331 | 0.343978816 | 0.00851062335 | 0.00425531167 | 0.012765935 | -0.339723504 | -0.331212881 | True | True | 1 | 1.26277286 | 1.75885084 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. Fractional separated-fragment charges persist in the RHF population diagnostic; this is an additional dissociation-model artifact, not a physically validated fragment limit. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 3413539.01 | 31 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| BeH2 | sto-3g | r(Be-H_i) = r_e(Be-H_i) + q for every H | 1.29053842 | -15.5613528 | -14.8977413 | 0.663611524 | 0.0113479909 | 0.00567399544 | 0.0170219863 | -0.657937529 | -0.646589538 | True | True | 1 | 1.14860752 | 1.43246932 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 1649025.61 | 30 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| H2O | sto-3g | r(O-H_i) = r_e(O-H_i) + q for every H | 0.989409176 | -74.9659012 | -74.2076777 | 0.758223457 | 0.0188609427 | 0.00943047134 | 0.028291414 | -0.748792986 | -0.729932043 | True | True | 1 | 0.876681982 | 1.10213637 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. 2 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 1570086.69 | 30 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| NH3 | sto-3g | r(N-H_i) = r_e(N-H_i) + q for every H | 1.03252271 | -55.4554198 | -54.1832595 | 1.27216031 | 0.0174576676 | 0.00872883378 | 0.0261865013 | -1.26343147 | -1.2459738 | True | True | 1 | 0.937645635 | 1.12739979 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. 4 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. Fractional separated-fragment charges persist in the RHF population diagnostic; this is an additional dissociation-model artifact, not a physically validated fragment limit. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 3162778.24 | 31 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| N2 | sto-3g | q = r(N-N) - r_e(N-N) | 1.13385101 | -107.500654 | -106.72767 | 0.772984272 | 0.0121628969 | 0.00608144847 | 0.0182443454 | -0.766902823 | -0.754739926 | True | True | 1 | 1.06029712 | 1.20740489 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. 2 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 3215903.34 | 31 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| CO | sto-3g | q = r(C-O) - r_e(C-O) | 1.14548068 | -111.22545 | -110.751404 | 0.474045977 | 0.0112180986 | 0.00560904931 | 0.0168271479 | -0.468436927 | -0.457218829 | True | True | 1 | 1.0680982 | 1.22286315 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 195.509901 | 17 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| HF | sto-3g | q = r(H-F) - r_e(H-F) | 0.955462699 | -98.5728473 | -98.011266 | 0.561581315 | 0.0203876306 | 0.0101938153 | 0.0305814459 | -0.5513875 | -0.530999869 | True | True | 1 | 0.801790953 | 1.10913444 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. Fractional separated-fragment charges persist in the RHF population diagnostic; this is an additional dissociation-model artifact, not a physically validated fragment limit. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 3122376.67 | 31 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| H2S | sto-3g | r(S-H_i) = r_e(S-H_i) + q for every H | 1.32865587 | -394.31163 | -393.578533 | 0.733097482 | 0.014915938 | 0.00745796899 | 0.022373907 | -0.725639513 | -0.710723575 | True | True | 1 | 1.20307019 | 1.45424154 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. 2 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 1659017.84 | 30 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |
| H2O2 | sto-3g | q = r(O-O) - r_e(O-O); both OH lengths, OOH angles and HOOH torsion fixed | 1.39624858 | -148.764997 | -148.287014 | 0.47798228 | 0.00678010846 | 0.00339005423 | 0.0101701627 | -0.474592225 | -0.467812117 | True | True | 1 | 1.3066257 | 1.48587146 | PLATEAU_CONFIRMED | D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental dissociation energy. Restricted closed-shell RHF cannot generally describe the separated open-shell fragments correctly. No UHF or correlated asymptote is used. Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry. | PASS | n=1 lies below the RHF-path dissociation threshold | [] | 3353474.98 | 31 | 0 | BOUND_LEVEL_SELECTED | NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL |

## LiH — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: q = r(Li-H) - r_e(Li-H). s_eq=1.510811853 A.

- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.
- Fractional separated-fragment charges persist in the RHF population diagnostic; this is an additional dissociation-model artifact, not a physically validated fragment limit.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| Li1 | 0.0763663988 |
| H2 | -0.0763663988 |

E_eq=-7.86338212892110 Eh; E_diss=-7.51940331329233 Eh; D_e^(RHF-path)=0.343978815629 Eh.

```text
D_e^(RHF-path)            = 0.343978815629 Hartree
n=0 energy above minimum = 0.004255311673 Hartree
n=0 relative to diss.    = -0.339723503956 Hartree BOUND
n=1 energy above minimum = 0.012765935019 Hartree
n=1 relative to diss.    = -0.331212880609 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [1.262772864, 1.758850842] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 5.457446539125499e-07,
  "pairwise_span_Eh": 5.457446539125499e-07,
  "max_neighbor_change_Eh": 3.118540883306764e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 1706768.7515893674,
  "confirmation_q_A": 3413539.0139905876,
  "window_q_A": [
    426691.05478845205,
    853383.6203887572,
    1706768.7515893674,
    3413539.0139905876
  ],
  "window_energies_Eh": [
    -7.519403859036988,
    -7.5194035471828995,
    -7.5194033912558496,
    -7.519403313292334
  ],
  "extension_change_Eh": 7.796351564337556e-08,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=3413539.01 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.51081185 | -7.86338212892110 | True | 9.34225181e-13 | True | True |
| 0.1 | 1.61081185 | -7.86149351213369 | True | 3.01519728e-16 | True | True |
| 0.2 | 1.71081185 | -7.85647791974663 | True | 6.93843859e-16 | True | True |
| 0.3 | 1.81081185 | -7.84913192448547 | True | 3.8368104e-16 | True | True |
| 0.5 | 2.01081185 | -7.82974068725217 | True | 6.96264075e-16 | True | True |
| 0.75 | 2.26081185 | -7.80061838915409 | True | 4.36993873e-16 | True | True |
| 1 | 2.51081185 | -7.76952162526570 | True | 1.83314128e-15 | True | True |
| 1.5 | 3.01081185 | -7.70962401212764 | True | 3.80616862e-16 | True | True |
| 2 | 3.51081185 | -7.66027690629447 | True | 1.1279419e-15 | True | True |
| 3 | 4.51081185 | -7.59998411206060 | True | 2.56307029e-13 | True | True |
| 4 | 5.51081185 | -7.57358145768659 | True | 6.63944164e-16 | True | True |
| 5 | 6.51081185 | -7.56168042029736 | True | 3.92447169e-13 | True | True |
| 11.5108119 | 13.0216237 | -7.53986299567033 | True | 8.01100431e-13 | True | True |
| 24.5324356 | 26.0432474 | -7.52962520941155 | True | 3.41803096e-16 | True | True |
| 50.575683 | 52.0864948 | -7.52451328171304 | True | 2.3803192e-16 | True | True |
| 102.662178 | 104.17299 | -7.52195809079209 | True | 2.32280189e-13 | True | True |
| 206.835167 | 208.345979 | -7.52068062562026 | True | 3.39829196e-15 | True | True |
| 415.181147 | 416.691959 | -7.52004192142796 | True | 4.43849924e-16 | True | True |
| 831.873105 | 833.383917 | -7.51972257614037 | True | 2.26422356e-16 | True | True |
| 1665.25702 | 1666.76783 | -7.51956290517705 | True | 3.71733906e-16 | True | True |
| 3332.02486 | 3333.53567 | -7.51948307011374 | True | 1.47732813e-16 | True | True |
| 6665.56053 | 6667.07134 | -7.51944315268650 | True | 8.02877843e-12 | True | True |
| 13332.6319 | 13334.1427 | -7.51942319399897 | True | 2.00172691e-12 | True | True |
| 26666.7745 | 26668.2854 | -7.51941321466171 | True | 4.99490522e-13 | True | True |
| 53335.0599 | 53336.5707 | -7.51940822499472 | True | 1.24722142e-13 | True | True |
| 106671.631 | 106673.141 | -7.51940573016163 | True | 3.09330543e-14 | True | True |
| 213344.772 | 213346.283 | -7.51940448274519 | True | 7.96078093e-15 | True | True |
| 426691.055 | 426692.566 | -7.51940385903699 | True | 1.90962773e-15 | True | True |
| 853383.62 | 853385.131 | -7.51940354718290 | True | 6.81864759e-16 | True | True |
| 1706768.75 | 1706770.26 | -7.51940339125585 | True | 2.84045982e-16 | True | True |
| 3413539.01 | 3413540.52 | -7.51940331329233 | True | 1.91373807e-16 | True | True |

## BeH2 — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: r(Be-H_i) = r_e(Be-H_i) + q for every H. s_eq=1.290538421 A.

- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| Be1 | 3.55271368e-15 |
| H2 | 6.66133815e-16 |
| H3 | 6.66133815e-16 |

E_eq=-15.56135280778207 Eh; E_diss=-14.89774128358228 Eh; D_e^(RHF-path)=0.663611524200 Eh.

```text
D_e^(RHF-path)            = 0.663611524200 Hartree
n=0 energy above minimum = 0.005673995442 Hartree
n=0 relative to diss.    = -0.657937528758 Hartree BOUND
n=1 energy above minimum = 0.017021986325 Hartree
n=1 relative to diss.    = -0.646589537875 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [1.148607521, 1.432469321] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 5.615797782354548e-07,
  "pairwise_span_Eh": 5.615797782354548e-07,
  "max_neighbor_change_Eh": 3.209027248374241e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 824512.1613786288,
  "confirmation_q_A": 1649025.6132956787,
  "window_q_A": [
    206127.07244084147,
    412255.4354201039,
    824512.1613786288,
    1649025.6132956787
  ],
  "window_energies_Eh": [
    -14.897741845162063,
    -14.897741524259338,
    -14.897741363807995,
    -14.897741283582285
  ],
  "extension_change_Eh": 8.022571051924388e-08,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=1649025.61 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.29053842 | -15.56135280778207 | True | 4.73855035e-13 | True | True |
| 0.1 | 1.39053842 | -15.55384967836462 | True | 3.23606753e-15 | True | True |
| 0.2 | 1.49053842 | -15.53451309035655 | True | 9.0543595e-16 | True | True |
| 0.3 | 1.59053842 | -15.50700972376832 | True | 2.66660455e-15 | True | True |
| 0.5 | 1.79053842 | -15.43723407565248 | True | 9.34815437e-16 | True | True |
| 0.75 | 2.04053842 | -15.33805888347827 | True | 5.7596781e-16 | True | True |
| 1 | 2.29053842 | -15.23918143329822 | True | 9.23421323e-16 | True | True |
| 1.5 | 2.79053842 | -15.07398931475581 | True | 6.8524108e-15 | True | True |
| 2 | 3.29053842 | -14.97656541222490 | True | 5.12691533e-16 | True | True |
| 3 | 4.29053842 | -14.92986639493140 | True | 1.22783313e-16 | True | True |
| 4 | 5.29053842 | -14.92277180113041 | True | 4.28248358e-16 | True | True |
| 5 | 6.29053842 | -14.91877214767153 | True | 2.35597644e-15 | True | True |
| 11.2905384 | 12.5810768 | -14.90825654352911 | True | 5.47617601e-18 | True | True |
| 23.8716153 | 25.1621537 | -14.90299887344287 | True | 9.68717151e-15 | True | True |
| 49.0337689 | 50.3243074 | -14.90037003839976 | True | 9.87382219e-15 | True | True |
| 99.3580763 | 100.648615 | -14.89905562087819 | True | 1.01220371e-14 | True | True |
| 200.006691 | 201.297229 | -14.89839841211738 | True | 1.03019669e-14 | True | True |
| 401.303921 | 402.594459 | -14.89806980773701 | True | 1.00981366e-14 | True | True |
| 803.898379 | 805.188918 | -14.89790550554680 | True | 1.08364027e-14 | True | True |
| 1609.0873 | 1610.37784 | -14.89782335445172 | True | 1.02194776e-14 | True | True |
| 3219.46513 | 3220.75567 | -14.89778227890417 | True | 1.03029829e-14 | True | True |
| 6440.2208 | 6441.51134 | -14.89776174113041 | True | 1.02726958e-14 | True | True |
| 12881.7321 | 12883.0227 | -14.89775147224352 | True | 1.07199702e-14 | True | True |
| 25764.7548 | 25766.0454 | -14.89774633780005 | True | 1.05158388e-14 | True | True |
| 51530.8002 | 51532.0907 | -14.89774377057835 | True | 1.03263542e-14 | True | True |
| 103062.891 | 103064.181 | -14.89774248696747 | True | 1.02859742e-14 | True | True |
| 206127.072 | 206128.363 | -14.89774184516206 | True | 1.02807659e-14 | True | True |
| 412255.435 | 412256.726 | -14.89774152425934 | True | 1.02825417e-14 | True | True |
| 824512.161 | 824513.452 | -14.89774136380800 | True | 1.02235552e-14 | True | True |
| 1649025.61 | 1649026.9 | -14.89774128358228 | True | 1.03120235e-14 | True | True |

## H2O — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: r(O-H_i) = r_e(O-H_i) + q for every H. s_eq=0.989409176 A.

- 2 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals.
- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| O1 | 1.77635684e-15 |
| H2 | -1.16464616e-11 |
| H3 | 1.16454624e-11 |

E_eq=-74.96590119229930 Eh; E_diss=-74.20767773502641 Eh; D_e^(RHF-path)=0.758223457273 Eh.

```text
D_e^(RHF-path)            = 0.758223457273 Hartree
n=0 energy above minimum = 0.009430471341 Hartree
n=0 relative to diss.    = -0.748792985932 Hartree BOUND
n=1 energy above minimum = 0.028291414023 Hartree
n=1 relative to diss.    = -0.729932043250 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [0.876681982, 1.102136371] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 7.697972108644535e-07,
  "pairwise_span_Eh": 7.697972108644535e-07,
  "max_neighbor_change_Eh": 4.398841610964155e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 785042.850157991,
  "confirmation_q_A": 1570086.6897251585,
  "window_q_A": [
    196259.9704826155,
    392520.9303744073,
    785042.850157991,
    1570086.6897251585
  ],
  "window_energies_Eh": [
    -74.20767850482362,
    -74.20767806493946,
    -74.20767784499745,
    -74.20767773502641
  ],
  "extension_change_Eh": 1.0997104027410387e-07,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=1570086.69 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.989409176 | -74.96590119229930 | True | 9.18945336e-13 | True | True |
| 0.1 | 1.08940918 | -74.94730016531081 | True | 4.49136707e-15 | True | True |
| 0.2 | 1.18940918 | -74.90300403811705 | True | 4.05354287e-15 | True | True |
| 0.3 | 1.28940918 | -74.84500562895876 | True | 4.70305022e-12 | True | True |
| 0.5 | 1.48940918 | -74.71337641545507 | True | 1.87067031e-15 | True | True |
| 0.75 | 1.73940918 | -74.54966812958406 | True | 3.59394456e-15 | True | True |
| 1 | 1.98940918 | -74.40753633270559 | True | 3.96897511e-14 | True | True |
| 1.5 | 2.48940918 | -74.29277947746333 | True | 1.59673024e-12 | True | True |
| 2 | 2.98940918 | -74.26751268054844 | True | 1.31386782e-13 | True | True |
| 3 | 3.98940918 | -74.25098360308559 | True | 1.44328115e-15 | True | True |
| 4 | 4.98940918 | -74.24228689945936 | True | 2.48152037e-13 | True | True |
| 5 | 5.98940918 | -74.23650689292857 | True | 5.18718866e-16 | True | True |
| 10.9894092 | 11.9788184 | -74.22209176320341 | True | 3.71992879e-13 | True | True |
| 22.9682275 | 23.9576367 | -74.21488468668809 | True | 6.64396304e-12 | True | True |
| 46.9258642 | 47.9152734 | -74.21128115575773 | True | 7.63473045e-14 | True | True |
| 94.8411376 | 95.8305468 | -74.20947939040482 | True | 1.02874725e-15 | True | True |
| 190.671684 | 191.661094 | -74.20857850773004 | True | 4.50466172e-16 | True | True |
| 382.332778 | 383.322187 | -74.20812806639270 | True | 3.56383742e-15 | True | True |
| 765.654965 | 766.644375 | -74.20790284572405 | True | 2.71006021e-15 | True | True |
| 1532.29934 | 1533.28875 | -74.20779023538965 | True | 4.49554986e-15 | True | True |
| 3065.58809 | 3066.5775 | -74.20773393022257 | True | 1.20882005e-15 | True | True |
| 6132.16559 | 6133.155 | -74.20770577763898 | True | 7.89227218e-12 | True | True |
| 12265.3206 | 12266.31 | -74.20769170134713 | True | 8.87941286e-12 | True | True |
| 24531.6306 | 24532.62 | -74.20768466320133 | True | 9.00283852e-12 | True | True |
| 49064.2506 | 49065.24 | -74.20768114412830 | True | 9.01841072e-12 | True | True |
| 98129.4905 | 98130.4799 | -74.20767938459180 | True | 9.02035629e-12 | True | True |
| 196259.97 | 196260.96 | -74.20767850482362 | True | 9.02063036e-12 | True | True |
| 392520.93 | 392521.92 | -74.20767806493946 | True | 9.0203049e-12 | True | True |
| 785042.85 | 785043.84 | -74.20767784499745 | True | 9.02082956e-12 | True | True |
| 1570086.69 | 1570087.68 | -74.20767773502641 | True | 9.02066266e-12 | True | True |

## NH3 — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: r(N-H_i) = r_e(N-H_i) + q for every H. s_eq=1.032522714 A.

- 4 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals.
- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.
- Fractional separated-fragment charges persist in the RHF population diagnostic; this is an additional dissociation-model artifact, not a physically validated fragment limit.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| N1 | 0.0498101301 |
| H2 | -0.0166033767 |
| H3 | -0.0166033767 |
| H4 | -0.0166033767 |

E_eq=-55.45541977879949 Eh; E_diss=-54.18325947352356 Eh; D_e^(RHF-path)=1.272160305276 Eh.

```text
D_e^(RHF-path)            = 1.272160305276 Hartree
n=0 energy above minimum = 0.008728833776 Hartree
n=0 relative to diss.    = -1.263431471499 Hartree BOUND
n=1 energy above minimum = 0.026186501329 Hartree
n=1 relative to diss.    = -1.245973803947 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [0.937645635, 1.127399792] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 9.57898478759489e-07,
  "pairwise_span_Eh": 9.57898478759489e-07,
  "max_neighbor_change_Eh": 5.473705400049766e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 1581388.601702368,
  "confirmation_q_A": 3162778.235927449,
  "window_q_A": [
    395346.37603355676,
    790693.7845898272,
    1581388.601702368,
    3162778.235927449
  ],
  "window_energies_Eh": [
    -54.183260431422035,
    -54.183259884051495,
    -54.183259610366164,
    -54.183259473523556
  ],
  "extension_change_Eh": 1.3684260835589157e-07,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=3162778.24 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.03252271 | -55.45541977879949 | True | 9.90090699e-13 | True | True |
| 0.1 | 1.13252271 | -55.43096218765861 | True | 3.6966233e-12 | True | True |
| 0.2 | 1.23252271 | -55.37219667624657 | True | 4.44667911e-15 | True | True |
| 0.3 | 1.33252271 | -55.29457405685081 | True | 5.65085941e-15 | True | True |
| 0.5 | 1.53252271 | -55.11712435316838 | True | 6.52615681e-15 | True | True |
| 0.75 | 1.78252271 | -54.89512224667610 | True | 6.14206664e-15 | True | True |
| 1 | 2.03252271 | -54.69884936503860 | True | 2.7248589e-15 | True | True |
| 1.5 | 2.53252271 | -54.45287543061595 | True | 1.21691262e-12 | True | True |
| 2 | 3.03252271 | -54.35843223808294 | True | 4.47138643e-13 | True | True |
| 3 | 4.03252271 | -54.29436666431690 | True | 6.65600968e-12 | True | True |
| 4 | 5.03252271 | -54.26991251756002 | True | 7.49018026e-14 | True | True |
| 5 | 6.03252271 | -54.25529573450163 | True | 1.04147409e-12 | True | True |
| 11.0325227 | 12.0650454 | -54.21916847085532 | True | 4.1032629e-15 | True | True |
| 23.0975681 | 24.1300909 | -54.20120047610676 | True | 7.9226995e-12 | True | True |
| 47.227659 | 48.2601817 | -54.19222815659272 | True | 1.73478528e-15 | True | True |
| 95.4878407 | 96.5203634 | -54.18774350641171 | True | 5.44289913e-15 | True | True |
| 192.008204 | 193.040727 | -54.18550138591446 | True | 3.18350411e-14 | True | True |
| 385.048931 | 386.081454 | -54.18438035542844 | True | 1.02808037e-12 | True | True |
| 771.130385 | 772.162907 | -54.18381984496642 | True | 2.13590617e-15 | True | True |
| 1543.29329 | 1544.32581 | -54.18353959059886 | True | 4.83460107e-13 | True | True |
| 3087.61911 | 3088.65163 | -54.18339946358963 | True | 1.03106771e-15 | True | True |
| 6176.27074 | 6177.30326 | -54.18332940012342 | True | 2.3297205e-14 | True | True |
| 12353.574 | 12354.6065 | -54.18329436839927 | True | 2.35964606e-15 | True | True |
| 24708.1805 | 24709.213 | -54.18327685253941 | True | 2.59806961e-12 | True | True |
| 49417.3935 | 49418.4261 | -54.18326809460992 | True | 6.49631595e-13 | True | True |
| 98835.8196 | 98836.8521 | -54.18326371564543 | True | 1.6243751e-13 | True | True |
| 197672.672 | 197673.704 | -54.18326152616317 | True | 4.0587808e-14 | True | True |
| 395346.376 | 395347.409 | -54.18326043142203 | True | 1.01110889e-14 | True | True |
| 790693.785 | 790694.817 | -54.18325988405149 | True | 2.51207203e-15 | True | True |
| 1581388.6 | 1581389.63 | -54.18325961036616 | True | 8.98034147e-15 | True | True |
| 3162778.24 | 3162779.27 | -54.18325947352356 | True | 1.29193106e-14 | True | True |

## N2 — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: q = r(N-N) - r_e(N-N). s_eq=1.133851007 A.

- 2 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals.
- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| N1 | 1.77635684e-15 |
| N2 | 1.77635684e-15 |

E_eq=-107.50065426275155 Eh; E_diss=-106.72766999119530 Eh; D_e^(RHF-path)=0.772984271556 Eh.

```text
D_e^(RHF-path)            = 0.772984271556 Hartree
n=0 energy above minimum = 0.006081448466 Hartree
n=0 relative to diss.    = -0.766902823091 Hartree BOUND
n=1 energy above minimum = 0.018244345397 Hartree
n=1 relative to diss.    = -0.754739926159 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [1.060297120, 1.207404895] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 5.75925184875814e-07,
  "pairwise_span_Eh": 5.75925184875814e-07,
  "max_neighbor_change_Eh": 3.2910004676978133e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 1607951.1046574996,
  "confirmation_q_A": 3215903.3431660067,
  "window_q_A": [
    401986.9257761193,
    803974.9854032461,
    1607951.1046574996,
    3215903.3431660067
  ],
  "window_energies_Eh": [
    -106.72767056712048,
    -106.72767023802044,
    -106.72767007347052,
    -106.7276699911953
  ],
  "extension_change_Eh": 8.227522130255238e-08,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=3215903.34 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.13385101 | -107.50065426275155 | True | 7.0811256e-13 | True | True |
| 0.1 | 1.23385101 | -107.47316536587796 | True | 9.97819491e-13 | True | True |
| 0.2 | 1.33385101 | -107.40980384424836 | True | 2.84561765e-13 | True | True |
| 0.3 | 1.43385101 | -107.32999972935380 | True | 1.23107492e-13 | True | True |
| 0.5 | 1.63385101 | -107.20898206508451 | True | 4.89701059e-14 | True | True |
| 0.75 | 1.88385101 | -107.10707025696750 | True | 1.53056066e-14 | True | True |
| 1 | 2.13385101 | -107.02579249393767 | True | 5.0873703e-15 | True | True |
| 1.5 | 2.63385101 | -106.90828121631571 | True | 4.77137465e-14 | True | True |
| 2 | 3.13385101 | -106.84194019727818 | True | 2.64005151e-15 | True | True |
| 3 | 4.13385101 | -106.79511567442343 | True | 3.7078809e-12 | True | True |
| 4 | 5.13385101 | -106.78020058293750 | True | 2.78532244e-15 | True | True |
| 5 | 6.13385101 | -106.77135512809676 | True | 4.40031041e-15 | True | True |
| 11.133851 | 12.267702 | -106.74930328504269 | True | 2.19005765e-15 | True | True |
| 23.401553 | 24.535404 | -106.73846190267604 | True | 2.80126334e-15 | True | True |
| 47.9369571 | 49.0708081 | -106.73306288276774 | True | 1.86055635e-15 | True | True |
| 97.0077651 | 98.1416161 | -106.73036602114991 | True | 3.71346589e-13 | True | True |
| 195.149381 | 196.283232 | -106.72901791837290 | True | 1.02717734e-14 | True | True |
| 391.432613 | 392.566464 | -106.72834390782401 | True | 2.76772296e-14 | True | True |
| 783.999078 | 785.132929 | -106.72800690764494 | True | 1.51761692e-13 | True | True |
| 1569.13201 | 1570.26586 | -106.72783840819172 | True | 1.61681274e-15 | True | True |
| 3139.39786 | 3140.53172 | -106.72775415854474 | True | 9.5368975e-16 | True | True |
| 6279.92958 | 6281.06343 | -106.72771203373115 | True | 8.59060151e-16 | True | True |
| 12560.993 | 12562.1269 | -106.72769097132556 | True | 1.25027275e-15 | True | True |
| 25123.1199 | 25124.2537 | -106.72768044012297 | True | 1.07273844e-15 | True | True |
| 50247.3736 | 50248.5075 | -106.72767517452162 | True | 2.36172797e-15 | True | True |
| 100495.881 | 100497.015 | -106.72767254172095 | True | 8.84199767e-16 | True | True |
| 200992.896 | 200994.03 | -106.72767122532063 | True | 1.54357304e-15 | True | True |
| 401986.926 | 401988.06 | -106.72767056712048 | True | 3.89570224e-12 | True | True |
| 803974.985 | 803976.119 | -106.72767023802044 | True | 4.8707094e-12 | True | True |
| 1607951.1 | 1607952.24 | -106.72767007347052 | True | 5.11326147e-12 | True | True |
| 3215903.34 | 3215904.48 | -106.72766999119530 | True | 5.1751919e-12 | True | True |

## CO — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: q = r(C-O) - r_e(C-O). s_eq=1.145480676 A.

- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| C1 | 8.8817842e-16 |
| O2 | -7.10542736e-15 |

E_eq=-111.22544951413573 Eh; E_diss=-110.75140353733572 Eh; D_e^(RHF-path)=0.474045976800 Eh.

```text
D_e^(RHF-path)            = 0.474045976800 Hartree
n=0 energy above minimum = 0.005609049309 Hartree
n=0 relative to diss.    = -0.468436927491 Hartree BOUND
n=1 energy above minimum = 0.016827147927 Hartree
n=1 relative to diss.    = -0.457218828873 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [1.068098198, 1.222863154] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 6.703159272092307e-08,
  "pairwise_span_Eh": 6.703159272092307e-08,
  "max_neighbor_change_Eh": 6.493891646641714e-08,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 97.18221013930999,
  "confirmation_q_A": 195.509900954574,
  "window_q_A": [
    23.436442027861997,
    48.018364731678,
    97.18221013930999,
    195.509900954574
  ],
  "window_energies_Eh": [
    -110.75140360436731,
    -110.7514035394284,
    -110.75140353739904,
    -110.75140353733572
  ],
  "extension_change_Eh": 6.332356861094013e-11,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=195.509901 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.14548068 | -111.22544951413573 | True | 9.76387486e-13 | True | True |
| 0.1 | 1.24548068 | -111.20249807967048 | True | 4.6477051e-12 | True | True |
| 0.2 | 1.34548068 | -111.14955770319571 | True | 6.95748004e-12 | True | True |
| 0.3 | 1.44548068 | -111.08279200766333 | True | 7.17471539e-15 | True | True |
| 0.5 | 1.64548068 | -110.94429835536695 | True | 5.34668779e-15 | True | True |
| 0.75 | 1.89548068 | -110.83318221815281 | True | 1.45875952e-13 | True | True |
| 1 | 2.14548068 | -110.79142593292588 | True | 1.35252326e-12 | True | True |
| 1.5 | 2.64548068 | -110.76170791654714 | True | 4.9129538e-16 | True | True |
| 2 | 3.14548068 | -110.75408982369788 | True | 2.44349231e-16 | True | True |
| 3 | 4.14548068 | -110.75190014617563 | True | 2.830081e-17 | True | True |
| 4 | 5.14548068 | -110.75157071736334 | True | 3.05863303e-16 | True | True |
| 5 | 6.14548068 | -110.75147226627655 | True | 1.10926496e-17 | True | True |
| 11.1454807 | 12.2909614 | -110.75140568270933 | True | 1.98834297e-18 | True | True |
| 23.436442 | 24.5819227 | -110.75140360436731 | True | 9.39957894e-15 | True | True |
| 48.0183647 | 49.1638454 | -110.75140353942840 | True | 2.2552303e-18 | True | True |
| 97.1822101 | 98.3276908 | -110.75140353739904 | True | 3.35272542e-16 | True | True |
| 195.509901 | 196.655382 | -110.75140353733572 | True | 6.94720784e-16 | True | True |

## HF — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: q = r(H-F) - r_e(H-F). s_eq=0.955462699 A.

- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.
- Fractional separated-fragment charges persist in the RHF population diagnostic; this is an additional dissociation-model artifact, not a physically validated fragment limit.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| H1 | 0.0522399014 |
| F2 | -0.0522399014 |

E_eq=-98.57284734715999 Eh; E_diss=-98.01126603214851 Eh; D_e^(RHF-path)=0.561581315011 Eh.

```text
D_e^(RHF-path)            = 0.561581315011 Hartree
n=0 energy above minimum = 0.010193815307 Hartree
n=0 relative to diss.    = -0.551387499705 Hartree BOUND
n=1 energy above minimum = 0.030581445920 Hartree
n=1 relative to diss.    = -0.530999869092 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [0.801790953, 1.109134444] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 5.947950398876856e-07,
  "pairwise_span_Eh": 5.947950398876856e-07,
  "max_neighbor_change_Eh": 3.3988283121288987e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 1561187.8582017212,
  "confirmation_q_A": 3122376.671866141,
  "window_q_A": [
    390296.24795340636,
    780593.4513695113,
    1561187.8582017212,
    3122376.671866141
  ],
  "window_energies_Eh": [
    -98.01126662694355,
    -98.01126628706072,
    -98.01126611711923,
    -98.01126603214851
  ],
  "extension_change_Eh": 8.497072201407718e-08,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=3122376.67 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.955462699 | -98.57284734715999 | True | 7.5475112e-13 | True | True |
| 0.1 | 1.0554627 | -98.56213027561877 | True | 7.51989036e-13 | True | True |
| 0.2 | 1.1554627 | -98.53695183949179 | True | 1.86918065e-13 | True | True |
| 0.3 | 1.2554627 | -98.50439083966518 | True | 4.81460491e-14 | True | True |
| 0.5 | 1.4554627 | -98.43127930597626 | True | 2.15496458e-12 | True | True |
| 0.75 | 1.7054627 | -98.34175739102142 | True | 5.07262404e-13 | True | True |
| 1 | 1.9554627 | -98.26654956423295 | True | 7.23301113e-15 | True | True |
| 1.5 | 2.4554627 | -98.16849439916098 | True | 3.14433259e-15 | True | True |
| 2 | 2.9554627 | -98.11900081516694 | True | 2.43936737e-15 | True | True |
| 3 | 3.9554627 | -98.08015885187410 | True | 3.87740248e-16 | True | True |
| 4 | 4.9554627 | -98.06512334866397 | True | 2.76983339e-12 | True | True |
| 5 | 5.9554627 | -98.05597694033598 | True | 2.48740837e-13 | True | True |
| 10.9554627 | 11.9109254 | -98.03356206539736 | True | 4.36404389e-12 | True | True |
| 22.8663881 | 23.8218508 | -98.02240629464734 | True | 7.02898824e-14 | True | True |
| 46.6882389 | 47.6437016 | -98.01683506577798 | True | 3.70442176e-15 | True | True |
| 94.3319405 | 95.2874032 | -98.01405035108425 | True | 6.32510309e-16 | True | True |
| 189.619344 | 190.574806 | -98.01265812378563 | True | 2.11384179e-15 | True | True |
| 380.19415 | 381.149613 | -98.01196203083039 | True | 3.04306247e-14 | True | True |
| 761.343763 | 762.299225 | -98.01161398805168 | True | 1.3483754e-13 | True | True |
| 1523.64299 | 1524.59845 | -98.01143996740291 | True | 3.06220223e-16 | True | True |
| 3048.24144 | 3049.1969 | -98.01135295724070 | True | 7.2934995e-12 | True | True |
| 6097.43834 | 6098.3938 | -98.01130945219724 | True | 1.82137844e-12 | True | True |
| 12195.8321 | 12196.7876 | -98.01128769968456 | True | 4.54575091e-13 | True | True |
| 24392.6198 | 24393.5752 | -98.01127682343048 | True | 1.12835515e-13 | True | True |
| 48786.195 | 48787.1504 | -98.01127138530400 | True | 2.76931081e-14 | True | True |
| 97573.3454 | 97574.3009 | -98.01126866624085 | True | 9.32861156e-15 | True | True |
| 195147.646 | 195148.602 | -98.01126730670931 | True | 7.54157529e-16 | True | True |
| 390296.248 | 390297.203 | -98.01126662694355 | True | 2.88450999e-16 | True | True |
| 780593.451 | 780594.407 | -98.01126628706072 | True | 2.64346425e-16 | True | True |
| 1561187.86 | 1561188.81 | -98.01126611711923 | True | 7.66065038e-16 | True | True |
| 3122376.67 | 3122377.63 | -98.01126603214851 | True | 1.37539617e-15 | True | True |

## H2S — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: r(S-H_i) = r_e(S-H_i) + q for every H. s_eq=1.328655869 A.

- 2 restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals.
- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| S1 | 1.77635684e-15 |
| H2 | -2.92721403e-12 |
| H3 | 2.92876834e-12 |

E_eq=-394.31163005927652 Eh; E_diss=-393.57853257752845 Eh; D_e^(RHF-path)=0.733097481748 Eh.

```text
D_e^(RHF-path)            = 0.733097481748 Hartree
n=0 energy above minimum = 0.007457968994 Hartree
n=0 relative to diss.    = -0.725639512754 Hartree BOUND
n=1 energy above minimum = 0.022373906982 Hartree
n=1 relative to diss.    = -0.710723574766 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [1.203070195, 1.454241543] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 7.725765840405074e-07,
  "pairwise_span_Eh": 7.725765840405074e-07,
  "max_neighbor_change_Eh": 4.4147282096673734e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 829508.2533848701,
  "confirmation_q_A": 1659017.8354256093,
  "window_q_A": [
    207376.06685431593,
    414753.4623645007,
    829508.2533848701,
    1659017.8354256093
  ],
  "window_energies_Eh": [
    -393.57853335010503,
    -393.5785329086322,
    -393.5785326878965,
    -393.57853257752845
  ],
  "extension_change_Eh": 1.1036803471142775e-07,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=1659017.84 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.32865587 | -394.31163005927652 | True | 4.92280127e-13 | True | True |
| 0.1 | 1.42865587 | -394.29934471457864 | True | 1.18172699e-13 | True | True |
| 0.2 | 1.52865587 | -394.26872778979958 | True | 7.0965389e-14 | True | True |
| 0.3 | 1.62865587 | -394.22675269471290 | True | 3.86250301e-14 | True | True |
| 0.5 | 1.82865587 | -394.12611049840245 | True | 5.90870854e-12 | True | True |
| 0.75 | 2.07865587 | -393.99461333180875 | True | 5.88684182e-12 | True | True |
| 1 | 2.32865587 | -393.87546908790978 | True | 3.59717286e-13 | True | True |
| 1.5 | 2.82865587 | -393.69983546349863 | True | 1.44363142e-14 | True | True |
| 2 | 3.32865587 | -393.64054222347880 | True | 2.50210472e-15 | True | True |
| 3 | 4.32865587 | -393.62091283270604 | True | 3.16962609e-15 | True | True |
| 4 | 5.32865587 | -393.61290628663141 | True | 8.00316356e-15 | True | True |
| 5 | 6.32865587 | -393.60746893905781 | True | 8.33377242e-15 | True | True |
| 11.3286559 | 12.6573117 | -393.59299869491997 | True | 5.15006726e-12 | True | True |
| 23.9859676 | 25.3146235 | -393.58576555091139 | True | 5.76967501e-15 | True | True |
| 49.3005911 | 50.629247 | -393.58214900857473 | True | 1.42555681e-13 | True | True |
| 99.929838 | 101.258494 | -393.58034073786013 | True | 8.64324723e-15 | True | True |
| 201.188332 | 202.516988 | -393.57943660251033 | True | 2.34249694e-15 | True | True |
| 403.70532 | 405.033976 | -393.57898453483517 | True | 1.59874846e-15 | True | True |
| 808.739295 | 810.067951 | -393.57875850099811 | True | 8.63474496e-15 | True | True |
| 1618.80725 | 1620.1359 | -393.57864548407872 | True | 3.24617705e-15 | True | True |
| 3238.94315 | 3240.2718 | -393.57858897561982 | True | 7.02986613e-15 | True | True |
| 6479.21495 | 6480.54361 | -393.57856072139003 | True | 3.7642723e-15 | True | True |
| 12959.7586 | 12961.0872 | -393.57854659427511 | True | 1.98538676e-12 | True | True |
| 25920.8458 | 25922.1744 | -393.57853953071822 | True | 2.23408306e-12 | True | True |
| 51843.0202 | 51844.3489 | -393.57853599893878 | True | 2.26518699e-12 | True | True |
| 103687.369 | 103688.698 | -393.57853423304960 | True | 2.26857383e-12 | True | True |
| 207376.067 | 207377.396 | -393.57853335010503 | True | 2.26902102e-12 | True | True |
| 414753.462 | 414754.791 | -393.57853290863221 | True | 2.26920821e-12 | True | True |
| 829508.253 | 829509.582 | -393.57853268789648 | True | 2.26901518e-12 | True | True |
| 1659017.84 | 1659019.16 | -393.57853257752845 | True | 2.26850603e-12 | True | True |

## H2O2 — PASS

Asymptote status: PLATEAU_CONFIRMED.

RHF basis: sto-3g. Coordinate: q = r(O-O) - r_e(O-O); both OH lengths, OOH angles and HOOH torsion fixed. s_eq=1.396248579 A.

- Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.

Separated-fragment Mulliken charge diagnostic (electron-charge units):

| Fragment | Charge [e] |
| --- | --- |
| O1 H3 | 5.28022071e-12 |
| O2 H4 | -5.28943556e-12 |

E_eq=-148.76499662127242 Eh; E_diss=-148.28701434160945 Eh; D_e^(RHF-path)=0.477982279663 Eh.

```text
D_e^(RHF-path)            = 0.477982279663 Hartree
n=0 energy above minimum = 0.003390054230 Hartree
n=0 relative to diss.    = -0.474592225433 Hartree BOUND
n=1 energy above minimum = 0.010170162691 Hartree
n=1 relative to diss.    = -0.467812116972 Hartree BOUND
selected n               = 1
```

n=1 lies below the RHF-path dissociation threshold

**Selected existing harmonic range: [1.306625699, 1.485871459] A.**

Final plateau assessment:

```json
{
  "passed": true,
  "spread_Eh": 5.522981041394814e-07,
  "pairwise_span_Eh": 5.522981041394814e-07,
  "max_neighbor_change_Eh": 3.1559929425384325e-07,
  "s_ratios": [
    2.0,
    2.0,
    2.0
  ],
  "candidate_q_A": 1676736.791279552,
  "confirmation_q_A": 3353474.978807683,
  "window_q_A": [
    419183.1506334536,
    838367.6975154864,
    1676736.791279552,
    3353474.978807683
  ],
  "window_energies_Eh": [
    -148.28701489390755,
    -148.28701457830826,
    -148.28701442050556,
    -148.28701434160945
  ],
  "extension_change_Eh": 7.889610742495279e-08,
  "reason": "Three-point plateau plus farther confirmation passed."
}
```

Last attempted q=3353474.98 A. Full scan follows; accepted means SCF, residual, internal stability and geometry checks passed.

| q [A] | s [A] | E_RHF [Eh] | SCF converged | Orbital residual | Internal stable | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.39624858 | -148.76499662127242 | True | 9.54461581e-13 | True | True |
| 0.1 | 1.49624858 | -148.75421207047788 | True | 1.68379676e-14 | True | True |
| 0.2 | 1.59624858 | -148.72816687171854 | True | 1.65110182e-14 | True | True |
| 0.3 | 1.69624858 | -148.69414658885165 | True | 1.32746948e-14 | True | True |
| 0.5 | 1.89624858 | -148.62013293191038 | True | 1.9665999e-14 | True | True |
| 0.75 | 2.14624858 | -148.53756939236723 | True | 1.44708652e-14 | True | True |
| 1 | 2.39624858 | -148.47441432329893 | True | 2.46576206e-14 | True | True |
| 1.5 | 2.89624858 | -148.40156486231774 | True | 1.63424415e-14 | True | True |
| 2 | 3.39624858 | -148.37143610809719 | True | 9.99005216e-15 | True | True |
| 3 | 4.39624858 | -148.34850201819194 | True | 8.68885098e-15 | True | True |
| 4 | 5.39624858 | -148.33671113980941 | True | 5.92480422e-15 | True | True |
| 5 | 6.39624858 | -148.32877601554577 | True | 1.08516593e-14 | True | True |
| 11.3962486 | 12.7924972 | -148.30774580272006 | True | 8.85985861e-14 | True | True |
| 24.1887457 | 25.5849943 | -148.29736182396991 | True | 2.42584834e-14 | True | True |
| 49.7737401 | 51.1699886 | -148.29218578828250 | True | 5.01802314e-12 | True | True |
| 100.943729 | 102.339977 | -148.28959974468287 | True | 3.76089231e-15 | True | True |
| 203.283706 | 204.679955 | -148.28830696865299 | True | 4.3011078e-15 | True | True |
| 407.96366 | 409.359909 | -148.28766061130332 | True | 1.54494887e-12 | True | True |
| 817.32357 | 818.719818 | -148.28733743645870 | True | 1.04804267e-12 | True | True |
| 1636.04339 | 1637.43964 | -148.28717584951497 | True | 2.5528923e-13 | True | True |
| 3273.48302 | 3274.87927 | -148.28709505610269 | True | 6.6548473e-14 | True | True |
| 6548.3623 | 6549.75855 | -148.28705465940399 | True | 6.88371519e-15 | True | True |
| 13098.1208 | 13099.5171 | -148.28703446105555 | True | 5.79289725e-15 | True | True |
| 26197.6379 | 26199.0342 | -148.28702436188212 | True | 4.40206883e-12 | True | True |
| 52396.6721 | 52398.0684 | -148.28701931229529 | True | 1.10107378e-12 | True | True |
| 104794.74 | 104796.137 | -148.28701678750187 | True | 2.74067963e-13 | True | True |
| 209590.877 | 209592.273 | -148.28701552510151 | True | 6.87888409e-14 | True | True |
| 419183.151 | 419184.547 | -148.28701489390755 | True | 1.7298692e-14 | True | True |
| 838367.698 | 838369.094 | -148.28701457830826 | True | 5.01317957e-15 | True | True |
| 1676736.79 | 1676738.19 | -148.28701442050556 | True | 6.15612153e-15 | True | True |
| 3353474.98 | 3353476.38 | -148.28701434160945 | True | 4.04043897e-15 | True | True |

## Provenance

```json
{
  "harmonic_json": "/Users/novaz/Desktop/qml2.0/new_rhf_harmonic_bond_ranges.json",
  "script_sha256": "dcad4f439eb36fe6d0617da39fe971411d53c87682f3ea388c4de6641fa35e03",
  "python_version": "3.13.2",
  "pyscf_version": "2.14.0",
  "nuclear_hessian_recomputed": false,
  "geometry_reoptimized": false,
  "energy_model": "conventional all-electron RHF without density fitting",
  "harmonic_json_sha256": "d698dde31c8252930dcd39aeea1721c5f4bde42efc81d3571a566a88cbfb5999",
  "harmonic_script_sha256": "e6687127ab5752b7a203cab27f177572774e85e7a71acec9f4b617e699b17bd9",
  "constants": {
    "Bohr_A": 0.52917721092,
    "Bohr_m": 5.2917721092e-11,
    "Hartree_J": 4.359744644911914e-18,
    "amu_kg": 1.660539040427164e-27,
    "hbar_J_s": 1.0545718001391127e-34,
    "c_m_per_s": 299792458,
    "source": "installed pyscf.data.nist"
  },
  "scan_calculation_script_sha256": "994ed070359dd3bed879a6c137a5a695179eec0c2b331b8dfb6b8c9fa4e10e7c",
  "unit_postprocessing": {
    "source_scan_json": "/Users/novaz/Desktop/qml2.0/json/rhf_bound_level_selection.json",
    "source_scan_json_sha256": "51ce2dbc0240bb2d27ec6d4f2af8b7eb077cba83de9914711ae291d3cc1e359a",
    "updated_utc": "2026-09-22T17:38:04.099497+00:00",
    "SCF_recomputed": false,
    "nuclear_hessian_recomputed": false,
    "scan_points_unchanged": true
  }
}
```

SCF convergence and electronic stability conventions were checked against the installed PySCF implementation and [the official SCF documentation](https://pyscf.org/user/scf.html#stability-analysis).
