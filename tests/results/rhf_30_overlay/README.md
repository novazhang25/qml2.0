# Plot 1 with 30 RHF points per molecule

The overview repeats the local harmonic panel in the 3x3 style of tests/plot.py. Each molecule also has an individual local-panel PNG and SVG.

Orange markers show all 30 accepted RHF energies, including both bond-range endpoints. The connecting lines guide the eye; no fitting or extra points are used. The saved optimized equilibrium (r_e, E_re) is included in the orange RHF curve in coordinate order, using the same circular marker and one shared legend entry, without a separate equilibrium text box. The curve has 30 scan points plus this saved equilibrium; no interpolation or new RHF calculation supplies the equilibrium. Blue dashed curves are the unchanged harmonic approximation. Green shading is the saved selected range. Purple/green horizontal segments are the existing n=0/n=1 total energies and turning points.

The y axis is absolute E in Hartree: the RHF curve uses E_RHF; the harmonic curve uses E_re + k*q^2/2; the two levels use E_total_n0 = E_re + E_n0 and E_total_n1 = E_re + E_n1. No equilibrium or sampled-minimum energy is subtracted. Axis tick offsets are disabled. The even 30-point scan grid is unchanged; the added equilibrium curve point uses saved Part 1/2 data.

The x axis is the saved absolute bond coordinate in Angstrom; H2O2 uses O-O separation. All RHF data are included in the y limits, including the compression side. No SCF, optimization, Hessian or new energy calculation was run by this plotting command.

Part 3 selects the highest n in {0,1} satisfying E_total_n = E_re + (n+0.5)*hbar_omega < 0 Ha. E_re remains the saved absolute RHF energy; the physical validity of zero as an allowed-state reference is unvalidated.

Scan input: /Users/novaz/Desktop/qml2.0/results/rhf_30_point_scans/rhf_scan_30.json
Scan SHA256: 261a03ce80c9eac0c3166579483bd4138148faf2349138b605973f9077a4c2c5
Part 3 input: /Users/novaz/Desktop/qml2.0/results/bond_length_part3/vibrational_levels.json
Part 3 SHA256: dbb2b85f3b594de3879d6aff03e5b94ccc6eddf4414f4b3e40691efc0b85f4bc

plotted_points.csv contains the exact plotted coordinates and absolute E_RHF_Ha values. It retains the 30 scan rows per molecule; the added equilibrium curve point comes from r_e_A and E_re_Ha in the Part 3 input. E_RHF_minus_E_re_Ha is retained only as an auxiliary diagnostic; it is not plotted.

- [overview.png](overview.png)
- [overview.svg](overview.svg)
- [LiH.png](LiH.png)
- [LiH.svg](LiH.svg)
- [BeH2.png](BeH2.png)
- [BeH2.svg](BeH2.svg)
- [H2O.png](H2O.png)
- [H2O.svg](H2O.svg)
- [NH3.png](NH3.png)
- [NH3.svg](NH3.svg)
- [N2.png](N2.png)
- [N2.svg](N2.svg)
- [CO.png](CO.png)
- [CO.svg](CO.svg)
- [HF.png](HF.png)
- [HF.svg](HF.svg)
- [H2S.png](H2S.png)
- [H2S.svg](H2S.svg)
- [H2O2.png](H2O2.png)
- [H2O2.svg](H2O2.svg)
