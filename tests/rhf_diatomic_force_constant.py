"""RHF diatomic bond curvature at a supplied, unmodified geometry.

RHF partial_hess_elec explicitly returns (A,B,dR_A,dR_B); hess_elec
adds Cartesian xy blocks at [A,B]; kernel adds hess_nuc in that layout.
thermo.harmonic_analysis uses 'pqxy,p,q->pqxy' and then transpose(0,2,1,3).
Thus H[a,b,x,y] = d^2 E / (d R[a,x] d R[b,y]), in Eh/Bohr^2.
Runtime checks below also verify dimensions and the nuclear-Coulomb blocks
against an independently written analytic expression before projection.
"""

import math
import sys

import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf.data import nist
from pyscf.hessian import rhf as rhf_hessian
from pyscf.hessian import thermo


# User settings: supply an RHF equilibrium distance for this exact basis.
# 0.917 is retained exactly as requested; its stationarity is checked, not assumed.
ATOM1 = "H"
ATOM2 = "F"
R_EQ_ANGSTROM = 0.917
BASIS = "cc-pVDZ"
CHARGE = 0
SPIN = 0
FD_STEPS_ANGSTROM = (1e-2, 5e-3, 2e-3, 1e-3)

SCF_ENERGY_TOL = 1e-13
SCF_GRADIENT_TOL = 1e-10
SCF_MAX_CYCLE = 400
DIRECT_SCF_TOL = 1e-14
CPHF_TOL = 1e-12
CPHF_MAX_CYCLE = 200
STATIONARY_GRADIENT_TOL = 1e-5  # Eh/Bohr; diagnostic, not an optimizer target

# Numerical validation tolerances, not claims about the physical accuracy of RHF.
# Finite energy differences have O(h^2) truncation and O(delta_E/h^2) noise.
FD_ATOL = 1e-6  # Eh/Bohr^2
FD_RTOL = 2e-5
FREQUENCY_ATOL = 1e-6  # cm^-1
FREQUENCY_RTOL = 1e-8
HESSIAN_ATOL = 1e-8  # Eh/Bohr^2
HESSIAN_RTOL = 1e-8

# Use the installed PySCF constants consistently, including its Angstrom/Bohr
# conversion and harmonic_analysis's SI constants. These may be from an older
# CODATA release than scipy.constants. Mixing releases can produce a small
# frequency discrepancy even when the projection and masses are correct.
ANGSTROM_PER_BOHR = nist.BOHR
BOHR_M = nist.BOHR_SI
HARTREE_J = nist.HARTREE2J
AMU_KG = nist.ATOMIC_MASS
LIGHT_SPEED_M_S = nist.LIGHT_SPEED_SI
AU_CURVATURE_TO_N_M = HARTREE_J / BOHR_M**2  # J/m^2 = N/m, about 1556.8931


def make_molecule(distance_angstrom):
    """Place the reference at z=(0,R_eq); keep its midpoint fixed for FD."""
    if not np.isfinite(distance_angstrom) or distance_angstrom <= 0:
        raise ValueError("Every bond distance must be finite and positive.")
    shift = 0.5 * (distance_angstrom - R_EQ_ANGSTROM)
    mol = gto.M(
        atom=[(ATOM1, (0.0, 0.0, -shift)),
              (ATOM2, (0.0, 0.0, R_EQ_ANGSTROM + shift))],
        unit="Angstrom", basis=BASIS, symmetry=False,
        charge=CHARGE, spin=SPIN, verbose=0,
    )
    # scf.RHF may select ROHF for an open-shell molecule: reject that here.
    if mol.natm != 2 or mol.spin != 0 or mol.nelectron <= 0 or mol.nelectron % 2:
        raise ValueError("This script requires a closed-shell, two-atom RHF system.")
    if np.any(mol.atom_charges() <= 0):
        raise ValueError("Both atoms must be real nuclei, not ghost centers.")
    return mol


def run_rhf(distance_angstrom, initial_density=None):
    """A fresh conventional RHF calculation with strict convergence checks."""
    mol = make_molecule(distance_angstrom)
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_ENERGY_TOL
    mf.conv_tol_grad = SCF_GRADIENT_TOL
    mf.max_cycle = SCF_MAX_CYCLE
    mf.direct_scf_tol = DIRECT_SCF_TOL
    mf.conv_tol_cpscf = CPHF_TOL
    mf.chkfile = None

    def strict_convergence(state):
        # PySCF's optional final cycle otherwise relaxes the thresholds and
        # accepts energy OR orbital-gradient convergence. Require BOTH here.
        return (abs(state["e_tot"] - state["last_hf_e"]) < mf.conv_tol
                and state["norm_gorb"] < mf.conv_tol_grad)

    mf.check_convergence = strict_convergence
    dm0 = None if initial_density is None else initial_density.copy()
    energy = mf.kernel(dm0=dm0)
    if not mf.converged or not np.isfinite(energy):
        raise RuntimeError(
            f"RHF did not converge at R={distance_angstrom:.12f} Angstrom."
        )
    if getattr(mf, "with_df", None) is not None:
        raise RuntimeError("Density fitting is not permitted.")
    if not np.all(np.isin(mf.mo_occ, [0.0, 2.0])):
        raise RuntimeError("The converged solution is not closed-shell RHF.")
    # Verify the final orbital residual with a freshly built, unmodified Fock.
    density = mf.make_rdm1()
    fock = mf.get_hcore() + mf.get_veff(mol, density)
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
    if not np.isfinite(residual) or residual > 1e-9:
        raise RuntimeError(f"Final RHF orbital-gradient norm is too large: {residual:.3e}")
    return mf, residual


def verify_hessian_layout(hessian_driver, hess):
    """Check the installed API's actual nuclear blocks and tensor dimensions."""
    if not isinstance(hessian_driver, rhf_hessian.Hessian):
        raise RuntimeError("Expected the conventional PySCF RHF Hessian implementation.")
    if hess.shape != (2, 2, 3, 3) or not np.all(np.isfinite(hess)):
        raise RuntimeError(f"Invalid Hessian: expected finite (2,2,3,3), got {hess.shape}.")

    mol = hessian_driver.mol
    delta = mol.atom_coords(unit="Bohr")[1] - mol.atom_coords(unit="Bohr")[0]
    r_bohr = float(np.linalg.norm(delta))
    direction = delta / r_bohr
    z1, z2 = mol.atom_charges()
    # For E_nuc=Z1*Z2/r, the same-atom block is Z1*Z2*(3ee^T-I)/r^3,
    # while the cross-atom block has the opposite sign. r is explicitly Bohr.
    block = z1 * z2 * (3 * np.outer(direction, direction) - np.eye(3)) / r_bohr**3
    expected = np.empty((2, 2, 3, 3))
    expected[0, 0] = expected[1, 1] = block
    expected[0, 1] = expected[1, 0] = -block
    actual = np.asarray(hessian_driver.hess_nuc())
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError("The nuclear Hessian failed the abxy layout/atomic-unit check.")
    return direction


def finite_difference_validation(reference, k_au):
    """Independent, fully reconverged total-energy central differences."""
    steps = sorted(set(FD_STEPS_ANGSTROM), reverse=True)
    if len(steps) < 2 or any(not np.isfinite(h) or h <= 0 for h in steps):
        raise ValueError("Use at least two distinct, finite, positive FD steps.")
    if max(steps) >= R_EQ_ANGSTROM:
        raise ValueError("Every FD step must be smaller than the reference bond length.")
    e0 = float(reference.e_tot)
    density = reference.make_rdm1()
    rows = []
    print("\nFinite-difference validation (curvatures and absolute errors in Eh/Bohr^2)")
    print(f"{'h [Angstrom]':>14} {'k_FD':>17} {'k_Hessian':>17} "
          f"{'absolute diff':>15} {'relative diff':>15}", flush=True)
    for h_angstrom in steps:
        # The reference density is only an initial guess. Each energy uses a
        # separate molecule and a fully converged RHF calculation at that R.
        plus, _ = run_rhf(R_EQ_ANGSTROM + h_angstrom, density)
        minus, _ = run_rhf(R_EQ_ANGSTROM - h_angstrom, density)
        h_bohr = h_angstrom / ANGSTROM_PER_BOHR
        # fsum reduces summation roundoff; it cannot remove SCF energy errors.
        k_fd = math.fsum([float(plus.e_tot), -2.0 * e0,
                          float(minus.e_tot)]) / h_bohr**2
        absolute = abs(k_fd - k_au)
        relative = absolute / abs(k_au) if k_au != 0 else float("nan")
        rows.append((h_angstrom, k_fd, absolute, relative))
        print(f"{h_angstrom:14.6g} {k_fd:17.11f} {k_au:17.11f} "
              f"{absolute:15.6e} {relative:15.6e}", flush=True)

    # Optional O(h^2) Richardson cancellation using the two smallest steps;
    # the requested raw central differences remain the main reported table.
    h_large, k_large, _, _ = rows[-2]
    h_small, k_small, _, _ = rows[-1]
    ratio_squared = (h_large / h_small)**2
    extrapolated = (ratio_squared * k_small - k_large) / (ratio_squared - 1)
    tolerance = FD_ATOL + FD_RTOL * abs(k_au)
    passed = (all(row[2] <= tolerance for row in rows[-2:])
              and abs(extrapolated - k_au) <= tolerance)
    print(f"Richardson estimate [Eh/Bohr^2]: {extrapolated:.12f}")
    print(f"Richardson absolute difference: {abs(extrapolated - k_au):.6e}")
    print(f"Acceptance: two smallest steps and Richardson error <= {tolerance:.3e}")
    print("Smaller steps need not improve agreement: energy cancellation amplifies noise.")
    return passed


def format_frequency(value):
    """Keep an imaginary frequency imaginary; never take abs(k) silently."""
    value = complex(value)
    if abs(value.imag) < 1e-10:
        return f"{value.real:.10f}"
    if abs(value.real) < 1e-10:
        return f"{value.imag:.10f}i"
    return f"{value.real:.10f}{value.imag:+.10f}i"


def main():
    if SPIN != 0:
        raise ValueError("SPIN must be zero for this closed-shell RHF script.")
    print(f"PySCF version: {pyscf.__version__}")
    print("Evaluating the supplied geometry without optimization.", flush=True)
    mf, orbital_residual = run_rhf(R_EQ_ANGSTROM)
    mol = mf.mol
    gradient = np.asarray(mf.nuc_grad_method().kernel())
    if gradient.shape != (2, 3) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("Invalid RHF Cartesian gradient.")
    print("RHF Cartesian gradient [Eh/Bohr], rows=atoms, columns=x,y,z:")
    for symbol, row in zip((ATOM1, ATOM2), gradient):
        print(f"  {symbol:>3s} " + " ".join(f"{component:+.12e}" for component in row))
    max_gradient = float(np.max(np.abs(gradient)))
    if max_gradient > STATIONARY_GRADIENT_TOL:
        print("WARNING: The supplied geometry is not stationary within the stated tolerance.")
        print("The reported k is the local curvature at this unchanged R, not a certified minimum.")

    hessian_driver = mf.Hessian()
    hessian_driver.max_cycle = CPHF_MAX_CYCLE
    hess = np.asarray(hessian_driver.kernel())
    e = verify_hessian_layout(hessian_driver, hess)
    print("Hessian convention verified: H[a,b,x,y], shape (2,2,3,3), Eh/Bohr^2.")

    # a,b index atoms; x,y index Cartesian components. R and r use Bohr.
    # R1 -> R1-dr*e/2 and R2 -> R2+dr*e/2 changes R2-R1 by dr*e,
    # so the bond distance changes by exactly dr. There is no factor-of-four
    # error: b has not been Euclidean-normalized, and k is not a Cartesian
    # normal-mode eigenvalue. The path is linear, so d^2R/dr^2=0 and no
    # gradient term is needed, even if the supplied geometry is nonstationary.
    b = np.stack((-0.5 * e, 0.5 * e))
    k_au = float(np.einsum("ax,abxy,by->", b, hess, b))
    k_si = k_au * AU_CURVATURE_TO_N_M
    slope = float(np.einsum("ax,ax->", b, gradient))
    print(f"dE/dr [Eh/Bohr]: {slope:+.12e}")
    print(f"Final SCF orbital-gradient norm: {orbital_residual:.3e}")

    # Measure invariance errors without modifying/symmetrizing the Hessian.
    symmetry_error = float(np.max(np.abs(hess - hess.transpose(1, 0, 3, 2))))
    translation_error = max(float(np.max(np.abs(hess.sum(axis=0)))),
                            float(np.max(np.abs(hess.sum(axis=1)))))
    hessian_tolerance = HESSIAN_ATOL + HESSIAN_RTOL * float(np.max(np.abs(hess)))
    invariance_ok = max(symmetry_error, translation_error) <= hessian_tolerance
    print(f"Hessian exchange-symmetry residual [Eh/Bohr^2]: {symmetry_error:.3e}")
    print(f"Hessian translation residual [Eh/Bohr^2]: {translation_error:.3e}")

    # Match harmonic_analysis's default isotope-average atomic masses explicitly.
    # atom_mass_list() without isotope_avg=True uses a different mass convention.
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True), dtype=float)
    if masses.shape != (2,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise RuntimeError("Atomic masses must be finite and positive.")
    m1, m2 = masses
    mu_amu = m1 * m2 / (m1 + m2)
    mu_kg = mu_amu * AMU_KG
    # k_SI [N/m] / mu [kg] has units s^-2. sqrt gives angular frequency
    # in rad/s. Divide by 2*pi and c [m/s] for m^-1, then by 100 for cm^-1.
    # scimath.sqrt deliberately preserves imaginary frequencies for k < 0.
    omega = np.lib.scimath.sqrt(k_si / mu_kg)
    frequency_k = complex(omega / (2 * np.pi * LIGHT_SPEED_M_S) / 100)
    if k_au <= 0:
        print("WARNING: k <= 0: the geometry is not a positive-curvature harmonic minimum")
        print("along the bond stretch. A negative k is reported as an imaginary frequency.")

    analysis = thermo.harmonic_analysis(
        mol, hess, exclude_trans=True, exclude_rot=True,
        imaginary_freq=True, mass=masses,
    )
    frequencies = np.asarray(analysis["freq_wavenumber"]).reshape(-1)
    if frequencies.size != 1 or not np.all(np.isfinite(frequencies)):
        raise RuntimeError(f"Expected exactly one finite vibrational mode; got {frequencies}.")
    frequency_pyscf = complex(frequencies[0])
    mode_mass = float(np.asarray(analysis["reduced_mass"]).reshape(-1)[0])
    frequency_difference = abs(frequency_k - frequency_pyscf)
    frequency_ok = frequency_difference <= (
        FREQUENCY_ATOL + FREQUENCY_RTOL * abs(frequency_pyscf)
    )

    # PySCF's reported mode mass is 1/sum(u_ax^2), where the Cartesian mode
    # satisfies sum(m_a*u_ax^2)=1. For a diatomic COM-fixed stretch this is
    # m1*m2*(m1+m2)/(m1^2+m2^2), NOT the bond-coordinate mass m1*m2/(m1+m2).
    # For equal masses the mode mass is m, while the bond reduced mass is m/2.
    # Do not insert the mode mass into omega=sqrt(k/mu).
    expected_mode_mass = m1 * m2 * (m1 + m2) / (m1**2 + m2**2)
    mass_ok = bool(np.isclose(mode_mass, expected_mode_mass, rtol=1e-8, atol=1e-10))

    # Midpoint-fixed and COM-fixed paths differ only by a translation, so they
    # have the same potential curvature. The physical kinetic mass mu belongs
    # to the COM-fixed path; b^T M b would be the wrong mass for this purpose.
    b_com = np.stack((-m2 / (m1 + m2) * e, m1 / (m1 + m2) * e))
    k_com = float(np.einsum("ax,abxy,by->", b_com, hess, b_com))
    print(f"COM-vs-midpoint curvature difference [Eh/Bohr^2]: {abs(k_com - k_au):.3e}")
    print(f"Atomic masses [amu], isotope averages: {m1:.10f}, {m2:.10f}")
    print(f"PySCF mode reduced mass [amu]: {mode_mass:.10f}")
    print(f"Expected mode mass from its normalization [amu]: {expected_mode_mass:.10f}")
    print("The PySCF mode mass differs by convention from the bond reduced mass.")
    print(f"Constants: Hartree={HARTREE_J:.14e} J; Bohr={BOHR_M:.14e} m;")
    print(f"           amu={AMU_KG:.14e} kg; c={LIGHT_SPEED_M_S:.0f} m/s")
    print(f"1 Eh/Bohr^2 = {AU_CURVATURE_TO_N_M:.10f} N/m", flush=True)

    fd_ok = finite_difference_validation(mf, k_au)
    print("\n========== RHF harmonic bond stretch ==========")
    print(f"Molecule: {ATOM1}-{ATOM2} (charge={CHARGE}, spin={SPIN})")
    print(f"Basis: {BASIS}")
    print(f"R_eq [Angstrom]: {R_EQ_ANGSTROM:.12f}")
    print(f"E_RHF [Eh]: {mf.e_tot:.15f}")
    print(f"k [Eh/Bohr^2]: {k_au:.12f}")
    print(f"k [N/m]: {k_si:.10f}")
    print(f"reduced mass [amu] (bond coordinate): {mu_amu:.10f}")
    print(f"frequency from projected k [cm^-1]: {format_frequency(frequency_k)}")
    print(f"PySCF harmonic frequency [cm^-1]: {format_frequency(frequency_pyscf)}")
    print(f"difference between the two frequencies [cm^-1] (absolute): {frequency_difference:.6e}")
    print(f"PySCF mode reduced mass [amu]: {mode_mass:.10f}")
    print(f"Maximum Cartesian gradient [Eh/Bohr]: {max_gradient:.6e}")
    print(f"Positive bond curvature: {'YES' if k_au > 0 else 'NO'}")
    print(f"Stationary within gradient tolerance: {'YES' if max_gradient <= STATIONARY_GRADIENT_TOL else 'NO'}")
    print(f"Hessian-vs-finite-difference validation: {'PASS' if fd_ok else 'FAIL'}")
    print(f"Frequency cross-check: {'PASS' if frequency_ok else 'FAIL'}")
    print(f"PySCF mode-mass convention check: {'PASS' if mass_ok else 'FAIL'}")
    print(f"Hessian symmetry/translation checks: {'PASS' if invariance_ok else 'FAIL'}")
    if not fd_ok:
        print("DIAGNOSIS: Inspect the step-size trend: smooth O(h^2) drift suggests truncation;")
        print("growing/scattered errors at small h suggest energy noise or different SCF roots.")
        print("A persistent offset suggests Hessian response error; the layout and units were checked.")
        print("SCF energy tolerances do not rigorously bound absolute energy errors.")
    if not frequency_ok:
        print("DIAGNOSIS: Both frequency routes use identical atomic masses and SI constants.")
        print("Compare the translation residual and COM-vs-midpoint curvature difference above;")
        print("a nonzero difference can explain the mismatch without changing either result.")
        print("If these are small, inspect the installed harmonic_analysis mode projection.")
    if not mass_ok:
        print("DIAGNOSIS: PySCF's returned mode mass does not follow the audited diatomic")
        print("normalization; inspect norm_mode and the installed harmonic_analysis implementation.")
    if not invariance_ok:
        print("DIAGNOSIS: Hessian invariance residuals exceed tolerance. Inspect RHF/CPHF")
        print("convergence and basis conditioning; no corrective projection has been applied.")
    return 0 if fd_ok and frequency_ok and mass_ok and invariance_ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FloatingPointError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
