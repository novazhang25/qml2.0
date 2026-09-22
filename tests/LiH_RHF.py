import numpy as np
import basis_set_exchange as bse
from pyscf import gto, scf
from scipy.optimize import minimize_scalar

BASIS = "sto-3g"

# Load once from Basis Set Exchange and reuse throughout the optimization.
BASIS_DATA = {
    element: gto.basis.parse(
        bse.get_basis(BASIS, elements=[element], fmt="nwchem")
    )
    for element in ("Li", "H")
}

def rhf_energy(R, verbose=False):
    mol = gto.M(
        atom=f"""
        Li  0.0  0.0  0.0
        H   0.0  0.0  {R:.14f}
        """,
        basis=BASIS_DATA,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )

    mf = scf.RHF(mol)

    # Tight SCF convergence
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.diis_space = 12

    # Tight direct-SCF screening
    mf.direct_scf_tol = 1e-14

    E = mf.kernel()

    if not mf.converged:
        raise RuntimeError(f"SCF did not converge at R = {R:.12f} Angstrom")

    if verbose:
        print(f"R = {R:.12f} A   E_RHF = {E:.15f} Eh")

    return E


# ============================================================
# 1. Coarse scan to locate the minimum
# ============================================================

Rs = np.linspace(1.45, 1.75, 31)

energies = []

for R in Rs:
    E = rhf_energy(R)
    energies.append(E)
    print(f"{R:12.8f}  {E:20.14f}")

energies = np.array(energies)

imin = np.argmin(energies)

if imin == 0 or imin == len(Rs) - 1:
    raise RuntimeError("Minimum lies at edge of scan range.")

R_left = Rs[imin - 1]
R_mid = Rs[imin]
R_right = Rs[imin + 1]

print("\nBrent bracket:")
print(R_left, R_mid, R_right)


# ============================================================
# 2. High-precision Brent minimization
# ============================================================

result = minimize_scalar(
    rhf_energy,
    bracket=(R_left, R_mid, R_right),
    method="brent",
    options={
        "xtol": 1e-11,
        "maxiter": 200,
    },
)

R_eq = result.x
E_eq = result.fun

print("RHF equilibrium geometry")
print(f"Basis   = {BASIS}")
print(f"R_e     = {R_eq:.12f} Angstrom")
print(f"E(R_e)  = {E_eq:.15f} Eh")
print(f"Success = {result.success}")
print(f"N eval  = {result.nfev}")


# ============================================================
# 3. Verify energies around the minimum
# ============================================================


for dR in [-1e-4, -5e-5, -1e-5, 0.0, 1e-5, 5e-5, 1e-4]:
    R = R_eq + dR
    E = rhf_energy(R)
    print(
        f"R = {R:.10f} A   "
        f"dR = {dR:+.1e} A   "
        f"E = {E:.15f} Eh   "
        f"dE = {(E-E_eq)*1e9:+.6f} nEh"
    )
