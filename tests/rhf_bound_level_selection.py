#!/usr/bin/env python3
"""Raw RHF/path utilities plus a compatibility entry point for current Part 3.

The current n=0/n=1 decision lives in codes/bond_length_part3.py and uses the
unshifted saved equilibrium energy and saved harmonic frequency. Running this
legacy filename delegates to that postprocessor and never launches a new scan.
The RHF solver and plateau assessment below remain available only as independent
raw-energy/path utilities for the 30-point scan and archived data validation.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pyscf
from scipy.linalg import expm
from pyscf import scf, lib
from pyscf.soscf import newton_ah

import new_rhf_harmonic_bond_ranges as harmonic


SCF_GRADIENT_LIMIT = 1e-10
STABILITY_LIMIT = -1e-7


def conversion_from_constants(constants):
    """Input wavenumber -> Hartree: h*c*100/Eh, using saved harmonic constants."""
    values = [constants[k] for k in ("Hartree_J", "hbar_J_s", "c_m_per_s")]
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Invalid saved harmonic physical constants.")
    return 2 * math.pi * values[1] * values[2] * 100 / values[0]



def assess_plateau(points, tolerance_Eh=1e-6):
    """Three-point candidate PLUS a geometrically farther fourth confirmation.

    All four consecutive attempted points must be accepted, with each absolute
    separation at least 1.5 times its predecessor. Failure points cannot be
    silently filtered out to manufacture a plateau.
    """
    result = {"passed": False, "spread_Eh": None, "max_neighbor_change_Eh": None}
    if len(points) < 4:
        return {**result, "reason": "Fewer than four consecutive points."}
    window = points[-4:]
    if not all(p.get("accepted", False) for p in window):
        return {**result, "reason": "A final-window RHF or geometry check failed."}
    energies = np.array([p["energy_Eh"] for p in window], float)
    separations = np.array([p["s_A"] for p in window], float)
    displacements = np.array([p["q_A"] for p in window], float)
    if not np.isfinite(energies).all() or not np.isfinite(separations).all():
        return {**result, "reason": "Nonfinite tail energy or separation."}
    if np.any(separations <= 0) or np.any(np.diff(displacements) <= 0):
        return {**result, "reason": "Tail displacements must increase strictly."}
    ratios = separations[1:] / separations[:-1]
    if np.any(ratios < 1.5):
        return {**result, "reason": "Tail separation ratios are too small for a distinct extension.", "s_ratios": ratios}
    spread = float(np.max(np.abs(energies - energies[-1])))
    pairwise_span = float(np.ptp(energies))
    adjacent = float(np.max(np.abs(np.diff(energies))))
    passed = max(spread, pairwise_span, adjacent) < tolerance_Eh
    return {"passed": passed, "spread_Eh": spread, "pairwise_span_Eh": pairwise_span,
            "max_neighbor_change_Eh": adjacent, "s_ratios": ratios,
            "candidate_q_A": displacements[-2], "confirmation_q_A": displacements[-1],
            "window_q_A": displacements, "window_energies_Eh": energies,
            "extension_change_Eh": float(abs(energies[-1] - energies[-2])),
            "reason": "Three-point plateau plus farther confirmation passed." if passed else "Tail energies have not converged."}


def load_harmonic_input(path):
    document = json.loads(path.read_text())
    expected_hash = document["runtime"]["script_sha256"]
    if harmonic.sha256(harmonic.__file__) != expected_hash:
        raise harmonic.InputProblem("The current harmonic builder does not match the validated output's source hash.")
    stored = document["runtime"]["constants"]
    actual = {"Bohr_A": harmonic.BOHR_A, "Hartree_J": harmonic.EH_J,
              "hbar_J_s": harmonic.HBAR, "c_m_per_s": harmonic.C_MS}
    if any(actual[key] != stored[key] for key in actual):
        raise harmonic.InputProblem("Installed and saved harmonic physical constants differ.")
    results = {r["summary"]["molecule"]: r for r in document["results"]}
    if len(results) != len(document["results"]):
        raise harmonic.InputProblem("Duplicate molecules in the harmonic output.")
    return document, results


def prepare_path(record):
    if not record or record.get("summary", {}).get("validation_status") != "PASS":
        raise harmonic.InputProblem("A validated PASS harmonic result is required.")
    saved = record["summary"]
    for key in ("s_eq_A", "harmonic_frequency_cm1", "Delta_q_n0_A", "Delta_q_n1_A",
                "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A"):
        if not math.isfinite(saved.get(key, float("nan"))) or saved[key] <= 0:
            raise harmonic.InputProblem(f"Missing or invalid harmonic quantity: {key}")
    symbols, coords = record["symbols"], np.array(record["equilibrium_cartesian_A"])
    basis = saved["RHF_basis"]
    masses = np.array(record["isotope_average_masses_amu"])
    mol = harmonic.make_molecule(symbols, coords, basis)
    if not np.array_equal(masses, mol.atom_mass_list(isotope_avg=True)):
        raise harmonic.InputProblem("Atomic masses differ from the validated harmonic result.")
    path = harmonic.StretchPath(saved["molecule"], symbols, coords, masses)
    tangent, validation = path.validate()
    if validation["failures"] or not np.allclose(tangent, record["tangent_dR_dq"], atol=2e-8, rtol=0):
        raise harmonic.InputProblem("Reused path does not reproduce its validated harmonic tangent.")
    if abs(path.s_eq - saved["s_eq_A"]) > 1e-10 or path.definition != saved["coordinate_definition"]:
        raise harmonic.InputProblem("Reused harmonic coordinate definition or equilibrium length differs.")
    return path, symbols, basis


def validate_scan_geometry(path, q, coords):
    original, current = path.internals(path.ref), path.internals(coords)
    length_tol = max(5e-10, 256 * np.finfo(float).eps * max(1, float(np.max(np.abs(coords)))))
    fixed_lengths = [v["value"] for v in original.values() if v["kind"] == "fixed_bond"]
    angle_tol = max(2e-7, math.degrees(8 * length_tol / min(fixed_lengths or [1.])))
    errors, passed = {}, bool(np.isfinite(coords).all())
    for label, item in original.items():
        expected = item["value"] + q if item["kind"] == "stretch" else item["value"]
        difference = (harmonic.angular_difference(current[label]["value"], expected)
                      if item["kind"] == "fixed_dihedral" else current[label]["value"] - expected)
        errors[label] = abs(difference)
        passed = passed and abs(difference) <= (length_tol if item["unit"] == "A" else angle_tol)
    return {"passed": passed, "absolute_internal_errors": errors,
            "length_tolerance_A": length_tol, "angle_tolerance_deg": angle_tol}


def orbital_stability(mf):
    """Dense internal RHF orbital stability; this is NOT a nuclear Hessian."""
    gradient, operator, _ = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    dimension = gradient.size
    matrix = np.column_stack([2 * operator(v) for v in np.eye(dimension)])
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return gradient * 2, matrix, eigenvalues, eigenvectors


def configured_rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-12, 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check = 400, 12, False
    mf.direct_scf = False  # Fully rebuild Fock matrices to avoid accumulated residuals.
    mf.chkfile = None
    return mf


def polish_orbitals(mf):
    """Resolve residuals with dense Newton steps on the SAME RHF energy functional.

    g and H below use the same factor of two as the RHF internal stability
    convention. Rotations preserve orthonormality and occupations exactly.
    Tiny null orbital rotations are omitted from the pseudoinverse; the full
    gradient is nevertheless checked, so an unresolved direction cannot pass.
    """
    history, previous_energy = [], None
    for iteration in range(30):
        energy = float(mf.energy_tot(dm=mf.make_rdm1()))
        gradient, matrix, eigenvalues, vectors = orbital_stability(mf)
        residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
        delta = None if previous_energy is None else abs(energy - previous_energy)
        history.append({"iteration": iteration, "energy_Eh": energy, "orbital_gradient_norm_Eh": residual,
                        "energy_change_Eh": delta, "lowest_internal_eigenvalue_Eh": float(eigenvalues[0])})
        if residual <= 1e-11 and delta is not None and delta <= 1e-13:
            mf.e_tot, mf.converged = energy, True
            return True, history
        previous_energy = energy
        if residual <= 1e-11:
            # A repeated full Fock rebuild confirms stationarity and energy convergence.
            continue
        keep = abs(eigenvalues) > 2e-10
        step = -vectors[:, keep] @ ((vectors[:, keep].T @ gradient) / eigenvalues[keep])
        step *= min(1., .1 / max(float(np.linalg.norm(step)), 1e-30))
        if not np.isfinite(step).all():
            break
        mf.mo_coeff = mf.mo_coeff @ expm(scf.hf.unpack_uniq_var(step, mf.mo_occ))
    mf.e_tot = float(mf.energy_tot(dm=mf.make_rdm1()))
    mf.converged = False
    return False, history


def newton_restart(mf, orbitals=None):
    solver = mf.newton()
    solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-12, 400
    solver.ah_conv_tol, solver.ah_lindep = 1e-14, 1e-20
    solver.kernel(mo_coeff=mf.mo_coeff if orbitals is None else orbitals, mo_occ=mf.mo_occ)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = solver.mo_coeff, solver.mo_occ, solver.mo_energy
    mf.e_tot, mf.converged = solver.e_tot, solver.converged
    return bool(solver.converged)


def perturbed_density(mf):
    count = int(np.count_nonzero(mf.mo_occ == 2) * np.count_nonzero(mf.mo_occ == 0))
    direction = np.sin(np.arange(1, count + 1))
    direction *= .15 / np.linalg.norm(direction)
    orbitals = mf.mo_coeff @ expm(scf.hf.unpack_uniq_var(direction, mf.mo_occ))
    return mf.make_rdm1(orbitals, mf.mo_occ)


def solve_point(symbols, coords, basis, density=None, independent=False, previous=None):
    """Strict, internally stable conventional RHF, with recorded solver attempts."""
    attempts = []
    guesses = [("fresh_minao" if density is None else "supplied_density", density)]
    if previous is not None:
        guesses.insert(0, ("orthonormalized_orbital_continuation", None))
    if density is not None and not independent:
        guesses.append(("fresh_minao_fallback", None))
    last_state = {"accepted": False, "scf_converged": False, "energy_Eh": None,
                  "orbital_gradient_norm_Eh": None, "internal_stable": False,
                  "restricted_stability_corrections": 0, "solver_attempts": attempts}
    for label, guess in guesses:
        attempt = {"initial_guess": label, "stability_repairs": []}
        attempts.append(attempt)
        try:
            mf = configured_rhf(harmonic.make_molecule(symbols, coords, basis))
            if label == "orthonormalized_orbital_continuation":
                # Transport paired orbitals in the atom-centered AO representation,
                # reorthonormalizing at the new geometry before any diagonalization
                # can swap occupations across nearly degenerate separated fragments.
                metric = previous.mo_coeff.T @ mf.get_ovlp() @ previous.mo_coeff
                metric_values, metric_vectors = np.linalg.eigh(metric)
                if np.min(metric_values) <= 1e-10:
                    raise ValueError("Transported orbital overlap is singular.")
                inverse_sqrt = (metric_vectors / np.sqrt(metric_values)) @ metric_vectors.T
                mf.mo_coeff = previous.mo_coeff @ inverse_sqrt
                mf.mo_occ, mf.mo_energy = previous.mo_occ.copy(), previous.mo_energy.copy()
                attempt["initial_solver"] = "transported paired orbitals; full-Fock Newton refinement"
            else:
                mf.kernel(dm0=guess)
                attempt["initial_pyscf_converged"] = bool(mf.converged)
            good, history = polish_orbitals(mf)
            attempt["initial_polish"] = history
            if not good:
                attempt["newton_restart_converged"] = newton_restart(mf)
                good, history = polish_orbitals(mf)
                attempt["restart_polish"] = history
            stable, eigenvalues = False, np.array([float("nan")])
            for repair in range(9):
                if not good:
                    break
                _, _, eigenvalues, vectors = orbital_stability(mf)
                stable = bool(eigenvalues[0] >= STABILITY_LIMIT)
                if stable or repair == 8:
                    break
                energy_before = float(mf.e_tot)
                alternatives = []
                for sign in (-1., 1.):
                    orbitals = mf.mo_coeff @ expm(scf.hf.unpack_uniq_var(sign * .35 * vectors[:, 0], mf.mo_occ))
                    trial_energy = float(mf.energy_tot(dm=mf.make_rdm1(orbitals, mf.mo_occ)))
                    alternatives.append((trial_energy, orbitals))
                trial_energy, orbitals = min(alternatives, key=lambda entry: entry[0])
                newton_converged = newton_restart(mf, orbitals)
                good, history = polish_orbitals(mf)
                attempt["stability_repairs"].append({
                    "energy_before_Eh": energy_before, "lowest_eigenvalue_before_Eh": float(eigenvalues[0]),
                    "initial_rotation_energy_Eh": trial_energy, "energy_after_Eh": float(mf.e_tot),
                    "newton_converged": newton_converged, "polish": history,
                })
            residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
            if not good:
                _, _, eigenvalues, _ = orbital_stability(mf)
                stable = bool(eigenvalues[0] >= STABILITY_LIMIT)
            electron_trace = float(np.einsum("ij,ji->", mf.make_rdm1(), mf.get_ovlp()))
            orthogonality = float(np.max(np.abs(mf.mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff - np.eye(mf.mo_coeff.shape[1]))))
            accepted = bool(good and residual <= SCF_GRADIENT_LIMIT and stable and np.isfinite(mf.e_tot)
                            and abs(electron_trace - mf.mol.nelectron) < 1e-8 and orthogonality < 1e-9
                            and np.all(np.isin(mf.mo_occ, [0, 2])) and getattr(mf, "with_df", None) is None)
            state = {"accepted": accepted, "scf_converged": bool(good), "energy_Eh": float(mf.e_tot),
                     "orbital_gradient_norm_Eh": residual, "internal_stable": stable,
                     "lowest_internal_stability_eigenvalue_Eh": float(eigenvalues[0]),
                     "restricted_stability_corrections": len(attempt["stability_repairs"]),
                     "electron_number": electron_trace, "orbital_orthogonality_max_error": orthogonality,
                     "solver_attempts": attempts}
            attempt["final_energy_Eh"], attempt["accepted"] = float(mf.e_tot), accepted
            last_state = state
            if accepted:
                _, charges = mf.mulliken_pop(verbose=0)
                state["mulliken_atomic_charges"] = charges
                return mf, state
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
    last_state["failure"] = "RHF failed strict energy/gradient, internal-stability or orbital consistency checks."
    return None, last_state



def main(argv=None):
    """Delegate this legacy command to the current zero-energy Part 3 workflow."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
    from bond_length_part3 import main as current_part3_main
    return current_part3_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
