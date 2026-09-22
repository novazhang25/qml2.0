#!/usr/bin/env python3
"""Select an existing n=0/n=1 harmonic range using a NEW constrained RHF path scan.

Uses only new_rhf_harmonic_bond_ranges.py and its validated JSON output. No
nuclear Hessian, geometry optimization, UHF, Morse model or fragment energy is
calculated. Every dissociation threshold must be an actually sampled RHF energy.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyscf
from scipy.linalg import expm
from pyscf import scf, lib
from pyscf.soscf import newton_ah

import new_rhf_harmonic_bond_ranges as harmonic


INITIAL_GRID_A = (0., .1, .2, .3, .5, .75, 1., 1.5, 2., 3., 4., 5.)
SCF_GRADIENT_LIMIT = 1e-10
STABILITY_LIMIT = -1e-7
DEFAULT_THRESHOLD_WARNING_EH = 4.5563352580524656e-5
GENERAL_RHF_WARNING = (
    "D_e^(RHF-path) is a constrained RHF-path well depth, not a physical or experimental "
    "dissociation energy. Restricted closed-shell RHF cannot generally describe the separated "
    "open-shell fragments correctly. No UHF or correlated asymptote is used."
)
FIELDS = [
    "molecule", "basis", "coordinate", "s_eq_A", "E_eq_Eh", "E_diss_Eh", "D_e_Eh",
    "harmonic_quantum_Eh", "half_hbar_omega_Eh", "three_halves_hbar_omega_Eh",
    "E0_rel_Eh", "E1_rel_Eh", "n0_bound", "n1_bound", "selected_n", "selected_range_min_A",
    "selected_range_max_A", "asymptote_status", "RHF_dissociation_warning", "validation_status",
    "selected_reason", "near_threshold_levels", "last_q_A", "accepted_scan_points", "failed_scan_points",
    "bound_level_test_status", "physical_dissociation_status",
]


def conversion_from_constants(constants):
    """Input wavenumber -> Hartree: h*c*100/Eh, using saved harmonic constants."""
    values = [constants[k] for k in ("Hartree_J", "hbar_J_s", "c_m_per_s")]
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Invalid saved harmonic physical constants.")
    return 2 * math.pi * values[1] * values[2] * 100 / values[0]


def select_level(depth_Eh, quantum_Eh, harmonic_summary, warning_Eh=DEFAULT_THRESHOLD_WARNING_EH):
    if not all(math.isfinite(x) and x > 0 for x in (depth_Eh, quantum_Eh, warning_Eh)):
        raise ValueError("A positive finite depth, harmonic energy quantum and warning window in Hartree are required.")
    e0, e1 = -depth_Eh + .5 * quantum_Eh, -depth_Eh + 1.5 * quantum_Eh
    if e1 < 0:
        selected = 1
        reason = "n=1 lies below the RHF-path dissociation threshold"
    elif e0 < 0:
        selected = 0
        reason = "n=0 is bound but n=1 is above the RHF-path dissociation threshold"
        if e1 == 0:
            reason = "n=0 is bound but n=1 is at the RHF-path dissociation threshold"
    else:
        selected = None
        reason = "even n=0 lies at or above the RHF-path dissociation threshold"
    near = [n for n, energy in enumerate((e0, e1)) if abs(energy) < warning_Eh]
    status = "NO_BOUND_LEVEL_IN_HARMONIC_TEST" if selected is None else "PASS"
    if near:
        status = "NEAR_THRESHOLD"
    return {
        "harmonic_quantum_Eh": quantum_Eh,
        "half_hbar_omega_Eh": .5 * quantum_Eh, "three_halves_hbar_omega_Eh": 1.5 * quantum_Eh,
        "E0_rel_Eh": e0, "E1_rel_Eh": e1, "n0_bound": e0 < 0, "n1_bound": e1 < 0,
        "selected_n": selected, "selected_reason": reason, "near_threshold_levels": near,
        "bound_level_test_status": "NO_BOUND_LEVEL_IN_HARMONIC_TEST" if selected is None else "BOUND_LEVEL_SELECTED",
        "selected_range_min_A": None if selected is None else harmonic_summary[f"n{selected}_min_A"],
        "selected_range_max_A": None if selected is None else harmonic_summary[f"n{selected}_max_A"],
        "validation_status": status,
    }


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


def harmonic_summary_in_hartree(summary, wavenumber_to_Eh):
    """Convert energy-equivalent source frequencies; never relabel a frequency as energy."""
    result = copy.deepcopy(summary)
    for source_key, target_key in (("harmonic_frequency_cm1", "harmonic_quantum_Eh"),
                                   ("dominant_mode_frequency_cm1", "dominant_mode_quantum_Eh")):
        if source_key in result:
            result[target_key] = result.pop(source_key) * wavenumber_to_Eh
    return result


def scan_one(record, settings, wavenumber_to_Eh, progress=None):
    name = record["summary"]["molecule"] if record else settings.get("molecule", "UNKNOWN")
    summary = {"molecule": name, "selected_n": None, "selected_range_min_A": None,
               "selected_range_max_A": None, "asymptote_status": "NOT_EVALUATED",
               "validation_status": "NEEDS_INPUT", "RHF_dissociation_warning": GENERAL_RHF_WARNING,
               "physical_dissociation_status": "NOT_PHYSICALLY_VALIDATED_CONSTRAINED_RHF_MODEL"}
    result = {"summary": summary, "scan": [], "warnings": [], "failure_reasons": []}
    start = time.monotonic()
    try:
        path, symbols, basis = prepare_path(record)
        saved = record["summary"]
        quantum_Eh = saved["harmonic_frequency_cm1"] * wavenumber_to_Eh
        result["harmonic_input"] = {"summary": harmonic_summary_in_hartree(saved, wavenumber_to_Eh), "equilibrium_cartesian_A": path.ref,
                                    "symbols": symbols, "isotope_average_masses_amu": path.masses,
                                    "input_provenance": record["input"]}
        summary.update(basis=basis, coordinate=path.definition, s_eq_A=saved["s_eq_A"],
                       harmonic_quantum_Eh=quantum_Eh,
                       half_hbar_omega_Eh=.5 * quantum_Eh,
                       three_halves_hbar_omega_Eh=1.5 * quantum_Eh)
        last_density, accepted_plateau, last_mf = None, None, None
        q_values = list(q for q in INITIAL_GRID_A if q <= settings["max_q_A"])
        # Subsequent spacing doubles the absolute coordinate s, not only q.
        while q_values[-1] < settings["max_q_A"]:
            q_values.append(min(settings["max_q_A"], 2 * (path.s_eq + q_values[-1]) - path.s_eq))
        consecutive_failures = 0
        for q in q_values:
            point = {"q_A": q, "s_A": saved["s_eq_A"] + q, "accepted": False,
                     "energy_Eh": None, "scf_converged": False, "orbital_gradient_norm_Eh": None}
            result["scan"].append(point)
            coords = path(q)
            point["cartesian_A"] = coords
            point["geometry_validation"] = validate_scan_geometry(path, q, coords)
            if not point["geometry_validation"]["passed"]:
                point["failure"] = "Reused path lost internal-coordinate accuracy at large separation."
                result["failure_reasons"].append(point["failure"])
                summary["validation_status"] = "FAIL_PATH_AT_LARGE_DISTANCE"
                break
            mf, solver = solve_point(symbols, coords, basis, last_density, previous=last_mf)
            point.update(solver)
            if mf is None:
                consecutive_failures += 1
                if q == 0 or consecutive_failures >= 3:
                    result["failure_reasons"].append("Three consecutive RHF points failed, or equilibrium RHF failed; no reliable outward continuation.")
                    break
                if progress:
                    progress(result)
                continue
            consecutive_failures = 0
            last_mf, last_density = mf, mf.make_rdm1()
            if q == 0:
                summary["E_eq_Eh"] = point["energy_Eh"]
                eq_difference = abs(point["energy_Eh"] - record["stationarity"]["E_RHF_Eh"])
                result["equilibrium_energy_reproduction_error_Eh"] = eq_difference
                if eq_difference > 1e-9:
                    result["failure_reasons"].append("Equilibrium RHF energy does not reproduce the validated harmonic calculation.")
                    summary["validation_status"] = "FAIL_EQUILIBRIUM_REPRODUCTION"
                    break
            plateau = assess_plateau(result["scan"], settings["plateau_tolerance_Eh"])
            result["last_plateau_test"] = plateau
            if plateau["passed"]:
                # Independent fresh guess and an orbital-perturbed guess test the
                # asymptotic solution for initial-density/branch dependence.
                checks = []
                for label, guess in (("fresh_minao", None), ("deterministic_orbital_perturbation", perturbed_density(mf))):
                    other, diag = solve_point(symbols, coords, basis, guess, independent=True)
                    check = {"guess": label, **diag}
                    if other is not None:
                        check["energy_difference_from_continuation_Eh"] = float(other.e_tot - mf.e_tot)
                    checks.append(check)
                result["asymptote_independent_checks"] = checks
                if not all(c["accepted"] for c in checks):
                    summary["asymptote_status"] = "RHF_ASYMPTOTE_BRANCH_UNVERIFIED"
                    result["failure_reasons"].append("An independent RHF asymptote check did not converge to an internally stable state.")
                    break
                if any(abs(c["energy_difference_from_continuation_Eh"]) >= settings["plateau_tolerance_Eh"] for c in checks):
                    summary["asymptote_status"] = "RHF_ASYMPTOTE_BRANCH_AMBIGUITY"
                    result["failure_reasons"].append("Different internally stable restricted solutions give incompatible asymptotic energies.")
                    break
                accepted_plateau = plateau
                break
            if progress:
                progress(result)

        accepted = [p for p in result["scan"] if p["accepted"]]
        summary.update(last_q_A=result["scan"][-1]["q_A"], accepted_scan_points=len(accepted),
                       failed_scan_points=len(result["scan"]) - len(accepted))
        corrections = sum(p.get("restricted_stability_corrections", 0) for p in result["scan"])
        if corrections:
            result["warnings"].append(f"{corrections} restricted orbital-stability corrections were required; the RHF solution was allowed to break spatial symmetry while retaining paired orbitals.")
        if summary["failed_scan_points"]:
            result["warnings"].append("Failed SCF attempts/points are retained; only consecutive accepted tail points enter the plateau test.")
        if summary["last_q_A"] > 100:
            result["warnings"].append("Very large separations are a numerical asymptote test for the constrained RHF model, not a physically meaningful bond geometry.")
        if accepted:
            groups = ([list(pair) for pair in path.fixed_bonds] if name == "H2O2"
                      else [[i] for i in range(len(symbols))])
            atomic_charges = accepted[-1]["mulliken_atomic_charges"]
            fragments = [{"atoms": [f"{symbols[i]}{i+1}" for i in group],
                          "mulliken_charge_e": float(sum(atomic_charges[i] for i in group))} for group in groups]
            result["asymptotic_fragment_population_diagnostic"] = fragments
            if any(abs(f["mulliken_charge_e"] - round(f["mulliken_charge_e"])) > 1e-3 for f in fragments):
                result["warnings"].append("Fractional separated-fragment charges persist in the RHF population diagnostic; this is an additional dissociation-model artifact, not a physically validated fragment limit.")
        if accepted_plateau is None:
            if summary["asymptote_status"] == "NOT_EVALUATED":
                summary["asymptote_status"] = "DISSOCIATION_ASYMPTOTE_NOT_REACHED"
            if summary["validation_status"] == "NEEDS_INPUT":
                summary["validation_status"] = summary["asymptote_status"]
            summary["selected_reason"] = "No validated, unambiguous RHF-path plateau; no level or range is selected."
            return result

        summary["asymptote_status"] = "PLATEAU_CONFIRMED"
        result["accepted_plateau"] = accepted_plateau
        summary["E_diss_Eh"] = accepted[-1]["energy_Eh"]
        depth = summary["E_diss_Eh"] - summary["E_eq_Eh"]
        summary["D_e_Eh"] = depth
        if not math.isfinite(depth) or depth <= 0:
            summary["validation_status"] = "NONPOSITIVE_RHF_PATH_WELL_DEPTH"
            summary["selected_reason"] = "The accepted RHF threshold is not above the equilibrium energy."
            return result
        selected = select_level(depth, quantum_Eh, saved, settings["threshold_warning_Eh"])
        summary.update(selected)
        result["relative_level_energies_Eh"] = {f"E{n}_rel_Eh": summary[f"E{n}_rel_Eh"] for n in (0, 1)}
        result["plateau_energy_spread_Eh"] = accepted_plateau["pairwise_span_Eh"]
        # Retain the exact same range floats from the existing harmonic artifact.
        result["selection_formula"] = "E_n_rel_Eh = -D_e_Eh + (n + 0.5) * harmonic_quantum_Eh"
    except harmonic.InputProblem as exc:
        result["failure_reasons"].append(str(exc))
        summary["validation_status"] = "NEEDS_INPUT"
    except Exception as exc:
        result["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        summary["validation_status"] = "ERROR"
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        if result["warnings"]:
            summary["RHF_dissociation_warning"] = GENERAL_RHF_WARNING + " " + " ".join(result["warnings"])
    return result


def print_summary(summary):
    print(f"{summary['molecule']}: {summary['validation_status']}", flush=True)
    if "D_e_Eh" in summary:
        print(f"  D_e^(RHF-path)          = {summary['D_e_Eh']:.12f} Hartree", flush=True)
        for n, key in ((0, "half_hbar_omega_Eh"), (1, "three_halves_hbar_omega_Eh")):
            print(f"  n={n} energy above minimum = {summary[key]:.12f} Hartree", flush=True)
            if f"E{n}_rel_Eh" in summary:
                bound = "BOUND" if summary[f"n{n}_bound"] else "UNBOUND"
                print(f"  n={n} relative to diss.    = {summary[f'E{n}_rel_Eh']:+.12f} Hartree {bound}", flush=True)
        print(f"  selected n             = {summary.get('selected_n')}", flush=True)


def build_report(document):
    lines = [
        "# RHF bound-level selection", "",
        "The harmonic level energy measured upward from the potential minimum is", "",
        "    epsilon_n = hbar * omega * (n + 1/2),", "",
        "which is always positive. To test whether the level is bound, set the RHF-path dissociation threshold to zero. "
        "Since the equilibrium minimum is then at -D_e,", "",
        "    E_n_rel = -D_e + epsilon_n.", "",
        "Therefore E_n_rel < 0 means the level lies below dissociation.", "",
        "All energies and energy thresholds in this report use Hartree (Eh). The harmonic quantum is hbar*omega "
        "expressed as an energy. Bond lengths and coordinate ranges retain Angstrom units.", "",
        "## Numerical results", "",
        GENERAL_RHF_WARNING, "",
        "The comparison is restricted to n=0 and n=1. Existing harmonic turning-point ranges are reused exactly. "
        "The plateau threshold is an actually sampled energy; no fit or extrapolated asymptote is substituted. "
        "A missing, unconverged or branch-ambiguous plateau produces no selection. PASS certifies only the numerical "
        "test within this constrained RHF model, not physically correct dissociation or actual bound vibrational levels.", "",
    ]
    headers = ["Molecule", "D_e^(RHF-path) [Eh]", "epsilon_0 [Eh]", "epsilon_1 [Eh]",
               "E0_rel [Eh]", "E1_rel [Eh]", "Selected n", "Range min [A]", "Range max [A]", "Status"]
    keys = ["molecule", "D_e_Eh", "half_hbar_omega_Eh", "three_halves_hbar_omega_Eh", "E0_rel_Eh", "E1_rel_Eh",
            "selected_n", "selected_range_min_A", "selected_range_max_A", "validation_status"]
    lines += harmonic.table(headers, [[r["summary"].get(key) for key in keys] for r in document["results"]])
    settings = document["settings"]
    lines += ["", "## Method and acceptance criteria", "",
        "The only structural and harmonic input is the validated current harmonic JSON. Its builder hash, constants, "
        "equilibrium coordinate and tangent are checked before scanning. The nuclear Hessian and turning points are not recalculated. "
        "All nine molecules retain their RHF basis and existing mass-weighted Eckart/Kabsch path. "
        "For H2O2, s is O-O separation with both O-H lengths, O-O-H angles and H-O-O-H dihedral fixed. "
        "Other polyatomic s values are their common participating bond length.", "",
        "Conventional all-electron RHF uses energy tolerance 1e-13 Eh, orbital-gradient acceptance <=1e-10, "
        "integral screening 1e-14 and at least 400 permitted SCF cycles. The numerical solver targets a tighter gradient "
        "and rebuilds the full Fock matrix. Every accepted point has a directly recomputed energy and orbital residual. "
        "All occupations remain 0 or 2. Neighboring paired orbitals are reorthonormalized in the new AO overlap metric "
        "and refined before diagonalization can reset nearly degenerate occupations. A second-order solver and restricted orbital rotations may improve convergence; "
        "they do not change the RHF energy model.", "",
        "An internal RHF orbital-stability check rejects stationary electronic saddle points. Any correction rotates "
        "paired spatial orbitals within RHF and is recorded. This electronic stability matrix is distinct from the "
        "nuclear Hessian used in harmonic analysis. The check does not establish physical open-shell dissociation accuracy. "
        "At a candidate asymptote, an independent fresh initial density and a deterministically perturbed restricted "
        "orbital guess must reproduce the continuation energy within the plateau tolerance; otherwise the branch is flagged.", "",
        f"Initial q grid [A]: {list(INITIAL_GRID_A)}. Afterwards the absolute separation s is doubled. "
        f"The documented maximum displacement is {settings['max_q_A']:.9g} A. Extremely large separations, if needed, "
        "only test the numerical constrained-RHF asymptote; Coulombic restricted tails can decay slowly.", "",
        f"A plateau requires four consecutive accepted points: a three-point candidate and a farther confirmation. "
        f"Each absolute-separation ratio must be >=1.5. All four energies must have full span, maximum difference "
        f"from the last energy, and maximum adjacent difference strictly below {settings['plateau_tolerance_Eh']:.3g} Eh. "
        "Failed points are never removed from the consecutive-window test. Scanning stops after three consecutive "
        "SCF failures because outward RHF continuation is then unresolved. Reaching the maximum distance alone never certifies a plateau.", "",
        "Geometry invariants are checked at every scan point with explicitly recorded roundoff-aware tolerances. "
        "The complete point list, failed solver attempts, energies, SCF residuals, internal stability and geometry diagnostics "
        "are preserved in JSON and per-molecule scan CSV files.", "",
        "At the input boundary, the saved harmonic frequency is converted to the energy quantum hbar*omega in Hartree "
        "using hbar, light speed and the Hartree energy from the original harmonic calculation. "
        "All subsequent well-depth and level comparisons are performed directly in Hartree.", "",
        "The exact decision is E1_rel<0 -> n=1, otherwise E0_rel<0 -> n=0, otherwise NONE. "
        f"Any |E_n_rel| < {settings['threshold_warning_Eh']:.12g} Eh is marked NEAR_THRESHOLD while retaining "
        "the exact sign decision. This warning window is not a silently applied sign tolerance. "
        "It covers the numerical plateau tolerance in Hartree; it does not quantify RHF model error. "
        "No Morse correction, experimental dissociation energy, UHF energy or historical scan is used.", "",
        "## Complete summary", "",
    ]
    lines += harmonic.table(FIELDS, [[r["summary"].get(key) for key in FIELDS] for r in document["results"]])
    for result in document["results"]:
        s = result["summary"]
        lines += ["", f"## {s['molecule']} — {s['validation_status']}", "",
                  f"Asymptote status: {s['asymptote_status']}.", ""]
        if "basis" in s:
            lines += [f"RHF basis: {s['basis']}. Coordinate: {s['coordinate']}. s_eq={s['s_eq_A']:.9f} A.", ""]
        if result["failure_reasons"]:
            lines += [f"- {reason}" for reason in result["failure_reasons"]] + [""]
        if result["warnings"]:
            lines += [f"- {warning}" for warning in result["warnings"]] + [""]
        if "asymptotic_fragment_population_diagnostic" in result:
            lines += ["Separated-fragment Mulliken charge diagnostic (electron-charge units):", ""]
            lines += harmonic.table(["Fragment", "Charge [e]"],
                [[" ".join(f["atoms"]), f["mulliken_charge_e"]] for f in result["asymptotic_fragment_population_diagnostic"]])
            lines += [""]
        if "D_e_Eh" in s:
            lines += [f"E_eq={s['E_eq_Eh']:.14f} Eh; E_diss={s['E_diss_Eh']:.14f} Eh; "
                      f"D_e^(RHF-path)={s['D_e_Eh']:.12f} Eh.", "", "```text",
                      f"D_e^(RHF-path)            = {s['D_e_Eh']:.12f} Hartree"]
            for n, key in ((0, "half_hbar_omega_Eh"), (1, "three_halves_hbar_omega_Eh")):
                lines += [f"n={n} energy above minimum = {s[key]:.12f} Hartree"]
                if f"E{n}_rel_Eh" in s:
                    lines += [f"n={n} relative to diss.    = {s[f'E{n}_rel_Eh']:+.12f} Hartree " + ("BOUND" if s[f"n{n}_bound"] else "UNBOUND")]
            lines += [f"selected n               = {s.get('selected_n')}", "```", "", s.get("selected_reason", ""), ""]
            if s.get("selected_n") is not None:
                lines += [f"**Selected existing harmonic range: [{s['selected_range_min_A']:.9f}, "
                          f"{s['selected_range_max_A']:.9f}] A.**", ""]
        if "last_plateau_test" in result:
            lines += ["Final plateau assessment:", "", "```json",
                      json.dumps(harmonic.jsonable(result["last_plateau_test"]), indent=2), "```", ""]
        if result["scan"]:
            lines += [f"Last attempted q={result['scan'][-1]['q_A']:.9g} A. Full scan follows; accepted means "
                      "SCF, residual, internal stability and geometry checks passed.", ""]
            lines += harmonic.table(["q [A]", "s [A]", "E_RHF [Eh]", "SCF converged", "Orbital residual", "Internal stable", "Accepted"],
                [[p["q_A"], p["s_A"], None if p.get("energy_Eh") is None else f"{p['energy_Eh']:.14f}",
                  p.get("scf_converged"), p.get("orbital_gradient_norm_Eh"), p.get("internal_stable"), p["accepted"]]
                 for p in result["scan"]])
    lines += ["", "## Provenance", "", "```json", json.dumps(document["provenance"], indent=2), "```", "",
              "SCF convergence and electronic stability conventions were checked against the installed PySCF implementation "
              "and [the official SCF documentation](https://pyscf.org/user/scf.html#stability-analysis).", ""]
    return "\n".join(lines)


def write_outputs(document, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = harmonic.jsonable(document)
    (output_dir / "rhf_bound_level_selection.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (output_dir / "rhf_bound_level_selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r["summary"] for r in payload["results"])
    (output_dir / "RHF_BOUND_LEVEL_SELECTION_REPORT.md").write_text(build_report(document))
    scan_dir = output_dir / "rhf_bound_level_scans"
    scan_dir.mkdir(exist_ok=True)
    scan_fields = ["q_A", "s_A", "energy_Eh", "scf_converged", "orbital_gradient_norm_Eh", "internal_stable",
                   "lowest_internal_stability_eigenvalue_Eh", "restricted_stability_corrections", "accepted", "failure"]
    for result in payload["results"]:
        with (scan_dir / f"{result['summary']['molecule']}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scan_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["scan"])


def convert_saved_results_to_hartree(document):
    """Pure postprocessing: retain every scanned energy and geometry verbatim."""
    converted = copy.deepcopy(document)
    factor = conversion_from_constants(converted["provenance"]["constants"])
    converted["schema_version"] = 2
    converted["energy_unit"] = "Hartree"
    converted.pop("hartree_to_cm1", None)
    converted["source_wavenumber_to_Eh"] = factor
    settings = converted["settings"]
    if "threshold_warning_cm1" in settings:
        settings["threshold_warning_Eh"] = settings.pop("threshold_warning_cm1") * factor
    if settings["threshold_warning_Eh"] < settings["plateau_tolerance_Eh"]:
        raise ValueError("The Hartree threshold-warning window must cover the plateau tolerance.")
    for result in converted["results"]:
        summary = result["summary"]
        previous_selection = (summary.get("selected_n"), summary.get("selected_range_min_A"), summary.get("selected_range_max_A"))
        for old_key, new_key in (("harmonic_frequency_cm1", "harmonic_quantum_Eh"),
                                 ("half_hbar_omega_cm1", "half_hbar_omega_Eh"),
                                 ("three_halves_hbar_omega_cm1", "three_halves_hbar_omega_Eh"),
                                 ("E0_rel_cm1", "E0_rel_Eh"), ("E1_rel_cm1", "E1_rel_Eh")):
            if old_key in summary:
                summary[new_key] = summary.pop(old_key) * factor
        # The native Hartree depth is authoritative; never round-trip it through a converted value.
        summary.pop("D_e_cm1", None)
        if "harmonic_input" in result:
            result["harmonic_input"]["summary"] = harmonic_summary_in_hartree(result["harmonic_input"]["summary"], factor)
        if "plateau_energy_spread_cm1" in result:
            result.pop("plateau_energy_spread_cm1")
            result["plateau_energy_spread_Eh"] = result["accepted_plateau"]["pairwise_span_Eh"]
        if "selection_formula" in result:
            result["selection_formula"] = "E_n_rel_Eh = -D_e_Eh + (n + 0.5) * harmonic_quantum_Eh"
        if summary.get("D_e_Eh", 0) > 0 and "E0_rel_Eh" in summary:
            selected = select_level(summary["D_e_Eh"], summary["harmonic_quantum_Eh"],
                                    result["harmonic_input"]["summary"], settings["threshold_warning_Eh"])
            summary.update(selected)
            result["relative_level_energies_Eh"] = {f"E{n}_rel_Eh": summary[f"E{n}_rel_Eh"] for n in (0, 1)}
            current_selection = (summary["selected_n"], summary["selected_range_min_A"], summary["selected_range_max_A"])
            if current_selection != previous_selection:
                raise ValueError(f"Unit conversion changed the threshold decision for {summary['molecule']}; inspect boundary rounding.")
    if any(a["scan"] != b["scan"] for a, b in zip(document["results"], converted["results"])):
        raise RuntimeError("Unit conversion must not modify any saved scan point.")
    return converted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--harmonic-json", type=Path, default=root / "new_rhf_harmonic_bond_ranges.json")
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--reuse-scan", type=Path, help="Regenerate Hartree outputs from saved scan JSON without running SCF.")
    parser.add_argument("--molecules", nargs="+", choices=harmonic.MOLECULES, default=list(harmonic.MOLECULES))
    parser.add_argument("--max-q", type=float, default=1e7, help="Maximum outward displacement in Angstrom, including numerical far-tail checks.")
    parser.add_argument("--plateau-tol", type=float, default=1e-6, help="Strict tail energy tolerance in Hartree.")
    parser.add_argument("--threshold-warning-eh", type=float, default=DEFAULT_THRESHOLD_WARNING_EH,
                        help="Near-threshold energy warning window in Hartree.")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if any(not math.isfinite(x) or x <= 0 for x in (args.max_q, args.plateau_tol, args.threshold_warning_eh)) or args.threads < 1:
        parser.error("Distance, tolerances and threads must be positive and finite.")
    if args.reuse_scan is not None:
        scan_path = args.reuse_scan.expanduser().resolve()
        try:
            document = convert_saved_results_to_hartree(json.loads(scan_path.read_text()))
        except (OSError, KeyError, ValueError) as exc:
            parser.error(f"Cannot convert saved scan: {exc}")
        provenance = document["provenance"]
        provenance.setdefault("scan_calculation_script_sha256", provenance["script_sha256"])
        provenance["script_sha256"] = harmonic.sha256(__file__)
        provenance["unit_postprocessing"] = {
            "source_scan_json": str(scan_path), "source_scan_json_sha256": harmonic.sha256(scan_path),
            "updated_utc": datetime.now(timezone.utc).isoformat(), "SCF_recomputed": False,
            "nuclear_hessian_recomputed": False, "scan_points_unchanged": True,
        }
        write_outputs(document, args.output_dir.expanduser().resolve())
        for result in document["results"]:
            print_summary(result["summary"])
        return 0
    lib.num_threads(args.threads)
    source_path = args.harmonic_json.expanduser().resolve()
    settings = {"max_q_A": args.max_q, "plateau_tolerance_Eh": args.plateau_tol,
                "threshold_warning_Eh": args.threshold_warning_eh, "threads": args.threads,
                "SCF_energy_tolerance_Eh": 1e-13, "orbital_gradient_acceptance_Eh": SCF_GRADIENT_LIMIT,
                "internal_stability_lower_limit_Eh": STABILITY_LIMIT}
    document = {"schema_version": 2, "energy_unit": "Hartree", "created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
                "source_wavenumber_to_Eh": None, "results": [], "provenance": {
                    "harmonic_json": str(source_path), "script_sha256": harmonic.sha256(__file__),
                    "python_version": platform.python_version(), "pyscf_version": pyscf.__version__,
                    "nuclear_hessian_recomputed": False, "geometry_reoptimized": False,
                    "energy_model": "conventional all-electron RHF without density fitting"}}
    try:
        source, records = load_harmonic_input(source_path)
        document["source_wavenumber_to_Eh"] = conversion_from_constants(source["runtime"]["constants"])
        if args.threshold_warning_eh < args.plateau_tol:
            parser.error(f"--threshold-warning-eh must be at least {args.plateau_tol:.9g} Hartree "
                         "to cover the requested plateau energy tolerance.")
        document["provenance"].update(harmonic_json_sha256=harmonic.sha256(source_path),
                                      harmonic_script_sha256=source["runtime"]["script_sha256"],
                                      constants=source["runtime"]["constants"])
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Cannot load validated harmonic input: {exc}")
    output_dir = args.output_dir.expanduser().resolve()
    for name in dict.fromkeys(args.molecules):
        print(f"Scanning {name} on the existing RHF collective coordinate...", flush=True)
        last_progress = [0.]
        def progress(partial):
            if time.monotonic() - last_progress[0] > 20:
                p = partial["scan"][-1]
                print(f"  {name}: q={p['q_A']:.6g} A; accepted={p['accepted']}; points={len(partial['scan'])}", flush=True)
                last_progress[0] = time.monotonic()
        result = scan_one(records.get(name), {**settings, "molecule": name}, document["source_wavenumber_to_Eh"], progress)
        document["results"].append(result)
        print_summary(result["summary"])
        for reason in result["failure_reasons"]:
            print(f"  {reason}", flush=True)
        write_outputs(document, output_dir)
    return 0 if all(r["summary"]["validation_status"] == "PASS" for r in document["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
