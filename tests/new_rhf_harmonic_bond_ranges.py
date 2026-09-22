#!/usr/bin/env python3
"""New equilibrium-centered RHF harmonic ranges; no scan or legacy-code inputs.

Requires numpy, scipy, pyscf. Input: a CSV of RHF optimization metadata and XYZs.
Run --help for the input contract. This file is independent of all other scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import math
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq
import pyscf
from pyscf import gto, lib, scf
from pyscf.data import nist
from pyscf.hessian import rhf as rhf_hessian, thermo


MOLECULES = ("LiH", "BeH2", "H2O", "NH3", "N2", "CO", "HF", "H2S", "H2O2")
FORMULAS = {
    "LiH": {"Li": 1, "H": 1}, "BeH2": {"Be": 1, "H": 2},
    "H2O": {"O": 1, "H": 2}, "NH3": {"N": 1, "H": 3},
    "N2": {"N": 2}, "CO": {"C": 1, "O": 1}, "HF": {"H": 1, "F": 1},
    "H2S": {"S": 1, "H": 2}, "H2O2": {"O": 2, "H": 2},
}
STARS = {"BeH2": "Be", "H2O": "O", "NH3": "N", "H2S": "S"}
DIATOMICS = {"LiH", "N2", "CO", "HF"}
BOHR_A = nist.BOHR
BOHR_M = BOHR_A * 1e-10
EH_J = nist.HARTREE2J
AMU_KG = nist.ATOMIC_MASS
HBAR = nist.HBAR
C_MS = nist.LIGHT_SPEED_SI
TANGENT_STEPS = (1e-3, 5e-4, 2e-4)
ENERGY_STEPS = (0.01, 0.005, 0.002, 0.001)
TOL = {
    "gradient_max_Eh_per_Bohr": 2e-6,
    "gradient_rms_Eh_per_Bohr": 1e-6,
    "common_bond_spread_A": 1e-6,
    "bond_derivative": 2e-8,
    "fixed_length_error_A": 2e-10,
    "fixed_angle_error_deg": 2e-7,
    "fixed_angle_derivative_deg_per_A": 2e-5,
    "tangent_convergence_max": 2e-7,
    "com_derivative": 2e-9,
    "rotation_residual_amu_A": 2e-7,
    "alignment_distance_error_A": 2e-12,
    "hessian_symmetry_Eh_per_Bohr2": 1e-8,
    "hessian_translation_Eh_per_Bohr2": 1e-7,
    "hessian_gradient_probe_relative": 2e-5,
    "hessian_gradient_probe_absolute_Eh_per_Bohr2": 2e-7,
    "fd_relative": 2e-4,
    "fd_absolute_Eh_per_Bohr2": 2e-6,
    "mode_orthogonality": 2e-7,
    "mode_weight_sum": 2e-7,
    "mode_curvature_relative": 2e-7,
    "diatomic_mass_relative": 2e-8,
}
FIELDS = [
    "molecule", "RHF_basis", "coordinate_definition", "s_eq_A",
    "k_q_Eh_per_Bohr2", "k_q_Eh_per_A2", "k_q_N_per_m", "mu_eff_amu", "mu_eff_kg",
    "omega_rad_per_s", "harmonic_frequency_cm1", "Delta_q_n0_A", "Delta_q_n1_A",
    "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A", "Hessian_FD_relative_difference",
    "dominant_normal_mode", "dominant_mode_frequency_cm1", "mode_overlap",
    "validation_status", "message",
]


class InputProblem(ValueError):
    """An equilibrium geometry or reliable basis/provenance was not supplied."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_equilibrium(name, row, metadata_path):
    if row is None:
        raise InputProblem("No equilibrium metadata row is available.")
    if row.get("method", "").strip().upper() != "RHF":
        raise InputProblem("Metadata must establish method=RHF.")
    basis = row.get("basis", "").strip()
    if not basis:
        raise InputProblem("The equilibrium RHF basis is missing.")
    if row.get("status", "").lower() != "converged":
        raise InputProblem("Equilibrium optimization is incomplete; provide a converged structure.")
    if row.get("optimizer_success", "true").strip().lower() not in ("true", "1", "yes"):
        raise InputProblem("Equilibrium optimizer did not report success.")
    xyz_string = row.get("xyz_file", "").strip()
    if not xyz_string:
        raise InputProblem("The optimized Cartesian geometry path is missing.")
    supplied_path = Path(xyz_string).expanduser()
    xyz_path = supplied_path if supplied_path.is_absolute() else metadata_path.parent / supplied_path
    # A relocated input directory may retain absolute paths in the metadata.
    if not xyz_path.is_file() and supplied_path.is_absolute():
        xyz_path = metadata_path.parent / supplied_path.name
    if not xyz_path.is_file():
        raise InputProblem(f"Optimized XYZ file is missing: {xyz_path}")
    lines = xyz_path.read_text().splitlines()
    try:
        n = int(lines[0])
        records = [line.split() for line in lines[2:2 + n]]
        symbols = [r[0] for r in records]
        coords = np.array([[float(v) for v in r[1:4]] for r in records])
    except (ValueError, IndexError) as exc:
        raise InputProblem(f"Invalid optimized XYZ: {exc}") from exc
    if coords.shape != (n, 3) or not np.isfinite(coords).all():
        raise InputProblem("XYZ coordinates are incomplete or nonfinite.")
    if Counter(symbols) != Counter(FORMULAS[name]):
        raise InputProblem(f"XYZ atom composition does not match {name}.")
    comment = lines[1]
    match = re.search(r"\bRHF/(\S+)", comment, flags=re.I)
    if match and match.group(1).lower() != basis.lower():
        raise InputProblem("XYZ comment and metadata disagree on the RHF basis.")
    return symbols, coords, basis, {
        "metadata_file": str(metadata_path.resolve()), "metadata_sha256": sha256(metadata_path),
        "xyz_file": str(xyz_path.resolve()), "xyz_sha256": sha256(xyz_path),
        "xyz_comment": comment, "metadata_row": dict(row),
        "geometry_units": "Angstrom", "reoptimized": False,
    }


def bond(coords, i, j):
    return float(np.linalg.norm(coords[j] - coords[i]))


def angle(coords, i, j, k):
    a, b = coords[i] - coords[j], coords[k] - coords[j]
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def dihedral(coords, i, j, k, l):
    axis = coords[k] - coords[j]
    axis = axis / np.linalg.norm(axis)
    v, w = coords[i] - coords[j], coords[l] - coords[k]
    v, w = v - np.dot(v, axis) * axis, w - np.dot(w, axis) * axis
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-10:
        raise ValueError("The H-O-O-H dihedral is undefined for this geometry.")
    return math.degrees(math.atan2(np.dot(np.cross(axis, v), w), np.dot(v, w)))


def angular_difference(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def center(coords, masses):
    return np.sum(masses[:, None] * coords, axis=0) / masses.sum()


def align(coords, reference, masses):
    """Mass-weighted, proper (det=+1) row-vector Kabsch/Eckart alignment."""
    ref_com = center(reference, masses)
    x, y = coords - center(coords, masses), reference - ref_com
    u, _, vt = np.linalg.svd(x.T @ (masses[:, None] * y))
    proper = np.eye(3)
    proper[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ proper @ vt
    return x @ rotation + ref_com, rotation


class StretchPath:
    """Internal-coordinate construction entirely determined by the input R_e.

    Star molecules: keep equilibrium bond-direction cosines and replace each
    radial coordinate r_i by r_i+q. Peroxide: replace O-O radial coordinate,
    carrying the two equilibrium OH vectors rigidly with their own oxygens.
    These vector internal coordinates preserve all specified angles/torsions.
    """

    def __init__(self, name, symbols, reference, masses):
        self.name, self.symbols = name, list(symbols)
        self.ref, self.masses = np.array(reference, float), np.array(masses, float)
        self.stretches, self.fixed_bonds, self.angles, self.dihedrals = [], [], [], []
        if name in DIATOMICS:
            self.stretches = [(0, 1)]
            self.definition = f"q = r({symbols[0]}-{symbols[1]}) - r_e({symbols[0]}-{symbols[1]})"
        elif name in STARS:
            self.heavy = symbols.index(STARS[name])
            self.hydrogens = [i for i, sym in enumerate(symbols) if sym == "H"]
            self.stretches = [(self.heavy, h) for h in self.hydrogens]
            self.angles = [(i, self.heavy, j) for i, j in itertools.combinations(self.hydrogens, 2)]
            self.definition = f"r({STARS[name]}-H_i) = r_e({STARS[name]}-H_i) + q for every H"
        elif name == "H2O2":
            self.oxygens = [i for i, sym in enumerate(symbols) if sym == "O"]
            hydrogens = [i for i, sym in enumerate(symbols) if sym == "H"]
            assignments = sorted(
                (sum(bond(self.ref, o, h) for o, h in zip(self.oxygens, perm)), perm)
                for perm in itertools.permutations(hydrogens)
            )
            if abs(assignments[0][0] - assignments[1][0]) < 1e-8:
                raise ValueError("Ambiguous OH connectivity in the optimized peroxide geometry.")
            h1, h2 = assignments[0][1]
            o1, o2 = self.oxygens
            self.peroxide_h = (h1, h2)
            self.stretches = [(o1, o2)]
            self.fixed_bonds = [(o1, h1), (o2, h2)]
            self.angles = [(h1, o1, o2), (o1, o2, h2)]
            self.dihedrals = [(h1, o1, o2, h2)]
            self.definition = "q = r(O-O) - r_e(O-O); both OH lengths, OOH angles and HOOH torsion fixed"
        else:
            raise ValueError(f"Unsupported molecule: {name}")
        self.lengths = np.array([bond(self.ref, *pair) for pair in self.stretches])
        if np.any(self.lengths < 1e-8):
            raise ValueError("A participating equilibrium bond has zero length.")
        self.s_eq = float(np.mean(self.lengths))

    def raw(self, q):
        if np.any(self.lengths + q <= 0):
            raise ValueError("A stretch displacement would produce a nonpositive bond length.")
        coords = self.ref.copy()
        if self.name in DIATOMICS:
            axis = (self.ref[1] - self.ref[0]) / self.lengths[0]
            coords[1] = coords[0] + (self.lengths[0] + q) * axis
        elif self.name in STARS:
            for (heavy, h), length in zip(self.stretches, self.lengths):
                ray = (self.ref[h] - self.ref[heavy]) / length
                coords[h] = coords[heavy] + (length + q) * ray
        else:
            o1, o2 = self.oxygens
            h1, h2 = self.peroxide_h
            axis = (self.ref[o2] - self.ref[o1]) / self.lengths[0]
            coords[o2] = coords[o1] + (self.lengths[0] + q) * axis
            coords[h1] = coords[o1] + (self.ref[h1] - self.ref[o1])
            coords[h2] = coords[o2] + (self.ref[h2] - self.ref[o2])
        return coords

    def __call__(self, q):
        return align(self.raw(q), self.ref, self.masses)[0]

    def internals(self, coords):
        def label(indices):
            return "-".join(f"{self.symbols[i]}{i + 1}" for i in indices)
        values = {}
        for pairs, role in ((self.stretches, "stretch"), (self.fixed_bonds, "fixed_bond")):
            values.update({label(pair): {"value": bond(coords, *pair), "kind": role, "unit": "A"}
                           for pair in pairs})
        for indices in self.angles:
            values[label(indices)] = {"value": angle(coords, *indices), "kind": "fixed_angle", "unit": "deg"}
        for indices in self.dihedrals:
            values[label(indices)] = {"value": dihedral(coords, *indices), "kind": "fixed_dihedral", "unit": "deg"}
        return values

    def validate(self):
        samples, tangents, failures = [], [], []
        equilibrium = self.internals(self.ref)
        if np.ptp(self.lengths) > TOL["common_bond_spread_A"]:
            failures.append("Participating equilibrium lengths differ; a single common absolute s is undefined.")
        for delta in TANGENT_STEPS:
            plus, minus = self(delta), self(-delta)
            t = (plus - minus) / (2 * delta)
            tangents.append(t)
            residual_com = np.linalg.norm(center(t, self.masses))
            residual_rot = np.linalg.norm(np.sum(
                self.masses[:, None] * np.cross(self.ref - center(self.ref, self.masses), t), axis=0))
            p, m = self.internals(plus), self.internals(minus)
            derivatives, deviations = {}, {}
            for key, item in equilibrium.items():
                diff = angular_difference if item["kind"] == "fixed_dihedral" else lambda a, b: a - b
                derivative = diff(p[key]["value"], m[key]["value"]) / (2 * delta)
                derivatives[key] = derivative
                if item["kind"] == "stretch":
                    error = max(abs(p[key]["value"] - item["value"] - delta),
                                abs(m[key]["value"] - item["value"] + delta))
                    good = abs(derivative - 1) < TOL["bond_derivative"] and error < TOL["fixed_length_error_A"]
                else:
                    error = max(abs(diff(p[key]["value"], item["value"])),
                                abs(diff(m[key]["value"], item["value"])))
                    if item["unit"] == "A":
                        good = abs(derivative) < TOL["bond_derivative"] and error < TOL["fixed_length_error_A"]
                    else:
                        good = abs(derivative) < TOL["fixed_angle_derivative_deg_per_A"] and error < TOL["fixed_angle_error_deg"]
                deviations[key] = error
                if not good:
                    failures.append(f"Internal coordinate validation failed for {key} at delta={delta} A.")
            alignment_error, determinants = 0.0, []
            for q in (-delta, delta):
                raw = self.raw(q)
                aligned, rot = align(raw, self.ref, self.masses)
                determinants.append(float(np.linalg.det(rot)))
                for i, j in itertools.combinations(range(len(self.ref)), 2):
                    alignment_error = max(alignment_error, abs(bond(raw, i, j) - bond(aligned, i, j)))
            if residual_com > TOL["com_derivative"] or residual_rot > TOL["rotation_residual_amu_A"]:
                failures.append(f"Translation/rotation residual too large at delta={delta} A.")
            if alignment_error > TOL["alignment_distance_error_A"] or not np.allclose(determinants, 1, atol=1e-12, rtol=0):
                failures.append("Alignment changed internal distances or was not a proper rotation.")
            samples.append({"delta_A": delta, "tangent": t, "internal_derivatives_per_A": derivatives,
                            "internal_max_deviations": deviations, "com_translation_residual": residual_com,
                            "rotation_residual_amu_A": residual_rot, "alignment_distance_error_A": alignment_error,
                            "rotation_determinants": determinants})
        convergence = [float(np.max(np.abs(t - tangents[-1]))) for t in tangents]
        if max(convergence) > TOL["tangent_convergence_max"]:
            failures.append("The Cartesian tangent did not converge with displacement step.")
        return tangents[-1], {"equilibrium_internals": equilibrium, "samples": samples,
                              "tangent_max_differences_from_finest": convergence, "failures": failures}


def make_molecule(symbols, coords, basis):
    mol = gto.M(atom=list(zip(symbols, coords.tolist())), unit="Angstrom", basis=basis,
                charge=0, spin=0, symmetry=False, cart=False, verbose=0)
    if mol.has_ecp() or mol.nelectron != sum(mol.atom_charges()):
        raise ValueError("This workflow requires a neutral all-electron molecule.")
    return mol


def run_rhf(symbols, coords, basis, dm0=None):
    mf = scf.RHF(make_molecule(symbols, coords, basis))
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-10, 1e-14
    mf.max_cycle, mf.diis_space = 400, 12
    mf.conv_check = False  # Do not accept PySCF's relaxed extra-cycle criterion.
    mf.chkfile = None
    mf.kernel(dm0=dm0)
    orbital_gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if not mf.converged or orbital_gradient > 1e-10:
        # Same all-electron RHF model; second-order SCF is only a solver fallback.
        solver = mf.newton()
        solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-10, 400
        solver.kernel()
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = solver.mo_coeff, solver.mo_occ, solver.mo_energy
        mf.e_tot, mf.converged = solver.e_tot, solver.converged
        orbital_gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if not mf.converged or orbital_gradient > 1e-10 or not np.isfinite(mf.e_tot):
        raise RuntimeError(f"RHF SCF failed tight convergence; orbital gradient norm={orbital_gradient:.3e}")
    if getattr(mf, "with_df", None) is not None:
        raise RuntimeError("Density fitting is forbidden in this workflow.")
    return mf


def probability_mass(n, a):
    """Probability in [-a*l,+a*l], using normalized oscillator densities."""
    if n == 0:
        return math.erf(a)
    if n == 1:
        return math.erf(a) - 2 * a * math.exp(-a * a) / math.sqrt(math.pi)
    raise ValueError("Only n=0 and n=1 are supported.")


def oscillator_ranges(k_bohr, mu_amu, s_eq):
    k_si, mu_kg = k_bohr * EH_J / BOHR_M**2, mu_amu * AMU_KG
    if k_si <= 0 or mu_kg <= 0 or not np.isfinite([k_si, mu_kg]).all():
        raise ValueError("A positive finite curvature and generalized mass are required.")
    omega = math.sqrt(k_si / mu_kg)
    length = math.sqrt(HBAR / (mu_kg * omega)) / 1e-10
    n1 = math.sqrt(3) * length
    summary = {
        "k_q_Eh_per_Bohr2": k_bohr, "k_q_Eh_per_A2": k_bohr / BOHR_A**2,
        "k_q_N_per_m": k_si, "mu_eff_amu": mu_amu, "mu_eff_kg": mu_kg,
        "omega_rad_per_s": omega, "harmonic_frequency_cm1": omega / (2 * math.pi * C_MS * 100),
        "Delta_q_n0_A": length, "Delta_q_n1_A": n1,
        "n0_min_A": s_eq - length, "n0_max_A": s_eq + length,
        "n1_min_A": s_eq - n1, "n1_max_A": s_eq + n1,
    }
    probability_intervals = []
    for n in (0, 1):
        for probability in (0.95, 0.99, 0.999):
            a = brentq(lambda z: probability_mass(n, z) - probability, 0, 10, xtol=1e-14)
            width = a * length
            probability_intervals.append({"n": n, "probability": probability, "half_width_in_l": a,
                                          "q_interval_A": [-width, width],
                                          "s_interval_A": [s_eq - width, s_eq + width]})
    return summary, {
        "oscillator_length_A": length, "q_n0_minus_A": -length, "q_n0_plus_A": length,
        "q_n1_minus_A": -n1, "q_n1_plus_A": n1,
        "probability_intervals": probability_intervals,
    }


def normal_modes(mol, hessian, tangent, masses, k_bohr, mu_amu):
    analysis = thermo.harmonic_analysis(mol, hessian, mass=masses, imaginary_freq=False)
    modes = analysis["norm_mode"]
    weighted = modes * np.sqrt(masses)[None, :, None]
    flat = weighted.reshape(len(modes), -1)
    unit_t = (np.sqrt(masses)[:, None] * tangent / math.sqrt(mu_amu)).ravel()
    overlaps = flat @ unit_t
    weights = overlaps**2
    reconstruction = float(np.dot(weights, analysis["force_const_au"]))
    target = k_bohr / mu_amu
    reconstruction_error = abs(reconstruction - target) / max(abs(target), 1e-30)
    rows = [{"mode": i + 1, "frequency_cm1": float(analysis["freq_wavenumber"][i]),
             "signed_overlap": float(overlaps[i]), "overlap": float(abs(overlaps[i])),
             "squared_overlap_weight": float(weights[i])} for i in range(len(modes))]
    dominant = int(np.argmax(weights))
    return {
        "modes": rows, "cartesian_mass_normalized_modes": modes, "mode_indexing": "1-based; ascending eigenvalue",
        "dominant_normal_mode": dominant + 1,
        "dominant_mode_frequency_cm1": rows[dominant]["frequency_cm1"],
        "mode_overlap": rows[dominant]["overlap"], "sum_squared_weights": float(weights.sum()),
        "mixed_modes": bool(np.count_nonzero(weights > 1e-6) > 1),
        "resolved_mode_weight_threshold": 1e-6,
        "secondary_weight_sum": float(weights.sum() - weights[dominant]),
        "substantial_mixing": bool(weights[dominant] < 0.99),
        "substantial_mixing_threshold_dominant_weight": 0.99,
        "mass_orthogonality_max_error": float(np.max(np.abs(flat @ flat.T - np.eye(len(modes))))),
        "curvature_reconstruction_relative_error": reconstruction_error,
        "imaginary_mode_count": int(np.sum(analysis["freq_wavenumber"] < 0)),
    }


def calculate(name, row, metadata_path):
    summary = {"molecule": name, "RHF_basis": (row or {}).get("basis", ""), "validation_status": "ERROR"}
    result = {"summary": summary, "failures": []}
    start = time.monotonic()
    try:
        symbols, reference, basis, provenance = load_equilibrium(name, row, metadata_path)
        result["input"] = provenance
        result["symbols"], result["equilibrium_cartesian_A"] = symbols, reference
        mol = make_molecule(symbols, reference, basis)
        masses = mol.atom_mass_list(isotope_avg=True)
        result["isotope_average_masses_amu"] = masses
        path = StretchPath(name, symbols, reference, masses)
        tangent, diagnostics = path.validate()
        result["path_validation"], result["tangent_dR_dq"] = diagnostics, tangent
        summary.update(RHF_basis=basis, coordinate_definition=path.definition, s_eq_A=path.s_eq)
        result["failures"].extend(diagnostics["failures"])
        if diagnostics["failures"]:
            summary["validation_status"] = "FAIL_PATH"
            return result

        mf = run_rhf(symbols, reference, basis)
        gradient = mf.nuc_grad_method().kernel()
        gradient_max = float(np.max(np.abs(gradient)))
        gradient_rms = float(np.sqrt(np.mean(gradient**2)))
        gradient_projection = float(np.sum(gradient * tangent))
        result["stationarity"] = {
            "E_RHF_Eh": float(mf.e_tot), "gradient_Eh_per_Bohr": gradient,
            "gradient_max_Eh_per_Bohr": gradient_max, "gradient_rms_Eh_per_Bohr": gradient_rms,
            "gradient_projection_Eh_per_Bohr": gradient_projection,
            "gradient_projection_Eh_per_A": gradient_projection / BOHR_A,
            "orbital_gradient_norm_Eh": float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))),
        }
        if gradient_max > TOL["gradient_max_Eh_per_Bohr"] or gradient_rms > TOL["gradient_rms_Eh_per_Bohr"]:
            result["failures"].append("FAIL_STATIONARITY: Cartesian nuclear gradient exceeds tolerance.")

        hessobj = mf.Hessian()
        hessobj.max_cycle = 100
        # The installed RHF Hessian solver uses mf.conv_tol_cpscf for CPHF.
        mf.conv_tol_cpscf = 1e-12
        hessian = hessobj.kernel()
        if hessian.shape != (mol.natm, mol.natm, 3, 3) or not np.isfinite(hessian).all():
            raise RuntimeError("Analytic Hessian has an invalid shape or nonfinite values.")
        symmetry_error = float(np.max(np.abs(hessian - hessian.transpose(1, 0, 3, 2))))
        translation_error = float(np.max(np.abs(hessian.sum(axis=1))))
        # Verify the complete H*v response against derivatives of analytic gradients.
        # The diagnostic probe is unrelated to, and does not rescale, the physical t.
        probe = np.sin(np.arange(1, mol.natm * 3 + 1)).reshape(mol.natm, 3)
        probe /= np.linalg.norm(probe)
        epsilon_bohr = 1e-4
        probe_gradients = []
        for sign in (1, -1):
            displaced = reference + sign * epsilon_bohr * BOHR_A * probe
            check_mf = run_rhf(symbols, displaced, basis, mf.make_rdm1())
            probe_gradients.append(check_mf.nuc_grad_method().kernel())
        numerical_response = (probe_gradients[0] - probe_gradients[1]) / (2 * epsilon_bohr)
        analytic_response = np.einsum("abxy,by->ax", hessian, probe)
        probe_error = float(np.max(np.abs(numerical_response - analytic_response)))
        probe_scale = float(np.max(np.abs(analytic_response)))
        result["hessian_validation"] = {
            "convention": "H[a,b,x,y] = d2E/(dR[a,x] dR[b,y]); Eh/Bohr^2",
            "shape": hessian.shape, "symmetry_max_error_Eh_per_Bohr2": symmetry_error,
            "translation_max_error_Eh_per_Bohr2": translation_error,
            "gradient_probe_step_Bohr": epsilon_bohr, "gradient_probe_vector": probe,
            "gradient_probe_max_error_Eh_per_Bohr2": probe_error,
            "gradient_probe_relative_error": probe_error / max(probe_scale, 1e-30),
            "analytic_gradient_response": analytic_response, "numerical_gradient_response": numerical_response,
        }
        result["cartesian_hessian_Eh_per_Bohr2"] = hessian
        if symmetry_error > TOL["hessian_symmetry_Eh_per_Bohr2"] or translation_error > TOL["hessian_translation_Eh_per_Bohr2"]:
            result["failures"].append("FAIL_HESSIAN: symmetry or translational sum rule failed.")
        if probe_error > TOL["hessian_gradient_probe_absolute_Eh_per_Bohr2"] + TOL["hessian_gradient_probe_relative"] * probe_scale:
            result["failures"].append("FAIL_HESSIAN_CONVENTION: analytic H*v disagrees with gradient differences.")

        k_bohr = float(np.einsum("ax,abxy,by->", tangent, hessian, tangent))
        mu_amu = float(np.einsum("a,ax,ax->", masses, tangent, tangent))
        summary.update(k_q_Eh_per_Bohr2=k_bohr, k_q_Eh_per_A2=k_bohr / BOHR_A**2,
                       k_q_N_per_m=k_bohr * EH_J / BOHR_M**2, mu_eff_amu=mu_amu, mu_eff_kg=mu_amu * AMU_KG)
        if name in DIATOMICS:
            reduced_mass = float(np.prod(masses) / np.sum(masses))
            error = abs(mu_amu / reduced_mass - 1)
            axis = (reference[1] - reference[0]) / path.s_eq
            ordinary_k = float(axis @ hessian[1, 1] @ axis)
            result["diatomic_checks"] = {
                "ordinary_reduced_mass_amu": reduced_mass, "mass_relative_error": error,
                "ordinary_bond_curvature_Eh_per_Bohr2": ordinary_k,
                "curvature_relative_error": abs(ordinary_k / k_bohr - 1) if k_bohr else None,
            }
            if error > TOL["diatomic_mass_relative"] or abs(ordinary_k - k_bohr) > 1e-7:
                result["failures"].append("FAIL_DIATOMIC: generalized mass or curvature disagrees with ordinary bond stretch.")

        # At imperfect stationarity d2E/dq2 = t H t + gradient . R''.
        acceleration_step = TANGENT_STEPS[-1]
        acceleration = (path(acceleration_step) - 2 * reference + path(-acceleration_step)) / acceleration_step**2
        gradient_correction = float(np.sum(gradient * acceleration) * BOHR_A)
        result["path_acceleration"] = {"R_second_derivative_per_A": acceleration,
                                       "gradient_correction_Eh_per_Bohr2": gradient_correction}
        fd_rows = []
        for h in ENERGY_STEPS:
            energies = [run_rhf(symbols, path(sign * h), basis, mf.make_rdm1()).e_tot for sign in (1, -1)]
            second = math.fsum([float(energies[0]), float(energies[1]), -2 * float(mf.e_tot)])
            fd_a = second / h**2
            fd_bohr = second / (h / BOHR_A)**2
            fd_rows.append({"h_A": h, "E_plus_Eh": float(energies[0]), "E_zero_Eh": float(mf.e_tot),
                            "E_minus_Eh": float(energies[1]), "k_FD_Eh_per_A2": fd_a,
                            "k_FD_Eh_per_Bohr2": fd_bohr,
                            "relative_difference": abs(fd_bohr - k_bohr) / max(abs(k_bohr), 1e-30),
                            "difference_from_acceleration_corrected_H_Eh_per_Bohr2": fd_bohr - k_bohr - gradient_correction})
        # Predetermine the two smallest steps; never select a favorable step after the fact.
        fine, coarse = fd_rows[-1], fd_rows[-2]
        richardson = (4 * fine["k_FD_Eh_per_Bohr2"] - coarse["k_FD_Eh_per_Bohr2"]) / 3
        allowed_fd = TOL["fd_absolute_Eh_per_Bohr2"] + TOL["fd_relative"] * abs(k_bohr)
        fd_ok = all(abs(r["k_FD_Eh_per_Bohr2"] - k_bohr) <= allowed_fd for r in (coarse, fine))
        fd_ok = fd_ok and abs(fine["k_FD_Eh_per_Bohr2"] - coarse["k_FD_Eh_per_Bohr2"]) <= allowed_fd
        result["finite_differences"] = {
            "rows": fd_rows, "comparison_h_A": fine["h_A"], "richardson_Eh_per_Bohr2": richardson,
            "richardson_relative_difference": abs(richardson - k_bohr) / max(abs(k_bohr), 1e-30),
            "acceptance_absolute_tolerance_Eh_per_Bohr2": allowed_fd,
            "convergence_pass": fd_ok,
        }
        summary["Hessian_FD_relative_difference"] = fine["relative_difference"]
        if not fd_ok:
            result["failures"].append("FAIL_FD: finest two energy curvatures disagree with H or each other; inspect gradient correction and convergence table.")
        if k_bohr <= 0 or mu_amu <= 0:
            result["failures"].append("FAIL_CURVATURE_OR_MASS: positive curvature and mass are required.")
        else:
            ranges, oscillator = oscillator_ranges(k_bohr, mu_amu, path.s_eq)
            summary.update(ranges)
            result["harmonic_oscillator"] = oscillator
            modes = normal_modes(mol, hessian, tangent, masses, k_bohr, mu_amu)
            result["normal_mode_diagnostic"] = modes
            for key in ("dominant_normal_mode", "dominant_mode_frequency_cm1", "mode_overlap"):
                summary[key] = modes[key]
            if modes["mass_orthogonality_max_error"] > TOL["mode_orthogonality"] or abs(modes["sum_squared_weights"] - 1) > TOL["mode_weight_sum"] or modes["curvature_reconstruction_relative_error"] > TOL["mode_curvature_relative"]:
                result["failures"].append("FAIL_MODE_CONVENTION: mass normalization or spectral curvature reconstruction failed.")
            if modes["imaginary_mode_count"]:
                result["failures"].append("FAIL_EQUILIBRIUM_MINIMUM: imaginary vibrational modes were found.")
            if not np.isfinite(list(ranges.values())).all() or ranges["n1_min_A"] <= 0:
                result["failures"].append("FAIL_RANGE: nonfinite extents or nonpositive bond interval endpoint.")
        summary["validation_status"] = "PASS" if not result["failures"] else "FAIL_VALIDATION"
    except InputProblem as exc:
        summary["validation_status"] = "NEEDS_INPUT"
        result["failures"].append(str(exc))
    except Exception as exc:
        summary["validation_status"] = "ERROR"
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        summary["message"] = "; ".join(result["failures"])
        result["elapsed_seconds"] = time.monotonic() - start
    return result


def fmt(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.9g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]


def build_report(document):
    results = document["results"]
    lines = [
        "# New RHF harmonic bond/stretch ranges", "",
        "These are newly derived equilibrium-centered RHF harmonic ranges, independent of any previous QML geometry grids. "
        "Only the supplied converged RHF optimization metadata, its XYZ geometries, and its basis labels are used. "
        "No geometry is reoptimized, and no previous scan or cutoff is read or compared.", "",
        f"Generated: {document['created_utc']}. PySCF {pyscf.__version__}; neutral, singlet, all-electron RHF; no density fitting.", "",
        "## Main result", "",
        "The default is the **n=1 classical turning-point range**, s_eq ± sqrt(3) l_q, in Angstrom. "
        "For collective stretches s is the common participating bond length; for H2O2 it is the O-O separation. "
        "Only PASS rows are validated results. Values attached to failed rows are diagnostic.", "",
    ]
    lines += table(["Molecule", "RHF basis", "s_eq [A]", "Delta_q(n=1) [A]", "n1_min [A]", "n1_max [A]", "Status"],
                   [[r["summary"].get(k) for k in ("molecule", "RHF_basis", "s_eq_A", "Delta_q_n1_A", "n1_min_A", "n1_max_A", "validation_status")] for r in results])
    lines += ["", "## Definitions and units", "",
        "Scientific chain: optimized R_e -> newly defined q -> analytic RHF Hessian H -> t=dR/dq -> "
        "k_q=t^T H t -> mu_eff=sum_A m_A |t_A|^2 -> omega -> harmonic extents -> new ranges.", "",
        "Both Cartesian R and q are measured in Angstrom when constructing t. Consequently t is dimensionless and "
        "has the same numerical components for dR_Bohr/dq_Bohr. It is never Euclidean-normalized for curvature or mass. "
        "H[a,b,x,y] has units Eh/Bohr^2. Flattening uses H.transpose(0,2,1,3).reshape(3N,3N). "
        "Thus k_A=k_Bohr/a0_A^2 and k_SI=k_Bohr*Eh_J/a0_m^2. An energy finite difference uses (h_A/a0_A)^2 in the denominator.", "",
        "omega=sqrt(k_SI/mu_kg); wavenumber=omega/(2*pi*c_m_s*100); l_q=sqrt(hbar/(mu_kg*omega)). "
        "The turning points are ±l_q (n=0) and ±sqrt(3)l_q (n=1). These are not RMS widths or probability cutoffs. "
        "The 1D oscillator has infinite tails; the ranges use the requested classical turning-point convention. "
        "The harmonic model is local; agreement of small-step curvatures does not establish anharmonic accuracy across the full interval.", "",
        "All constants are taken consistently from the installed pyscf.data.nist, including its Bohr-to-Angstrom conversion. "
        "Masses are explicitly mol.atom_mass_list(isotope_avg=True), also passed to the normal-mode analysis.", "",
        "For star molecules the optimized bond directions define fixed angular internal coordinates and each radial length is increased by q. "
        "For peroxide, changing the optimized O-O radial length translates each rigid OH fragment consistently. "
        "A mass-weighted proper Kabsch rotation and center-of-mass alignment then impose the equilibrium Eckart frame. "
        "No historical length, angle, torsion, builder, or nominal equilibrium value enters this construction.", "",
        "Normal-mode overlaps use u=sqrt(M)t/sqrt(mu_eff) and v_j=sqrt(M)L_j, with L_j^T M L_k=delta_jk. "
        "The diagnostic squared weights sum to one. Mode phases are arbitrary; absolute overlaps and signed overlaps are both saved. "
        "More than one weight above 1e-6 is labeled mixed; a dominant weight below 0.99 indicates substantial mixing. "
        "Small admixtures are reported explicitly. The collective path is never replaced by a normal mode.", "",
        "## Validation policy", "",
        "SCF thresholds: energy 1e-13 Eh, orbital gradient norm 1e-10, integral screening 1e-14, at most 400 cycles. "
        "The relaxed extra SCF convergence cycle is disabled. A second-order RHF solver is used only if needed. "
        "CPHF tolerance is 1e-12. Stationarity requires max Cartesian gradient <=2e-6 and RMS <=1e-6 Eh/Bohr; "
        "these limits accommodate finite optimizer convergence while the independent curvature tests remain required.", "",
        "The actual installed Hessian convention is checked by shape, symmetry, translational sum rules, and an independent "
        "analytic-gradient finite difference along a generic Cartesian probe (step 1e-4 Bohr). "
        "Energy finite differences are evaluated at all four prescribed steps. Both 0.002 and 0.001 A must agree with the "
        "projected Hessian and with each other within 2e-6 Eh/Bohr^2 + 2e-4 |k_q|. "
        "This allows absolute SCF cancellation noise and O(h^2) truncation without choosing a favorable step after seeing results. "
        "The table's relative difference always uses h=0.001 A; Richardson extrapolation is an additional diagnostic.", "",
        "At a nonexact stationary point, the path energy curvature includes g dot R'' in addition to t^T H t. "
        "That correction is measured and reported; a large nuclear gradient is flagged and never silently accepted. "
        "Additional checks include internal-coordinate preservation, converged tangent, translation/rotation removal, "
        "positive mass/curvature, diatomic reduced masses, normal-mode normalization, spectral curvature reconstruction, "
        "absence of imaginary vibrational modes, and finite positive bond intervals.", "",
        "All numerical tolerances, constants, source hashes, input hashes, Cartesian Hessians, tangents, energies and modes "
        "are preserved in the JSON output. Missing geometry, basis, or converged RHF provenance produces NEEDS_INPUT.", "",
        "## Complete summary", "",
    ]
    lines += table(FIELDS, [[r["summary"].get(k) for k in FIELDS] for r in results])
    for r in results:
        s = r["summary"]
        lines += ["", f"## {s['molecule']} — {s['validation_status']}", ""]
        if r["failures"]:
            lines += [f"- {message}" for message in r["failures"]] + [""]
        if "input" not in r:
            continue
        lines += [f"Basis: {s['RHF_basis']}. Coordinate: {s.get('coordinate_definition', 'unavailable')}.", "",
                  f"Input: `{r['input']['xyz_file']}`. SHA-256: `{r['input']['xyz_sha256']}`.", ""]
        lines += table(["Atom", "x [A]", "y [A]", "z [A]", "Average mass [amu]"],
                       [[f"{sym}{i+1}", *coords, r["isotope_average_masses_amu"][i]]
                        for i, (sym, coords) in enumerate(zip(r["symbols"], r["equilibrium_cartesian_A"]))])
        if "path_validation" in r:
            p = r["path_validation"]
            lines += ["", "Optimized internal coordinates and finest-step derivatives:", ""]
            lines += table(["Internal coordinate", "Role", "Equilibrium value", "Unit", "Derivative per A"],
                           [[key, v["kind"], v["value"], v["unit"], p["samples"][-1]["internal_derivatives_per_A"][key]]
                            for key, v in p["equilibrium_internals"].items()])
            lines += ["", "Unnormalized tangent t (dimensionless):", ""]
            lines += table(["Atom", "t_x", "t_y", "t_z"],
                           [[f"{sym}{i+1}", *r["tangent_dR_dq"][i]] for i, sym in enumerate(r["symbols"])])
            lines += ["", "Tangent convergence and Eckart residuals:", ""]
            lines += table(["delta [A]", "max |t-t_finest|", "COM derivative norm", "Rotation residual [amu A]"],
                           [[sample["delta_A"], error, sample["com_translation_residual"], sample["rotation_residual_amu_A"]]
                            for sample, error in zip(p["samples"], p["tangent_max_differences_from_finest"])])
        if "stationarity" in r:
            g = r["stationarity"]
            lines += ["", f"RHF energy: {g['E_RHF_Eh']:.14f} Eh. Maximum/RMS nuclear gradient: "
                      f"{g['gradient_max_Eh_per_Bohr']:.6g} / {g['gradient_rms_Eh_per_Bohr']:.6g} Eh/Bohr. "
                      f"Gradient projection: {g['gradient_projection_Eh_per_A']:.6g} Eh/A.", ""]
        if "hessian_validation" in r:
            h = r["hessian_validation"]
            lines += [f"Hessian shape: {h['shape']}; symmetry error {h['symmetry_max_error_Eh_per_Bohr2']:.3g}; "
                      f"translation error {h['translation_max_error_Eh_per_Bohr2']:.3g} Eh/Bohr^2. "
                      f"Gradient-probe relative error: {h['gradient_probe_relative_error']:.3g}.", ""]
        if "finite_differences" in r:
            fd = r["finite_differences"]
            lines += [f"Analytic k_q: {s['k_q_Eh_per_Bohr2']:.10g} Eh/Bohr^2 = {s['k_q_Eh_per_A2']:.10g} Eh/A^2 "
                      f"= {s['k_q_N_per_m']:.10g} N/m. Generalized mass: {s['mu_eff_amu']:.10g} amu "
                      f"= {s['mu_eff_kg']:.10g} kg.", ""]
            lines += table(["h [A]", "E(-h) [Eh]", "E(0) [Eh]", "E(+h) [Eh]", "k_FD [Eh/Bohr^2]", "Relative difference"],
                           [[v["h_A"], f"{v['E_minus_Eh']:.14f}", f"{v['E_zero_Eh']:.14f}", f"{v['E_plus_Eh']:.14f}",
                             v["k_FD_Eh_per_Bohr2"], v["relative_difference"]] for v in fd["rows"]])
            lines += ["", f"Richardson curvature: {fd['richardson_Eh_per_Bohr2']:.10g} Eh/Bohr^2; relative difference "
                      f"{fd['richardson_relative_difference']:.3g}. Residual-gradient path correction: "
                      f"{r['path_acceleration']['gradient_correction_Eh_per_Bohr2']:.3g} Eh/Bohr^2.", ""]
        if "diatomic_checks" in r:
            d = r["diatomic_checks"]
            lines += [f"Ordinary diatomic reduced mass: {d['ordinary_reduced_mass_amu']:.10g} amu "
                      f"(relative discrepancy {d['mass_relative_error']:.3g}); ordinary d2E/dr2: "
                      f"{d['ordinary_bond_curvature_Eh_per_Bohr2']:.10g} Eh/Bohr^2.", ""]
        if "harmonic_oscillator" in r:
            lines += [f"omega = {s['omega_rad_per_s']:.10g} rad/s; collective frequency = {s['harmonic_frequency_cm1']:.6f} cm^-1.", "",
                      f"n=0: q in [{-s['Delta_q_n0_A']:.9f}, {s['Delta_q_n0_A']:.9f}] A; "
                      f"s in [{s['n0_min_A']:.9f}, {s['n0_max_A']:.9f}] A.", "",
                      f"n=1: q in [{-s['Delta_q_n1_A']:.9f}, {s['Delta_q_n1_A']:.9f}] A.", "",
                      f"**NEW n=1 harmonic range: [{s['n1_min_A']:.9f}, {s['n1_max_A']:.9f}] A.**", "",
                      "Separate symmetric probability intervals (normalized oscillator densities; not turning points):", ""]
            lines += table(["n", "Probability", "q_min [A]", "q_max [A]", "s_min [A]", "s_max [A]"],
                           [[v["n"], v["probability"], *v["q_interval_A"], *v["s_interval_A"]]
                            for v in r["harmonic_oscillator"]["probability_intervals"]])
        if "normal_mode_diagnostic" in r:
            m = r["normal_mode_diagnostic"]
            lines += ["", f"Dominant vibrational mode: {m['dominant_normal_mode']} at {m['dominant_mode_frequency_cm1']:.6f} cm^-1. "
                      f"Mixed coordinate (multiple weights >1e-6): {m['mixed_modes']}. "
                      f"Total secondary-mode weight: {100*m['secondary_weight_sum']:.6f}%; substantial mixing "
                      f"(dominant weight <0.99): {m['substantial_mixing']}. Weight sum: {m['sum_squared_weights']:.12f}. "
                      f"Spectral curvature reconstruction relative error: {m['curvature_reconstruction_relative_error']:.3g}.", ""]
            lines += table(["Mode (1-based)", "Frequency [cm^-1]", "Absolute overlap", "Squared weight"],
                           [[v["mode"], v["frequency_cm1"], v["overlap"], v["squared_overlap_weight"]] for v in m["modes"]])
    lines += ["", "## Reproducibility", "", "```json", json.dumps(jsonable(document["runtime"]), indent=2), "```", "",
              "Installed implementation was inspected and hashed. Convention references: "
              "[PySCF RHF Hessian](https://pyscf.org/_modules/pyscf/hessian/rhf.html), "
              "[PySCF harmonic analysis](https://pyscf.org/_modules/pyscf/hessian/thermo.html), "
              "[PySCF atomic masses](https://pyscf.org/_modules/pyscf/gto/mole.html#atom_mass_list).", ""]
    return "\n".join(lines)


def write_outputs(document, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "new_rhf_harmonic_bond_ranges"
    payload = jsonable(document)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r["summary"] for r in payload["results"])
    (output_dir / "NEW_RHF_HARMONIC_BOND_RANGE_REPORT.md").write_text(build_report(document))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Input CSV columns: molecule, method=RHF, basis, status=converged, xyz_file. "
        "Optional optimizer_success must be true. XYZ coordinates must be in Angstrom. "
        "Relative XYZ paths resolve against the metadata directory. Missing input is reported per molecule; "
        "no automatic optimization or fallback basis is used.")
    parser.add_argument("--metadata", type=Path, default=Path(__file__).resolve().parent / "rhf_geometries/rhf_equilibrium_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--molecules", nargs="+", choices=MOLECULES, default=list(MOLECULES))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    lib.num_threads(args.threads)
    metadata_path = args.metadata.expanduser().resolve()
    rows = {}
    if metadata_path.is_file():
        with metadata_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                name = row.get("molecule", "")
                if name in rows:
                    parser.error(f"Duplicate equilibrium metadata for {name}")
                rows[name] = row
    runtime = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "pyscf": pyscf.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "threads": args.threads, "script_sha256": sha256(__file__),
        "source_hashes": {name: hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
                          for name, obj in (("rhf_hessian", rhf_hessian), ("harmonic_analysis", thermo.harmonic_analysis),
                                            ("atom_mass_list", gto.mole.atom_mass_list))},
        "constants": {"Bohr_A": BOHR_A, "Bohr_m": BOHR_M, "Hartree_J": EH_J, "amu_kg": AMU_KG,
                      "hbar_J_s": HBAR, "c_m_per_s": C_MS, "source": "installed pyscf.data.nist"},
        "SCF": {"conv_tol": 1e-13, "conv_tol_grad": 1e-10, "direct_scf_tol": 1e-14, "max_cycle": 400,
                "density_fitting": False, "all_electron": True, "conv_tol_cpscf": 1e-12},
    }
    document = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime, "tolerances": TOL, "results": []}
    for name in dict.fromkeys(args.molecules):
        print(f"Calculating {name} from supplied RHF equilibrium geometry...", flush=True)
        result = calculate(name, rows.get(name), metadata_path)
        document["results"].append(result)
        s = result["summary"]
        print(f"{name}: {s['validation_status']}" +
              (f"; NEW n=1 range [{s['n1_min_A']:.9f}, {s['n1_max_A']:.9f}] Angstrom" if "n1_min_A" in s else "") +
              (f"; {s['message']}" if s["message"] else ""), flush=True)
        write_outputs(document, args.output_dir.expanduser().resolve())
    return 0 if all(r["summary"]["validation_status"] == "PASS" for r in document["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
