#!/usr/bin/env python3
"""Methods 4.1 descriptors from consistent second-order MP2 wavefunction RDMs.

Standard PySCF RDMs are retained as a separate audit. No FCI energy is solved.
See md/METHODS41.md for the explicit density definition. Inputs are read-only.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from methods41_data import MOLECULES, discover_dataset
from methods41_rdm import build_consistent_rdms
from methods41_validation import dual_rdm_diagnostics

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "methods41-v3-fullspace"
RDM_DEFINITION = "normalized truncated MP2 wavefunction, expectation values through second order"
RDM_CONVENTION = "Gamma[p,q,r,s] = sum_spin <p^dagger r^dagger s q>; real spatial AOs"
RDM_SOURCE = RDM_DEFINITION
PYSCF_RDM_SOURCE = "https://pyscf.org/_modules/pyscf/mp/mp2.html#make_rdm2"
TOL = {"matrix": 1e-8, "rdm": 1e-8, "energy_Ha": 1e-10,
       "overlap_min": 1e-8, "gradient": 1e-10, "imaginary": 1e-12}


class ValidationError(ValueError):
    """Invalid scientific inputs or failed numerical checks."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def real_array(value, name):
    array = np.asarray(value)
    require(np.isfinite(array).all(), f"{name} contains nonfinite values")
    if np.iscomplexobj(array):
        require(float(np.max(np.abs(array.imag), initial=0)) <= TOL["imaginary"],
                f"{name} contains complex values")
        array = array.real
    return np.asarray(array, dtype=float)


def maxabs(array):
    return float(np.max(np.abs(array), initial=0))


def check(diagnostics, name, value, limit):
    diagnostics.setdefault("checks", {})[name] = {
        "value": float(value), "limit": float(limit), "passed": bool(value <= limit)}


def finish_checks(diagnostics):
    diagnostics["passed"] = all(c["passed"] for c in diagnostics["checks"].values())
    diagnostics["failed_checks"] = [key for key, c in diagnostics["checks"].items() if not c["passed"]]
    return diagnostics


def disconnected_rhf(P):
    """Spin-summed RHF product in PySCF's chemist-index convention."""
    P = real_array(P, "RHF density")
    require(P.ndim == 2 and P.shape[0] == P.shape[1], "RHF density must be square")
    return np.einsum("pq,rs->pqrs", P, P) - 0.5 * np.einsum("ps,rq->pqrs", P, P)


def lowdin_factors(S, min_eigenvalue=1e-8):
    S = real_array(S, "overlap")
    require(S.ndim == 2 and S.shape[0] == S.shape[1], "Overlap must be square")
    require(maxabs(S - S.T) < 1e-12, "Overlap is not symmetric")
    values, U = np.linalg.eigh(S)
    require(values.min() > min_eigenvalue,
            f"Near-linear AO dependency: minimum overlap eigenvalue {values.min():.3e}")
    return (U * np.sqrt(values)) @ U.T, (U / np.sqrt(values)) @ U.T, values


def transform_rank4(tensor, A):
    """Contravariant density transform, performed in the full AO space."""
    return np.einsum("ap,bq,cr,ds,pqrs->abcd", A, A, A, A, tensor, optimize=True)


def explicit_rank4(tensor, A):
    """Independent explicit output-index loops (small molecules only)."""
    n = A.shape[0]
    result = np.empty((n, n, n, n))
    for a, b, c, d in np.ndindex(result.shape):
        weights = (A[a, :, None, None, None] * A[b, None, :, None, None]
                   * A[c, None, None, :, None] * A[d, None, None, None, :])
        result[a, b, c, d] = np.sum(weights * tensor)
    return result


def select_valence_aos(mol):
    """Preserve the uniform legacy highest-principal-shell/chemical-core rule.

    Retain H 1s, second-period 2s/2p, and third-period 3s/3p. Sulfur therefore
    loses 1s/2s/2p by the same rule, not by element-specific special handling.
    This historical descriptor selection does not freeze occupied MP2 MOs.
    The current 1s-only descriptor selector is in codes/descriptor.py.
    """
    require(str(mol.basis).lower().replace(" ", "") == "sto-3g", "Valence rule requires STO-3G")
    require(not mol.cart and not mol.has_ecp(), "Requires spherical all-electron STO-3G")
    labels = mol.ao_labels()
    structured = mol.ao_labels(fmt=False)
    ao_loc = mol.ao_loc_nr()
    core, valence, shell_records = [], [], []
    l_letters = "spdfgh"
    for shell in range(mol.nbas):
        atom, ell = mol.bas_atom(shell), mol.bas_angular(shell)
        Z = mol.atom_charge(atom)
        require(Z == 1 or 3 <= Z <= 18, f"Unsupported atom for valence rule: Z={Z}")
        top = 1 if Z == 1 else 2 if Z <= 10 else 3
        for i in range(int(ao_loc[shell]), int(ao_loc[shell + 1])):
            atom_label, symbol, nl, component = structured[i]
            match = re.fullmatch(r"(\d+)([spdfgh])", nl.strip())
            require(match is not None, f"Unrecognized AO principal shell: {structured[i]}")
            principal = int(match[1])
            require(atom_label == atom and symbol == mol.atom_symbol(atom)
                    and match[2] == l_letters[ell], f"AO/basis shell mismatch at {i}")
            require(principal <= top, f"Unexpected higher shell in STO-3G: {labels[i]}")
            (valence if principal == top else core).append(i)
            shell_records.append({"ao_index": i, "atom_index": atom, "element": symbol,
                                  "shell_index": shell, "principal_shell": principal,
                                  "angular_momentum": ell, "component": component,
                                  "n_contractions": mol.bas_nctr(shell),
                                  "n_primitives": mol.bas_nprim(shell)})
    expected_valence = sum(1 if z == 1 else 4 for z in mol.atom_charges())
    require(len(valence) == expected_valence, "Unexpected STO-3G valence shell dimensions")
    return {"ao_labels": np.asarray(labels), "core_indices": np.asarray(core, dtype=int),
            "valence_indices": np.asarray(valence, dtype=int),
            "core_labels": np.asarray([labels[i] for i in core]),
            "valence_labels": np.asarray([labels[i] for i in valence]),
            "ao_shell_metadata_json": np.asarray(json.dumps(shell_records, sort_keys=True))}


def extract_descriptors(P_L, F_L, Lambda_L, valence_indices):
    indices = np.asarray(valence_indices, dtype=int)
    n = len(indices)
    P = P_L[np.ix_(indices, indices)]
    F = F_L[np.ix_(indices, indices)]
    L = Lambda_L[np.ix_(indices, indices, indices, indices)]
    pairs = np.asarray(list(itertools.combinations(range(n), 2)), dtype=int).reshape(-1, 2)
    triples = np.asarray(list(itertools.combinations(range(n), 3)), dtype=int).reshape(-1, 3)
    # Raw[p,q,r,s] = Methods[p,r,q,s]; the final index is retained, not summed.
    lambda_methods = L.transpose(0, 2, 1, 3)
    T = np.diagonal(lambda_methods, axis1=2, axis2=3).copy()
    return {"P_mu": np.diag(P).copy(), "F_mu": np.diag(F).copy(),
            "pair_indices": pairs, "P_munu": P[pairs[:, 0], pairs[:, 1]],
            "T_full": T, "triple_indices": triples,
            "T_munulambda": T[triples[:, 0], triples[:, 1], triples[:, 2]]}


def validate_rdms(P, S, Gamma, P_mp2, nelectron, *, mo_coeff=None, mo_occ=None, pyscf_audit=False):
    P, S = real_array(P, "P_AO"), real_array(S, "S_AO")
    Gamma, D = real_array(Gamma, "Gamma_MP2_AO"), real_array(P_mp2, "P_MP2_AO")
    n = P.shape[0]
    require(P.shape == S.shape == D.shape == (n, n), "1-RDM/overlap shape mismatch")
    require(Gamma.shape == (n,) * 4, "2-RDM shape mismatch")
    d = {"checks": {}, "rdm_definition": RDM_DEFINITION, "rdm_convention": RDM_CONVENTION}
    contraction = np.einsum("pqrs,rs->pq", Gamma, S)
    zero = disconnected_rhf(P)
    cumulant = Gamma - zero
    check(d, "rhf_electron_number_error", abs(np.einsum("pq,qp", P, S) - nelectron), TOL["matrix"])
    check(d, "mp2_electron_number_error", abs(np.einsum("pq,qp", D, S) - nelectron), TOL["matrix"])
    check(d, "rhf_idempotency_ao_max_error", maxabs(P @ S @ P - 2 * P), TOL["matrix"])
    check(d, "rdm2_contraction_max_error", maxabs(contraction - (nelectron - 1) * D.T), TOL["rdm"])
    check(d, "rdm2_particle_number_error", abs(np.einsum("pq,qp", contraction, S)
                                              - nelectron * (nelectron - 1)), TOL["rdm"])
    check(d, "rhf_rdm2_contraction_max_error",
          maxabs(np.einsum("pqrs,rs->pq", zero, S) - (nelectron - 1) * P.T), TOL["rdm"])
    check(d, "cumulant_contraction_max_error",
          maxabs(np.einsum("pqrs,rs->pq", cumulant, S) - (nelectron - 1) * (D - P).T), TOL["rdm"])
    if pyscf_audit and mo_coeff is not None:
        C = real_array(mo_coeff, "mo_coeff")
        O = np.diag(np.asarray(mo_occ) / 2)
        delta = C.T @ S @ (D - P) @ S @ C
        expected_defect = C @ (delta - O @ delta - delta @ O) @ C.T
        # This is diagnostic of PySCF's truncated density; it cannot replace the physical tests.
        check(d, "pyscf_truncation_identity_max_error",
              maxabs(contraction - (nelectron - 1) * D.T - expected_defect), TOL["rdm"])
        d["pyscf_expected_contraction_defect_max"] = maxabs(expected_defect)
    return finish_checks(d)


def make_molecule(record):
    from pyscf import gto
    require(record.spin == 0 and record.electron_count % 2 == 0,
            "System is not closed-shell: RHF/RMP2 calculation stopped")
    return gto.M(atom=list(zip(record.symbols, record.coordinates.tolist())),
                 unit="Angstrom", basis=record.basis, charge=record.charge,
                 spin=record.spin, cart=False, symmetry=False, verbose=0)


def calculate_rhf_rmp2(record, mol):
    from pyscf import scf
    from pyscf.mp import mp2
    with np.load(record.state_path, allow_pickle=False) as saved:
        dm0 = saved["rhf_density_ao"].copy()
    mf = scf.RHF(mol)
    lowdin_factors(mf.get_ovlp(), TOL["overlap_min"])
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, TOL["gradient"], 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check = 400, 12, False
    mf.direct_scf, mf.chkfile = False, None
    mf.kernel(dm0=dm0)
    require(mf.converged, "RHF did not converge")
    gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    require(gradient <= TOL["gradient"], f"RHF orbital gradient too large: {gradient:.3e}")
    require(abs(mf.e_tot - record.E_RHF_input) <= TOL["energy_Ha"],
            f"RHF energy mismatch: {mf.e_tot - record.E_RHF_input:.3e} Ha")
    require(np.all(np.isin(mf.mo_occ, [0, 2])), "RHF occupations are not closed shell")
    require(np.all(np.diff(mf.mo_occ) <= 0), "RHF occupied MOs are not first")
    mf.mo_energy, mf.mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    require(np.all(np.diff(mf.mo_energy[mf.mo_occ == 2]) >= -1e-10), "Occupied MOs are not energy ordered")
    # All occupied and virtual MOs participate; target FCI freezing is label
    # metadata only and never enters the descriptor calculation.
    occupied = mf.mo_energy[:mol.nelectron // 2]
    virtual = mf.mo_energy[mol.nelectron // 2:]
    gap = float(np.min(virtual) - np.max(occupied))
    require(gap > 1e-10, f"MP2 denominator near zero or inverted gap: {gap:.3e}")
    pt = mp2.RMP2(mf, frozen=0)
    e_mp2, t2 = pt.kernel()
    require(np.isfinite(e_mp2) and np.isfinite(t2).all(), "Nonfinite RMP2 result")
    P, S = mf.make_rdm1(), mf.get_ovlp()
    # Full AO representation with every electron correlated, no core embedding.
    Gamma = pt.make_rdm2(ao_repr=True)
    D = pt.make_rdm1(ao_repr=True, with_frozen=True)
    # Direct library zero-amplitude limit, without another electronic calculation.
    Gamma_no_correlation = pt.make_rdm2(t2=np.zeros_like(t2), ao_repr=True)
    return {"S_AO": S, "P_AO": P, "F_AO": mf.get_fock(dm=P),
            "Gamma_MP2_AO": Gamma, "P_MP2_AO": D,
            "Gamma_MP2_MO_pyscf": pt.make_rdm2(ao_repr=False),
            "P_MP2_MO_pyscf": pt.make_rdm1(ao_repr=False, with_frozen=True),
            "hcore_AO": mf.get_hcore(), "eri_AO": mol.intor("int2e", aosym="s1"),
            "E_nuclear": np.asarray(mol.energy_nuc()),
            "E_RHF_check": np.asarray(mf.e_tot), "E_MP2_correlation": np.asarray(e_mp2),
            "mo_coeff": mf.mo_coeff, "mo_occ": mf.mo_occ, "mo_energy": mf.mo_energy,
            "mp2_t2": t2}, {"rhf_converged": bool(mf.converged),
            "rhf_gradient_norm": gradient, "mp2_convergence_status": "canonical noniterative RMP2 completed",
            "mp2_min_occupied_virtual_gap_Ha": gap,
            "zero_t2_cumulant_max_error": maxabs(Gamma_no_correlation - disconnected_rhf(P))}


def record_identity(record):
    return {"molecule": record.molecule, "geometry_id": record.geometry_id,
            "point_index": record.point_index, "input_geometry_path": str(record.geometry_path),
            "input_state_path": str(record.state_path), "scan_coordinate": record.q_A,
            "q_A": record.q_A, "bond_length_A": record.bond_length_A,
            "units": "Angstrom", "energy_units": "Hartree", "charge": record.charge,
            "spin": record.spin, "basis": record.basis, "electron_count": record.electron_count,
            "frozen_core": 0, "mp2_frozen_core": 0, "rdm_frozen_core": 0,
            "frozen_core_setting": "none; all electrons and all STO-3G MOs participate",
            "input_fci_frozen_core": record.input_fci_frozen_core,
            "input_fci_convention": "frozen-core target energies retained unchanged from initialdata",
            "label_descriptor_correlation_spaces_match": record.input_fci_frozen_core == 0,
            "descriptor_selection_stage": "valence AOs selected only after full-space RDM and full-AO Lowdin transformation",
            "E_RHF_input": record.E_RHF_input, "E_FCI": record.E_FCI,
            "E_corr_input": record.E_corr_input,
            "E_FCI_minus_E_RHF_input": record.E_FCI - record.E_RHF_input,
            "symbols": np.asarray(record.symbols), "cartesian_A": record.coordinates,
            "schema_version": SCHEMA_VERSION, "rdm_definition": RDM_DEFINITION,
            "rdm_convention": RDM_CONVENTION}


def add_consistent_rdms(arrays, record):
    """Keep both tracks, but make legacy tensor names explicit formal aliases."""
    C, S = arrays["mo_coeff"], arrays["S_AO"]
    arrays["P_MP2_AO_pyscf"] = arrays["P_MP2_AO"]
    arrays["Gamma_MP2_AO_pyscf"] = arrays["Gamma_MP2_AO"]
    consistent, metadata = build_consistent_rdms(
        arrays["mp2_t2"], C.shape[1], record.electron_count, frozen_core=0)
    arrays.update(consistent)
    for order in range(3):
        arrays[f"P_AO_order{order}"] = C @ arrays[f"P_MO_order{order}"] @ C.T
        arrays[f"Gamma_AO_order{order}"] = transform_rank4(arrays[f"Gamma_MO_order{order}"], C)
    arrays["P_MP2_AO_consistent"] = C @ arrays["P_MP2_MO_consistent"] @ C.T
    arrays["Gamma_MP2_AO_consistent"] = transform_rank4(arrays["Gamma_MP2_MO_consistent"], C)
    arrays["P_MP2_AO"] = arrays["P_MP2_AO_consistent"]
    arrays["Gamma_MP2_AO"] = arrays["Gamma_MP2_AO_consistent"]
    arrays["Gamma0_HF_AO"] = disconnected_rhf(arrays["P_AO"])
    arrays["Lambda_HFref_AO"] = arrays["Gamma_MP2_AO_consistent"] - arrays["Gamma0_HF_AO"]
    arrays["Gamma0_correlated_AO"] = disconnected_rhf(arrays["P_MP2_AO_consistent"])
    arrays["Lambda_conventional_AO"] = arrays["Gamma_MP2_AO_consistent"] - arrays["Gamma0_correlated_AO"]
    metadata.update(
        index_ordering_MO="P[p,q]=sum_spin <q^dagger p>; Gamma[p,q,r,s]=sum_spin <p^dagger r^dagger s q>",
        index_ordering_AO=RDM_CONVENTION,
        MO_to_AO="P_AO=C P_MO C.T; Gamma_AO[p,q,r,s]=sum_ijkl C[p,i] C[q,j] C[r,k] C[s,l] Gamma_MO[i,j,k,l]; all orbitals real",
        contraction_AO="sum_rs Gamma_AO[p,q,r,s]*S[r,s]=(N-1)*P_AO[q,p]",
        formal_cumulant="Lambda_HFref_AO=Gamma_MP2_AO_consistent-Gamma0_HF_AO",
        diagnostic_cumulant="Lambda_conventional_AO=Gamma_MP2_AO_consistent-Gamma0[P_MP2_AO_consistent]",
        formal_cumulant_contraction="(N-1)*(P_MP2_AO_consistent-P_AO).T",
        conventional_cumulant_contraction="(0.5*P_MP2_AO_consistent@S_AO@P_MP2_AO_consistent-P_MP2_AO_consistent).T",
        alias_definitions="Gamma_MP2_AO and P_MP2_AO alias consistent RDMs; Gamma0_AO aliases Gamma0_HF_AO; Lambda_AO aliases Lambda_HFref_AO",
        calculation_space="all electrons; all occupied and virtual STO-3G MOs; no frozen orbitals; no AO removal before descriptor extraction",
        input_fci_frozen_core=record.input_fci_frozen_core,
        input_fci_used_only_as_label=True,
        pyscf_audit_source=PYSCF_RDM_SOURCE)
    arrays["rdm_metadata_json"] = np.asarray(json.dumps(metadata))
    return metadata


def energy_audit(arrays):
    """Energy contractions; no target FCI energy is calculated here.

    E^[2]=E_HF+<H>_R1+Tr(epsilon P2). Contracting the full physical H
    with R0+R1+R2 also includes <V>_R2, formally third order in MP counting.
    It is recorded separately and need not equal E_HF+E_MP2.
    """
    h, eri = arrays["hcore_AO"], arrays["eri_AO"]
    def electronic(D, G):
        return float(np.einsum("pq,qp", h, D) + .5*np.einsum("pqrs,pqrs", eri, G))
    nuclear = float(arrays["E_nuclear"])
    energies = {}
    for track in ("consistent", "pyscf"):
        try:
            D, G = arrays[f"P_MP2_AO_{track}"], arrays[f"Gamma_MP2_AO_{track}"]
            require(np.isrealobj(D) and np.isrealobj(G) and np.isfinite(D).all() and np.isfinite(G).all(),
                    "Invalid density in energy contraction")
            energies[track + "_full_H_expectation_Ha"] = electronic(D, G) + nuclear
        except (ValueError, TypeError) as exc:
            if track == "consistent":
                raise
            energies["pyscf_full_H_expectation_Ha"] = None
            energies["pyscf_error"] = str(exc)
    linear = electronic(arrays["P_AO_order1"], arrays["Gamma_AO_order1"])
    h0_quadratic = float(np.einsum("p,pp", arrays["mo_energy"], arrays["P_MO_order2"]))
    correlation = float(arrays["E_MP2_correlation"])
    energies.update(canonical_RMP2_correlation_Ha=correlation,
                    canonical_RMP2_total_Ha=float(arrays["E_RHF_check"]) + correlation,
                    H_on_R1_Ha=linear, H0_on_R2_Ha=h0_quadratic,
                    consistent_second_order_correlation_Ha=linear+h0_quadratic,
                    consistent_second_order_total_Ha=float(arrays["E_RHF_check"]) + linear+h0_quadratic,
                    reference_from_R0_Ha=electronic(arrays["P_AO_order0"], arrays["Gamma_AO_order0"])+nuclear,
                    full_H_difference_Ha=(energies["consistent_full_H_expectation_Ha"]-energies["pyscf_full_H_expectation_Ha"]
                                          if energies["pyscf_full_H_expectation_Ha"] is not None else None))
    energies["interpretation"] = "Full-H expectations are direct contractions of the named density. Formal MP2 energy retains H:R1 plus H0:R2; V:R2 is higher order. No FCI energy recomputation."
    return energies


def validate_record(arrays, record, independent=False):
    for key, value in arrays.items():
        a = np.asarray(value)
        if a.dtype.kind in "fci" and "pyscf" not in key:
            real_array(a, key)
    P, S, Gamma = arrays["P_AO"], arrays["S_AO"], arrays["Gamma_MP2_AO"]
    d = validate_rdms(P, S, Gamma, arrays["P_MP2_AO"], record.electron_count,
                      mo_coeff=arrays["mo_coeff"], mo_occ=arrays["mo_occ"])
    dual, contraction_terms = dual_rdm_diagnostics(arrays, record.electron_count, ncore=0)
    d["checks"].update(dual.pop("checks"))
    d["dual_rdm"] = dual
    arrays.update(contraction_terms)
    metadata = json.loads(str(arrays["rdm_metadata_json"]))
    d["rdm_construction"] = metadata
    nocc, nvir = record.electron_count//2, arrays["mo_coeff"].shape[1]-record.electron_count//2
    require(arrays["mp2_t2"].shape == (nocc,nocc,nvir,nvir), "MP2 amplitudes must include every occupied and virtual MO")
    require(metadata["frozen_core"] == 0, "Frozen orbitals are forbidden before descriptor selection")
    for key in ("frozen_core", "mp2_frozen_core", "rdm_frozen_core"):
        require(int(arrays[key]) == 0, f"{key} must be zero")
    d["calculation_space"] = {"full_occupied_MOs": nocc, "full_virtual_MOs": nvir,
                              "frozen_MOs": 0, "input_fci_frozen_core": record.input_fci_frozen_core,
                              "input_fci_is_label_only": True}
    for alias, official in (("P_MP2_AO", "P_MP2_AO_consistent"), ("Gamma_MP2_AO", "Gamma_MP2_AO_consistent"),
                            ("Gamma0_AO", "Gamma0_HF_AO"), ("Lambda_AO", "Lambda_HFref_AO")):
        require(arrays[alias].shape == arrays[official].shape, f"Formal alias shape mismatch: {alias}")
        check(d, alias + "_formal_alias_error", maxabs(arrays[alias]-arrays[official]), TOL["rdm"])
    check(d, "HFref_formula_error", maxabs(arrays["Lambda_HFref_AO"]-(arrays["Gamma_MP2_AO_consistent"]-disconnected_rhf(P))), TOL["rdm"])
    check(d, "conventional_formula_error", maxabs(arrays["Lambda_conventional_AO"]-(arrays["Gamma_MP2_AO_consistent"]-disconnected_rhf(arrays["P_MP2_AO_consistent"]))), TOL["rdm"])
    check(d, "MO_orthonormality", maxabs(arrays["mo_coeff"].T@S@arrays["mo_coeff"]-np.eye(S.shape[0])), TOL["matrix"])
    check(d, "wavefunction_norm_eta1", abs(float(np.vdot(arrays["ci_normalized_eta1"], arrays["ci_normalized_eta1"]))-1), TOL["rdm"])
    for order in range(3):
        check(d, f"wavefunction_norm_order{order}", abs(float(arrays[f"wavefunction_norm_order{order}"])-float(order == 0)), TOL["rdm"])
    check(d, "reference_P_matches_RHF", maxabs(arrays["P_AO_order0"]-P), TOL["rdm"])
    check(d, "zero_t2_cumulant_max_error", maxabs(arrays["Gamma_AO_order0"]-disconnected_rhf(P)), TOL["rdm"])
    # Conventions are source-derived; this identity is diagnostic for the audit only.
    try:
        d["pyscf_standard_audit"] = validate_rdms(P, S, arrays["Gamma_MP2_AO_pyscf"], arrays["P_MP2_AO_pyscf"], record.electron_count,
                                                 mo_coeff=arrays["mo_coeff"], mo_occ=arrays["mo_occ"], pyscf_audit=True)
    except (ValueError, TypeError) as exc:
        d["pyscf_standard_audit"] = {"passed": False, "error": str(exc)}
    d["pyscf_standard_audit"]["rdm_definition"] = "PySCF standard unrelaxed MP2 energy-derivative density; audit only"
    energies = energy_audit(arrays)
    d["energy_audit"] = energies
    check(d, "R0_energy_matches_RHF_Ha", abs(energies["reference_from_R0_Ha"]-float(arrays["E_RHF_check"])), TOL["energy_Ha"])
    check(d, "consistent_second_order_energy_error_Ha", abs(energies["consistent_second_order_correlation_Ha"]-float(arrays["E_MP2_correlation"])), TOL["energy_Ha"])
    h, inv, eigenvalues = lowdin_factors(S, TOL["overlap_min"])
    n = S.shape[0]
    v = arrays["valence_indices"]
    nv = len(v)
    shapes = {"S_AO": (n,n), "F_AO": (n,n), "S_half": (n,n), "S_minus_half": (n,n),
              "P_L": (n,n), "F_L": (n,n), "Gamma0_AO": (n,)*4, "Lambda_AO": (n,)*4,
              "Lambda_L": (n,)*4, "P_mu": (nv,), "F_mu": (nv,), "P_munu": (math.comb(nv,2),),
              "T_full": (nv,)*3, "T_munulambda": (math.comb(nv,3),),
              "pair_indices": (math.comb(nv,2),2), "triple_indices": (math.comb(nv,3),3)}
    for key, shape in shapes.items():
        require(arrays[key].shape == shape, f"{key} shape mismatch: {arrays[key].shape} != {shape}")
    comparisons = {
        "lowdin_metric_max_error": inv @ S @ inv - np.eye(n),
        "S_half_max_error": arrays["S_half"] - h,
        "S_minus_half_max_error": arrays["S_minus_half"] - inv,
        "P_L_transform_max_error": arrays["P_L"] - h @ P @ h,
        "F_L_transform_max_error": arrays["F_L"] - inv @ arrays["F_AO"] @ inv,
        "rhf_idempotency_lowdin_max_error": arrays["P_L"] @ arrays["P_L"] - 2 * arrays["P_L"],
        "Gamma0_max_error": arrays["Gamma0_AO"] - disconnected_rhf(P),
        "Lambda_AO_max_error": arrays["Lambda_AO"] - (Gamma - arrays["Gamma0_AO"]),
        "Lambda_L_transform_max_error": arrays["Lambda_L"] - transform_rank4(arrays["Lambda_AO"], h)}
    for key, residual in comparisons.items():
        check(d, key, maxabs(residual), TOL["matrix"])
    check(d, "lowdin_electron_number_error", abs(np.trace(arrays["P_L"]) - record.electron_count), TOL["matrix"])
    check(d, "cumulant_lowdin_contraction_max_error",
          maxabs(np.einsum("abcc->ab", arrays["Lambda_L"])
                 - (record.electron_count - 1) * (h @ arrays["P_MP2_AO"] @ h - arrays["P_L"]).T), TOL["rdm"])
    check(d, "rhf_energy_error_Ha", abs(float(arrays["E_RHF_check"]) - record.E_RHF_input), TOL["energy_Ha"])
    check(d, "input_correlation_error_Ha", abs(record.E_corr_input - record.E_FCI + record.E_RHF_input), 1e-12)
    expected = extract_descriptors(arrays["P_L"], arrays["F_L"], arrays["Lambda_L"], v)
    for key, values in expected.items():
        check(d, key + "_extraction_max_error", maxabs(arrays[key] - values), TOL["matrix"])
    # Independent indexing check includes all repeated lambda positions and signs.
    explicit_T = np.empty((nv, nv, nv))
    for mu, nu, lam in np.ndindex(explicit_T.shape):
        explicit_T[mu, nu, lam] = arrays["Lambda_L"][v[mu], v[lam], v[nu], v[lam]]
    check(d, "T_full_explicit_index_max_error", maxabs(arrays["T_full"] - explicit_T), 0.0)
    if independent:
        check(d, "rank4_explicit_loop_max_error",
              maxabs(explicit_rank4(arrays["Lambda_AO"], h) - arrays["Lambda_L"]), TOL["matrix"])
    d["overlap_eigenvalue_min"], d["overlap_eigenvalue_max"] = float(eigenvalues.min()), float(eigenvalues.max())
    d["tensor_shapes"] = {key: list(shape) for key, shape in shapes.items()}
    d["tensor_shapes"]["Gamma_MP2_AO"] = list(Gamma.shape)
    d["descriptor_dimensions"] = {"one_body": nv, "pair": math.comb(nv,2), "triple": math.comb(nv,3)}
    return finish_checks(d)


def calculate_record(record, ao_metadata, independent=False):
    mol = make_molecule(record)
    arrays, convergence = calculate_rhf_rmp2(record, mol)
    convergence["pyscf_zero_t2_cumulant_max_error"] = convergence.pop("zero_t2_cumulant_max_error")
    add_consistent_rdms(arrays, record)
    arrays.update(record_identity(record))
    import pyscf
    arrays["pyscf_version"] = pyscf.__version__
    arrays["rdm_source"] = RDM_SOURCE
    arrays.update(ao_metadata)
    h, inv, _ = lowdin_factors(arrays["S_AO"], TOL["overlap_min"])
    arrays["Gamma0_AO"] = arrays["Gamma0_HF_AO"]
    arrays["Lambda_AO"] = arrays["Lambda_HFref_AO"]
    arrays.update(S_half=h, S_minus_half=inv, P_L=h @ arrays["P_AO"] @ h,
                  F_L=inv @ arrays["F_AO"] @ inv,
                  Lambda_L=transform_rank4(arrays["Lambda_HFref_AO"], h))
    arrays.update(extract_descriptors(arrays["P_L"], arrays["F_L"], arrays["Lambda_L"], arrays["valence_indices"]))
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    diagnostics = validate_record(arrays, record, independent=independent)
    diagnostics.update(convergence)
    finish_checks(diagnostics)
    arrays["energy_audit_json"] = np.asarray(json.dumps(diagnostics["energy_audit"], allow_nan=False))
    arrays["E_MP2_consistent_second_order"] = np.asarray(diagnostics["energy_audit"]["consistent_second_order_total_Ha"])
    arrays["rhf_converged"] = np.asarray(convergence["rhf_converged"])
    arrays["mp2_convergence_status"] = np.asarray(convergence["mp2_convergence_status"])
    return arrays, diagnostics


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(document), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_record(output, record, arrays, diagnostics):
    base = output / record.molecule / record.geometry_id
    base.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if diagnostics["passed"] else "FAIL"
    # Explicit failed retries preserve the previous evidence rather than replacing it.
    if base.with_suffix(".json").exists() or base.with_suffix(".npz").exists():
        revision = 1
        while base.with_suffix(f".attempt{revision}.json").exists() or base.with_suffix(f".attempt{revision}.npz").exists():
            revision += 1
        for extension in (".json", ".npz"):
            if base.with_suffix(extension).exists():
                base.with_suffix(extension).replace(base.with_suffix(f".attempt{revision}{extension}"))
    if arrays is not None:
        arrays["status"] = np.asarray(status)
        arrays["validation_json"] = np.asarray(json.dumps(jsonable(diagnostics), allow_nan=False))
        with base.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        base.with_suffix(".npz.tmp").replace(base.with_suffix(".npz"))
    write_json(base.with_suffix(".json"), {"molecule": record.molecule, "geometry_id": record.geometry_id,
               "status": status, "validation": diagnostics})
    return {"molecule": record.molecule, "geometry_id": record.geometry_id, "status": status,
            "q_A": record.q_A, "record": str(base.with_suffix(".npz")) if arrays is not None else None,
            "validation": str(base.with_suffix(".json")),
            "error": diagnostics.get("error", "; ".join(diagnostics.get("failed_checks", []))),
            "E_RHF_error_Ha": diagnostics.get("checks", {}).get("rhf_energy_error_Ha", {}).get("value")}


def print_failed_contraction(record, arrays, diagnostics):
    """Print a reproducer and all formal contraction terms on physical failure."""
    report = {"molecule": record.molecule, "geometry_id": record.geometry_id,
              "geometry_path": str(record.geometry_path), "symbols": record.symbols,
              "cartesian_A": record.coordinates, "basis": record.basis,
              "charge": record.charge, "spin": record.spin, "frozen_core": 0,
              "input_fci_frozen_core": record.input_fci_frozen_core,
              "formula": "sum_rs Gamma[p,q,r,s]*S[r,s] = (N-1)*P[q,p]",
              "index_convention": RDM_CONVENTION,
              "failed_checks": diagnostics.get("failed_checks", []),
              "error": diagnostics.get("error")}
    if arrays is not None:
        report["mp2_t2"] = arrays["mp2_t2"]
        report["rdm_metadata"] = json.loads(str(arrays["rdm_metadata_json"]))
        report["contraction_terms"] = {key: value for key,value in arrays.items()
                                       if "contraction" in key and "pyscf" not in key}
    print(json.dumps(jsonable(report), indent=2, allow_nan=False), flush=True)


def read_success(output, record, ao_metadata, independent=False):
    path = output / record.molecule / f"{record.geometry_id}.npz"
    require(path.is_file(), "Missing saved NPZ")
    sidecar = json.loads(path.with_suffix(".json").read_text())
    require(sidecar["status"] == "PASS", "Saved record failed validation; use --rerun-failed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    require(str(arrays["status"]) == "PASS", "Saved NPZ is not successful")
    for key, value in record_identity(record).items():
        require(np.array_equal(arrays[key], np.asarray(value)), f"Saved input metadata mismatch: {key}")
    for key, value in ao_metadata.items():
        require(np.array_equal(arrays[key], value), f"Saved AO metadata mismatch: {key}")
    require(bool(arrays["rhf_converged"]), "Saved RHF was not converged")
    old_diagnostics = json.loads(str(arrays["validation_json"]))
    require(old_diagnostics["passed"], "Saved validation status is not successful")
    diagnostics = validate_record(arrays, record, independent=independent)
    check(diagnostics, "zero_t2_cumulant_max_error",
          old_diagnostics["checks"]["zero_t2_cumulant_max_error"]["value"], TOL["rdm"])
    diagnostics.update({k: v for k, v in old_diagnostics.items()
                        if k not in ("checks", "passed", "failed_checks") and k not in diagnostics})
    finish_checks(diagnostics)
    require(diagnostics["passed"], f"Saved tensor validation failed: {diagnostics['failed_checks']}")
    return arrays, diagnostics


def validate_output_path(input_dir, output_dir):
    source, output = input_dir.resolve(), output_dir.resolve()
    require(source != output and source not in output.parents and output not in source.parents,
            "Output and original input directories must be separate, non-nested directories")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "results/initialdata")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/methods41_fullspace")
    parser.add_argument("--molecules", nargs="+", choices=MOLECULES, default=list(MOLECULES))
    parser.add_argument("--geometry-id", help="One scan ID (requires exactly one molecule)")
    parser.add_argument("--smoke-test", action="store_true", help="H2O scan 001 only, with explicit rank-4 verification")
    parser.add_argument("--dry-run", action="store_true", help="Read-only data/AO audit; no RHF/MP2 calculation or output writes")
    parser.add_argument("--resume", action="store_true", help="Revalidate and skip successful records")
    parser.add_argument("--rerun-failed", action="store_true", help="Retry failed/invalid outputs (successful records are kept)")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.molecules, args.geometry_id = ["H2O"], "001"
    if args.geometry_id:
        if len(args.molecules) != 1:
            parser.error("--geometry-id requires exactly one molecule")
        if not args.geometry_id.isdigit() or not 1 <= int(args.geometry_id) <= 30:
            parser.error("--geometry-id must be between 001 and 030")
        args.geometry_id = f"{int(args.geometry_id):03d}"
    if args.threads < 1:
        parser.error("--threads must be positive")
    if len(set(args.molecules)) != len(args.molecules):
        parser.error("Duplicate molecule selections are not allowed")
    return args


def run(args):
    import pyscf
    from pyscf import lib
    validate_output_path(args.input_dir, args.output_dir)
    # The query returns 1 even in this build's no-OpenMP fallback; the setter's
    # return value is the capability probe. Suppress only its known warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"OpenMP is not available\. Setting omp_threads to .* has no effects\.",
                                category=UserWarning, module=r"pyscf\.lib\.misc")
        openmp_available = lib.num_threads(args.threads) > 0
    if not openmp_available and args.threads != 1:
        print("OpenMP is unavailable; the requested OpenMP thread count cannot be applied.", flush=True)
    records, audit = discover_dataset(args.input_dir, molecules=args.molecules)
    # Build AO metadata at every geometry, without any electronic calculation.
    ao_metadata, ao_errors = {}, {}
    for name in args.molecules:
        reference = None
        for record in (r for r in records if r.molecule == name):
            try:
                metadata = select_valence_aos(make_molecule(record))
                if reference is not None:
                    require(all(np.array_equal(metadata[k], reference[k]) for k in metadata),
                            "AO ordering or core/valence selection changes within molecule")
                reference = metadata
                ao_metadata[(name, record.geometry_id)] = metadata
            except Exception as exc:
                ao_errors[name] = str(exc)
                break
        if reference is not None and name not in ao_errors:
            nv = len(reference["valence_indices"])
            audit["molecules"][name]["descriptor_dimensions"] = {
                "one_body": nv, "pair": math.comb(nv, 2), "triple": math.comb(nv, 3)}
            audit["molecules"][name]["core_ao_indices"] = reference["core_indices"].tolist()
            audit["molecules"][name]["valence_ao_indices"] = reference["valence_indices"].tolist()
    audit["ao_errors"] = ao_errors
    if args.dry_run:
        print(json.dumps(jsonable(audit), indent=2, allow_nan=False))
        return 0 if audit["status"] == "PASS" and not ao_errors else 1
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.resume or args.rerun_failed):
        raise ValidationError("Output exists; choose a new directory or use --resume / --rerun-failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {"schema_version": SCHEMA_VERSION, "input_dir": str(args.input_dir.resolve()),
                "rdm_definition": RDM_DEFINITION, "rdm_convention": RDM_CONVENTION,
                "rdm_source": RDM_SOURCE, "pyscf_version": pyscf.__version__,
                "calculation_space": "all electrons and all STO-3G MOs; no freezing or AO removal before final descriptor selection",
                "mp2_frozen_core": 0, "rdm_frozen_core": 0,
                "input_fci_policy": "Frozen-core FCI labels retained unchanged; their correlation space differs from the full-space descriptor calculation",
                "openmp_available": openmp_available, "requested_openmp_threads": args.threads,
                "tolerances": TOL, "input_audit": audit, "records": [],
                "execution_order": "H2O/001 smoke gate, then molecule order and numerical geometry ID"}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        require(all(previous[k] == manifest[k] for k in ("schema_version", "input_dir", "rdm_definition", "tolerances")),
                "Output convention/settings differ; choose a new output directory")
        manifest["records"] = previous["records"]
    manifest["input_audits_by_molecule"] = {
        **(previous.get("input_audits_by_molecule", {}) if manifest_path.exists() else {}),
        **audit["molecules"]}
    entries = {(row["molecule"], row["geometry_id"]): row for row in manifest["records"]}
    requested_keys = {(name, args.geometry_id or f"{i:03d}")
                      for name in args.molecules for i in range(1, 31)}

    def checkpoint():
        manifest["records"] = sorted(entries.values(), key=lambda row: (MOLECULES.index(row["molecule"]), int(row["geometry_id"])))
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["successful_geometry_count"] = sum(row["status"] == "PASS" for row in entries.values())
        manifest["failed_geometry_count"] = sum(row["status"] == "FAIL" for row in entries.values())
        manifest["input_excluded_geometry_count"] = audit["failed_geometry_count"]
        manifest["ao_excluded_geometry_count"] = 30 * len(ao_errors)
        manifest["requested_geometry_count"] = len(requested_keys)
        manifest["not_run_requested_geometry_count"] = len(requested_keys - entries.keys())
        manifest["status"] = ("FAIL" if manifest["failed_geometry_count"] or audit["status"] != "PASS" or ao_errors
                              else "INCOMPLETE" if manifest["not_run_requested_geometry_count"] else "PASS")
        manifest["dataset_expected_geometry_count"] = 30 * len(MOLECULES)
        manifest["dataset_not_run_geometry_count"] = 30 * len(MOLECULES) - len(entries)
        manifest["dataset_status"] = ("FAIL" if manifest["status"] == "FAIL" else "PASS"
                                      if manifest["successful_geometry_count"] == 30 * len(MOLECULES) else "PARTIAL")
        write_json(manifest_path, manifest)
        for name in set(args.molecules) | {key[0] for key in entries}:
            folder = args.output_dir / name
            folder.mkdir(exist_ok=True)
            rows = [entries.get((name, f"{i:03d}"), {"molecule": name, "geometry_id": f"{i:03d}", "status": "NOT_RUN"})
                    for i in range(1,31)]
            input_error = ao_errors.get(name) or "; ".join(audit["molecules"].get(name, {}).get("errors", []))
            if input_error:
                for row in rows:
                    if row["status"] == "NOT_RUN":
                        row.update(status="INPUT_FAIL", error=input_error)
            with (folder / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["molecule", "geometry_id", "status", "q_A", "record", "validation", "error", "E_RHF_error_Ha"])
                writer.writeheader()
                writer.writerows(rows)

    def execute(record, independent=False):
        key = (record.molecule, record.geometry_id)
        base = args.output_dir / record.molecule / record.geometry_id
        if base.with_suffix(".npz").exists() or base.with_suffix(".json").exists():
            try:
                arrays, diagnostics = read_success(args.output_dir, record, ao_metadata[key], independent)
                entries[key] = {"molecule": record.molecule, "geometry_id": record.geometry_id,
                                "status": "PASS", "q_A": record.q_A,
                                "record": str(base.with_suffix(".npz")), "validation": str(base.with_suffix(".json")),
                                "error": "", "E_RHF_error_Ha": diagnostics["checks"]["rhf_energy_error_Ha"]["value"]}
                print(f"SKIP {record.molecule}/{record.geometry_id}: saved tensors revalidated", flush=True)
                checkpoint()
                return True
            except Exception as exc:
                if not args.rerun_failed:
                    entries[key] = {"molecule": record.molecule, "geometry_id": record.geometry_id,
                                    "status": "FAIL", "error": f"Resume validation: {exc}"}
                    checkpoint()
                    print(f"FAIL {record.molecule}/{record.geometry_id}: {exc}", flush=True)
                    return False
        arrays = None
        try:
            arrays, diagnostics = calculate_record(record, ao_metadata[key], independent)
        except Exception as exc:
            diagnostics = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        entries[key] = save_record(args.output_dir, record, arrays, diagnostics)
        checkpoint()
        print(f"{entries[key]['status']} {record.molecule}/{record.geometry_id}: {entries[key]['error']}", flush=True)
        if not diagnostics["passed"]:
            print(f"See {base.with_suffix('.json')} for validation details", flush=True)
            print_failed_contraction(record, arrays, diagnostics)
        return diagnostics["passed"]

    checkpoint()
    selected = [r for r in records if r.molecule not in ao_errors and (args.geometry_id is None or r.geometry_id == args.geometry_id)]
    if args.geometry_id is None:
        # The full dataset is never calculated before a successful H2O smoke test.
        smoke = next((r for r in records if r.molecule == "H2O" and r.geometry_id == "001"), None)
        if smoke is None:
            smoke_records, smoke_audit = discover_dataset(args.input_dir, molecules=["H2O"])
            require(smoke_audit["status"] == "PASS", "H2O smoke input audit failed")
            smoke = next(r for r in smoke_records if r.geometry_id == "001")
            ao_metadata[("H2O", "001")] = select_valence_aos(make_molecule(smoke))
        if "H2O" in ao_errors or not execute(smoke, independent=True):
            manifest["stopped_reason"] = "H2O smoke test did not pass; full generation stopped"
            checkpoint()
            return 1
        selected = [r for r in selected if not (r.molecule == "H2O" and r.geometry_id == "001")]
    for record in selected:
        if not execute(record, independent=record.geometry_id == "001" and record.molecule in ("H2O", "LiH")):
            manifest["stopped_reason"] = f"Formal validation failed for {record.molecule}/{record.geometry_id}; batch stopped"
            checkpoint()
            return 1
    checkpoint()
    print(f"Successful: {manifest['successful_geometry_count']}; calculation/record failures: {manifest['failed_geometry_count']}; "
          f"input-excluded: {manifest['input_excluded_geometry_count']}; AO-excluded: {manifest['ao_excluded_geometry_count']}; "
          f"dataset not run: {manifest['dataset_not_run_geometry_count']}", flush=True)
    return 0 if manifest["status"] == "PASS" else 1


def main(argv=None):
    try:
        return run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
