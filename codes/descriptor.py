#!/usr/bin/env python3
"""Descriptor calculation and optional solver-free repair of saved T fields.

All electrons and STO-3G orbitals participate in MP2/RDM construction. Only
non-hydrogen/non-helium 1s AOs are excluded from descriptor extraction after
the full-AO Lowdin transformation.
Normal calculation does not perform scientific acceptance validation.
The explicit --repair-t mode validates saved tensors and replaces only T.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MOLECULES = ("LiH", "HF", "BeH2", "H2O", "H2S", "NH3", "N2", "CO", "H2O2")
SCHEMA_VERSION = "descriptor-v1-calculation-only"
RDM_SOURCE = "normalized truncated MP2 wavefunction, expectation values through second order"
RDM_CONVENTION = "Gamma[p,q,r,s]=sum_spin <p^dagger r^dagger s q>; P[p,q]=sum_spin <q^dagger p>"
T_AXIS_MAPPING = "T[i,j,k] = Lambda_methods[i,j,k,k] = Lambda_raw[i,k,j,k]"
AO_SELECTION_RULE = "post-lowdin-exclude-Z-gt-2-principal-1-only-v1"


def load_records(input_dir, molecules):
    """Read each scan point directly, retaining the source coordinate precision."""
    records = []
    for name in molecules:
        paths = sorted((input_dir / name / "scan").glob("*/point.json"), key=lambda p: int(p.parent.name))
        for path in paths:
            point = json.loads(path.read_text())
            records.append({**point, "geometry_id": path.parent.name,
                            "input_geometry_path": str(path.with_name("geometry.xyz").resolve()),
                            "input_state_path": str(path.with_name("electronic_state.npz").resolve())})
    return records


def disconnected_rhf(P):
    return np.einsum("pq,rs->pqrs", P, P) - .5*np.einsum("ps,rq->pqrs", P, P)


def lowdin_factors(S):
    eigenvalues, U = np.linalg.eigh(S)
    return (U*np.sqrt(eigenvalues))@U.T, (U/np.sqrt(eigenvalues))@U.T


def transform_rank4(tensor, A):
    return np.einsum("ap,bq,cr,ds,pqrs->abcd", A, A, A, A, tensor, optimize=True)


def build_consistent_rdms(t2, nmo, nelectron):
    """R[2] = R0 + R1 + R_chichi - w R0, for both RDMs, without checks.

    Psi(eta)=(Phi+eta*chi)/sqrt(1+eta**2*w), w=<chi|chi>.
    Raw RMP2 t2 maps directly to restricted CISD c2. Only determinant
    conversion and density algebra are used; no FCI energy solver is run.
    """
    from pyscf.ci import cisd
    from pyscf.fci import direct_spin1
    nocc = nelectron//2
    singles = np.zeros((nocc, nmo-nocc))
    def state(c0, doubles):
        return np.asarray(cisd.to_fcivec(cisd.amplitudes_to_cisdvec(c0, singles, doubles),
                                        nmo, nelectron, frozen=0))
    reference = state(1., np.zeros_like(t2))
    chi = state(0., t2)
    w = float(np.vdot(chi, chi).real)
    p0, g0 = direct_spin1.make_rdm12(reference, nmo, nelectron, reorder=True)
    pc, gc = direct_spin1.make_rdm12(chi, nmo, nelectron, reorder=True)
    pr, gr = direct_spin1.make_rdm12(reference+chi, nmo, nelectron, reorder=True)
    p1, g1 = pr-p0-pc, gr-g0-gc
    p2, g2 = pc-w*p0, gc-w*g0
    return {"P_MP2_MO_consistent": p0+p1+p2, "Gamma_MP2_MO_consistent": g0+g1+g2,
            "P_MO_order0": p0, "P_MO_order1": p1, "P_MO_order2": p2,
            "Gamma_MO_order0": g0, "Gamma_MO_order1": g1, "Gamma_MO_order2": g2,
            "ci_reference": reference, "ci_first_order": chi,
            "ci_second_order_normalization": -.5*w*reference,
            "ci_normalized_eta1": (reference+chi)/np.sqrt(1+w),
            "mp2_first_order_norm_squared": np.asarray(w)}


def select_valence_aos(mol):
    """Exclude only Z > 2, principal-1 AOs after the full-AO Lowdin transform.

    Legacy selection uniformly retained the highest principal shell on each
    atom: H 1s, second-period 2s/2p, and S 3s/3p. Thus it also excluded S
    2s/2p; sulfur had no separate selection rule. The revised rule retains
    those AOs, changing only H2S from 6 to 10 descriptor AOs in this benchmark.
    These AO exclusions do not freeze occupied MOs: MP2 remains frozen=0.
    """
    labels = mol.ao_labels()
    structured = mol.ao_labels(fmt=False)
    core, valence, shells = [], [], []
    locations = mol.ao_loc_nr()
    for shell in range(mol.nbas):
        atom = mol.bas_atom(shell)
        Z = mol.atom_charge(atom)
        for i in range(int(locations[shell]), int(locations[shell+1])):
            principal = int(structured[i][2][:-1])
            is_core = Z > 2 and principal == 1
            (core if is_core else valence).append(i)
            shells.append({"ao_index": i, "atom_index": atom, "shell_index": shell,
                           "principal_shell": principal, "angular_momentum": mol.bas_angular(shell),
                           "n_contractions": mol.bas_nctr(shell), "n_primitives": mol.bas_nprim(shell)})
    return {"ao_selection_rule": np.asarray(AO_SELECTION_RULE),
            "ao_labels": np.asarray(labels), "core_indices": np.asarray(core, dtype=int),
            "valence_indices": np.asarray(valence, dtype=int),
            "core_labels": np.asarray([labels[i] for i in core]),
            "valence_labels": np.asarray([labels[i] for i in valence]),
            "ao_shell_metadata_json": np.asarray(json.dumps(shells))}


def extract_descriptors(P_L, F_L, Lambda_L, valence_indices):
    v = np.asarray(valence_indices)
    n = len(v)
    P, F = P_L[np.ix_(v,v)], F_L[np.ix_(v,v)]
    L = Lambda_L[np.ix_(v,v,v,v)]
    pairs = np.asarray(list(itertools.combinations(range(n),2)), dtype=int).reshape(-1,2)
    triples = np.asarray(list(itertools.combinations(range(n),3)), dtype=int).reshape(-1,3)
    # Stored rank-4 tensors stay in PySCF order: raw[p,q,r,s] = Methods[p,r,q,s].
    # Keep k as an output index: T[i,j,k] = raw[i,k,j,k], with no summation.
    lambda_methods = L.transpose(0, 2, 1, 3)
    T = np.diagonal(lambda_methods, axis1=2, axis2=3).copy()
    return {"P_mu": np.diag(P).copy(), "F_mu": np.diag(F).copy(),
            "pair_indices": pairs, "P_munu": P[pairs[:,0],pairs[:,1]],
            "T_full": T, "triple_indices": triples,
            "T_munulambda": T[triples[:,0],triples[:,1],triples[:,2]]}


def calculate_record(record):
    import pyscf
    from pyscf import gto
    from pyscf.scf.hf import RHF
    from pyscf.mp import mp2
    mol = gto.M(atom=list(zip(record["symbols"], record["cartesian_A"])), unit="Angstrom",
                basis=record["basis"], charge=record["charge"], spin=record["spin"],
                cart=False, symmetry=False, verbose=0)
    with np.load(record["input_state_path"], allow_pickle=False) as saved:
        dm0 = saved["rhf_density_ao"]
    mf = RHF(mol)
    # These are solver settings, not post-calculation acceptance tests.
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-13, 1e-10, 1e-14
    mf.max_cycle, mf.diis_space, mf.conv_check = 400, 12, False
    mf.direct_scf, mf.chkfile = False, None
    mf.kernel(dm0=dm0)
    mf.mo_energy, mf.mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    pt = mp2.RMP2(mf, frozen=0)
    e_mp2, t2 = pt.kernel()
    C, S, P = mf.mo_coeff, mf.get_ovlp(), mf.make_rdm1()
    F = mf.get_fock(dm=P)
    arrays = {"S_AO": S, "P_AO": P, "F_AO": F,
              "hcore_AO": mf.get_hcore(), "eri_AO": mol.intor("int2e", aosym="s1"),
              "E_nuclear": mol.energy_nuc(), "E_RHF_calculated": mf.e_tot,
              "E_MP2_correlation": e_mp2, "E_MP2_total": mf.e_tot+e_mp2,
              "rhf_converged": mf.converged, "mp2_t2": t2,
              "mo_coeff": C, "mo_occ": mf.mo_occ, "mo_energy": mf.mo_energy,
              "P_MP2_MO_pyscf": pt.make_rdm1(ao_repr=False),
              "Gamma_MP2_MO_pyscf": pt.make_rdm2(ao_repr=False),
              "P_MP2_AO_pyscf": pt.make_rdm1(ao_repr=True),
              "Gamma_MP2_AO_pyscf": pt.make_rdm2(ao_repr=True)}
    arrays.update(build_consistent_rdms(t2, C.shape[1], mol.nelectron))
    for order in range(3):
        arrays[f"P_AO_order{order}"] = C@arrays[f"P_MO_order{order}"]@C.T
        arrays[f"Gamma_AO_order{order}"] = transform_rank4(arrays[f"Gamma_MO_order{order}"], C)
    D = C@arrays["P_MP2_MO_consistent"]@C.T
    Gamma = transform_rank4(arrays["Gamma_MP2_MO_consistent"], C)
    gamma0 = disconnected_rhf(P)
    L = Gamma-gamma0
    arrays.update(P_MP2_AO_consistent=D, Gamma_MP2_AO_consistent=Gamma,
                  P_MP2_AO=D, Gamma_MP2_AO=Gamma, Gamma0_HF_AO=gamma0, Gamma0_AO=gamma0,
                  Lambda_HFref_AO=L, Lambda_AO=L, Gamma0_correlated_AO=disconnected_rhf(D),
                  Lambda_conventional_AO=Gamma-disconnected_rhf(D))
    half, inverse = lowdin_factors(S)
    arrays.update(S_half=half, S_minus_half=inverse, P_L=half@P@half,
                  F_L=inverse@F@inverse, Lambda_L=transform_rank4(L, half))
    arrays.update(select_valence_aos(mol))
    arrays.update(extract_descriptors(arrays["P_L"], arrays["F_L"], arrays["Lambda_L"], arrays["valence_indices"]))
    metadata = {"rdm_source": RDM_SOURCE, "perturbative_order": 2,
                "wavefunction": "Psi(eta)=(Phi+eta*chi)/sqrt(1+eta^2*w); w=<chi|chi>",
                "expectation": "R0+R1+R_chichi-w*R0; no exact division by 1+w",
                "normalization": "wavefunction coefficient at order2 is -w*Phi/2",
                "spin_convention": "spin-summed spatial-orbital, closed-shell singlet",
                "index_ordering_MO": RDM_CONVENTION, "index_ordering_AO": RDM_CONVENTION,
                "frozen_core": 0, "input_fci_frozen_core": record["n_frozen_orbitals"],
                "calculation_space": "all occupied and virtual MOs; full AOs until final descriptor extraction",
                "formal_cumulant": "Gamma_MP2_AO_consistent-Gamma0[P_HF]",
                "conventional_cumulant": "Gamma_MP2_AO_consistent-Gamma0[P_MP2_AO_consistent]; stored only",
                "pyscf_version": pyscf.__version__}
    arrays.update(molecule=record["molecule"], geometry_id=record["geometry_id"],
                  point_index=record["point_index"], input_geometry_path=record["input_geometry_path"],
                  symbols=record["symbols"], cartesian_A=record["cartesian_A"],
                  input_state_path=record["input_state_path"], scan_coordinate=record["q_A"],
                  q_A=record["q_A"], bond_length_A=record["bond_length_A"], units="Angstrom", energy_units="Hartree",
                  charge=mol.charge, spin=mol.spin, basis=record["basis"], electron_count=mol.nelectron,
                  frozen_core=0, mp2_frozen_core=0, rdm_frozen_core=0,
                  input_fci_frozen_core=record["n_frozen_orbitals"], input_fci_convention=record["fci_convention"],
                  E_RHF_input=record["E_RHF_Ha"], E_FCI=record["E_FCI_Ha"], E_corr_input=record["E_corr_Ha"],
                  E_FCI_minus_E_RHF_input=record["E_FCI_Ha"]-record["E_RHF_Ha"],
                  schema_version=SCHEMA_VERSION, rdm_source=RDM_SOURCE, rdm_convention=RDM_CONVENTION,
                  rdm_metadata_json=json.dumps(metadata), pyscf_version=pyscf.__version__,
                  validation_performed=False)
    return {key: np.asarray(value) for key,value in arrays.items()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def save_record(output, record, arrays=None, error=None):
    base = output/record["molecule"]/record["geometry_id"]
    base.parent.mkdir(parents=True, exist_ok=True)
    # Keep previous attempts when the user explicitly retries an execution error.
    if base.with_suffix(".json").exists() or base.with_suffix(".npz").exists():
        attempt = 1
        while any(base.with_suffix(f".attempt{attempt}{ext}").exists() for ext in (".json", ".npz")):
            attempt += 1
        for ext in (".json", ".npz"):
            if base.with_suffix(ext).exists():
                base.with_suffix(ext).replace(base.with_suffix(f".attempt{attempt}{ext}"))
    row = {"molecule": record["molecule"], "geometry_id": record["geometry_id"],
           "q_A": record["q_A"],
           "record": str(base.with_suffix(".npz")) if arrays is not None else "", "error": error or ""}
    if arrays is not None:
        with base.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        base.with_suffix(".npz.tmp").replace(base.with_suffix(".npz"))
    write_json(base.with_suffix(".json"), {**row, "validation_performed": False})
    return row


T_REPAIR_VERSION = "descriptor-v2-methods-t"
RDM_TOL = 1e-8  # Existing strict tolerance in tests/methods41_validation.py.
T_FIELDS = frozenset(("T_full", "T_munulambda"))
T_REPAIR_MAPPING = {
    "raw_tensor": "G[p,q,r,s] = Lambda_Methods[p,r,q,s]",
    "corrected_T": "T[i,j,k] = Lambda_Methods[i,j,k,k] = G[i,k,j,k]",
    "previous_T": "T_old[i,j,k] = G[i,j,k,k]",
    "selection": "valence-local i < j < k, lexicographic combinations; unchanged",
    "rank4_storage": "unchanged PySCF/raw ordering",
}


class AuditFailure(ValueError):
    """One or more records failed preflight; no output dataset was created."""

    def __init__(self, report):
        self.report = report
        super().__init__(f"Saved-data audit failed for {len(report['failures'])} record(s)")


def maxabs(value):
    return float(np.max(np.abs(value), initial=0))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _write_repair_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def finite_array(arrays, key, shape=None):
    if key not in arrays:
        raise ValueError(f"Missing required array: {key}")
    value = arrays[key]
    if shape is not None and value.shape != shape:
        raise ValueError(f"{key}: expected full shape {shape}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value) or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain finite real numbers")
    return value


def audit_record(arrays):
    """Validate saved scientific inputs, then compare production T with a literal loop."""
    S = finite_array(arrays, "S_AO")
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not S.shape[0]:
        raise ValueError("S_AO must be a nonempty square full-AO metric")
    n = len(S)
    matrix, rank4 = (n, n), (n,) * 4
    for key in ("Lambda_L", "Lambda_AO", "Gamma0_HF_AO", "Gamma_MP2_AO_consistent",
                "Gamma_MP2_MO_consistent"):
        finite_array(arrays, key, rank4)
    for key in ("P_AO", "F_AO", "P_L", "F_L", "S_half", "S_minus_half", "mo_coeff",
                "P_MP2_AO_consistent", "P_MP2_MO_consistent"):
        finite_array(arrays, key, matrix)
    N = float(finite_array(arrays, "electron_count", ()))
    if not N.is_integer() or N < 2 or N > 2*n:
        raise ValueError("electron_count must be an integer in [2, 2*n]")
    for key in ("frozen_core", "mp2_frozen_core", "rdm_frozen_core"):
        if float(finite_array(arrays, key, ())) != 0:
            raise ValueError(f"{key}: expected the saved all-electron production convention")

    residuals = {}

    def check(key, residual, tolerance=RDM_TOL):
        if not np.isfinite(residual).all():
            raise ValueError(f"{key}: nonfinite validation residual")
        residuals[key] = maxabs(residual)
        if residuals[key] > tolerance:
            raise ValueError(f"{key}: {residuals[key]:.17g} exceeds {tolerance:.17g}")

    if np.linalg.eigvalsh(S).min() <= 1e-8:
        raise ValueError("S_AO minimum eigenvalue must exceed the existing 1e-8 Lowdin threshold")
    half, inverse, C = arrays["S_half"], arrays["S_minus_half"], arrays["mo_coeff"]
    check("S_symmetry", S-S.T)
    check("S_half_symmetry", half-half.T)
    if np.linalg.eigvalsh(half).min() <= 0:
        raise ValueError("S_half must be the positive Lowdin square root")
    check("S_half_squared", half@half-S)
    check("S_minus_half_inverse", half@inverse-np.eye(n))
    check("MO_orthonormality", C.T@S@C-np.eye(n))
    check("RHF_particle_number", np.einsum("pq,qp", arrays["P_AO"], S)-N)
    check("P_L_saved_transform", half@arrays["P_AO"]@half-arrays["P_L"])
    check("F_L_saved_transform", inverse@arrays["F_AO"]@inverse-arrays["F_L"])

    for basis, metric in (("MO", np.eye(n)), ("AO", S)):
        D, G = arrays[f"P_MP2_{basis}_consistent"], arrays[f"Gamma_MP2_{basis}_consistent"]
        contracted = np.einsum("pqrs,rs->pq", G, metric)
        check(f"{basis}_particle_number", np.einsum("pq,qp", D, metric)-N)
        check(f"{basis}_pair_number", np.einsum("pq,qp", contracted, metric)-N*(N-1))
        check(f"{basis}_contraction", contracted-(N-1)*D.T)
        check(f"{basis}_P_hermiticity", D-D.T)
        check(f"{basis}_Gamma_hermiticity", G-G.transpose(1, 0, 3, 2))
        check(f"{basis}_Gamma_particle_exchange", G-G.transpose(2, 3, 0, 1))
    check("P_MO_to_AO", C@arrays["P_MP2_MO_consistent"]@C.T-arrays["P_MP2_AO_consistent"])
    check("Gamma_MO_to_AO", transform_rank4(arrays["Gamma_MP2_MO_consistent"], C)
          - arrays["Gamma_MP2_AO_consistent"])
    gamma0 = disconnected_rhf(arrays["P_AO"])
    check("Gamma0_HF", gamma0-arrays["Gamma0_HF_AO"])
    check("Lambda_AO_subtraction", arrays["Gamma_MP2_AO_consistent"]-gamma0-arrays["Lambda_AO"])
    for alias, canonical in (("Gamma0_AO", "Gamma0_HF_AO"),
                             ("Lambda_HFref_AO", "Lambda_AO")):
        if alias in arrays:
            check(f"{alias}_alias", finite_array(arrays, alias, rank4)-arrays[canonical])
    check("Lambda_L_saved_transform", transform_rank4(arrays["Lambda_AO"], half)-arrays["Lambda_L"])

    v = arrays.get("valence_indices")
    if v is None or v.ndim != 1 or not np.issubdtype(v.dtype, np.integer):
        raise ValueError("valence_indices must be a one-dimensional integer array")
    if len(v) < 3 or len(np.unique(v)) != len(v) or np.any(v < 0) or np.any(v >= n):
        raise ValueError("valence_indices must contain at least three distinct full-AO indices")
    nv = len(v)
    pairs = np.asarray(list(itertools.combinations(range(nv), 2)), dtype=int).reshape(-1, 2)
    triples = np.asarray(list(itertools.combinations(range(nv), 3)), dtype=int).reshape(-1, 3)
    for key, expected in (("pair_indices", pairs), ("triple_indices", triples)):
        actual = arrays.get(key)
        if actual is None or not np.issubdtype(actual.dtype, np.integer) or not np.array_equal(actual, expected):
            raise ValueError(f"{key} must retain the existing lexicographic valence-local combinations")
    finite_array(arrays, "T_full", (nv,)*3)
    finite_array(arrays, "T_munulambda", (len(triples),))
    for key, shape in (("P_mu", (nv,)), ("F_mu", (nv,)), ("P_munu", (len(pairs),))):
        finite_array(arrays, key, shape)
    extracted = extract_descriptors(arrays["P_L"], arrays["F_L"], arrays["Lambda_L"], v)
    for key in ("P_mu", "F_mu", "P_munu", "pair_indices", "triple_indices"):
        if not np.array_equal(extracted[key], arrays[key]):
            raise ValueError(f"Non-T descriptor {key} disagrees with saved extraction")
    raw = arrays["Lambda_L"][np.ix_(v, v, v, v)]
    literal = np.empty((nv,)*3, dtype=raw.dtype)
    for i in range(nv):
        for j in range(nv):
            for k in range(nv):
                literal[i, j, k] = raw[i, k, j, k]
    check("T_vectorized_vs_literal_loop", extracted["T_full"]-literal, tolerance=0.0)
    selected = np.asarray([literal[i, j, k] for i, j, k in triples], dtype=literal.dtype)
    check("T_selection_vs_literal_loop", extracted["T_munulambda"]-selected, tolerance=0.0)
    replacements = {key: extracted[key] for key in T_FIELDS}
    for key, value in replacements.items():
        if value.dtype != arrays[key].dtype or value.shape != arrays[key].shape:
            raise ValueError(f"{key}: repair would change dtype or shape")
    differences = {}
    for key, value in replacements.items():
        delta = np.abs(value-arrays[key])
        differences[key] = {"max_abs_difference": maxabs(delta),
                            "mean_abs_difference": float(delta.mean()),
                            "sum_abs_difference": float(delta.sum()),
                            "element_count": int(delta.size),
                            "exactly_unchanged": bool(np.array_equal(value, arrays[key])),
                            "unchanged_at_1e_8": bool(maxabs(delta) <= RDM_TOL)}
    return replacements, {"full_AO_count": n, "valence_AO_count": nv,
                          "Lambda_L_shape": list(rank4), "T_full_shape": list(literal.shape),
                          "T_munulambda_shape": list(selected.shape),
                          "residuals": residuals, "differences": differences}


def patch_archive(source, output, replacements):
    """Preserve every non-T NPY member byte-for-byte, including its header and dtype."""
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(output, "x") as repaired:
        names = original.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate NPZ member in {source}")
        if not {key+".npy" for key in T_FIELDS}.issubset(names):
            raise ValueError(f"Missing T payload in {source}")
        for member in original.infolist():
            key = member.filename.removesuffix(".npy")
            payload = original.read(member.filename)
            if key in replacements and member.filename == key+".npy":
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, replacements[key], allow_pickle=False)
                payload = buffer.getvalue()
            repaired.writestr(member, payload)
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(output, "r") as repaired:
        if original.namelist() != repaired.namelist():
            raise RuntimeError("NPZ member order or population changed")
        count = 0
        for name in original.namelist():
            if name not in {key+".npy" for key in T_FIELDS}:
                if original.read(name) != repaired.read(name):
                    raise RuntimeError(f"Non-T payload changed: {name}")
                count += 1
    with np.load(output, allow_pickle=False) as saved:
        for key, expected in replacements.items():
            if saved[key].dtype != expected.dtype or not np.array_equal(saved[key], expected):
                raise RuntimeError(f"Repaired {key} did not round-trip exactly")
    return count


def repair_dataset(source, output, expected_count=270, audit_only=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    before = snapshot(source)
    manifest = json.loads((source/"manifest.json").read_text())
    rows = manifest["records"]
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} manifest records, found {len(rows)}")
    identities = [(row["molecule"], row["geometry_id"]) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Duplicate molecule/geometry identity in source manifest")
    for molecule, geometry_id in identities:
        if Path(molecule).name != molecule or Path(geometry_id).name != geometry_id:
            raise ValueError("Molecule and geometry identifiers cannot contain path components")
    if expected_count == 270:
        expected = {(name, f"{index:03d}") for name in MOLECULES for index in range(1, 31)}
        if set(identities) != expected:
            raise ValueError("Expected all nine molecules and geometry IDs 001 through 030")
    report = {"version": T_REPAIR_VERSION, "status": "AUDITING", "source": str(source), "output": str(output),
              "mapping": T_REPAIR_MAPPING, "strict_rdm_absolute_tolerance": RDM_TOL,
              "T_indexing_absolute_tolerance": 0.0, "geometries_audited": 0,
              "geometries_regenerated": 0, "missing_usable_full_cumulant": 0,
              "failures": [], "records": [], "per_molecule": {}, "maximum_residuals": {},
              "source_sha256_before": before, "non_T_NPY_payloads_verified_unchanged": 0,
              "solver_calls": 0, "RDM_regenerations": 0}
    prepared = []
    for row in rows:
        identity = f"{row['molecule']}/{row['geometry_id']}"
        report["geometries_audited"] += 1
        usable = False
        try:
            if row.get("error") or not row.get("record"):
                raise ValueError("Manifest record is missing or failed")
            path = Path(row["record"])
            if not path.is_absolute():
                path = source/path
            path = path.resolve()
            if source not in path.parents:
                raise ValueError("Manifest record is outside the source dataset")
            relative = path.relative_to(source)
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            n = len(arrays["S_AO"])
            finite_array(arrays, "Lambda_L", (n,)*4)
            usable = True
            for key in ("molecule", "geometry_id"):
                if str(arrays[key]) != row[key]:
                    raise ValueError(f"Manifest and NPZ {key} disagree")
            replacements, evidence = audit_record(arrays)
            evidence.update(identity=identity, source_record=str(path), output_record=str(output/relative),
                            source_sha256=before[str(relative)])
            report["records"].append(evidence)
            prepared.append((row, path, relative, replacements, evidence))
            for key, residual in evidence["residuals"].items():
                report["maximum_residuals"][key] = max(report["maximum_residuals"].get(key, 0), residual)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile, TypeError) as exc:
            report["failures"].append({"identity": identity, "error": f"{type(exc).__name__}: {exc}"})
            if not usable:
                report["missing_usable_full_cumulant"] += 1
    if snapshot(source) != before:
        raise RuntimeError("Source dataset changed during the audit; no output created")
    report["source_files_unchanged"] = True
    if report["failures"]:
        report["status"] = "FAIL"
        raise AuditFailure(report)
    for name in dict.fromkeys(row["molecule"] for row in rows):
        records = [item for item in report["records"] if item["identity"].split("/")[0] == name]
        summary = {"geometry_count": len(records)}
        for key in sorted(T_FIELDS):
            diffs = [item["differences"][key] for item in records]
            summary[key] = {"max_abs_difference": max(item["max_abs_difference"] for item in diffs),
                            "mean_abs_difference": sum(item["sum_abs_difference"] for item in diffs)
                            / sum(item["element_count"] for item in diffs),
                            "exactly_unchanged_geometry_count": sum(item["exactly_unchanged"] for item in diffs),
                            "unchanged_at_1e_8_geometry_count": sum(item["unchanged_at_1e_8"] for item in diffs)}
        report["per_molecule"][name] = summary
    if audit_only:
        report["status"] = "AUDIT_PASS"
        return report

    output.mkdir(parents=True, exist_ok=False)
    new_rows = []
    for row, path, relative, replacements, evidence in prepared:
        if sha256(path) != evidence["source_sha256"]:
            raise RuntimeError(f"Source changed after audit: {path}")
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = patch_archive(path, destination, replacements)
        report["non_T_NPY_payloads_verified_unchanged"] += count
        evidence.update(non_T_NPY_payloads_verified_unchanged=count, output_sha256=sha256(destination))
        new_row = {**row, "record": str(destination)}
        new_rows.append(new_row)
        sidecar_source = path.with_suffix(".json")
        sidecar = json.loads(sidecar_source.read_text()) if sidecar_source.exists() else {}
        _write_repair_json(destination.with_suffix(".json"), {**sidecar, **new_row,
                   "T_repair": {"version": T_REPAIR_VERSION, "mapping": T_REPAIR_MAPPING, "evidence": evidence}})
        report["geometries_regenerated"] += 1
    after = snapshot(source)
    if after != before:
        raise RuntimeError("Source dataset changed during migration; output is incomplete and has no manifest")
    report["source_sha256_after"] = after
    report["status"] = "PASS"
    report["all_non_T_fields_unchanged"] = True
    fields = ["molecule", "geometry_id", "q_A", "record", "error"]
    for name in report["per_molecule"]:
        with (output/name/"summary.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(row for row in new_rows if row["molecule"] == name)
    _write_repair_json(output/"t_repair_audit.json", report)
    _write_repair_json(output/"manifest.json", {**manifest, "records": new_rows,
               "t_descriptor_version": 2, "t_axis_mapping": T_AXIS_MAPPING,
               "T_descriptor_version": T_REPAIR_VERSION, "T_axis_mapping": T_REPAIR_MAPPING,
               "T_repair_source": str(source), "T_repair_audit": str(output/"t_repair_audit.json")})
    return report



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-t", action="store_true", help="Repair saved T fields without electronic-structure calculations")
    parser.add_argument("--input-dir", "--source-dir", dest="input_dir", type=Path,
                        help="Input geometries, or saved descriptors with --repair-t")
    parser.add_argument("--expected-count", type=int, help="Expected saved record count for --repair-t; default 270")
    parser.add_argument("--audit-only", action="store_true", help="With --repair-t, validate without writing a dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT/"results/descriptor-v2-methods-t")
    parser.add_argument("--molecules", nargs="+", choices=MOLECULES)
    parser.add_argument("--geometry-id", help="Single scan ID; use with one molecule")
    parser.add_argument("--smoke-test", action="store_true", help="Calculate H2O/001 only; no validation")
    parser.add_argument("--dry-run", action="store_true", help="List selected input records only; no validation")
    parser.add_argument("--resume", action="store_true", help="Skip completed files without revalidation")
    parser.add_argument("--rerun-failed", action="store_true", help="Retry execution failures; keep completed files")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repair_t:
        if (args.molecules is not None or args.geometry_id or args.smoke_test or args.dry_run
                or args.resume or args.rerun_failed or args.threads != 1):
            parser.error("--repair-t cannot be combined with calculation or resume options")
        args.input_dir = args.input_dir if args.input_dir is not None else PROJECT/"results/descriptor"
        args.expected_count = 270 if args.expected_count is None else args.expected_count
        if args.expected_count < 1:
            parser.error("--expected-count must be positive")
    else:
        if args.audit_only or args.expected_count is not None:
            parser.error("--audit-only and --expected-count require --repair-t")
        args.input_dir = args.input_dir if args.input_dir is not None else PROJECT/"results/initialdata"
    args.molecules = list(MOLECULES) if args.molecules is None else args.molecules
    if args.smoke_test:
        args.molecules, args.geometry_id = ["H2O"], "001"
    if args.geometry_id:
        if len(args.molecules) != 1:
            parser.error("--geometry-id requires one molecule")
        args.geometry_id = f"{int(args.geometry_id):03d}"
    return args


def run(args):
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    # File protection only: input data are never an output destination.
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must be separate and non-nested")
    records = load_records(source, args.molecules)
    if args.geometry_id:
        records = [r for r in records if r["geometry_id"] == args.geometry_id]
    if args.dry_run:
        print(json.dumps({"geometry_count": len(records), "records": [
            {k:r[k] for k in ("molecule", "geometry_id", "input_geometry_path")} for r in records]}, indent=2))
        return 0
    if output.exists() and any(output.iterdir()) and not (args.resume or args.rerun_failed):
        raise FileExistsError("Output exists; use --resume or --rerun-failed, or a new output directory")
    from pyscf import lib
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"OpenMP is not available\. Setting omp_threads to .* has no effects\.",
                                category=UserWarning, module=r"pyscf\.lib\.misc")
        lib.num_threads(args.threads)
    manifest_path = output/"manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"records": []}
    if previous["records"] and (
            previous.get("t_descriptor_version") != 2
            or previous.get("t_axis_mapping") != T_AXIS_MAPPING):
        raise ValueError("Existing output uses an older or unknown T mapping; "
                         "use descriptor.py --repair-t to create a new versioned dataset")
    if previous["records"] and previous.get("ao_selection_rule") != AO_SELECTION_RULE:
        raise ValueError("Existing output uses an older or unknown post-Lowdin AO selection; "
                         "use a new output directory for the 1s-only descriptor space. "
                         "T-only repair does not change AO selection.")
    entries = {(r["molecule"],r["geometry_id"]): r for r in previous["records"]}
    fields = ["molecule", "geometry_id", "q_A", "record", "error"]
    def checkpoint():
        rows = sorted(entries.values(), key=lambda r: (MOLECULES.index(r["molecule"]),int(r["geometry_id"])))
        manifest = {"schema_version": SCHEMA_VERSION, "validation_performed": False,
                    "t_descriptor_version": 2, "t_axis_mapping": T_AXIS_MAPPING,
                    "ao_selection_rule": AO_SELECTION_RULE,
                    "rdm_source": RDM_SOURCE, "mp2_frozen_core": 0,
                    "input_dir": str(source), "records": rows,
                    "completed_geometry_count": sum(bool(r.get("record")) and not r.get("error") for r in rows),
                    "failed_geometry_count": sum(bool(r.get("error")) for r in rows)}
        write_json(manifest_path, manifest)
        for name in sorted({r["molecule"] for r in rows}):
            with (output/name/"summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(r for r in rows if r["molecule"] == name)
    exit_code = 0
    for record in records:
        key = record["molecule"],record["geometry_id"]
        base = output/key[0]/key[1]
        sidecar = json.loads(base.with_suffix(".json").read_text()) if base.with_suffix(".json").exists() else {}
        failed = bool(sidecar.get("error")) or sidecar.get("status") in ("FAILED", "FAIL")
        if base.with_suffix(".npz").exists() and not failed:
            entries[key] = {"molecule": key[0], "geometry_id": key[1], "q_A": record["q_A"],
                            "record": str(base.with_suffix(".npz")), "error": ""}
            print(f"SKIP {key[0]}/{key[1]}", flush=True)
            continue
        if failed and not args.rerun_failed:
            entries[key] = {k:sidecar[k] for k in fields}
            print(f"FAILED {key[0]}/{key[1]}: use --rerun-failed to retry", flush=True)
            exit_code = 1
            continue
        try:
            arrays = calculate_record(record)
            entries[key] = save_record(output, record, arrays)
        except Exception as exc:
            entries[key] = save_record(output, record, error=f"{type(exc).__name__}: {exc}")
            exit_code = 1
        checkpoint()
        print(f"{key[0]}/{key[1]} {entries[key]['error'] or entries[key]['record']}", flush=True)
    checkpoint()
    print(f"Completed: {sum(bool(r.get('record')) and not r.get('error') for r in entries.values())}; execution failures: "
          f"{sum(bool(r.get('error')) for r in entries.values())}. Scientific validation was not run.", flush=True)
    return exit_code


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.repair_t:
            report = repair_dataset(args.input_dir, args.output_dir, args.expected_count, args.audit_only)
            print(json.dumps({key: value for key, value in report.items()
                              if key not in ("records", "source_sha256_before", "source_sha256_after")}, indent=2))
            return 0
        return run(args)
    except AuditFailure as exc:
        print(json.dumps(exc.report, indent=2))
        return 1
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
