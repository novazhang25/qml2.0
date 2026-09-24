"""RDM algebra for the normalized first-order MP2 wavefunction ansatz.

For Psi(eta) = (Phi + eta chi) / sqrt(1 + eta**2 <chi|chi>),
retain expectation-value coefficients through eta**2.  Here chi contains the
unmodified RMP2 doubles.  This definition does not include an independent
second-order wavefunction or orbital-response contribution.

Only determinant conversion and RDM contractions are used.  No electronic
energy solver is called by this module.
"""
from __future__ import annotations

import numpy as np


RDM_SOURCE = "normalized truncated MP2 wavefunction, expectation values through second order"
RDM_CONVENTION = "Gamma[p,q,r,s] = sum_spin <p^dagger r^dagger s q>; real spatial MOs"
PRIMARY_SOURCES = {
    "amplitude_mapping": "https://pyscf.org/_modules/pyscf/ci/cisd.html#to_fcivec",
    "density_convention": "https://pyscf.org/_modules/pyscf/fci/direct_spin1.html#make_rdm12",
    "mp2_amplitudes": "https://pyscf.org/_modules/pyscf/mp/mp2.html#kernel",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _maxabs(value):
    return float(np.max(np.abs(value), initial=0.0))


def build_consistent_rdms(t2, nmo, nelectron, frozen_core):
    """Return full-MO density coefficients and construction metadata.

    ``t2`` has PySCF RMP2 order ``[i,j,a,b]`` and contains only the active
    occupied orbitals.  ``nmo`` and ``nelectron`` include the frozen core.
    Integer ``frozen_core`` denotes the first canonical occupied spatial MOs.

    PySCF's CISD-to-determinant conversion takes the opposite-spin doubles
    directly from c2 and constructs same-spin doubles as c2[i,j,a,b] minus
    c2[j,i,a,b].  Consequently raw RMP2 t2 is used as c2 without an extra
    spin factor.  The converter also restores the frozen occupations and
    associated determinant signs.  Packed CISD amplitudes are not treated
    as an orthonormal determinant vector when computing the norm.

    The returned consistent RDM is R0 + R1 + R2, where
    R1 = <Phi|O|chi> + <chi|O|Phi> and
    R2 = <chi|O|chi> - <chi|chi> R0.
    ``ci_normalized_eta1`` documents the normalized ansatz at eta=1; taking
    its exact RDM would retain higher orders and is not the returned RDM.
    """
    from pyscf import __version__ as pyscf_version
    from pyscf.ci import cisd
    from pyscf.fci import direct_spin1

    for value, label in ((nmo, "nmo"), (nelectron, "nelectron"),
                         (frozen_core, "frozen_core")):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, bool),
                 f"{label} must be an integer")
    nmo, nelectron, frozen_core = int(nmo), int(nelectron), int(frozen_core)
    _require(nmo > 0 and 0 < nelectron <= 2 * nmo and nelectron % 2 == 0,
             "Requires a valid closed-shell full-space electron count")
    nocc_full = nelectron // 2
    _require(0 <= frozen_core < nocc_full, "Invalid frozen occupied orbital count")
    nocc, nvir = nocc_full - frozen_core, nmo - nocc_full
    _require(nvir > 0, "The MP2 descriptor requires at least one virtual orbital")
    t2 = np.asarray(t2)
    _require(np.isrealobj(t2) and np.isfinite(t2).all(), "RMP2 amplitudes must be finite and real")
    _require(t2.shape == (nocc, nocc, nvir, nvir),
             f"RMP2 t2 shape differs from full-space/frozen-core counts: {t2.shape}")
    t2 = np.asarray(t2, dtype=float)
    symmetry_error = _maxabs(t2 - t2.transpose(1, 0, 3, 2))
    _require(symmetry_error <= 1e-10, "RMP2 amplitudes violate t[i,j,a,b] = t[j,i,b,a]")
    singles = np.zeros((nocc, nvir))

    def determinant_vector(c0, doubles):
        packed = cisd.amplitudes_to_cisdvec(c0, singles, doubles)
        return np.asarray(cisd.to_fcivec(packed, nmo, nelectron, frozen=frozen_core))

    reference = determinant_vector(1.0, np.zeros_like(t2))
    first_order = determinant_vector(0.0, t2)
    reference_norm = float(np.vdot(reference, reference).real)
    reference_overlap = float(np.vdot(reference, first_order).real)
    norm_squared = float(np.vdot(first_order, first_order).real)
    # The packed restricted-spin metric independently verifies the norm used
    # in both RDMs.  It is not the ordinary Euclidean norm of packed c2.
    amplitude_norm_squared = float(2 * np.einsum("ijab,ijab->", t2, t2)
                                   - np.einsum("ijab,jiab->", t2, t2))
    norm_error = abs(norm_squared - amplitude_norm_squared)
    _require(abs(reference_norm - 1.0) <= 1e-12, "Reference determinant is not normalized")
    _require(abs(reference_overlap) <= 1e-12, "First-order wavefunction overlaps the reference")
    _require(norm_error <= 1e-10 * max(1.0, norm_squared),
             "Determinant and restricted-amplitude norms disagree")

    second_order_normalization = -0.5 * norm_squared * reference
    normalized_eta1 = (reference + first_order) / np.sqrt(1.0 + norm_squared)
    normalized_norm = float(np.vdot(normalized_eta1, normalized_eta1).real)
    norm_coefficients = (
        reference_norm,
        2.0 * reference_overlap,
        norm_squared + 2.0 * float(np.vdot(reference, second_order_normalization).real),
    )
    _require(abs(normalized_norm - 1.0) <= 1e-12, "Normalized ansatz norm differs from one")
    _require(max(abs(norm_coefficients[0] - 1.0), abs(norm_coefficients[1]),
                 abs(norm_coefficients[2])) <= 1e-12,
             "Wavefunction norm coefficients are not (1, 0, 0)")

    # make_rdm12 is bilinear in its determinant coefficients and does not
    # normalize them.  Thus these three calls extract all polynomial orders
    # without an arbitrary finite-difference step or empirical rescaling.
    p0, gamma0 = direct_spin1.make_rdm12(reference, nmo, nelectron, reorder=True)
    p_chichi, gamma_chichi = direct_spin1.make_rdm12(first_order, nmo, nelectron, reorder=True)
    p_raw, gamma_raw = direct_spin1.make_rdm12(reference + first_order, nmo, nelectron, reorder=True)
    p1, gamma1 = p_raw - p0 - p_chichi, gamma_raw - gamma0 - gamma_chichi
    p2, gamma2 = p_chichi - norm_squared * p0, gamma_chichi - norm_squared * gamma0
    arrays = {
        "P_MP2_MO_consistent": p0 + p1 + p2,
        "Gamma_MP2_MO_consistent": gamma0 + gamma1 + gamma2,
        "P_MO_order0": p0, "P_MO_order1": p1, "P_MO_order2": p2,
        "Gamma_MO_order0": gamma0, "Gamma_MO_order1": gamma1, "Gamma_MO_order2": gamma2,
        "ci_reference": reference, "ci_first_order": first_order,
        "ci_second_order_normalization": second_order_normalization,
        "ci_normalized_eta1": normalized_eta1,
        "mp2_first_order_norm_squared": np.asarray(norm_squared),
        "wavefunction_norm_order0": np.asarray(norm_coefficients[0]),
        "wavefunction_norm_order1": np.asarray(norm_coefficients[1]),
        "wavefunction_norm_order2": np.asarray(norm_coefficients[2]),
    }
    for key, value in arrays.items():
        _require(np.isrealobj(value) and np.isfinite(value).all(), f"Nonfinite or complex {key}")
    metadata = {
        "rdm_source": RDM_SOURCE,
        "rdm_convention": RDM_CONVENTION,
        "rdm1_convention": "P[p,q] = sum_spin <q^dagger p>",
        "spin_convention": "closed-shell singlet; spin-summed spatial-orbital RDMs",
        "wavefunction_ansatz": "Psi(eta) = (Phi + eta chi) / sqrt(1 + eta^2 w); w = <chi|chi>",
        "expectation_expansion": "R(eta) = R0 + eta R1 + eta^2 (R_chichi - w R0) + O(eta^3)",
        "retained_wavefunction_norm_coefficients": list(norm_coefficients),
        "wavefunction_norm_eta1": normalized_norm,
        "first_order_norm_squared": norm_squared,
        "restricted_amplitude_norm_squared": amplitude_norm_squared,
        "amplitude_to_determinant_norm_error": norm_error,
        "amplitude_pair_symmetry_max_error": symmetry_error,
        "amplitude_mapping": "raw RMP2 t2[i,j,a,b] -> CISD c2; c1=0; no amplitude rescaling",
        "frozen_core_embedding": ("none; all electrons and orbitals active" if frozen_core == 0
                                  else "cisd.to_fcivec with full nmo, full nelectron, frozen=frozen_core"),
        "frozen_core": frozen_core, "nmo": nmo, "nelectron": nelectron,
        "definition_scope": "Expectation expansion of the normalized Phi+eta chi ansatz only; "
                            "no independent second-order wavefunction or orbital-response terms",
        "normalized_eta1_vector_note": "Stored for normalization checks; its exact expectation "
                                       "contains higher orders than the saved consistent RDM",
        "pyscf_version": pyscf_version,
        "primary_sources": PRIMARY_SOURCES.copy(),
    }
    return arrays, metadata
