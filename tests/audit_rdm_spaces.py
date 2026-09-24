"""Manual full-space RDM audit; never imported by production descriptor code."""

import numpy as np


def _shape(x):
    return None if x is None else tuple(np.asarray(x).shape)


def audit_rdm_spaces(mf, post=None, rdm_tracks=None, print_ao_labels=True):
    """
    Parameters
    ----------
    mf : converged PySCF RHF object
    post : optional MP2/CC/etc. object
    rdm_tracks : dict, optional
        Example:
        {
            "standard_mo":   {"dm1": dm1_mo, "dm2": dm2_mo, "basis": "MO"},
            "consistent_mo": {"dm1": cdm1_mo, "dm2": cdm2_mo, "basis": "MO"},
            "standard_ao":   {"dm1": dm1_ao, "dm2": dm2_ao, "basis": "AO"},
        }
    """
    mol = mf.mol
    S = mf.get_ovlp()
    C = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)

    nao = mol.nao_nr()
    nmo = C.shape[1]
    nelec = mol.nelectron
    nalpha, nbeta = mol.nelec
    nocc = int(np.count_nonzero(mo_occ > 0))
    ndoubly = int(np.count_nonzero(np.isclose(mo_occ, 2.0)))
    nsingly = int(np.count_nonzero(np.isclose(mo_occ, 1.0)))
    nvir = int(np.count_nonzero(np.isclose(mo_occ, 0.0)))

    s_eval = np.linalg.eigvalsh(S)
    s_tol = max(S.shape) * np.finfo(float).eps * max(abs(s_eval))
    s_rank = int(np.count_nonzero(s_eval > s_tol))
    mo_metric = C.T @ S @ C
    mo_orth_err = np.max(np.abs(mo_metric - np.eye(nmo)))

    print("=" * 72)
    print(f"Molecule                 : {mol.atom}")
    print(f"Basis                    : {mol.basis}")
    print(f"Charge / spin (2S)       : {mol.charge} / {mol.spin}")
    print(f"Electrons N (alpha,beta) : {nelec} ({nalpha},{nbeta})")
    print(f"AO count M_AO            : {nao}")
    print(f"MO coefficient shape     : {C.shape}  [AO, MO]")
    print(f"MO count M_MO            : {nmo}")
    print(f"MO occupations shape     : {mo_occ.shape}")
    print(f"Occupied / virtual MOs   : {nocc} / {nvir}")
    print(f"Doubly / singly occupied : {ndoubly} / {nsingly}")
    print(f"Overlap S shape/rank     : {S.shape} / {s_rank}")
    print(f"min/max eig(S)           : {s_eval.min():.6e} / {s_eval.max():.6e}")
    print(f"max|C^T S C - I|         : {mo_orth_err:.3e}")
    print(f"Expected full 1-RDM      : ({nmo}, {nmo}) = {nmo**2:,} elements")
    print(f"Expected full 2-RDM      : ({nmo}, {nmo}, {nmo}, {nmo}) = {nmo**4:,} elements")

    if nao != nmo:
        print("WARNING: nao != nmo; possible linear-dependence removal or MO truncation.")
    if s_rank != nao:
        print("WARNING: AO overlap is numerically rank deficient.")
    if mo_orth_err > 1e-8:
        print("WARNING: canonical MOs are not orthonormal under the AO metric.")

    print("\nMO occupation vector:")
    print(np.array2string(mo_occ, precision=6, suppress_small=True))

    if post is not None:
        frozen = getattr(post, "frozen", None)
        print("\nPost-HF correlation space")
        print(f"Method object            : {type(post).__name__}")
        print(f"frozen setting           : {frozen!r}")
        try:
            active_mask = np.asarray(post.get_frozen_mask(), dtype=bool)
            frozen_idx = np.flatnonzero(~active_mask)
            active_idx = np.flatnonzero(active_mask)
            corr_occ = np.flatnonzero(active_mask & (mo_occ > 0))
            corr_vir = np.flatnonzero(active_mask & np.isclose(mo_occ, 0.0))
            print(f"Active MO mask shape     : {active_mask.shape}")
            print(f"Active/frozen MO count   : {active_mask.sum()} / {(~active_mask).sum()}")
            print(f"Correlated occupied MOs  : {corr_occ.tolist()}")
            print(f"Correlated virtual MOs   : {corr_vir.tolist()}")
            print(f"Frozen MO indices        : {frozen_idx.tolist()}")
            print(f"All active MO indices    : {active_idx.tolist()}")
            if frozen_idx.size == 0:
                print("CHECK: no frozen orbitals; occupied-core electrons are correlated.")
            else:
                print("WARNING: frozen orbitals exist; this is not all-electron correlation.")
        except (AttributeError, TypeError, ValueError) as exc:
            print(f"Could not inspect frozen mask: {exc}")

    if print_ao_labels:
        print("\nFull AO list (core AOs must appear here):")
        labels = mol.ao_labels()
        for i, label in enumerate(labels):
            print(f"  AO {i:3d}: {label}")
        if len(labels) != nao:
            print(f"WARNING: len(ao_labels)={len(labels)} != nao={nao}")

    if not rdm_tracks:
        return

    print("\n" + "=" * 72)
    print("RDM TRACK AUDIT")
    for name, track in rdm_tracks.items():
        basis = str(track.get("basis", "MO")).upper()
        dm1 = track.get("dm1")
        dm2 = track.get("dm2")
        expected_m = nmo if basis == "MO" else nao
        expected1 = (expected_m, expected_m)
        expected2 = (expected_m,) * 4

        print(f"\n[{name}] basis={basis}")
        print(f"1-RDM shape              : {_shape(dm1)}; expected {expected1}")
        print(f"2-RDM shape              : {_shape(dm2)}; expected {expected2}")

        if dm1 is not None:
            dm1 = np.asarray(dm1)
            shape_ok = dm1.shape == expected1
            print(f"1-RDM full-space shape   : {'PASS' if shape_ok else 'FAIL'}")
            if not shape_ok:
                print("Electron trace skipped: incompatible full-space shape.")
                continue
            if basis == "MO":
                electron_trace = np.trace(dm1).real
                trace_formula = "Tr(dm1)"
            elif basis == "AO":
                electron_trace = np.einsum("pq,qp->", S, dm1).real
                trace_formula = "Tr(S @ dm1)"
            else:
                electron_trace = np.nan
                trace_formula = "unknown basis"
            print(f"Electron trace           : {electron_trace:.12f} via {trace_formula}")
            print(f"Trace target/error       : {nelec} / {electron_trace-nelec:+.3e}")

        if dm2 is not None:
            dm2 = np.asarray(dm2)
            print(f"2-RDM full-space shape   : {'PASS' if dm2.shape == expected2 else 'FAIL'}")
            print(f"2-RDM memory             : {dm2.nbytes / 1024**2:.3f} MiB")


def compare_track_shapes(rdm_tracks):
    """Verify that every supplied track uses identical 1-RDM and 2-RDM shapes."""
    shapes1 = {name: _shape(t.get("dm1")) for name, t in rdm_tracks.items()}
    shapes2 = {name: _shape(t.get("dm2")) for name, t in rdm_tracks.items()}
    print("1-RDM shapes:", shapes1)
    print("2-RDM shapes:", shapes2)
    print("All 1-RDM shapes equal:", len(set(shapes1.values())) <= 1)
    print("All 2-RDM shapes equal:", len(set(shapes2.values())) <= 1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
    from descriptor import (
        build_consistent_rdms, disconnected_rhf, lowdin_factors,
        transform_rank4, select_valence_aos, extract_descriptors,
    )
    from pyscf import gto, scf, mp

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    pt = mp.MP2(mf, frozen=0).run()

    dm1_mo = pt.make_rdm1(ao_repr=False)
    dm2_mo = pt.make_rdm2(ao_repr=False)
    dm1_ao = pt.make_rdm1(ao_repr=True)
    dm2_ao = pt.make_rdm2(ao_repr=True)
    consistent = build_consistent_rdms(pt.t2, mf.mo_coeff.shape[1], mol.nelectron)
    cp = consistent["P_MP2_MO_consistent"]
    cg = consistent["Gamma_MP2_MO_consistent"]
    C = mf.mo_coeff
    cp_ao = C @ cp @ C.T
    cg_ao = transform_rank4(cg, C)

    tracks = {
        "standard_mo": {"dm1": dm1_mo, "dm2": dm2_mo, "basis": "MO"},
        "standard_ao": {"dm1": dm1_ao, "dm2": dm2_ao, "basis": "AO"},
        "consistent_mo": {"dm1": cp, "dm2": cg, "basis": "MO"},
        "consistent_ao": {"dm1": cp_ao, "dm2": cg_ao, "basis": "AO"},
    }

    audit_rdm_spaces(mf, post=pt, rdm_tracks=tracks)
    compare_track_shapes(tracks)

    # Complete every transformation in the full AO space before selecting AOs.
    assert np.all(pt.get_frozen_mask()), "All occupied and virtual MOs must be active"
    S = mf.get_ovlp()
    half, inverse = lowdin_factors(S)
    P = mf.make_rdm1()
    P_L = half @ P @ half
    F_L = inverse @ mf.get_fock(dm=P) @ inverse
    Lambda_L = transform_rank4(cg_ao - disconnected_rhf(P), half)
    for name, p, g in (("standard", dm1_ao, dm2_ao), ("consistent", cp_ao, cg_ao)):
        p_l = half @ p @ half
        g_l = transform_rank4(g, half)
        assert p_l.shape == (mol.nao_nr(),) * 2
        assert g_l.shape == (mol.nao_nr(),) * 4
        print(f"{name} full Lowdin RDM shapes: {p_l.shape}, {g_l.shape}")
        print(f"{name} full Lowdin electron count: {np.trace(p_l):.12f}")
    print(f"Full Lowdin metric residual: {np.max(np.abs(inverse @ S @ inverse - np.eye(mol.nao_nr()))):.3e}")
    selection = select_valence_aos(mol)
    descriptors = extract_descriptors(P_L, F_L, Lambda_L, selection["valence_indices"])
    print("Core AOs excluded only after full Lowdin transformation:", selection["core_indices"])
    print("Valence AO indices:", selection["valence_indices"])
    print("Descriptor shapes:", {key: value.shape for key, value in descriptors.items()})
