"""Current Part 3 geometry validation and explicitly requested missing-FCI work.

Importing this module never imports an electronic-structure engine or solves a
geometry. Input loading uses only the current project's saved JSON documents.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
import json
import math
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
MOLECULES = ("LiH", "BeH2", "H2O", "NH3", "N2", "CO", "HF", "H2S", "H2O2")
# (doubly occupied frozen spatial orbitals, active spatial orbitals, active electrons)
ACTIVE_SPACES = dict(zip(MOLECULES, ((1, 5, 2), (1, 6, 4), (1, 6, 8),
    (1, 7, 8), (2, 8, 10), (2, 8, 10), (1, 5, 8), (5, 6, 8), (2, 10, 14))))
COMPOSITIONS = dict(zip(MOLECULES, (Counter(x) for x in (
    ["Li", "H"], ["Be", "H", "H"], ["O", "H", "H"], ["N", "H", "H", "H"],
    ["N", "N"], ["C", "O"], ["F", "H"], ["S", "H", "H"], ["O", "O", "H", "H"]))))
IDENTITY_FIELDS = ("molecule", "geom_index", "source_point_index", "basis", "symbols",
                   "cartesian_A", "bond_length_angstrom", "q_A", "E_RHF_Ha", "charge", "spin")
FCI_SCHEMA = "current-range-frozen-core-fci-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(not isinstance(value, bool), f"{label}: boolean is not a number.")
    result = float(value)
    require(math.isfinite(result), f"{label}: nonfinite number.")
    return result


def close(actual, expected, label, tolerance=1e-12):
    require(abs(finite(actual, label) - finite(expected, label)) <= tolerance,
            f"{label}: does not match the current saved input.")


def coordinates(value, count):
    require(isinstance(value, list) and len(value) == count, "Invalid coordinate count.")
    require(all(isinstance(v, list) and len(v) == 3 for v in value), "Invalid Cartesian shape.")
    return [[finite(x, "Cartesian coordinate") for x in v] for v in value]


def raw_path(name, symbols, reference, q):
    """Reproduce saved StretchPath.raw geometry without PySCF or rigid alignment."""
    expected = [list(v) for v in reference]
    if name in ("LiH", "N2", "CO", "HF"):
        stretches = [(0, 1)]
    elif name == "H2O2":
        oxygens = [i for i, symbol in enumerate(symbols) if symbol == "O"]
        hydrogens = [i for i, symbol in enumerate(symbols) if symbol == "H"]
        assignments = sorted((sum(math.dist(reference[o], reference[h]) for o, h in zip(oxygens, perm)), perm)
                             for perm in permutations(hydrogens))
        require(assignments[1][0] - assignments[0][0] > 1e-8, "Ambiguous peroxide connectivity.")
        stretches = [tuple(oxygens)]
    else:
        heavy = next(i for i, symbol in enumerate(symbols) if symbol != "H")
        stretches = [(heavy, i) for i, symbol in enumerate(symbols) if symbol == "H"]
    lengths = [math.dist(reference[a], reference[b]) for a, b in stretches]
    for (a, b), length in zip(stretches, lengths):
        require(length > 1e-8 and length + q > 0, "Nonpositive stretch coordinate.")
        expected[b] = [reference[a][k] + (length + q) * (reference[b][k] - reference[a][k]) / length
                       for k in range(3)]
    if name == "H2O2":
        for oxygen, hydrogen in zip(oxygens, assignments[0][1]):
            expected[hydrogen] = [expected[oxygen][k] + reference[hydrogen][k] - reference[oxygen][k]
                                  for k in range(3)]
    return expected, sum(lengths) / len(lengths)


def index_results(document, label, issues):
    grouped = {}
    for position, record in enumerate(document.get("results", [])):
        name = record.get("summary", {}).get("molecule")
        if name not in MOLECULES:
            issues.append(f"{label}: unexpected molecule {name!r} at results[{position}].")
        grouped.setdefault(name, []).append((position, record))
    for name, records in grouped.items():
        if len(records) > 1:
            issues.append(f"{label}: duplicate molecule {name}; all its records excluded.")
    return {name: entries[0] for name, entries in grouped.items() if len(entries) == 1}


def load_current_inputs(ranges_json, scan_json, harmonic_json):
    """Return (validated points, warning strings, molecule -> Part 3 summary)."""
    documents = [json.loads(Path(path).read_text()) for path in (ranges_json, scan_json, harmonic_json)]
    require(documents[0].get("schema_version") == "part3-zero-energy-v2", "Unsupported Part 3 schema.")
    require(documents[1].get("schema_version") == "rhf-current-range-30-v1", "Unsupported current RHF scan schema.")
    require(documents[1].get("points_per_molecule") == 30, "Expected 30 scan points per molecule.")
    issues, points, ranges = [], [], {}
    tables = [index_results(doc, label, issues) for doc, label in zip(documents, ("Part 3", "RHF scan", "harmonic"))]
    for name in MOLECULES:
        if any(name not in table for table in tables):
            issues.append(f"{name}: missing or duplicated molecule in current inputs; geometries 0-29 unavailable.")
            continue
        (_, selected), (scan_position, scan), (_, harmonic) = [table[name] for table in tables]
        summary, saved, hs = selected["summary"], scan["summary"], harmonic["summary"]
        ranges[name] = dict(summary)
        try:
            require(selected.get("input_validation", {}).get("passed") is True, "Part 3 validation failed.")
            require(summary.get("validation_status") == hs.get("validation_status") == "PASS", "Invalid range source.")
            require(not harmonic.get("failures"), "Harmonic source has failures.")
            require(summary["basis"].lower() == saved["basis"].lower() == hs["RHF_basis"].lower() == "sto-3g",
                    "Basis mismatch or unsupported frozen-core basis.")
            require(summary["coordinate"] == saved["coordinate"] == hs["coordinate_definition"], "Coordinate definition mismatch.")
            n = summary["selected_n"]
            require(type(n) is int and n in (0, 1) and saved["selected_n"] == n, "Selected level mismatch.")
            require(summary["reference_shift_Ha"] == saved["reference_shift_Ha"] == 0, "Shifted energy reference.")
            close(summary["r_e_A"], hs["s_eq_A"], "Equilibrium bond coordinate")
            close(saved["r_e_A"], summary["r_e_A"], "Saved equilibrium coordinate")
            close(saved["E_re_Ha"], summary["E_re_Ha"], "Saved equilibrium RHF energy")
            for side in ("min", "max"):
                key = f"selected_range_{side}_A"
                close(summary[key], saved[key], key)
                close(summary[key], hs[f"n{n}_{side}_A"], f"Harmonic {key}")
            lower, upper = [finite(summary[f"selected_range_{side}_A"], side) for side in ("min", "max")]
            require(0 < lower < upper, "Invalid selected range endpoints.")
            grid = [lower + (upper - lower) * j / 29 for j in range(30)]
            grid[0], grid[-1] = lower, upper
            symbols = scan["symbols"]
            require(symbols == harmonic["symbols"] and Counter(symbols) == COMPOSITIONS[name], "Atom identity/order mismatch.")
            reference = coordinates(harmonic["equilibrium_cartesian_A"], len(symbols))
            _, reference_length = raw_path(name, symbols, reference, 0)
            close(reference_length, summary["r_e_A"], "Harmonic geometry equilibrium length", 1e-10)
        except (KeyError, ValueError, TypeError) as exc:
            issues.append(f"{name}: inconsistent current source; geometries 0-29 unavailable: {exc}")
            continue
        rows = scan.get("points", [])
        counts = Counter(row.get("point_index") for row in rows)
        missing = [j - 1 for j in range(1, 31) if counts[j] == 0]
        duplicates = [j - 1 for j in range(1, 31) if counts[j] > 1]
        unexpected = [index for index in counts if type(index) is not int or index not in range(1, 31)]
        if missing:
            issues.append(f"{name}: missing geom_index values {missing}.")
        if duplicates:
            issues.append(f"{name}: duplicate geom_index values {duplicates}; excluded.")
        if unexpected:
            issues.append(f"{name}: unexpected source point indices {unexpected}.")
        for row_position, row in enumerate(rows):
            index = row.get("point_index")
            if type(index) is not int or index not in range(1, 31) or counts[index] != 1:
                continue
            try:
                require(row["molecule"] == name and row["basis"].lower() == "sto-3g", "Point molecule/basis mismatch.")
                require(row["selected_n"] == n, "Point selected level mismatch.")
                require(row.get("status") == "PASS" and row.get("scf_converged") is True
                        and row.get("internal_stable") is True and row.get("geometry_validation", {}).get("passed") is True,
                        "Saved RHF/geometry did not pass validation.")
                close(row["bond_length_A"], grid[index - 1], "Current range grid point")
                close(row["q_A"], grid[index - 1] - summary["r_e_A"], "Displacement q")
                coords = coordinates(row["cartesian_A"], len(symbols))
                expected, _ = raw_path(name, symbols, reference, row["q_A"])
                for a, b in combinations(range(len(symbols)), 2):
                    close(math.dist(coords[a], coords[b]), math.dist(expected[a], expected[b]),
                          f"Nuclear geometry distance {a}-{b}", 5e-10)
                energy = finite(row["E_RHF_Ha"], "Absolute RHF energy")
                close(row["diagnostics"]["energy_Eh"], energy, "RHF diagnostics energy")
                close(row["E_re_Ha"], summary["E_re_Ha"], "Point equilibrium reference")
                close(row["delta_E_RHF_Ha"], energy - row["E_re_Ha"], "RHF absolute/relative convention")
                points.append({"molecule": name, "geom_index": index - 1, "source_point_index": index,
                    "basis": row["basis"], "symbols": symbols, "cartesian_A": coords,
                    "bond_length_angstrom": row["bond_length_A"], "q_A": row["q_A"], "E_RHF_Ha": energy,
                    "charge": 0, "spin": 0,
                    "rhf_source_file": str(Path(scan_json).resolve()),
                    "rhf_source_key": f"results[{scan_position}].points[{row_position}].E_RHF_Ha"})
            except (KeyError, ValueError, TypeError) as exc:
                issues.append(f"{name} geom_index={index - 1}: excluded: {exc}")
    return sorted(points, key=lambda p: (MOLECULES.index(p["molecule"]), p["geom_index"])), issues, ranges


def validate_fci_record(record, point):
    """Reject cache records from another grid, reference, or energy convention."""
    require(record.get("schema_version") == FCI_SCHEMA, "Unsupported FCI cache schema.")
    for key in IDENTITY_FIELDS:
        require(record.get(key) == point[key], f"FCI cache identity mismatch: {key}.")
    ncore, ncas, nelec = ACTIVE_SPACES[point["molecule"]]
    for key, value in (("n_frozen_orbitals", ncore), ("n_frozen_electrons", 2 * ncore),
                       ("n_active_orbitals", ncas), ("n_active_electrons", nelec)):
        require(type(record.get(key)) is int and record[key] == value, f"FCI cache convention mismatch: {key}.")
    require(record.get("fci_converged") is True and record.get("total_energy_confirmed") is True,
            "FCI total energy or convergence is unconfirmed.")
    require(record.get("canonical_rhf_orbitals") is True, "Canonical RHF frozen core is unconfirmed.")
    close(record["E_FCI_frozen_core_Ha"], finite(record["E_CAS_active_Ha"], "Active CI energy")
          + finite(record["E_core_including_nuclear_Ha"], "Core energy"), "FCI total decomposition", 1e-9)
    require(finite(record["E_nuclear_Ha"], "Nuclear repulsion") > 0, "Invalid nuclear energy.")
    close(record["E_RHF_reconstructed_Ha"], point["E_RHF_Ha"], "Reconstructed RHF reference", 1e-9)
    require(bool(record.get("source_function")) and bool(record.get("pyscf_version")), "Missing FCI provenance.")


def compute_missing_fci(point, threads=1, allow_rhf_rebuild=False):
    """Compute only missing FCI; rebuilding unavailable RHF orbitals needs opt-in."""
    require(allow_rhf_rebuild is True,
            "Missing FCI requires unavailable RHF orbitals; explicitly allow RHF orbital reconstruction first.")
    require(type(threads) is int and threads > 0, "threads must be a positive integer.")
    sys.path.insert(0, str(PROJECT / "tests"))
    import numpy as np
    import pyscf
    from pyscf import lib, mcscf
    import rhf_bound_level_selection as engine
    lib.num_threads(threads)
    mf, state = engine.solve_point(point["symbols"], np.array(point["cartesian_A"]), point["basis"])
    require(mf is not None and state.get("accepted") is True, "RHF orbital reconstruction failed.")
    close(mf.e_tot, point["E_RHF_Ha"], "Reconstructed RHF reference", 1e-9)
    ncore, ncas, nelec = ACTIVE_SPACES[point["molecule"]]
    require(mf.mol.nelectron == nelec + 2 * ncore and mf.mo_coeff.shape[1] == ncas + ncore,
            "RHF orbital/electron count differs from the frozen-core convention.")
    mf.mo_energy, mf.mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    require(np.all(np.isin(mf.mo_occ, [0, 2])) and np.all(np.diff(mf.mo_occ) <= 0)
            and np.all(mf.mo_occ[:ncore] == 2), "Invalid closed-shell orbital occupations.")
    occupied = mf.mo_energy[mf.mo_occ == 2]
    require(np.all(np.diff(occupied) >= -1e-10), "Occupied canonical orbitals are not energy ordered.")
    mc = mcscf.CASCI(mf, ncas, nelec, ncore=ncore)
    mc.fcisolver.conv_tol, mc.fcisolver.max_cycle = 1e-12, 200
    mc.verbose = 0
    total = finite(mc.kernel()[0], "CASCI total energy")
    core = finite(mc.get_h1eff()[1], "Core including nuclear energy")
    result = {key: point[key] for key in IDENTITY_FIELDS}
    result.update(schema_version=FCI_SCHEMA, n_frozen_orbitals=ncore, n_frozen_electrons=2 * ncore,
        n_active_orbitals=ncas, n_active_electrons=nelec, E_FCI_frozen_core_Ha=total,
        E_CAS_active_Ha=finite(mc.e_cas, "Active CI energy"), E_core_including_nuclear_Ha=core,
        E_nuclear_Ha=finite(mf.mol.energy_nuc(), "Nuclear repulsion"),
        E_RHF_reconstructed_Ha=float(mf.e_tot), fci_converged=bool(mc.fcisolver.converged),
        total_energy_confirmed=True, canonical_rhf_orbitals=True, rhf_orbitals_reconstructed=True,
        pyscf_version=pyscf.__version__, source_function="pyscf.mcscf.CASCI.kernel()[0]",
        construction="E_FCI_total = mc.e_cas + mc.get_h1eff()[1]; core includes nuclear repulsion",
        rhf_source_file=point["rhf_source_file"], rhf_source_key=point["rhf_source_key"])
    validate_fci_record(result, point)
    return result
