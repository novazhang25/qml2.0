#!/usr/bin/env python3
"""Calculate 30 fresh RHF energies per molecule over its current selected range.

Reads the corrected Part 3 JSON and its matching saved harmonic geometry.
Uses evenly spaced bond lengths, including both endpoints, and the existing
collective stretching path. No geometry optimization, nuclear Hessian, range
selection, vibrational-energy addition, or dissociation scan is performed.
Use --rebind-part3 to update only saved selection metadata after verifying that
the existing numerical grid, geometry source, and RHF inputs are unchanged.
Run with the project's .venv-rhf Python. Nothing is calculated on import.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
POINT_COUNT = 30
PART3_SCHEMA = "part3-zero-energy-v2"
MOLECULES = ("LiH", "BeH2", "H2O", "NH3", "N2", "CO", "HF", "H2S", "H2O2")
CSV_FIELDS = (
    "molecule", "basis", "selected_n", "point_index", "bond_length_A", "q_A",
    "E_re_Ha", "E_RHF_Ha", "delta_E_RHF_Ha", "status", "scf_converged",
    "internal_stable", "orbital_gradient_norm_Eh", "failure",
)
SUMMARY_FIELDS = (
    "molecule", "basis", "selected_n", "selected_range_min_A", "selected_range_max_A",
    "requested_points", "accepted_points", "failed_points", "status",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    value = float(value)
    require(math.isfinite(value), f"{name} must be finite.")
    return value


def equal_number(actual, expected, name, tolerance=1e-12):
    require(abs(finite(actual, name) - finite(expected, name)) <= tolerance,
            f"{name} differs from the saved input.")


def build_grid(lower, upper):
    """Exactly 30 ascending, equally spaced coordinates including saved endpoints."""
    lower, upper = finite(lower, "lower endpoint"), finite(upper, "upper endpoint")
    require(0 < lower < upper, "Endpoints must satisfy 0 < lower < upper.")
    grid = [lower + (upper - lower) * index / (POINT_COUNT - 1) for index in range(POINT_COUNT)]
    grid[0], grid[-1] = lower, upper
    require(all(a < b for a, b in zip(grid, grid[1:])), "Range is too narrow for 30 distinct floats.")
    return grid


def validate_range_record(record, harmonic_record):
    """Consume the saved selection unchanged; never infer or widen its endpoints."""
    summary, harmonic = record["summary"], harmonic_record["summary"]
    name = summary["molecule"]
    require(record["input_validation"]["passed"] is True, f"{name}: Part 3 input validation failed.")
    require(summary["validation_status"] == "PASS", f"{name}: no valid selected range.")
    require(harmonic["validation_status"] == "PASS" and not harmonic_record.get("failures"),
            f"{name}: harmonic input did not pass validation.")
    require(name == harmonic["molecule"], "Molecule mismatch between range and geometry.")
    require(summary["basis"].lower() == harmonic["RHF_basis"].lower(), f"{name}: basis mismatch.")
    require(summary["coordinate"] == harmonic["coordinate_definition"], f"{name}: coordinate mismatch.")
    equal_number(summary["r_e_A"], harmonic["s_eq_A"], f"{name}: equilibrium coordinate")
    re = finite(summary["E_re_Ha"], "E_re")
    equal_number(re, harmonic_record["stationarity"]["E_RHF_Eh"], "equilibrium RHF energy", 1e-9)
    require(summary["decision_criterion"] == "E_total_n < 0 Ha", f"{name}: incorrect Part 3 criterion.")
    require(summary["reference_shift_Ha"] == 0, f"{name}: Part 3 must not shift the equilibrium energy.")
    require(summary["energy_reference_status"] == "UNVALIDATED_ZERO_REFERENCE"
            and bool(summary["energy_reference_warning"]), f"{name}: missing energy-reference warning.")
    for key, harmonic_key in (("k_Ha_per_Bohr2", "k_q_Eh_per_Bohr2"),
                              ("k_Ha_per_A2", "k_q_Eh_per_A2"),
                              ("k_N_per_m", "k_q_N_per_m"),
                              ("mu_eff_amu", "mu_eff_amu"), ("mu_eff_kg", "mu_eff_kg"),
                              ("omega_rad_per_s", "omega_rad_per_s")):
        require(summary[key] == harmonic[harmonic_key], f"{name}: changed saved {key}.")
    quantum = finite(summary["hbar_omega_Ha"], "hbar_omega")
    require(quantum > 0, "Harmonic energy quantum must be positive.")
    for n in (0, 1):
        excitation = finite(summary[f"E_n{n}_Ha"], f"E_n{n}")
        require(excitation > 0, "Harmonic level energy must be positive above equilibrium.")
        equal_number(excitation, (n + 0.5) * quantum, f"E_n{n}")
        total = finite(summary[f"E_total_n{n}_Ha"], f"E_total_n{n}")
        equal_number(total, re + excitation, f"E_total_n{n}")
        require(summary[f"n{n}_pass"] is (total < 0), f"{name}: inconsistent n={n} zero-test flag.")
    selected = summary["selected_n"]
    require(type(selected) is int and selected in (0, 1), f"{name}: no selected n=0/1 range.")
    expected = 1 if summary["n1_pass"] else 0 if summary["n0_pass"] else None
    require(selected == expected, f"{name}: selected level disagrees with current Part 3 classification.")
    for side in ("min", "max"):
        endpoint = summary[f"selected_range_{side}_A"]
        require(endpoint == summary[f"n{selected}_{side}_A"] == harmonic[f"n{selected}_{side}_A"],
                f"{name}: selected {side} endpoint differs from the saved harmonic range.")
    grid = build_grid(summary["selected_range_min_A"], summary["selected_range_max_A"])
    require(grid[0] < summary["r_e_A"] < grid[-1], f"{name}: selected range does not bracket equilibrium.")
    return copy.deepcopy(summary)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_records(document):
    indexed = {}
    for record in document["results"]:
        name = record["summary"]["molecule"]
        require(name not in indexed, f"Duplicate molecule: {name}")
        indexed[name] = record
    return indexed


def load_inputs(ranges_path, harmonic_path, molecules):
    require(bool(molecules) and len(molecules) == len(set(molecules)), "Molecules must be nonempty and unique.")
    range_doc = json.loads(Path(ranges_path).read_text())
    harmonic_doc = json.loads(Path(harmonic_path).read_text())
    require(range_doc["schema_version"] == PART3_SCHEMA, "Expected the zero-energy Part 3 JSON schema.")
    source_hashes = range_doc["provenance"]["source_files_sha256"]
    require(sha256(harmonic_path) in source_hashes.values(),
            "The harmonic geometry file is not the input used by the current Part 3 ranges.")
    ranges, harmonic = index_records(range_doc), index_records(harmonic_doc)
    constants = harmonic_doc["runtime"]["constants"]
    hbar = finite(constants["hbar_J_s"], "saved hbar_J_s")
    hartree = finite(constants["Hartree_J"], "saved Hartree_J")
    require(hbar > 0 and hartree > 0, "Saved energy-conversion constants must be positive.")
    pairs = []
    for name in molecules:
        require(name in ranges and name in harmonic, f"Missing current inputs for {name}.")
        validate_range_record(ranges[name], harmonic[name])
        summary = ranges[name]["summary"]
        equal_number(summary["hbar_omega_Ha"], hbar * summary["omega_rad_per_s"] / hartree,
                     f"{name}: saved harmonic-frequency conversion to Hartree")
        pairs.append((ranges[name], harmonic[name]))
    return range_doc, harmonic_doc, pairs


def prepare_plans(pairs, engine):
    """Validate every geometry for every molecule before starting any RHF solve."""
    plans = []
    for record, harmonic in pairs:
        summary = validate_range_record(record, harmonic)
        path, symbols, basis = engine.prepare_path(harmonic)
        points = []
        for index, length in enumerate(build_grid(summary["selected_range_min_A"],
                                                   summary["selected_range_max_A"]), 1):
            q = length - path.s_eq
            coords = path(q)
            geometry = engine.validate_scan_geometry(path, q, coords)
            require(geometry["passed"] is True, f"{summary['molecule']}: invalid geometry at point {index}.")
            points.append({"point_index": index, "bond_length_A": length, "q_A": q,
                           "cartesian_A": coords, "geometry_validation": geometry})
        plans.append({"summary": summary, "path": path, "symbols": symbols, "basis": basis, "points": points})
    return plans


def scan_molecule(plan, engine, on_progress=None):
    """Calculate each distinct geometry once, retaining failed attempts explicitly."""
    summary = {**copy.deepcopy(plan["summary"]), "requested_points": POINT_COUNT,
               "accepted_points": 0, "failed_points": 0, "status": "RUNNING"}
    require(len(plan["points"]) == POINT_COUNT, "A molecule must have exactly 30 planned points.")
    result = {"summary": summary, "symbols": plan["symbols"], "points": []}
    previous = None
    for point in plan["points"]:
        row = {**point, "molecule": summary["molecule"], "basis": plan["basis"],
               "selected_n": summary["selected_n"], "E_re_Ha": summary["E_re_Ha"],
               "E_RHF_Ha": None, "delta_E_RHF_Ha": None, "status": "FAIL_RHF",
               "scf_converged": False, "internal_stable": False,
               "orbital_gradient_norm_Eh": None, "failure": "", "diagnostics": {}}
        try:
            require(point["geometry_validation"]["passed"] is True, "Geometry validation failed.")
            mf, state = engine.solve_point(plan["symbols"], point["cartesian_A"], plan["basis"], previous=previous)
            row["diagnostics"] = state
            for key in ("scf_converged", "internal_stable", "orbital_gradient_norm_Eh"):
                row[key] = state.get(key)
            require(mf is not None and state.get("accepted") is True
                    and state.get("scf_converged") is True and state.get("internal_stable") is True,
                    state.get("failure", "RHF convergence or internal stability failed."))
            energy = finite(state["energy_Eh"], "RHF energy")
            gradient = finite(state["orbital_gradient_norm_Eh"], "RHF orbital gradient")
            require(0 <= gradient <= 1e-10, "RHF orbital gradient exceeds 1e-10 Ha.")
            row.update(E_RHF_Ha=energy, delta_E_RHF_Ha=energy - summary["E_re_Ha"], status="PASS")
            summary["accepted_points"] += 1
            previous = mf
        except Exception as exc:
            row["failure"] = f"{type(exc).__name__}: {exc}"
            summary["failed_points"] += 1
            previous = None  # A failed state is never used to seed the next point.
        result["points"].append(row)
        if len(result["points"]) == POINT_COUNT:
            summary["status"] = "PASS" if summary["accepted_points"] == POINT_COUNT else "FAIL_RHF"
        if on_progress is not None:
            on_progress(result, row)
    return result


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_outputs(document, output_dir, engine, current_name):
    """Checkpoint after each point, including complete coordinates and diagnostics."""
    payload = engine.harmonic.jsonable(document)
    all_rows = [row for result in payload["results"] for row in result["points"]]
    atomic_text(output_dir / "rhf_scan_30.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    atomic_text(output_dir / "rhf_scan_30.csv", csv_text(all_rows, CSV_FIELDS))
    atomic_text(output_dir / "summary.csv", csv_text((r["summary"] for r in payload["results"]), SUMMARY_FIELDS))
    current = next((r for r in payload["results"] if r["summary"]["molecule"] == current_name), None)
    if current is not None:
        atomic_text(output_dir / "molecules" / f"{current_name}.csv", csv_text(current["points"], CSV_FIELDS))
        xyz = []
        for row in current["points"]:
            xyz += [str(len(current["symbols"])),
                    f"{current_name} RHF/{row['basis']} point={row['point_index']} "
                    f"bond_length_A={row['bond_length_A']:.15g} E_RHF_Ha={row['E_RHF_Ha']} status={row['status']}"]
            xyz += [f"{symbol} {r[0]:.14f} {r[1]:.14f} {r[2]:.14f}"
                    for symbol, r in zip(current["symbols"], row["cartesian_A"])]
        atomic_text(output_dir / "geometries" / f"{current_name}.xyz", "\n".join(xyz) + "\n")


def numerical_payload_digest(document):
    """Fingerprint every saved atom, point, energy, geometry check and solver diagnostic."""
    numerical = [{"molecule": result["summary"]["molecule"],
                  "symbols": result["symbols"], "points": result["points"]}
                 for result in document["results"]]
    return hashlib.sha256(json.dumps(numerical, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def rebind_saved_scan(scan_path, ranges_path, harmonic_path):
    """Migrate selection metadata without importing or executing a numerical engine.

    Refuse changed ranges or physical inputs: no existing point may be reassigned
    to a different level, geometry, basis, or equilibrium reference. Historical
    source hashes remain attached to the original numerical run, while a separate
    record identifies the code and inputs used only for this metadata migration.
    """
    scan_path, ranges_path, harmonic_path = map(Path, (scan_path, ranges_path, harmonic_path))
    original = json.loads(scan_path.read_text())
    require(original["schema_version"] == "rhf-current-range-30-v1" and original["status"] == "PASS",
            "Metadata rebinding requires a completed PASS RHF scan.")
    require(original["points_per_molecule"] == POINT_COUNT, "Expected 30 saved points per molecule.")
    names = original["requested_molecules"]
    require(original["requested_total_points"] == POINT_COUNT * len(names), "Incorrect saved point count.")
    range_doc, harmonic_doc, pairs = load_inputs(ranges_path, harmonic_path, names)
    provenance = original["provenance"]
    require(provenance["harmonic_sha256"] == sha256(harmonic_path),
            "The harmonic geometry input changed; existing RHF data cannot be rebound.")
    require(provenance["saved_constants"] == harmonic_doc["runtime"]["constants"],
            "The physical constants changed; existing RHF data cannot be rebound.")
    geometry_path = PROJECT / "tests/new_rhf_harmonic_bond_ranges.py"
    require(provenance["source_hashes"]["geometry"] == sha256(geometry_path),
            "The geometry builder changed; existing RHF data cannot be rebound.")
    saved_by_name = index_records(original)
    require(set(saved_by_name) == set(names), "The saved molecule set is incomplete.")
    migrated = copy.deepcopy(original)
    migrated_by_name = index_records(migrated)
    unchanged = ("molecule", "basis", "coordinate", "r_e_A", "E_re_Ha", "selected_n",
                 "selected_range_min_A", "selected_range_max_A", "k_Ha_per_Bohr2", "k_Ha_per_A2",
                 "k_N_per_m", "mu_eff_amu", "mu_eff_kg", "omega_rad_per_s",
                 "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A", "Delta_q_n0_A", "Delta_q_n1_A")
    for part3_record, harmonic_record in pairs:
        current = part3_record["summary"]
        name = current["molecule"]
        saved = saved_by_name[name]
        old = saved["summary"]
        for key in unchanged:
            require(old[key] == current[key], f"{name}: changed {key}; refuse to reuse the saved RHF grid.")
        for current_key, legacy_key in (("hbar_omega_Ha", "harmonic_quantum_Ha"),
                                        ("E_n0_Ha", "E_0_Ha"), ("E_n1_Ha", "E_1_Ha"),
                                        ("E_total_n0_Ha", "E_total_0_Ha"),
                                        ("E_total_n1_Ha", "E_total_1_Ha")):
            equal_number(old[current_key] if current_key in old else old[legacy_key],
                         current[current_key], f"{name}: saved {current_key}")
        require(saved["symbols"] == harmonic_record["symbols"], f"{name}: atom identities changed.")
        require(old["status"] == "PASS" and old["requested_points"] == POINT_COUNT
                and old["accepted_points"] == POINT_COUNT and old["failed_points"] == 0,
                f"{name}: incomplete RHF results.")
        points = saved["points"]
        require(len(points) == POINT_COUNT
                and [point["point_index"] for point in points] == list(range(1, POINT_COUNT + 1)),
                f"{name}: missing or unordered RHF grid.")
        expected = build_grid(current["selected_range_min_A"], current["selected_range_max_A"])
        for point, length in zip(points, expected):
            require(point["bond_length_A"] == length and point["q_A"] == length - current["r_e_A"],
                    f"{name}: changed numeric grid; refuse to reuse the saved RHF energies.")
            for key in ("molecule", "basis", "selected_n", "E_re_Ha"):
                require(point[key] == current[key], f"{name}: changed point {key}.")
            require(point["status"] == "PASS" and point["scf_converged"] is True
                    and point["internal_stable"] is True and point["geometry_validation"]["passed"] is True,
                    f"{name}: a saved RHF point failed validation.")
            energy = finite(point["E_RHF_Ha"], f"{name}: RHF energy")
            require(point["delta_E_RHF_Ha"] == energy - current["E_re_Ha"],
                    f"{name}: changed energy reference.")
            coords = point["cartesian_A"]
            require(len(coords) == len(saved["symbols"])
                    and all(len(atom) == 3 and all(math.isfinite(value) for value in atom) for atom in coords),
                    f"{name}: invalid saved Cartesian geometry.")
        migrated_by_name[name]["summary"] = {
            **copy.deepcopy(current), **{key: old[key] for key in
                                        ("requested_points", "accepted_points", "failed_points", "status")}}
    before = numerical_payload_digest(original)
    require(numerical_payload_digest(migrated) == before, "Metadata migration altered numerical RHF data.")
    target_hash = sha256(ranges_path)
    if provenance["ranges_sha256"] == target_hash and migrated["results"] == original["results"]:
        return {"changed": False, "molecules": len(names), "numerical_payload_sha256": before}
    updated = migrated["provenance"]
    updated.setdefault("original_numerical_run_provenance", copy.deepcopy(provenance))
    updated.setdefault("part3_metadata_migrations", []).append({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "old_ranges_sha256": provenance["ranges_sha256"], "new_ranges_sha256": target_hash,
        "metadata_script_sha256": sha256(__file__), "harmonic_sha256": sha256(harmonic_path),
        "numerical_payload_sha256_before": before, "numerical_payload_sha256_after": before,
        "SCF_run": False, "points_modified": False,
        "verified": "Unchanged grid, selected range, basis, coordinate, geometry source, harmonic input, "
                    "equilibrium energy, force constant, mass, frequency and complete numerical payload.",
    })
    updated.update(ranges_json=str(ranges_path.resolve()), ranges_sha256=target_hash,
                   ranges_created_utc=range_doc["created_utc"],
                   source_hashes_scope="Original numerical RHF run; metadata migration hashes are recorded separately.")
    atomic_text(scan_path, json.dumps(migrated, indent=2, allow_nan=False) + "\n")
    # Point CSVs and XYZ files already contain the same data and are left intact.
    atomic_text(scan_path.parent / "summary.csv",
                csv_text((record["summary"] for record in migrated["results"]), SUMMARY_FIELDS))
    return {"changed": True, "molecules": len(names), "numerical_payload_sha256": before}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranges-json", type=Path, default=PROJECT / "results/bond_length_part3/vibrational_levels.json")
    parser.add_argument("--harmonic-json", type=Path, default=PROJECT / "json/new_rhf_harmonic_bond_ranges.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/rhf_30_point_scans")
    parser.add_argument("--molecules", nargs="+", choices=MOLECULES, default=list(MOLECULES))
    parser.add_argument("--threads", type=int, default=1, help="PySCF threads; default: 1.")
    parser.add_argument("--rebind-part3", action="store_true",
                        help="Update only existing scan metadata after verifying unchanged numerical inputs; no RHF run.")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    return args


def main(argv=None):
    args = parse_args(argv)
    start = time.monotonic()
    try:
        if args.rebind_part3:
            report = rebind_saved_scan(args.output_dir / "rhf_scan_30.json", args.ranges_json, args.harmonic_json)
            print(f"Part 3 metadata verified for {report['molecules']} molecules; changed={report['changed']}. "
                  "All RHF points and energies preserved; no SCF run.")
            return 0
        range_doc, harmonic_doc, pairs = load_inputs(args.ranges_json, args.harmonic_json, args.molecules)
        # The existing numerical engines live in tests/, independently of cwd.
        sys.path.insert(0, str(PROJECT / "tests"))
        import rhf_bound_level_selection as engine
        engine.lib.num_threads(args.threads)
        engine.load_harmonic_input(args.harmonic_json)  # Verify saved builder hash and physical constants.
        plans = prepare_plans(pairs, engine)
        document = {
            "schema_version": "rhf-current-range-30-v1", "status": "RUNNING",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "requested_molecules": args.molecules, "points_per_molecule": POINT_COUNT,
            "requested_total_points": POINT_COUNT * len(plans),
            "grid": "30 uniformly spaced absolute bond coordinates, both endpoints included",
            "energy_definition": "E_RHF_Ha is absolute RHF total energy including nuclear repulsion; "
                                 "delta_E_RHF_Ha = E_RHF_Ha - saved E_re_Ha. No vibrational energy is added.",
            "provenance": {
                "ranges_json": str(args.ranges_json.resolve()), "ranges_sha256": sha256(args.ranges_json),
                "harmonic_json": str(args.harmonic_json.resolve()), "harmonic_sha256": sha256(args.harmonic_json),
                "source_hashes": {"scan": sha256(__file__), "solver": sha256(engine.__file__),
                                  "geometry": sha256(engine.harmonic.__file__)},
                "python": sys.version, "pyscf": engine.pyscf.__version__, "threads": args.threads,
                "ranges_created_utc": range_doc["created_utc"],
                "saved_constants": harmonic_doc["runtime"]["constants"],
                "method": "all-electron restricted closed-shell Hartree-Fock; no density fitting",
                "geometry_reoptimized": False, "harmonic_ranges_recomputed": False,
                "SCF_acceptance": "energy convergence 1e-13 Ha; orbital gradient <=1e-10 Ha; "
                                  "internal RHF stability and orbital/electron consistency required",
            },
            "results": [{"summary": {**copy.deepcopy(plan["summary"]), "requested_points": POINT_COUNT,
                                     "accepted_points": 0, "failed_points": 0, "status": "PENDING"},
                         "symbols": plan["symbols"], "points": []} for plan in plans],
        }
        for index, plan in enumerate(plans):
            name = plan["summary"]["molecule"]
            print(f"{name}: 30 RHF points from {plan['summary']['selected_range_min_A']:.12f} "
                  f"to {plan['summary']['selected_range_max_A']:.12f} Angstrom", flush=True)
            def progress(result, row):
                document["results"][index] = result
                document["elapsed_seconds"] = time.monotonic() - start
                write_outputs(document, args.output_dir, engine, name)
                energy = "FAILED" if row["E_RHF_Ha"] is None else f"{row['E_RHF_Ha']:.14f} Ha"
                print(f"  {name} {row['point_index']:02d}/30: r={row['bond_length_A']:.9f} A; {energy}", flush=True)
            document["results"][index] = scan_molecule(plan, engine, progress)
        passed = all(r["summary"]["status"] == "PASS" for r in document["results"])
        document["status"] = "PASS" if passed else "FAIL_RHF"
        document["elapsed_seconds"] = time.monotonic() - start
        write_outputs(document, args.output_dir, engine, plans[-1]["summary"]["molecule"])
        accepted = sum(r["summary"]["accepted_points"] for r in document["results"])
        print(f"{document['status']}: {accepted}/{document['requested_total_points']} accepted RHF points. "
              f"Results: {args.output_dir.resolve()}", flush=True)
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, TypeError, ImportError, ArithmeticError) as exc:
        print(f"RHF scan failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
