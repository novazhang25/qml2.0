"""Read-only discovery and cross-file validation for initialdata-v1 scans.

The JSON Cartesian coordinates retain the producer's full precision. XYZ files
are independently checked against them, allowing only the writer's rounding.
No energy is calculated or repaired by this module.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MOLECULES = ("LiH", "HF", "BeH2", "H2O", "H2S", "NH3", "N2", "CO", "H2O2")
ELEMENTS = ("H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar")
ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(ELEMENTS)}
FORMULAS = {
    "LiH": {"Li": 1, "H": 1}, "HF": {"H": 1, "F": 1},
    "BeH2": {"Be": 1, "H": 2}, "H2O": {"O": 1, "H": 2},
    "H2S": {"S": 1, "H": 2}, "NH3": {"N": 1, "H": 3},
    "N2": {"N": 2}, "CO": {"C": 1, "O": 1}, "H2O2": {"H": 2, "O": 2},
}
POINT_FIELDS = (
    "molecule", "point_index", "basis", "charge", "spin", "symbols",
    "cartesian_A", "bond_length_A", "q_A", "status", "E_RHF_Ha", "E_FCI_Ha",
    "E_FCI_frozen_core_Ha", "E_corr_Ha", "E_corr_mHa", "E_CAS_active_Ha",
    "E_core_including_nuclear_Ha", "E_nuclear_Ha", "n_frozen_orbitals",
    "n_frozen_electrons", "n_active_orbitals", "n_active_electrons",
    "canonical_rhf_orbitals", "fci_converged", "fci_convention",
    "total_energy_includes_nuclear_repulsion", "electronic_state_stage",
    "rhf_diagnostics", "geometry_validation",
)
CSV_NUMERIC_FIELDS = (
    "bond_length_A", "q_A", "E_RHF_Ha", "E_FCI_Ha", "E_corr_Ha", "E_corr_mHa",
    "E_CAS_active_Ha", "E_core_including_nuclear_Ha", "E_nuclear_Ha",
    "n_frozen_orbitals", "n_active_orbitals", "n_active_electrons",
)


class InputError(ValueError):
    """An input convention, geometry, or energy record is inconsistent."""


@dataclass
class InputGeometry:
    molecule: str
    geometry_id: str
    point_index: int
    geometry_path: Path
    state_path: Path
    symbols: list[str]
    coordinates: np.ndarray
    charge: int
    spin: int
    basis: str
    electron_count: int
    input_fci_frozen_core: int
    q_A: float
    bond_length_A: float
    E_RHF_input: float
    E_FCI: float
    E_corr_input: float
    raw: dict


def require(condition, message):
    if not condition:
        raise InputError(message)


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise InputError(f"Cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def read_xyz(path):
    """Return (symbols, Cartesian coordinates, comment) from a strict XYZ file."""
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
        count = int(lines[0])
        require(count > 0 and len(lines) >= 2, f"Invalid XYZ header: {path}")
        rows = [line.split() for line in lines[2:] if line.strip()]
        require(len(rows) == count and all(len(row) == 4 for row in rows),
                f"XYZ atom count or coordinate columns differ: {path}")
        symbols = [row[0] for row in rows]
        coordinates = np.asarray([[float(x) for x in row[1:]] for row in rows])
        require(np.isfinite(coordinates).all(), f"Nonfinite XYZ coordinates: {path}")
        return symbols, coordinates, lines[1]
    except (OSError, ValueError, IndexError) as exc:
        raise InputError(f"Cannot read XYZ {path}: {exc}") from exc


def _same_point(left, right, location):
    for field in POINT_FIELDS:
        require(field in left and field in right and left[field] == right[field],
                f"{location}: duplicated {field} differs or is missing")
    for field in ("geometry_file", "electronic_state_file"):
        require(left[field]["path"] == right[field]["path"],
                f"{location}: duplicated {field} path differs")


def _source_active_spaces(path):
    """Read the producer's literal convention without importing/running it."""
    try:
        module = ast.parse(Path(path).read_text())
        for node in module.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "ACTIVE_SPACES"
                    for target in node.targets):
                return ast.literal_eval(node.value)
    except (OSError, SyntaxError, ValueError) as exc:
        raise InputError(f"Cannot inspect source active spaces: {exc}") from exc
    raise InputError("Source snapshot has no literal ACTIVE_SPACES convention")


def _point_record(root, name, point, local_point, csv_row, settings):
    index = point["point_index"]
    require(type(index) is int and 1 <= index <= 30, "Invalid scan point_index")
    identifier = f"{index:03d}"
    _same_point(point, local_point, f"{name}/{identifier} result.json")
    folder = root / name / "scan" / identifier
    individual = _read_json(folder / "point.json")
    _same_point(point, individual, f"{name}/{identifier} point.json")
    require(point["geometry_file"]["path"] == f"{name}/scan/{identifier}/geometry.xyz",
            "Geometry path does not agree with molecule and geometry ID")
    require(point["electronic_state_file"]["path"] == f"{name}/scan/{identifier}/electronic_state.npz",
            "Electronic-state path does not agree with molecule and geometry ID")
    geometry_path, state_path = folder / "geometry.xyz", folder / "electronic_state.npz"
    require(point["molecule"] == name, "Point molecule does not match its directory")
    require(point["status"] == "PASS" and point["fci_converged"] is True,
            "Stored target point did not pass FCI validation")
    require(point["rhf_diagnostics"]["accepted"] is True
            and point["rhf_diagnostics"]["scf_converged"] is True,
            "Stored RHF point is not converged and accepted")
    require(point["geometry_validation"]["passed"] is True,
            "Stored geometry validation did not pass")
    require(point["electronic_state_stage"] == "RHF_AND_FCI", "Incomplete target state")
    require(point["basis"] == settings["basis"] == "sto-3g", "STO-3G convention differs")
    require(type(point["charge"]) is int and point["charge"] == settings["charge"],
            "Charge metadata differs")
    require(type(point["spin"]) is int and point["spin"] == settings["spin"] == 0,
            "System is not an explicitly documented closed-shell singlet")
    require(point["fci_convention"] == settings["fci_convention"] == "frozen-core",
            "FCI frozen-core convention differs or is not explicit")
    require(point["canonical_rhf_orbitals"] is True
            and point["total_energy_includes_nuclear_repulsion"] is True,
            "Target canonical-orbital/total-energy convention differs")
    symbols = point["symbols"]
    require(Counter(symbols) == Counter(FORMULAS[name]), "Molecular atom composition differs")
    coordinates = np.asarray(point["cartesian_A"], dtype=float)
    require(coordinates.shape == (len(symbols), 3) and np.isfinite(coordinates).all(),
            "Invalid Cartesian geometry")
    xyz_symbols, xyz_coordinates, comment = read_xyz(geometry_path)
    require(xyz_symbols == symbols, "XYZ atom ordering differs from energy row")
    xyz_error = float(np.max(np.abs(xyz_coordinates - coordinates)))
    require(xyz_error <= 1e-14, f"XYZ/metadata coordinates differ by {xyz_error:.3e} Angstrom")
    match = re.fullmatch(rf"{re.escape(name)} point=(\d+); q=([^;]+); units=Angstrom", comment)
    require(match is not None and int(match.group(1)) == index,
            "XYZ comment does not establish the same geometry ID and Angstrom units")
    require(abs(float(match.group(2)) - point["q_A"]) <= 1e-14,
            "XYZ comment scan displacement differs")
    require(csv_row["molecule"] == name and int(csv_row["point_index"]) == index
            and csv_row["basis"] == point["basis"] and csv_row["status"] == "PASS",
            "CSV identity, basis, or status differs")
    for field in CSV_NUMERIC_FIELDS:
        require(math.isfinite(float(point[field])) and float(csv_row[field]) == point[field],
                f"scan.csv {field} differs or is nonfinite")
    correlation_error = abs(point["E_corr_Ha"] - (point["E_FCI_Ha"] - point["E_RHF_Ha"]))
    require(correlation_error <= 1e-12, f"E_corr identity residual is {correlation_error:.3e} Ha")
    require(abs(point["E_corr_mHa"] - 1000 * point["E_corr_Ha"]) <= 1e-10,
            "Correlation energy units disagree")
    require(point["E_FCI_Ha"] == point["E_FCI_frozen_core_Ha"], "FCI total fields disagree")
    require(abs(point["E_FCI_Ha"] - point["E_CAS_active_Ha"]
                - point["E_core_including_nuclear_Ha"]) <= 1e-9, "FCI energy decomposition differs")
    require(abs(point["rhf_diagnostics"]["energy_Eh"] - point["E_RHF_Ha"]) <= 1e-12,
            "Stored RHF diagnostics energy differs")
    electrons = sum(ATOMIC_NUMBERS[s] for s in symbols) - point["charge"]
    core = point["n_frozen_orbitals"]
    require(type(core) is int and core >= 0, "Invalid frozen occupied orbital count")
    require(electrons > 0 and electrons % 2 == 0, "System has an odd or invalid electron count")
    require([core, point["n_active_orbitals"], point["n_active_electrons"]]
            == settings["active_spaces"][name], "Point active space differs from dataset convention")
    require(point["n_frozen_electrons"] == 2 * core
            and electrons == 2 * core + point["n_active_electrons"],
            "Frozen-core and active electron counts disagree")
    expected_nao = core + point["n_active_orbitals"]
    with np.load(state_path, allow_pickle=False) as arrays:
        require(arrays["symbols"].tolist() == symbols
                and np.array_equal(arrays["cartesian_A"], coordinates),
                "NPZ geometry differs from the energy row")
        for field in ("E_RHF_Ha", "E_FCI_Ha", "E_corr_Ha"):
            require(float(arrays[field]) == point[field], f"NPZ {field} differs")
        require(float(arrays["core_energy_Ha"]) == point["E_core_including_nuclear_Ha"],
                "NPZ core energy differs")
        for field in ("mo_coeff", "rhf_density_ao", "overlap_ao", "fock_ao_Ha", "hcore_ao_Ha"):
            value = arrays[field]
            require(value.shape == (expected_nao, expected_nao) and np.isrealobj(value)
                    and np.isfinite(value).all(), f"Invalid stored {field} shape or values")
        occupation = arrays["mo_occ"]
        require(occupation.shape == (expected_nao,) and np.isin(occupation, (0., 2.)).all()
                and np.sum(occupation) == electrons and np.all(occupation[:core] == 2)
                and np.all(np.diff(occupation) <= 0), "Stored occupations are not closed-shell ordered RHF")
        spec = json.loads(str(arrays["molecule_pyscf_json"]))
        require(ast.literal_eval(spec["basis"]) == point["basis"]
                and spec["unit"] == "Angstrom" and spec["charge"] == point["charge"]
                and spec.get("spin", 0) == point["spin"] and spec["cart"] is False,
                "Stored PySCF molecule convention differs")
        require(not spec.get("_ecp") and not spec.get("_pseudo"), "ECP/pseudopotential is unsupported")
        atoms = ast.literal_eval(spec["atom"])
        require([atom[0] for atom in atoms] == symbols
                and np.array_equal([atom[1] for atom in atoms], coordinates),
                "Stored PySCF molecule input geometry differs")
        shell_signature = [(int(shell[0]), int(shell[1]), int(shell[2]), int(shell[3]))
                           for shell in spec["_bas"]]
    # Retain useful original metadata without copying existing checksum fields.
    raw = {field: point[field] for field in POINT_FIELDS}
    record = InputGeometry(name, identifier, index, geometry_path, state_path,
                           list(symbols), coordinates, point["charge"], point["spin"],
                           point["basis"], electrons, core, float(point["q_A"]),
                           float(point["bond_length_A"]), float(point["E_RHF_Ha"]),
                           float(point["E_FCI_Ha"]), float(point["E_corr_Ha"]), raw)
    return record, xyz_error, correlation_error, shell_signature


def discover_dataset(input_dir, molecules=MOLECULES):
    """Return sorted valid records and a JSON-ready read-only validation report.

    A failed point excludes its entire molecule. Other valid molecules remain
    available. A failure in shared dataset metadata excludes all requested
    molecules. The report records the failure instead of repairing input.
    """
    root = Path(input_dir).expanduser().resolve()
    molecules = list(molecules)
    report = {"input_dir": str(root), "schema": "initialdata-v1", "status": "PASS",
              "molecules": {}, "errors": [], "valid_geometry_count": 0,
              "failed_molecule_count": 0, "failed_geometry_count": 0,
              "expected_geometry_count": 30 * len(molecules),
              "energy_units": "Hartree", "geometry_units": "Angstrom",
              "geometry_coordinate_source": "point metadata cartesian_A, verified against XYZ and NPZ",
              "fci_convention": "frozen-core canonical RHF spatial orbitals",
              "source_snapshot": str(root / "source_initialdata.py")}
    records = []
    try:
        require(len(set(molecules)) == len(molecules), "Duplicate requested molecules")
        require(all(name in MOLECULES for name in molecules), "Unsupported requested molecule")
        document = _read_json(root / "initialdata.json")
        require(document["schema_version"] == "initialdata-v1", "Unsupported input schema")
        settings = document["settings"]
        require(settings["scan_points"] == 30, "Dataset metadata does not specify 30 geometries")
        require(document["units"]["length"] == "Angstrom"
                and document["units"]["energy"] == "Hartree (Ha = Eh)", "Input units differ or are missing")
        source_spaces = _source_active_spaces(root / "source_initialdata.py")
        for name in molecules:
            require(name in settings["molecules"], f"Requested molecule missing from settings: {name}")
            require(list(source_spaces[name]) == settings["active_spaces"][name],
                    f"Source snapshot and metadata frozen-core convention differ: {name}")
        manifest_rows = defaultdict(list)
        for result in document["results"]:
            manifest_rows[result["molecule"]].append(result)
        with (root / "scan.csv").open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        indexed_csv = defaultdict(list)
        for row in csv_rows:
            indexed_csv[(row["molecule"], int(row["point_index"]))].append(row)
    except (InputError, OSError, ValueError, KeyError, TypeError) as exc:
        report["status"] = "FAIL"
        report["errors"].append(f"Dataset: {exc}")
        report["failed_molecule_count"] = len(molecules)
        report["failed_geometry_count"] = 30 * len(molecules)
        for name in molecules:
            report["molecules"][name] = {"status": "FAIL", "errors": [str(exc)], "geometry_count": 0}
        return [], report

    for name in molecules:
        entry = {"status": "PASS", "errors": [], "geometry_count": 0,
                 "max_xyz_rounding_A": 0., "max_correlation_residual_Ha": 0.}
        report["molecules"][name] = entry
        collected = []
        try:
            require(len(manifest_rows[name]) == 1, "Missing or duplicate molecule in initialdata.json")
            result = manifest_rows[name][0]
            local = _read_json(root / name / "result.json")
            require(result["molecule"] == local["molecule"] == name
                    and result["status"] == local["status"] == "PASS", "Molecule result is not successful")
            require(result["summary"] == local["summary"], "Molecule scan summaries differ")
            points, local_points = result["scan"], local["scan"]
            entry["geometry_count"] = len(points)
            ids = [point["point_index"] for point in points]
            require(len(points) == 30 and sorted(ids) == list(range(1, 31)),
                    f"Expected exactly 30 unique geometry IDs 001..030; found {ids}")
            require(len(local_points) == 30 and sorted(p["point_index"] for p in local_points) == sorted(ids),
                    "result.json scan IDs do not match initialdata.json")
            local_by_id = {p["point_index"]: p for p in local_points}
            directories = sorted(path.name for path in (root / name / "scan").iterdir() if path.is_dir())
            require(directories == [f"{i:03d}" for i in range(1, 31)],
                    "Scan directory IDs/count differ from energy metadata")
            molecule_csv = [row for row in csv_rows if row["molecule"] == name]
            require(len(molecule_csv) == 30 and sorted(int(row["point_index"]) for row in molecule_csv) == sorted(ids),
                    "scan.csv has missing, duplicate, or extra geometry rows")
            shell_reference = None
            geometries = set()
            for point in sorted(points, key=lambda point: point["point_index"]):
                index = point["point_index"]
                try:
                    matches = indexed_csv[(name, index)]
                    require(len(matches) == 1, "Missing or duplicate CSV energy row")
                    record, xyz_error, correlation_error, shells = _point_record(
                        root, name, point, local_by_id[index], matches[0], settings)
                    if shell_reference is None:
                        shell_reference = (record.symbols, shells)
                    require(shell_reference == (record.symbols, shells),
                            "Atom ordering or basis-shell ordering changes within molecule")
                    identity = tuple(record.coordinates.ravel())
                    require(identity not in geometries, "Duplicate Cartesian geometry")
                    geometries.add(identity)
                    collected.append(record)
                    entry["max_xyz_rounding_A"] = max(entry["max_xyz_rounding_A"], xyz_error)
                    entry["max_correlation_residual_Ha"] = max(entry["max_correlation_residual_Ha"], correlation_error)
                except (InputError, OSError, ValueError, KeyError, TypeError) as exc:
                    entry["errors"].append(f"{name}/{index:03d}: {exc}")
            if entry["errors"]:
                raise InputError("One or more geometry/energy records failed validation")
            lengths = np.array([r.bond_length_A for r in collected])
            displacements = np.array([r.q_A for r in collected])
            summary = result["summary"]
            require(np.all(np.diff(lengths) > 0) and np.all(np.diff(displacements) > 0),
                    "Geometry IDs do not follow strictly increasing scan coordinates")
            require(np.allclose(lengths, np.linspace(summary["n1_min_A"], summary["n1_max_A"], 30),
                                rtol=0, atol=1e-14), "Scan lengths do not match documented 30-point grid")
            require(np.allclose(displacements, lengths - summary["r_e_A"], rtol=0, atol=1e-14),
                    "q_A is inconsistent with the documented equilibrium coordinate")
            entry.update(coordinate_definition=summary["coordinate"],
                         geometry_ids=[r.geometry_id for r in collected],
                         scan_range_A=[float(lengths[0]), float(lengths[-1])],
                         electron_count=collected[0].electron_count,
                         input_fci_frozen_core=collected[0].input_fci_frozen_core,
                         full_ao_count=sum((2 * s[1] + 1) * s[3] for s in shell_reference[1]),
                         basis=collected[0].basis, charge=collected[0].charge, spin=collected[0].spin)
            records.extend(collected)
        except (InputError, OSError, ValueError, KeyError, TypeError) as exc:
            entry["status"] = "FAIL"
            entry["errors"].append(f"{name}: {exc}")
            report["status"] = "FAIL"
            report["failed_molecule_count"] += 1
    report["valid_geometry_count"] = len(records)
    report["failed_geometry_count"] = 30 * report["failed_molecule_count"]
    return records, report
