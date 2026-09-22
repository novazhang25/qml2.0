#!/usr/bin/env python3
"""Complete RHF bond-length pipeline: equilibrium -> k -> n=0/n=1 selection.

Run this file from any working directory. It reuses the validated numerical
engines in new_rhf_harmonic_bond_ranges.py and rhf_bound_level_selection.py.
It does not import historical scan/geometry scripts.

Default: reuse matching, validated current results and compute missing stages.
--reoptimize requests a new equilibrium optimization and invalidates downstream
stages. --recompute recomputes k and dissociation but retains valid equilibria.
All reported energies are Hartree, lengths Angstrom, k Eh/Bohr^2 or Eh/A^2.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyscf
from pyscf import lib
from scipy.optimize import minimize

import new_rhf_harmonic_bond_ranges as harmonic
import rhf_bound_level_selection as selection


PROJECT = Path(__file__).resolve().parent.parent
MOLECULES = harmonic.MOLECULES
SUMMARY_FIELDS = [
    "molecule", "basis", "coordinate", "r_e_A", "E_eq_Eh", "max_gradient_Eh_per_Bohr",
    "k_q_Eh_per_Bohr2", "k_q_Eh_per_A2", "mu_eff_amu", "harmonic_quantum_Eh",
    "Delta_q_n0_A", "Delta_q_n1_A", "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A",
    "E_diss_Eh", "D_e_Eh", "half_hbar_omega_Eh", "three_halves_hbar_omega_Eh",
    "E0_rel_Eh", "E1_rel_Eh", "n0_bound", "n1_bound", "selected_n",
    "selected_range_min_A", "selected_range_max_A", "Hessian_FD_relative_difference",
    "equilibrium_status", "harmonic_status", "selection_status", "asymptote_status",
    "validation_status", "message",
]
ENERGY_KEYS = {"harmonic_frequency_cm1": "harmonic_quantum_Eh",
               "dominant_mode_frequency_cm1": "dominant_mode_quantum_Eh",
               "frequency_cm1": "mode_quantum_Eh"}


def constants():
    return {"Bohr_A": harmonic.BOHR_A, "Hartree_J": harmonic.EH_J,
            "hbar_J_s": harmonic.HBAR, "c_m_per_s": harmonic.C_MS,
            "amu_kg": harmonic.AMU_KG}


def source_hashes():
    return {"pipeline": harmonic.sha256(__file__),
            "harmonic": harmonic.sha256(harmonic.__file__),
            "selection": harmonic.sha256(selection.__file__)}


def energy_view(value, inverse=False):
    """Store mode energy quanta in Hartree, reconstruct native engine inputs only in memory."""
    factor = selection.conversion_from_constants(constants())
    mapping = {v: k for k, v in ENERGY_KEYS.items()} if inverse else ENERGY_KEYS
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in mapping:
                result[mapping[key]] = None if item is None else item / factor if inverse else item * factor
            else:
                result[key] = energy_view(item, inverse)
        return result
    if isinstance(value, list):
        return [energy_view(item, inverse) for item in value]
    return value


def read_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def read_metadata(path):
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if row["molecule"] in result:
            raise ValueError(f"Duplicate equilibrium metadata for {row['molecule']}")
        result[row["molecule"]] = row
    return result


def starting_geometries(name):
    """Loose optimization seeds, never equilibrium values or fixed scan parameters.

    All Cartesian coordinates are free during optimization. Peroxide uses three
    torsional starts; its final scan retains only the OPTIMIZED internal values.
    """
    if name in harmonic.DIATOMICS:
        symbols = {"LiH": ["Li", "H"], "N2": ["N", "N"], "CO": ["C", "O"], "HF": ["H", "F"]}[name]
        length = {"LiH": 1.8, "N2": 1.2, "CO": 1.25, "HF": 1.05}[name]
        return [(symbols, np.array([[0., 0., 0.], [0., 0., length]]))]
    if name == "BeH2":
        return [(["Be", "H", "H"], np.array([[0., 0., 0.], [0., 0., -1.4], [0., 0., 1.4]]))]
    if name in ("H2O", "H2S"):
        x, z = (.8, .65) if name == "H2O" else (1., .9)
        return [(["O" if name == "H2O" else "S", "H", "H"], np.array([[0., 0., 0.], [x, 0., z], [-x, 0., z]]))]
    if name == "NH3":
        coords = [[0., 0., 0.]] + [[math.cos(phi), math.sin(phi), .4] for phi in (0., 2*math.pi/3, 4*math.pi/3)]
        return [(["N", "H", "H", "H"], np.array(coords))]
    if name == "H2O2":
        return [(["O", "O", "H", "H"], np.array([[0., 0., -.75], [0., 0., .75],
                  [1., 0., -1.], [math.cos(phi), math.sin(phi), 1.]]))
                for phi in np.deg2rad([60., 120., 160.])]
    raise ValueError(f"Unsupported molecule: {name}")


def optimize_equilibrium(name, basis, geometry_dir, gtol=1e-6, maxiter=200, starts=None):
    """Stage 1: unconstrained Cartesian BFGS using tight RHF energies and gradients."""
    candidates, errors = [], []
    for index, (symbols, initial) in enumerate(starts or starting_geometries(name), 1):
        history = []
        def objective(flat):
            coords_A = flat.reshape(-1, 3) * harmonic.BOHR_A
            mf = harmonic.run_rhf(symbols, coords_A, basis)
            gradient = mf.nuc_grad_method().kernel()
            if not np.isfinite(gradient).all():
                raise ValueError("Nonfinite RHF nuclear gradient during optimization.")
            history.append({"E_RHF_Eh": float(mf.e_tot),
                            "max_gradient_Eh_per_Bohr": float(np.max(np.abs(gradient)))})
            return float(mf.e_tot), gradient.ravel()
        try:
            initial = np.array(initial, float)
            initial -= initial.mean(axis=0)
            optimized = minimize(objective, (initial / harmonic.BOHR_A).ravel(), method="BFGS", jac=True,
                                 options={"gtol": gtol, "maxiter": maxiter})
            coords_A = optimized.x.reshape(-1, 3) * harmonic.BOHR_A
            # Fresh final evaluation verifies the geometry being written, rather than a cached iteration.
            energy, gradient = objective(optimized.x)
            max_gradient = float(np.max(np.abs(gradient)))
            passed = bool(optimized.success and max_gradient <= gtol)
            candidates.append({"start": index, "symbols": symbols, "cartesian_A": coords_A,
                               "E_eq_Eh": energy, "max_gradient_Eh_per_Bohr": max_gradient,
                               "optimizer_success": bool(optimized.success), "iterations": int(optimized.nit),
                               "evaluations": history, "converged": passed, "message": str(optimized.message)})
        except Exception as exc:
            errors.append(f"Start {index}: {type(exc).__name__}: {exc}")
    converged = [c for c in candidates if c["converged"]]
    if not converged:
        return {"status": "FAIL_RHF_EQUILIBRIUM", "candidates": candidates, "errors": errors,
                "message": "No starting geometry reached a converged RHF equilibrium."}
    best = min(converged, key=lambda c: c["E_eq_Eh"])
    geometry_dir.mkdir(parents=True, exist_ok=True)
    xyz = geometry_dir / f"{name}_converged.xyz"
    lines = [str(len(best["symbols"])), f"{name} RHF/{basis} converged E={best['E_eq_Eh']:.15f} Eh"]
    lines += [f"{symbol} {r[0]:.14f} {r[1]:.14f} {r[2]:.14f}" for symbol, r in zip(best["symbols"], best["cartesian_A"])]
    xyz.write_text("\n".join(lines) + "\n")
    row = {"molecule": name, "method": "RHF", "basis": basis, "status": "converged",
           "optimizer_success": "True", "energy_hartree": best["E_eq_Eh"], "xyz_file": str(xyz.resolve()),
           "max_gradient_hartree_per_bohr": best["max_gradient_Eh_per_Bohr"], "iterations": best["iterations"]}
    metadata = geometry_dir / f"{name}_equilibrium.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return {"status": "PASS", "row": row, "metadata_path": str(metadata), "source": "computed",
            "symbols": best["symbols"], "cartesian_A": best["cartesian_A"], "E_eq_Eh": best["E_eq_Eh"],
            "max_gradient_Eh_per_Bohr": best["max_gradient_Eh_per_Bohr"],
            "selected_start": best["start"], "candidates": candidates, "errors": errors}


def reuse_equilibrium(name, basis, row, metadata, gtol):
    symbols, coords, stored_basis, provenance = harmonic.load_equilibrium(name, row, metadata)
    if stored_basis.lower() != basis.lower():
        raise harmonic.InputProblem("Equilibrium basis differs from the requested basis.")
    energy = float(row.get("energy_hartree", "nan"))
    gradient = float(row.get("max_gradient_hartree_per_bohr", "nan"))
    if not np.isfinite([energy, gradient]).all() or gradient < 0 or gradient > gtol:
        raise harmonic.InputProblem("Saved equilibrium lacks a sufficiently small verified nuclear gradient.")
    # Resolve relocated XYZ paths now so every later stage consumes this exact geometry.
    resolved_row = {**row, "xyz_file": provenance["xyz_file"]}
    return {"status": "PASS", "source": "reused", "row": resolved_row, "metadata_path": str(metadata),
            "symbols": symbols, "cartesian_A": coords, "E_eq_Eh": energy,
            "max_gradient_Eh_per_Bohr": gradient, "input_provenance": provenance}


def same_geometry(symbols, coordinates, other_symbols, other_coordinates):
    left, right = np.asarray(coordinates), np.asarray(other_coordinates)
    return (list(symbols) == list(other_symbols) and left.shape == right.shape
            and np.isfinite(left).all() and np.isfinite(right).all()
            and bool(np.allclose(left, right, atol=1e-11, rtol=0)))


def harmonic_cache_matches(record, equilibrium):
    """A changed geometry, basis or equilibrium energy invalidates the Hessian cache."""
    try:
        summary = record["summary"]
        if summary["validation_status"] != "PASS" or record.get("failures"):
            return False
        if summary["RHF_basis"].lower() != equilibrium["row"]["basis"].lower():
            return False
        if not same_geometry(equilibrium["symbols"], equilibrium["cartesian_A"],
                             record["symbols"], record["equilibrium_cartesian_A"]):
            return False
        if abs(record["stationarity"]["E_RHF_Eh"] - equilibrium["E_eq_Eh"]) > 1e-9:
            return False
        selection.prepare_path(record)  # Geometry/tangent checks only; no SCF or Hessian.
        return (summary["k_q_Eh_per_Bohr2"] > 0 and summary["mu_eff_amu"] > 0
                and record["finite_differences"]["convergence_pass"])
    except (KeyError, ValueError, TypeError):
        return False


def selection_cache_matches(record, harmonic_record, settings):
    """Recheck the actual sampled plateau and its relation to the current harmonic path."""
    try:
        old, now = record["harmonic_input"], harmonic_record
        hs, ss = now["summary"], record["summary"]
        if ss["validation_status"] not in ("PASS", "NEAR_THRESHOLD", "NO_BOUND_LEVEL_IN_HARMONIC_TEST"):
            return False
        if ss["asymptote_status"] != "PLATEAU_CONFIRMED" or record.get("failure_reasons"):
            return False
        if ss["basis"].lower() != hs["RHF_basis"].lower() or ss["coordinate"] != hs["coordinate_definition"]:
            return False
        if not same_geometry(now["symbols"], now["equilibrium_cartesian_A"], old["symbols"], old["equilibrium_cartesian_A"]):
            return False
        quantum = hs["harmonic_frequency_cm1"] * selection.conversion_from_constants(constants())
        if not math.isclose(ss["harmonic_quantum_Eh"], quantum, rel_tol=1e-12, abs_tol=1e-15):
            return False
        if any(abs(old["summary"][key] - hs[key]) > 1e-12 for key in ("s_eq_A", "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A")):
            return False
        if abs(ss["E_eq_Eh"] - now["stationarity"]["E_RHF_Eh"]) > 1e-9:
            return False
        points = record["scan"]
        if not points or max(p["q_A"] for p in points) > settings["max_q_A"]:
            return False
        if not selection.assess_plateau(points, settings["plateau_tolerance_Eh"])["passed"]:
            return False
        if any(not p["internal_stable"] or p["orbital_gradient_norm_Eh"] > selection.SCF_GRADIENT_LIMIT
               or not p["geometry_validation"]["passed"] for p in points[-4:]):
            return False
        checks = record["asymptote_independent_checks"]
        if len(checks) < 2 or any(not c["accepted"] or abs(c["energy_difference_from_continuation_Eh"]) >= settings["plateau_tolerance_Eh"] for c in checks):
            return False
        return (abs(ss["E_diss_Eh"] - points[-1]["energy_Eh"]) < 1e-12
                and abs(ss["D_e_Eh"] - (ss["E_diss_Eh"] - ss["E_eq_Eh"])) < 1e-12
                and ss["D_e_Eh"] > 0)
    except (KeyError, ValueError, TypeError):
        return False


def cached_records(output_dir, harmonic_file, selection_file):
    """Only current validated output files and this pipeline's own saved output are considered."""
    caches = {"equilibrium": {}, "harmonic": {}, "selection": {}, "notes": []}
    hashes = source_hashes()
    for kind, path in (("pipeline", output_dir / "bond_length.json"),
                       ("harmonic", harmonic_file), ("selection", selection_file)):
        try:
            document = read_json(path)
            if document is None:
                continue
            if kind == "pipeline":
                if document["provenance"]["source_hashes"] != hashes or document["provenance"]["constants"] != constants():
                    caches["notes"].append(f"Ignored changed pipeline cache: {path}")
                    continue
                for result in document["results"]:
                    name = result["summary"]["molecule"]
                    if result.get("equilibrium", {}).get("status") == "PASS":
                        caches["equilibrium"][name] = result["equilibrium"]
                    if "harmonic" in result:
                        caches["harmonic"][name] = energy_view(result["harmonic"], inverse=True)
                    if "selection" in result:
                        caches["selection"][name] = result["selection"]
            else:
                provenance = document["runtime"] if kind == "harmonic" else document["provenance"]
                if provenance["script_sha256"] != hashes[kind]:
                    caches["notes"].append(f"Ignored {kind} cache with changed implementation: {path}")
                    continue
                if any(provenance["constants"][key] != value for key, value in constants().items()):
                    caches["notes"].append(f"Ignored {kind} cache with different physical constants: {path}")
                    continue
                for result in document["results"]:
                    caches[kind].setdefault(result["summary"]["molecule"], result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            caches["notes"].append(f"Could not reuse {kind} cache {path}: {exc}")
    return caches


def joined_summary(name, basis, result):
    summary = {"molecule": name, "basis": basis, "selected_n": None,
               "selected_range_min_A": None, "selected_range_max_A": None,
               "equilibrium_status": "NOT_RUN", "harmonic_status": "NOT_RUN", "selection_status": "NOT_RUN"}
    equilibrium, h, bound = result.get("equilibrium"), result.get("harmonic"), result.get("selection")
    if equilibrium:
        summary.update(equilibrium_status=equilibrium["status"], E_eq_Eh=equilibrium.get("E_eq_Eh"),
                       max_gradient_Eh_per_Bohr=equilibrium.get("max_gradient_Eh_per_Bohr"))
    if h:
        hs = h["summary"]
        for key in ("k_q_Eh_per_Bohr2", "k_q_Eh_per_A2", "mu_eff_amu", "Delta_q_n0_A", "Delta_q_n1_A",
                    "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A", "Hessian_FD_relative_difference"):
            summary[key] = hs.get(key)
        summary.update(harmonic_status=hs["validation_status"], coordinate=hs.get("coordinate_definition"), r_e_A=hs.get("s_eq_A"))
        if "harmonic_frequency_cm1" in hs:
            summary["harmonic_quantum_Eh"] = hs["harmonic_frequency_cm1"] * selection.conversion_from_constants(constants())
    if bound:
        for key in ("E_diss_Eh", "D_e_Eh", "harmonic_quantum_Eh", "half_hbar_omega_Eh", "three_halves_hbar_omega_Eh",
                    "E0_rel_Eh", "E1_rel_Eh", "n0_bound", "n1_bound", "selected_n", "selected_range_min_A", "selected_range_max_A", "asymptote_status"):
            summary[key] = bound["summary"].get(key)
        summary["selection_status"] = bound["summary"]["validation_status"]
    summary["validation_status"] = result.get("status", "INCOMPLETE")
    summary["message"] = "; ".join(result.get("messages", []))
    return summary


def process_molecule(name, args, rows, caches):
    result = {"sources": {}, "messages": [], "status": "INCOMPLETE"}
    start = time.monotonic()
    settings = {"max_q_A": args.max_q, "plateau_tolerance_Eh": args.plateau_tol,
                "threshold_warning_Eh": args.threshold_warning_eh, "molecule": name}
    try:
        equilibrium = None
        if not args.reoptimize:
            own = caches["equilibrium"].get(name)
            options = ([(own["row"], Path(own["metadata_path"]))] if own else []) + [(rows.get(name), args.metadata)]
            for row, metadata in options:
                try:
                    equilibrium = reuse_equilibrium(name, args.basis, row, metadata, args.gtol)
                    break
                except (harmonic.InputProblem, OSError, ValueError, TypeError):
                    continue
        if equilibrium is None:
            if args.reuse_only:
                result["status"] = "NEEDS_INPUT"
                result["messages"].append("No matching validated RHF equilibrium is available in reuse-only mode.")
                return result
            print(f"{name}: 1/3 optimizing RHF equilibrium ({args.basis})...", flush=True)
            equilibrium = optimize_equilibrium(name, args.basis, args.output_dir / "equilibria", args.gtol, args.maxiter)
        result["equilibrium"] = equilibrium
        result["sources"]["equilibrium"] = equilibrium.get("source", "failed")
        if equilibrium["status"] != "PASS":
            result["status"] = equilibrium["status"]
            result["messages"].append(equilibrium["message"])
            return result

        allow_harmonic_cache = equilibrium["source"] == "reused" and not args.recompute and not args.reoptimize
        h = caches["harmonic"].get(name) if allow_harmonic_cache else None
        if h is not None and harmonic_cache_matches(h, equilibrium):
            h = copy.deepcopy(h)
            result["sources"]["harmonic"] = "reused"
        elif args.reuse_only:
            result["status"] = "NEEDS_INPUT"
            result["messages"].append("No matching validated harmonic curvature is available in reuse-only mode.")
            return result
        else:
            print(f"{name}: 2/3 calculating the collective curvature k...", flush=True)
            h = harmonic.calculate(name, equilibrium["row"], Path(equilibrium["metadata_path"]))
            result["sources"]["harmonic"] = "computed"
        result["harmonic"] = h
        if h["summary"]["validation_status"] != "PASS":
            result["status"] = "FAIL_HARMONIC"
            result["messages"].extend(h.get("failures") or [h["summary"].get("message", "Harmonic validation failed.")])
            return result

        allow_selection_cache = result["sources"]["harmonic"] == "reused" and not args.recompute
        bound = caches["selection"].get(name) if allow_selection_cache else None
        if bound is not None and selection_cache_matches(bound, h, settings):
            bound = copy.deepcopy(bound)
            # Reapply the exact decision using the requested warning window, retaining the saved range floats.
            bound["summary"].update(selection.select_level(bound["summary"]["D_e_Eh"], bound["summary"]["harmonic_quantum_Eh"],
                                                         h["summary"], args.threshold_warning_eh))
            result["sources"]["selection"] = "reused"
        elif args.reuse_only:
            result["status"] = "NEEDS_INPUT"
            result["messages"].append("No matching, independently confirmed RHF plateau is available in reuse-only mode.")
            return result
        else:
            print(f"{name}: 3/3 scanning the same RHF path and selecting n=0/1...", flush=True)
            progress_time = [time.monotonic()]
            def progress(partial):
                if time.monotonic() - progress_time[0] >= 20:
                    point = partial["scan"][-1]
                    print(f"  {name}: q={point['q_A']:.6g} A; accepted={point['accepted']}", flush=True)
                    progress_time[0] = time.monotonic()
            bound = selection.scan_one(h, settings, selection.conversion_from_constants(constants()), progress)
            result["sources"]["selection"] = "computed"
        result["selection"] = bound
        result["status"] = bound["summary"]["validation_status"]
        result["messages"].extend(bound.get("failure_reasons", []))
    except Exception as exc:
        result["status"] = "ERROR"
        result["messages"].append(f"{type(exc).__name__}: {exc}")
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["summary"] = joined_summary(name, args.basis, result)
    return result


def report_text(document):
    rows = document["results"]
    lines = ["# Complete RHF bond-length pipeline", "",
             "This pipeline connects (1) RHF equilibrium geometry, (2) the collective stretching force constant, "
             "and (3) the n=0/n=1 bound-level test on the same constrained RHF dissociation path.", "",
             "All energies are Hartree (Eh). Lengths are Angstrom (A). A force constant has dimensions of energy/length^2; "
             "its reported units are Eh/Bohr^2 and Eh/A^2. Atomic masses are PySCF isotope-average masses in amu.", "",
             "## Summary", ""]
    keys = ("molecule", "r_e_A", "k_q_Eh_per_Bohr2", "mu_eff_amu", "harmonic_quantum_Eh", "D_e_Eh",
            "E0_rel_Eh", "E1_rel_Eh", "selected_n", "selected_range_min_A", "selected_range_max_A", "validation_status")
    lines += harmonic.table(list(keys), [[r["summary"].get(k) for k in keys] for r in rows])
    lines += ["", "## 1. RHF equilibrium r_e", "",
              "A converged equilibrium is reused only if its molecule, requested RHF basis and nuclear-gradient tolerance match. "
              "Otherwise analytic all-electron RHF gradients drive an unconstrained Cartesian BFGS optimization. "
              "The built-in geometries are only optimization seeds. Peroxide uses three torsional starts and the lowest-energy "
              "converged candidate is retained. --reoptimize explicitly requests a new optimization.", "",
              "No RHF equilibrium length, angle or torsion is hard-coded into the stretching path. "
              "The resulting Cartesian geometry is the sole geometric input for stages 2 and 3.", "",
              "## 2. Collective force constant k", "",
              "At q=0, t=dR/dq. The path increases each participating bond by q, with optimized angular geometry fixed. "
              "For H2O2 only the O-O separation increases; the optimized OH groups and dihedral remain fixed. "
              "Mass-weighted Eckart/Kabsch alignment removes overall translation and rotation.", "",
              "    k_q = t^T H_RHF t", "    mu_eff = sum_A m_A |t_A|^2", "    omega = sqrt(k_SI / mu_kg)",
              "    l_q = sqrt(hbar / (mu_kg * omega))", "    Delta_q(n) = sqrt(2*n+1) * l_q", "",
              "The tangent is not Euclidean-normalized. The RHF nuclear Hessian is analytic, with shape and convention "
              "checks and independent energy finite differences. Stationarity, fixed internal coordinates, positive curvature, "
              "masses and normal modes must pass the existing harmonic validation. A failed stage cannot yield a selected range.", "",
              "## 3. n=0 or n=1", "",
              "    D_e^(RHF-path) = E_diss - E_eq", "    E0_rel = -D_e + 0.5 * hbar * omega",
              "    E1_rel = -D_e + 1.5 * hbar * omega", "",
              "If E1_rel<0, choose n=1. Otherwise, if E0_rel<0, choose n=0. Otherwise choose NONE. "
              "Use the corresponding already validated interval r_e +/- Delta_q(n). Epsilon_n above the minimum is positive; "
              "the negative sign of E_n_rel relative to dissociation is the bound-level test.", "",
              f"Near-threshold warning window: {document['settings']['threshold_warning_Eh']:.12g} Eh. "
              f"Plateau tolerance: {document['settings']['plateau_tolerance_Eh']:.6g} Eh. The plateau requires four "
              "consecutive accepted points with geometrically increasing separation, including an outward confirmation, "
              "and independent RHF initial guesses that reproduce the asymptotic energy. A single last point is never sufficient. "
              "No stable, unambiguous plateau means no selected n or range.", "",
              selection.GENERAL_RHF_WARNING, "",
              "## Reuse and output files", "",
              "The program checks code hashes, constants, basis, geometry, equilibrium energy, harmonic ranges and actual plateau "
              "evidence before reusing a stage. --recompute invalidates harmonic and dissociation results. --reuse-only "
              "forbids numerical calculations and reports NEEDS_INPUT when a stage is unavailable. "
              "Optimization, harmonic analysis and scans are never launched merely by importing the module.", "",
              "bond_length.csv contains the joined result. bond_length.json preserves all three stages, Hessians, path checks "
              "and complete scans. scans/<molecule>.csv provides each energy scan. New equilibrium XYZ and metadata files "
              "are stored in equilibria/. Partial results are saved after each molecule.", ""]
    for r in rows:
        s = r["summary"]
        lines += [f"## {s['molecule']}", "", f"Status: {s['validation_status']}. Basis: {s['basis']}.", "",
                  f"Stage sources: {r['sources']}.", ""]
        if s.get("message"):
            lines += [s["message"], ""]
        if "selection" in r:
            lines += [r["selection"]["summary"].get("selected_reason", ""), ""]
            lines += [f"- {w}" for w in r["selection"].get("warnings", [])] + [""]
    lines += ["## Provenance", "", "```json", json.dumps(document["provenance"], indent=2), "```", ""]
    return "\n".join(lines)


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_outputs(document, output_dir):
    """Export joined results and full evidence without changing the previous workflow artifacts."""
    payload = copy.deepcopy(harmonic.jsonable(document))
    for result in payload["results"]:
        if "harmonic" in result:
            result["harmonic"] = energy_view(result["harmonic"])
    atomic_text(output_dir / "bond_length.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(r["summary"] for r in payload["results"])
    atomic_text(output_dir / "bond_length.csv", stream.getvalue())
    atomic_text(output_dir / "BOND_LENGTH_REPORT.md", report_text(document))
    scan_fields = ("q_A", "s_A", "energy_Eh", "scf_converged", "orbital_gradient_norm_Eh", "internal_stable", "accepted", "failure")
    for result in payload["results"]:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=scan_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.get("selection", {}).get("scan", []))
        atomic_text(output_dir / "scans" / f"{result['summary']['molecule']}.csv", stream.getvalue())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--basis", default="sto-3g", help="RHF basis used consistently by all three stages.")
    parser.add_argument("--molecules", nargs="+", choices=MOLECULES, default=list(MOLECULES))
    parser.add_argument("--metadata", type=Path, default=PROJECT / "results/rhf_geometries/rhf_equilibrium_summary.csv")
    parser.add_argument("--harmonic-json", type=Path, default=PROJECT / "json/new_rhf_harmonic_bond_ranges.json")
    parser.add_argument("--selection-json", type=Path, default=PROJECT / "json/rhf_bound_level_selection.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/bond_length")
    parser.add_argument("--reoptimize", action="store_true", help="Explicitly optimize new equilibria and recompute both subsequent stages.")
    parser.add_argument("--recompute", action="store_true", help="Recompute k and dissociation, retaining valid RHF equilibrium inputs.")
    parser.add_argument("--reuse-only", action="store_true", help="Use verified saved results only; never run SCF, optimization or Hessians.")
    parser.add_argument("--gtol", type=float, default=1e-6, help="Maximum nuclear gradient in Eh/Bohr for equilibrium optimization/reuse.")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--max-q", type=float, default=1e7, help="Maximum numerical dissociation displacement in Angstrom.")
    parser.add_argument("--plateau-tol", type=float, default=1e-6, help="RHF asymptote energy tolerance in Hartree.")
    parser.add_argument("--threshold-warning-eh", type=float, default=selection.DEFAULT_THRESHOLD_WARNING_EH)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.reuse_only and (args.recompute or args.reoptimize):
        parser.error("--reuse-only cannot be combined with --recompute or --reoptimize")
    if any(not math.isfinite(x) or x <= 0 for x in (args.gtol, args.max_q, args.plateau_tol, args.threshold_warning_eh)) or min(args.maxiter, args.threads) < 1:
        parser.error("Tolerances, limits and thread count must be positive and finite")
    if args.threshold_warning_eh < args.plateau_tol:
        parser.error("--threshold-warning-eh must cover --plateau-tol")
    if args.gtol > harmonic.TOL["gradient_max_Eh_per_Bohr"]:
        parser.error("--gtol cannot exceed the harmonic stationarity tolerance")
    for key in ("metadata", "harmonic_json", "selection_json", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.molecules = list(dict.fromkeys(args.molecules))
    return args


def main(argv=None):
    args = parse_args(argv)
    rows = read_metadata(args.metadata)
    caches = cached_records(args.output_dir, args.harmonic_json, args.selection_json)
    if not args.reuse_only:
        lib.num_threads(args.threads)
    document = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "energy_unit": "Hartree", "length_unit": "Angstrom", "results": [],
                "settings": {"basis": args.basis, "equilibrium_gtol_Eh_per_Bohr": args.gtol,
                             "max_q_A": args.max_q, "plateau_tolerance_Eh": args.plateau_tol,
                             "threshold_warning_Eh": args.threshold_warning_eh,
                             "reoptimize": args.reoptimize, "recompute": args.recompute, "reuse_only": args.reuse_only},
                "provenance": {"source_hashes": source_hashes(), "constants": constants(),
                               "python_version": platform.python_version(), "pyscf_version": pyscf.__version__,
                               "metadata": str(args.metadata), "harmonic_input": str(args.harmonic_json),
                               "selection_input": str(args.selection_json), "cache_notes": caches["notes"]}}
    for name in args.molecules:
        print(f"{name}: RHF equilibrium -> collective k -> n=0/1 selection", flush=True)
        result = process_molecule(name, args, rows, caches)
        document["results"].append(result)
        summary = result["summary"]
        print(f"{name}: {summary['validation_status']}; sources={result['sources']}; selected_n={summary['selected_n']}", flush=True)
        if summary["message"]:
            print(summary["message"], flush=True)
        write_outputs(document, args.output_dir)
    print(f"Summary CSV: {args.output_dir / 'bond_length.csv'}", flush=True)
    return 0 if all(r["status"] == "PASS" for r in document["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
