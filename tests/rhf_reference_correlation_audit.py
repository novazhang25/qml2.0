#!/usr/bin/env python3
"""Audit correlation energies on the CURRENT Part 3 bond-length ranges.

Use saved absolute RHF energies and exact Cartesian nuclei from qml2.0. No old
grid, external cutoff or Protocol-7 manifest is read. Missing frozen-core FCI
requires --compute-missing-fci; rebuilding its unavailable RHF orbitals also
requires --rebuild-rhf-orbitals. Nothing executes on import.
Exit codes: 0 complete; 2 incomplete/failed audit with reports; 1 setup failure.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from current_range_fci import (ACTIVE_SPACES, MOLECULES, FCI_SCHEMA,
                               load_current_inputs, compute_missing_fci, validate_fci_record)

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "rhf-reference-current-range-v3"
THRESHOLD_HA = 0.0016
ENERGY_FIELDS = (
    "molecule", "geom_index", "bond_length_angstrom", "E_RHF_Ha", "E_FCI_frozen_core_Ha",
    "E_corr_Ha", "E_corr_mHa", "E_corr_over_1p6_mHa", "RHF_within_1p6_mHa_of_FCI")
SUMMARY_FIELDS = (
    "molecule", "N_geometry", "min_E_corr_Ha", "min_E_corr_mHa", "geom_index_at_min",
    "max_E_corr_Ha", "max_E_corr_mHa", "geom_index_at_max", "span_Ha", "span_mHa",
    "span_over_1p6_mHa", "span_within_1p6_mHa", "N_within_1p6_mHa",
    "N_outside_1p6_mHa", "all_30_within_1p6_mHa")
OUTPUTS = ("per_geometry.csv", "per_molecule_summary.csv", "provenance.json", "REPORT.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    value = float(value)
    require(math.isfinite(value), f"Non-finite {name}")
    return value


def read_cache(path):
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    require(document.get("schema_version") == FCI_SCHEMA, f"Not a current-range FCI cache: {path}")
    records = {}
    for row in document["records"]:
        key = (row["molecule"], row["geom_index"])
        require(key[0] in MOLECULES and type(key[1]) is int and 0 <= key[1] < 30,
                f"Unexpected cached geometry: {key}")
        require(key not in records, f"Duplicate cached geometry: {key}")
        records[key] = row
    return records


def write_cache(path, records):
    """Persist each successful new calculation so a later run can reuse it."""
    require(not path.is_symlink(), f"Cache cannot be a symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=FCI_SCHEMA, records=[records[key] for key in sorted(records)])
    with tempfile.TemporaryDirectory(prefix=".fci-cache-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        staged.replace(path)


def summarize(rows):
    summaries = []
    for name in MOLECULES:
        group = [r for r in rows if r["molecule"] == name]
        within = sum(r["RHF_within_1p6_mHa_of_FCI"] for r in group)
        summary = dict.fromkeys(SUMMARY_FIELDS)
        summary.update(molecule=name, N_geometry=len(group), N_within_1p6_mHa=within,
                       N_outside_1p6_mHa=len(group)-within, all_30_within_1p6_mHa=len(group) == within == 30)
        if group:
            low = min(group, key=lambda r: (r["E_corr_Ha"], r["geom_index"]))
            high = min(group, key=lambda r: (-r["E_corr_Ha"], r["geom_index"]))
            span = high["E_corr_Ha"]-low["E_corr_Ha"]
            summary.update(min_E_corr_Ha=low["E_corr_Ha"], min_E_corr_mHa=low["E_corr_mHa"],
                           geom_index_at_min=low["geom_index"], max_E_corr_Ha=high["E_corr_Ha"],
                           max_E_corr_mHa=high["E_corr_mHa"], geom_index_at_max=high["geom_index"],
                           span_Ha=span, span_mHa=1000*span, span_over_1p6_mHa=span/THRESHOLD_HA,
                           span_within_1p6_mHa=span <= THRESHOLD_HA)
        summaries.append(summary)
    return summaries


def run_audit(args):
    points, input_issues, ranges = load_current_inputs(args.ranges_json, args.rhf_scan_json, args.harmonic_json)
    issues = [dict(code="INPUT_VALIDATION", message=message) for message in input_issues]
    records = read_cache(args.fci_cache)
    expected = {(p["molecule"], p["geom_index"]): p for p in points}
    # Preflight every existing entry; never match a different geometry by index alone.
    for key, record in records.items():
        if key in expected:
            validate_fci_record(record, expected[key])
    rows, provenance = [], []
    computed = reused = 0
    for point in points:
        name, index = point["molecule"], point["geom_index"]
        key = (name, index)
        try:
            was_reused = key in records
            if not was_reused:
                require(args.compute_missing_fci, "Missing FCI for this new-range nuclear geometry. "
                        "Use --compute-missing-fci --rebuild-rhf-orbitals; old-grid substitution is forbidden.")
                require(not input_issues, "Fix input validation issues before computing missing FCI")
                print(f"{name} geometry {index:02d}/29: missing FCI at "
                      f"{point['bond_length_angstrom']:.12g} Angstrom", flush=True)
                record = compute_missing_fci(point, threads=args.threads,
                                             allow_rhf_rebuild=args.rebuild_rhf_orbitals)
                validate_fci_record(record, point)
                records[key] = record
                write_cache(args.fci_cache, records)
                computed += 1
            record = records[key]
            validate_fci_record(record, point)
            reused += int(was_reused)
            rhf = finite(point["E_RHF_Ha"], "saved RHF energy")
            fci = finite(record["E_FCI_frozen_core_Ha"], "FCI total")
            corr = finite(fci-rhf, "correlation energy")
            rows.append(dict(molecule=name, geom_index=index, bond_length_angstrom=point["bond_length_angstrom"],
                             E_RHF_Ha=rhf, E_FCI_frozen_core_Ha=fci, E_corr_Ha=corr, E_corr_mHa=1000*corr,
                             E_corr_over_1p6_mHa=corr/THRESHOLD_HA,
                             RHF_within_1p6_mHa_of_FCI=-THRESHOLD_HA <= corr <= THRESHOLD_HA))
            provenance.append(dict(
                **point, frozen_orbitals=ACTIVE_SPACES[name][0], frozen_electrons=2*ACTIVE_SPACES[name][0],
                energy_sources={"E_RHF": dict(file=point["rhf_source_file"], key=point["rhf_source_key"]),
                                "E_FCI": dict(file=str(args.fci_cache),
                                              key=f"records[molecule={name},geom_index={index}].E_FCI_frozen_core_Ha")},
                fci_record=record, fci_reused_this_run=was_reused,
                total_energy_confirmation="CASCI_TOTAL_WITH_CORE_AND_NUCLEAR"))
            if corr > args.variational_tolerance_ha:
                issues.append(dict(code="FCI_ABOVE_RHF", molecule=name, geom_index=index,
                                   message=f"E_FCI - E_RHF = {corr:.17g} Ha exceeds tolerance"))
        except (ValueError, KeyError, TypeError, ArithmeticError, RuntimeError) as exc:
            issues.append(dict(code="GEOMETRY_NOT_EVALUATED", molecule=name, geom_index=index, message=str(exc)))
            print(f"{name} geometry {index:02d}: {exc}", file=sys.stderr, flush=True)
    population = {}
    for name in MOLECULES:
        accepted = {r["geom_index"] for r in rows if r["molecule"] == name}
        population[name] = dict(evaluated_count=len(accepted), unevaluated_indices=sorted(set(range(30))-accepted))
    successful = len(rows) == 270 and not issues
    return dict(
        schema_version=SCHEMA, grid_source="bond_length_part3_selected_range",
        status="PASS" if successful else "FAIL", created_utc=datetime.now(timezone.utc).isoformat(),
        expected_count=270, evaluated_count=len(rows), all_270_evaluated=len(rows) == 270,
        all_270_successfully_validated=successful, population=population, issues=issues,
        inputs=dict(ranges_json=str(args.ranges_json), rhf_scan_json=str(args.rhf_scan_json),
                    harmonic_json=str(args.harmonic_json), fci_cache=str(args.fci_cache)),
        selected_ranges=ranges, results=rows, summary=summarize(rows), per_geometry_provenance=provenance,
        fci_computed_this_run=computed, fci_reused_this_run=reused,
        rhf_orbitals_rebuilt_this_run=computed, saved_RHF_energy_replaced=False,
        variational_tolerance_Ha=args.variational_tolerance_ha,
        variational_violation_count=sum(i["code"] == "FCI_ABOVE_RHF" for i in issues),
        chemical_accuracy_threshold_mHa=1.6, chemical_accuracy_threshold_Ha=THRESHOLD_HA,
        protocol7_selection_applied=False, external_cutoff_read=False, energy_shift_Ha=0.0)


def display(value):
    if value is None:
        return "—"
    return f"{value:.12g}" if isinstance(value, float) else str(value).replace("|", "\\|").replace("\n", " ")


def table(rows, columns):
    return "\n".join([
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(display(row[key]) for key, _ in columns) + " |" for row in rows)])


def report_text(doc):
    sources = doc["inputs"]
    lines = ["# RHF-reference correlation: current Part 3 ranges", "",
             f"Status: **{doc['status']}**; evaluated **{doc['evaluated_count']}/270**; "
             f"all 270 successfully validated: **{doc['all_270_successfully_validated']}**.", "",
             "The sole geometry population is the new Part 3 selected range: 30 points per molecule. "
             "No external cutoff or Protocol-7 population is read. Source RHF point indices 1–30 map "
             "to geom_index 0–29. E_corr = E_FCI_frozen_core_total - saved_E_RHF_total. All totals "
             "are absolute. Min is most negative, max least negative; span = max - min. "
             "No absolute-value transformation is applied. Ties use the smallest geometry index.", "",
             table(doc["summary"], (("molecule", "Molecule"), ("N_geometry", "N"),
                   ("min_E_corr_mHa", "min E_corr (mHa)"), ("geom_index_at_min", "geom"),
                   ("max_E_corr_mHa", "max E_corr (mHa)"), ("geom_index_at_max", "geom"),
                   ("span_mHa", "span (mHa)"))), "",
             "CSV min/max/span also appear in Hartree; machine-readable values retain full binary64 "
             "round-trip precision. Incomplete populations are explicit below.", "",
             "## Actual bond-length ranges", "",
             "| Molecule | Selected n | Lower (Angstrom) | Upper (Angstrom) | Frozen orbitals | Frozen electrons |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in MOLECULES:
        r, core = doc["selected_ranges"].get(name, {}), ACTIVE_SPACES[name][0]
        lines.append(f"| {name} | {display(r.get('selected_n'))} | {display(r.get('selected_range_min_A'))} | "
                     f"{display(r.get('selected_range_max_A'))} | {core} | {2*core} |")
    lines += ["", "## Sources, geometry and total-energy convention", "",
              f"- Range: `{sources['ranges_json']}` → `results[].summary.selected_range_min_A/selected_range_max_A`.",
              f"- RHF: `{sources['rhf_scan_json']}` → `results[].points[].E_RHF_Ha`; "
              "nuclei: `results[].symbols` and `results[].points[].cartesian_A`; same-record basis.",
              f"- Harmonic geometry: `{sources['harmonic_json']}`; used to validate the new path.",
              f"- FCI: `{sources['fci_cache']}` → `records[].E_FCI_frozen_core_Ha`. "
              "Reuse requires matching molecule/index, basis, exact nuclear coordinates, reference and core convention.",
              "- Per-geometry source fields, nuclei and FCI components are recorded in `provenance.json`.", "",
              "Saved RHF energies remain unchanged. Their orbitals were not serialized. Only explicit "
              "missing-FCI/orbital-rebuild options reconstruct them; the strict local RHF solver must "
              "reproduce the saved energy within 1e-9 Ha. The rebuilt value never replaces E_RHF. "
              "Canonicalization identifies the lowest occupied RHF core orbitals before CASCI.", "",
              f"Frozen-core counts are recorded in `{PROJECT / 'codes/current_range_fci.py'}` → `ACTIVE_SPACES`. "
              "They retain the previously inspected paper convention in "
              "`encoding_qml/canonical_pipeline/canonical_pipeline/molecules.py` → "
              "`MOLECULES[molecule].expected_core_count`; this analysis does not read that project's grid "
              "or cutoff. Basis STO-3G, charge 0, spin 0. "
              "ncas = n_AO - ncore; nelecas = N_electrons - 2*ncore; all non-core orbitals are active.", "",
              "PySCF CASCI kernel()[0] is the molecular total. The cache also records mc.e_cas and "
              "mc.get_h1eff()[1], verifying total = active + core. The core scalar includes nuclear repulsion. "
              "Neither core nor nuclear energy is added twice. UHF and shifted PES energies are not used.", "",
              f"FCI totals computed this run: {doc['fci_computed_this_run']}; reused: {doc['fci_reused_this_run']}. "
              f"Successful RHF orbital reconstructions for missing FCI: {doc['rhf_orbitals_rebuilt_this_run']}.", "",
              "## Comparison with 1.6 mHa", "",
              "Within threshold means -0.0016 <= E_corr_Ha <= 0.0016. E_corr_over_1p6_mHa stays signed. "
              "span_over_1p6_mHa = span_Ha/0.0016; span_within_1p6_mHa tests span <= 0.0016 Ha. "
              "Equality passes without extra tolerance. Span measures the largest RHF-versus-FCI "
              "discrepancy in an energy difference between sampled geometries; a small span does not "
              "imply small absolute RHF errors. The reference is frozen-core STO-3G, not experiment "
              "or exact all-electron energy. Missing points are excluded from counts and extrema.", "",
              table(doc["summary"], (("molecule", "Molecule"), ("N_geometry", "N"),
                    ("N_within_1p6_mHa", "Within 1.6 mHa"), ("N_outside_1p6_mHa", "Outside 1.6 mHa"),
                    ("span_over_1p6_mHa", "span / 1.6 mHa"), ("span_within_1p6_mHa", "span <= 1.6 mHa"))), "",
              "## Validation and missing geometries", "",
              f"FCI <= RHF tolerance: {doc['variational_tolerance_Ha']:.17g} Ha; "
              f"violations: {doc['variational_violation_count']}. Chemical accuracy does not alter this test.", ""]
    for name, status in doc["population"].items():
        lines.append(f"- {name}: {status['evaluated_count']}/30; unevaluated indices {status['unevaluated_indices']}.")
    lines += [""]
    if doc["issues"]:
        lines.extend(f"- {i['code']} {i.get('molecule', '')} {i.get('geom_index', '')}: {display(i['message'])}"
                     for i in doc["issues"])
    else:
        lines.append("No missing, duplicate, geometry-consistency or variational violations detected.")
    return "\n".join(lines)+"\n"


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: repr(row[key]) if isinstance(row[key], float) else row[key] for key in fields} for row in rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges-json", type=Path, default=PROJECT / "results/bond_length_part3/vibrational_levels.json")
    parser.add_argument("--rhf-scan-json", type=Path, default=PROJECT / "results/rhf_30_point_scans/rhf_scan_30.json")
    parser.add_argument("--harmonic-json", type=Path, default=PROJECT / "json/new_rhf_harmonic_bond_ranges.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/rhf_reference_correlation")
    parser.add_argument("--fci-cache", type=Path, help="Default: output-dir/fci_current_range_cache.json")
    parser.add_argument("--compute-missing-fci", action="store_true", help="Calculate missing matching FCI totals only")
    parser.add_argument("--rebuild-rhf-orbitals", action="store_true",
                        help="Allow missing-FCI orbital reconstruction while keeping saved RHF reference energies")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Replace four audit outputs; preserve the FCI cache")
    parser.add_argument("--variational-tolerance-ha", type=float, default=1e-8)
    args = parser.parse_args(argv)
    try:
        require(args.threads > 0, "Threads must be positive")
        require(finite(args.variational_tolerance_ha, "tolerance") >= 0, "Tolerance must be nonnegative")
        require(not args.rebuild_rhf_orbitals or args.compute_missing_fci,
                "--rebuild-rhf-orbitals requires --compute-missing-fci")
        for key in ("ranges_json", "rhf_scan_json", "harmonic_json", "output_dir"):
            setattr(args, key, getattr(args, key).resolve())
        args.fci_cache = (args.fci_cache or args.output_dir / "fci_current_range_cache.json").resolve()
        output = args.output_dir
        require(not output.exists() or args.overwrite, f"Output directory exists: {output}; use --overwrite")
        require(not output.exists() or output.is_dir(), "Output path must be a directory")
        for name in OUTPUTS:
            target = output / name
            require(not target.is_symlink() and (not target.exists() or target.is_file()), f"Invalid output target: {target}")
        protected = {args.ranges_json, args.rhf_scan_json, args.harmonic_json}
        targets = {output / name for name in OUTPUTS}
        require(args.fci_cache not in protected | targets, "FCI cache would overwrite an input or report")
        require(not protected.intersection(targets), "Outputs overlap inputs")
        document = run_audit(args)
        output.mkdir(parents=True, exist_ok=args.overwrite or args.compute_missing_fci)
        with tempfile.TemporaryDirectory(prefix=".audit-", dir=output) as temporary:
            staging = Path(temporary)
            write_csv(staging / "per_geometry.csv", document["results"], ENERGY_FIELDS)
            write_csv(staging / "per_molecule_summary.csv", document["summary"], SUMMARY_FIELDS)
            (staging / "provenance.json").write_text(json.dumps(document, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            (staging / "REPORT.md").write_text(report_text(document), encoding="utf-8")
            for name in OUTPUTS:
                (staging / name).replace(output / name)
        print(f"{document['status']}: {document['evaluated_count']}/270 new-range geometries. Outputs: {output}")
        return 0 if document["status"] == "PASS" else 2
    except (OSError, ValueError, KeyError, TypeError, ImportError, ArithmeticError) as exc:
        print(f"Current-range correlation audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
