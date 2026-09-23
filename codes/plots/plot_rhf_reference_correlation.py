#!/usr/bin/env python3
"""Plot saved absolute RHF/FCI totals and signed correlation energies.

Reads the completed audit only. No energy calculations or source-data changes.
Produces nine two-panel figures and a 3x3 overview as PNG/PDF. Requires Matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# Allow direct execution as well as imports through the plots package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path(__file__).resolve().parents[2]
MOLECULES = ("LiH", "BeH2", "H2O", "NH3", "N2", "CO", "HF", "H2S", "H2O2")
ENERGIES = ("E_RHF_Ha", "E_FCI_frozen_core_Ha", "E_corr_Ha", "E_corr_mHa")
LABELS = dict(zip(MOLECULES, (
    "Li-H distance", "Mean Be-H distance", "Mean O-H distance", "Mean N-H distance",
    "N-N distance", "C-O distance", "H-F distance", "Mean S-H distance", "O-O distance")))
TITLES = {"BeH2": r"BeH$_2$", "H2O": r"H$_2$O", "NH3": r"NH$_3$",
          "N2": r"N$_2$", "H2S": r"H$_2$S", "H2O2": r"H$_2$O$_2$"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(records, label):
    result = {}
    for row in records:
        index = int(row["geom_index"])
        require(str(index) == str(row["geom_index"]), f"{label}: invalid geometry index")
        key = (row["molecule"], index)
        require(key not in result, f"{label}: duplicate {key}")
        result[key] = row
    expected = {(name, index) for name in MOLECULES for index in range(30)}
    require(set(result) == expected,
            f"{label}: missing {sorted(expected-set(result))}; unexpected {sorted(set(result)-expected)}")
    return result


def load_data(directory):
    document = json.loads((directory / "provenance.json").read_text())
    require(document.get("schema_version") == "rhf-reference-current-range-v3"
            and document.get("grid_source") == "bond_length_part3_selected_range",
            "These are old-grid results. Run the corrected current-range correlation audit first.")
    require(document["status"] == "PASS" and document["all_270_successfully_validated"] is True,
            "A completed PASS audit with all 270 geometries is required")
    from current_range_fci import load_current_inputs
    inputs = document["inputs"]
    current, issues, _ = load_current_inputs(Path(inputs["ranges_json"]), Path(inputs["rhf_scan_json"]),
                                            Path(inputs["harmonic_json"]))
    require(not issues, f"Current range inputs are invalid: {issues}")
    current = keyed(current, "Current new-range geometries")
    with (directory / "per_geometry.csv").open(newline="") as handle:
        rows = keyed(list(csv.DictReader(handle)), "CSV")
    saved = keyed(document["results"], "Saved results")
    geometry = keyed(document["per_geometry_provenance"], "Geometry provenance")
    grouped = {}
    for name in MOLECULES:
        group = []
        for index in range(30):
            key = (name, index)
            require(geometry[key]["cartesian_A"] == current[key]["cartesian_A"]
                    and geometry[key]["symbols"] == current[key]["symbols"]
                    and geometry[key]["basis"] == current[key]["basis"]
                    and geometry[key]["bond_length_angstrom"] == current[key]["bond_length_angstrom"],
                    f"{key}: results no longer match the current Part 3 geometry")
            row = {field: float(rows[key][field]) for field in ENERGIES}
            require(all(math.isfinite(value) for value in row.values()), f"{key}: non-finite energy")
            require(all(row[field] == float(saved[key][field]) for field in ENERGIES),
                    f"{key}: CSV and saved audit energies disagree")
            require(row["E_RHF_Ha"] == current[key]["E_RHF_Ha"], f"{key}: saved RHF reference changed")
            require(row["E_corr_Ha"] == row["E_FCI_frozen_core_Ha"] - row["E_RHF_Ha"],
                    f"{key}: correlation formula mismatch")
            require(row["E_corr_mHa"] == 1000.0 * row["E_corr_Ha"], f"{key}: unit mismatch")
            r = float(geometry[key]["bond_length_angstrom"])
            require(math.isfinite(r) and r > 0, f"{key}: invalid bond length")
            require(float(rows[key]["bond_length_angstrom"]) == r,
                    f"{key}: CSV bond length differs from the saved current geometry")
            require(geometry[key]["basis"] == "sto-3g", f"{key}: unexpected basis")
            group.append(dict(**row, geom_index=index, r=r))
        require(all(a["r"] < b["r"] for a, b in zip(group, group[1:])),
                f"{name}: duplicate or non-increasing bond lengths")
        grouped[name] = group
    return grouped


def draw_panels(top, bottom, name, rows, compact=False):
    x = [row["r"] for row in rows]
    size = 2.3 if compact else 3.5
    top.plot(x, [row["E_FCI_frozen_core_Ha"] for row in rows],
             color="#215EA8", marker="o", markersize=size, linewidth=1.6, label="Frozen-core FCI")
    top.plot(x, [row["E_RHF_Ha"] for row in rows],
             color="#CF6B22", marker="s", markersize=size, linewidth=1.6, linestyle="--", label="RHF")
    bottom.plot(x, [row["E_corr_mHa"] for row in rows],
                color="#207C63", marker="o", markersize=size, linewidth=1.6,
                label=r"$E_{corr}=E_{FCI}-E_{RHF}$")
    bottom.axhspan(-1.6, 1.6, color="#DFB84C", alpha=.22, label="Within ±1.6 mHa")
    bottom.axhline(-1.6, color="#947015", linewidth=.8, linestyle=":")
    bottom.axhline(1.6, color="#947015", linewidth=.8, linestyle=":")
    bottom.axhline(0, color="#777777", linewidth=.7)
    top.set_title(f"{TITLES.get(name, name)}  ·  30 geometries", loc="left", fontsize=11 if compact else 14)
    top.set_ylabel("Total energy (Ha)")
    bottom.set_ylabel(r"$E_{corr}$ (mHa)")
    bottom.set_xlabel(LABELS[name] + " (Å)")
    top.tick_params(labelbottom=False)
    for axis in (top, bottom):
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18, linewidth=.6)
        axis.margins(x=.025, y=.15)
        axis.set_axisbelow(True)
    if not compact:
        top.legend(loc="best", fontsize=9)
        bottom.legend(loc="best", fontsize=9)


def save_figure(figure, staging, stem):
    for extension in ("png", "pdf"):
        figure.savefig(staging / f"{stem}.{extension}", dpi=250, bbox_inches="tight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "results/rhf_reference_correlation")
    parser.add_argument("--output-dir", type=Path, help="Default: input-dir/plots")
    parser.add_argument("--overwrite", action="store_true", help="Replace this script's PNG/PDF figures only")
    args = parser.parse_args(argv)
    try:
        source = args.input_dir.resolve()
        destination = (args.output_dir or source / "plots").resolve()
        names = [f"{stem}.{ext}" for stem in (*MOLECULES, "overview") for ext in ("png", "pdf")]
        require(not destination.exists() or destination.is_dir(), "Output path is not a directory")
        for name in names:
            target = destination / name
            require(not target.is_symlink() and (not target.exists() or target.is_file()),
                    f"Output is not a regular file: {target}")
            require(args.overwrite or not target.exists(), f"Figure exists: {target}; use --overwrite")
        groups = load_data(source)
        # Use temporary Matplotlib configuration and staged exports; do not alter audit outputs.
        with tempfile.TemporaryDirectory(prefix="rhf-energy-plots-") as runtime:
            os.environ["MPLCONFIGDIR"] = runtime
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                                 "pdf.fonttype": 42, "axes.titleweight": "semibold"})
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".staging-", dir=destination) as temporary:
                staging = Path(temporary)
                for name in MOLECULES:
                    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 7),
                                                       layout="constrained", height_ratios=(1.25, 1))
                    draw_panels(top, bottom, name, groups[name])
                    save_figure(fig, staging, name)
                    plt.close(fig)
                overview = plt.figure(figsize=(18, 17), layout="constrained")
                grid = overview.add_gridspec(3, 3)
                for position, name in enumerate(MOLECULES):
                    subgrid = grid[position // 3, position % 3].subgridspec(2, 1, height_ratios=(1.25, 1))
                    top = overview.add_subplot(subgrid[0])
                    bottom = overview.add_subplot(subgrid[1], sharex=top)
                    draw_panels(top, bottom, name, groups[name], compact=True)
                save_figure(overview, staging, "overview")
                plt.close(overview)
                for name in names:
                    require((staging / name).stat().st_size > 0, f"Empty figure: {name}")
                for name in names:
                    (staging / name).replace(destination / name)
        print(f"Created 9 molecule figures and a 3x3 overview (PNG/PDF): {destination}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Energy plotting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
