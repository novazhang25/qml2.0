#!/usr/bin/env python3
"""Overlay 30 saved RHF points on plot 1, in the style of tests/plot.py.

Writes a 3x3 overview and individual local harmonic-range figures as PNG/SVG.
By default plotted energies are absolute; --relative uses mHa referenced to
the minimum of the 30 saved RHF scan points, shifting every overlay equally.
In absolute mode: RHF samples are E_RHF, the harmonic curve
is E_re + k*q^2/2, and the level segments are E_re + E_n0 and E_re + E_n1.
The saved equilibrium (r_e, E_re) joins the RHF curve as a normal RHF point.
No display reference is subtracted; the zero test in Part 3 is unchanged.
No electronic calculations are performed. Matplotlib is imported only on run.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# Allow direct execution as well as imports through the plots package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhf_bond_scan_30 import PART3_SCHEMA, POINT_COUNT, build_grid, equal_number, finite, require, sha256


PROJECT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#255D9B", "#C56517", "#268567", "#7952A1", "#56616E"
NAMES = {"BeH2": r"BeH$_2$", "H2O": r"H$_2$O", "NH3": r"NH$_3$",
         "N2": r"N$_2$", "H2S": r"H$_2$S", "H2O2": r"H$_2$O$_2$"}
COORDINATES = {
    "LiH": "Li-H distance", "BeH2": "Common Be-H length", "H2O": "Common O-H length",
    "NH3": "Common N-H length", "N2": "N-N distance", "CO": "C-O distance",
    "HF": "H-F distance", "H2S": "Common S-H length", "H2O2": "O-O separation",
}


def load_plot_data(scan_path, levels_path):
    """Reject partial, mismatched or invalid scans rather than hide missing points."""
    scan = json.loads(Path(scan_path).read_text())
    levels = json.loads(Path(levels_path).read_text())
    require(scan["schema_version"] == "rhf-current-range-30-v1" and scan["status"] == "PASS",
            "A completed PASS 30-point RHF scan is required.")
    require(scan["points_per_molecule"] == POINT_COUNT, "The scan must contain 30 points per molecule.")
    require(levels["schema_version"] == PART3_SCHEMA, "Expected zero-energy Part 3 levels.")
    require(scan["provenance"]["ranges_sha256"] == sha256(levels_path),
            "The level/range file differs from the one used for these RHF scans.")
    by_name = {}
    for record in levels["results"]:
        name = record["summary"]["molecule"]
        require(name not in by_name, f"Duplicate Part 3 molecule: {name}")
        by_name[name] = record
    results, names = [], set()
    for record in scan["results"]:
        summary = record["summary"]
        name = summary["molecule"]
        require(name in COORDINATES and name in by_name and name not in names, f"Invalid/duplicate molecule: {name}")
        names.add(name)
        current = by_name[name]["summary"]
        require(by_name[name]["input_validation"]["passed"] is True
                and current["validation_status"] == "PASS", f"{name}: invalid Part 3 inputs.")
        require(summary["status"] == "PASS" and summary["accepted_points"] == POINT_COUNT
                and summary["requested_points"] == POINT_COUNT and summary["failed_points"] == 0,
                f"{name}: not all 30 points passed.")
        for key in ("basis", "coordinate", "selected_n", "selected_range_min_A", "selected_range_max_A",
                    "r_e_A", "E_re_Ha", "k_Ha_per_A2", "hbar_omega_Ha", "E_n0_Ha", "E_n1_Ha", "E_total_n0_Ha", "E_total_n1_Ha",
                    "n0_pass", "n1_pass", "decision_criterion", "reference_shift_Ha",
                    "energy_reference_status", "energy_reference_warning",
                    "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A", "Delta_q_n1_A"):
            require(summary[key] == current[key], f"{name}: mismatched saved {key}.")
        require(type(current["selected_n"]) is int and current["selected_n"] in (0, 1), "Invalid selected level.")
        require(current["decision_criterion"] == "E_total_n < 0 Ha" and current["reference_shift_Ha"] == 0,
                f"{name}: incorrect zero-energy decision or shifted energy reference.")
        require(current["energy_reference_status"] == "UNVALIDATED_ZERO_REFERENCE"
                and bool(current["energy_reference_warning"]), f"{name}: missing zero-reference warning.")
        re = finite(current["E_re_Ha"], "E_re")
        quantum = finite(current["hbar_omega_Ha"], "hbar_omega")
        require(quantum > 0, "Harmonic energy quantum must be positive.")
        curvature = finite(current["k_Ha_per_A2"], "harmonic curvature")
        require(curvature > 0, "The harmonic curvature must be positive.")
        for n in (0, 1):
            excitation = finite(current[f"E_n{n}_Ha"], f"E_{n}")
            amplitude = finite(current[f"Delta_q_n{n}_A"], f"n={n} amplitude")
            require(excitation > 0 and amplitude > 0, "Invalid harmonic level or amplitude.")
            equal_number(excitation, (n + 0.5) * quantum, f"n={n} excitation")
            equal_number(0.5 * curvature * amplitude**2, excitation, f"n={n} harmonic units")
            equal_number(current[f"E_total_n{n}_Ha"], re + excitation, f"n={n} absolute level")
            require(current[f"n{n}_pass"] is (current[f"E_total_n{n}_Ha"] < 0),
                    f"{name}: incorrect n={n} zero-test result.")
        selected = 1 if current["n1_pass"] else 0 if current["n0_pass"] else None
        require(current["selected_n"] == selected, f"{name}: wrong selected n.")
        points = sorted(record["points"], key=lambda point: point["point_index"])
        require(len(points) == POINT_COUNT and [p["point_index"] for p in points] == list(range(1, POINT_COUNT + 1)),
                f"{name}: missing or duplicate grid points.")
        expected = build_grid(current["selected_range_min_A"], current["selected_range_max_A"])
        plotted = []
        for point, length in zip(points, expected):
            require(point["status"] == "PASS" and point["scf_converged"] is True
                    and point["internal_stable"] is True and point["geometry_validation"]["passed"] is True,
                    f"{name}: a plotted point did not pass validation.")
            require(point["molecule"] == name and point["basis"] == current["basis"], "Point identity mismatch.")
            equal_number(point["bond_length_A"], length, f"{name}: grid coordinate")
            equal_number(point["q_A"], length - current["r_e_A"], f"{name}: displacement")
            equal_number(point["E_re_Ha"], re, f"{name}: energy zero")
            energy = finite(point["E_RHF_Ha"], f"{name}: RHF energy")
            relative = energy - re  # Retained only as a checked CSV diagnostic, not the plotted energy.
            equal_number(point["delta_E_RHF_Ha"], relative, f"{name}: relative RHF energy")
            plotted.append({"molecule": name, "point_index": point["point_index"],
                            "bond_length_A": point["bond_length_A"], "q_A": point["q_A"],
                            "E_RHF_Ha": energy, "E_re_Ha": re, "E_RHF_minus_E_re_Ha": relative})
        results.append({"summary": current, "points": plotted})
    require(1 <= len(results) <= 9 and names == set(scan["requested_molecules"]), "Incomplete molecule set.")
    require(scan["requested_total_points"] == POINT_COUNT * len(results), "Incorrect total point count.")
    return results


def local_panel(ax, result, ticker, relative=False, compact_style=False):
    """The original plot-1 composition, overlaid with all 30 new RHF samples."""
    s, points = result["summary"], result["points"]
    equilibrium_reference = "fci_reference" in result
    compact_style = compact_style or equilibrium_reference
    reference_at_re = equilibrium_reference or result.get("relative_to_re", False)
    relative = relative or reference_at_re
    reference = (s["E_re_Ha"] if reference_at_re else
                 min(p["E_RHF_Ha"] for p in points) if relative else 0.0)
    def display(energy):
        return 1000.0 * (energy - reference) if relative else energy
    width = 1.28 * s["Delta_q_n1_A"]
    q = [-width + 2 * width * i / 500 for i in range(501)]
    model = [display(s["E_re_Ha"] + 0.5 * s["k_Ha_per_A2"] * displacement**2) for displacement in q]
    if reference_at_re:
        model = [1000.0 * 0.5 * s["k_Ha_per_A2"] * displacement**2 for displacement in q]
    ax.axvspan(s["selected_range_min_A"], s["selected_range_max_A"],
               color=GREEN, alpha=.12, label="Selected bond range")
    if not equilibrium_reference:
        ax.plot([s["r_e_A"] + displacement for displacement in q], model,
                color=BLUE, ls="--", lw=1.7, label="Local harmonic model")
    rhf_curve = sorted([(p["bond_length_A"], p["E_RHF_Ha"]) for p in points]
                       + [(s["r_e_A"], s["E_re_Ha"])])
    ax.plot([r for r, energy in rhf_curve], [display(energy) for r, energy in rhf_curve],
            "o-", color=ORANGE, lw=1.05, ms=3.5, markeredgecolor="white", markeredgewidth=.35,
            zorder=4, label="RHF calculated points")
    fci_values = []
    if equilibrium_reference:
        ref = s["E_re_Ha"]
        fci_curve = sorted([(p["bond_length_A"], 1000.0 * (p["E_FCI_frozen_core_Ha"] - ref))
                            for p in points] + [(s["r_e_A"],
                                1000.0 * (result["fci_reference"]["E_FCI_frozen_core_Ha"] - ref))])
        fci_values = [energy for r, energy in fci_curve]
        ax.plot([r for r, energy in fci_curve], fci_values, "s-", color="#B13E64",
                lw=1.05, ms=3.1, markeredgecolor="white", markeredgewidth=.35,
                zorder=4, label="CASCI calculated points")
    levels_to_show = () if equilibrium_reference else ((0, PURPLE), (1, GREEN))
    for n, color in levels_to_show:
        level = 1000.0 * s[f"E_n{n}_Ha"] if reference_at_re else display(s[f"E_total_n{n}_Ha"])
        ax.hlines(level, s[f"n{n}_min_A"], s[f"n{n}_max_A"], color=color, lw=1.8)
        ax.scatter([s[f"n{n}_min_A"], s[f"n{n}_max_A"]], [level, level], s=16, color=color, zorder=5)
        ax.annotate(f"n={n}", (s["r_e_A"], level), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=color, fontsize=9)
    ax.axvline(s["r_e_A"], color=GRAY, lw=.8, ls=":")
    energies = [display(s["E_re_Ha"]), *model, *(display(p["E_RHF_Ha"]) for p in points),
                display(s["E_total_n0_Ha"]), display(s["E_total_n1_Ha"])]
    if equilibrium_reference:
        energies = [display(s["E_re_Ha"]), *(display(p["E_RHF_Ha"]) for p in points),
                    1000*s["E_n0_Ha"], 1000*s["E_n1_Ha"]]
    energies.extend(fci_values)
    low, high = min(energies), max(energies)
    span = max(high - low, 1e-12)
    ax.set_xlim(s["r_e_A"] - width, s["r_e_A"] + width)
    ax.set_ylim(low - .055 * span, high + .30 * span)  # Include RHF compression-side excursions.
    if compact_style:
        ax.set_ylim(low - .055*span, high + (.44 if s["molecule"] == "LiH" else .16)*span)
        ax.text(.97, .96, NAMES.get(s["molecule"], s["molecule"]), transform=ax.transAxes,
                ha="right", va="top", fontweight="semibold", fontsize=12)
    else:
        ax.set_title(NAMES.get(s["molecule"], s["molecule"]), loc="left", fontweight="semibold")
    coordinate = COORDINATES[s["molecule"]]
    if compact_style:
        coordinate = coordinate.replace("Common ", "")
    ax.set_xlabel(coordinate + " (Å)")
    ax.set_ylabel("Relative energy (mHa)" if relative else r"$E$ (Hartree)")
    if reference_at_re:
        ax.set_ylabel(r"Energy relative to $E_{RHF}(R_e)$ (mHa)")
    if relative:
        ax.axhline(0, color=GRAY, lw=.8)
    ax.xaxis.set_major_locator(ticker(5))
    ax.yaxis.set_major_locator(ticker(5))
    if equilibrium_reference and s["molecule"] == "CO":
        # Use a uniform 50 mHa scale covering both curves and positive energies.
        ax.set_ylim(-160, 55)
        ax.set_yticks([-150, -100, -50, 0, 50])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if compact_style and not relative:
        ax.tick_params(axis="y", labelsize=8)
    if not compact_style:
        ax.text(.98, .97, f"[{s['selected_range_min_A']:.6f}, {s['selected_range_max_A']:.6f}] Å",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GREEN)
    ax.grid(axis="y", alpha=.85)
    ax.set_axisbelow(True)


def write_plots(results, output, relative=False, compact_style=False):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qml-rhf-overlay-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#CAD1D8",
        "axes.labelcolor": "#273444", "text.color": "#273444", "xtick.color": GRAY, "ytick.color": GRAY,
        "grid.color": "#E3E8EC", "grid.linewidth": .6, "legend.frameon": False,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    }
    compact = compact_style or "fci_reference" in results[0]
    if compact:
        order = ("LiH", "HF", "BeH2", "NH3", "H2O", "H2S", "N2", "CO", "H2O2")
        results = sorted(results, key=lambda result: order.index(result["summary"]["molecule"]))
    def inset_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        labels = [{"RHF calculated points": "RHF", "CASCI calculated points": "CASCI"}.get(label, label) for label in labels]
        ax.legend(handles, labels,
                  loc="upper left", fontsize=8, borderaxespad=.35, labelspacing=.2, handlelength=1.7)
    filenames = []
    def save(fig, name):
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=190)
            filenames.append(path.name)
    with plt.rc_context(style):
        rows, columns = math.ceil(len(results) / 3), 3
        fig, axes = plt.subplots(rows, columns, figsize=(12, 2.9 * rows) if compact else (14, 4 * rows), squeeze=False)
        try:
            if compact:
                absolute_compact = not relative and "fci_reference" not in results[0]
                fig.subplots_adjust(left=.084 if absolute_compact else .066, right=.992,
                                    bottom=.065, top=.988, hspace=.25,
                                    wspace=.25 if absolute_compact else .12)
                ylabel = ("Total energy (Ha)" if absolute_compact else
                          r"Energy relative to $E_{RHF}(R_e)$ (mHa)" if "fci_reference" in results[0] or results[0].get("relative_to_re")
                          else "Relative energy (mHa)")
                fig.supylabel(ylabel, x=.008, fontsize=11)
            else:
                fig.subplots_adjust(left=.085, right=.98, bottom=.085, top=.92, hspace=.48, wspace=.4)
            for ax, result in zip(axes.flat, results):
                local_panel(ax, result, MaxNLocator, relative=relative, compact_style=compact)
                if compact:
                    ax.set_ylabel("")
                    if result["summary"]["molecule"] == "LiH":
                        inset_legend(ax)
            for ax in list(axes.flat)[len(results):]:
                ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if not compact:
                fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.075, .98), ncol=3,
                           borderaxespad=0, fontsize=10, handlelength=2.7, columnspacing=2.5)
            if relative and not compact:
                fig.text(.5, .02, "RHF relative energies · Each scan referenced to its sampled minimum · STO-3G", ha="center")
            save(fig, "overview")
        finally:
            plt.close(fig)
        for result in results:
            fig, ax = plt.subplots(figsize=(6, 4.4) if compact else (7.4, 5.8))
            try:
                fig.subplots_adjust(left=.16, right=.98, bottom=.14, top=.98) if compact else fig.subplots_adjust(left=.16, right=.97, bottom=.17, top=.85)
                local_panel(ax, result, MaxNLocator, relative=relative, compact_style=compact)
                handles, labels = ax.get_legend_handles_labels()
                if compact:
                    if result["summary"]["molecule"] == "LiH":
                        inset_legend(ax)
                else:
                    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, .99), fontsize=9, ncol=2)
                save(fig, result["summary"]["molecule"])
            finally:
                plt.close(fig)
    return filenames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=PROJECT / "results/rhf_30_point_scans/rhf_scan_30.json")
    parser.add_argument("--levels", type=Path, default=PROJECT / "results/bond_length_part3/vibrational_levels.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact-style", action="store_true", help="Use compact ordered panels, inside labels, and a LiH legend")
    parser.add_argument("--relative", action="store_true", help="Reference all overlays to the 30-point RHF sampled minimum, in mHa")
    parser.add_argument("--relative-to-re", action="store_true", help="Reference energies to the saved optimized RHF equilibrium energy, in mHa")
    args = parser.parse_args(argv)
    if args.relative and args.relative_to_re:
        parser.error("Choose --relative or --relative-to-re, not both")
    try:
        args.output = args.output or PROJECT / ("results/plots/rhf_30_overlay_relative" if args.relative else "results/plots/rhf_30_overlay")
        before = (sha256(args.scan), sha256(args.levels))
        results = load_plot_data(args.scan, args.levels)
        references = []
        if args.relative_to_re:
            for result in results:
                result["relative_to_re"] = True
                for point in result["points"]:
                    point["E_RHF_relative_mHa"] = 1000.0 * (point["E_RHF_Ha"] - result["summary"]["E_re_Ha"])
        if args.relative:
            for result in results:
                points = result["points"]
                minimum = min(p["E_RHF_Ha"] for p in points)
                minima = [i for i, point in enumerate(points) if point["E_RHF_Ha"] == minimum]
                for point in points:
                    point["E_RHF_relative_mHa"] = 1000.0 * (point["E_RHF_Ha"] - minimum)
                require(min(p["E_RHF_relative_mHa"] for p in points) == 0.0, "Relative minimum must be zero")
                for index in minima:
                    point = points[index]
                    location = "left_boundary" if index == 0 else "right_boundary" if index == len(points)-1 else "interior"
                    references.append(dict(molecule=point["molecule"], method="RHF",
                        minimum_geometry_index=point["point_index"], minimum_coordinate_A=point["bond_length_A"],
                        minimum_absolute_energy_Ha=minimum, minimum_location=location,
                        equilibrium_minimum_established_within_scan="true" if all(0 < i < len(points)-1 for i in minima) else "false"))
        filenames = write_plots(results, args.output, relative=args.relative or args.relative_to_re, compact_style=args.compact_style)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(results[0]["points"][0]))
        writer.writeheader()
        writer.writerows(point for result in results for point in result["points"])
        (args.output / "plotted_points.csv").write_text(stream.getvalue(), encoding="utf-8")
        notes = ["# Plot 1 with 30 RHF points per molecule", "",
                 "The overview repeats the local harmonic panel in the 3x3 style of tests/plot.py. "
                 "Each molecule also has an individual local-panel PNG and SVG.", "",
                 "Orange markers show all 30 accepted RHF energies, including both bond-range endpoints. "
                 "The connecting lines guide the eye; no fitting or extra points are used. "
                 "The saved optimized equilibrium (r_e, E_re) is included in the orange RHF curve in coordinate order, "
                 "using the same circular marker and one shared legend entry, without a separate equilibrium text box. "
                 "The curve has 30 scan points plus this saved equilibrium; "
                 "no interpolation or new RHF calculation supplies the equilibrium. "
                 "Blue dashed curves are the unchanged harmonic approximation. Green shading is the saved selected range. "
                 "Purple/green horizontal segments are the existing n=0/n=1 total energies and turning points.", "",
                 "The y axis is absolute E in Hartree: the RHF curve uses E_RHF; the harmonic curve uses E_re + k*q^2/2; "
                 "the two levels use E_total_n0 = E_re + E_n0 and E_total_n1 = E_re + E_n1. "
                 "No equilibrium or sampled-minimum energy is subtracted. Axis tick offsets are disabled. "
                 "The even 30-point scan grid is unchanged; the added equilibrium curve point uses saved Part 1/2 data.", "",
                 "The x axis is the saved absolute bond coordinate in Angstrom; H2O2 uses O-O separation. "
                 "All RHF data are included in the y limits, including the compression side. "
                 "No SCF, optimization, Hessian or new energy calculation was run by this plotting command.", "",
                 "Part 3 selects the highest n in {0,1} satisfying E_total_n = E_re + (n+0.5)*hbar_omega < 0 Ha. "
                 "E_re remains the saved absolute RHF energy; "
                 "the physical validity of zero as an allowed-state reference is unvalidated.", "",
                 f"Scan input: {args.scan.resolve()}", f"Scan SHA256: {sha256(args.scan)}",
                 f"Part 3 input: {args.levels.resolve()}", f"Part 3 SHA256: {sha256(args.levels)}", "",
                 "plotted_points.csv contains the exact plotted coordinates and absolute E_RHF_Ha values. "
                 "It retains the 30 scan rows per molecule; the added equilibrium curve point comes from r_e_A and E_re_Ha "
                 "in the Part 3 input. E_RHF_minus_E_re_Ha is retained only as an auxiliary diagnostic; it is not plotted.", ""]
        if args.relative:
            notes = ["# Relative RHF 30-point overlay", "",
                     "All 30 saved RHF scan points are unchanged. The reference for each molecule is the minimum of these 30 absolute RHF energies.",
                     "All displayed energies use 1000 * (E_absolute - E_scan_min) in mHa, including the saved equilibrium point, harmonic model and both vibrational levels.",
                     "The saved equilibrium point is separate from the scan and may appear below zero. No new energies, geometry optimization or extrapolation are performed.",
                     "Interior minima denote sampled equilibrium-region minima, not newly optimized equilibrium geometries. Boundary minima do not establish an equilibrium minimum within the scan.",
                     "minimum_references.csv preserves the source point_index convention (1-based, 1–30). plotted_points.csv retains absolute energies and adds E_RHF_relative_mHa.",
                     "The selected ranges and Part 3 absolute-energy selection criterion are unchanged. Original absolute overlay files are preserved.",
                     f"Scan input: {args.scan.resolve()}", f"Scan SHA256: {before[0]}",
                     f"Part 3 input: {args.levels.resolve()}", f"Part 3 SHA256: {before[1]}", ""]
            for destination in (sys.stdout,):
                writer = csv.DictWriter(destination, fieldnames=list(references[0]))
                writer.writeheader()
                writer.writerows(references)
            with (args.output / "minimum_references.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(references[0]))
                writer.writeheader()
                writer.writerows(references)
            for row in references:
                if row["minimum_location"] != "interior":
                    warning = f"{row['molecule']}: minimum at scan boundary; equilibrium minimum not established within current range"
                    print(warning)
                    notes.append(warning)
            print("PASS: every RHF scan minimum is exactly 0 mHa.")
        if args.relative_to_re:
            notes = ["# RHF overlay relative to the optimized RHF equilibrium energy", "",
                     "All displayed energies use 1000 * (E - E_re) in mHa. The saved equilibrium point is exactly zero; the 30 scan energies are not independently zeroed.",
                     "The harmonic model is 1000*k*q^2/2. The n=0/n=1 levels are 1000*E_n0 and 1000*E_n1. All coordinates, absolute energies, harmonic parameters and selected ranges are unchanged.",
                     "No electronic-structure calculations were performed. plotted_points.csv retains all 270 absolute energies and their displayed relative energies.",
                     f"Scan input: {args.scan.resolve()}", f"Scan SHA256: {before[0]}",
                     f"Part 3 input: {args.levels.resolve()}", f"Part 3 SHA256: {before[1]}", ""]
        require(before == (sha256(args.scan), sha256(args.levels)), "Source files changed")
        if args.compact_style:
            notes += ["Compact layout: LiH/HF/BeH2, NH3/H2O/H2S, N2/CO/H2O2. Molecule labels are inside the top-right corners; the legend is inside LiH. Common is omitted from x-axis labels; bracketed range annotations are hidden. The harmonic model and n=0/n=1 levels are retained.", ""]
        notes += [f"- [{name}]({name})" for name in filenames]
        (args.output / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"Saved {len(results)} molecules with 30 RHF points each: {args.output.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Overlay plot failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
