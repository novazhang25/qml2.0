#!/usr/bin/env python3
"""Plot saved initialdata results without running any molecular calculations.

The energy panel uses the saved RHF equilibrium energy as its zero by default.
Uses the compact layout from plot_rhf_fci_equilibrium_overlay.py: one energy
panel per molecule, shaded n=1 range, RHF and frozen-core CASCI curves.
Correlation energies remain in the saved data and exported table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SUMMARY_NUMBERS = (
    "r_e_A", "E_RHF_re_Ha", "k_Ha_per_Bohr2", "k_Ha_per_A2", "mu_eff_amu",
    "omega_rad_per_s", "hbar_omega_Ha", "E_n1_Ha", "E_total_n1_Ha",
    "n1_min_A", "n1_max_A",
)
COLORS = {"rhf": "#C56517", "fci": "#B13E64", "range": "#268567", "gray": "#56616E"}
PANEL_ORDER = ("LiH", "HF", "BeH2", "NH3", "H2O", "H2S", "N2", "CO", "H2O2")
NAMES = {"BeH2": r"BeH$_2$", "H2O": r"H$_2$O", "NH3": r"NH$_3$",
         "N2": r"N$_2$", "H2S": r"H$_2$S", "H2O2": r"H$_2$O$_2$"}
COORDINATES = {
    "LiH": "Li-H distance", "HF": "H-F distance", "BeH2": "Be-H length",
    "NH3": "N-H length", "H2O": "O-H length", "H2S": "S-H length",
    "N2": "N-N distance", "CO": "C-O distance", "H2O2": "O-O separation",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label}: a finite number is required.")
    return float(value)


def same(actual, expected, label):
    require(math.isclose(number(actual, label), expected, rel_tol=1e-9, abs_tol=1e-10),
            f"{label}: saved value does not match the other saved data.")


def validate_document(document):
    """Validate complete saved results, including all points and energy differences."""
    require(isinstance(document, dict), "The input must contain a JSON object.")
    require(document.get("schema_version") == "initialdata-v1", "Expected initialdata-v1 results.")
    require(document.get("status") == "PASS", "Only complete PASS initialdata results can be plotted.")
    settings = document.get("settings")
    require(isinstance(settings, dict), "Missing calculation settings.")
    require(settings.get("fci_convention") == "frozen-core",
            "Expected an explicit frozen-core FCI convention.")
    count = settings.get("scan_points")
    require(type(count) is int and count >= 2, "settings.scan_points must be an integer of at least two.")
    basis = settings.get("basis")
    require(isinstance(basis, str) and bool(basis), "Missing calculation basis.")
    results = document.get("results")
    require(isinstance(results, list) and bool(results), "No saved molecule results.")
    names = set()
    for result in results:
        require(isinstance(result, dict), "Each molecule result must be a JSON object.")
        name = result.get("molecule")
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name),
                "Invalid molecule name.")
        require(name not in names, f"Duplicate molecule: {name}.")
        names.add(name)
        require(result.get("status") == "PASS", f"{name}: incomplete molecule result.")
        summary = result.get("summary")
        require(isinstance(summary, dict), f"{name}: missing summary.")
        require(summary.get("molecule") == name and summary.get("basis") == basis,
                f"{name}: summary identity or basis does not match the settings.")
        for key in SUMMARY_NUMBERS:
            number(summary.get(key), f"{name}.{key}")
        for key in ("r_e_A", "k_Ha_per_Bohr2", "k_Ha_per_A2", "mu_eff_amu",
                    "omega_rad_per_s", "hbar_omega_Ha", "E_n1_Ha"):
            require(summary[key] > 0, f"{name}.{key}: must be positive.")
        lower, upper, equilibrium = summary["n1_min_A"], summary["n1_max_A"], summary["r_e_A"]
        require(0 < lower < equilibrium < upper, f"{name}: invalid n=1 bond range.")
        same(lower + upper, 2 * equilibrium, f"{name}: symmetric n=1 range")
        same(summary["E_n1_Ha"], 1.5 * summary["hbar_omega_Ha"], f"{name}: n=1 level")
        same(summary["E_total_n1_Ha"], summary["E_RHF_re_Ha"] + summary["E_n1_Ha"],
             f"{name}: total n=1 energy")
        same(0.5 * summary["k_Ha_per_A2"] * (upper - equilibrium) ** 2,
             summary["E_n1_Ha"], f"{name}: n=1 turning point")
        energies = result.get("equilibrium_energies")
        require(isinstance(energies, dict), f"{name}: missing equilibrium energies.")
        for key in ("E_RHF_Ha", "E_FCI_Ha", "E_corr_Ha"):
            number(energies.get(key), f"{name}: equilibrium {key}")
        same(energies["E_RHF_Ha"], summary["E_RHF_re_Ha"], f"{name}: RHF equilibrium energy")
        same(energies["E_corr_Ha"], energies["E_FCI_Ha"] - energies["E_RHF_Ha"],
             f"{name}: equilibrium correlation energy")
        points = result.get("scan")
        require(isinstance(points, list) and len(points) == count,
                f"{name}: expected all {count} scan points.")
        require(all(isinstance(point, dict) and type(point.get("point_index")) is int for point in points),
                f"{name}: invalid scan point indices.")
        points = sorted(points, key=lambda point: point["point_index"])
        require([point["point_index"] for point in points] == list(range(1, count + 1)),
                f"{name}: missing or duplicate scan indices.")
        for index, point in enumerate(points):
            label = f"{name}: point {index + 1}"
            require(point.get("status") == "PASS", f"{label}: incomplete scan point.")
            for key in ("bond_length_A", "q_A", "E_RHF_Ha", "E_FCI_Ha", "E_corr_Ha", "E_corr_mHa"):
                number(point.get(key), f"{label} {key}")
            expected = lower + (upper - lower) * index / (count - 1)
            same(point["bond_length_A"], expected, f"{label}: bond grid/endpoints")
            same(point["q_A"], point["bond_length_A"] - equilibrium, f"{label}: displacement")
            same(point["E_corr_Ha"], point["E_FCI_Ha"] - point["E_RHF_Ha"],
                 f"{label}: correlation energy")
            same(point["E_corr_mHa"], 1000 * point["E_corr_Ha"], f"{label}: correlation units")
            coordinates = point.get("cartesian_A")
            require(isinstance(coordinates, list) and len(coordinates) >= 2,
                    f"{label}: missing Cartesian geometry.")
            for atom in coordinates:
                require(isinstance(atom, list) and len(atom) == 3, f"{label}: invalid Cartesian geometry.")
                for coordinate in atom:
                    number(coordinate, f"{label}: Cartesian coordinate")
    for requested in (document.get("requested_molecules"), settings.get("molecules")):
        if requested is not None:
            require(isinstance(requested, list) and all(isinstance(name, str) for name in requested)
                    and len(requested) == len(names) and set(requested) == names,
                    "The saved molecule set does not match the requested molecules.")
    return document


def load_plot_data(path):
    """Read and validate a standalone initialdata JSON file."""
    return validate_document(json.loads(Path(path).read_text(encoding="utf-8")))


def plot_panel(ax, result, *, absolute=False):
    """Match the reference overlay, retaining every scan and equilibrium point."""
    from matplotlib.ticker import MaxNLocator

    summary = result["summary"]
    name, re = result["molecule"], summary["r_e_A"]
    reference = 0.0 if absolute else summary["E_RHF_re_Ha"]
    factor = 1.0 if absolute else 1000.0
    ax.axvspan(summary["n1_min_A"], summary["n1_max_A"], color=COLORS["range"],
               alpha=.12, label="Selected bond range")
    displayed = []
    for field, color, label, marker, size in (
        ("E_RHF_Ha", COLORS["rhf"], "RHF", "o", 3.5),
        ("E_FCI_Ha", COLORS["fci"], "CASCI", "s", 3.1),
    ):
        curve = sorted([(point["bond_length_A"], point[field]) for point in result["scan"]]
                       + [(re, result["equilibrium_energies"][field])])
        values = [factor * (energy - reference) for distance, energy in curve]
        displayed.extend(values)
        ax.plot([distance for distance, energy in curve], values, marker + "-", color=color,
                lw=1.05, ms=size, markeredgecolor="white", markeredgewidth=.35,
                zorder=4, label=label)
    # Preserve the reference plot's vertical headroom, without drawing level lines.
    displayed.extend(factor * (summary["E_RHF_re_Ha"] + multiplier * summary["hbar_omega_Ha"] - reference)
                     for multiplier in (0.5, 1.5))
    low, high = min(displayed), max(displayed)
    span = max(high - low, 1e-12)
    half_width = 1.28 * (summary["n1_max_A"] - summary["n1_min_A"]) / 2
    ax.set_xlim(re - half_width, re + half_width)
    ax.set_ylim(low - .055 * span, high + (.44 if name == "LiH" else .16) * span)
    if name == "CO" and not absolute and low >= -160 and high <= 55:
        ax.set_ylim(-160, 55)
        ax.set_yticks([-150, -100, -50, 0, 50])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.axvline(re, color=COLORS["gray"], lw=.8, ls=":")
    if not absolute:
        ax.axhline(0, color=COLORS["gray"], lw=.8)
    ax.text(.97, .96, NAMES.get(name, name), transform=ax.transAxes,
            ha="right", va="top", fontweight="semibold", fontsize=12)
    ax.set_xlabel(COORDINATES.get(name, "Bond distance") + " (Å)")
    ax.set_ylabel(r"$E$ (Hartree)" if absolute else r"Energy relative to $E_{RHF}(R_e)$ (mHa)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if absolute:
        ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=.85)
    ax.set_axisbelow(True)


def write_plots(document, output_dir, *, absolute=False):
    """Save the reference-style compact overview, single panels, and data table."""
    validate_document(document)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qml-initialdata-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    order = {name: index for index, name in enumerate(PANEL_ORDER)}
    results = sorted(document["results"], key=lambda result: order.get(result["molecule"], len(order)))
    filenames = []
    style = {
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#CAD1D8",
        "axes.labelcolor": "#273444", "text.color": "#273444",
        "xtick.color": COLORS["gray"], "ytick.color": COLORS["gray"],
        "grid.color": "#E3E8EC", "grid.linewidth": .6, "legend.frameon": False,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    }

    def save(figure, name):
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            figure.savefig(path, dpi=190)
            filenames.append(path.name)

    def inset_legend(ax):
        ax.legend(loc="upper left", fontsize=8, borderaxespad=.35,
                  labelspacing=.2, handlelength=1.7)

    with plt.rc_context(style):
        columns = min(3, len(results))
        rows = math.ceil(len(results) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.9 * rows), squeeze=False)
        try:
            figure.subplots_adjust(left=.084 if absolute else .066, right=.992,
                                   bottom=.065, top=.988, wspace=.25 if absolute else .12, hspace=.25)
            figure.supylabel(r"$E$ (Hartree)" if absolute else r"Energy relative to $E_{RHF}(R_e)$ (mHa)",
                             x=.008, fontsize=11)
            for index, result in enumerate(results):
                ax = axes.flat[index]
                plot_panel(ax, result, absolute=absolute)
                ax.set_ylabel("")
                if index == 0:
                    inset_legend(ax)
            for ax in list(axes.flat)[len(results):]:
                ax.set_visible(False)
            save(figure, "overview")
        finally:
            plt.close(figure)
        for result in results:
            figure, ax = plt.subplots(figsize=(6, 4.4))
            try:
                figure.subplots_adjust(left=.16, right=.98, bottom=.14, top=.98)
                plot_panel(ax, result, absolute=absolute)
                if result["molecule"] == results[0]["molecule"]:
                    inset_legend(ax)
                save(figure, result["molecule"])
            finally:
                plt.close(figure)
    fields = ["molecule", "point_index", "bond_length_A", "q_A", "E_RHF_Ha", "E_FCI_Ha",
              "E_corr_Ha", "E_corr_mHa", "display_reference_Ha", "display_RHF", "display_FCI", "display_unit"]
    with (output / "plotted_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            reference = 0.0 if absolute else result["summary"]["E_RHF_re_Ha"]
            factor = 1.0 if absolute else 1000.0
            for point in sorted(result["scan"], key=lambda row: row["point_index"]):
                row = {key: point[key] for key in fields[1:8]}
                row.update(molecule=result["molecule"], display_reference_Ha=reference,
                           display_RHF=factor * (point["E_RHF_Ha"] - reference),
                           display_FCI=factor * (point["E_FCI_Ha"] - reference),
                           display_unit="Ha" if absolute else "mHa")
                writer.writerow(row)
    filenames.append("plotted_points.csv")
    return [output / name for name in filenames]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "results/initialdata/initialdata.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/plots/initialdata")
    parser.add_argument("--absolute", action="store_true", help="Plot absolute electronic energies in Ha.")
    parser.add_argument("--no-harmonic", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        document = load_plot_data(args.input)
        files = write_plots(document, args.output_dir, absolute=args.absolute)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {len(files)} plot files and data tables to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
