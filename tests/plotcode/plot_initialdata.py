#!/usr/bin/env python3
"""Plot saved initialdata results without running any molecular calculations.

The energy panel uses the saved RHF equilibrium energy as its zero by default.
The correlation panel always plots E_FCI - E_RHF in mHa.  FCI means frozen-core
FCI for the saved initialdata-v1 format.  Matplotlib is needed only for rendering.
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


PROJECT = Path(__file__).resolve().parents[2]
SUMMARY_NUMBERS = (
    "r_e_A", "E_RHF_re_Ha", "k_Ha_per_Bohr2", "k_Ha_per_A2", "mu_eff_amu",
    "omega_rad_per_s", "hbar_omega_Ha", "E_n1_Ha", "E_total_n1_Ha",
    "n1_min_A", "n1_max_A",
)
COLORS = {"rhf": "#2463A6", "fci": "#B33F62", "harmonic": "#687586", "n1": "#20846A"}


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


def plot_panels(energy_ax, correlation_ax, result, *, absolute=False, harmonic=True):
    """Plot saved electronic energies and their saved correlation difference."""
    summary = result["summary"]
    points = sorted(result["scan"], key=lambda point: point["point_index"])
    x = [point["bond_length_A"] for point in points]
    reference = 0.0 if absolute else summary["E_RHF_re_Ha"]
    factor = 1.0 if absolute else 1000.0
    for field, color, label, marker in (
        ("E_RHF_Ha", COLORS["rhf"], "RHF", "o"),
        ("E_FCI_Ha", COLORS["fci"], "Frozen-core FCI", "s"),
    ):
        energy_ax.plot(x, [factor * (point[field] - reference) for point in points],
                       color=color, marker=marker, markersize=3, linewidth=1.3, label=label)
        energy_ax.scatter([summary["r_e_A"]],
                          [factor * (result["equilibrium_energies"][field] - reference)],
                          color=color, marker="*", s=65, zorder=5)
    if harmonic:
        grid = [x[0] + (x[-1] - x[0]) * index / 200 for index in range(201)]
        energy_ax.plot(grid, [factor * (summary["E_RHF_re_Ha"] - reference
                       + 0.5 * summary["k_Ha_per_A2"] * (distance - summary["r_e_A"]) ** 2)
                       for distance in grid], color=COLORS["harmonic"], linestyle="--",
                       linewidth=1.1, label="RHF harmonic model")
        energy_ax.hlines(factor * (summary["E_total_n1_Ha"] - reference),
                         summary["n1_min_A"], summary["n1_max_A"], colors=COLORS["n1"],
                         linestyles=":", linewidth=1.4, label="Harmonic n=1")
    correlation_ax.plot(x, [point["E_corr_mHa"] for point in points],
                        color=COLORS["fci"], marker="o", markersize=3, linewidth=1.3)
    correlation_ax.scatter([summary["r_e_A"]],
                           [1000 * result["equilibrium_energies"]["E_corr_Ha"]],
                           color=COLORS["fci"], marker="*", s=65, zorder=5)
    energy_ax.set_title(f"{result['molecule']} · {summary['basis']}", loc="left", fontweight="semibold")
    energy_ax.set_ylabel("Total energy (Ha)" if absolute else "Energy relative to RHF equilibrium (mHa)")
    correlation_ax.set_ylabel(r"$E_{\mathrm{corr}}$ (mHa)")
    correlation_ax.set_xlabel("Bond distance (Å)")
    energy_ax.tick_params(labelbottom=False)
    for axis in (energy_ax, correlation_ax):
        axis.axvline(summary["r_e_A"], color="#8C959F", linewidth=0.8, linestyle=":")
        axis.grid(axis="y", color="#E4E8EC", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.margins(x=0.035, y=0.15)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)


def write_plots(document, output_dir, *, absolute=False, harmonic=True):
    """Save per-molecule and overview figures and an exact table of plotted values."""
    validate_document(document)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qml-initialdata-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results, filenames = document["results"], []
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
             "axes.spines.right": False, "axes.edgecolor": "#BCC5CE", "legend.frameon": False,
             "svg.fonttype": "none", "savefig.facecolor": "white"}

    def save(figure, name):
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            figure.savefig(path, dpi=180)
            filenames.append(path.name)

    with plt.rc_context(style):
        for result in results:
            figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7),
                                       gridspec_kw={"height_ratios": (2, 1)}, layout="constrained")
            try:
                plot_panels(*axes, result, absolute=absolute, harmonic=harmonic)
                axes[0].legend(fontsize=8, loc="best")
                save(figure, result["molecule"])
            finally:
                plt.close(figure)
        columns = min(3, len(results))
        rows = math.ceil(len(results) / columns)
        figure = plt.figure(figsize=(5.2 * columns, 5.1 * rows), layout="constrained")
        try:
            outer = figure.add_gridspec(rows, columns)
            for index, result in enumerate(results):
                inner = outer[index // columns, index % columns].subgridspec(2, 1, height_ratios=(2, 1))
                energy_ax = figure.add_subplot(inner[0])
                correlation_ax = figure.add_subplot(inner[1], sharex=energy_ax)
                plot_panels(energy_ax, correlation_ax, result, absolute=absolute, harmonic=harmonic)
                energy_ax.legend(fontsize=7, loc="best")
            figure.suptitle("RHF, frozen-core FCI, and correlation energy", fontsize=14)
            save(figure, "overview")
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
    parser.add_argument("--no-harmonic", action="store_true", help="Hide the saved RHF harmonic model and n=1 level.")
    args = parser.parse_args(argv)
    try:
        document = load_plot_data(args.input)
        files = write_plots(document, args.output_dir, absolute=args.absolute, harmonic=not args.no_harmonic)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {len(files)} plot files and data tables to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
