#!/usr/bin/env python3
"""Plot saved RHF scans and harmonic ranges; no electronic calculations are run.

Requires numpy and matplotlib. Run with the system Python used for plotting.
Energies use Hartree; geometric coordinates use Angstrom.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qml-rhf-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qml-rhf-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


BLUE = "#255D9B"
ORANGE = "#C56517"
GREEN = "#268567"
PURPLE = "#7952A1"
GRAY = "#56616E"
NAMES = {"BeH2": r"BeH$_2$", "H2O": r"H$_2$O", "NH3": r"NH$_3$",
         "N2": r"N$_2$", "H2S": r"H$_2$S", "H2O2": r"H$_2$O$_2$"}


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CAD1D8", "axes.labelcolor": "#273444",
        "text.color": "#273444", "xtick.color": GRAY, "ytick.color": GRAY,
        "grid.color": "#E3E8EC", "grid.linewidth": .6,
        "legend.frameon": False, "legend.fontsize": 8,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def coordinate_label(name):
    return {
        "LiH": "Li-H distance", "BeH2": "Common Be-H length",
        "H2O": "Common O-H length", "NH3": "Common N-H length",
        "N2": "N-N distance", "CO": "C-O distance", "HF": "H-F distance",
        "H2S": "Common S-H length", "H2O2": "O-O separation",
    }[name]


def validate_record(record):
    s = record["summary"]
    if s.get("validation_status") != "PASS" or s.get("asymptote_status") != "PLATEAU_CONFIRMED":
        raise ValueError(f"{s['molecule']}: a validated plateau is required for these plots.")
    points = [p for p in record["scan"] if p["accepted"]]
    if len(points) != s["accepted_scan_points"]:
        raise ValueError(f"{s['molecule']}: inconsistent accepted point count.")
    q = np.array([p["q_A"] for p in points])
    energies = np.array([p["energy_Eh"] for p in points])
    if not np.isfinite(energies).all() or not np.all(np.diff(q) > 0):
        raise ValueError(f"{s['molecule']}: invalid scan data.")
    h = record["harmonic_input"]["summary"]
    k = h["k_q_Eh_per_A2"]
    for n, factor in ((0, .5), (1, 1.5)):
        if not np.isclose(.5 * k * h[f"Delta_q_n{n}_A"]**2,
                          factor * s["harmonic_quantum_Eh"], rtol=1e-9, atol=1e-14):
            raise ValueError(f"{s['molecule']}: harmonic units or turning points are inconsistent.")
    return points, q, energies


def style_axis(ax):
    ax.grid(axis="y", alpha=.85)
    ax.set_axisbelow(True)


def full_scan_panel(ax, record, compact=False):
    s = record["summary"]
    points, q, energies = validate_record(record)
    above_minimum = energies - s["E_eq_Eh"]
    ax.plot(q, above_minimum, "o-", color=BLUE, lw=1.5, ms=3.3,
            label="Accepted RHF samples")
    ax.axhline(s["D_e_Eh"], color=ORANGE, ls="--", lw=1.4,
               label=r"RHF-path threshold $D_e$")
    ax.set_xscale("symlog", linthresh=1., linscale=.8)
    ax.set_xlim(0, q[-1] * 1.25)
    ax.set_ylim(-.04 * s["D_e_Eh"], 1.13 * s["D_e_Eh"])
    ax.set_xlabel(r"Stretch displacement $q$ (Å; symlog)")
    ax.set_ylabel(r"$E(q)-E_{eq}$ (Hartree)")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if compact:
        title = NAMES.get(s["molecule"], s["molecule"])
        ax.set_title(f"{title}  |  {len(points)} accepted points", loc="left", fontweight="semibold")
        ax.text(.97, .07, f"Selected n = {s['selected_n']}\n"
                f"[{s['selected_range_min_A']:.6f}, {s['selected_range_max_A']:.6f}] Å",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 3})
    else:
        tail = record["accepted_plateau"]["window_q_A"]
        ax.axvspan(tail[0], tail[-1], color=GREEN, alpha=.12, label="Validated tail window")
        ax.set_title("2  Full outward RHF scan", loc="left", fontweight="semibold")
        ax.legend(loc="lower right", fontsize=8)
    style_axis(ax)


def local_panel(ax, record, compact=False):
    s, h = record["summary"], record["harmonic_input"]["summary"]
    _, q, energy = validate_record(record)
    width = 1.28 * h["Delta_q_n1_A"]
    local_q = np.linspace(-width, width, 501)
    model = .5 * h["k_q_Eh_per_A2"] * local_q**2
    ax.axvspan(s["selected_range_min_A"], s["selected_range_max_A"],
               color=GREEN, alpha=.12, label=f"Selected n={s['selected_n']} range")
    ax.plot(s["s_eq_A"] + local_q, model, color=BLUE, ls="--", lw=1.7,
            label="Local harmonic model")
    use = q <= width
    ax.scatter(s["s_eq_A"] + q[use], energy[use] - s["E_eq_Eh"],
               color=ORANGE, s=28, zorder=4, label="Existing RHF samples")
    for n, field, color in ((0, "half_hbar_omega_Eh", PURPLE), (1, "three_halves_hbar_omega_Eh", GREEN)):
        level = s[field]
        ax.hlines(level, h[f"n{n}_min_A"], h[f"n{n}_max_A"], color=color, lw=1.8)
        ax.scatter([h[f"n{n}_min_A"], h[f"n{n}_max_A"]], [level, level],
                   s=16, color=color, zorder=5)
        ax.annotate(f"n={n}", (s["s_eq_A"], level), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=color, fontsize=9)
    ax.axvline(s["s_eq_A"], color=GRAY, lw=.8, ls=":")
    ax.set_xlim(s["s_eq_A"] - width, s["s_eq_A"] + width)
    ax.set_ylim(-.055 * model.max(), 1.3 * model.max())
    title = NAMES.get(s["molecule"], s["molecule"]) if compact else "1  Local harmonic range"
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(coordinate_label(s["molecule"]) + " (Å)")
    ax.set_ylabel(r"$E-E_{eq}$ (Hartree)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if compact:
        ax.text(.98, .97,
                f"[{s['selected_range_min_A']:.6f}, {s['selected_range_max_A']:.6f}] Å",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GREEN)
    else:
        ax.legend(loc="upper center", fontsize=7.5, ncol=1)
    style_axis(ax)


def convergence_panel(ax, record, tolerance):
    s = record["summary"]
    _, q, energies = validate_record(record)
    difference = np.abs(energies - s["E_diss_Eh"])
    # The reference point is exactly zero by definition. Omit it from log-y;
    # never invent a positive floor and portray that floor as computed data.
    use = (q >= 5) & (difference > 0)
    ax.loglog(q[use], difference[use], "o-", color=BLUE, lw=1.5, ms=4)
    tail = record["accepted_plateau"]["window_q_A"]
    ax.axvspan(tail[0], tail[-1], color=GREEN, alpha=.12)
    ax.axhline(tolerance, color=ORANGE, ls="--", lw=1.4,
               label=f"Plateau tolerance: {tolerance:.0e} Hartree")
    ax.set_xlim(4., q[-1] * 1.3)
    ax.set_title("3  Asymptote convergence", loc="left", fontweight="semibold")
    ax.set_xlabel(r"Stretch displacement $q$ (Å; log)")
    ax.set_ylabel(r"$|E(q)-E_{diss}|$ (Hartree; log)")
    last = record["accepted_plateau"]
    ax.text(.04, .04,
            f"Final four-point span: {last['pairwise_span_Eh']:.2e} Hartree\n"
            f"Farther-step change: {last['extension_change_Eh']:.2e} Hartree\n"
            "Final reference has zero error; omitted from log-y.",
            transform=ax.transAxes, va="bottom", fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 4})
    ax.legend(loc="upper right", fontsize=7.5)
    style_axis(ax)


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=190)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def molecule_figure(record, document, output):
    s = record["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.subplots_adjust(left=.065, right=.985, bottom=.25, top=.76, wspace=.33)
    fig.suptitle(f"{NAMES.get(s['molecule'], s['molecule'])}  |  RHF/{s['basis'].upper()}",
                 x=.065, y=.975, ha="left", fontsize=21, fontweight="bold")
    fig.text(.065, .87,
             f"{coordinate_label(s['molecule'])}  ·  selected n = {s['selected_n']}  ·  "
             f"range [{s['selected_range_min_A']:.6f}, {s['selected_range_max_A']:.6f}] Å  ·  "
             f"{s['accepted_scan_points']} accepted scan points", fontsize=11)
    local_panel(axes[0], record)
    full_scan_panel(axes[1], record)
    convergence_panel(axes[2], record, document["settings"]["plateau_tolerance_Eh"])
    fig.text(.065, .145,
             f"RHF-path well depth = {s['D_e_Eh']:.9f} Hartree    |    "
             f"n=0 energy above minimum = {s['half_hbar_omega_Eh']:.9f} Hartree    |    "
             f"n=1 energy above minimum = {s['three_halves_hbar_omega_Eh']:.9f} Hartree", fontsize=10)
    fig.text(.065, .095,
             f"Relative to dissociation: E0 = {s['E0_rel_Eh']:+.9f} Hartree; "
             f"E1 = {s['E1_rel_Eh']:+.9f} Hartree.  Negative means below the RHF-path threshold.", fontsize=10)
    fig.text(.065, .042,
             "Constrained RHF model only. Lines between RHF samples guide the eye; the local dashed curve is a harmonic approximation. "
             "No negative-q RHF scan was performed.", fontsize=8, color=GRAY)
    save_figure(fig, output / s["molecule"])


def overview_figure(results, output):
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.subplots_adjust(left=.075, right=.98, bottom=.065, top=.92, hspace=.48, wspace=.3)
    for ax, record in zip(axes.flat, results):
        local_panel(ax, record, compact=True)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.075, .98), ncol=3,
               borderaxespad=0, fontsize=10, handlelength=2.7, columnspacing=2.5)
    save_figure(fig, output / "overview")


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "json/rhf_bound_level_selection.json")
    parser.add_argument("--output", type=Path, default=root / "results/plots")
    args = parser.parse_args(argv)
    document = json.loads(args.input.read_text())
    if document.get("energy_unit") != "Hartree":
        parser.error("The input must be the Hartree version of the saved results.")
    results = document["results"]
    if not 1 <= len(results) <= 9:
        parser.error("Expected one to nine molecule results.")
    for record in results:
        validate_record(record)
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    overview_figure(results, args.output)
    readme = ["# Local harmonic ranges: 3 x 3 figure", "",
              "overview.png and overview.svg contain only the local harmonic-range panel for all nine molecules. "
              "No dissociation-scan or convergence panels are included in the remade figure.", "",
              "Energies use Hartree and are measured relative to each equilibrium minimum. Lengths use Angstrom. "
              "Blue dashed curves are local harmonic approximations; orange markers are existing RHF samples. "
              "The n=0/n=1 horizontal segments terminate at their classical turning points. Green shading marks the selected range.", "",
              "Each panel uses its own linear axis limits. H2O2 uses O-O separation. "
              "All graphics reuse saved results without new electronic calculations.", ""]
    (args.output / "README.md").write_text("\n".join(readme))
    print(f"Saved overview and figure notes to {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
