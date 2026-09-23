"""Plot the saved Part 3 absolute-energy comparison with literal zero.

``write_plots(results, output_dir)`` writes one PNG/SVG pair per molecule.
The local harmonic panel and the zero-comparison panel share the same
unshifted absolute energy reference. A negative total passes the requested
algebraic test only; the plots do not infer physical vibrational binding.
No electronic calculations, optimizations, fitting, or scans are performed.
Matplotlib is imported lazily; importing this module creates no output.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path


LEVEL_COLORS = ("#1664A5", "#8052A6")
CURVE_COLOR = "#4B5969"
ZERO_COLOR = "#AE4C22"
EQUILIBRIUM_COLOR = "#5A7861"
REFERENCE_WARNING = (
    "UNVALIDATED ZERO REFERENCE: E_re is the unshifted absolute RHF total energy, including nuclear repulsion.\n"
    "No physical zero threshold is established. Passing E_total < 0 is an algebraic result,\n"
    "not a determination that a physical vibrational level is bound."
)


def _finite(value, description):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Cannot plot nonfinite {description}.")
    return number


def _validated_fields(result):
    """Check the saved sums, units, flags, and unshifted energy reference."""
    summary = result.get("summary", result)
    name = str(summary["molecule"])
    numeric_keys = (
        "E_re_Ha", "r_e_A", "k_Ha_per_A2", "hbar_omega_Ha",
        "E_n0_Ha", "E_n1_Ha", "E_total_n0_Ha", "E_total_n1_Ha",
        "Delta_q_n0_A", "Delta_q_n1_A", "reference_shift_Ha",
        "n0_min_A", "n0_max_A", "n1_min_A", "n1_max_A",
    )
    values = {key: _finite(summary[key], f"{name}: {key}") for key in numeric_keys}
    if values["reference_shift_Ha"] != 0.0:
        raise ValueError(f"{name}: these plots require unshifted absolute energies.")
    if summary["energy_reference_status"] != "UNVALIDATED_ZERO_REFERENCE":
        raise ValueError(f"{name}: the unvalidated zero reference must be explicit.")
    if not str(summary["energy_reference_warning"]).strip():
        raise ValueError(f"{name}: the energy-reference warning is required.")
    if "input_validation" in result and result["input_validation"]["passed"] is not True:
        raise ValueError(f"{name}: saved input validation failed.")
    if values["k_Ha_per_A2"] <= 0 or values["hbar_omega_Ha"] <= 0:
        raise ValueError(f"{name}: positive curvature and harmonic quantum are required.")
    for n in (0, 1):
        level = values[f"E_n{n}_Ha"]
        total = values[f"E_total_n{n}_Ha"]
        amplitude = values[f"Delta_q_n{n}_A"]
        if level <= 0 or amplitude <= 0:
            raise ValueError(f"{name}: harmonic level and amplitude must be positive.")
        if not math.isclose(level, (n + 0.5) * values["hbar_omega_Ha"],
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name}: inconsistent n={n} harmonic energy.")
        if not math.isclose(total, values["E_re_Ha"] + level,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name}: E_total({n}) is not E_re + E_n.")
        if not math.isclose(0.5 * values["k_Ha_per_A2"] * amplitude ** 2,
                            level, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{name}: n={n} harmonic turning points are inconsistent.")
        for side, sign in (("min", -1), ("max", 1)):
            if not math.isclose(values[f"n{n}_{side}_A"], values["r_e_A"] + sign * amplitude,
                                rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"{name}: inconsistent n={n} {side} turning coordinate.")
        passed = summary[f"n{n}_pass"]
        if not isinstance(passed, bool) or passed != (total < 0.0):
            raise ValueError(f"{name}: n={n} pass flag must equal E_total({n}) < 0.")
    expected_selection = 1 if summary["n1_pass"] else 0 if summary["n0_pass"] else None
    if summary["selected_n"] != expected_selection:
        raise ValueError(f"{name}: selected_n disagrees with the strict zero comparison.")
    return summary, values


def _label(n, passed):
    return f"n={n}: passes <0? {'YES' if passed else 'NO'}"


def _limits(values, fraction=0.1):
    lower, upper = min(values), max(values)
    span = max(upper - lower, 1e-12)
    return lower - fraction * span, upper + fraction * span


def _local_panel(ax, summary, values):
    equilibrium = values["E_re_Ha"]
    extent = 1.22 * max(values["Delta_q_n0_A"], values["Delta_q_n1_A"])
    q = [-extent + 2 * extent * index / 400 for index in range(401)]
    curve = [equilibrium + 0.5 * values["k_Ha_per_A2"] * displacement ** 2
             for displacement in q]
    ax.plot(q, curve, color=CURVE_COLOR, lw=1.7,
            label=r"$E_{re} + \frac{1}{2}kq^2$")
    ax.axhline(equilibrium, color=EQUILIBRIUM_COLOR, lw=1.1, ls=":",
               label=r"$E_{re}$ (absolute RHF minimum)")
    for n, color in enumerate(LEVEL_COLORS):
        total = values[f"E_total_n{n}_Ha"]
        turning_point = values[f"Delta_q_n{n}_A"]
        ax.hlines(total, -turning_point, turning_point, color=color, lw=2.5,
                  linestyles="solid" if summary[f"n{n}_pass"] else "dashed",
                  label=rf"$E_{{re}}+E_{{n={n}}}$; {_label(n, summary[f'n{n}_pass'])}")
        ax.plot([-turning_point, turning_point], [total, total], "o", color=color, ms=4)
    ax.set(xlim=(-extent, extent), ylim=_limits([equilibrium, *curve], fraction=0.07),
           xlabel=r"Stretch displacement $q$ (Å)", ylabel="Absolute energy (Ha)",
           title="1  Local harmonic levels: absolute energy zoom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=8)


def _zero_panel(ax, summary, values):
    """Show literal zero and both totals on one unbroken absolute energy axis."""
    ax.axhline(0.0, color=ZERO_COLOR, lw=1.8, ls="--", label="Literal zero: 0 Ha")
    totals = [values[f"E_total_n{n}_Ha"] for n in (0, 1)]
    for n, color in enumerate(LEVEL_COLORS):
        total = totals[n]
        ax.vlines(n, min(0.0, total), max(0.0, total), color=color, alpha=0.13, lw=18)
        ax.hlines(total, n - 0.25, n + 0.25, color=color, lw=2.8,
                  linestyles="solid" if summary[f"n{n}_pass"] else "dashed")
        ax.plot(n, total, "o", color=color, ms=6)
        ax.annotate(f"{total:.10f} Ha\n{_label(n, summary[f'n{n}_pass'])}", (n, total),
                    xytext=(0, 14 if total <= 0 else -32), textcoords="offset points",
                    ha="center", va="bottom" if total <= 0 else "top", fontsize=9, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 3})
    ax.set_xticks([0, 1], [r"$E_{re}+E_{n=0}$", r"$E_{re}+E_{n=1}$"])
    ax.set(xlim=(-0.55, 1.55), ylim=_limits([0.0, *totals], 0.10),
           xlabel="Absolute harmonic level energy", ylabel="Absolute energy (Ha)",
           title=r"2  Requested algebraic comparison: $E_{total}(n) < 0$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=9)


def write_plots(results, output_dir):
    """Write absolute-level plots with the energy-reference warning visible.

    Strict comparisons of the saved unrounded sums determine pass flags.
    Existing per-molecule filenames are retained when figures are regenerated.
    Returned filenames are relative to ``output_dir``.
    """
    validated = [_validated_fields(result) for result in results]
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qml-part3-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qml-part3-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    filenames = []
    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.frameon": False, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        for summary, values in validated:
            fig, axes = plt.subplots(1, 2, figsize=(14.5, 8.0))
            try:
                _local_panel(axes[0], summary, values)
                _zero_panel(axes[1], summary, values)
                for ax in axes:
                    ax.grid(axis="y", alpha=0.22)
                    ax.set_axisbelow(True)
                    ax.yaxis.set_major_locator(MaxNLocator(6))
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                name = str(summary["molecule"])
                basis = str(summary.get("basis", ""))
                selected = summary["selected_n"]
                selection_text = "none" if selected is None else str(selected)
                fig.suptitle(f"{name} | {basis} | Harmonic n=0 / n=1: absolute totals and literal zero",
                             fontsize=14, y=0.98)
                fig.text(0.5, 0.922, REFERENCE_WARNING, ha="center", va="top", fontsize=10,
                         color="#873917", linespacing=1.45,
                         bbox={"facecolor": "#FFF2E7", "edgecolor": "#E2B998", "pad": 8})
                coordinate = str(summary.get("coordinate", "saved harmonic coordinate"))
                fig.text(0.5, 0.805,
                         f"Coordinate: {coordinate}\nEquilibrium = {values['r_e_A']:.6f} Å; "
                         f"highest n passing <0 among n=0,1: {selection_text}; reference shift = 0 Ha",
                         ha="center", va="center", fontsize=9)
                sums = [
                    f"n={n}: E_re ({values['E_re_Ha']:.10f}) + E_n ({values[f'E_n{n}_Ha']:.10f}) "
                    f"= E_total ({values[f'E_total_n{n}_Ha']:.10f}) Ha; "
                    f"passes <0? {'YES' if summary[f'n{n}_pass'] else 'NO'}"
                    for n in (0, 1)
                ]
                ranges = " | ".join(
                    f"n={n} turning coordinates: [{values[f'n{n}_min_A']:.6f}, "
                    f"{values[f'n{n}_max_A']:.6f}] Å" for n in (0, 1)
                )
                fig.text(0.5, 0.115, "\n".join(sums), ha="center", va="top", fontsize=9)
                fig.text(0.5, 0.047, ranges, ha="center", va="top", fontsize=9)
                fig.text(0.5, 0.012, "Both panels use the same unshifted absolute reference. "
                         "The left panel magnifies the harmonic energy spacing; the right includes literal zero.",
                         ha="center", va="bottom", fontsize=8, color=CURVE_COLOR)
                fig.subplots_adjust(left=0.085, right=0.98, bottom=0.31, top=0.72, wspace=0.30)
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "molecule"
                for extension in ("png", "svg"):
                    path = plot_dir / f"{safe_name}_vibrational_levels.{extension}"
                    fig.savefig(path, dpi=180, bbox_inches="tight")
                    filenames.append(str(path.relative_to(output_dir)))
            finally:
                plt.close(fig)
    return filenames


def main(argv=None):
    """Plot existing Part 3 results into a new directory, without rerunning Part 3."""
    import argparse
    import json
    import sys

    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path,
                        default=project / "results/bond_length_part3/vibrational_levels.json")
    parser.add_argument("--output-dir", type=Path,
                        default=project / "results/bond_length_part3_plots")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.input_json.read_text())
        if document.get("schema_version") != "part3-zero-energy-v2":
            raise ValueError("Expected current saved Part 3 results.")
        results = document["results"]
        if not results:
            raise ValueError("No saved Part 3 results to plot.")
        for result in results:
            _validated_fields(result)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        files = write_plots(results, output)
        print(f"Created {len(files)} PNG/SVG files in {output / 'plots'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Part 3 plotting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
