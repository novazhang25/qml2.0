#!/usr/bin/env python3
"""Plot Run 1 FG/FE/MB-1 seed metrics and signed-error curves.

Adapted from encoding_qml/plot_seed_sweep_5abc_results.py. Uses the custom palette,
mean +/- SE bars with per-seed scatter, mean +/- sample SD signed-error bands,
and chemical-accuracy lines.
Reads one run1 timestamp directory; does not train models or change results.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qml_run1_plot_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qml_run1_plot_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SUMMARY_CSV = "metrics.csv"
PREDICTIONS_CSV = "predictions.csv"
CHEMICAL_ACCURACY_MHA = 1.593
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
DISPLAY_RUN_LABELS = {"FG": "FG", "FE": "FE", "MB-1": "MB1", "MB1_PRIME": "MB1'"}
COLORS = {
    "MB-1": "#0072B2",       # blue
    "MB-2": "#009E73",       # green
    "MB-3": "#CC79A7",       # pink
    "FG": "#E69F00",         # orange
    "FE": "#6F52A2",         # purple
    "Random T": "#22A7C6",   # cyan
    "Shuffled T": "#A78BCE", # lavender
    "Ansatz": "#707070",     # gray
    "Control 1": "#4F6DB8",  # periwinkle blue
    "Control 2": "#C07AB8",  # mauve
}
RUN_COLORS = {run: COLORS[run] for run in ("FG", "FE", "MB-1")}
RUN_COLORS['MB1_PRIME'] = '#009E73'


@dataclass(frozen=True)
class SeedRun:
    molecule: str
    run: str
    seed: int | None
    path: Path
    summary_path: Path
    predictions_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path,
                        help="One completed results/run1/<timestamp> directory.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--molecules", nargs="+", default=["CO"])
    parser.add_argument("--runs", nargs="+", choices=list(RUN_COLORS), default=["FG", "FE", "MB-1"])
    parser.add_argument("--expected-seeds", type=int, default=32,
                        help="Require exactly seeds 0 through N-1 for every selected model.")
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="all",
                        help="Comparison curve population; all merges all saved splits. Bars always use test metrics.")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--font-scale", type=float, default=1.60)
    parser.add_argument("--overall-title", default="Signed error along the bond-stretching scan")
    parser.add_argument("--protocol-label", default="Run 1")
    parser.add_argument("--run-label-prefix", default="run1")
    args = parser.parse_args()
    if args.expected_seeds < 2 or args.dpi < 1:
        parser.error("--expected-seeds must be at least 2 and --dpi must be positive")
    args.molecules = list(dict.fromkeys(args.molecules))
    args.runs = list(dict.fromkeys(args.runs))
    return args

def seed_number(path: Path) -> int | None:
    match = re.fullmatch(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else None

def existing_seed_run(molecule: str, run: str, seed_dir: Path) -> SeedRun | None:
    summary = seed_dir / SUMMARY_CSV
    predictions = seed_dir / PREDICTIONS_CSV
    if seed_number(seed_dir) is not None and summary.is_file() and predictions.is_file():
        return SeedRun(molecule, run, seed_number(seed_dir), seed_dir, summary, predictions)
    return None

def discover_seed_runs(input_root: Path, molecules: Iterable[str], runs: Iterable[str]) -> list[SeedRun]:
    """Read only runs/<molecule>/<model>/seed_XX within one result directory."""
    found = []
    for molecule in molecules:
        for run in runs:
            model_dir = input_root / "runs" / molecule / run
            for seed_dir in sorted(model_dir.glob("seed_*")):
                record = existing_seed_run(molecule, run, seed_dir)
                if record is not None:
                    found.append(record)
    return found

def load_metric_table(records: list[SeedRun]) -> pd.DataFrame:
    rows = []
    for record in records:
        frame = pd.read_csv(record.summary_path)
        required = {"molecule", "model", "seed", "split", "mae_mHa", "rmse_mHa"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing metric columns in {record.summary_path}: {required - set(frame.columns)}")
        row = frame[frame["split"] == "test"]
        if len(row) != 1:
            raise ValueError(f"Exactly one test metric row required: {record.summary_path}")
        row = row.iloc[0]
        if (row["molecule"], row["model"], row["seed"]) != (record.molecule, record.run, record.seed):
            raise ValueError(f"Metric identity mismatch: {record.summary_path}")
        values = np.asarray([row["mae_mHa"], row["rmse_mHa"]], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Nonfinite or negative metrics: {record.summary_path}")
        rows.append(dict(molecule=record.molecule, run=record.run, seed=record.seed,
                         MAE_mHa=values[0], RMSE_mHa=values[1]))
    return pd.DataFrame(rows)

def summarize_metrics(metrics: pd.DataFrame, molecules: list[str], runs: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for molecule in molecules:
        for run in runs:
            subset = metrics[(metrics["molecule"] == molecule) & (metrics["run"] == run)]
            rows.append(
                {
                    "molecule": molecule,
                    "run": run,
                    "n_seeds": int(len(subset)),
                    "mean_MAE_mHa": float(subset["MAE_mHa"].mean()) if len(subset) else math.nan,
                    "std_MAE_mHa": float(subset["MAE_mHa"].std(ddof=1)) if len(subset) > 1 else 0.0,
                    "se_MAE_mHa": float(subset["MAE_mHa"].std(ddof=1) / math.sqrt(len(subset))) if len(subset) > 1 else math.nan,
                    "mean_RMSE_mHa": float(subset["RMSE_mHa"].mean()) if len(subset) else math.nan,
                    "std_RMSE_mHa": float(subset["RMSE_mHa"].std(ddof=1)) if len(subset) > 1 else 0.0,
                    "se_RMSE_mHa": float(subset["RMSE_mHa"].std(ddof=1) / math.sqrt(len(subset))) if len(subset) > 1 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def validate_seed_runs(records: list[SeedRun], molecules: list[str], runs: list[str], expected: int) -> None:
    """Require complete coverage and identical geometry/target/split populations."""
    for molecule in molecules:
        reference = None
        for run in runs:
            selected = [r for r in records if r.molecule == molecule and r.run == run]
            seeds = sorted(r.seed for r in selected)
            if seeds != list(range(expected)):
                raise ValueError(f"{molecule}/{run}: expected seeds 0..{expected - 1}; found {seeds}. "
                                 "Use a completed run directory.")
            for record in selected:
                frame = load_prediction(record, "all").sort_values("geometry_id")
                identity = frame[["geometry_id", "split", "x", "target_Ha"]].reset_index(drop=True)
                if reference is None:
                    reference = identity
                elif not identity.equals(reference):
                    raise ValueError(f"Geometry, split, bond length or target differs across models/seeds: {record.predictions_path}")

def format_molecule_label(molecule: str) -> str:
    """Render molecular formula digits as Unicode subscripts in plot labels."""

    return str(molecule).translate(SUBSCRIPT_DIGITS)


def format_run_label(run: str, prefix: str) -> str:
    """Render an internal run ID with the requested displayed protocol prefix."""

    if run in DISPLAY_RUN_LABELS:
        return DISPLAY_RUN_LABELS[run]
    if len(run) >= 2 and run[0].isdigit():
        return f"{prefix}{run[1:]}"
    return run


def run_color(run: str) -> str:
    """Use the custom palette for the supported Run 1 models."""
    return RUN_COLORS[run]

def setup_matplotlib(font_scale: float) -> None:
    if font_scale <= 0.0:
        raise ValueError("--font-scale must be positive.")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 11 * font_scale,
            "axes.labelsize": 12 * font_scale,
            "axes.titlesize": 13 * font_scale,
            "figure.titlesize": 15 * font_scale,
            "legend.fontsize": 10 * font_scale,
            "xtick.labelsize": 11 * font_scale,
            "ytick.labelsize": 11 * font_scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def compact_legend(ax, handles=None, labels=None, *, single_row=False):
    """Place a smaller legend in the upper-left corner, optionally in one row."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper left",
        frameon=False,
        ncol=max(1, len(handles) if single_row else math.ceil(len(handles) / 2)),
        fontsize=0.75 * plt.rcParams["legend.fontsize"],
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0.4,
    )


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    """Save each plot in both raster and vector formats."""

    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".pdf"))


def plot_metric_bars(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    molecules: list[str],
    runs: list[str],
    metric: str,
    output_path: Path,
    dpi: int,
    title: str | None = None,
    run_label_prefix: str = "5",
    ncols: int | None = None,
    model_only_legend: bool = False,
) -> None:
    mean_col = f"mean_{metric}_mHa"
    se_col = f"se_{metric}_mHa"
    ncols = min(ncols or len(molecules), len(molecules))
    nrows = math.ceil(len(molecules) / ncols)
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 if len(molecules) == 1 else 3.8 * ncols, 4.8 if len(molecules) == 1 else 3.5 * nrows),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    # Same symmetric, shuffled jitter recipe as plot_benchmark_32seed_nature.py.
    # Scale its 0.040 jitter / 0.145 bar-width ratio to these 0.8-wide bars.
    rng = np.random.default_rng(20260825)
    jitter_half_width = 0.8 * 0.040 / 0.145

    for ax, molecule in zip(axes, molecules):
        x = np.arange(len(runs))
        means = []
        ses = []
        for run in runs:
            row = summary[(summary["molecule"] == molecule) & (summary["run"] == run)]
            means.append(float(row[mean_col].iloc[0]) if not row.empty else math.nan)
            ses.append(float(row[se_col].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            x,
            means,
            yerr=ses,
            capsize=4,
            error_kw={"zorder": 4.5, "elinewidth": 0.9, "capthick": 0.9},
            color=[run_color(run) for run in runs],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.82,
            zorder=2,
        )

        for position, run in zip(x, runs):
            seed_rows = metrics[(metrics["molecule"] == molecule) & (metrics["run"] == run)].sort_values("seed")
            values = seed_rows[f"{metric}_mHa"].to_numpy(dtype=float)
            positive = rng.uniform(0.05, 1.0, size=len(values) // 2)
            horizontal = np.concatenate((-positive, positive, np.zeros(len(values) % 2)))
            if horizontal.size and np.max(np.abs(horizontal)) > 0:
                horizontal *= jitter_half_width / np.max(np.abs(horizontal))
            horizontal = horizontal[rng.permutation(horizontal.size)]
            ax.scatter(
                position + horizontal, values,
                s=12.0, marker="o", facecolors=run_color(run),
                edgecolors="#555555" if run == "FG" else "none",
                linewidths=0.25 if run == "FG" else 0.0,
                alpha=0.60 if run == "FG" else 0.45,
                zorder=3.2, clip_on=True,
            )

        ax.set_xticks(x, [format_run_label(run, run_label_prefix) for run in runs])
        if metric == "MAE":
            ax.tick_params(axis="y", which="both", left=True, labelleft=True)
            ax.set_yticks([0, 2, 4, 6])
        ax.axhline(
            CHEMICAL_ACCURACY_MHA,
            color="0.25",
            linewidth=0.9,
            linestyle=":",
            alpha=0.85,
            label="Chemical accuracy",
        )
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)

    for ax in axes[len(molecules):]:
        ax.set_visible(False)

    for row in range(nrows):
        axes[row * ncols].set_ylabel(f"Test {metric} / mHa")
    for ax in axes[:len(molecules)]:
        if model_only_legend:
            handles = [Patch(facecolor=run_color(run), edgecolor="black", alpha=0.82)
                       for run in runs]
            compact_legend(ax, handles, [format_run_label(run, run_label_prefix) for run in runs], single_row=True)
            continue
        handles, labels = ax.get_legend_handles_labels()
        if metric == "MAE":
            bar_handles = [Patch(facecolor=run_color(run), edgecolor="black", alpha=0.82)
                           for run in runs]
            errorbar_handle = Line2D([], [], color="black", marker="|", linestyle="none",
                                    markersize=10, markeredgewidth=1.0)
            handles = bar_handles + handles + [errorbar_handle]
            labels = [format_run_label(run, run_label_prefix) for run in runs] + labels + ["Error bars: mean +/- SE"]
        else:
            handles.append(Line2D([], [], color="black", marker="|", linestyle="none",
                                  markersize=10, markeredgewidth=1.0))
            labels.append("Error bars: mean +/- SE")
        handles.append(Line2D([], [], marker="o", linestyle="none", color="#555555",
                              markersize=3.5, alpha=0.6))
        labels.append("Each seed")
        compact_legend(ax, handles, labels)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)
    plt.close(fig)


def load_prediction(record: SeedRun, split: str) -> pd.DataFrame:
    frame = pd.read_csv(record.predictions_path, dtype={"geometry_id": str})
    required = {"molecule", "model", "seed", "split", "geometry_id", "bond_length_A",
                "target_Ha", "prediction_Ha", "error_Ha"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prediction columns in {record.predictions_path}: {required - set(frame.columns)}")
    if (len(frame) != 30 or frame.geometry_id.duplicated().any()
            or set(frame.geometry_id) != {f"{i:03d}" for i in range(1, 31)}):
        raise ValueError(f"Expected each geometry 001..030 exactly once: {record.predictions_path}")
    if not ((frame.molecule == record.molecule) & (frame.model == record.run)
            & (frame.seed == record.seed)).all():
        raise ValueError(f"Prediction identity mismatch: {record.predictions_path}")
    if set(frame["split"]) not in ({"train", "test"}, {"train", "validation", "test"}):
        raise ValueError(f"Invalid prediction splits: {record.predictions_path}")
    values = frame[["bond_length_A", "target_Ha", "prediction_Ha", "error_Ha"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame.bond_length_A <= 0).any():
        raise ValueError(f"Nonfinite predictions or invalid bond lengths: {record.predictions_path}")
    if not np.allclose(frame.error_Ha, frame.prediction_Ha - frame.target_Ha, rtol=0, atol=1e-12):
        raise ValueError(f"error_Ha must equal prediction_Ha - target_Ha: {record.predictions_path}")
    if split != "all":
        frame = frame[frame["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No {split} predictions in {record.predictions_path}")
    return pd.DataFrame({
        "molecule": record.molecule, "run": record.run, "seed": record.seed,
        "geometry_id": frame.geometry_id.to_numpy(), "split": frame["split"].to_numpy(),
        "x_name": "bond_length", "x": frame.bond_length_A.to_numpy(dtype=float),
        "target_Ha": frame.target_Ha.to_numpy(dtype=float),
        "signed_error_mHa": 1000.0 * frame.error_Ha.to_numpy(dtype=float),
    })

def signed_error_stats(records: list[SeedRun], molecule: str, run: str, split: str) -> tuple[pd.DataFrame, str]:
    frames = [
        load_prediction(record, split)
        for record in records
        if record.molecule == molecule and record.run == run
    ]
    if not frames:
        return pd.DataFrame(), "geom_idx"

    all_rows = pd.concat(frames, ignore_index=True)
    x_name = "bond_length" if (all_rows["x_name"] == "bond_length").any() else "geom_idx"
    grouped = (
        all_rows.groupby(["geometry_id", "x"], as_index=False)
        .agg(
            mean_signed_error_mHa=("signed_error_mHa", "mean"),
            std_signed_error_mHa=("signed_error_mHa", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            n_seeds=("seed", "count"),
        )
        .sort_values("x")
    )
    return grouped, x_name


def plot_signed_error_curves(
    records: list[SeedRun],
    molecules: list[str],
    runs: list[str],
    output_dir: Path,
    split: str,
    dpi: int,
    protocol_label: str,
    run_label_prefix: str,
    molecule_label_fontsize: float,
    *,
    band: str = "sd",
    model_only_legend: bool = False,
) -> None:
    if band not in ("sd", "se"):
        raise ValueError("Signed-error band must be 'sd' or 'se'")
    for molecule in molecules:
        molecule_output_dir = output_dir / molecule
        molecule_output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 5.1))
        x_label = "Geometry index"
        for run in runs:
            stats, x_name = signed_error_stats(records, molecule, run, split)
            if stats.empty:
                warnings.warn(f"{molecule} {run}: no prediction data found; skipping curve.")
                continue
            x = stats["x"].to_numpy(dtype=float)
            y = stats["mean_signed_error_mHa"].to_numpy(dtype=float)
            ystd = stats["std_signed_error_mHa"].to_numpy(dtype=float)
            if band == "se":
                ystd = ystd / np.sqrt(stats["n_seeds"].to_numpy(dtype=float))
            color = run_color(run)

            ax.plot(
                x,
                y,
                label=format_run_label(run, run_label_prefix),
                color=color,
                linestyle="-",
                linewidth=2.0,
                zorder=3,
            )
            ax.fill_between(x, y - ystd, y + ystd, color=color, alpha=0.13, linewidth=0, zorder=2)
            if x_name == "bond_length":
                x_label = "Bond length / Å"

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.85, zorder=1)
        ax.axhline(
            CHEMICAL_ACCURACY_MHA,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
            zorder=1,
            label="Chemical accuracy",
        )
        ax.axhline(
            -CHEMICAL_ACCURACY_MHA,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
            zorder=1,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Signed error / mHa")
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        if model_only_legend:
            handles, labels = ax.get_legend_handles_labels()
            model_labels = [format_run_label(run, run_label_prefix) for run in runs]
            selected = [(handle, label) for handle, label in zip(handles, labels)
                        if label in model_labels]
            compact_legend(ax, [h for h, _ in selected], [label for _, label in selected], single_row=True)
        elif band == "se":
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor="0.5", alpha=0.13, edgecolor="none"))
            labels.append("Shading: mean +/- SE")
            compact_legend(ax, handles, labels)
        else:
            compact_legend(ax)
        fig.tight_layout()
        save_figure(
            fig,
            molecule_output_dir / f"signed_error_{run_label_prefix}_comparison.png",
            dpi=dpi,
        )
        plt.close(fig)


def plot_overall_signed_error(
    records: list[SeedRun],
    molecules: list[str],
    runs: list[str],
    output_path: Path,
    split: str,
    dpi: int,
    title: str,
    run_label_prefix: str,
    molecule_label_fontsize: float,
    title_fontsize: float,
    ncols: int = 3,
) -> None:
    ncols = min(ncols, len(molecules))
    nrows = math.ceil(len(molecules) / ncols)
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.7 * nrows),
        squeeze=False,
    )
    axes = list(axes_grid.flat)

    for index, (ax, molecule) in enumerate(zip(axes, molecules)):
        x_label = "Geometry index"
        for run in runs:
            stats, x_name = signed_error_stats(records, molecule, run, split)
            if stats.empty:
                warnings.warn(f"{molecule} {run}: no prediction data found; skipping curve.")
                continue
            x = stats["x"].to_numpy(dtype=float)
            y = stats["mean_signed_error_mHa"].to_numpy(dtype=float)
            ystd = stats["std_signed_error_mHa"].to_numpy(dtype=float)
            color = run_color(run)
            ax.plot(
                x,
                y,
                label=format_run_label(run, run_label_prefix),
                color=color,
                linestyle="-",
                linewidth=2.0,
                zorder=3,
            )
            ax.fill_between(
                x,
                y - ystd,
                y + ystd,
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            if x_name == "bond_length":
                x_label = "Bond length / Å"

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.85, zorder=1)
        ax.axhline(
            CHEMICAL_ACCURACY_MHA,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
            zorder=1,
            label="Chemical accuracy",
        )
        ax.axhline(
            -CHEMICAL_ACCURACY_MHA,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
            zorder=1,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Signed error / mHa")
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)

    for ax in axes[len(molecules):]:
        ax.set_visible(False)

    for ax in axes[:len(molecules)]:
        compact_legend(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)
    plt.close(fig)


def zoom_ylim(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    lower = min(float(np.nanmin(y - ystd)) for y, ystd in curves)
    upper = max(float(np.nanmax(y + ystd)) for y, ystd in curves)
    lower = min(lower, -CHEMICAL_ACCURACY_MHA)
    upper = max(upper, CHEMICAL_ACCURACY_MHA)
    if math.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    pad = 0.12 * (upper - lower)
    return lower - pad, upper + pad


def plot_model_split_curves(
    records: list[SeedRun],
    molecules: list[str],
    output_dir: Path,
    split: str,
    dpi: int,
    protocol_label: str,
    run_label_prefix: str,
    molecule_label_fontsize: float,
    zoom_run: str,
) -> None:
    color = run_color(zoom_run)
    split_styles = {"train": "-", "validation": "-.", "test": "--"}

    for molecule in molecules:
        molecule_output_dir = output_dir / molecule
        molecule_output_dir.mkdir(parents=True, exist_ok=True)
        curve_data: list[tuple[str, pd.DataFrame, str]] = []
        x_label = "Geometry index"
        example = next(r for r in records if r.molecule == molecule and r.run == zoom_run)
        available_splits = set(load_prediction(example, "all")["split"])
        for curve_split in [name for name in ("train", "validation", "test") if name in available_splits]:
            stats, x_name = signed_error_stats(records, molecule, zoom_run, curve_split)
            if stats.empty:
                warnings.warn(f"{molecule} {zoom_run} {curve_split}: no prediction data found; skipping zoom curve.")
                continue
            curve_data.append((curve_split, stats, x_name))
            if x_name == "bond_length":
                x_label = "Bond length / Å"

        if not curve_data:
            continue

        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        ylim_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for curve_split, stats, _ in curve_data:
            x = stats["x"].to_numpy(dtype=float)
            y = stats["mean_signed_error_mHa"].to_numpy(dtype=float)
            ystd = stats["std_signed_error_mHa"].to_numpy(dtype=float)
            ylim_curves.append((y, ystd))
            ax.plot(
                x,
                y,
                label=f"{format_run_label(zoom_run, run_label_prefix)} {curve_split.title()}",
                color=color,
                linestyle=split_styles.get(curve_split, "-"),
                linewidth=2.0,
                zorder=3,
            )
            ax.fill_between(x, y - ystd, y + ystd, color=color, alpha=0.13, linewidth=0, zorder=2)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.85, zorder=1)
        ax.axhline(
            CHEMICAL_ACCURACY_MHA,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
            zorder=1,
            label="Chemical accuracy",
        )
        ax.axhline(
            -CHEMICAL_ACCURACY_MHA,
            color="0.45",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
            zorder=1,
        )
        ax.set_ylim(*zoom_ylim(ylim_curves))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Signed error / mHa")
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        compact_legend(ax)
        fig.tight_layout()
        save_figure(
            fig,
            molecule_output_dir
            / f"signed_error_{format_run_label(zoom_run, run_label_prefix)}.png",
            dpi=dpi,
        )
        plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_matplotlib(args.font_scale)
    molecule_label_fontsize = 18.0 * args.font_scale
    signed_error_title_fontsize = 18.0 * args.font_scale
    records = discover_seed_runs(args.input_root, args.molecules, args.runs)
    if not records:
        raise SystemExit(f"No seed result directories found under {args.input_root}")

    validate_seed_runs(records, args.molecules, args.runs, args.expected_seeds)
    metrics = load_metric_table(records)
    summary = summarize_metrics(metrics, args.molecules, args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "per_seed_test_metrics.csv", index=False)
    summary_path = args.output_dir / "summary_run1_seed_metrics.csv"
    summary.to_csv(summary_path, index=False)

    for molecule in args.molecules:
        molecule_output_dir = args.output_dir / molecule
        molecule_output_dir.mkdir(parents=True, exist_ok=True)
        plot_metric_bars(
            metrics,
            summary,
            [molecule],
            args.runs,
            "MAE",
            molecule_output_dir / "barplot_MAE.png",
            args.dpi,
        )
        plot_metric_bars(
            metrics,
            summary,
            [molecule],
            args.runs,
            "RMSE",
            molecule_output_dir / "barplot_RMSE.png",
            args.dpi,
        )

    if len(args.molecules) > 1:
        plot_metric_bars(
            metrics,
            summary,
            args.molecules,
            args.runs,
            "MAE",
            args.output_dir / "overall_barplot_MAE.png",
            args.dpi,
            title="Test MAE across seeds",
            run_label_prefix=args.run_label_prefix,
            ncols=3,
        )
        plot_metric_bars(
            metrics,
            summary,
            args.molecules,
            args.runs,
            "RMSE",
            args.output_dir / "overall_barplot_RMSE.png",
            args.dpi,
            run_label_prefix=args.run_label_prefix,
            ncols=3,
        )
        plot_overall_signed_error(
            records,
            args.molecules,
            args.runs,
            args.output_dir
            / f"overall_signed_error_{args.run_label_prefix}_comparison.png",
            args.split,
            args.dpi,
            args.overall_title,
            args.run_label_prefix,
            molecule_label_fontsize,
            signed_error_title_fontsize,
            ncols=3,
        )

    plot_signed_error_curves(
        records,
        args.molecules,
        args.runs,
        args.output_dir,
        args.split,
        args.dpi,
        args.protocol_label,
        args.run_label_prefix,
        molecule_label_fontsize,
    )
    plot_model_split_curves(
        records,
        args.molecules,
        args.output_dir,
        args.split,
        args.dpi,
        args.protocol_label,
        args.run_label_prefix,
        molecule_label_fontsize,
        "MB-1" if "MB-1" in args.runs else args.runs[-1],
    )

    print(summary.to_string(index=False))
    print(f"\nSaved plots and summary to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
