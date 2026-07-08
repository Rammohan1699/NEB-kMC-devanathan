#!/usr/bin/env python3
"""Regenerate KMC postprocess plots as presentation-style PNG files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


BAND_COLORS = {
    "grain_boundary": "#f4b6b6",
    "transition": "#ffe2a8",
    "bulk_grain_0": "#d7ecff",
    "bulk_grain_1": "#ddf3df",
}

ZONE_COLORS = {
    "bulk_grain_0": "#1f6fdf",
    "bulk_grain_1": "#2ca02c",
    "transition": "#ff9f1c",
    "grain_boundary": "#d62728",
}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 2.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.minor.width": 1.6,
            "ytick.minor.width": 1.6,
            "xtick.major.size": 9,
            "ytick.major.size": 9,
            "xtick.minor.size": 5,
            "ytick.minor.size": 5,
        }
    )
    return plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def i(row: dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def parse_window_from_name(name: str) -> Optional[tuple[int, int]]:
    match = re.search(r"steps_(\d+)_(\d+)", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def output_path(post_dir: Path, out_dir: Path, source_svg: Path) -> Path:
    rel = source_svg.relative_to(post_dir)
    return out_dir / rel.with_suffix(".png")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)


def style_axes(ax, *, tick_size: int = 24) -> None:
    ax.tick_params(axis="both", which="both", top=True, right=True, labelsize=tick_size, pad=9)
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)


def nice_ylim(values: np.ndarray, *, include_zero: bool = True) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(finite.min())
    hi = float(finite.max())
    if include_zero:
        lo = min(0.0, lo)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi + 0.10 * (hi - lo)


def parse_region_bands_from_svg(svg_path: Path, lx: float) -> list[tuple[float, float, str, str]]:
    if not svg_path.exists() or lx <= 0.0:
        return []
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    bands: list[tuple[float, float, str, str]] = []
    rect_re = re.compile(
        r'<rect x="(?P<x>[-+0-9.eE]+)" y="50" width="(?P<w>[-+0-9.eE]+)" height="405" '
        r'fill="(?P<color>#[0-9A-Fa-f]{6})" fill-opacity="0\.34"'
    )
    color_to_label = {color.lower(): label for label, color in BAND_COLORS.items()}
    for match in rect_re.finditer(text):
        color = match.group("color").lower()
        label = color_to_label.get(color)
        if not label:
            continue
        x_px = float(match.group("x"))
        w_px = float(match.group("w"))
        left = (x_px - 80.0) / 785.0 * lx
        right = (x_px + w_px - 80.0) / 785.0 * lx
        if right > 0.0 and left < lx and right > left:
            bands.append((max(0.0, left), min(lx, right), color, label))
    return bands


def add_region_bands(ax, bands: list[tuple[float, float, str, str]]) -> None:
    seen: set[str] = set()
    for left, right, color, label in bands:
        legend_label = label if label not in seen else None
        ax.axvspan(left, right, color=color, alpha=0.22, lw=0, label=legend_label)
        seen.add(label)


def plot_x_summary(
    plt,
    rows: list[dict[str, str]],
    out_png: Path,
    *,
    x_label: str,
    y_label: str,
    y_field: str = "mean_barrier_eV",
    x_bounds: Optional[tuple[float, float]] = None,
    bands: Optional[list[tuple[float, float, str, str]]] = None,
    legend_loc: str = "lower right",
) -> None:
    rows = [row for row in rows if row.get("mean_x_A", "") and row.get(y_field, "")]
    if not rows:
        return
    x = np.asarray([f(row, "mean_x_A") for row in rows], dtype=float)
    y = np.asarray([f(row, y_field) for row in rows], dtype=float)
    counts = np.asarray([max(i(row, "count", 1), 1) for row in rows], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    counts = counts[order]

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    if bands:
        add_region_bands(ax, bands)
    ax.plot(x, y, color="#4d4d4d", lw=3.5, solid_capstyle="round", zorder=3)
    sizes = 36.0 + 160.0 * np.sqrt(counts / counts.max())
    ax.scatter(x, y, s=sizes, color="#1f6fdf", edgecolor="#4d4d4d", linewidth=0.6, zorder=4)
    ax.set_xlabel(x_label, fontsize=34, labelpad=14)
    ax.set_ylabel(y_label, fontsize=34, labelpad=16)
    if x_bounds:
        ax.set_xlim(*x_bounds)
    else:
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(*nice_ylim(y, include_zero=True))
    if bands:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:4], labels[:4], loc=legend_loc, frameon=False, fontsize=14, ncol=1)
    style_axes(ax)
    fig.tight_layout(pad=0.35)
    save(fig, out_png)


def plot_move_class_fraction(plt, csv_path: Path, out_png: Path) -> None:
    rows = read_csv(csv_path)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        klass = str(row.get("move_class", ""))
        if not klass:
            continue
        by_class[klass].append(row)
        totals[klass] += f(row, "fraction_of_selected_moves", 0.0)
    if not by_class:
        return

    order = sorted(by_class, key=lambda key: totals[key], reverse=True)
    label_map = {
        "bulk_grain_0_internal": "bulk 0 internal",
        "bulk_grain_1_internal": "bulk 1 internal",
        "grain_boundary_internal": "GB internal",
        "transition_internal": "transition internal",
        "transition_entry_to_gb": "transition -> GB",
        "gb_exit_to_transition": "GB -> transition",
        "bulk_grain_0_entry_to_transition": "bulk 0 -> transition",
        "bulk_grain_1_entry_to_transition": "bulk 1 -> transition",
        "transition_exit_to_bulk_grain_0": "transition -> bulk 0",
        "transition_exit_to_bulk_grain_1": "transition -> bulk 1",
    }
    colors = [
        "#4d4d4d",
        "#1f6fdf",
        "#ff3b3f",
        "#2ca02c",
        "#ff9f1c",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#bcbd22",
        "#e377c2",
    ]
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    for idx, klass in enumerate(order):
        members = sorted(by_class[klass], key=lambda row: f(row, "mean_step"))
        x = np.asarray([f(row, "mean_step") / 1e6 for row in members], dtype=float)
        y = np.asarray([f(row, "fraction_of_selected_moves") for row in members], dtype=float)
        ax.plot(x, y, lw=2.8, color=colors[idx % len(colors)], label=label_map.get(klass, klass.replace("_", " ")))
    ax.set_xlabel(r"KMC step ($10^6$)", fontsize=34, labelpad=14)
    ax.set_ylabel("Move fraction", fontsize=34, labelpad=16)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10, handlelength=1.8, labelspacing=0.45)
    style_axes(ax, tick_size=23)
    fig.tight_layout(pad=0.35)
    save(fig, out_png)


def plot_neb_timing(plt, csv_path: Path, out_png: Path) -> None:
    rows = [row for row in read_csv(csv_path) if i(row, "steps_observed") > 0]
    if not rows:
        return
    x = np.asarray([f(row, "mean_step") / 1e6 for row in rows], dtype=float)
    wall = np.asarray([f(row, "neb_wall_time_per_step_s") for row in rows], dtype=float)
    cpu = np.asarray([f(row, "neb_cpu_time_per_step_s") for row in rows], dtype=float)
    speed = np.asarray([f(row, "relative_wall_speedup_vs_initial") for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax2 = ax.twinx()
    l1, = ax.plot(x, wall, color="#4d4d4d", lw=3.2, label="Wall / step")
    l2, = ax.plot(x, cpu, color="#1f6fdf", lw=3.0, label="CPU / step")
    l3, = ax2.plot(x, speed, color="#ff3b3f", lw=3.0, ls="--", dash_capstyle="round", label="Wall speedup")
    ax.set_xlabel(r"KMC step ($10^6$)", fontsize=34, labelpad=14)
    ax.set_ylabel("NEB time / step (s)", fontsize=31, labelpad=15)
    ax2.set_ylabel("Speedup", fontsize=31, labelpad=15)
    ax.set_xlim(left=0.0)
    ax.set_ylim(*nice_ylim(np.concatenate([wall, cpu]), include_zero=True))
    ax2.set_ylim(*nice_ylim(speed, include_zero=True))
    style_axes(ax, tick_size=22)
    ax2.tick_params(axis="y", which="both", direction="in", labelsize=22, width=2.0, size=9, pad=9)
    for spine in ax2.spines.values():
        spine.set_linewidth(2.2)
    ax.legend(
        [l1, l2, l3],
        [line.get_label() for line in [l1, l2, l3]],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=3,
        frameon=False,
        fontsize=13,
        handlelength=1.7,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.35)
    save(fig, out_png)


def plot_from_zone_bars(
    plt,
    rows: list[dict[str, str]],
    out_png: Path,
    *,
    label_order: list[str],
    ymax: Optional[float] = None,
) -> None:
    rows_by_label = {str(row.get("from_label", "")): row for row in rows}
    labels = [label for label in label_order if label in rows_by_label]
    if not labels:
        return
    values = np.asarray([f(rows_by_label[label], "mean_barrier_eV") for label in labels], dtype=float)
    counts = [i(rows_by_label[label], "count") for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.bar(x, values, color=[ZONE_COLORS.get(label, "#4d4d4d") for label in labels], edgecolor="#222222", linewidth=0.8)
    for xpos, value, count in zip(x, values, counts):
        ax.text(xpos, value + 0.002, f"n={count}", ha="center", va="bottom", fontsize=12, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in labels], fontsize=18)
    ax.set_ylabel("Barrier (eV)", fontsize=34, labelpad=16)
    ax.set_ylim(0.0, ymax if ymax else max(float(values.max()) * 1.20, 0.01))
    style_axes(ax, tick_size=22)
    fig.tight_layout(pad=0.35)
    save(fig, out_png)


def plot_zone_pair_heatmap(
    plt,
    rows: list[dict[str, str]],
    out_png: Path,
    *,
    label_order: list[str],
    vmin: float,
    vmax: float,
) -> None:
    row_by_pair = {(str(row.get("from_label", "")), str(row.get("to_label", ""))): row for row in rows}
    labels = [
        label
        for label in label_order
        if any((label, other) in row_by_pair or (other, label) in row_by_pair for other in label_order)
    ]
    if not labels:
        return
    matrix = np.full((len(labels), len(labels)), np.nan)
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for r, from_label in enumerate(labels):
        for c, to_label in enumerate(labels):
            row = row_by_pair.get((from_label, to_label))
            if row is not None:
                matrix[r, c] = f(row, "mean_barrier_eV")
                counts[r, c] = i(row, "count")

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f2f2f2")
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([label.replace("_", "\n") for label in labels], fontsize=16)
    ax.set_yticklabels([label.replace("_", "\n") for label in labels], fontsize=16)
    ax.set_xlabel("Destination zone", fontsize=28, labelpad=12)
    ax.set_ylabel("Origin zone", fontsize=28, labelpad=12)
    for r in range(len(labels)):
        for c in range(len(labels)):
            if math.isfinite(float(matrix[r, c])):
                ax.text(c, r, f"{matrix[r, c]:.3f}\n{counts[r, c]}", ha="center", va="center", fontsize=13)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18, width=1.8)
    cbar.set_label("Barrier (eV)", fontsize=24, labelpad=10)
    ax.tick_params(axis="both", which="both", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    fig.tight_layout(pad=0.35)
    save(fig, out_png)


def write_manifest(paths: list[Path], out_dir: Path) -> None:
    manifest = out_dir / "PNG_MANIFEST.txt"
    manifest.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    post_dir = args.postprocess_dir.expanduser().resolve()
    out_dir = (args.out_dir or post_dir / "presentation_png").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    written: list[Path] = []

    x_csv = post_dir / "avg_barrier_vs_path_x_position.csv"
    x_svg = post_dir / "avg_barrier_vs_path_x_position.svg"
    x_rows = read_csv(x_csv) if x_csv.exists() else []
    lx = max((f(row, "x_bin_right_A", 0.0) for row in x_rows), default=0.0)
    bands = parse_region_bands_from_svg(x_svg, lx)
    if x_rows:
        out = output_path(post_dir, out_dir, x_svg)
        plot_x_summary(
            plt,
            x_rows,
            out,
            x_label=r"x position ($\mathrm{\AA}$)",
            y_label="Barrier (eV)",
            x_bounds=(0.0, lx) if lx > 0.0 else None,
            bands=bands,
        )
        written.append(out)

        max_x_out = out_dir / "max_selected_barrier_vs_path_x_position.png"
        plot_x_summary(
            plt,
            x_rows,
            max_x_out,
            x_label=r"x position ($\mathrm{\AA}$)",
            y_label="Maximum barrier (eV)",
            y_field="max_barrier_eV",
            x_bounds=(0.0, lx) if lx > 0.0 else None,
            bands=bands,
            legend_loc="upper right",
        )
        written.append(max_x_out)

    max_csv = post_dir / "max_selected_barrier_vs_path_x_displacement.csv"
    max_svg = post_dir / "max_selected_barrier_vs_path_x_displacement.svg"
    if max_csv.exists():
        out = output_path(post_dir, out_dir, max_svg)
        plot_x_summary(
            plt,
            read_csv(max_csv),
            out,
            x_label=r"x displacement ($\mathrm{\AA}$)",
            y_label="Maximum barrier (eV)",
            y_field="max_barrier_eV",
        )
        written.append(out)

    move_csv = post_dir / "move_class_fraction_vs_time.csv"
    move_svg = post_dir / "move_class_fraction_vs_time.svg"
    if move_csv.exists():
        out = output_path(post_dir, out_dir, move_svg)
        plot_move_class_fraction(plt, move_csv, out)
        written.append(out)

    timing_csv = post_dir / "cache_speedup" / "neb_timing_vs_step.csv"
    timing_svg = post_dir / "cache_speedup" / "neb_timing_vs_step.svg"
    if timing_csv.exists():
        out = output_path(post_dir, out_dir, timing_svg)
        plot_neb_timing(plt, timing_csv, out)
        written.append(out)

    by_step_dir = post_dir / "avg_barrier_vs_path_x_position_by_timestep"
    for csv_path in sorted(by_step_dir.glob("*.csv")):
        svg_path = csv_path.with_suffix(".svg")
        out = output_path(post_dir, out_dir, svg_path)
        plot_x_summary(
            plt,
            read_csv(csv_path),
            out,
            x_label=r"x position ($\mathrm{\AA}$)",
            y_label="Barrier (eV)",
            x_bounds=(0.0, lx) if lx > 0.0 else None,
            bands=bands,
        )
        written.append(out)

    tw_dir = post_dir / "time_window_region_barriers"
    label_order = ["bulk_grain_0", "transition", "grain_boundary", "bulk_grain_1"]
    tw_x_csv = tw_dir / "time_window_avg_barrier_vs_path_x_position.csv"
    tw_from_csv = tw_dir / "time_window_selected_barrier_summary_by_from_zone.csv"
    tw_pair_csv = tw_dir / "time_window_selected_barrier_summary_by_zone_pair.csv"
    tw_x_rows = read_csv(tw_x_csv) if tw_x_csv.exists() else []
    tw_from_rows = read_csv(tw_from_csv) if tw_from_csv.exists() else []
    tw_pair_rows = read_csv(tw_pair_csv) if tw_pair_csv.exists() else []
    ymax_from = max((f(row, "mean_barrier_eV", 0.0) for row in tw_from_rows), default=0.0) * 1.20
    pair_values = [f(row, "mean_barrier_eV") for row in tw_pair_rows if math.isfinite(f(row, "mean_barrier_eV"))]
    vmin = min(pair_values) if pair_values else 0.0
    vmax = max(pair_values) if pair_values else 1.0

    for svg_path in sorted(tw_dir.glob("avg_barrier_vs_path_x_position_steps_*.svg")):
        window = parse_window_from_name(svg_path.name)
        if not window:
            continue
        start, end = window
        rows = [
            row
            for row in tw_x_rows
            if i(row, "window_start_step") == start and i(row, "window_end_step") == end
        ]
        if not rows:
            continue
        out = output_path(post_dir, out_dir, svg_path)
        plot_x_summary(
            plt,
            rows,
            out,
            x_label=r"x position ($\mathrm{\AA}$)",
            y_label="Barrier (eV)",
            x_bounds=(0.0, lx) if lx > 0.0 else None,
            bands=bands,
        )
        written.append(out)

    for svg_path in sorted(tw_dir.glob("avg_selected_barrier_by_from_zone_steps_*.svg")):
        window = parse_window_from_name(svg_path.name)
        if not window:
            continue
        start, end = window
        rows = [
            row
            for row in tw_from_rows
            if i(row, "window_start_step") == start and i(row, "window_end_step") == end
        ]
        if not rows:
            continue
        out = output_path(post_dir, out_dir, svg_path)
        plot_from_zone_bars(plt, rows, out, label_order=label_order, ymax=ymax_from)
        written.append(out)

    for svg_path in sorted(tw_dir.glob("avg_selected_barrier_zone_pair_heatmap_steps_*.svg")):
        window = parse_window_from_name(svg_path.name)
        if not window:
            continue
        start, end = window
        rows = [
            row
            for row in tw_pair_rows
            if i(row, "window_start_step") == start and i(row, "window_end_step") == end
        ]
        if not rows:
            continue
        out = output_path(post_dir, out_dir, svg_path)
        plot_zone_pair_heatmap(plt, rows, out, label_order=label_order, vmin=vmin, vmax=vmax)
        written.append(out)

    write_manifest(written, out_dir)
    print(f"Wrote {len(written)} PNG files to {out_dir}")


if __name__ == "__main__":
    main()
