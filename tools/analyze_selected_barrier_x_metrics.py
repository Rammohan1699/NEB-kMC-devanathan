#!/usr/bin/env python3
"""Summarize selected KMC barriers versus x position."""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN = Path(
    "hyperion_results/runs/consolidated/"
    "devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p7215_full_from_step0"
    "__through_step1012564_partial_segment20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--x-mode", choices=("to", "from", "midpoint"), default="to")
    parser.add_argument("--x-length-a", type=float, default=255.44840574)
    parser.add_argument("--source-x-min-a", type=float, default=16.12518061159)
    parser.add_argument("--source-x-max-a", type=float, default=26.12518061159)
    parser.add_argument("--right-sink-x-min-a", type=float, default=204.19906933748)
    return parser.parse_args()


def finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def selected_x(row: dict[str, str], mode: str) -> float | None:
    from_x = finite_float(row.get("from_x_A", ""))
    to_x = finite_float(row.get("to_x_A", ""))
    if mode == "from":
        return from_x
    if mode == "to":
        return to_x
    if from_x is None or to_x is None:
        return None
    return 0.5 * (from_x + to_x)


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean_barrier_eV": float(arr.mean()),
        "median_barrier_eV": float(median(values)),
        "std_barrier_eV": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_barrier_eV": float(arr.min()),
        "max_barrier_eV": float(arr.max()),
    }


def read_points(path: Path, x_mode: str) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    seen_steps: set[int] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                step = int(row["step"])
            except Exception:
                continue
            if step in seen_steps:
                continue
            x = selected_x(row, x_mode)
            barrier = finite_float(row.get("barrier_eV", ""))
            if x is None or barrier is None:
                continue
            seen_steps.add(step)
            points.append((step, x, barrier))
    return points


def binned_summary(
    points: list[tuple[int, float, float]],
    *,
    x_length: float,
    bins: int,
    x_field: str,
) -> list[dict[str, object]]:
    edges = np.linspace(0.0, x_length, bins + 1)
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for _, x, barrier in points:
        if x < 0.0 or x > x_length:
            continue
        index = int(np.searchsorted(edges, x, side="right") - 1)
        index = min(max(index, 0), bins - 1)
        grouped[index].append((x, barrier))

    rows: list[dict[str, object]] = []
    for index in range(bins):
        members = grouped.get(index, [])
        if not members:
            continue
        xs = np.asarray([item[0] for item in members], dtype=float)
        barriers = [item[1] for item in members]
        row: dict[str, object] = {
            "x_bin_field": x_field,
            "x_bin_left_A": float(edges[index]),
            "x_bin_right_A": float(edges[index + 1]),
            "count": len(members),
            "mean_x_A": float(xs.mean()),
        }
        row.update(summarize(barriers))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "x_bin_field",
        "x_bin_left_A",
        "x_bin_right_A",
        "count",
        "mean_x_A",
        "mean_barrier_eV",
        "median_barrier_eV",
        "std_barrier_eV",
        "min_barrier_eV",
        "max_barrier_eV",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_csv(path: Path, rows: list[dict[str, object]], metric: str) -> None:
    fields = ["x_bin_left_A", "x_bin_right_A", "count", "mean_x_A", metric]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict[str, object]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    path_png: Path,
    path_svg: Path,
    source_min: float,
    source_max: float,
    sink_min: float,
) -> None:
    xs = np.asarray([float(row["mean_x_A"]) for row in rows], dtype=float)
    ys = np.asarray([float(row[metric]) for row in rows], dtype=float)
    counts = np.asarray([int(row["count"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=160)
    ax.axvspan(source_min, source_max, color="#f4c95d", alpha=0.28, label="source")
    ax.axvspan(sink_min, max(float(xs.max()) if xs.size else sink_min, sink_min), color="#8ecae6", alpha=0.18, label="right sink")
    ax.plot(xs, ys, color="#234b6d", linewidth=1.4)
    sizes = 24.0 + 96.0 * np.sqrt(counts / max(float(counts.max()), 1.0))
    scatter = ax.scatter(xs, ys, s=sizes, color="#2f7da8", edgecolor="white", linewidth=0.7, alpha=0.88)
    for x, y, count in zip(xs, ys, counts):
        if count > 0:
            scatter.set_label("_nolegend_")
    ax.set_xlabel("Selected-event x position (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_svg)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, object]], points: list[tuple[int, float, float]], x_mode: str) -> None:
    if not rows:
        text = "# Selected Barrier X Metrics\n\nNo selected barrier rows were found.\n"
    else:
        max_mean = max(rows, key=lambda row: float(row["mean_barrier_eV"]))
        max_max = max(rows, key=lambda row: float(row["max_barrier_eV"]))
        all_barriers = np.asarray([barrier for _, _, barrier in points], dtype=float)
        text = "\n".join(
            [
                "# Selected Barrier X Metrics",
                "",
                f"- Selected-event x mode: `{x_mode}`",
                f"- Selected events included: {len(points):,}",
                f"- Overall mean selected barrier: {float(all_barriers.mean()):.8f} eV",
                f"- Overall max selected barrier: {float(all_barriers.max()):.8f} eV",
                f"- Highest binned mean barrier: {float(max_mean['mean_barrier_eV']):.8f} eV at x = {float(max_mean['mean_x_A']):.8f} A",
                f"- Highest binned max barrier: {float(max_max['max_barrier_eV']):.8f} eV at x = {float(max_max['mean_x_A']):.8f} A",
                "",
            ]
        )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_events = args.run_dir / "diagnostics" / "kmc_selected_events.csv"
    if not selected_events.exists():
        raise FileNotFoundError(selected_events)

    out_dir = args.out_dir or args.run_dir / "analysis" / "selected_barrier_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    points = read_points(selected_events, args.x_mode)
    rows = binned_summary(
        points,
        x_length=args.x_length_a,
        bins=args.bins,
        x_field=f"{args.x_mode}_x_A",
    )
    write_csv(out_dir / "selected_barrier_vs_x_position.csv", rows)
    write_metric_csv(out_dir / "mean_selected_barrier_vs_x_position.csv", rows, "mean_barrier_eV")
    write_metric_csv(out_dir / "max_selected_barrier_vs_x_position.csv", rows, "max_barrier_eV")
    write_summary(out_dir / "SELECTED_BARRIER_SUMMARY.md", rows, points, args.x_mode)
    if rows:
        plot_metric(
            rows,
            metric="mean_barrier_eV",
            ylabel="Mean selected barrier (eV)",
            title="Mean selected barrier vs x",
            path_png=out_dir / "mean_selected_barrier_vs_x_position.png",
            path_svg=out_dir / "mean_selected_barrier_vs_x_position.svg",
            source_min=args.source_x_min_a,
            source_max=args.source_x_max_a,
            sink_min=args.right_sink_x_min_a,
        )
        plot_metric(
            rows,
            metric="max_barrier_eV",
            ylabel="Max selected barrier (eV)",
            title="Max selected barrier vs x",
            path_png=out_dir / "max_selected_barrier_vs_x_position.png",
            path_svg=out_dir / "max_selected_barrier_vs_x_position.svg",
            source_min=args.source_x_min_a,
            source_max=args.source_x_max_a,
            sink_min=args.right_sink_x_min_a,
        )
    print(f"results={out_dir}")
    print(f"selected_events={len(points)}")
    print(f"bins_with_events={len(rows)}")


if __name__ == "__main__":
    main()
