#!/usr/bin/env python3
"""Plot candidate barrier statistics versus x-position for the four pulse runs."""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional.
    plt = None


DEFAULT_CAMPAIGN = Path(
    "practice-version/devanathan-kmc-base/runs/"
    "external_pulse50_barrier_pair_20260706_143812"
)


RUNS = (
    ("lammps_original_8_10", "LAMMPS original 8/10 A", "lammps_only", "#4c78a8"),
    ("lammps_shell_5_6", "LAMMPS 5/6 A", "lammps_only_ase_shell", "#f58518"),
    (
        "lammps_shell_5_6_endpoint_opt",
        "LAMMPS 5/6 A + endpoint opt",
        "lammps_only_lmp_endpoint_opt_ase_shell",
        "#54a24b",
    ),
    ("ase_neb_5_6", "ASE-NEB 5/6 A", "ase_neb", "#e45756"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--site-map", type=Path, default=Path("kmc_map_inputs/sigma5_stage3_unified_sites.npz"))
    parser.add_argument("--site-regions", type=Path, default=Path("kmc_map_inputs/sigma5_site_regions.npz"))
    parser.add_argument("--x-bin-a", type=float, default=1.0, help="Midpoint-x bin width in Angstrom.")
    parser.add_argument("--min-count-for-plot", type=int, default=20)
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def x_bin_center(mid_x: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("--x-bin-a must be positive")
    return round((math.floor(float(mid_x) / width) + 0.5) * width, 10)


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def gb_window(site_map: Path, site_regions: Path) -> tuple[float, float, float, float]:
    smap = np.load(site_map, allow_pickle=True)
    sreg = np.load(site_regions, allow_pickle=True)
    positions = np.asarray(smap["positions"], dtype=float)
    labels = sreg["site_selection_label"]
    box_x = float(np.asarray(smap["box_lengths"], dtype=float)[0])
    left = float(positions[labels == "bulk_grain_0", 0].max())
    right = float(positions[labels == "bulk_grain_1", 0].min())
    return left, right, 0.5 * box_x, box_x


def collect_run(run_dir: Path, *, x_bin_a: float) -> dict[float, list[float]]:
    path = run_dir / "diagnostics" / "rates_allranks.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing rates_allranks.csv: {path}")

    by_x: dict[float, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            from_x = finite_float(row.get("from_x_A"))
            to_x = finite_float(row.get("to_x_A"))
            barrier = finite_float(row.get("barrier_eV"))
            if from_x is None or to_x is None or barrier is None:
                continue
            mid_x = 0.5 * (from_x + to_x)
            by_x[x_bin_center(mid_x, x_bin_a)].append(barrier)
    return by_x


def summarize_runs(campaign: Path, *, x_bin_a: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_id, label, subdir, _color in RUNS:
        run_dir = campaign / subdir
        cfg = read_env(run_dir / "run_config_snapshot.env")
        by_x = collect_run(run_dir, x_bin_a=x_bin_a)
        for x_value in sorted(by_x):
            values = by_x[x_value]
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "run_dir": str(run_dir),
                    "neb_engine": cfg.get("NEB_ENGINE", ""),
                    "shell_inner_radius_A": cfg.get("SHELL_INNER_RADIUS_A", ""),
                    "shell_outer_radius_A": cfg.get("SHELL_OUTER_RADIUS_A", ""),
                    "lammps_endpoint_optimize": cfg.get("LAMMPS_ONLY_ENDPOINT_OPTIMIZE", ""),
                    "x_mid_bin_center_A": x_value,
                    "x_bin_width_A": float(x_bin_a),
                    "count": int(arr.size),
                    "barrier_mean_eV": float(arr.mean()),
                    "barrier_median_eV": float(median(values)),
                    "barrier_max_eV": float(arr.max()),
                    "barrier_min_eV": float(arr.min()),
                    "barrier_p90_eV": float(np.quantile(arr, 0.90)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shade_gb(ax, central_left: float, central_right: float, gb_center: float) -> None:
    ax.axvspan(central_left, central_right, color="#f0b84f", alpha=0.18, linewidth=0)
    ax.axvline(gb_center, color="#7a3b00", linewidth=0.9, alpha=0.7)


def filtered_run_rows(
    rows: list[dict[str, object]],
    *,
    run_id: str,
    min_count: int,
    xlim: tuple[float, float] | None,
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        x = float(row["x_mid_bin_center_A"])
        if row["run_id"] != run_id or int(row["count"]) < int(min_count):
            continue
        if xlim is not None and not (xlim[0] <= x <= xlim[1]):
            continue
        out.append(row)
    out.sort(key=lambda row: float(row["x_mid_bin_center_A"]))
    return out


def plot_metric(
    rows: list[dict[str, object]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    min_count: int,
    central_left: float,
    central_right: float,
    gb_center: float,
    xlim: tuple[float, float] | None = None,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(11.2, 5.8), dpi=170)
    shade_gb(ax, central_left, central_right, gb_center)
    for run_id, label, _subdir, color in RUNS:
        run_rows = filtered_run_rows(rows, run_id=run_id, min_count=min_count, xlim=xlim)
        if not run_rows:
            continue
        x = [float(row["x_mid_bin_center_A"]) for row in run_rows]
        y = [float(row[metric]) for row in run_rows]
        ax.plot(x, y, marker="o", markersize=2.4, linewidth=1.35, alpha=0.92, color=color, label=label)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Hop midpoint x-position (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_combined(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    min_count: int,
    central_left: float,
    central_right: float,
    gb_center: float,
    xlim: tuple[float, float] | None = None,
    title: str,
) -> None:
    if plt is None:
        return
    metrics = (
        ("barrier_mean_eV", "Mean barrier (eV)", "Mean"),
        ("barrier_median_eV", "Median barrier (eV)", "Median"),
        ("barrier_max_eV", "Max barrier (eV)", "Max"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 12.0), dpi=170, sharex=True)
    for ax, (metric, ylabel, panel_title) in zip(axes, metrics):
        shade_gb(ax, central_left, central_right, gb_center)
        for run_id, label, _subdir, color in RUNS:
            run_rows = filtered_run_rows(rows, run_id=run_id, min_count=min_count, xlim=xlim)
            if not run_rows:
                continue
            x = [float(row["x_mid_bin_center_A"]) for row in run_rows]
            y = [float(row[metric]) for row in run_rows]
            ax.plot(x, y, marker="o", markersize=2.2, linewidth=1.25, alpha=0.92, color=color, label=label)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.75)
    axes[-1].set_xlabel("Hop midpoint x-position (A)")
    axes[0].legend(frameon=False, ncols=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def write_markdown(
    path: Path,
    *,
    rows: list[dict[str, object]],
    x_bin_a: float,
    min_count: int,
    central_left: float,
    central_right: float,
    gb_center: float,
) -> None:
    lines = [
        "# Barrier Statistics vs X-Position",
        "",
        "- X coordinate is hop midpoint: `(from_x_A + to_x_A) / 2`.",
        f"- Midpoint x is binned at `{x_bin_a:g}` A.",
        f"- Plot minimum count per run/bin: `{min_count}`. The CSV includes all bins.",
        f"- Central GB window: `{central_left:.6f}` to `{central_right:.6f}` A; center `{gb_center:.6f}` A.",
        "",
        "## Runs",
        "",
    ]
    for run_id, label, _subdir, _color in RUNS:
        run_rows = [row for row in rows if row["run_id"] == run_id]
        count = sum(int(row["count"]) for row in run_rows)
        bins = len(run_rows)
        first = run_rows[0] if run_rows else {}
        lines.append(
            f"- `{label}`: {count:,} candidate barriers across {bins:,} x bins; "
            f"engine `{first.get('neb_engine', '')}`, shell "
            f"`{first.get('shell_inner_radius_A', '')}/{first.get('shell_outer_radius_A', '')}` A, "
            f"native endpoint opt `{first.get('lammps_endpoint_optimize', '')}`."
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `barrier_stats_vs_mid_x_four_runs_combined_full.png` / `.svg`",
            "- `barrier_stats_vs_mid_x_four_runs_combined_gb_zoom.png` / `.svg`",
            "- `barrier_mean_vs_mid_x_four_runs_full.png` / `.svg`",
            "- `barrier_median_vs_mid_x_four_runs_full.png` / `.svg`",
            "- `barrier_max_vs_mid_x_four_runs_full.png` / `.svg`",
            "- `barrier_mean_vs_mid_x_four_runs_gb_zoom.png` / `.svg`",
            "- `barrier_median_vs_mid_x_four_runs_gb_zoom.png` / `.svg`",
            "- `barrier_max_vs_mid_x_four_runs_gb_zoom.png` / `.svg`",
            "",
            "## Data",
            "",
            "- `barrier_stats_vs_mid_x_four_runs.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.campaign / "analysis" / "barrier_stats_vs_x_four_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    central_left, central_right, gb_center, _box_x = gb_window(args.site_map, args.site_regions)
    rows = summarize_runs(args.campaign, x_bin_a=float(args.x_bin_a))
    write_csv(out_dir / "barrier_stats_vs_mid_x_four_runs.csv", rows)

    zoom_xlim = (central_left - 15.0, central_right + 15.0)
    metrics = (
        ("barrier_mean_eV", "Mean barrier (eV)", "Mean Candidate Barrier vs Hop Midpoint X", "barrier_mean_vs_mid_x_four_runs"),
        ("barrier_median_eV", "Median barrier (eV)", "Median Candidate Barrier vs Hop Midpoint X", "barrier_median_vs_mid_x_four_runs"),
        ("barrier_max_eV", "Max barrier (eV)", "Max Candidate Barrier vs Hop Midpoint X", "barrier_max_vs_mid_x_four_runs"),
    )
    for metric, ylabel, title, stem in metrics:
        plot_metric(
            rows,
            metric=metric,
            ylabel=ylabel,
            title=f"{title}: Full Cell",
            out_path=out_dir / f"{stem}_full",
            min_count=int(args.min_count_for_plot),
            central_left=central_left,
            central_right=central_right,
            gb_center=gb_center,
        )
        plot_metric(
            rows,
            metric=metric,
            ylabel=ylabel,
            title=f"{title}: Central GB Zoom",
            out_path=out_dir / f"{stem}_gb_zoom",
            min_count=int(args.min_count_for_plot),
            central_left=central_left,
            central_right=central_right,
            gb_center=gb_center,
            xlim=zoom_xlim,
        )

    plot_combined(
        rows,
        out_path=out_dir / "barrier_stats_vs_mid_x_four_runs_combined_full",
        min_count=int(args.min_count_for_plot),
        central_left=central_left,
        central_right=central_right,
        gb_center=gb_center,
        title="Candidate Barrier Statistics vs Hop Midpoint X: Full Cell",
    )
    plot_combined(
        rows,
        out_path=out_dir / "barrier_stats_vs_mid_x_four_runs_combined_gb_zoom",
        min_count=int(args.min_count_for_plot),
        central_left=central_left,
        central_right=central_right,
        gb_center=gb_center,
        xlim=zoom_xlim,
        title="Candidate Barrier Statistics vs Hop Midpoint X: Central GB Zoom",
    )
    write_markdown(
        out_dir / "BARRIER_STATS_VS_X_FOUR_RUNS.md",
        rows=rows,
        x_bin_a=float(args.x_bin_a),
        min_count=int(args.min_count_for_plot),
        central_left=central_left,
        central_right=central_right,
        gb_center=gb_center,
    )
    print(f"out_dir={out_dir}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
