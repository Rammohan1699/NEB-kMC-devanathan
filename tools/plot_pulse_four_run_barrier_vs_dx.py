#!/usr/bin/env python3
"""Plot candidate barrier statistics versus x-displacement for four pulse runs."""
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
    parser.add_argument(
        "--dx-bin-a",
        type=float,
        default=0.001,
        help="Bin width for abs(dx_A), in Angstrom. Default preserves existing 0.001 A-style bins.",
    )
    parser.add_argument(
        "--min-count-for-plot",
        type=int,
        default=1,
        help="Minimum candidate count in a bin to draw that run/bin in plots. CSV always includes all bins.",
    )
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def bin_dx(dx: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("--dx-bin-a must be positive")
    return round(round(abs(float(dx)) / width) * width, 10)


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


def collect_run(run_dir: Path, *, dx_bin_a: float) -> dict[float, list[float]]:
    rates_path = run_dir / "diagnostics" / "rates_allranks.csv"
    if not rates_path.exists() or rates_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing rates_allranks.csv: {rates_path}")

    by_dx: dict[float, list[float]] = defaultdict(list)
    with rates_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            barrier = finite_float(row.get("barrier_eV"))
            dx = finite_float(row.get("dx_A"))
            if barrier is None or dx is None:
                continue
            by_dx[bin_dx(dx, dx_bin_a)].append(barrier)
    return by_dx


def summarize_runs(campaign: Path, *, dx_bin_a: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_id, label, subdir, _color in RUNS:
        run_dir = campaign / subdir
        cfg = read_env(run_dir / "run_config_snapshot.env")
        by_dx = collect_run(run_dir, dx_bin_a=dx_bin_a)
        for dx_value in sorted(by_dx):
            values = by_dx[dx_value]
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
                    "dx_abs_bin_A": dx_value,
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


def plot_metric(
    rows: list[dict[str, object]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    min_count: int,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=170)
    for run_id, label, _subdir, color in RUNS:
        run_rows = [
            row
            for row in rows
            if row["run_id"] == run_id and int(row["count"]) >= int(min_count)
        ]
        run_rows.sort(key=lambda row: float(row["dx_abs_bin_A"]))
        if not run_rows:
            continue
        x = [float(row["dx_abs_bin_A"]) for row in run_rows]
        y = [float(row[metric]) for row in run_rows]
        ax.plot(x, y, marker="o", markersize=2.2, linewidth=1.35, alpha=0.9, color=color, label=label)
    ax.set_xlabel("|x displacement|, |dx_A| (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_combined(rows: list[dict[str, object]], *, out_path: Path, min_count: int) -> None:
    if plt is None:
        return
    metrics = (
        ("barrier_mean_eV", "Mean barrier (eV)", "Mean"),
        ("barrier_median_eV", "Median barrier (eV)", "Median"),
        ("barrier_max_eV", "Max barrier (eV)", "Max"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 12.0), dpi=170, sharex=True)
    for ax, (metric, ylabel, panel_title) in zip(axes, metrics):
        for run_id, label, _subdir, color in RUNS:
            run_rows = [
                row
                for row in rows
                if row["run_id"] == run_id and int(row["count"]) >= int(min_count)
            ]
            run_rows.sort(key=lambda row: float(row["dx_abs_bin_A"]))
            if not run_rows:
                continue
            x = [float(row["dx_abs_bin_A"]) for row in run_rows]
            y = [float(row[metric]) for row in run_rows]
            ax.plot(x, y, marker="o", markersize=2.0, linewidth=1.25, alpha=0.9, color=color, label=label)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.75)
    axes[-1].set_xlabel("|x displacement|, |dx_A| (A)")
    axes[0].legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def write_markdown(path: Path, *, rows: list[dict[str, object]], dx_bin_a: float, min_count: int) -> None:
    lines = [
        "# Barrier vs X-Displacement",
        "",
        f"- Source: candidate rows from each run's `diagnostics/rates_allranks.csv`.",
        f"- X displacement is `abs(dx_A)` binned at `{dx_bin_a:g}` A.",
        f"- Plot minimum count per run/bin: `{min_count}`. The CSV includes all bins.",
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
            f"- `{label}`: {count:,} candidate barriers across {bins:,} dx bins; "
            f"engine `{first.get('neb_engine', '')}`, shell "
            f"`{first.get('shell_inner_radius_A', '')}/{first.get('shell_outer_radius_A', '')}` A, "
            f"native endpoint opt `{first.get('lammps_endpoint_optimize', '')}`."
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `barrier_mean_vs_abs_dx_four_runs.png` / `.svg`",
            "- `barrier_median_vs_abs_dx_four_runs.png` / `.svg`",
            "- `barrier_max_vs_abs_dx_four_runs.png` / `.svg`",
            "- `barrier_stats_vs_abs_dx_four_runs_combined.png` / `.svg`",
            "",
            "## Data",
            "",
            "- `barrier_stats_vs_abs_dx_four_runs.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    campaign = args.campaign
    out_dir = args.out_dir or campaign / "analysis" / "barrier_vs_dx_four_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = summarize_runs(campaign, dx_bin_a=float(args.dx_bin_a))
    write_csv(out_dir / "barrier_stats_vs_abs_dx_four_runs.csv", rows)

    plot_metric(
        rows,
        metric="barrier_mean_eV",
        ylabel="Mean barrier (eV)",
        title="Mean Candidate Barrier vs |x displacement|",
        out_path=out_dir / "barrier_mean_vs_abs_dx_four_runs",
        min_count=int(args.min_count_for_plot),
    )
    plot_metric(
        rows,
        metric="barrier_median_eV",
        ylabel="Median barrier (eV)",
        title="Median Candidate Barrier vs |x displacement|",
        out_path=out_dir / "barrier_median_vs_abs_dx_four_runs",
        min_count=int(args.min_count_for_plot),
    )
    plot_metric(
        rows,
        metric="barrier_max_eV",
        ylabel="Max barrier (eV)",
        title="Max Candidate Barrier vs |x displacement|",
        out_path=out_dir / "barrier_max_vs_abs_dx_four_runs",
        min_count=int(args.min_count_for_plot),
    )
    plot_combined(
        rows,
        out_path=out_dir / "barrier_stats_vs_abs_dx_four_runs_combined",
        min_count=int(args.min_count_for_plot),
    )
    write_markdown(out_dir / "BARRIER_VS_DX_FOUR_RUNS.md", rows=rows, dx_bin_a=float(args.dx_bin_a), min_count=int(args.min_count_for_plot))

    print(f"out_dir={out_dir}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
