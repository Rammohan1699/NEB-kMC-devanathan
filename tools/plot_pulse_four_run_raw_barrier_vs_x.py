#!/usr/bin/env python3
"""Plot raw candidate barriers versus x-position for the four pulse runs."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

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
    parser.add_argument("--point-size", type=float, default=0.18)
    parser.add_argument("--alpha", type=float, default=0.035)
    parser.add_argument("--y-max", type=float, default=2.0)
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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


def load_raw_points(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "diagnostics" / "rates_allranks.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing rates_allranks.csv: {path}")

    xs: list[float] = []
    barriers: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            from_x = finite_float(row.get("from_x_A"))
            to_x = finite_float(row.get("to_x_A"))
            barrier = finite_float(row.get("barrier_eV"))
            if from_x is None or to_x is None or barrier is None:
                continue
            xs.append(0.5 * (from_x + to_x))
            barriers.append(barrier)
    return np.asarray(xs, dtype=np.float32), np.asarray(barriers, dtype=np.float32)


def load_all(campaign: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for run_id, label, subdir, color in RUNS:
        run_dir = campaign / subdir
        cfg = read_env(run_dir / "run_config_snapshot.env")
        x, barrier = load_raw_points(run_dir)
        out.append(
            {
                "run_id": run_id,
                "label": label,
                "run_dir": run_dir,
                "color": color,
                "config": cfg,
                "x": x,
                "barrier": barrier,
            }
        )
    return out


def shade_gb(ax, central_left: float, central_right: float, gb_center: float) -> None:
    ax.axvspan(central_left, central_right, color="#f0b84f", alpha=0.18, linewidth=0)
    ax.axvline(gb_center, color="#7a3b00", linewidth=0.85, alpha=0.65)


def plot_panels(
    datasets: list[dict[str, object]],
    *,
    out_path: Path,
    central_left: float,
    central_right: float,
    gb_center: float,
    xlim: tuple[float, float],
    y_max: float,
    point_size: float,
    alpha: float,
    title: str,
) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), dpi=180, sharex=True, sharey=True)
    for ax, data in zip(axes.flat, datasets):
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["barrier"], dtype=np.float32)
        mask = (x >= xlim[0]) & (x <= xlim[1]) & np.isfinite(y)
        x_plot = x[mask]
        y_plot = y[mask]
        ax.scatter(
            x_plot,
            y_plot,
            s=point_size,
            alpha=alpha,
            c=str(data["color"]),
            edgecolors="none",
            rasterized=True,
        )
        shade_gb(ax, central_left, central_right, gb_center)
        cfg = data["config"]
        ax.set_title(
            f"{data['label']}\n"
            f"{len(x_plot):,} points, shell {cfg.get('SHELL_INNER_RADIUS_A', '')}/{cfg.get('SHELL_OUTER_RADIUS_A', '')} A"
        )
        ax.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.65)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.02, y_max)
    for ax in axes[:, 0]:
        ax.set_ylabel("Candidate barrier (eV)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Hop midpoint x-position (A)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_overlay(
    datasets: list[dict[str, object]],
    *,
    out_path: Path,
    central_left: float,
    central_right: float,
    gb_center: float,
    xlim: tuple[float, float],
    y_max: float,
    point_size: float,
    alpha: float,
    title: str,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=180)
    shade_gb(ax, central_left, central_right, gb_center)
    for data in datasets:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["barrier"], dtype=np.float32)
        mask = (x >= xlim[0]) & (x <= xlim[1]) & np.isfinite(y)
        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            alpha=alpha,
            c=str(data["color"]),
            edgecolors="none",
            rasterized=True,
            label=str(data["label"]),
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.02, y_max)
    ax.set_xlabel("Hop midpoint x-position (A)")
    ax.set_ylabel("Candidate barrier (eV)")
    ax.set_title(title)
    ax.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.65)
    ax.legend(frameon=False, markerscale=8)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def write_summary(path: Path, datasets: list[dict[str, object]], *, central_left: float, central_right: float, gb_center: float) -> None:
    lines = [
        "# Raw Barrier vs X-Position",
        "",
        "- These are raw candidate barrier points, not mean/median/max binned statistics.",
        "- X coordinate is hop midpoint: `(from_x_A + to_x_A) / 2`.",
        f"- Central GB window: `{central_left:.6f}` to `{central_right:.6f}` A; center `{gb_center:.6f}` A.",
        "",
        "## Runs",
        "",
    ]
    for data in datasets:
        cfg = data["config"]
        x = np.asarray(data["x"])
        y = np.asarray(data["barrier"])
        lines.append(
            f"- `{data['label']}`: {len(x):,} finite candidate barriers; "
            f"x range `{float(np.min(x)):.3f}` to `{float(np.max(x)):.3f}` A; "
            f"barrier max `{float(np.max(y)):.4g}` eV; "
            f"engine `{cfg.get('NEB_ENGINE', '')}`, shell "
            f"`{cfg.get('SHELL_INNER_RADIUS_A', '')}/{cfg.get('SHELL_OUTER_RADIUS_A', '')}` A, "
            f"native endpoint opt `{cfg.get('LAMMPS_ONLY_ENDPOINT_OPTIMIZE', '')}`."
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `raw_barrier_vs_x_four_runs_panels_full.png` / `.svg`",
            "- `raw_barrier_vs_x_four_runs_panels_gb_zoom.png` / `.svg`",
            "- `raw_barrier_vs_x_four_runs_overlay_gb_zoom.png` / `.svg`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.campaign / "analysis" / "raw_barrier_vs_x_four_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    central_left, central_right, gb_center, box_x = gb_window(args.site_map, args.site_regions)
    datasets = load_all(args.campaign)

    plot_panels(
        datasets,
        out_path=out_dir / "raw_barrier_vs_x_four_runs_panels_full",
        central_left=central_left,
        central_right=central_right,
        gb_center=gb_center,
        xlim=(0.0, box_x),
        y_max=float(args.y_max),
        point_size=float(args.point_size),
        alpha=float(args.alpha),
        title="Raw Candidate Barrier vs Hop Midpoint X",
    )
    plot_panels(
        datasets,
        out_path=out_dir / "raw_barrier_vs_x_four_runs_panels_gb_zoom",
        central_left=central_left,
        central_right=central_right,
        gb_center=gb_center,
        xlim=(central_left - 15.0, central_right + 15.0),
        y_max=float(args.y_max),
        point_size=max(float(args.point_size), 0.25),
        alpha=min(0.12, max(float(args.alpha), 0.055)),
        title="Raw Candidate Barrier vs Hop Midpoint X: Central GB Zoom",
    )
    plot_overlay(
        datasets,
        out_path=out_dir / "raw_barrier_vs_x_four_runs_overlay_gb_zoom",
        central_left=central_left,
        central_right=central_right,
        gb_center=gb_center,
        xlim=(central_left - 15.0, central_right + 15.0),
        y_max=float(args.y_max),
        point_size=max(float(args.point_size), 0.25),
        alpha=min(0.09, max(float(args.alpha), 0.045)),
        title="Raw Candidate Barrier vs Hop Midpoint X: Central GB Zoom",
    )
    write_summary(
        out_dir / "RAW_BARRIER_VS_X_FOUR_RUNS.md",
        datasets,
        central_left=central_left,
        central_right=central_right,
        gb_center=gb_center,
    )
    print(f"out_dir={out_dir}")
    for data in datasets:
        print(f"{data['run_id']} points={len(data['x'])}")


if __name__ == "__main__":
    main()
