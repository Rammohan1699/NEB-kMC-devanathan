#!/usr/bin/env python3
"""Fit a global KMC attempt frequency by rescaling an MSD time axis.

For fixed barriers, changing the attempt frequency multiplies every event rate
by the same factor. The selected-event probabilities are unchanged, so an
existing KMC trajectory can be re-timed without rerunning NEB/KMC:

    t_new = t_old * nu_old / nu_new

This script fits ``nu_new`` against either a numeric MD MSD CSV or a target MD
MSD slope/point read from a reference plot.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


KB_EV_PER_K = 8.617333262145e-5


def _float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _find_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_to_original = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def load_msd_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        time_s_col = _find_column(fieldnames, ["time_s", "t_s", "time_seconds"])
        time_ps_col = _find_column(fieldnames, ["time_ps", "t_ps", "time"])
        msd_col = _find_column(fieldnames, ["msd_A2", "msd", "<d2>", "d2_A2"])
        if msd_col is None:
            raise SystemExit(f"Could not find an MSD column in {path}")
        if time_s_col is None and time_ps_col is None:
            raise SystemExit(f"Could not find a time column in {path}")

        times_ps: list[float] = []
        msd_a2: list[float] = []
        for row in reader:
            if time_s_col is not None:
                t_ps = _float(row.get(time_s_col)) * 1e12
            else:
                t_ps = _float(row.get(time_ps_col))
            msd = _float(row.get(msd_col))
            if math.isfinite(t_ps) and math.isfinite(msd):
                times_ps.append(t_ps)
                msd_a2.append(msd)

    if len(times_ps) < 2:
        raise SystemExit(f"Need at least two MSD rows in {path}")
    return np.asarray(times_ps, dtype=float), np.asarray(msd_a2, dtype=float)


def infer_attempt_frequency(run_dir: Path, temperature_k: float) -> float:
    selected = _first_existing(
        run_dir / "diagnostics" / "kmc_selected_events.csv",
        run_dir / "kmc_selected_events.csv",
    )
    if selected is None:
        raise SystemExit("Pass --attempt-frequency-hz or provide diagnostics/kmc_selected_events.csv")
    with selected.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            barrier = _float(row.get("barrier_eV"))
            rate = _float(row.get("rate_Hz"))
            if math.isfinite(barrier) and math.isfinite(rate) and rate > 0.0:
                return rate / math.exp(-barrier / (KB_EV_PER_K * temperature_k))
    raise SystemExit(f"Could not infer attempt frequency from {selected}")


def slope_fit(
    time_ps: np.ndarray,
    msd_a2: np.ndarray,
    min_ps: Optional[float],
    max_ps: Optional[float],
    through_origin: bool,
) -> tuple[float, float, int, float, float]:
    mask = np.ones_like(time_ps, dtype=bool)
    if min_ps is not None:
        mask &= time_ps >= min_ps
    if max_ps is not None:
        mask &= time_ps <= max_ps
    t = time_ps[mask]
    y = msd_a2[mask]
    if t.size < 2:
        raise SystemExit("Fit window contains fewer than two MSD points")

    if through_origin:
        denom = float(np.dot(t, t))
        slope = float(np.dot(t, y) / denom)
        intercept = 0.0
        pred = slope * t
    else:
        slope, intercept = np.polyfit(t, y, 1)
        slope = float(slope)
        intercept = float(intercept)
        pred = slope * t + intercept

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, int(t.size), float(t[0]), float(t[-1]), r2


def parse_md_point(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected TIME_PS,MSD_A2")
    return float(parts[0]), float(parts[1])


def write_rescaled_msd(path: Path, time_ps: np.ndarray, msd_a2: np.ndarray, scale: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "time_ps", "time_s", "msd_A2"])
        for step, (t_ps, msd) in enumerate(zip(time_ps * scale, msd_a2)):
            writer.writerow([step, f"{t_ps:.12e}", f"{t_ps * 1e-12:.12e}", f"{msd:.12e}"])


def write_scan(path: Path, nu_values: list[float], nu0: float, kmc_slope: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["attempt_frequency_Hz", "predicted_msd_slope_A2_per_ps"])
        for nu in nu_values:
            writer.writerow([f"{nu:.12e}", f"{kmc_slope * (nu / nu0):.12e}"])


def maybe_plot(
    out_png: Path,
    kmc_time_ps: np.ndarray,
    kmc_msd: np.ndarray,
    rescaled_time_ps: np.ndarray,
    md_time_ps: Optional[np.ndarray],
    md_msd: Optional[np.ndarray],
    md_slope: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=180)
    ax.plot(kmc_time_ps, kmc_msd, color="#1f77b4", lw=1.6, label="KMC original")
    ax.plot(rescaled_time_ps, kmc_msd, color="#333333", lw=1.8, label="KMC fitted nu")
    if md_time_ps is not None and md_msd is not None:
        ax.plot(md_time_ps, md_msd, color="#d62728", lw=1.8, label="MD")
    else:
        x = np.asarray([0.0, float(np.max(rescaled_time_ps))])
        ax.plot(x, md_slope * x, color="#d62728", lw=1.5, ls="--", label="MD target slope")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (A$^2$)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--msd-csv", type=Path, help="Defaults to RUN_DIR/msd_vs_time.csv")
    parser.add_argument("--md-msd-csv", type=Path, help="Optional MD curve CSV with time and MSD columns")
    parser.add_argument("--md-slope-a2-per-ps", type=float, help="Target MD MSD slope in A^2/ps")
    parser.add_argument("--md-point", type=parse_md_point, help="Target point as TIME_PS,MSD_A2")
    parser.add_argument("--attempt-frequency-hz", type=float, help="Attempt frequency used by the KMC run")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--fit-min-ps", type=float)
    parser.add_argument("--fit-max-ps", type=float)
    parser.add_argument("--free-intercept", action="store_true", help="Use y=m*x+b instead of forcing y=m*x")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    msd_csv = args.msd_csv or run_dir / "msd_vs_time.csv"
    out_dir = (args.out_dir or run_dir / "attempt_frequency_fit").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kmc_time_ps, kmc_msd = load_msd_csv(msd_csv)
    nu0 = args.attempt_frequency_hz or infer_attempt_frequency(run_dir, args.temperature_k)
    kmc_slope, intercept, nfit, fit_start, fit_stop, r2 = slope_fit(
        kmc_time_ps,
        kmc_msd,
        args.fit_min_ps,
        args.fit_max_ps,
        through_origin=not args.free_intercept,
    )

    md_time_ps = None
    md_msd = None
    if args.md_msd_csv is not None:
        md_time_ps, md_msd = load_msd_csv(args.md_msd_csv)
        md_slope, md_intercept, md_nfit, md_start, md_stop, md_r2 = slope_fit(
            md_time_ps,
            md_msd,
            args.fit_min_ps,
            args.fit_max_ps,
            through_origin=not args.free_intercept,
        )
        md_source = str(args.md_msd_csv)
    elif args.md_slope_a2_per_ps is not None:
        md_slope = args.md_slope_a2_per_ps
        md_intercept = 0.0
        md_nfit = 0
        md_start = 0.0
        md_stop = float("nan")
        md_r2 = float("nan")
        md_source = "manual slope"
    elif args.md_point is not None:
        t_ps, msd = args.md_point
        md_slope = msd / t_ps
        md_intercept = 0.0
        md_nfit = 1
        md_start = t_ps
        md_stop = t_ps
        md_r2 = float("nan")
        md_source = f"manual point {t_ps:g} ps,{msd:g} A^2"
    else:
        raise SystemExit("Provide --md-msd-csv, --md-slope-a2-per-ps, or --md-point")

    nu_fit = nu0 * (md_slope / kmc_slope)
    time_scale = nu0 / nu_fit
    rescaled_time_ps = kmc_time_ps * time_scale

    scan_values = sorted(
        set(
            [
                1e12,
                2e12,
                3e12,
                nu_fit,
                4e12,
                5e12,
                1e13,
                2e13,
            ]
        )
    )
    write_scan(out_dir / "attempt_frequency_scan.csv", scan_values, nu0, kmc_slope)
    write_rescaled_msd(out_dir / "msd_rescaled_best_fit.csv", kmc_time_ps, kmc_msd, time_scale)
    maybe_plot(
        out_dir / "attempt_frequency_fit.png",
        kmc_time_ps,
        kmc_msd,
        rescaled_time_ps,
        md_time_ps,
        md_msd,
        md_slope,
    )

    summary = [
        "Attempt-frequency MSD fit",
        "=========================",
        f"run_dir: {run_dir}",
        f"kmc_msd_csv: {msd_csv}",
        f"md_source: {md_source}",
        f"temperature_K_for_nu0_inference: {args.temperature_k:.6g}",
        f"nu0_Hz: {nu0:.12e}",
        f"kmc_fit_window_ps: {fit_start:.12e} to {fit_stop:.12e}",
        f"kmc_fit_points: {nfit}",
        f"kmc_slope_A2_per_ps_at_nu0: {kmc_slope:.12e}",
        f"kmc_intercept_A2: {intercept:.12e}",
        f"kmc_fit_R2: {r2:.12e}",
        f"md_slope_A2_per_ps: {md_slope:.12e}",
        f"md_intercept_A2: {md_intercept:.12e}",
        f"md_fit_points: {md_nfit}",
        f"md_fit_window_ps: {md_start:.12e} to {md_stop:.12e}",
        f"md_fit_R2: {md_r2:.12e}",
        f"best_attempt_frequency_Hz: {nu_fit:.12e}",
        f"time_rescale_factor_t_new_over_t_old: {time_scale:.12e}",
        f"original_end_time_ps: {kmc_time_ps[-1]:.12e}",
        f"rescaled_end_time_ps: {rescaled_time_ps[-1]:.12e}",
        f"end_msd_A2: {kmc_msd[-1]:.12e}",
        "",
        "Generated files:",
        f"- {out_dir / 'attempt_frequency_scan.csv'}",
        f"- {out_dir / 'msd_rescaled_best_fit.csv'}",
        f"- {out_dir / 'attempt_frequency_fit.png'}",
    ]
    (out_dir / "fit_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
