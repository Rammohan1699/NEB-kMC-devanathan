#!/usr/bin/env python3
"""Fit cumulative sink detections and estimate concentration-driven permeability."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sink-permeability")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-sink-permeability")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np


DEFAULT_AREA_A2 = 57.202 * 57.202
DEFAULT_SOURCE_X_MIN_A = 0.0
DEFAULT_SOURCE_X_MAX_A = 10.0
DEFAULT_SINK_X_A = 205.9272
DEFAULT_SOURCE_H = 28.0
DPI = 260
SINGLE_SIZE = (8.7, 6.1)
BLACK = "#4a4a4a"
BLUE = "#2a6fdb"


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 2.6,
            "axes.labelsize": 34,
            "axes.titlesize": 28,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 25,
            "lines.linewidth": 3.4,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2.6)
    ax.tick_params(
        which="major",
        direction="in",
        length=10,
        width=2.4,
        top=True,
        right=True,
        colors="black",
        pad=8,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        length=5,
        width=2.0,
        top=True,
        right=True,
        colors="black",
    )
    try:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    except ValueError:
        pass
    try:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    except ValueError:
        pass


def latex_sci(value: float, precision: int = 3) -> str:
    mantissa, exponent = f"{value:.{precision}e}".split("e")
    return rf"{float(mantissa):g}\times10^{{{int(exponent)}}}"


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_final_time_ns(events_csv: Path, fallback: float) -> float:
    study = events_csv.parent / "sink_deletion_study.md"
    if study.is_file():
        for line in study.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("- final_KMC_time_ns:"):
                final_time = finite_float(line.split(":", 1)[1].strip())
                if final_time is not None:
                    return final_time

    candidates = [
        events_csv.parent / "sink_deletions_by_time_bin.csv",
        events_csv.parent / "sink_detections_by_time_bin.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        final_time = fallback
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                end_time = finite_float(row.get("end_time_ns"))
                if end_time is not None:
                    final_time = max(final_time, end_time)
        return final_time
    return fallback


def read_sink_events(events_csv: Path) -> list[dict[str, float | int | str]]:
    events: list[dict[str, float | int | str]] = []
    with events_csv.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            time_ns = finite_float(row.get("time_ns"))
            if time_ns is None:
                time_s = finite_float(row.get("time_s"))
                if time_s is None:
                    continue
                time_ns = time_s * 1.0e9
            event_index = int(finite_float(row.get("event_index")) or index)
            cumulative = int(
                finite_float(row.get("cumulative_H_deleted_at_sink"))
                or finite_float(row.get("cumulative_detected_H"))
                or event_index
            )
            events.append(
                {
                    "event_index": event_index,
                    "step": row.get("step", ""),
                    "time_ns": time_ns,
                    "time_s": time_ns * 1.0e-9,
                    "cumulative_H_detected_at_sink": cumulative,
                }
            )
    events.sort(key=lambda row: (float(row["time_ns"]), int(row["event_index"])))
    for cumulative, row in enumerate(events, start=1):
        row["cumulative_H_detected_at_sink"] = cumulative
    return events


def build_events_from_diagnostics(diagnostics_csv: Path) -> list[dict[str, float | int | str]]:
    events: list[dict[str, float | int | str]] = []
    with diagnostics_csv.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.DictReader(handle):
            removed = int(finite_float(row.get("removed_this_step")) or 0)
            if removed <= 0:
                continue
            sites = (row.get("deleted_sink_sites") or "").split()
            time_s = float(row["time_s"])
            for local_index in range(removed):
                events.append(
                    {
                        "event_index": len(events) + 1,
                        "step": row.get("step", ""),
                        "time_ns": time_s * 1.0e9,
                        "time_s": time_s,
                        "cumulative_H_detected_at_sink": len(events) + 1,
                        "deleted_sink_site": sites[local_index] if local_index < len(sites) else "",
                    }
                )
    return events


def linear_fit(
    events: list[dict[str, float | int | str]],
    fit_start_ns: float | None,
    fit_end_ns: float | None,
) -> dict[str, float | int]:
    selected = [
        row
        for row in events
        if (fit_start_ns is None or float(row["time_ns"]) >= fit_start_ns)
        and (fit_end_ns is None or float(row["time_ns"]) <= fit_end_ns)
    ]
    if len(selected) < 2:
        raise ValueError("At least two sink detections are required in the fit window")

    x_s = np.asarray([float(row["time_s"]) for row in selected], dtype=float)
    y = np.asarray([float(row["cumulative_H_detected_at_sink"]) for row in selected], dtype=float)
    slope_h_per_s, intercept_h = np.polyfit(x_s, y, 1)
    fitted = slope_h_per_s * x_s + intercept_h
    residuals = y - fitted
    ss_res = float(np.sum(residuals * residuals))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    dof = len(selected) - 2
    sxx = float(np.sum((x_s - float(np.mean(x_s))) ** 2))
    if dof > 0 and sxx > 0.0:
        residual_variance = ss_res / dof
        slope_se = math.sqrt(residual_variance / sxx)
        intercept_se = math.sqrt(residual_variance * (1.0 / len(selected) + float(np.mean(x_s)) ** 2 / sxx))
    else:
        residual_variance = math.nan
        slope_se = math.nan
        intercept_se = math.nan

    return {
        "fit_points": len(selected),
        "fit_start_ns": float(np.min(x_s) * 1.0e9),
        "fit_end_ns": float(np.max(x_s) * 1.0e9),
        "slope_H_per_s": float(slope_h_per_s),
        "slope_H_per_ns": float(slope_h_per_s * 1.0e-9),
        "intercept_H": float(intercept_h),
        "slope_standard_error_H_per_s": float(slope_se),
        "intercept_standard_error_H": float(intercept_se),
        "residual_variance_H2": float(residual_variance),
        "r_squared": float(r_squared),
    }


def permeability_from_slope(
    fit: dict[str, float | int],
    *,
    area_a2: float,
    source_h: float,
    source_x_min_a: float,
    source_x_max_a: float,
    sink_x_a: float,
    sink_concentration_h_per_a3: float,
    final_time_ns: float,
    total_events: int,
) -> dict[str, float | int]:
    source_width_a = source_x_max_a - source_x_min_a
    if area_a2 <= 0.0:
        raise ValueError("Cross-sectional area must be positive")
    if source_width_a <= 0.0:
        raise ValueError("Source width must be positive")
    transport_length_a = sink_x_a - source_x_max_a
    if transport_length_a <= 0.0:
        raise ValueError("Sink x must be larger than source_x_max_a")

    source_concentration = source_h / (area_a2 * source_width_a)
    delta_concentration = source_concentration - sink_concentration_h_per_a3
    if delta_concentration <= 0.0:
        raise ValueError("Source concentration must exceed sink concentration")

    slope_h_per_s = float(fit["slope_H_per_s"])
    slope_flux_h_per_a2_s = slope_h_per_s / area_a2
    slope_flux_h_per_m2_s = slope_flux_h_per_a2_s * 1.0e20
    permeance_a_per_s = slope_flux_h_per_a2_s / delta_concentration
    permeability_a2_per_s = slope_flux_h_per_a2_s * transport_length_a / delta_concentration
    final_time_s = final_time_ns * 1.0e-9
    full_run_rate_h_per_s = total_events / final_time_s if final_time_s > 0.0 else math.nan
    full_run_flux_h_per_a2_s = full_run_rate_h_per_s / area_a2 if final_time_s > 0.0 else math.nan
    full_run_permeability_a2_per_s = (
        full_run_flux_h_per_a2_s * transport_length_a / delta_concentration
        if final_time_s > 0.0
        else math.nan
    )

    return {
        "total_sink_events": total_events,
        "final_time_ns": final_time_ns,
        "cross_section_area_A2": area_a2,
        "source_H": source_h,
        "source_x_min_A": source_x_min_a,
        "source_x_max_A": source_x_max_a,
        "source_width_A": source_width_a,
        "right_sink_x_A": sink_x_a,
        "transport_length_A": transport_length_a,
        "source_concentration_H_per_A3": source_concentration,
        "sink_concentration_H_per_A3": sink_concentration_h_per_a3,
        "delta_concentration_H_per_A3": delta_concentration,
        "slope_flux_H_per_A2_s": slope_flux_h_per_a2_s,
        "slope_flux_H_per_m2_s": slope_flux_h_per_m2_s,
        "slope_permeance_A_per_s": permeance_a_per_s,
        "slope_permeance_m_per_s": permeance_a_per_s * 1.0e-10,
        "slope_effective_permeability_A2_per_s": permeability_a2_per_s,
        "slope_effective_permeability_m2_per_s": permeability_a2_per_s * 1.0e-20,
        "full_run_rate_H_per_s": full_run_rate_h_per_s,
        "full_run_flux_H_per_A2_s": full_run_flux_h_per_a2_s,
        "full_run_effective_permeability_A2_per_s": full_run_permeability_a2_per_s,
        "full_run_effective_permeability_m2_per_s": full_run_permeability_a2_per_s * 1.0e-20,
    }


def plot_fit(
    output: Path,
    events: list[dict[str, float | int | str]],
    fit: dict[str, float | int],
    permeability: dict[str, float | int],
) -> None:
    configure_publication_style()
    event_time_ns = np.asarray([float(row["time_ns"]) for row in events], dtype=float)
    cumulative = np.asarray([int(row["cumulative_H_detected_at_sink"]) for row in events], dtype=int)
    final_time_ns = float(permeability["final_time_ns"])
    fit_x_ns = np.linspace(float(fit["fit_start_ns"]), float(fit["fit_end_ns"]), 200)
    fit_y = float(fit["slope_H_per_s"]) * fit_x_ns * 1.0e-9 + float(fit["intercept_H"])

    fig, ax = plt.subplots(figsize=SINGLE_SIZE)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.96)
    ax.step(
        np.r_[0.0, event_time_ns, final_time_ns],
        np.r_[0, cumulative, cumulative[-1] if cumulative.size else 0],
        where="post",
        color=BLUE,
        label="Sink detections",
    )
    ax.plot(
        event_time_ns,
        cumulative,
        linestyle="none",
        marker="o",
        markersize=6.2,
        color=BLUE,
    )
    ax.plot(fit_x_ns, fit_y, color=BLACK, linewidth=3.0, label="Linear fit")
    ax.set_xlabel("KMC time (ns)")
    ax.set_ylabel("H detected at sink")
    ax.set_xlim(0.0, max(3.5, final_time_ns))
    ax.set_ylim(0.0, max(1.0, float(cumulative.max()) if cumulative.size else 1.0) * 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    style_axis(ax)
    slope_ns = float(fit["slope_H_per_ns"])
    intercept = float(fit["intercept_H"])
    flux = float(permeability["slope_flux_H_per_A2_s"])
    p_eff = float(permeability["slope_effective_permeability_m2_per_s"])
    annotation = "\n".join(
        [
            rf"$N(t_{{\mathrm{{ns}}}})={slope_ns:.3f}t_{{\mathrm{{ns}}}}{intercept:+.3f}$",
            rf"$J=m/A={latex_sci(flux)}\ \mathrm{{H\,\AA^{{-2}}\,s^{{-1}}}}$",
            rf"$P_{{\mathrm{{eff}}}}=J L/\Delta C={latex_sci(p_eff)}\ \mathrm{{m^2\,s^{{-1}}}}$",
        ]
    )
    ax.text(
        0.055,
        0.93,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
    )
    ax.legend(frameon=False, handlelength=2.6, borderpad=0.2, labelspacing=0.6, loc="lower right")
    fig.savefig(output / "cumulative_sink_linear_fit_permeability.png")
    fig.savefig(output / "cumulative_sink_linear_fit_permeability.pdf")
    plt.close(fig)


def write_markdown(
    output: Path,
    events_csv: Path,
    fit: dict[str, float | int],
    permeability: dict[str, float | int],
) -> None:
    lines = [
        "# Sink Linear Fit And Permeability Analysis",
        "",
        f"- Sink event source: `{events_csv}`",
        f"- Total sink detections: {int(permeability['total_sink_events']):,}",
        f"- Final KMC time: {float(permeability['final_time_ns']):.9f} ns",
        f"- Fit window: {float(fit['fit_start_ns']):.9f} to {float(fit['fit_end_ns']):.9f} ns ({int(fit['fit_points'])} event points)",
        f"- Linear slope: {float(fit['slope_H_per_ns']):.8f} H/ns = {float(fit['slope_H_per_s']):.8e} H/s",
        f"- Fit intercept: {float(fit['intercept_H']):.8f} H",
        f"- R^2: {float(fit['r_squared']):.8f}",
        f"- Slope-derived flux: {float(permeability['slope_flux_H_per_A2_s']):.8e} H A^-2 s^-1 = {float(permeability['slope_flux_H_per_m2_s']):.8e} H m^-2 s^-1",
        f"- Source concentration: {float(permeability['source_concentration_H_per_A3']):.8e} H A^-3",
        f"- Sink concentration assumed: {float(permeability['sink_concentration_H_per_A3']):.8e} H A^-3",
        f"- Transport length: {float(permeability['transport_length_A']):.8f} A",
        f"- Slope-derived permeance: {float(permeability['slope_permeance_m_per_s']):.8e} m/s",
        f"- Slope-derived effective permeability: {float(permeability['slope_effective_permeability_m2_per_s']):.8e} m^2/s",
        f"- Full-run count/time effective permeability, for comparison: {float(permeability['full_run_effective_permeability_m2_per_s']):.8e} m^2/s",
        "",
        "## Mathematical Formulation",
        "",
        "Let `N(t)` be the cumulative number of H atoms detected and deleted at the absorbing sink. The analysis fits the detected event points to",
        "",
        "```text",
        "N(t) = m t + b",
        "```",
        "",
        "where `t` is KMC time in seconds, `m = dN/dt` is the transmitted-H rate in H/s, and `b` is the fitted intercept. The default fit uses event points from the first through the last sink detection, so it estimates the post-breakthrough arrival rate rather than the initial no-arrival transient.",
        "",
        "The sink flux is the fitted arrival rate divided by the cross-sectional area:",
        "",
        "```text",
        "J = m / A",
        "```",
        "",
        "with units H A^-2 s^-1 when `A` is in A^2. The source-side concentration is estimated from the maintained source inventory and source slab volume:",
        "",
        "```text",
        "C_source = N_source / (A W_source)",
        "Delta C = C_source - C_sink",
        "```",
        "",
        "where `W_source = x_source,max - x_source,min`. The absorbing sink is taken as `C_sink = 0` unless another value is supplied.",
        "",
        "For a concentration-driven Devanathan boundary, Fick's law gives",
        "",
        "```text",
        "J = P_eff Delta C / L",
        "P_eff = J L / Delta C",
        "```",
        "",
        "where `L = x_sink - x_source,max`. `P_eff` has units of A^2/s, converted here to m^2/s by multiplying by 1e-20. This is an effective concentration-normalized permeability/transport coefficient. It is not a pressure-normalized Sieverts-law permeability because the simulation output does not provide gas pressure or solubility.",
        "",
        "## Output Files",
        "",
        "- `sink_linear_fit_permeability.csv`",
        "- `cumulative_sink_linear_fit_permeability.png`",
        "- `cumulative_sink_linear_fit_permeability.pdf`",
        "- `SINK_LINEAR_PERMEABILITY_ANALYSIS.md`",
    ]
    (output / "SINK_LINEAR_PERMEABILITY_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--events-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--area-a2", type=float, default=DEFAULT_AREA_A2)
    parser.add_argument("--source-h", type=float, default=DEFAULT_SOURCE_H)
    parser.add_argument("--source-x-min-a", type=float, default=DEFAULT_SOURCE_X_MIN_A)
    parser.add_argument("--source-x-max-a", type=float, default=DEFAULT_SOURCE_X_MAX_A)
    parser.add_argument("--sink-x-a", type=float, default=DEFAULT_SINK_X_A)
    parser.add_argument("--sink-concentration-h-per-a3", type=float, default=0.0)
    parser.add_argument("--fit-start-ns", type=float, default=None)
    parser.add_argument("--fit-end-ns", type=float, default=None)
    parser.add_argument("--final-time-ns", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.events_csv is None and args.run_dir is None:
        raise SystemExit("Provide --events-csv or --run-dir")

    run_dir = args.run_dir.resolve() if args.run_dir else None
    if args.events_csv is not None:
        events_csv = args.events_csv.resolve()
    elif run_dir is not None:
        events_csv = run_dir / "analysis" / "sink_deletions" / "sink_deletion_events.csv"
    else:
        raise AssertionError("unreachable")

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    elif events_csv.is_file():
        out_dir = events_csv.parent
    elif run_dir is not None:
        out_dir = run_dir / "analysis" / "sink_deletions"
    else:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    if events_csv.is_file():
        events = read_sink_events(events_csv)
    elif run_dir is not None:
        diagnostics = run_dir / "diagnostics" / "kmc_devanathan.csv"
        events = build_events_from_diagnostics(diagnostics)
        events_csv = out_dir / "sink_deletion_events.csv"
        write_csv(
            events_csv,
            ["event_index", "step", "time_s", "time_ns", "cumulative_H_detected_at_sink", "deleted_sink_site"],
            events,
        )
    else:
        raise FileNotFoundError(events_csv)
    if len(events) < 2:
        raise ValueError("At least two sink detections are required for a linear fit")

    final_time_ns = args.final_time_ns
    if final_time_ns is None:
        final_time_ns = infer_final_time_ns(events_csv, max(float(row["time_ns"]) for row in events))
    fit = linear_fit(events, args.fit_start_ns, args.fit_end_ns)
    permeability = permeability_from_slope(
        fit,
        area_a2=args.area_a2,
        source_h=args.source_h,
        source_x_min_a=args.source_x_min_a,
        source_x_max_a=args.source_x_max_a,
        sink_x_a=args.sink_x_a,
        sink_concentration_h_per_a3=args.sink_concentration_h_per_a3,
        final_time_ns=final_time_ns,
        total_events=len(events),
    )
    result = {**fit, **permeability}
    fields = list(result)
    write_csv(out_dir / "sink_linear_fit_permeability.csv", fields, [result])
    plot_fit(out_dir, events, fit, permeability)
    write_markdown(out_dir, events_csv, fit, permeability)

    print(f"results={out_dir}")
    print(f"slope={float(fit['slope_H_per_ns']):.8f} H/ns")
    print(f"flux={float(permeability['slope_flux_H_per_A2_s']):.8e} H A^-2 s^-1")
    print(f"effective_permeability={float(permeability['slope_effective_permeability_m2_per_s']):.8e} m^2/s")


if __name__ == "__main__":
    main()
