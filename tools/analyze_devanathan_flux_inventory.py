#!/usr/bin/env python3
"""Analyze Devanathan/GCMC flux and active-H inventory for a consolidated run."""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-devanathan-flux")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-devanathan-flux")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_lammps_frame(
    handle: TextIO,
) -> tuple[int, np.ndarray, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] | None:
    first = handle.readline()
    if not first:
        return None
    if first.rstrip("\r\n") != "ITEM: TIMESTEP":
        raise ValueError(f"Expected TIMESTEP header, got {first!r}")
    step = int(handle.readline().strip())
    if handle.readline().rstrip("\r\n") != "ITEM: NUMBER OF ATOMS":
        raise ValueError(f"Malformed atom count at step {step}")
    count = int(handle.readline().strip())
    box_header = handle.readline()
    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"Malformed box header at step {step}")
    bounds: list[tuple[float, float]] = []
    for _ in range(3):
        lo, hi, *_ = handle.readline().split()
        bounds.append((float(lo), float(hi)))
    atom_header = handle.readline().strip().split()
    if atom_header[:2] != ["ITEM:", "ATOMS"]:
        raise ValueError(f"Malformed atom header at step {step}")
    columns = atom_header[2:]
    x_index = columns.index("x")
    positions = np.empty(count, dtype=float)
    for index in range(count):
        values = handle.readline().split()
        positions[index] = float(values[x_index])
    return step, positions, (bounds[0], bounds[1], bounds[2])


def load_devanathan_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Devanathan diagnostics: {path}")
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def integer(row: dict[str, str], name: str) -> int:
    value = row.get(name, "")
    return int(value) if value not in {"", None} else 0


def floating(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in {"", None} else 0.0


def analyze_active_h(rows: list[dict[str, str]]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    series = {
        "step": np.asarray([integer(row, "step") for row in rows], dtype=np.int64),
        "time_s": np.asarray([floating(row, "time_s") for row in rows], dtype=float),
        "source_occupied": np.asarray([integer(row, "source_occupied") for row in rows], dtype=np.int64),
        "total_occupied": np.asarray([integer(row, "total_occupied") for row in rows], dtype=np.int64),
        "left_sink_occupied": np.asarray([integer(row, "left_sink_occupied") for row in rows], dtype=np.int64),
        "right_sink_occupied": np.asarray([integer(row, "sink_occupied") for row in rows], dtype=np.int64),
        "inserted_this_step": np.asarray([integer(row, "inserted_this_step") for row in rows], dtype=np.int64),
        "left_removed_this_step": np.asarray([integer(row, "left_removed_this_step") for row in rows], dtype=np.int64),
        "right_removed_this_step": np.asarray([integer(row, "removed_this_step") for row in rows], dtype=np.int64),
        "trimmed_this_step": np.asarray([integer(row, "trimmed_this_step") for row in rows], dtype=np.int64),
        "cumulative_inserted": np.asarray([integer(row, "cumulative_inserted") for row in rows], dtype=np.int64),
        "cumulative_left_removed": np.asarray([integer(row, "cumulative_left_removed") for row in rows], dtype=np.int64),
        "cumulative_right_removed": np.asarray([integer(row, "cumulative_removed") for row in rows], dtype=np.int64),
        "gcmc_attempts_this_step": np.asarray([integer(row, "gcmc_attempts_this_step") for row in rows], dtype=np.int64),
        "gcmc_accepts_this_step": np.asarray([integer(row, "gcmc_accepts_this_step") for row in rows], dtype=np.int64),
    }
    series["time_ns"] = series["time_s"] * 1.0e9
    total = series["total_occupied"]
    source = series["source_occupied"]
    min_index = int(np.argmin(total))
    max_index = int(np.argmax(total))
    source_min_index = int(np.argmin(source))
    source_max_index = int(np.argmax(source))
    summary = {
        "rows": int(len(rows)),
        "first_step": int(series["step"][0]),
        "last_step": int(series["step"][-1]),
        "first_time_s": float(series["time_s"][0]),
        "last_time_s": float(series["time_s"][-1]),
        "duration_s": float(series["time_s"][-1] - series["time_s"][0]),
        "initial_total_H": int(total[0]),
        "final_total_H": int(total[-1]),
        "min_total_H": int(total[min_index]),
        "min_total_step": int(series["step"][min_index]),
        "max_total_H": int(total[max_index]),
        "max_total_step": int(series["step"][max_index]),
        "mean_total_H": float(np.mean(total)),
        "std_total_H": float(np.std(total)),
        "initial_source_H": int(source[0]),
        "final_source_H": int(source[-1]),
        "min_source_H": int(source[source_min_index]),
        "min_source_step": int(series["step"][source_min_index]),
        "max_source_H": int(source[source_max_index]),
        "max_source_step": int(series["step"][source_max_index]),
        "mean_source_H": float(np.mean(source)),
        "std_source_H": float(np.std(source)),
        "included_inserted_H": int(np.sum(series["inserted_this_step"])),
        "included_left_removed_H": int(np.sum(series["left_removed_this_step"])),
        "included_right_removed_H": int(np.sum(series["right_removed_this_step"])),
        "included_source_trimmed_H": int(np.sum(series["trimmed_this_step"])),
        "included_gcmc_attempts": int(np.sum(series["gcmc_attempts_this_step"])),
        "included_gcmc_accepts": int(np.sum(series["gcmc_accepts_this_step"])),
        "final_cumulative_inserted_H": int(series["cumulative_inserted"][-1]),
        "final_cumulative_left_removed_H": int(series["cumulative_left_removed"][-1]),
        "final_cumulative_right_removed_H": int(series["cumulative_right_removed"][-1]),
    }
    return series, summary


def write_active_h_outputs(output: Path, series: dict[str, np.ndarray], sample_every: int) -> None:
    fields = [
        "step",
        "time_s",
        "time_ns",
        "total_occupied",
        "source_occupied",
        "left_sink_occupied",
        "right_sink_occupied",
        "inserted_this_step",
        "left_removed_this_step",
        "right_removed_this_step",
        "trimmed_this_step",
        "cumulative_inserted",
        "cumulative_left_removed",
        "cumulative_right_removed",
    ]
    rows = []
    sampled = []
    for index in range(len(series["step"])):
        row = {field: series[field][index] for field in fields}
        rows.append(row)
        step = int(series["step"][index])
        if (
            index == 0
            or index == len(series["step"]) - 1
            or step % sample_every == 0
            or int(series["inserted_this_step"][index])
            or int(series["left_removed_this_step"][index])
            or int(series["right_removed_this_step"][index])
            or int(series["trimmed_this_step"][index])
        ):
            sampled.append(row)
    write_csv(output / "active_hydrogen_vs_time.csv", fields, rows)
    write_csv(output / "active_hydrogen_vs_time_sampled.csv", fields, sampled)

    total = series["total_occupied"]
    source = series["source_occupied"]
    non_source = total - source
    steps = series["step"]
    times_ns = series["time_ns"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times_ns, total, linewidth=0.8, label="Total active H")
    axes[0].plot(times_ns, non_source, linewidth=0.75, alpha=0.8, label="Active H outside source")
    axes[0].set(ylabel="Active H count", title="Active hydrogen inventory vs time")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(times_ns, source, linewidth=0.75, color="tab:green", label="Source-region H")
    axes[1].set(xlabel="KMC time (ns)", ylabel="Source H count")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "active_hydrogen_vs_time.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, total, linewidth=0.8, label="Total active H")
    ax.plot(steps, source, linewidth=0.75, label="Source-region H")
    ax.set(xlabel="KMC step", ylabel="H count", title="Active hydrogen inventory vs step")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "active_hydrogen_vs_step.png", dpi=200)
    plt.close(fig)


def analyze_trajectory(
    path: Path,
    final_time_s: float,
    n_sections: int,
) -> tuple[
    list[dict[str, object]],
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    dict[str, object],
]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory: {path}")
    count_sum: np.ndarray | None = None
    count_sum_sq: np.ndarray | None = None
    edges: np.ndarray | None = None
    first_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    frames = 0
    first_step = -1
    last_step = -1

    with path.open("r", encoding="utf-8", newline="") as handle:
        while True:
            frame = read_lammps_frame(handle)
            if frame is None:
                break
            step, x_positions, bounds = frame
            if first_step < 0:
                first_step = step
                first_bounds = bounds
                xlo, xhi = bounds[0]
                edges = np.linspace(xlo, xhi, n_sections + 1)
                count_sum = np.zeros(n_sections, dtype=float)
                count_sum_sq = np.zeros(n_sections, dtype=float)
            last_step = step
            if edges is None or count_sum is None or count_sum_sq is None:
                raise RuntimeError("Trajectory accumulator was not initialized")
            counts, _ = np.histogram(x_positions, bins=edges)
            count_sum += counts
            count_sum_sq += counts * counts
            frames += 1

    if frames == 0 or first_bounds is None or edges is None or count_sum is None or count_sum_sq is None:
        raise RuntimeError(f"No trajectory frames read from {path}")

    means = count_sum / frames
    stds = np.sqrt(np.maximum(0.0, count_sum_sq / frames - means * means))
    ylo, yhi = first_bounds[1]
    zlo, zhi = first_bounds[2]
    area_a2 = (yhi - ylo) * (zhi - zlo)
    width = edges[1] - edges[0]
    rows: list[dict[str, object]] = []
    for index in range(n_sections):
        rows.append(
            {
                "section": index + 1,
                "x_start_A": edges[index],
                "x_end_A": edges[index + 1],
                "x_center_A": 0.5 * (edges[index] + edges[index + 1]),
                "mean_H_count": means[index],
                "std_H_count": stds[index],
                "mean_H_number_density_per_A3": means[index] / (area_a2 * width),
                "frames": frames,
                "sampled_duration_s": final_time_s,
            }
        )
    summary = {
        "trajectory_first_step": first_step,
        "trajectory_last_step": last_step,
        "trajectory_frames": frames,
        "x_length_A": first_bounds[0][1] - first_bounds[0][0],
        "y_length_A": first_bounds[1][1] - first_bounds[1][0],
        "z_length_A": first_bounds[2][1] - first_bounds[2][0],
        "cross_section_area_A2": area_a2,
    }
    return rows, first_bounds, summary


def analyze_cross_section_flux(
    selected_events: Path,
    final_time_s: float,
    included_duration_s: float,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    n_sections: int,
) -> list[dict[str, object]]:
    if not selected_events.is_file():
        raise FileNotFoundError(f"Missing selected-event diagnostics: {selected_events}")
    xlo, xhi = bounds[0]
    ylo, yhi = bounds[1]
    zlo, zhi = bounds[2]
    edges = np.linspace(xlo, xhi, n_sections + 1)
    planes = edges[1:]
    forward = np.zeros(n_sections, dtype=np.int64)
    backward = np.zeros(n_sections, dtype=np.int64)

    with selected_events.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.DictReader(handle):
            x0 = float(row["from_x_A"])
            x1 = float(row["to_x_A"])
            if x1 > x0:
                forward += (x0 < planes) & (planes <= x1)
            elif x1 < x0:
                backward += (x1 < planes) & (planes <= x0)

    area_a2 = (yhi - ylo) * (zhi - zlo)
    rows: list[dict[str, object]] = []
    for index, plane in enumerate(planes):
        net = int(forward[index] - backward[index])
        gross = int(forward[index] + backward[index])
        rows.append(
            {
                "section": index + 1,
                "x_start_A": edges[index],
                "x_end_A": edges[index + 1],
                "measurement_plane_x_A": plane,
                "forward_crossings": int(forward[index]),
                "backward_crossings": int(backward[index]),
                "net_forward_crossings": net,
                "net_flux_practice_style_H_per_A2_s": net / area_a2 / final_time_s if final_time_s > 0 else 0.0,
                "gross_flux_practice_style_H_per_A2_s": gross / area_a2 / final_time_s if final_time_s > 0 else 0.0,
                "net_flux_included_window_H_per_A2_s": net / area_a2 / included_duration_s if included_duration_s > 0 else 0.0,
                "gross_flux_included_window_H_per_A2_s": gross / area_a2 / included_duration_s if included_duration_s > 0 else 0.0,
            }
        )
    return rows


def analyze_sink_flux(
    rows: list[dict[str, str]],
    area_a2: float,
    final_time_s: float,
    included_duration_s: float,
    time_bin_ns: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    right_events: list[tuple[int, float]] = []
    left_events: list[tuple[int, float]] = []
    for row in rows:
        step = integer(row, "step")
        time_ns = floating(row, "time_s") * 1.0e9
        for _ in range(integer(row, "removed_this_step")):
            right_events.append((step, time_ns))
        for _ in range(integer(row, "left_removed_this_step")):
            left_events.append((step, time_ns))

    last_time_ns = final_time_s * 1.0e9
    n_bins = int(math.floor(last_time_ns / time_bin_ns)) + 1 if time_bin_ns > 0 else 1
    right_by_bin: Counter[int] = Counter()
    left_by_bin: Counter[int] = Counter()
    for _, time_ns in right_events:
        right_by_bin[int(time_ns // time_bin_ns)] += 1
    for _, time_ns in left_events:
        left_by_bin[int(time_ns // time_bin_ns)] += 1

    binned: list[dict[str, object]] = []
    cumulative_right = 0
    cumulative_left = 0
    for index in range(n_bins):
        start_ns = index * time_bin_ns
        end_ns = min((index + 1) * time_bin_ns, last_time_ns)
        duration_s = max((end_ns - start_ns) * 1.0e-9, 0.0)
        right_count = right_by_bin[index]
        left_count = left_by_bin[index]
        cumulative_right += right_count
        cumulative_left += left_count
        binned.append(
            {
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "mid_time_ns": 0.5 * (start_ns + end_ns),
                "right_sink_detected_H": right_count,
                "left_absorber_detected_H": left_count,
                "total_absorbed_H": right_count + left_count,
                "cumulative_right_sink_detected_H": cumulative_right,
                "cumulative_left_absorber_detected_H": cumulative_left,
                "right_sink_flux_H_per_A2_s": right_count / area_a2 / duration_s if duration_s > 0 else 0.0,
                "left_absorber_flux_H_per_A2_s": left_count / area_a2 / duration_s if duration_s > 0 else 0.0,
            }
        )

    first = rows[0]
    last = rows[-1]
    included_right = len(right_events)
    included_left = len(left_events)
    final_cumulative_right = integer(last, "cumulative_removed")
    final_cumulative_left = integer(last, "cumulative_left_removed")
    initial_cumulative_right = integer(first, "cumulative_removed")
    initial_cumulative_left = integer(first, "cumulative_left_removed")
    summary = {
        "included_right_sink_removed_H": included_right,
        "included_left_absorber_removed_H": included_left,
        "initial_cumulative_right_sink_removed_H": initial_cumulative_right,
        "initial_cumulative_left_absorber_removed_H": initial_cumulative_left,
        "final_cumulative_right_sink_removed_H": final_cumulative_right,
        "final_cumulative_left_absorber_removed_H": final_cumulative_left,
        "right_sink_flux_practice_style_H_per_A2_s": included_right / area_a2 / final_time_s if final_time_s > 0 else 0.0,
        "left_absorber_flux_practice_style_H_per_A2_s": included_left / area_a2 / final_time_s if final_time_s > 0 else 0.0,
        "right_sink_flux_included_window_H_per_A2_s": included_right / area_a2 / included_duration_s if included_duration_s > 0 else 0.0,
        "left_absorber_flux_included_window_H_per_A2_s": included_left / area_a2 / included_duration_s if included_duration_s > 0 else 0.0,
        "right_sink_flux_cumulative_counter_H_per_A2_s": final_cumulative_right / area_a2 / final_time_s if final_time_s > 0 else 0.0,
        "left_absorber_flux_cumulative_counter_H_per_A2_s": final_cumulative_left / area_a2 / final_time_s if final_time_s > 0 else 0.0,
    }
    return binned, summary


def write_sink_plot(output: Path, binned: list[dict[str, object]], time_bin_ns: float) -> None:
    mids = [float(row["mid_time_ns"]) for row in binned]
    right_counts = [int(row["right_sink_detected_H"]) for row in binned]
    left_counts = [int(row["left_absorber_detected_H"]) for row in binned]
    cumulative_right = [int(row["cumulative_right_sink_detected_H"]) for row in binned]
    cumulative_left = [int(row["cumulative_left_absorber_detected_H"]) for row in binned]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].step(mids, cumulative_right, where="mid", label="Right sink")
    axes[0].step(mids, cumulative_left, where="mid", label="Left absorber")
    axes[0].set(ylabel="Cumulative absorbed H", title="Absorbing-boundary detections")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].bar(mids, right_counts, width=time_bin_ns * 0.9, label="Right sink")
    axes[1].bar(mids, left_counts, bottom=right_counts, width=time_bin_ns * 0.9, label="Left absorber")
    axes[1].set(xlabel="KMC time (ns)", ylabel=f"Absorbed H per {time_bin_ns:g} ns")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "sink_flux_and_absorption.png", dpi=200)
    plt.close(fig)


def write_cross_section_plot(
    output: Path,
    flux_rows: list[dict[str, object]],
    occupancy_rows: list[dict[str, object]],
    sink_x_min_a: float | None,
) -> None:
    x_flux = [float(row["measurement_plane_x_A"]) for row in flux_rows]
    net_flux = [float(row["net_flux_practice_style_H_per_A2_s"]) for row in flux_rows]
    gross_flux = [float(row["gross_flux_practice_style_H_per_A2_s"]) for row in flux_rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x_flux, net_flux, marker="o", label="Net forward flux")
    axes[0].plot(x_flux, gross_flux, marker="s", alpha=0.7, label="Gross crossing flux")
    axes[0].set(ylabel="Flux (H A^-2 s^-1)", title="Practice-style cross-sectional H transport")
    axes[0].set_yscale("symlog", linthresh=1.0e6)
    if sink_x_min_a is not None:
        axes[0].axvline(sink_x_min_a, color="tab:red", linestyle="--", label="Right sink")
        axes[0].axvspan(sink_x_min_a, x_flux[-1], color="tab:red", alpha=0.06)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].errorbar(
        [float(row["x_center_A"]) for row in occupancy_rows],
        [float(row["mean_H_count"]) for row in occupancy_rows],
        yerr=[float(row["std_H_count"]) for row in occupancy_rows],
        marker="o",
        capsize=3,
    )
    axes[1].set(xlabel="x (A)", ylabel="Mean H per section", title="Trajectory-averaged H distribution")
    if sink_x_min_a is not None:
        axes[1].axvline(sink_x_min_a, color="tab:red", linestyle="--")
        axes[1].axvspan(sink_x_min_a, x_flux[-1], color="tab:red", alpha=0.06)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "cross_section_flux_and_occupancy.png", dpi=200)
    plt.close(fig)


def write_summary(
    output: Path,
    run_dir: Path,
    active_summary: dict[str, object],
    trajectory_summary: dict[str, object],
    sink_summary: dict[str, object],
    flux_rows: list[dict[str, object]],
    source_x_min_a: float | None,
    source_x_max_a: float | None,
    right_sink_x_min_a: float | None,
) -> None:
    max_net = max(flux_rows, key=lambda row: abs(float(row["net_flux_practice_style_H_per_A2_s"])))
    gcmc_acceptance = (
        active_summary["included_gcmc_accepts"] / active_summary["included_gcmc_attempts"]
        if active_summary["included_gcmc_attempts"]
        else 0.0
    )
    lines = [
        "# Devanathan Flux And Active-H Inventory Analysis",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Diagnostic rows: {active_summary['rows']:,}",
        f"- Step range: {active_summary['first_step']:,} to {active_summary['last_step']:,}",
        f"- Time range: {active_summary['first_time_s'] * 1e9:.9f} to {active_summary['last_time_s'] * 1e9:.9f} ns",
        f"- Included-window duration: {active_summary['duration_s'] * 1e9:.9f} ns",
        f"- Cross-sectional area: {trajectory_summary['cross_section_area_A2']:.8f} A^2",
        f"- Trajectory frames: {trajectory_summary['trajectory_frames']:,} ({trajectory_summary['trajectory_first_step']:,} to {trajectory_summary['trajectory_last_step']:,})",
        "",
        "## Active H Inventory",
        "",
        f"- Initial / final active H: {active_summary['initial_total_H']:,} / {active_summary['final_total_H']:,}",
        f"- Min / max active H: {active_summary['min_total_H']:,} at step {active_summary['min_total_step']:,} / {active_summary['max_total_H']:,} at step {active_summary['max_total_step']:,}",
        f"- Mean +/- SD active H: {active_summary['mean_total_H']:.4f} +/- {active_summary['std_total_H']:.4f}",
        f"- Initial / final source H: {active_summary['initial_source_H']:,} / {active_summary['final_source_H']:,}",
        f"- Mean +/- SD source H: {active_summary['mean_source_H']:.4f} +/- {active_summary['std_source_H']:.4f}",
        "",
        "## Boundary Flux",
        "",
        f"- Included left-absorber deletions: {sink_summary['included_left_absorber_removed_H']:,}",
        f"- Included right-sink deletions: {sink_summary['included_right_sink_removed_H']:,}",
        f"- Included-window left-absorber flux: {sink_summary['left_absorber_flux_included_window_H_per_A2_s']:.8e} H A^-2 s^-1",
        f"- Included-window right-sink flux: {sink_summary['right_sink_flux_included_window_H_per_A2_s']:.8e} H A^-2 s^-1",
        f"- Practice-style left-absorber flux: {sink_summary['left_absorber_flux_practice_style_H_per_A2_s']:.8e} H A^-2 s^-1",
        f"- Practice-style right-sink flux: {sink_summary['right_sink_flux_practice_style_H_per_A2_s']:.8e} H A^-2 s^-1",
        f"- Cumulative-counter left-absorber flux: {sink_summary['left_absorber_flux_cumulative_counter_H_per_A2_s']:.8e} H A^-2 s^-1",
        f"- Cumulative-counter right-sink flux: {sink_summary['right_sink_flux_cumulative_counter_H_per_A2_s']:.8e} H A^-2 s^-1",
        "",
        "## GCMC Source Maintenance",
        "",
        f"- Included inserted H: {active_summary['included_inserted_H']:,}",
        f"- Included source trims: {active_summary['included_source_trimmed_H']:,}",
        f"- Included GCMC attempts / accepts: {active_summary['included_gcmc_attempts']:,} / {active_summary['included_gcmc_accepts']:,}",
        f"- Included GCMC acceptance fraction: {gcmc_acceptance:.6%}",
        "",
        "## Cross-Section Flux",
        "",
        "Practice-style cross-section flux divides selected-event plane crossings by area and cumulative KMC time, matching the practice-run analysis.",
        f"- Largest absolute net plane: section {max_net['section']} at x = {float(max_net['measurement_plane_x_A']):.8f} A",
        f"- Largest absolute net flux: {float(max_net['net_flux_practice_style_H_per_A2_s']):.8e} H A^-2 s^-1",
    ]
    if source_x_min_a is not None and source_x_max_a is not None:
        lines.append(f"- Source region: x = {source_x_min_a:.8f} to {source_x_max_a:.8f} A")
    if right_sink_x_min_a is not None:
        lines.append(f"- Right sink: x >= {right_sink_x_min_a:.8f} A")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `active_hydrogen_vs_time.csv`",
            "- `active_hydrogen_vs_time_sampled.csv`",
            "- `active_hydrogen_vs_time.png`",
            "- `active_hydrogen_vs_step.png`",
            "- `sink_flux_time_bins.csv`",
            "- `sink_flux_and_absorption.png`",
            "- `cross_section_flux_10_sections.csv`",
            "- `cross_section_H_occupancy_10_sections.csv`",
            "- `cross_section_flux_and_occupancy.png`",
            "- `flux_inventory_summary.csv`",
        ]
    )
    (output / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Consolidated run directory")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sections", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--time-bin-ns", type=float, default=0.05)
    parser.add_argument("--source-x-min-a", type=float, default=16.12518061159)
    parser.add_argument("--source-x-max-a", type=float, default=26.12518061159)
    parser.add_argument("--right-sink-x-min-a", type=float, default=204.19906933748)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else run_dir / "analysis" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = run_dir / "diagnostics" / "kmc_devanathan.csv"
    selected = run_dir / "diagnostics" / "kmc_selected_events.csv"
    trajectory = run_dir / "trajectories" / "H_trajectory_onlyH.lammpstrj"

    rows = load_devanathan_rows(diagnostics)
    series, active_summary = analyze_active_h(rows)
    write_active_h_outputs(out_dir, series, args.sample_every)

    occupancy_rows, bounds, trajectory_summary = analyze_trajectory(
        trajectory,
        float(active_summary["last_time_s"]),
        args.sections,
    )
    write_csv(out_dir / "cross_section_H_occupancy_10_sections.csv", list(occupancy_rows[0]), occupancy_rows)

    sink_bins, sink_summary = analyze_sink_flux(
        rows,
        float(trajectory_summary["cross_section_area_A2"]),
        float(active_summary["last_time_s"]),
        float(active_summary["duration_s"]),
        args.time_bin_ns,
    )
    write_csv(out_dir / "sink_flux_time_bins.csv", list(sink_bins[0]), sink_bins)
    write_sink_plot(out_dir, sink_bins, args.time_bin_ns)

    flux_rows = analyze_cross_section_flux(
        selected,
        float(active_summary["last_time_s"]),
        float(active_summary["duration_s"]),
        bounds,
        args.sections,
    )
    write_csv(out_dir / "cross_section_flux_10_sections.csv", list(flux_rows[0]), flux_rows)
    write_cross_section_plot(out_dir, flux_rows, occupancy_rows, args.right_sink_x_min_a)

    summary_row = {**active_summary, **trajectory_summary, **sink_summary}
    write_csv(out_dir / "flux_inventory_summary.csv", list(summary_row), [summary_row])
    write_summary(
        out_dir,
        run_dir,
        active_summary,
        trajectory_summary,
        sink_summary,
        flux_rows,
        args.source_x_min_a,
        args.source_x_max_a,
        args.right_sink_x_min_a,
    )

    print(f"results={out_dir}")
    print(f"rows={active_summary['rows']} steps={active_summary['first_step']}-{active_summary['last_step']}")
    print(
        "active_H_initial_final="
        f"{active_summary['initial_total_H']}/{active_summary['final_total_H']} "
        f"left_removed_included={sink_summary['included_left_absorber_removed_H']} "
        f"right_removed_included={sink_summary['included_right_sink_removed_H']}"
    )
    print(
        "left_flux_included_window="
        f"{sink_summary['left_absorber_flux_included_window_H_per_A2_s']:.8e} "
        "H A^-2 s^-1"
    )


if __name__ == "__main__":
    main()
