#!/usr/bin/env python3
"""Plot selected-barrier regional averages split by KMC time windows."""
from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np


DEFAULT_LABEL_ORDER = ["bulk_grain_0", "transition", "grain_boundary", "bulk_grain_1"]
REGION_COLORS = {
    "bulk_grain_0": "#d7ecff",
    "transition": "#ffe2a8",
    "grain_boundary": "#f4b6b6",
    "bulk_grain_1": "#ddf3df",
}


def region_color(label: object) -> str:
    return REGION_COLORS.get(str(label), "#cfcfcf")


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{self.start}-{self.end}"

    @property
    def stem(self) -> str:
        return f"steps_{self.start:07d}_{self.end:07d}"


def finite_float(value: object) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def make_windows(start: int, end: int, size: int) -> list[Window]:
    windows: list[Window] = []
    left = int(start)
    while left <= int(end):
        right = min(left + int(size) - 1, int(end))
        windows.append(Window(left, right))
        left = right + 1
    return windows


def summarize(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean_barrier_eV": float(arr.mean()),
        "median_barrier_eV": float(median(values)),
        "std_barrier_eV": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_barrier_eV": float(arr.min()),
        "max_barrier_eV": float(arr.max()),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_site_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["positions"], dtype=float), np.asarray(data["box_lengths"], dtype=float)


def x_region_bands(positions: np.ndarray, labels: np.ndarray, lx: float) -> list[dict[str, object]]:
    bands: list[dict[str, object]] = []
    preferred = ["grain_boundary", "transition", "bulk_grain_0", "bulk_grain_1"]
    all_labels = preferred + sorted(str(x) for x in set(labels.tolist()) if str(x) not in preferred)
    gap_threshold = max(1.5, float(lx) / 100.0)
    for label in all_labels:
        xs = np.sort(positions[labels == label, 0])
        if xs.size == 0:
            continue
        start = float(xs[0])
        prev = float(xs[0])
        for x_val in xs[1:]:
            x = float(x_val)
            if x - prev > gap_threshold:
                bands.append({"label": label, "left": start, "right": prev, "color": region_color(label)})
                start = x
            prev = x
        bands.append({"label": label, "left": start, "right": prev, "color": region_color(label)})
    return bands


def fixed_x_barrier_summary(
    points: list[dict[str, object]],
    *,
    n_bins: int,
    lx: float,
) -> list[dict[str, object]]:
    edges = np.linspace(0.0, float(lx), max(2, int(n_bins) + 1))
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == len(edges) - 2:
            members = [point for point in points if left <= float(point["x_position_A"]) <= right]
        else:
            members = [point for point in points if left <= float(point["x_position_A"]) < right]
        if not members:
            continue
        x_values = np.asarray([float(point["x_position_A"]) for point in members], dtype=float)
        barriers = [float(point["barrier_eV"]) for point in members]
        row = {
            "x_bin_left_A": left,
            "x_bin_right_A": right,
            "count": int(len(members)),
            "mean_x_A": float(x_values.mean()),
        }
        row.update(summarize(barriers))
        rows.append(row)
    return rows


def plot_x_position_summary(
    rows: list[dict[str, object]],
    out_svg: Path,
    *,
    title: str,
    lx: float,
    x_bands: list[dict[str, object]],
) -> None:
    points = [
        (float(row["mean_x_A"]), float(row["mean_barrier_eV"]), int(row.get("count", 0)))
        for row in rows
        if row.get("mean_x_A", "") != "" and row.get("mean_barrier_eV", "") != ""
    ]
    if not points:
        return

    x_vals = np.asarray([p[0] for p in points], dtype=float)
    y_vals = np.asarray([p[1] for p in points], dtype=float)
    counts = np.asarray([p[2] for p in points], dtype=int)

    width, height = 900, 520
    left, right, top, bottom = 80, 35, 50, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_lo, x_hi = 0.0, float(lx)
    y_span = max(float(np.ptp(y_vals)), 1.0e-6)
    y_lo = max(0.0, float(y_vals.min()) - 0.08 * y_span)
    y_hi = float(y_vals.max()) + 0.08 * y_span
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0e-3

    def sx(x: float) -> float:
        return left + (x - x_lo) * plot_w / (x_hi - x_lo)

    def sy(y: float) -> float:
        return top + plot_h - (y - y_lo) * plot_h / (y_hi - y_lo)

    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_vals, y_vals))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    legend_x = left
    legend_y = top - 14
    seen_labels: set[str] = set()
    for band in x_bands:
        band_left = max(x_lo, float(band["left"]))
        band_right = min(x_hi, float(band["right"]))
        if band_right <= x_lo or band_left >= x_hi or band_right <= band_left:
            continue
        x1 = sx(band_left)
        x2 = sx(band_right)
        label = str(band["label"])
        color = str(band["color"])
        lines.append(f'<rect x="{x1:.1f}" y="{top}" width="{max(0.0, x2 - x1):.1f}" height="{plot_h}" fill="{color}" fill-opacity="0.34"/>')
        if label not in seen_labels:
            seen_labels.add(label)
            lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="10" fill="{color}" fill-opacity="0.7"/>')
            lines.append(f'<text x="{legend_x + 16}" y="{legend_y}" font-family="Arial, sans-serif" font-size="11">{html.escape(label)}</text>')
            legend_x += 18 + len(label) * 6.2 + 18
    for tick in np.linspace(x_lo, x_hi, 6):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 40}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick:.3g}</text>')
    for tick in np.linspace(y_lo, y_hi, 5):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.3g}</text>')
    lines.append(f'<polyline points="{poly}" fill="none" stroke="#2b6c9f" stroke-width="1.5"/>')
    max_count = max(int(counts.max()), 1)
    for x, y, count in points:
        radius = 3.0 + 6.0 * math.sqrt(max(count, 1) / max_count)
        lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius:.1f}" fill="#3579a8" fill-opacity="0.78"><title>n={count}</title></circle>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">Mean post-move H x position (Angstrom)</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">Mean selected barrier (eV)</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        frac = 0.5
    else:
        frac = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
    # Blue -> light -> red, restrained for readability.
    if frac < 0.5:
        t = frac / 0.5
        r = int(70 + (245 - 70) * t)
        g = int(120 + (245 - 120) * t)
        b = int(190 + (245 - 190) * t)
    else:
        t = (frac - 0.5) / 0.5
        r = int(245 + (190 - 245) * t)
        g = int(245 + (70 - 245) * t)
        b = int(245 + (65 - 245) * t)
    return f"rgb({r},{g},{b})"


def plot_zone_pair_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    labels: list[str],
    *,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    cell = 118
    left = 190
    top = 78
    right = 150
    bottom = 90
    width = left + cell * len(labels) + right
    height = top + cell * len(labels) + bottom
    row_by_pair = {(str(r["from_label"]), str(r["to_label"])): r for r in rows}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="28" y="32" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left + cell * len(labels) / 2:.1f}" y="{height - 28}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">destination zone</text>',
        f'<text x="24" y="{top + cell * len(labels) / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 24 {top + cell * len(labels) / 2:.1f})">origin zone</text>',
    ]
    for j, label in enumerate(labels):
        x = left + j * cell + cell / 2
        lines.append(f'<text x="{x:.1f}" y="{top - 12}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{html.escape(label)}</text>')
    for i, label in enumerate(labels):
        y = top + i * cell + cell / 2 + 4
        lines.append(f'<text x="{left - 10}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{html.escape(label)}</text>')

    for i, from_label in enumerate(labels):
        for j, to_label in enumerate(labels):
            x = left + j * cell
            y = top + i * cell
            row = row_by_pair.get((from_label, to_label))
            if row is None:
                fill = "#f7f7f7"
                label = "n=0"
                title_text = f"{from_label} -> {to_label}; no events"
            else:
                mean = float(row["mean_barrier_eV"])
                count = int(row["count"])
                fill = color_scale(mean, vmin, vmax)
                label = f'{mean:.4f} eV'
                title_text = f"{from_label} -> {to_label}; mean={mean:.6g} eV; n={count}"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="2"><title>{html.escape(title_text)}</title></rect>')
            lines.append(f'<text x="{x + cell/2:.1f}" y="{y + cell/2 - 4:.1f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" text-anchor="middle">{html.escape(label)}</text>')
            if row is not None:
                lines.append(f'<text x="{x + cell/2:.1f}" y="{y + cell/2 + 17:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">n={int(row["count"])}</text>')

    legend_x = left + cell * len(labels) + 38
    legend_y = top + 6
    legend_h = cell * len(labels) - 12
    for k in range(80):
        frac = k / 79
        y = legend_y + (1.0 - frac) * legend_h
        lines.append(f'<rect x="{legend_x}" y="{y:.1f}" width="18" height="{legend_h / 79 + 1:.1f}" fill="{color_scale(vmin + frac * (vmax - vmin), vmin, vmax)}"/>')
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="18" height="{legend_h}" fill="none" stroke="#999999"/>')
    lines.append(f'<text x="{legend_x + 28}" y="{legend_y + 5}" font-family="Arial, sans-serif" font-size="11">{vmax:.4f}</text>')
    lines.append(f'<text x="{legend_x + 28}" y="{legend_y + legend_h:.1f}" font-family="Arial, sans-serif" font-size="11">{vmin:.4f}</text>')
    lines.append(f'<text x="{legend_x}" y="{legend_y + legend_h + 24:.1f}" font-family="Arial, sans-serif" font-size="11">mean eV</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_from_zone_bars(path: Path, rows: list[dict[str, object]], labels: list[str], *, title: str, ymax: float) -> None:
    width, height = 820, 440
    left, right, top, bottom = 92, 28, 54, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    rows_by_label = {str(r["from_label"]): r for r in rows}
    bar_gap = 24
    bar_w = (plot_w - bar_gap * (len(labels) - 1)) / max(len(labels), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    for tick in np.linspace(0.0, ymax, 6):
        y = top + plot_h - tick * plot_h / ymax
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.3g}</text>')
    for i, label in enumerate(labels):
        row = rows_by_label.get(label)
        x = left + i * (bar_w + bar_gap)
        if row is None:
            h = 0.0
            mean = 0.0
            count = 0
        else:
            mean = float(row["mean_barrier_eV"])
            count = int(row["count"])
            h = mean * plot_h / ymax
        y = top + plot_h - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{REGION_COLORS.get(label, "#cccccc")}" stroke="#666666"><title>{html.escape(label)}; mean={mean:.6g}; n={count}</title></rect>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{mean:.4f}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 42}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle">{html.escape(label)}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 24}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle">n={count}</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">mean selected barrier (eV)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-events", type=Path, required=True)
    parser.add_argument("--region-file", type=Path, required=True)
    parser.add_argument("--site-map", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=500000)
    parser.add_argument("--x-bins", type=int, default=25)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1999999)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = make_windows(args.start, args.end, args.window_size)
    regions = np.load(args.region_file, allow_pickle=True)
    labels = np.asarray(regions["site_selection_label"], dtype=object)
    site_map_path = args.site_map or (args.region_file.parent / "sigma5_stage3_unified_sites.npz")
    positions, box_lengths = load_site_map(site_map_path)
    lx = float(box_lengths[0])
    region_bands = x_region_bands(positions, labels, lx)

    label_order = [label for label in DEFAULT_LABEL_ORDER if label in {str(x) for x in labels.tolist()}]
    for label in sorted({str(x) for x in labels.tolist()}):
        if label not in label_order:
            label_order.append(label)

    pair_groups: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    from_groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    x_points_by_window: dict[int, list[dict[str, object]]] = defaultdict(list)

    with args.selected_events.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                step = int(row["step"])
                h_site = int(row["h_site"])
                n_site = int(row["n_site"])
            except Exception:
                continue
            if step < args.start or step > args.end:
                continue
            barrier = finite_float(row.get("barrier_eV"))
            if barrier is None:
                continue
            window_idx = (step - args.start) // args.window_size
            if window_idx < 0 or window_idx >= len(windows):
                continue
            from_label = str(labels[h_site])
            to_label = str(labels[n_site])
            pair_groups[(window_idx, from_label, to_label)].append(barrier)
            from_groups[(window_idx, from_label)].append(barrier)
            x_position = finite_float(row.get("to_x_A"))
            if x_position is None:
                x_position = float(positions[n_site, 0])
            x_points_by_window[window_idx].append(
                {
                    "x_position_A": float(x_position) % lx,
                    "barrier_eV": barrier,
                }
            )

    pair_rows: list[dict[str, object]] = []
    from_rows: list[dict[str, object]] = []
    for (window_idx, from_label, to_label), values in sorted(pair_groups.items()):
        window = windows[window_idx]
        row = {
            "window_start_step": window.start,
            "window_end_step": window.end,
            "from_label": from_label,
            "to_label": to_label,
        }
        row.update(summarize(values))
        pair_rows.append(row)
    for (window_idx, from_label), values in sorted(from_groups.items()):
        window = windows[window_idx]
        row = {
            "window_start_step": window.start,
            "window_end_step": window.end,
            "from_label": from_label,
        }
        row.update(summarize(values))
        from_rows.append(row)

    write_csv(
        args.out_dir / "time_window_selected_barrier_summary_by_zone_pair.csv",
        pair_rows,
        [
            "window_start_step",
            "window_end_step",
            "from_label",
            "to_label",
            "count",
            "mean_barrier_eV",
            "median_barrier_eV",
            "std_barrier_eV",
            "min_barrier_eV",
            "max_barrier_eV",
        ],
    )
    write_csv(
        args.out_dir / "time_window_selected_barrier_summary_by_from_zone.csv",
        from_rows,
        [
            "window_start_step",
            "window_end_step",
            "from_label",
            "count",
            "mean_barrier_eV",
            "median_barrier_eV",
            "std_barrier_eV",
            "min_barrier_eV",
            "max_barrier_eV",
        ],
    )

    x_rows_all: list[dict[str, object]] = []
    for window_idx, window in enumerate(windows):
        rows = fixed_x_barrier_summary(x_points_by_window.get(window_idx, []), n_bins=args.x_bins, lx=lx)
        for row in rows:
            row["window_start_step"] = window.start
            row["window_end_step"] = window.end
        x_rows_all.extend(rows)
    write_csv(
        args.out_dir / "time_window_avg_barrier_vs_path_x_position.csv",
        x_rows_all,
        [
            "window_start_step",
            "window_end_step",
            "x_bin_left_A",
            "x_bin_right_A",
            "count",
            "mean_x_A",
            "mean_barrier_eV",
            "median_barrier_eV",
            "std_barrier_eV",
            "min_barrier_eV",
            "max_barrier_eV",
        ],
    )

    means = [float(row["mean_barrier_eV"]) for row in pair_rows]
    vmin = min(means) if means else 0.0
    vmax = max(means) if means else 1.0
    from_means = [float(row["mean_barrier_eV"]) for row in from_rows]
    ymax = max(from_means) * 1.12 if from_means else 1.0

    for window in windows:
        pair_members = [
            row for row in pair_rows
            if int(row["window_start_step"]) == window.start and int(row["window_end_step"]) == window.end
        ]
        from_members = [
            row for row in from_rows
            if int(row["window_start_step"]) == window.start and int(row["window_end_step"]) == window.end
        ]
        title_suffix = f"steps {window.start:,}-{window.end:,}"
        plot_zone_pair_heatmap(
            args.out_dir / f"avg_selected_barrier_zone_pair_heatmap_{window.stem}.svg",
            pair_members,
            label_order,
            title=f"Average selected barrier by zone pair, {title_suffix}",
            vmin=vmin,
            vmax=vmax,
        )
        plot_from_zone_bars(
            args.out_dir / f"avg_selected_barrier_by_from_zone_{window.stem}.svg",
            from_members,
            label_order,
            title=f"Average selected barrier by origin zone, {title_suffix}",
            ymax=ymax,
        )
        x_members = [
            row for row in x_rows_all
            if int(row["window_start_step"]) == window.start and int(row["window_end_step"]) == window.end
        ]
        plot_x_position_summary(
            x_members,
            args.out_dir / f"avg_barrier_vs_path_x_position_{window.stem}.svg",
            title=f"All H selected barriers averaged by x position, {title_suffix}",
            lx=lx,
            x_bands=region_bands,
        )

    print(f"Wrote {len(windows)} time-window regional barrier image sets to {args.out_dir}")


if __name__ == "__main__":
    main()
