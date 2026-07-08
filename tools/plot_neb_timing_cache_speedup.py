#!/usr/bin/env python3
"""Aggregate NEB timing over a restart-chain KMC run and plot cache speedup."""
from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Segment:
    root: Path
    start: int
    end: int


def parse_segment(value: str) -> Segment:
    try:
        root_s, range_s = value.split(":", 1)
        start_s, end_s = range_s.split("-", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("segment must be RUN_DIR:START-END") from exc
    return Segment(Path(root_s).expanduser(), int(start_s), int(end_s))


def timing_files(root: Path) -> list[Path]:
    files = sorted((root / "logs").glob("timing_rank*.csv"))
    if files:
        return files
    return sorted(root.glob("timing_rank*.csv"))


def rank_from_path(path: Path) -> int:
    match = re.search(r"timing_rank(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else -1


def bin_index(step: int, start: int, bin_size: int, n_bins: int) -> int | None:
    if step < start:
        return None
    idx = (step - start) // bin_size
    if idx < 0 or idx >= n_bins:
        return None
    return idx


def aggregate_segments(segments: list[Segment], *, start: int, end: int, bin_size: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    n_bins = (end - start + bin_size) // bin_size
    step_max_by_bin = [dict() for _ in range(n_bins)]
    sum_by_rank = [dict() for _ in range(n_bins)]
    rank_rows_by_bin = [dict() for _ in range(n_bins)]
    all_rank_ids: set[int] = set()
    segment_ranges: list[str] = []

    for segment in segments:
        segment_ranges.append(f"{segment.root}:{segment.start}-{segment.end}")
        for path in timing_files(segment.root):
            rank = rank_from_path(path)
            if rank < 0:
                continue
            all_rank_ids.add(rank)
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        step = int(row["step"])
                    except Exception:
                        continue
                    if step < segment.start or step > segment.end or step < start or step > end:
                        continue
                    idx = bin_index(step, start, bin_size, n_bins)
                    if idx is None:
                        continue
                    try:
                        barrier_neb = float(row.get("barrier_neb", "0") or 0.0)
                    except Exception:
                        barrier_neb = 0.0
                    if not math.isfinite(barrier_neb):
                        barrier_neb = 0.0
                    rank_rows_by_bin[idx][rank] = int(rank_rows_by_bin[idx].get(rank, 0)) + 1
                    step_max_by_bin[idx][step] = max(float(step_max_by_bin[idx].get(step, 0.0)), barrier_neb)
                    sum_by_rank[idx][rank] = float(sum_by_rank[idx].get(rank, 0.0)) + barrier_neb

    rows: list[dict[str, object]] = []
    ranks = sorted(all_rank_ids)
    for idx in range(n_bins):
        left = start + idx * bin_size
        right = min(left + bin_size - 1, end)
        rank_sum_values = [float(sum_by_rank[idx].get(rank, 0.0)) for rank in ranks]
        rank_row_counts = [int(rank_rows_by_bin[idx].get(rank, 0)) for rank in ranks]
        steps_observed = max(rank_row_counts) if rank_row_counts else 0
        neb_wall = sum(float(value) for value in step_max_by_bin[idx].values())
        neb_cpu = sum(rank_sum_values)
        rows.append(
            {
                "step_bin_left": left,
                "step_bin_right": right,
                "mean_step": (left + right) / 2.0,
                "steps_observed": steps_observed,
                "ranks_observed": sum(1 for count in rank_row_counts if count > 0),
                "neb_wall_time_s": neb_wall,
                "neb_cpu_time_s": neb_cpu,
                "neb_wall_time_per_step_s": neb_wall / steps_observed if steps_observed else 0.0,
                "neb_cpu_time_per_step_s": neb_cpu / steps_observed if steps_observed else 0.0,
            }
        )

    baseline_candidates = [float(row["neb_wall_time_per_step_s"]) for row in rows[: max(1, min(5, len(rows)))] if float(row["neb_wall_time_per_step_s"]) > 0.0]
    cpu_baseline_candidates = [float(row["neb_cpu_time_per_step_s"]) for row in rows[: max(1, min(5, len(rows)))] if float(row["neb_cpu_time_per_step_s"]) > 0.0]
    wall_baseline = float(np.mean(baseline_candidates)) if baseline_candidates else 0.0
    cpu_baseline = float(np.mean(cpu_baseline_candidates)) if cpu_baseline_candidates else 0.0
    for row in rows:
        wall = float(row["neb_wall_time_per_step_s"])
        cpu = float(row["neb_cpu_time_per_step_s"])
        row["relative_wall_speedup_vs_initial"] = wall_baseline / wall if wall > 0.0 and wall_baseline > 0.0 else ""
        row["relative_cpu_speedup_vs_initial"] = cpu_baseline / cpu if cpu > 0.0 and cpu_baseline > 0.0 else ""

    summary = {
        "segments": "; ".join(segment_ranges),
        "rank_count": len(ranks),
        "bin_size": bin_size,
        "wall_baseline_per_step_s": wall_baseline,
        "cpu_baseline_per_step_s": cpu_baseline,
        "total_neb_wall_time_s": sum(float(row["neb_wall_time_s"]) for row in rows),
        "total_neb_cpu_time_s": sum(float(row["neb_cpu_time_s"]) for row in rows),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "step_bin_left",
        "step_bin_right",
        "mean_step",
        "steps_observed",
        "ranks_observed",
        "neb_wall_time_s",
        "neb_cpu_time_s",
        "neb_wall_time_per_step_s",
        "neb_cpu_time_per_step_s",
        "relative_wall_speedup_vs_initial",
        "relative_cpu_speedup_vs_initial",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scale_points(values: list[float], lo: float, hi: float, out_lo: float, out_hi: float) -> list[float]:
    if hi <= lo:
        return [(out_lo + out_hi) / 2.0 for _ in values]
    return [out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo) for value in values]


def plot_timing(path: Path, rows: list[dict[str, object]]) -> None:
    rows = [row for row in rows if int(row["steps_observed"]) > 0]
    if not rows:
        return
    x = [float(row["mean_step"]) for row in rows]
    wall = [float(row["neb_wall_time_per_step_s"]) for row in rows]
    cpu = [float(row["neb_cpu_time_per_step_s"]) for row in rows]
    speed = [
        float(row["relative_wall_speedup_vs_initial"])
        for row in rows
        if row["relative_wall_speedup_vs_initial"] != ""
    ]

    width, height = 1120, 680
    left, right, top, bottom = 92, 115, 54, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_lo, x_hi = min(x), max(x)
    y_values = wall + cpu
    y_lo = 0.0
    y_hi = max(y_values) * 1.08 if y_values else 1.0
    speed_hi = max(speed) * 1.08 if speed else 1.0

    sx = scale_points(x, x_lo, x_hi, left, left + plot_w)
    wall_y = scale_points(wall, y_lo, y_hi, top + plot_h, top)
    cpu_y = scale_points(cpu, y_lo, y_hi, top + plot_h, top)
    speed_values = [
        float(row["relative_wall_speedup_vs_initial"]) if row["relative_wall_speedup_vs_initial"] != "" else 0.0
        for row in rows
    ]
    speed_y = scale_points(speed_values, 0.0, speed_hi, top + plot_h, top)

    def poly(xs: list[float], ys: list[float]) -> str:
        return " ".join(f"{px:.1f},{py:.1f}" for px, py in zip(xs, ys))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">NEB timing over 2M KMC steps</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    for tick in np.linspace(x_lo, x_hi, 6):
        px = scale_points([float(tick)], x_lo, x_hi, left, left + plot_w)[0]
        lines.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{px:.1f}" y="{height - 42}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick/1_000_000:.2f}M</text>')
    for tick in np.linspace(y_lo, y_hi, 6):
        py = scale_points([float(tick)], y_lo, y_hi, top + plot_h, top)[0]
        lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{py + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.3g}</text>')
    for tick in np.linspace(0.0, speed_hi, 6):
        py = scale_points([float(tick)], 0.0, speed_hi, top + plot_h, top)[0]
        lines.append(f'<text x="{left + plot_w + 8}" y="{py + 4:.1f}" font-family="Arial, sans-serif" font-size="11">{tick:.3g}x</text>')

    lines.append(f'<polyline points="{poly(sx, cpu_y)}" fill="none" stroke="#8c6bb1" stroke-width="1.8"/>')
    lines.append(f'<polyline points="{poly(sx, wall_y)}" fill="none" stroke="#2b6c9f" stroke-width="2.0"/>')
    lines.append(f'<polyline points="{poly(sx, speed_y)}" fill="none" stroke="#d95f02" stroke-width="1.8" stroke-dasharray="5 4"/>')
    max_count = max(int(row["steps_observed"]) for row in rows)
    for px, py, row in zip(sx, wall_y, rows):
        radius = 2.5 + 3.0 * math.sqrt(max(int(row["steps_observed"]), 1) / max_count)
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="#2b6c9f" fill-opacity="0.65">'
            f'<title>steps {int(row["step_bin_left"])}-{int(row["step_bin_right"])}; '
            f'wall NEB/step={float(row["neb_wall_time_per_step_s"]):.6g}s; '
            f'speedup={row["relative_wall_speedup_vs_initial"]}</title></circle>'
        )

    legend_x = left + plot_w - 300
    legend_y = top + 22
    legend = [
        ("#2b6c9f", "rank-max NEB wall time / KMC step", ""),
        ("#8c6bb1", "rank-sum NEB CPU time / KMC step", ""),
        ("#d95f02", "relative wall speedup vs first bins", "5 4"),
    ]
    for i, (color, label, dash) in enumerate(legend):
        y = legend_y + i * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="2"{dash_attr}/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12">{html.escape(label)}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 14}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">KMC step</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">NEB time per KMC step (s)</text>')
    lines.append(f'<text x="{width - 24}" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(90 {width - 24} {top + plot_h / 2:.1f})">relative speedup</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    first = next((row for row in rows if float(row["neb_wall_time_per_step_s"]) > 0.0), None)
    last = next((row for row in reversed(rows) if int(row["steps_observed"]) > 0), None)
    lines = ["# NEB Timing Cache Speedup", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    if first and last:
        first_wall = float(first["neb_wall_time_per_step_s"])
        last_wall = float(last["neb_wall_time_per_step_s"])
        lines.extend(
            [
                "",
                f"- first_nonzero_window: {int(first['step_bin_left'])}-{int(first['step_bin_right'])}",
                f"- first_nonzero_wall_neb_per_step_s: {first_wall}",
                f"- last_window: {int(last['step_bin_left'])}-{int(last['step_bin_right'])}",
                f"- last_wall_neb_per_step_s: {last_wall}",
                f"- first_to_last_wall_speedup: {first_wall / last_wall if last_wall > 0 else ''}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--segment", action="append", type=parse_segment, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1999999)
    parser.add_argument("--bin-size", type=int, default=10000)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = [Segment(s.root.resolve(), s.start, s.end) for s in args.segment]
    rows, summary = aggregate_segments(segments, start=args.start, end=args.end, bin_size=args.bin_size)
    write_csv(out_dir / "neb_timing_vs_step.csv", rows)
    plot_timing(out_dir / "neb_timing_vs_step.svg", rows)
    write_summary(out_dir / "neb_timing_summary.md", summary, rows)
    print(f"Wrote NEB timing graph to {out_dir / 'neb_timing_vs_step.svg'}")
    print(f"Wrote NEB timing CSV to {out_dir / 'neb_timing_vs_step.csv'}")


if __name__ == "__main__":
    main()
