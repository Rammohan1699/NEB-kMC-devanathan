#!/usr/bin/env python3
"""Analyze paired pulse barrier differences by x displacement and GB region."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--lammps-run", type=Path, default=None)
    parser.add_argument("--ase-run", type=Path, default=None)
    parser.add_argument("--site-map", type=Path, default=Path("kmc_map_inputs/sigma5_stage3_unified_sites.npz"))
    parser.add_argument("--site-regions", type=Path, default=Path("kmc_map_inputs/sigma5_site_regions.npz"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--x-bins", type=int, default=80)
    parser.add_argument("--large-delta-e-v", type=float, default=0.05)
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def int_or_none(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except Exception:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


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


def q(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def summarize_values(values: list[float], prefix: str = "") -> dict[str, object]:
    if not values:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean": "",
            f"{prefix}median": "",
            f"{prefix}std": "",
            f"{prefix}min": "",
            f"{prefix}p10": "",
            f"{prefix}p90": "",
            f"{prefix}max": "",
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}count": int(arr.size),
        f"{prefix}mean": float(arr.mean()),
        f"{prefix}median": float(median(values)),
        f"{prefix}std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}min": float(arr.min()),
        f"{prefix}p10": q(values, 0.10),
        f"{prefix}p90": q(values, 0.90),
        f"{prefix}max": float(arr.max()),
    }


@dataclass(frozen=True)
class RegionContext:
    positions: np.ndarray
    labels: np.ndarray
    regions: np.ndarray
    grains: np.ndarray
    box_x: float
    central_left: float
    central_right: float
    gb_center: float

    def site_label(self, site: int) -> str:
        if 0 <= site < len(self.labels):
            return str(self.labels[site])
        return "unknown"

    def site_region(self, site: int) -> str:
        if 0 <= site < len(self.regions):
            return str(self.regions[site])
        return "unknown"

    def site_grain(self, site: int) -> int | None:
        if 0 <= site < len(self.grains):
            return int(self.grains[site])
        return None

    def classify_x(self, from_x: float, to_x: float) -> str:
        mid = 0.5 * (from_x + to_x)
        lo = min(from_x, to_x)
        hi = max(from_x, to_x)
        if lo <= 10.0 or hi >= self.box_x - 10.0:
            return "periodic_gb_edge"
        if hi < self.central_left:
            return "bulk_grain0_side"
        if lo > self.central_right:
            return "bulk_grain1_side"
        if self.central_left <= mid <= self.central_right or hi >= self.central_left and lo <= self.central_right:
            return "central_gb_window"
        return "transition_near_gb"


def load_region_context(site_map: Path, site_regions: Path) -> RegionContext:
    smap = np.load(site_map, allow_pickle=True)
    sreg = np.load(site_regions, allow_pickle=True)
    positions = np.asarray(smap["positions"], dtype=float)
    labels = sreg["site_selection_label"]
    regions = sreg["site_region_name"]
    grains = sreg["site_grain_id"]
    box_x = float(np.asarray(smap["box_lengths"], dtype=float)[0])

    bulk0 = positions[labels == "bulk_grain_0", 0]
    bulk1 = positions[labels == "bulk_grain_1", 0]
    central_left = float(bulk0.max())
    central_right = float(bulk1.min())
    return RegionContext(
        positions=positions,
        labels=labels,
        regions=regions,
        grains=grains,
        box_x=box_x,
        central_left=central_left,
        central_right=central_right,
        gb_center=0.5 * box_x,
    )


def row_key(row: dict[str, str]) -> tuple[int, int, int] | None:
    step = int_or_none(row.get("step"))
    h_site = int_or_none(row.get("h_site"))
    n_site = int_or_none(row.get("n_site"))
    if step is None or h_site is None or n_site is None:
        return None
    return (step, h_site, n_site)


def pair_key(row: dict[str, str]) -> tuple[int, int] | None:
    h_site = int_or_none(row.get("h_site"))
    n_site = int_or_none(row.get("n_site"))
    if h_site is None or n_site is None:
        return None
    return (h_site, n_site)


def enriched_fields(row: dict[str, str], ctx: RegionContext) -> dict[str, object] | None:
    h_site = int_or_none(row.get("h_site"))
    n_site = int_or_none(row.get("n_site"))
    from_x = finite_float(row.get("from_x_A"))
    to_x = finite_float(row.get("to_x_A"))
    dx = finite_float(row.get("dx_A"))
    barrier = finite_float(row.get("barrier_eV"))
    if h_site is None or n_site is None or from_x is None or to_x is None or dx is None or barrier is None:
        return None
    from_label = ctx.site_label(h_site)
    to_label = ctx.site_label(n_site)
    from_region = ctx.site_region(h_site)
    to_region = ctx.site_region(n_site)
    from_grain = ctx.site_grain(h_site)
    to_grain = ctx.site_grain(n_site)
    mid_x = 0.5 * (from_x + to_x)
    return {
        "step": int_or_none(row.get("step")),
        "h_site": h_site,
        "n_site": n_site,
        "from_x_A": from_x,
        "to_x_A": to_x,
        "mid_x_A": mid_x,
        "dx_A": dx,
        "dx_bin_A": round(dx, 3),
        "hop_distance_A": finite_float(row.get("hop_distance_A")),
        "barrier_eV": barrier,
        "selected": int_or_none(row.get("selected")) or 0,
        "x_region_class": ctx.classify_x(from_x, to_x),
        "from_label": from_label,
        "to_label": to_label,
        "label_pair": f"{from_label}->{to_label}",
        "from_region": from_region,
        "to_region": to_region,
        "region_pair": f"{from_region}->{to_region}",
        "from_grain": "" if from_grain is None else from_grain,
        "to_grain": "" if to_grain is None else to_grain,
        "grain_pair": f"{from_grain}->{to_grain}",
    }


def iter_finite_rate_rows(path: Path, ctx: RegionContext) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            fields = enriched_fields(row, ctx)
            if fields is None:
                continue
            yield fields


def group_summary(rows: Iterable[dict[str, object]], keys: tuple[str, ...], value_field: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get(value_field))
        if value is None:
            continue
        grouped[tuple(row.get(key, "") for key in keys)].append(value)
    out: list[dict[str, object]] = []
    for group_key, values in grouped.items():
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(summarize_values(values, ""))
        out.append(row)
    out.sort(key=lambda r: tuple(str(r[k]) for k in keys))
    return out


def load_rate_minimal(path: Path, ctx: RegionContext) -> tuple[dict[tuple[int, int, int], dict[str, object]], dict[tuple[int, int], list[dict[str, object]]]]:
    by_step: dict[tuple[int, int, int], dict[str, object]] = {}
    by_pair: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in iter_finite_rate_rows(path, ctx):
        step_key = (int(row["step"]), int(row["h_site"]), int(row["n_site"]))
        # Prefer selected duplicate if it ever exists.
        existing = by_step.get(step_key)
        if existing is None or int(row.get("selected", 0)) == 1:
            by_step[step_key] = row
        by_pair[(int(row["h_site"]), int(row["n_site"]))].append(row)
    return by_step, by_pair


def exact_step_delta_rows(
    lammps: dict[tuple[int, int, int], dict[str, object]],
    ase: dict[tuple[int, int, int], dict[str, object]],
    large_delta: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(set(lammps) & set(ase)):
        lmp = lammps[key]
        ase_row = ase[key]
        delta = float(ase_row["barrier_eV"]) - float(lmp["barrier_eV"])
        rows.append(
            {
                "step": key[0],
                "h_site": key[1],
                "n_site": key[2],
                "from_x_A": lmp["from_x_A"],
                "to_x_A": lmp["to_x_A"],
                "mid_x_A": lmp["mid_x_A"],
                "dx_A": lmp["dx_A"],
                "dx_bin_A": lmp["dx_bin_A"],
                "x_region_class": lmp["x_region_class"],
                "label_pair": lmp["label_pair"],
                "region_pair": lmp["region_pair"],
                "lammps_barrier_eV": lmp["barrier_eV"],
                "ase_barrier_eV": ase_row["barrier_eV"],
                "delta_ase_minus_lammps_eV": delta,
                "abs_delta_eV": abs(delta),
                "large_delta": int(abs(delta) >= large_delta),
                "lammps_selected": lmp["selected"],
                "ase_selected": ase_row["selected"],
            }
        )
    rows.sort(key=lambda r: (-float(r["abs_delta_eV"]), int(r["step"]), int(r["h_site"]), int(r["n_site"])))
    return rows


def median_pair_rows(
    lammps: dict[tuple[int, int], list[dict[str, object]]],
    ase: dict[tuple[int, int], list[dict[str, object]]],
    large_delta: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(set(lammps) & set(ase)):
        lvals = [float(row["barrier_eV"]) for row in lammps[key]]
        avals = [float(row["barrier_eV"]) for row in ase[key]]
        ref = lammps[key][0]
        l_med = float(median(lvals))
        a_med = float(median(avals))
        delta = a_med - l_med
        rows.append(
            {
                "h_site": key[0],
                "n_site": key[1],
                "from_x_A": ref["from_x_A"],
                "to_x_A": ref["to_x_A"],
                "mid_x_A": ref["mid_x_A"],
                "dx_A": ref["dx_A"],
                "dx_bin_A": ref["dx_bin_A"],
                "x_region_class": ref["x_region_class"],
                "label_pair": ref["label_pair"],
                "region_pair": ref["region_pair"],
                "lammps_count": len(lvals),
                "ase_count": len(avals),
                "lammps_mean_barrier_eV": float(mean(lvals)),
                "ase_mean_barrier_eV": float(mean(avals)),
                "lammps_median_barrier_eV": l_med,
                "ase_median_barrier_eV": a_med,
                "delta_median_ase_minus_lammps_eV": delta,
                "abs_delta_median_eV": abs(delta),
                "large_delta": int(abs(delta) >= large_delta),
            }
        )
    rows.sort(key=lambda r: (-float(r["abs_delta_median_eV"]), int(r["h_site"]), int(r["n_site"])))
    return rows


def delta_group_summary(rows: list[dict[str, object]], keys: tuple[str, ...], delta_field: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    abs_grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        delta = finite_float(row.get(delta_field))
        if delta is None:
            continue
        key = tuple(row.get(k, "") for k in keys)
        grouped[key].append(delta)
        abs_grouped[key].append(abs(delta))
    out: list[dict[str, object]] = []
    for key, values in grouped.items():
        row = {k: v for k, v in zip(keys, key)}
        row.update(summarize_values(values, "delta_"))
        row.update(summarize_values(abs_grouped[key], "abs_delta_"))
        out.append(row)
    out.sort(key=lambda r: tuple(str(r[k]) for k in keys))
    return out


def candidate_engine_summaries(run_paths: dict[str, Path], ctx: RegionContext, out_dir: Path) -> dict[str, object]:
    overall: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    dx_rows: list[dict[str, object]] = []
    region_dx_rows: list[dict[str, object]] = []
    label_pair_rows: list[dict[str, object]] = []
    xbin_rows: list[dict[str, object]] = []
    edges = np.linspace(0.0, ctx.box_x, 81)

    for engine, run in run_paths.items():
        rate_path = run / "diagnostics" / "rates_allranks.csv"
        rows = list(iter_finite_rate_rows(rate_path, ctx))
        barriers = [float(row["barrier_eV"]) for row in rows]
        overall_row = {"engine": engine, "rows": len(rows), "unique_steps": len({row["step"] for row in rows})}
        overall_row.update(summarize_values(barriers, "barrier_"))
        overall.append(overall_row)

        for table, keys in [
            (region_rows, ("x_region_class",)),
            (dx_rows, ("dx_bin_A",)),
            (region_dx_rows, ("x_region_class", "dx_bin_A")),
            (label_pair_rows, ("label_pair",)),
        ]:
            for row in group_summary(rows, keys, "barrier_eV"):
                row = {"engine": engine, **row}
                table.append(row)

        grouped: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            mid = float(row["mid_x_A"])
            idx = int(np.searchsorted(edges, mid, side="right") - 1)
            idx = min(max(idx, 0), len(edges) - 2)
            grouped[idx].append(float(row["barrier_eV"]))
        for idx, values in grouped.items():
            out = {
                "engine": engine,
                "x_bin_left_A": float(edges[idx]),
                "x_bin_right_A": float(edges[idx + 1]),
                "x_mid_A": float(0.5 * (edges[idx] + edges[idx + 1])),
            }
            out.update(summarize_values(values, "barrier_"))
            xbin_rows.append(out)

    write_csv(out_dir / "candidate_barrier_overall_by_engine.csv", overall)
    write_csv(out_dir / "candidate_barrier_summary_by_x_region.csv", region_rows)
    write_csv(out_dir / "candidate_barrier_summary_by_dx.csv", dx_rows)
    write_csv(out_dir / "candidate_barrier_summary_by_x_region_and_dx.csv", region_dx_rows)
    write_csv(out_dir / "candidate_barrier_summary_by_label_pair.csv", label_pair_rows)
    write_csv(out_dir / "candidate_barrier_summary_by_mid_x_bin.csv", xbin_rows)
    return {"overall": overall, "region_rows": region_rows, "dx_rows": dx_rows, "xbin_rows": xbin_rows}


def selected_summaries(run_paths: dict[str, Path], ctx: RegionContext, out_dir: Path) -> dict[str, object]:
    selected_by_engine: dict[str, list[dict[str, object]]] = {}
    selected_region_rows: list[dict[str, object]] = []
    selected_dx_rows: list[dict[str, object]] = []
    for engine, run in run_paths.items():
        rows: list[dict[str, object]] = []
        for raw in read_csv(run / "diagnostics" / "kmc_selected_events.csv"):
            fields = enriched_fields({**raw, "selected": "1"}, ctx)
            if fields is not None:
                rows.append(fields)
        selected_by_engine[engine] = rows
        for table, keys in [(selected_region_rows, ("x_region_class",)), (selected_dx_rows, ("dx_bin_A",))]:
            for row in group_summary(rows, keys, "barrier_eV"):
                table.append({"engine": engine, **row})

    write_csv(out_dir / "selected_barrier_summary_by_x_region.csv", selected_region_rows)
    write_csv(out_dir / "selected_barrier_summary_by_dx.csv", selected_dx_rows)

    lmp_by_step = {int(row["step"]): row for row in selected_by_engine.get("lammps_only", [])}
    ase_by_step = {int(row["step"]): row for row in selected_by_engine.get("ase_neb", [])}
    step_rows: list[dict[str, object]] = []
    for step in sorted(set(lmp_by_step) | set(ase_by_step)):
        lmp = lmp_by_step.get(step)
        ase = ase_by_step.get(step)
        same = bool(lmp and ase and lmp["h_site"] == ase["h_site"] and lmp["n_site"] == ase["n_site"])
        row: dict[str, object] = {"step": step, "same_selected_hop": int(same)}
        if lmp:
            row.update(
                {
                    "lammps_h_site": lmp["h_site"],
                    "lammps_n_site": lmp["n_site"],
                    "lammps_mid_x_A": lmp["mid_x_A"],
                    "lammps_dx_A": lmp["dx_A"],
                    "lammps_x_region_class": lmp["x_region_class"],
                    "lammps_barrier_eV": lmp["barrier_eV"],
                }
            )
        if ase:
            row.update(
                {
                    "ase_h_site": ase["h_site"],
                    "ase_n_site": ase["n_site"],
                    "ase_mid_x_A": ase["mid_x_A"],
                    "ase_dx_A": ase["dx_A"],
                    "ase_x_region_class": ase["x_region_class"],
                    "ase_barrier_eV": ase["barrier_eV"],
                }
            )
        if lmp and ase:
            row["selected_barrier_delta_ase_minus_lammps_eV"] = float(ase["barrier_eV"]) - float(lmp["barrier_eV"])
        step_rows.append(row)
    write_csv(out_dir / "selected_event_step_comparison.csv", step_rows)
    return {"selected_by_engine": selected_by_engine, "step_rows": step_rows}


def timing_summaries(run_paths: dict[str, Path], selected: dict[str, list[dict[str, object]]], out_dir: Path) -> None:
    rows_out: list[dict[str, object]] = []
    summary_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for engine, run in run_paths.items():
        timing_path = run / "logs" / "timing_rank0.csv"
        if not timing_path.exists():
            continue
        sel_by_step = {int(row["step"]): row for row in selected.get(engine, [])}
        for raw in read_csv(timing_path):
            step = int_or_none(raw.get("step"))
            if step is None:
                continue
            sel = sel_by_step.get(step)
            region = str(sel["x_region_class"]) if sel else "no_selected_row"
            barrier_neb = finite_float(raw.get("barrier_neb")) or 0.0
            wall = finite_float(raw.get("wall_total")) or 0.0
            row = {
                "engine": engine,
                "step": step,
                "selected_x_region_class": region,
                "selected_mid_x_A": "" if sel is None else sel["mid_x_A"],
                "selected_dx_A": "" if sel is None else sel["dx_A"],
                "barrier_neb_s": barrier_neb,
                "wall_total_s": wall,
                "job_comm_s": finite_float(raw.get("job_comm")) or "",
                "result_comm_s": finite_float(raw.get("result_comm")) or "",
                "cache_update_s": finite_float(raw.get("cache_update")) or "",
            }
            rows_out.append(row)
            summary_values[(engine, region)].append(barrier_neb)
    write_csv(out_dir / "timing_rank0_by_selected_step.csv", rows_out)

    summary_rows: list[dict[str, object]] = []
    for (engine, region), values in sorted(summary_values.items()):
        row = {"engine": engine, "selected_x_region_class": region}
        row.update(summarize_values(values, "barrier_neb_s_"))
        summary_rows.append(row)
    write_csv(out_dir / "timing_rank0_summary_by_selected_region.csv", summary_rows)


def plot_xbin(xbin_rows: list[dict[str, object]], out_dir: Path, ctx: RegionContext) -> None:
    if plt is None or not xbin_rows:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    for engine, color in [("lammps_only", "#355f8a"), ("ase_neb", "#c44e52")]:
        rows = [row for row in xbin_rows if row["engine"] == engine]
        rows.sort(key=lambda r: float(r["x_mid_A"]))
        if not rows:
            continue
        ax.plot(
            [float(row["x_mid_A"]) for row in rows],
            [float(row["barrier_mean"]) for row in rows],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            color=color,
            label=engine,
        )
    ax.axvspan(ctx.central_left, ctx.central_right, color="#f0b84f", alpha=0.22, label="central GB window")
    ax.axvline(ctx.gb_center, color="#7a3b00", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Candidate hop midpoint x (A)")
    ax.set_ylabel("Mean candidate barrier (eV)")
    ax.set_title("Candidate barrier vs x")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_mean_barrier_vs_mid_x.png")
    fig.savefig(out_dir / "candidate_mean_barrier_vs_mid_x.svg")
    plt.close(fig)


def plot_delta_x(rows: list[dict[str, object]], delta_field: str, out_dir: Path, ctx: RegionContext, stem: str) -> None:
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    xs = [float(row["mid_x_A"]) for row in rows]
    ys = [float(row[delta_field]) for row in rows]
    colors = ["#c44e52" if y >= 0 else "#355f8a" for y in ys]
    ax.scatter(xs, ys, s=10, c=colors, alpha=0.28, linewidths=0)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.axvspan(ctx.central_left, ctx.central_right, color="#f0b84f", alpha=0.22, label="central GB window")
    ax.axvline(ctx.gb_center, color="#7a3b00", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Hop midpoint x (A)")
    ax.set_ylabel("ASE - LAMMPS barrier (eV)")
    ax.set_title(stem.replace("_", " "))
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)


def write_markdown(
    path: Path,
    *,
    ctx: RegionContext,
    configs: dict[str, dict[str, str]],
    candidate_summary: dict[str, object],
    exact_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    selected_info: dict[str, object],
    large_delta: float,
) -> None:
    def fmt(value: object, digits: int = 4) -> str:
        try:
            f = float(value)
        except Exception:
            return str(value)
        return f"{f:.{digits}g}"

    exact_deltas = [float(row["delta_ase_minus_lammps_eV"]) for row in exact_rows]
    pair_deltas = [float(row["delta_median_ase_minus_lammps_eV"]) for row in pair_rows]
    exact_gb = [row for row in exact_rows if row["x_region_class"] == "central_gb_window"]
    pair_gb = [row for row in pair_rows if row["x_region_class"] == "central_gb_window"]
    selected_rows = selected_info["step_rows"]
    same_selected = sum(int(row.get("same_selected_hop", 0)) for row in selected_rows)
    compared_selected = sum(1 for row in selected_rows if "lammps_h_site" in row and "ase_h_site" in row)

    lammps_shell = (
        configs["lammps_only"].get("SHELL_INNER_RADIUS_A", ""),
        configs["lammps_only"].get("SHELL_OUTER_RADIUS_A", ""),
    )
    ase_shell = (
        configs["ase_neb"].get("SHELL_INNER_RADIUS_A", ""),
        configs["ase_neb"].get("SHELL_OUTER_RADIUS_A", ""),
    )
    shell_note = (
        "Note: LAMMPS-only and ASE-NEB use the same extraction shell in this comparison; "
        "remaining deltas are dominated by engine/optimizer/path differences and trajectory divergence."
        if lammps_shell == ase_shell
        else (
            "Note: LAMMPS-only and ASE-NEB use different extraction shells in this comparison. "
            "Treat absolute deltas as workflow deltas, not pure engine-only deltas."
        )
    )

    lines = [
        "# Pulse Barrier Difference Analysis",
        "",
        "## Inputs",
        "",
        f"- Central GB window inferred from site labels: `{ctx.central_left:.6f}` to `{ctx.central_right:.6f}` A; GB center `{ctx.gb_center:.6f}` A.",
        f"- LAMMPS-only engine: `{configs['lammps_only'].get('NEB_ENGINE', '')}`.",
        f"- LAMMPS-only shell: `{lammps_shell[0]}/{lammps_shell[1]}` A.",
        f"- LAMMPS-only native endpoint optimization: `{configs['lammps_only'].get('LAMMPS_ONLY_ENDPOINT_OPTIMIZE', '')}`.",
        f"- LAMMPS-only endpoint optimization settings: ftol `{configs['lammps_only'].get('LAMMPS_ENDPOINT_OPT_FTOL', '')}`, steps `{configs['lammps_only'].get('LAMMPS_ENDPOINT_OPT_STEPS', '')}`.",
        f"- ASE-NEB engine: `{configs['ase_neb'].get('NEB_ENGINE', '')}`.",
        f"- ASE-NEB shell: `{ase_shell[0]}/{ase_shell[1]}` A.",
        f"- ASE endpoint optimization: `{configs['ase_neb'].get('OPTIMIZE_ENDPOINTS', '')}`.",
        f"- Large-delta threshold: `{large_delta:g}` eV.",
        "",
        shell_note,
        "",
        "## Overall Candidate Barrier Distributions",
        "",
    ]
    for row in candidate_summary["overall"]:
        lines.append(
            f"- `{row['engine']}`: {row['rows']:,} finite candidate rows over {row['unique_steps']:,} steps; "
            f"mean barrier {fmt(row['barrier_mean'])} eV, median {fmt(row['barrier_median'])} eV, "
            f"p90 {fmt(row['barrier_p90'])} eV."
        )
    lines.extend(
        [
            "",
            "## Exact Same-Step Same-Hop Deltas",
            "",
            f"- Exact matched candidate hops: {len(exact_rows):,}.",
            f"- Central-GB exact matched hops: {len(exact_gb):,}.",
            f"- Overall exact mean delta ASE-LAMMPS: {fmt(mean(exact_deltas) if exact_deltas else '')} eV; median {fmt(median(exact_deltas) if exact_deltas else '')} eV.",
            f"- Overall exact mean |delta|: {fmt(mean(abs(v) for v in exact_deltas) if exact_deltas else '')} eV.",
            f"- Exact large deltas >= {large_delta:g} eV: {sum(abs(v) >= large_delta for v in exact_deltas):,}.",
        ]
    )
    if exact_gb:
        gb_d = [float(row["delta_ase_minus_lammps_eV"]) for row in exact_gb]
        lines.append(
            f"- Central-GB exact mean delta: {fmt(mean(gb_d))} eV; median {fmt(median(gb_d))} eV; "
            f"mean |delta| {fmt(mean(abs(v) for v in gb_d))} eV."
        )
    lines.extend(
        [
            "",
            "## Site-Pair Median Barrier Deltas",
            "",
            "This compares the median barrier for the same `h_site -> n_site` pair across each run, ignoring the fact that the co-occupied H environment may differ after the trajectories diverge.",
            "",
            f"- Common site pairs: {len(pair_rows):,}.",
            f"- Central-GB common site pairs: {len(pair_gb):,}.",
            f"- Overall pair-median mean delta ASE-LAMMPS: {fmt(mean(pair_deltas) if pair_deltas else '')} eV; median {fmt(median(pair_deltas) if pair_deltas else '')} eV.",
            f"- Overall pair-median mean |delta|: {fmt(mean(abs(v) for v in pair_deltas) if pair_deltas else '')} eV.",
            f"- Pair-median large deltas >= {large_delta:g} eV: {sum(abs(v) >= large_delta for v in pair_deltas):,}.",
        ]
    )
    if pair_gb:
        gb_d = [float(row["delta_median_ase_minus_lammps_eV"]) for row in pair_gb]
        lines.append(
            f"- Central-GB pair-median mean delta: {fmt(mean(gb_d))} eV; median {fmt(median(gb_d))} eV; "
            f"mean |delta| {fmt(mean(abs(v) for v in gb_d))} eV."
        )
    lines.extend(
        [
            "",
            "## Selected Trajectories",
            "",
            f"- Same selected hop at same step: {same_selected:,}/{compared_selected:,}.",
        ]
    )

    selected_by_engine = selected_info["selected_by_engine"]
    for engine, rows in selected_by_engine.items():
        if not rows:
            continue
        gb_count = sum(1 for row in rows if row["x_region_class"] == "central_gb_window")
        barriers = [float(row["barrier_eV"]) for row in rows]
        lines.append(
            f"- `{engine}` selected events: {len(rows):,}; central-GB selected events {gb_count:,}; "
            f"mean selected barrier {fmt(mean(barriers))} eV."
        )

    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `candidate_barrier_summary_by_x_region.csv`",
            "- `candidate_barrier_summary_by_dx.csv`",
            "- `candidate_barrier_summary_by_x_region_and_dx.csv`",
            "- `exact_step_hop_deltas.csv`",
            "- `exact_step_hop_delta_summary_by_x_region_and_dx.csv`",
            "- `site_pair_median_deltas.csv`",
            "- `site_pair_median_delta_summary_by_x_region_and_dx.csv`",
            "- `selected_barrier_summary_by_x_region.csv`",
            "- `timing_rank0_summary_by_selected_region.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    campaign = args.campaign
    lammps_run = args.lammps_run or campaign / "lammps_only"
    ase_run = args.ase_run or campaign / "ase_neb"
    out_dir = args.out_dir or campaign / "analysis" / "barrier_dx_gb_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_region_context(args.site_map, args.site_regions)
    run_paths = {"lammps_only": lammps_run, "ase_neb": ase_run}
    configs = {engine: read_env(run / "run_config_snapshot.env") for engine, run in run_paths.items()}
    (out_dir / "region_context.json").write_text(
        json.dumps(
            {
                "box_x_A": ctx.box_x,
                "central_gb_left_A": ctx.central_left,
                "central_gb_right_A": ctx.central_right,
                "central_gb_center_A": ctx.gb_center,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate_summary = candidate_engine_summaries(run_paths, ctx, out_dir)
    selected_info = selected_summaries(run_paths, ctx, out_dir)
    timing_summaries(run_paths, selected_info["selected_by_engine"], out_dir)

    lmp_step, lmp_pair = load_rate_minimal(lammps_run / "diagnostics" / "rates_allranks.csv", ctx)
    ase_step, ase_pair = load_rate_minimal(ase_run / "diagnostics" / "rates_allranks.csv", ctx)

    exact_rows = exact_step_delta_rows(lmp_step, ase_step, args.large_delta_e_v)
    pair_rows = median_pair_rows(lmp_pair, ase_pair, args.large_delta_e_v)

    write_csv(out_dir / "exact_step_hop_deltas.csv", exact_rows)
    write_csv(out_dir / "site_pair_median_deltas.csv", pair_rows)

    write_csv(
        out_dir / "exact_step_hop_delta_summary_by_x_region.csv",
        delta_group_summary(exact_rows, ("x_region_class",), "delta_ase_minus_lammps_eV"),
    )
    write_csv(
        out_dir / "exact_step_hop_delta_summary_by_dx.csv",
        delta_group_summary(exact_rows, ("dx_bin_A",), "delta_ase_minus_lammps_eV"),
    )
    write_csv(
        out_dir / "exact_step_hop_delta_summary_by_x_region_and_dx.csv",
        delta_group_summary(exact_rows, ("x_region_class", "dx_bin_A"), "delta_ase_minus_lammps_eV"),
    )
    write_csv(
        out_dir / "site_pair_median_delta_summary_by_x_region.csv",
        delta_group_summary(pair_rows, ("x_region_class",), "delta_median_ase_minus_lammps_eV"),
    )
    write_csv(
        out_dir / "site_pair_median_delta_summary_by_dx.csv",
        delta_group_summary(pair_rows, ("dx_bin_A",), "delta_median_ase_minus_lammps_eV"),
    )
    write_csv(
        out_dir / "site_pair_median_delta_summary_by_x_region_and_dx.csv",
        delta_group_summary(pair_rows, ("x_region_class", "dx_bin_A"), "delta_median_ase_minus_lammps_eV"),
    )
    write_csv(
        out_dir / "site_pair_median_delta_summary_by_label_pair.csv",
        delta_group_summary(pair_rows, ("label_pair",), "delta_median_ase_minus_lammps_eV"),
    )

    plot_xbin(candidate_summary["xbin_rows"], out_dir, ctx)
    plot_delta_x(exact_rows, "delta_ase_minus_lammps_eV", out_dir, ctx, "exact_step_hop_delta_vs_mid_x")
    plot_delta_x(pair_rows, "delta_median_ase_minus_lammps_eV", out_dir, ctx, "site_pair_median_delta_vs_mid_x")

    write_markdown(
        out_dir / "BARRIER_DX_GB_ANALYSIS.md",
        ctx=ctx,
        configs=configs,
        candidate_summary=candidate_summary,
        exact_rows=exact_rows,
        pair_rows=pair_rows,
        selected_info=selected_info,
        large_delta=args.large_delta_e_v,
    )
    print(f"analysis_dir={out_dir}")
    print(f"exact_matched_hops={len(exact_rows)}")
    print(f"common_site_pairs={len(pair_rows)}")


if __name__ == "__main__":
    main()
