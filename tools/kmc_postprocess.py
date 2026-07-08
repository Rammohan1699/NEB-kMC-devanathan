#!/usr/bin/env python3
"""
Post-process KMC H diffusion runs.

This helper is the offline analysis companion for completed KMC/NEB runs. It
discovers the run log, H-only trajectory, optional previous-run trajectory,
site-region metadata, and timestep CSVs, then rebuilds selected-event context
without rerunning the simulation.

It produces:
  - selected event table with barriers, positions, regions, and atom IDs
  - H-neighbor counts/lists around each selected event
  - average barrier summaries and plots for region/grain transitions
  - average selected barrier versus signed x displacement
  - average local H-neighbor count and barrier change versus signed x displacement
  - one CSV "sheet" per H atom with the barriers on its selected path
  - optional path plot for a selected H atom

Example:
    python tools/kmc_postprocess.py \
      --run-dir one-crystal-exp \
      --previous-run . \
      --h-id 1
"""
from __future__ import annotations

import argparse
import csv
import html
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np


REGION_COLORS = {
    "bulk_grain_0": "#d7ecff",
    "bulk_grain_1": "#ddf3df",
    "transition": "#ffe2a8",
    "grain_boundary": "#f4b6b6",
    "bulk": "#e6eef7",
}


SELECTED_RE = re.compile(
    r"Step\s+(?P<step>\d+):\s+selected H\s+"
    r"(?P<h_site>\d+)->(?P<n_site>\d+),\s+"
    r"barrier=(?P<barrier>[^,]+),\s+"
    r"rate=(?P<rate>[^,\s]+)\s+Hz,\s+"
    r"dt=(?P<dt>[^,\s]+)\s+s"
)


@dataclass(frozen=True)
class RunInputs:
    root: Path
    log: Path
    trajectory: Path
    timestep_csv: Optional[Path]


def _existing(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_inputs(run_dir: Path, previous_run: Optional[Path]) -> list[RunInputs]:
    roots: list[Path] = []
    if previous_run is not None:
        roots.append(previous_run.resolve())
    roots.append(run_dir.resolve())

    inputs: list[RunInputs] = []
    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        log = _first_existing(root / "logs" / "log_rank0.txt", root / "log_rank0.txt")
        traj = root / "H_trajectory_onlyH.lammpstrj"
        if log is None or not log.exists():
            # Fall back to diagnostics for move endpoints only; barriers will be absent.
            log = root / "kmc_diagnostics_rank0.log"
        if not log.exists() or not traj.exists():
            continue
        inputs.append(
            RunInputs(
                root=root,
                log=log,
                trajectory=traj,
                timestep_csv=_first_existing(
                    root / "diagnostics" / "kmc_timestep_vs_step.csv",
                    root / "kmc_timestep_vs_step.csv",
                ),
            )
        )
    if not inputs:
        raise SystemExit("No usable run inputs found. Need logs/log_rank0.txt or kmc_diagnostics_rank0.log, plus H_trajectory_onlyH.lammpstrj.")
    return inputs


def discover_rate_files(inputs: Iterable[RunInputs]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for inp in inputs:
        for path in (inp.root / "diagnostics" / "rates_allranks.csv", inp.root / "rates_allranks.csv"):
            if path.exists() and path.resolve() not in seen:
                paths.append(path)
                seen.add(path.resolve())
                break
    return paths


def load_site_metadata(map_dir: Path) -> dict[str, np.ndarray]:
    site_map = np.load(map_dir / "sigma5_stage3_unified_sites.npz", allow_pickle=True)
    regions = np.load(map_dir / "sigma5_site_regions.npz", allow_pickle=True)
    positions = np.asarray(site_map["positions"], dtype=float)
    labels = np.asarray(regions["site_selection_label"], dtype=object)
    region_names = np.asarray(regions["site_region_name"], dtype=object)
    grain_ids = np.asarray(regions["site_grain_id"], dtype=int)
    if not (len(positions) == len(labels) == len(region_names) == len(grain_ids)):
        raise ValueError("Site map and region metadata lengths do not match")
    return {
        "positions": positions,
        "box_lengths": np.asarray(site_map["box_lengths"], dtype=float),
        "labels": labels,
        "region_names": region_names,
        "grain_ids": grain_ids,
    }


def region_color(label: object) -> str:
    return REGION_COLORS.get(str(label), "#cfcfcf")


def x_region_bands(meta: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Cluster site labels into x intervals for lightweight plot overlays."""
    positions = meta["positions"]
    labels = meta["labels"]
    lx = float(meta["box_lengths"][0])
    bands: list[dict[str, object]] = []
    preferred = ["grain_boundary", "transition", "bulk_grain_0", "bulk_grain_1"]
    all_labels = preferred + sorted(str(x) for x in set(labels.tolist()) if str(x) not in preferred)
    gap_threshold = max(1.5, lx / 100.0)
    for label in all_labels:
        xs = np.sort(positions[labels == label, 0])
        if xs.size == 0:
            continue
        start = float(xs[0])
        prev = float(xs[0])
        for x in xs[1:]:
            x = float(x)
            if x - prev > gap_threshold:
                bands.append({"label": label, "left": start, "right": prev, "color": region_color(label)})
                start = x
            prev = x
        bands.append({"label": label, "left": start, "right": prev, "color": region_color(label)})
    return bands


def parse_float(value: str) -> float:
    value = value.strip()
    if value.lower() in {"none", "nan"}:
        return float("nan")
    if value.lower() in {"inf", "+inf", "infinity"}:
        return float("inf")
    return float(value)


def read_selected_events(log_paths: Iterable[Path]) -> list[dict[str, object]]:
    events: dict[int, dict[str, object]] = {}
    diag_re = re.compile(
        r"^\s*(?P<step>\d+)\s+\|\s+H index\s+"
        r"(?P<h_site>\d+)\s+->\s+(?P<n_site>\d+).*?"
        r"dt\s*=\s*(?P<dt>[-+0-9.eE]+)\s+s.*?"
        r"t_total\s*=\s*(?P<total>[-+0-9.eE]+)\s+s"
    )
    for path in log_paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = SELECTED_RE.search(line)
                if m:
                    step = int(m.group("step"))
                    events[step] = {
                        "step": step,
                        "h_site": int(m.group("h_site")),
                        "n_site": int(m.group("n_site")),
                        "barrier_eV": parse_float(m.group("barrier")),
                        "rate_Hz": parse_float(m.group("rate")),
                        "dt_s": parse_float(m.group("dt")),
                    }
                    continue
                m = diag_re.search(line)
                if m and int(m.group("step")) not in events:
                    step = int(m.group("step"))
                    events[step] = {
                        "step": step,
                        "h_site": int(m.group("h_site")),
                        "n_site": int(m.group("n_site")),
                        "barrier_eV": float("nan"),
                        "rate_Hz": float("nan"),
                        "dt_s": parse_float(m.group("dt")),
                    }
    return [events[k] for k in sorted(events)]


def read_selected_events_csv(path: Path) -> list[dict[str, object]]:
    """Read the newer machine-readable selected-event dump written during KMC."""
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                event = {
                    "step": int(row["step"]),
                    "atom_id": int(row["atom_id"]) if str(row.get("atom_id", "")).strip() else "",
                    "h_site": int(row["h_site"]),
                    "n_site": int(row["n_site"]),
                    "barrier_eV": parse_float(str(row.get("barrier_eV", "nan"))),
                    "rate_Hz": parse_float(str(row.get("rate_Hz", "nan"))),
                    "dt_s": parse_float(str(row.get("dt_s", "nan"))),
                    "total_time_s": parse_float(str(row.get("total_time_s", "nan"))),
                    "post_timestep": int(row["step"]) + 1,
                    "post_x": parse_float(str(row.get("to_x_A", "nan"))),
                    "post_y": parse_float(str(row.get("to_y_A", "nan"))),
                    "post_z": parse_float(str(row.get("to_z_A", "nan"))),
                }
            except Exception:
                continue
            events.append(event)
    return events


def read_timestep_totals(paths: Iterable[Optional[Path]]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for path in paths:
        if path is None or not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    totals[int(row["step"])] = float(row["total"])
                except Exception:
                    continue
    return totals


def iter_lammps_h_frames(path: Path) -> Iterator[tuple[int, list[dict[str, object]]]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"{path}: expected ITEM: TIMESTEP, got {line!r}")
            step = int(fh.readline().strip())
            marker = fh.readline()
            if marker.strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"{path}: expected atom count marker at timestep {step}")
            natoms = int(fh.readline().strip())
            bounds_marker = fh.readline()
            if not bounds_marker.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{path}: expected BOX BOUNDS at timestep {step}")
            for _ in range(3):
                fh.readline()
            header = fh.readline().strip().split()[2:]
            col = {name: idx for idx, name in enumerate(header)}
            required = {"id", "x", "y", "z", "site"}
            missing = required.difference(col)
            if missing:
                raise ValueError(f"{path}: missing trajectory columns {sorted(missing)}")
            rows: list[dict[str, object]] = []
            for _ in range(natoms):
                parts = fh.readline().split()
                rows.append(
                    {
                        "atom_id": int(parts[col["id"]]),
                        "x": float(parts[col["x"]]),
                        "y": float(parts[col["y"]]),
                        "z": float(parts[col["z"]]),
                        "site": int(parts[col["site"]]),
                    }
                )
            yield step, rows


def pbc_delta(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image delta for orthogonal periodic cell lengths."""
    return delta - box * np.round(delta / box)


def annotate_event(event: dict[str, object], meta: dict[str, np.ndarray]) -> None:
    h = int(event["h_site"])
    n = int(event["n_site"])
    pos = meta["positions"]
    labels = meta["labels"]
    names = meta["region_names"]
    grains = meta["grain_ids"]
    delta = pbc_delta(pos[n] - pos[h], meta["box_lengths"])
    event.update(
        {
            "from_x": float(pos[h, 0]),
            "from_y": float(pos[h, 1]),
            "from_z": float(pos[h, 2]),
            "to_x": float(pos[n, 0]),
            "to_y": float(pos[n, 1]),
            "to_z": float(pos[n, 2]),
            "from_label": str(labels[h]),
            "to_label": str(labels[n]),
            "from_region": str(names[h]),
            "to_region": str(names[n]),
            "from_grain_id": int(grains[h]),
            "to_grain_id": int(grains[n]),
            "dx_A": float(delta[0]),
            "dy_A": float(delta[1]),
            "dz_A": float(delta[2]),
            "abs_dx_A": float(abs(delta[0])),
            "hop_distance_A": float(np.linalg.norm(delta)),
        }
    )
    event["move_class"] = classify_move(str(labels[h]), str(labels[n]), int(grains[h]), int(grains[n]))


def classify_move(from_label: str, to_label: str, from_grain: int, to_grain: int) -> str:
    if from_label == to_label:
        return f"{from_label}_internal"
    if from_label == "grain_boundary" and to_label == "transition":
        return "gb_exit_to_transition"
    if from_label == "transition" and to_label == "grain_boundary":
        return "transition_entry_to_gb"
    if from_label.startswith("bulk") and to_label == "transition":
        return f"{from_label}_entry_to_transition"
    if from_label == "transition" and to_label.startswith("bulk"):
        return f"transition_exit_to_{to_label}"
    if from_grain != to_grain and from_grain >= 0 and to_grain >= 0:
        return "grain_id_crossing"
    return f"{from_label}_to_{to_label}"


def count_occupied_regions(occupied_sites: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    occupied_labels = labels[occupied_sites]
    return {
        "bulk0": int(np.sum(occupied_labels == "bulk_grain_0")),
        "bulk1": int(np.sum(occupied_labels == "bulk_grain_1")),
        "transition": int(np.sum(occupied_labels == "transition")),
        "gb": int(np.sum(occupied_labels == "grain_boundary")),
        "gb_transition": int(np.sum((occupied_labels == "grain_boundary") | (occupied_labels == "transition"))),
    }


def format_number_list(values: Iterable[float], digits: int = 4) -> str:
    return ";".join(f"{float(v):.{digits}f}" for v in values)


def format_int_list(values: Iterable[int]) -> str:
    return ";".join(str(int(v)) for v in values)


def event_neighbor_summary(
    occupied_sites: np.ndarray,
    site_to_atom: dict[int, int],
    moving_site: int,
    meta: dict[str, np.ndarray],
    *,
    cutoff_a: float,
    k: int,
) -> dict[str, object]:
    pos = meta["positions"]
    box = meta["box_lengths"]
    other_sites = occupied_sites[occupied_sites != int(moving_site)]
    if other_sites.size == 0:
        return {"count": 0, "nearest_distances": "", "nearest_sites": "", "nearest_atom_ids": ""}
    deltas = pbc_delta(pos[other_sites] - pos[int(moving_site)], box)
    distances = np.linalg.norm(deltas, axis=1)
    order = np.argsort(distances)
    nearest = order[: max(0, int(k))]
    nearest_sites = other_sites[nearest]
    nearest_distances = distances[nearest]
    nearest_atom_ids = [site_to_atom.get(int(site), -1) for site in nearest_sites]
    return {
        "count": int(np.sum(distances <= float(cutoff_a))),
        "nearest_distances": format_number_list(nearest_distances),
        "nearest_sites": format_int_list(nearest_sites),
        "nearest_atom_ids": format_int_list(nearest_atom_ids),
    }


def annotate_events_from_trajectory(
    events: list[dict[str, object]],
    trajectories: Iterable[Path],
    meta: dict[str, np.ndarray],
    *,
    neighbor_cutoff_a: float,
    neighbor_k: int,
) -> None:
    by_post_step: dict[int, list[int]] = defaultdict(list)
    for idx, event in enumerate(events):
        by_post_step[int(event["step"]) + 1].append(idx)
        event["atom_id"] = ""
        event["h_neighbors_before_count"] = ""
        event["h_neighbors_after_count"] = ""
        event["h_neighbor_sites_before"] = ""
        event["h_neighbor_sites_after"] = ""
        event["h_neighbor_atom_ids_before"] = ""
        event["h_neighbor_atom_ids_after"] = ""
        event["h_neighbor_distances_before_A"] = ""
        event["h_neighbor_distances_after_A"] = ""

    for path in trajectories:
        for timestep, rows in iter_lammps_h_frames(path):
            wanted = by_post_step.get(timestep)
            if not wanted:
                continue
            site_to_atom = {int(row["site"]): int(row["atom_id"]) for row in rows}
            occupied_after = np.asarray([int(row["site"]) for row in rows], dtype=int)
            labels = meta["labels"]
            for idx in wanted:
                event = events[idx]
                h_site = int(event["h_site"])
                n_site = int(event["n_site"])
                post_row = next((row for row in rows if int(row["site"]) == n_site), None)
                atom_id = site_to_atom.get(n_site)
                if atom_id is not None:
                    event["atom_id"] = atom_id
                if post_row is not None:
                    event["post_timestep"] = int(timestep)
                    event["post_x"] = float(post_row["x"])
                    event["post_y"] = float(post_row["y"])
                    event["post_z"] = float(post_row["z"])
                occupied_before = occupied_after.copy()
                occupied_before[occupied_before == n_site] = h_site
                counts_before = count_occupied_regions(occupied_before, labels)
                counts_after = count_occupied_regions(occupied_after, labels)
                before = event_neighbor_summary(
                    occupied_before,
                    site_to_atom,
                    h_site,
                    meta,
                    cutoff_a=neighbor_cutoff_a,
                    k=neighbor_k,
                )
                after = event_neighbor_summary(
                    occupied_after,
                    site_to_atom,
                    n_site,
                    meta,
                    cutoff_a=neighbor_cutoff_a,
                    k=neighbor_k,
                )
                event["h_neighbors_before_count"] = before["count"]
                event["h_neighbors_after_count"] = after["count"]
                event["h_neighbor_sites_before"] = before["nearest_sites"]
                event["h_neighbor_sites_after"] = after["nearest_sites"]
                event["h_neighbor_atom_ids_before"] = before["nearest_atom_ids"]
                event["h_neighbor_atom_ids_after"] = after["nearest_atom_ids"]
                event["h_neighbor_distances_before_A"] = before["nearest_distances"]
                event["h_neighbor_distances_after_A"] = after["nearest_distances"]
                event["bulk0_h_before_count"] = counts_before["bulk0"]
                event["bulk1_h_before_count"] = counts_before["bulk1"]
                event["transition_h_before_count"] = counts_before["transition"]
                event["gb_h_before_count"] = counts_before["gb"]
                event["gb_transition_h_before_count"] = counts_before["gb_transition"]
                event["bulk0_h_after_count"] = counts_after["bulk0"]
                event["bulk1_h_after_count"] = counts_after["bulk1"]
                event["transition_h_after_count"] = counts_after["transition"]
                event["gb_h_after_count"] = counts_after["gb"]
                event["gb_transition_h_after_count"] = counts_after["gb_transition"]


def annotate_barrier_changes(events: list[dict[str, object]]) -> None:
    previous_by_atom: dict[int, float] = {}
    for event in sorted(events, key=lambda e: int(e["step"])):
        event["delta_barrier_from_previous_eV"] = ""
        atom_id = event.get("atom_id")
        barrier = finite_barrier(event.get("barrier_eV"))
        if atom_id == "" or barrier is None:
            continue
        atom_id_i = int(atom_id)
        if atom_id_i in previous_by_atom:
            event["delta_barrier_from_previous_eV"] = float(barrier - previous_by_atom[atom_id_i])
        previous_by_atom[atom_id_i] = barrier


def write_selected_events(path: Path, events: list[dict[str, object]]) -> None:
    fields = [
        "step",
        "atom_id",
        "h_site",
        "n_site",
        "barrier_eV",
        "rate_Hz",
        "dt_s",
        "total_time_s",
        "from_label",
        "to_label",
        "from_region",
        "to_region",
        "from_grain_id",
        "to_grain_id",
        "move_class",
        "dx_A",
        "dy_A",
        "dz_A",
        "abs_dx_A",
        "hop_distance_A",
        "delta_barrier_from_previous_eV",
        "h_neighbors_before_count",
        "h_neighbors_after_count",
        "h_neighbor_sites_before",
        "h_neighbor_sites_after",
        "h_neighbor_atom_ids_before",
        "h_neighbor_atom_ids_after",
        "h_neighbor_distances_before_A",
        "h_neighbor_distances_after_A",
        "bulk0_h_before_count",
        "bulk1_h_before_count",
        "transition_h_before_count",
        "gb_h_before_count",
        "gb_transition_h_before_count",
        "bulk0_h_after_count",
        "bulk1_h_after_count",
        "transition_h_after_count",
        "gb_h_after_count",
        "gb_transition_h_after_count",
        "post_timestep",
        "post_x",
        "post_y",
        "post_z",
        "from_x",
        "from_y",
        "from_z",
        "to_x",
        "to_y",
        "to_z",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)


def finite_barrier(value: object) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def summarize_barriers(events: list[dict[str, object]], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for event in events:
        b = finite_barrier(event.get("barrier_eV"))
        if b is None:
            continue
        key = tuple(event.get(k, "") for k in key_fields)
        groups[key].append(b)
    rows: list[dict[str, object]] = []
    for key, values in sorted(groups.items(), key=lambda kv: (kv[0],)):
        arr = np.asarray(values, dtype=float)
        row = {field: key[i] for i, field in enumerate(key_fields)}
        row.update(
            {
                "count": int(arr.size),
                "mean_barrier_eV": float(arr.mean()),
                "median_barrier_eV": float(np.median(arr)),
                "std_barrier_eV": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min_barrier_eV": float(arr.min()),
                "max_barrier_eV": float(arr.max()),
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: Optional[list[str]] = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_physics_classes(events: list[dict[str, object]]) -> list[str]:
    priority = [
        "bulk_grain_0_entry_to_transition",
        "bulk_grain_1_entry_to_transition",
        "transition_entry_to_gb",
        "grain_boundary_internal",
        "gb_exit_to_transition",
        "transition_exit_to_bulk_grain_0",
        "transition_exit_to_bulk_grain_1",
        "transition_internal",
        "bulk_grain_0_internal",
        "bulk_grain_1_internal",
    ]
    present = {str(event.get("move_class", "")) for event in events}
    ordered = [cls for cls in priority if cls in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def time_binned_move_class_summary(
    events: list[dict[str, object]],
    *,
    n_bins: int,
    classes: Optional[list[str]] = None,
) -> list[dict[str, object]]:
    finite_events = [event for event in events if finite_barrier(event.get("barrier_eV")) is not None]
    if not finite_events:
        return []
    steps = np.asarray([int(event["step"]) for event in finite_events], dtype=float)
    edges = np.linspace(float(steps.min()), float(steps.max()) + 1.0, max(2, int(n_bins) + 1))
    classes = classes or selected_physics_classes(finite_events)
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        members = [event for event in finite_events if left <= int(event["step"]) < right]
        if not members:
            continue
        total = len(members)
        for cls in classes:
            cls_members = [event for event in members if event.get("move_class") == cls]
            if not cls_members:
                continue
            barriers = np.asarray([float(event["barrier_eV"]) for event in cls_members], dtype=float)
            gb_occ = np.asarray([float(event.get("gb_transition_h_before_count", 0)) for event in cls_members], dtype=float)
            rows.append(
                {
                    "step_bin_left": int(left),
                    "step_bin_right": int(right),
                    "mean_step": float(np.mean([int(event["step"]) for event in cls_members])),
                    "move_class": cls,
                    "count": int(len(cls_members)),
                    "fraction_of_selected_moves": float(len(cls_members) / total),
                    "mean_barrier_eV": float(barriers.mean()),
                    "median_barrier_eV": float(np.median(barriers)),
                    "mean_gb_transition_h_before_count": float(gb_occ.mean()) if gb_occ.size else "",
                }
            )
    return rows


def occupancy_binned_move_class_summary(
    events: list[dict[str, object]],
    *,
    classes: Optional[list[str]] = None,
) -> list[dict[str, object]]:
    finite_events = [
        event for event in events
        if finite_barrier(event.get("barrier_eV")) is not None and event.get("gb_transition_h_before_count", "") != ""
    ]
    if not finite_events:
        return []
    classes = classes or selected_physics_classes(finite_events)
    rows: list[dict[str, object]] = []
    for cls in classes:
        cls_events = [event for event in finite_events if event.get("move_class") == cls]
        if not cls_events:
            continue
        occ_values = sorted({int(event["gb_transition_h_before_count"]) for event in cls_events})
        for occ in occ_values:
            members = [event for event in cls_events if int(event["gb_transition_h_before_count"]) == occ]
            barriers = np.asarray([float(event["barrier_eV"]) for event in members], dtype=float)
            rows.append(
                {
                    "gb_transition_h_before_count": occ,
                    "move_class": cls,
                    "count": int(len(members)),
                    "mean_barrier_eV": float(barriers.mean()),
                    "median_barrier_eV": float(np.median(barriers)),
                    "std_barrier_eV": float(barriers.std(ddof=1)) if barriers.size > 1 else 0.0,
                    "min_barrier_eV": float(barriers.min()),
                    "max_barrier_eV": float(barriers.max()),
                }
            )
    return rows


def cumulative_first_passage_summary(events: list[dict[str, object]]) -> list[dict[str, object]]:
    """First event step where each H reaches transition, GB, and bulk_grain_1."""
    first: dict[int, dict[str, object]] = defaultdict(dict)
    for event in sorted(events, key=lambda e: int(e["step"])):
        atom_id = event.get("atom_id")
        if atom_id == "":
            continue
        atom_id_i = int(atom_id)
        to_label = str(event.get("to_label", ""))
        for target, predicate in [
            ("transition", lambda label: label == "transition"),
            ("grain_boundary", lambda label: label == "grain_boundary"),
            ("bulk_grain_1", lambda label: label == "bulk_grain_1"),
        ]:
            if target not in first[atom_id_i] and predicate(to_label):
                first[atom_id_i][target] = int(event["step"])
    rows: list[dict[str, object]] = []
    for atom_id in sorted(first):
        rows.append(
            {
                "atom_id": atom_id,
                "first_transition_step": first[atom_id].get("transition", ""),
                "first_grain_boundary_step": first[atom_id].get("grain_boundary", ""),
                "first_bulk_grain_1_step": first[atom_id].get("bulk_grain_1", ""),
            }
        )
    return rows


def plot_multi_line_summary(
    rows: list[dict[str, object]],
    *,
    x_field: str,
    y_field: str,
    series_field: str,
    out_svg: Path,
    title: str,
    x_label: str,
    y_label: str,
    series_order: Optional[list[str]] = None,
) -> None:
    rows = [row for row in rows if row.get(x_field, "") != "" and row.get(y_field, "") != ""]
    if not rows:
        return
    series_order = series_order or sorted({str(row[series_field]) for row in rows})
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    width, height = 1050, 620
    left, right, top, bottom = 86, 260, 52, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_vals = np.asarray([float(row[x_field]) for row in rows], dtype=float)
    y_vals = np.asarray([float(row[y_field]) for row in rows], dtype=float)

    def pad(lo: float, hi: float) -> tuple[float, float]:
        if lo == hi:
            return lo - 0.5, hi + 0.5
        margin = 0.08 * (hi - lo)
        return lo - margin, hi + margin

    x_lo, x_hi = pad(float(x_vals.min()), float(x_vals.max()))
    y_lo, y_hi = pad(float(y_vals.min()), float(y_vals.max()))

    def sx(x: float) -> float:
        return left + (x - x_lo) * plot_w / (x_hi - x_lo)

    def sy(y: float) -> float:
        return top + plot_h - (y - y_lo) * plot_h / (y_hi - y_lo)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    for tick in np.linspace(x_lo, x_hi, 6):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 40}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick:.3g}</text>')
    for tick in np.linspace(y_lo, y_hi, 6):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.3g}</text>')
    for i, series in enumerate(series_order):
        members = sorted([row for row in rows if str(row[series_field]) == series], key=lambda row: float(row[x_field]))
        if not members:
            continue
        color = palette[i % len(palette)]
        pts = " ".join(f'{sx(float(row[x_field])):.1f},{sy(float(row[y_field])):.1f}' for row in members)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.7" stroke-opacity="0.9"/>')
        for row in members:
            lines.append(f'<circle cx="{sx(float(row[x_field])):.1f}" cy="{sy(float(row[y_field])):.1f}" r="2.5" fill="{color}"><title>{html.escape(series)}; n={row.get("count", "")}</title></circle>')
        ly = top + 18 + i * 20
        lines.append(f'<rect x="{width - right + 20}" y="{ly - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{width - right + 38}" y="{ly}" font-family="Arial, sans-serif" font-size="11">{html.escape(series)}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">{html.escape(x_label)}</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">{html.escape(y_label)}</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def binned_x_motion_summary(events: list[dict[str, object]], *, n_bins: int) -> list[dict[str, object]]:
    finite_events = [
        event for event in events
        if finite_barrier(event.get("barrier_eV")) is not None and "dx_A" in event
    ]
    if not finite_events:
        return []
    dx_values = np.asarray([float(event["dx_A"]) for event in finite_events], dtype=float)
    lo = float(dx_values.min())
    hi = float(dx_values.max())
    edges = np.asarray([lo - 0.5, hi + 0.5]) if lo == hi else np.linspace(lo, hi, max(2, int(n_bins) + 1))

    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == len(edges) - 2:
            members = [event for event in finite_events if left <= float(event["dx_A"]) <= right]
        else:
            members = [event for event in finite_events if left <= float(event["dx_A"]) < right]
        if not members:
            continue
        barriers = np.asarray([float(event["barrier_eV"]) for event in members], dtype=float)
        dx = np.asarray([float(event["dx_A"]) for event in members], dtype=float)
        abs_dx = np.asarray([float(event["abs_dx_A"]) for event in members], dtype=float)
        neighbor_counts = np.asarray(
            [float(event["h_neighbors_before_count"]) for event in members if event.get("h_neighbors_before_count") != ""],
            dtype=float,
        )
        barrier_deltas = np.asarray(
            [
                float(event["delta_barrier_from_previous_eV"])
                for event in members
                if event.get("delta_barrier_from_previous_eV") != ""
            ],
            dtype=float,
        )
        rows.append(
            {
                "dx_bin_left_A": left,
                "dx_bin_right_A": right,
                "count": int(len(members)),
                "mean_dx_A": float(dx.mean()),
                "mean_abs_dx_A": float(abs_dx.mean()),
                "mean_selected_barrier_eV": float(barriers.mean()),
                "median_selected_barrier_eV": float(np.median(barriers)),
                "std_selected_barrier_eV": float(barriers.std(ddof=1)) if barriers.size > 1 else 0.0,
                "mean_h_neighbors_before": float(neighbor_counts.mean()) if neighbor_counts.size else "",
                "mean_delta_barrier_from_previous_eV": float(barrier_deltas.mean()) if barrier_deltas.size else "",
                "median_delta_barrier_from_previous_eV": float(np.median(barrier_deltas)) if barrier_deltas.size else "",
            }
        )
    return rows


def all_h_path_barrier_points(events: list[dict[str, object]]) -> list[dict[str, object]]:
    """Barrier points on each H atom path, using post-move trajectory coordinates."""
    first_x_by_atom: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for event in sorted(events, key=lambda e: int(e["step"])):
        barrier = finite_barrier(event.get("barrier_eV"))
        atom_id = event.get("atom_id")
        if barrier is None or atom_id == "" or event.get("post_x", "") == "":
            continue
        atom_id_i = int(atom_id)
        post_x = float(event["post_x"])
        first_x_by_atom.setdefault(atom_id_i, post_x)
        rows.append(
            {
                "step": int(event["step"]),
                "post_timestep": int(event.get("post_timestep", int(event["step"]) + 1)),
                "atom_id": atom_id_i,
                "x_position_A": post_x,
                "x_displacement_from_first_event_A": post_x - first_x_by_atom[atom_id_i],
                "barrier_eV": float(barrier),
                "site": int(event["n_site"]),
                "site_label": event.get("to_label", ""),
                "site_region": event.get("to_region", ""),
                "site_grain_id": event.get("to_grain_id", ""),
            }
        )
    return rows


def binned_path_barrier_summary(
    points: list[dict[str, object]],
    *,
    x_field: str,
    n_bins: int,
) -> list[dict[str, object]]:
    if not points:
        return []
    x_values = np.asarray([float(point[x_field]) for point in points], dtype=float)
    lo = float(x_values.min())
    hi = float(x_values.max())
    edges = np.asarray([lo - 0.5, hi + 0.5]) if lo == hi else np.linspace(lo, hi, max(2, int(n_bins) + 1))
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == len(edges) - 2:
            members = [point for point in points if left <= float(point[x_field]) <= right]
        else:
            members = [point for point in points if left <= float(point[x_field]) < right]
        if not members:
            continue
        x = np.asarray([float(point[x_field]) for point in members], dtype=float)
        barriers = np.asarray([float(point["barrier_eV"]) for point in members], dtype=float)
        rows.append(
            {
                "x_bin_field": x_field,
                "x_bin_left_A": left,
                "x_bin_right_A": right,
                "count": int(len(members)),
                "mean_x_A": float(x.mean()),
                "mean_barrier_eV": float(barriers.mean()),
                "median_barrier_eV": float(np.median(barriers)),
                "std_barrier_eV": float(barriers.std(ddof=1)) if barriers.size > 1 else 0.0,
                "min_barrier_eV": float(barriers.min()),
                "max_barrier_eV": float(barriers.max()),
            }
        )
    return rows


def fixed_x_barrier_summary(
    points: list[dict[str, object]],
    *,
    x_field: str,
    n_bins: int,
    lx: float,
) -> list[dict[str, object]]:
    if not points:
        return []
    edges = np.linspace(0.0, float(lx), max(2, int(n_bins) + 1))
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == len(edges) - 2:
            members = [point for point in points if left <= float(point[x_field]) <= right]
        else:
            members = [point for point in points if left <= float(point[x_field]) < right]
        if not members:
            continue
        x = np.asarray([float(point[x_field]) for point in members], dtype=float)
        barriers = np.asarray([float(point["barrier_eV"]) for point in members], dtype=float)
        rows.append(
            {
                "x_bin_field": x_field,
                "x_bin_left_A": left,
                "x_bin_right_A": right,
                "count": int(len(members)),
                "mean_x_A": float(x.mean()),
                "mean_barrier_eV": float(barriers.mean()),
                "median_barrier_eV": float(np.median(barriers)),
                "std_barrier_eV": float(barriers.std(ddof=1)) if barriers.size > 1 else 0.0,
                "min_barrier_eV": float(barriers.min()),
                "max_barrier_eV": float(barriers.max()),
            }
        )
    return rows


def timestep_path_barrier_summary(points: list[dict[str, object]], *, n_bins: int) -> list[dict[str, object]]:
    if not points:
        return []
    t_values = np.asarray([int(point["post_timestep"]) for point in points], dtype=float)
    lo = float(t_values.min())
    hi = float(t_values.max())
    edges = np.asarray([lo - 0.5, hi + 0.5]) if lo == hi else np.linspace(lo, hi, max(2, int(n_bins) + 1))
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == len(edges) - 2:
            members = [point for point in points if left <= float(point["post_timestep"]) <= right]
        else:
            members = [point for point in points if left <= float(point["post_timestep"]) < right]
        if not members:
            continue
        t = np.asarray([float(point["post_timestep"]) for point in members], dtype=float)
        x = np.asarray([float(point["x_displacement_from_first_event_A"]) for point in members], dtype=float)
        barriers = np.asarray([float(point["barrier_eV"]) for point in members], dtype=float)
        rows.append(
            {
                "timestep_bin_left": left,
                "timestep_bin_right": right,
                "count": int(len(members)),
                "mean_timestep": float(t.mean()),
                "mean_x_displacement_from_first_event_A": float(x.mean()),
                "mean_barrier_eV": float(barriers.mean()),
                "median_barrier_eV": float(np.median(barriers)),
            }
        )
    return rows


def staggered_step_windows(max_step: int, *, window_steps: int) -> list[tuple[int, int]]:
    """Return inclusive windows: 0-4999, 4999-9999, 9999-14999, ..."""
    window_steps = max(1, int(window_steps))
    if int(max_step) < 0:
        return []
    windows: list[tuple[int, int]] = []
    start = 0
    end = min(window_steps - 1, int(max_step))
    windows.append((start, end))
    while end < int(max_step):
        start = end
        end = min(start + window_steps, int(max_step))
        windows.append((start, end))
    return windows


def write_staggered_x_position_barrier_maps(
    points: list[dict[str, object]],
    out_dir: Path,
    *,
    n_bins: int,
    window_steps: int,
    lx: float,
    region_bands: list[dict[str, object]],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not points:
        return 0
    max_step = max(int(point["step"]) for point in points)
    written = 0
    for start, end in staggered_step_windows(max_step, window_steps=window_steps):
        members = [point for point in points if start <= int(point["step"]) <= end]
        if not members:
            continue
        rows = fixed_x_barrier_summary(members, x_field="x_position_A", n_bins=n_bins, lx=lx)
        for row in rows:
            row["step_window_start"] = start
            row["step_window_end"] = end
        stem = f"avg_barrier_vs_path_x_position_steps_{start:06d}_{end:06d}"
        write_csv(out_dir / f"{stem}.csv", rows)
        plot_xy_summary(
            rows,
            x_field="mean_x_A",
            y_field="mean_barrier_eV",
            out_svg=out_dir / f"{stem}.svg",
            title=f"All H path barriers averaged by x position, steps {start}-{end}",
            x_label="Mean post-move H x position (Angstrom)",
            y_label="Mean selected barrier (eV)",
            x_bounds=(0.0, lx),
            x_bands=region_bands,
        )
        written += 1
    return written


def _empty_allranks_accumulators(n_groups: int, n_bins: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros((n_groups, n_bins), dtype=np.int64),
        "sum_x": np.zeros((n_groups, n_bins), dtype=float),
        "sum_barrier": np.zeros((n_groups, n_bins), dtype=float),
        "sum_barrier2": np.zeros((n_groups, n_bins), dtype=float),
        "min_barrier": np.full((n_groups, n_bins), np.inf, dtype=float),
        "max_barrier": np.full((n_groups, n_bins), -np.inf, dtype=float),
    }


def _allranks_window_indices(step: int, n_windows: int, window_steps: int) -> list[int]:
    if n_windows <= 0:
        return []
    if step < 0:
        return []
    window_steps = max(1, int(window_steps))
    if step < window_steps:
        indices = [0]
    else:
        indices = [min((step - window_steps) // window_steps + 1, n_windows - 1)]
    first_boundary = window_steps - 1
    if step >= first_boundary and (step - first_boundary) % window_steps == 0:
        nxt = (step - first_boundary) // window_steps + 1
        if nxt < n_windows and nxt not in indices:
            indices.append(nxt)
    return indices


def _add_allranks_point(
    acc: dict[str, np.ndarray],
    group_idx: int,
    *,
    x: float,
    barrier: float,
    lx: float,
) -> None:
    n_bins = int(acc["count"].shape[1])
    if n_bins <= 0 or not math.isfinite(x) or not math.isfinite(barrier):
        return
    x_clamped = min(max(float(x), 0.0), float(lx))
    if float(lx) <= 0.0:
        bin_idx = 0
    elif x_clamped >= float(lx):
        bin_idx = n_bins - 1
    else:
        bin_idx = int((x_clamped / float(lx)) * n_bins)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
    acc["count"][group_idx, bin_idx] += 1
    acc["sum_x"][group_idx, bin_idx] += x_clamped
    acc["sum_barrier"][group_idx, bin_idx] += barrier
    acc["sum_barrier2"][group_idx, bin_idx] += barrier * barrier
    acc["min_barrier"][group_idx, bin_idx] = min(acc["min_barrier"][group_idx, bin_idx], barrier)
    acc["max_barrier"][group_idx, bin_idx] = max(acc["max_barrier"][group_idx, bin_idx], barrier)


def _allranks_rows_from_accumulator(
    acc: dict[str, np.ndarray],
    *,
    group_idx: int,
    lx: float,
    window: Optional[tuple[int, int]] = None,
) -> list[dict[str, object]]:
    n_bins = int(acc["count"].shape[1])
    edges = np.linspace(0.0, float(lx), n_bins + 1)
    rows: list[dict[str, object]] = []
    for bin_idx in range(n_bins):
        count = int(acc["count"][group_idx, bin_idx])
        if count <= 0:
            continue
        mean_barrier = float(acc["sum_barrier"][group_idx, bin_idx] / count)
        variance = max(float(acc["sum_barrier2"][group_idx, bin_idx] / count) - mean_barrier * mean_barrier, 0.0)
        row: dict[str, object] = {
            "x_bin_field": "to_x_A",
            "x_bin_left_A": float(edges[bin_idx]),
            "x_bin_right_A": float(edges[bin_idx + 1]),
            "count": count,
            "mean_x_A": float(acc["sum_x"][group_idx, bin_idx] / count),
            "mean_barrier_eV": mean_barrier,
            "std_barrier_eV": float(math.sqrt(variance)),
            "min_barrier_eV": float(acc["min_barrier"][group_idx, bin_idx]),
            "max_barrier_eV": float(acc["max_barrier"][group_idx, bin_idx]),
        }
        if window is not None:
            row["step_window_start"] = int(window[0])
            row["step_window_end"] = int(window[1])
        rows.append(row)
    return rows


def allranks_input_files(inputs: Iterable[RunInputs], out_dir: Path) -> list[Path]:
    split_dir = out_dir / "allranks_atom_id_per_h"
    split_files = sorted(split_dir.glob("H_*_allranks_rates.csv"))
    if split_files:
        return split_files
    return discover_rate_files(inputs)


def write_allranks_x_position_barrier_maps(
    rate_files: Iterable[Path],
    out_dir: Path,
    window_dir: Path,
    *,
    max_step: int,
    n_bins: int,
    window_steps: int,
    lx: float,
    region_bands: list[dict[str, object]],
) -> int:
    windows = staggered_step_windows(max_step, window_steps=window_steps)
    acc = _empty_allranks_accumulators(len(windows) + 1, max(1, int(n_bins)))
    full_idx = len(windows)
    for path in rate_files:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    barrier = float(row.get("barrier_eV", "nan"))
                    x = float(row.get("to_x_A", "nan"))
                    step = int(row.get("step", -1))
                except Exception:
                    continue
                if not math.isfinite(barrier) or not math.isfinite(x):
                    continue
                _add_allranks_point(acc, full_idx, x=x, barrier=barrier, lx=lx)
                for group_idx in _allranks_window_indices(step, len(windows), window_steps):
                    _add_allranks_point(acc, group_idx, x=x, barrier=barrier, lx=lx)

    full_rows = _allranks_rows_from_accumulator(acc, group_idx=full_idx, lx=lx)
    write_csv(out_dir / "allranks_avg_barrier_vs_path_x_position.csv", full_rows)
    plot_xy_summary(
        full_rows,
        x_field="mean_x_A",
        y_field="mean_barrier_eV",
        out_svg=out_dir / "allranks_avg_barrier_vs_path_x_position.svg",
        title="All-ranks candidate barriers averaged by x position",
        x_label="Mean candidate destination x position (Angstrom)",
        y_label="Mean candidate barrier (eV)",
        x_bounds=(0.0, lx),
        x_bands=region_bands,
    )

    window_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for group_idx, window in enumerate(windows):
        rows = _allranks_rows_from_accumulator(acc, group_idx=group_idx, lx=lx, window=window)
        if not rows:
            continue
        start, end = window
        stem = f"allranks_avg_barrier_vs_path_x_position_steps_{start:06d}_{end:06d}"
        write_csv(window_dir / f"{stem}.csv", rows)
        plot_xy_summary(
            rows,
            x_field="mean_x_A",
            y_field="mean_barrier_eV",
            out_svg=window_dir / f"{stem}.svg",
            title=f"All-ranks candidate barriers averaged by x position, steps {start}-{end}",
            x_label="Mean candidate destination x position (Angstrom)",
            y_label="Mean candidate barrier (eV)",
            x_bounds=(0.0, lx),
            x_bands=region_bands,
        )
        written += 1
    return written


def classify_xy_direction(dx: float, dy: float) -> str:
    if abs(float(dx)) >= abs(float(dy)):
        if float(dx) > 0.0:
            return "right"
        if float(dx) < 0.0:
            return "left"
    else:
        if float(dy) > 0.0:
            return "up"
        if float(dy) < 0.0:
            return "down"
    return ""


def directional_bias_rows_from_allranks(
    rate_files: Iterable[Path],
    meta: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    dirs = ("left", "right", "down", "up")
    sums: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: {
        "obs": 0.0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "left_count": 0.0,
        "right_count": 0.0,
        "down_count": 0.0,
        "up_count": 0.0,
        "left_sum": 0.0,
        "right_sum": 0.0,
        "down_sum": 0.0,
        "up_sum": 0.0,
    })
    labels = meta["labels"]
    region_names = meta["region_names"]
    grains = meta["grain_ids"]
    positions = meta["positions"]

    for path in rate_files:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    atom_id = int(row.get("atom_id", ""))
                    h_site = int(row.get("h_site", ""))
                    barrier = float(row.get("barrier_eV", "nan"))
                    dx = float(row.get("dx_A", "nan"))
                    dy = float(row.get("dy_A", "nan"))
                    from_x = float(row.get("from_x_A", positions[h_site, 0]))
                    from_y = float(row.get("from_y_A", positions[h_site, 1]))
                except Exception:
                    continue
                if not math.isfinite(barrier) or not math.isfinite(dx) or not math.isfinite(dy):
                    continue
                direction = classify_xy_direction(dx, dy)
                if not direction:
                    continue
                label = str(labels[h_site])
                key = (atom_id, label)
                item = sums[key]
                item["obs"] += 1.0
                item["sum_x"] += from_x
                item["sum_y"] += from_y
                item[f"{direction}_count"] += 1.0
                item[f"{direction}_sum"] += barrier

    rows: list[dict[str, object]] = []
    for (atom_id, label), item in sorted(sums.items()):
        h_site_for_label = int(np.where(labels == label)[0][0]) if np.any(labels == label) else 0
        means: dict[str, Optional[float]] = {}
        for direction in dirs:
            count = int(item[f"{direction}_count"])
            means[direction] = float(item[f"{direction}_sum"] / count) if count else None
        available = {direction: value for direction, value in means.items() if value is not None}
        if not available:
            continue
        preferred_direction = min(available, key=lambda direction: float(available[direction]))
        preferred_mean = float(available[preferred_direction])
        worst_mean = max(float(value) for value in available.values())
        horizontal_bias = ""
        horizontal_preferred = ""
        if means["left"] is not None and means["right"] is not None:
            horizontal_bias = abs(float(means["left"]) - float(means["right"]))
            horizontal_preferred = "right" if float(means["right"]) < float(means["left"]) else "left"
        vertical_bias = ""
        vertical_preferred = ""
        if means["down"] is not None and means["up"] is not None:
            vertical_bias = abs(float(means["down"]) - float(means["up"]))
            vertical_preferred = "up" if float(means["up"]) < float(means["down"]) else "down"
        rows.append(
            {
                "atom_id": atom_id,
                "from_label": label,
                "from_region": str(region_names[h_site_for_label]),
                "from_grain_id": int(grains[h_site_for_label]),
                "mean_x_A": float(item["sum_x"] / item["obs"]),
                "mean_y_A": float(item["sum_y"] / item["obs"]),
                "candidate_count": int(item["obs"]),
                "left_count": int(item["left_count"]),
                "right_count": int(item["right_count"]),
                "down_count": int(item["down_count"]),
                "up_count": int(item["up_count"]),
                "mean_left_barrier_eV": "" if means["left"] is None else float(means["left"]),
                "mean_right_barrier_eV": "" if means["right"] is None else float(means["right"]),
                "mean_down_barrier_eV": "" if means["down"] is None else float(means["down"]),
                "mean_up_barrier_eV": "" if means["up"] is None else float(means["up"]),
                "preferred_direction": preferred_direction,
                "preferred_mean_barrier_eV": preferred_mean,
                "directional_spread_eV": float(worst_mean - preferred_mean),
                "horizontal_preferred": horizontal_preferred,
                "horizontal_bias_eV": horizontal_bias,
                "vertical_preferred": vertical_preferred,
                "vertical_bias_eV": vertical_bias,
            }
        )
    return rows


def plot_directional_bias(rows: list[dict[str, object]], out_svg: Path, box_lengths: np.ndarray) -> None:
    if not rows:
        return
    width, height = 1180, 430
    left, right, top, bottom = 70, 210, 42, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    lx = float(box_lengths[0])
    ly = float(box_lengths[1])
    max_spread = max(float(row.get("directional_spread_eV", 0.0) or 0.0) for row in rows) or 1.0
    dir_vec = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "down": (0.0, 1.0),
        "up": (0.0, -1.0),
    }
    color_by_region = {
        "bulk_grain_0": "#1f77b4",
        "bulk_grain_1": "#2ca02c",
        "transition": "#ff7f0e",
        "grain_boundary": "#d62728",
        "bulk": "#7f7f7f",
    }

    def sx(x: float) -> float:
        return left + min(max(float(x), 0.0), lx) * plot_w / lx

    def sy(y: float) -> float:
        return top + plot_h - min(max(float(y), 0.0), ly) * plot_h / ly

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="context-stroke"/></marker></defs>',
        f'<text x="{left}" y="26" font-family="Arial, sans-serif" font-size="18" font-weight="700">Directional bias from all-ranks candidate barriers</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    for tick in np.linspace(0.0, lx, 7):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 32}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick:.0f}</text>')
    for tick in np.linspace(0.0, ly, 5):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.0f}</text>')

    for row in rows:
        direction = str(row.get("preferred_direction", ""))
        vx, vy = dir_vec.get(direction, (0.0, 0.0))
        if vx == 0.0 and vy == 0.0:
            continue
        x0 = sx(float(row["mean_x_A"]))
        y0 = sy(float(row["mean_y_A"]))
        strength = float(row.get("directional_spread_eV", 0.0) or 0.0) / max_spread
        length = 8.0 + 24.0 * math.sqrt(max(0.0, strength))
        color = color_by_region.get(str(row.get("from_label", "")), "#444444")
        x1 = x0 + vx * length
        y1 = y0 + vy * length
        opacity = 0.35 + 0.6 * min(max(strength, 0.0), 1.0)
        lines.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
            f'stroke="{color}" stroke-width="1.6" stroke-opacity="{opacity:.3f}" marker-end="url(#arrowhead)">'
            f'<title>H {int(row["atom_id"]):03d}; {html.escape(str(row.get("from_label", "")))}; preferred {direction}; '
            f'spread {float(row.get("directional_spread_eV", 0.0) or 0.0):.4g} eV</title></line>'
        )

    legend_x = width - right + 30
    legend_y = top + 18
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" font-weight="700">origin region</text>')
    for i, (label, color) in enumerate(color_by_region.items()):
        y = legend_y + 20 + i * 19
        lines.append(f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 18}" y2="{y - 4}" stroke="{color}" stroke-width="2.2" marker-end="url(#arrowhead)"/>')
        lines.append(f'<text x="{legend_x + 26}" y="{y}" font-family="Arial, sans-serif" font-size="11">{html.escape(label)}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">x coordinate (Angstrom)</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 18 {top + plot_h / 2:.1f})">y coordinate (Angstrom)</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def _optional_float(value: object) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def read_dict_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def allranks_x_position_outputs_exist(out_dir: Path) -> bool:
    window_dir = out_dir / "allranks_avg_barrier_vs_path_x_position_by_timestep"
    return (
        (out_dir / "allranks_avg_barrier_vs_path_x_position.csv").exists()
        and (out_dir / "allranks_avg_barrier_vs_path_x_position.svg").exists()
        and window_dir.exists()
        and any(window_dir.glob("*.csv"))
        and any(window_dir.glob("*.svg"))
    )


def average_directional_bias_by_region(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    region_order = ["bulk_grain_0", "transition", "grain_boundary", "bulk_grain_1"]
    direction_fields = {
        "left": "mean_left_barrier_eV",
        "right": "mean_right_barrier_eV",
        "down": "mean_down_barrier_eV",
        "up": "mean_up_barrier_eV",
    }
    out: list[dict[str, object]] = []
    for region in region_order:
        members = [row for row in rows if str(row.get("from_label", "")) == region]
        if not members:
            out.append({"from_label": region})
            continue
        result: dict[str, object] = {
            "from_label": region,
            "atom_region_group_count": len(members),
        }
        available: dict[str, float] = {}
        for direction, field in direction_fields.items():
            values = [_optional_float(row.get(field)) for row in members]
            finite = [float(value) for value in values if value is not None]
            result[f"mean_{direction}_barrier_eV"] = float(np.mean(finite)) if finite else ""
            result[f"{direction}_atom_group_count"] = len(finite)
            if finite:
                available[direction] = float(np.mean(finite))
        if available:
            preferred = min(available, key=lambda key: available[key])
            result["preferred_direction"] = preferred
            result["preferred_mean_barrier_eV"] = available[preferred]
            result["directional_spread_eV"] = max(available.values()) - available[preferred]
        out.append(result)
    return out


def plot_master_directional_bias_star(rows: list[dict[str, object]], out_svg: Path) -> None:
    region_order = ["bulk_grain_0", "transition", "grain_boundary", "bulk_grain_1"]
    direction_fields = {
        "left": "mean_left_barrier_eV",
        "right": "mean_right_barrier_eV",
        "down": "mean_down_barrier_eV",
        "up": "mean_up_barrier_eV",
    }
    direction_vectors = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "down": (0.0, 1.0),
        "up": (0.0, -1.0),
    }
    colors = {
        "left": "#1f77b4",
        "right": "#d62728",
        "down": "#2ca02c",
        "up": "#9467bd",
    }
    all_barriers = [
        value
        for row in rows
        for field in direction_fields.values()
        for value in [_optional_float(row.get(field))]
        if value is not None
    ]
    if not all_barriers:
        return
    min_b = min(all_barriers)
    max_b = max(all_barriers)
    span = max(max_b - min_b, 1.0e-12)

    width, height = 980, 720
    panel_w, panel_h = 440, 270
    margin_x, margin_y = 60, 78
    gap_x, gap_y = 42, 58
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="masterStarArrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="context-stroke"/></marker></defs>',
        '<text x="34" y="32" font-family="Arial, sans-serif" font-size="19" font-weight="700">Average directional barrier bias across all H atoms</text>',
        '<text x="34" y="54" font-family="Arial, sans-serif" font-size="12" fill="#555">Each region panel averages per-H directional mean barriers equally. Longer arrows indicate lower average barrier.</text>',
    ]
    row_by_region = {str(row.get("from_label", "")): row for row in rows}
    for idx, region in enumerate(region_order):
        col = idx % 2
        row_idx = idx // 2
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_y + row_idx * (panel_h + gap_y)
        cx = x0 + panel_w / 2.0
        cy = y0 + panel_h / 2.0 + 10.0
        lines.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfbfb" stroke="#cfcfcf"/>')
        lines.append(f'<text x="{x0 + 14}" y="{y0 + 24}" font-family="Arial, sans-serif" font-size="15" font-weight="700">{html.escape(region)}</text>')
        row = row_by_region.get(region, {})
        if not row or not any(_optional_float(row.get(field)) is not None for field in direction_fields.values()):
            lines.append(f'<text x="{cx}" y="{cy}" font-family="Arial, sans-serif" font-size="13" fill="#777" text-anchor="middle">no candidate data</text>')
            continue
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#222"/>')
        for direction, field in direction_fields.items():
            barrier = _optional_float(row.get(field))
            if barrier is None:
                continue
            vx, vy = direction_vectors[direction]
            ease = (max_b - barrier) / span
            length = 34.0 + 76.0 * ease
            x1 = cx + vx * length
            y1 = cy + vy * length
            color = colors[direction]
            lines.append(
                f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                f'stroke="{color}" stroke-width="4" marker-end="url(#masterStarArrow)">'
                f'<title>{direction}: {barrier:.6g} eV</title></line>'
            )
            label_x = x1 + vx * 26.0
            label_y = y1 + vy * 18.0 + (4.0 if direction in {"left", "right"} else 0.0)
            anchor = "middle"
            if direction == "left":
                anchor = "end"
            elif direction == "right":
                anchor = "start"
            lines.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-family="Arial, sans-serif" font-size="11" '
                f'fill="{color}" text-anchor="{anchor}">{direction}: {barrier:.4f}</text>'
            )
        pref = str(row.get("preferred_direction", ""))
        spread = _optional_float(row.get("directional_spread_eV")) or 0.0
        group_count = int(float(row.get("atom_region_group_count", 0) or 0))
        lines.append(f'<text x="{x0 + 14}" y="{y0 + panel_h - 34}" font-family="Arial, sans-serif" font-size="12">preferred: {html.escape(pref)}</text>')
        lines.append(f'<text x="{x0 + 14}" y="{y0 + panel_h - 16}" font-family="Arial, sans-serif" font-size="12">spread: {spread:.4f} eV, atom groups: {group_count}</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def plot_per_atom_directional_bias_stars(rows: list[dict[str, object]], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_atom: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        try:
            by_atom[int(row["atom_id"])].append(row)
        except Exception:
            continue

    region_order = ["bulk_grain_0", "transition", "grain_boundary", "bulk_grain_1"]
    direction_fields = {
        "left": "mean_left_barrier_eV",
        "right": "mean_right_barrier_eV",
        "down": "mean_down_barrier_eV",
        "up": "mean_up_barrier_eV",
    }
    direction_vectors = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "down": (0.0, 1.0),
        "up": (0.0, -1.0),
    }
    colors = {
        "left": "#1f77b4",
        "right": "#d62728",
        "down": "#2ca02c",
        "up": "#9467bd",
    }

    written = 0
    for atom_id, atom_rows in sorted(by_atom.items()):
        row_by_region = {str(row.get("from_label", "")): row for row in atom_rows}
        all_barriers = [
            value
            for row in atom_rows
            for field in direction_fields.values()
            for value in [_optional_float(row.get(field))]
            if value is not None
        ]
        if not all_barriers:
            continue
        min_b = min(all_barriers)
        max_b = max(all_barriers)
        span = max(max_b - min_b, 1.0e-12)

        width, height = 980, 720
        panel_w, panel_h = 440, 270
        margin_x, margin_y = 60, 78
        gap_x, gap_y = 42, 58
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<defs><marker id="starArrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="context-stroke"/></marker></defs>',
            f'<text x="34" y="32" font-family="Arial, sans-serif" font-size="19" font-weight="700">H atom {atom_id:03d}: directional barrier bias by region</text>',
            f'<text x="34" y="54" font-family="Arial, sans-serif" font-size="12" fill="#555">Longer arrows indicate lower average barrier in that direction; labels show mean barrier in eV.</text>',
        ]

        for idx, region in enumerate(region_order):
            col = idx % 2
            row_idx = idx // 2
            x0 = margin_x + col * (panel_w + gap_x)
            y0 = margin_y + row_idx * (panel_h + gap_y)
            cx = x0 + panel_w / 2.0
            cy = y0 + panel_h / 2.0 + 10.0
            lines.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfbfb" stroke="#cfcfcf"/>')
            lines.append(f'<text x="{x0 + 14}" y="{y0 + 24}" font-family="Arial, sans-serif" font-size="15" font-weight="700">{html.escape(region)}</text>')
            bias_row = row_by_region.get(region)
            if bias_row is None:
                lines.append(f'<text x="{cx}" y="{cy}" font-family="Arial, sans-serif" font-size="13" fill="#777" text-anchor="middle">no candidate data</text>')
                continue
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#222"/>')
            for direction, field in direction_fields.items():
                barrier = _optional_float(bias_row.get(field))
                if barrier is None:
                    continue
                vx, vy = direction_vectors[direction]
                ease = (max_b - barrier) / span
                length = 34.0 + 76.0 * ease
                x1 = cx + vx * length
                y1 = cy + vy * length
                color = colors[direction]
                lines.append(
                    f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                    f'stroke="{color}" stroke-width="4" marker-end="url(#starArrow)">'
                    f'<title>{direction}: {barrier:.6g} eV</title></line>'
                )
                label_x = x1 + vx * 26.0
                label_y = y1 + vy * 18.0 + (4.0 if direction in {"left", "right"} else 0.0)
                anchor = "middle"
                if direction == "left":
                    anchor = "end"
                elif direction == "right":
                    anchor = "start"
                lines.append(
                    f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-family="Arial, sans-serif" font-size="11" '
                    f'fill="{color}" text-anchor="{anchor}">{direction}: {barrier:.4f}</text>'
                )
            pref = str(bias_row.get("preferred_direction", ""))
            spread = _optional_float(bias_row.get("directional_spread_eV")) or 0.0
            count = int(float(bias_row.get("candidate_count", 0) or 0))
            lines.append(f'<text x="{x0 + 14}" y="{y0 + panel_h - 34}" font-family="Arial, sans-serif" font-size="12">preferred: {html.escape(pref)}</text>')
            lines.append(f'<text x="{x0 + 14}" y="{y0 + panel_h - 16}" font-family="Arial, sans-serif" font-size="12">spread: {spread:.4f} eV, candidates: {count}</text>')

        lines.append("</svg>")
        (out_dir / f"H_{atom_id:03d}_directional_bias_by_region.svg").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


def plot_xy_summary(
    rows: list[dict[str, object]],
    *,
    x_field: str,
    y_field: str,
    out_svg: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_bounds: Optional[tuple[float, float]] = None,
    y_bounds: Optional[tuple[float, float]] = None,
    x_bands: Optional[list[dict[str, object]]] = None,
) -> None:
    points = [
        (float(row[x_field]), float(row[y_field]), int(row.get("count", 0)))
        for row in rows
        if row.get(x_field, "") != "" and row.get(y_field, "") != ""
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

    def pad(lo: float, hi: float) -> tuple[float, float]:
        if lo == hi:
            return lo - 0.5, hi + 0.5
        margin = 0.08 * (hi - lo)
        return lo - margin, hi + margin

    x_lo, x_hi = x_bounds if x_bounds is not None else pad(float(x_vals.min()), float(x_vals.max()))
    y_lo, y_hi = y_bounds if y_bounds is not None else pad(float(y_vals.min()), float(y_vals.max()))

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
    if x_bands:
        legend_x = left
        legend_y = top - 14
        seen_labels: set[str] = set()
        for band in x_bands:
            b_left = max(x_lo, float(band["left"]))
            b_right = min(x_hi, float(band["right"]))
            if b_right <= x_lo or b_left >= x_hi or b_right <= b_left:
                continue
            x1 = sx(b_left)
            x2 = sx(b_right)
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
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">{html.escape(x_label)}</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">{html.escape(y_label)}</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(rows: list[dict[str, object]], label_fields: tuple[str, ...], out_svg: Path, title: str) -> None:
    if not rows:
        return
    labels = [" -> ".join(str(row[f]) for f in label_fields) for row in rows]
    values = [float(row["mean_barrier_eV"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    counts = [counts[i] for i in order]

    width = 1100
    left = 360
    right = 40
    top = 54
    row_h = 28
    bottom = 52
    height = top + bottom + row_h * len(labels)
    max_value = max(values) if values else 1.0
    plot_w = width - left - right

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left}" y="{height - 15}" font-family="Arial, sans-serif" font-size="13">Mean selected barrier (eV)</text>',
    ]
    for tick in np.linspace(0.0, max_value, 5):
        x = left + (float(tick) / max_value) * plot_w if max_value else left
        lines.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - bottom + 8}" stroke="#dddddd" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 30}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick:.3g}</text>')
    for i, (label, value, count) in enumerate(zip(labels, values, counts)):
        y = top + i * row_h
        bar_w = (value / max_value) * plot_w if max_value else 0.0
        lines.append(f'<text x="{left - 8}" y="{y + 17}" font-family="Arial, sans-serif" font-size="12" text-anchor="end">{html.escape(label)} (n={count})</text>')
        lines.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w:.1f}" height="17" fill="#3579a8"/>')
        lines.append(f'<text x="{left + bar_w + 5:.1f}" y="{y + 18}" font-family="Arial, sans-serif" font-size="11">{value:.4g}</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def write_per_h_sheets(events: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_atom: dict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        atom_id = event.get("atom_id")
        if atom_id == "":
            continue
        by_atom[int(atom_id)].append(event)
    for atom_id, rows in sorted(by_atom.items()):
        write_selected_events(out_dir / f"H_{atom_id:03d}_barrier_events.csv", rows)


def split_rates_allranks_by_atom_id(rate_files: Iterable[Path], out_dir: Path) -> dict[int, int]:
    """Stream rates_allranks.csv into one CSV per atom_id without loading it all."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, object] = {}
    writers: dict[int, csv.DictWriter] = {}
    counts: dict[int, int] = defaultdict(int)
    fields: Optional[list[str]] = None
    try:
        for path in rate_files:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    continue
                if fields is None:
                    fields = list(reader.fieldnames)
                for row in reader:
                    raw_atom_id = str(row.get("atom_id", "")).strip()
                    if not raw_atom_id:
                        continue
                    try:
                        atom_id = int(raw_atom_id)
                    except ValueError:
                        continue
                    if atom_id not in writers:
                        out_path = out_dir / f"H_{atom_id:03d}_allranks_rates.csv"
                        handle = out_path.open("w", encoding="utf-8", newline="")
                        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                        writer.writeheader()
                        handles[atom_id] = handle
                        writers[atom_id] = writer
                    writers[atom_id].writerow(row)
                    counts[atom_id] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return dict(sorted(counts.items()))


def cleanup_postprocess_dir(out_dir: Path, keep_names: set[str], keep_dirs: set[str]) -> None:
    if not out_dir.exists():
        return
    for path in out_dir.iterdir():
        if path.is_dir():
            if path.name not in keep_dirs:
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
        elif path.name not in keep_names:
            path.unlink()


def parse_semicolon_ints(value: object) -> list[int]:
    text = str(value or "")
    return [int(part) for part in text.split(";") if part not in {"", "-1"}]


def parse_semicolon_floats(value: object) -> list[float]:
    text = str(value or "")
    return [float(part) for part in text.split(";") if part != ""]


def write_event_neighbor_detail(
    event: dict[str, object],
    meta: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    labels = meta["labels"]
    names = meta["region_names"]
    grains = meta["grain_ids"]
    fields = [
        "step",
        "phase",
        "selected_atom_id",
        "selected_site",
        "neighbor_rank",
        "neighbor_atom_id",
        "neighbor_site",
        "distance_A",
        "neighbor_label",
        "neighbor_region",
        "neighbor_grain_id",
    ]
    rows: list[dict[str, object]] = []
    for phase, selected_site_key, atom_key, site_key, dist_key in [
        ("before", "h_site", "h_neighbor_atom_ids_before", "h_neighbor_sites_before", "h_neighbor_distances_before_A"),
        ("after", "n_site", "h_neighbor_atom_ids_after", "h_neighbor_sites_after", "h_neighbor_distances_after_A"),
    ]:
        atom_ids = parse_semicolon_ints(event.get(atom_key))
        sites = parse_semicolon_ints(event.get(site_key))
        distances = parse_semicolon_floats(event.get(dist_key))
        for rank, (atom_id, site, distance) in enumerate(zip(atom_ids, sites, distances), start=1):
            rows.append(
                {
                    "step": int(event["step"]),
                    "phase": phase,
                    "selected_atom_id": event.get("atom_id", ""),
                    "selected_site": int(event[selected_site_key]),
                    "neighbor_rank": rank,
                    "neighbor_atom_id": atom_id,
                    "neighbor_site": site,
                    "distance_A": distance,
                    "neighbor_label": str(labels[site]),
                    "neighbor_region": str(names[site]),
                    "neighbor_grain_id": int(grains[site]),
                }
            )
    write_csv(out_path, rows, fields)


def write_atom_path_csv(
    out_path: Path,
    trajectories: Iterable[Path],
    meta: dict[str, np.ndarray],
    atom_id_filter: Optional[int] = None,
) -> list[dict[str, object]]:
    fields = [
        "timestep",
        "atom_id",
        "site",
        "x",
        "y",
        "z",
        "site_label",
        "site_region",
        "site_grain_id",
    ]
    selected_rows: list[dict[str, object]] = []
    labels = meta["labels"]
    names = meta["region_names"]
    grains = meta["grain_ids"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        seen: set[tuple[int, int]] = set()
        for path in trajectories:
            for timestep, rows in iter_lammps_h_frames(path):
                for row in rows:
                    atom_id = int(row["atom_id"])
                    if atom_id_filter is not None and atom_id != atom_id_filter:
                        continue
                    key = (int(timestep), atom_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    site = int(row["site"])
                    out = {
                        "timestep": int(timestep),
                        "atom_id": atom_id,
                        "site": site,
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z"]),
                        "site_label": str(labels[site]),
                        "site_region": str(names[site]),
                        "site_grain_id": int(grains[site]),
                    }
                    writer.writerow(out)
                    selected_rows.append(out)
    return selected_rows


def plot_h_path(path_rows: list[dict[str, object]], event_rows: list[dict[str, object]], atom_id: int, out_svg: Path) -> None:
    if not path_rows:
        return
    t = np.asarray([int(r["timestep"]) for r in path_rows], dtype=int)
    x = np.asarray([float(r["x"]) for r in path_rows], dtype=float)
    y = np.asarray([float(r["y"]) for r in path_rows], dtype=float)
    barriers = [finite_barrier(r.get("barrier_eV")) for r in event_rows]
    event_steps = np.asarray([int(r["step"]) + 1 for r, b in zip(event_rows, barriers) if b is not None], dtype=int)
    event_barriers = np.asarray([b for b in barriers if b is not None], dtype=float)
    event_x = np.interp(event_steps, t, x) if event_steps.size else np.asarray([])
    event_y = np.interp(event_steps, t, y) if event_steps.size else np.asarray([])

    width, height = 1000, 720
    left, right, top, bottom = 70, 40, 55, 55
    panel_gap = 70
    panel_h = (height - top - bottom - panel_gap) / 2.0
    plot_w = width - left - right

    def scale(vals: np.ndarray, lo: float, hi: float, out_lo: float, out_hi: float) -> np.ndarray:
        if hi == lo:
            return np.full_like(vals, (out_lo + out_hi) / 2.0, dtype=float)
        return out_lo + (vals - lo) * (out_hi - out_lo) / (hi - lo)

    def polyline(xs: np.ndarray, ys: np.ndarray) -> str:
        return " ".join(f"{float(px):.1f},{float(py):.1f}" for px, py in zip(xs, ys))

    t_x = scale(t.astype(float), float(t.min()), float(t.max()), left, left + plot_w)
    x_y = scale(x, float(x.min()), float(x.max()), top + panel_h, top)
    xy_x = scale(x, float(x.min()), float(x.max()), left, left + plot_w)
    xy_y = scale(y, float(y.min()), float(y.max()), top + panel_h * 2 + panel_gap, top + panel_h + panel_gap)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">H atom {atom_id} path and selected barriers</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>',
        f'<polyline points="{polyline(t_x, x_y)}" fill="none" stroke="#2b6c9f" stroke-width="1.2"/>',
        f'<text x="20" y="{top + panel_h / 2:.1f}" font-family="Arial, sans-serif" font-size="12" transform="rotate(-90 20 {top + panel_h / 2:.1f})">x (Angstrom)</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{top + panel_h + 34:.1f}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">timestep</text>',
        f'<rect x="{left}" y="{top + panel_h + panel_gap}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>',
        f'<polyline points="{polyline(xy_x, xy_y)}" fill="none" stroke="#444444" stroke-width="0.9"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 16}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">x (Angstrom)</text>',
        f'<text x="20" y="{top + panel_h + panel_gap + panel_h / 2:.1f}" font-family="Arial, sans-serif" font-size="12" transform="rotate(-90 20 {top + panel_h + panel_gap + panel_h / 2:.1f})">y (Angstrom)</text>',
    ]
    if event_steps.size:
        b_min = float(event_barriers.min())
        b_max = float(event_barriers.max())
        ex = scale(event_x, float(x.min()), float(x.max()), left, left + plot_w)
        ey = scale(event_y, float(y.min()), float(y.max()), top + panel_h * 2 + panel_gap, top + panel_h + panel_gap)
        for px, py, b in zip(ex, ey, event_barriers):
            frac = 0.5 if b_max == b_min else (float(b) - b_min) / (b_max - b_min)
            red = int(40 + 200 * frac)
            blue = int(210 - 150 * frac)
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.0" fill="rgb({red},90,{blue})"><title>{b:.6g} eV</title></circle>')
        lines.append(f'<text x="{left}" y="{top + panel_h + panel_gap - 12:.1f}" font-family="Arial, sans-serif" font-size="12">Barrier color: {b_min:.3g} to {b_max:.3g} eV</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def plot_h_path_boxed(
    path_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
    atom_id: int,
    out_svg: Path,
    box_lengths: np.ndarray,
    x_bands: Optional[list[dict[str, object]]] = None,
) -> None:
    """Plot one H path with coordinate axes fixed to the structure box."""
    if not path_rows:
        return
    t = np.asarray([int(r["timestep"]) for r in path_rows], dtype=int)
    x = np.asarray([float(r["x"]) for r in path_rows], dtype=float)
    y = np.asarray([float(r["y"]) for r in path_rows], dtype=float)
    barriers = [finite_barrier(r.get("barrier_eV")) for r in event_rows]
    event_steps = np.asarray([int(r["step"]) + 1 for r, b in zip(event_rows, barriers) if b is not None], dtype=int)
    event_barriers = np.asarray([b for b in barriers if b is not None], dtype=float)
    event_x = np.interp(event_steps, t, x) if event_steps.size else np.asarray([])
    event_y = np.interp(event_steps, t, y) if event_steps.size else np.asarray([])

    width, height = 1000, 720
    left, right, top, bottom = 70, 40, 55, 55
    panel_gap = 70
    panel_h = (height - top - bottom - panel_gap) / 2.0
    plot_w = width - left - right
    lx, ly = float(box_lengths[0]), float(box_lengths[1])

    def scale(vals: np.ndarray, lo: float, hi: float, out_lo: float, out_hi: float) -> np.ndarray:
        if hi == lo:
            return np.full_like(vals, (out_lo + out_hi) / 2.0, dtype=float)
        return out_lo + (vals - lo) * (out_hi - out_lo) / (hi - lo)

    def polyline(xs: np.ndarray, ys: np.ndarray) -> str:
        return " ".join(f"{float(px):.1f},{float(py):.1f}" for px, py in zip(xs, ys))

    x_wrapped = np.mod(x, lx)
    event_x_wrapped = np.mod(event_x, lx) if event_x.size else event_x
    t_x = scale(t.astype(float), float(t.min()), float(t.max()), left, left + plot_w)
    x_y = scale(x_wrapped, 0.0, lx, top + panel_h, top)
    xy_x = scale(x_wrapped, 0.0, lx, left, left + plot_w)
    xy_y = scale(np.mod(y, ly), 0.0, ly, top + panel_h * 2 + panel_gap, top + panel_h + panel_gap)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">H atom {atom_id} path and selected barriers</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>',
        f'<polyline points="{polyline(t_x, x_y)}" fill="none" stroke="#2b6c9f" stroke-width="1.2"/>',
        f'<text x="20" y="{top + panel_h / 2:.1f}" font-family="Arial, sans-serif" font-size="12" transform="rotate(-90 20 {top + panel_h / 2:.1f})">wrapped x, 0-{lx:.3g} A</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{top + panel_h + 34:.1f}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">timestep</text>',
        f'<rect x="{left}" y="{top + panel_h + panel_gap}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>',
        f'<polyline points="{polyline(xy_x, xy_y)}" fill="none" stroke="#444444" stroke-width="0.9"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 16}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">wrapped x, 0-{lx:.3g} A</text>',
        f'<text x="20" y="{top + panel_h + panel_gap + panel_h / 2:.1f}" font-family="Arial, sans-serif" font-size="12" transform="rotate(-90 20 {top + panel_h + panel_gap + panel_h / 2:.1f})">wrapped y, 0-{ly:.3g} A</text>',
    ]
    if x_bands:
        legend_x = left
        legend_y = top - 13
        seen_labels: set[str] = set()
        for band in x_bands:
            b_left = max(0.0, float(band["left"]))
            b_right = min(lx, float(band["right"]))
            if b_right <= b_left:
                continue
            color = str(band["color"])
            label = str(band["label"])
            # Top panel: x is the vertical coordinate, so region bands are horizontal.
            y1 = float(scale(np.asarray([b_right]), 0.0, lx, top + panel_h, top)[0])
            y2 = float(scale(np.asarray([b_left]), 0.0, lx, top + panel_h, top)[0])
            lines.append(f'<rect x="{left}" y="{y1:.1f}" width="{plot_w}" height="{max(0.0, y2 - y1):.1f}" fill="{color}" fill-opacity="0.23"/>')
            # Bottom panel: x is the horizontal coordinate.
            x1 = float(scale(np.asarray([b_left]), 0.0, lx, left, left + plot_w)[0])
            x2 = float(scale(np.asarray([b_right]), 0.0, lx, left, left + plot_w)[0])
            lines.append(f'<rect x="{x1:.1f}" y="{top + panel_h + panel_gap}" width="{max(0.0, x2 - x1):.1f}" height="{panel_h}" fill="{color}" fill-opacity="0.23"/>')
            if label not in seen_labels:
                seen_labels.add(label)
                lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="10" fill="{color}" fill-opacity="0.7"/>')
                lines.append(f'<text x="{legend_x + 16}" y="{legend_y}" font-family="Arial, sans-serif" font-size="11">{html.escape(label)}</text>')
                legend_x += 18 + len(label) * 6.2 + 18
    if event_steps.size:
        b_min = float(event_barriers.min())
        b_max = float(event_barriers.max())
        ex = scale(event_x_wrapped, 0.0, lx, left, left + plot_w)
        ey = scale(np.mod(event_y, ly), 0.0, ly, top + panel_h * 2 + panel_gap, top + panel_h + panel_gap)
        for px, py, b in zip(ex, ey, event_barriers):
            frac = 0.5 if b_max == b_min else (float(b) - b_min) / (b_max - b_min)
            red = int(40 + 200 * frac)
            blue = int(210 - 150 * frac)
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.0" fill="rgb({red},90,{blue})"><title>{b:.6g} eV</title></circle>')
        lines.append(f'<text x="{left}" y="{top + panel_h + panel_gap - 12:.1f}" font-family="Arial, sans-serif" font-size="12">Barrier color: {b_min:.3g} to {b_max:.3g} eV</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def plot_barrier_vs_x_displacement_points(
    points_by_atom: dict[int, list[dict[str, object]]],
    atom_ids: list[int],
    out_svg: Path,
    *,
    title: str,
    box_lengths: np.ndarray,
) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 82, 180, 52, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    selected = {atom_id: points_by_atom.get(atom_id, []) for atom_id in atom_ids if points_by_atom.get(atom_id)}
    if not selected:
        return
    xs = np.asarray([float(p["x_displacement_from_first_event_A"]) for rows in selected.values() for p in rows], dtype=float)
    ys = np.asarray([float(p["barrier_eV"]) for rows in selected.values() for p in rows], dtype=float)
    lx = float(box_lengths[0])
    x_lo = min(-0.05 * lx, float(xs.min()))
    x_hi = max(1.05 * lx, float(xs.max()))
    y_lo = 0.0
    y_hi = max(float(ys.max()) * 1.08, 0.01)

    def sx(x: float) -> float:
        return left + (x - x_lo) * plot_w / (x_hi - x_lo)

    def sy(y: float) -> float:
        return top + plot_h - (y - y_lo) * plot_h / (y_hi - y_lo)

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    region_labels = ["bulk_grain_0", "bulk_grain_1", "transition", "grain_boundary"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    for tick in np.linspace(x_lo, x_hi, 7):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 40}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick:.3g}</text>')
    for tick in np.linspace(y_lo, y_hi, 6):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.3g}</text>')
    for i, atom_id in enumerate(atom_ids):
        rows = selected.get(atom_id, [])
        if not rows:
            continue
        color = palette[i % len(palette)]
        pts = " ".join(
            f'{sx(float(p["x_displacement_from_first_event_A"])):.1f},{sy(float(p["barrier_eV"])):.1f}'
            for p in rows
        )
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.0" stroke-opacity="0.75"/>')
        for p in rows[:: max(1, len(rows) // 450)]:
            fill = region_color(p.get("site_label", ""))
            lines.append(
                f'<circle cx="{sx(float(p["x_displacement_from_first_event_A"])):.1f}" '
                f'cy="{sy(float(p["barrier_eV"])):.1f}" r="2.2" fill="{fill}" '
                f'stroke="{color}" stroke-width="0.7" fill-opacity="0.88">'
                f'<title>H {atom_id:03d}; {html.escape(str(p.get("site_label", "")))}; {float(p["barrier_eV"]):.6g} eV</title></circle>'
            )
        ly = top + 20 + i * 20
        lines.append(f'<rect x="{width - right + 22}" y="{ly - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{width - right + 40}" y="{ly}" font-family="Arial, sans-serif" font-size="12">H {atom_id:03d}</text>')
    region_legend_y = top + 20 + max(len(atom_ids), 1) * 20 + 18
    lines.append(f'<text x="{width - right + 22}" y="{region_legend_y}" font-family="Arial, sans-serif" font-size="12" font-weight="700">point region</text>')
    for j, label in enumerate(region_labels):
        y = region_legend_y + 18 + j * 18
        color = region_color(label)
        lines.append(f'<circle cx="{width - right + 28}" cy="{y - 4}" r="5" fill="{color}" stroke="#555555" stroke-width="0.5"/>')
        lines.append(f'<text x="{width - right + 40}" y="{y}" font-family="Arial, sans-serif" font-size="11">{html.escape(label)}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">x displacement from first selected event (Angstrom); x-cell length {lx:.3g} A</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">selected barrier (eV)</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def first_gb_entry_event(events: list[dict[str, object]]) -> Optional[dict[str, object]]:
    gb_events = [
        event for event in events
        if event.get("atom_id") != "" and str(event.get("to_label", "")) == "grain_boundary"
    ]
    if not gb_events:
        return None
    return min(gb_events, key=lambda event: int(event["step"]))


def current_site_for_atom_events(atom_events: list[dict[str, object]], step: int) -> Optional[int]:
    if not atom_events:
        return None
    if step <= int(atom_events[0]["step"]):
        return int(atom_events[0]["h_site"])
    lo = 0
    hi = len(atom_events) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if int(atom_events[mid]["step"]) <= step:
            lo = mid + 1
        else:
            hi = mid - 1
    if int(atom_events[hi]["step"]) == int(step):
        return int(atom_events[hi]["h_site"])
    return int(atom_events[hi]["n_site"])


def extract_atom_candidate_barriers(
    rate_files: Iterable[Path],
    atom_events: list[dict[str, object]],
    meta: dict[str, np.ndarray],
    out_csv: Path,
) -> list[dict[str, object]]:
    """Write all candidate rate rows belonging to one tracked H atom."""
    atom_events = sorted(atom_events, key=lambda event: int(event["step"]))
    selected_by_step = {int(event["step"]): int(event["n_site"]) for event in atom_events}
    labels = meta["labels"]
    names = meta["region_names"]
    grains = meta["grain_ids"]
    fields = [
        "step",
        "atom_id",
        "h_site",
        "n_site",
        "barrier_eV",
        "rate_Hz",
        "source",
        "env_kind",
        "status",
        "candidate_selected",
        "from_label",
        "to_label",
        "from_region",
        "to_region",
        "from_grain_id",
        "to_grain_id",
        "move_class",
    ]
    rows_for_plot: list[dict[str, object]] = []
    atom_id = int(atom_events[0]["atom_id"]) if atom_events else -1
    with out_csv.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fields)
        writer.writeheader()
        for path in rate_files:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        step = int(row["step"])
                        h_site = int(row["h_site"])
                        n_site = int(row["n_site"])
                    except Exception:
                        continue
                    current_site = current_site_for_atom_events(atom_events, step)
                    if current_site is None or h_site != current_site:
                        continue
                    barrier = finite_barrier(row.get("barrier_eV"))
                    if barrier is None:
                        continue
                    from_label = str(labels[h_site])
                    to_label = str(labels[n_site])
                    from_grain = int(grains[h_site])
                    to_grain = int(grains[n_site])
                    out = {
                        "step": step,
                        "atom_id": atom_id,
                        "h_site": h_site,
                        "n_site": n_site,
                        "barrier_eV": float(barrier),
                        "rate_Hz": row.get("rate_Hz", ""),
                        "source": row.get("source", ""),
                        "env_kind": row.get("env_kind", ""),
                        "status": row.get("status", ""),
                        "candidate_selected": int(selected_by_step.get(step, -1) == n_site),
                        "from_label": from_label,
                        "to_label": to_label,
                        "from_region": str(names[h_site]),
                        "to_region": str(names[n_site]),
                        "from_grain_id": from_grain,
                        "to_grain_id": to_grain,
                        "move_class": classify_move(from_label, to_label, from_grain, to_grain),
                    }
                    writer.writerow(out)
                    rows_for_plot.append(out)
    return rows_for_plot


def candidate_barrier_timestep_summary(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[int(row["step"])].append(row)
    rows: list[dict[str, object]] = []
    for step in sorted(grouped):
        members = grouped[step]
        barriers = np.asarray([float(row["barrier_eV"]) for row in members], dtype=float)
        selected = [row for row in members if int(row.get("candidate_selected", 0)) == 1]
        rows.append(
            {
                "step": step,
                "candidate_count": int(len(members)),
                "min_candidate_barrier_eV": float(barriers.min()),
                "mean_candidate_barrier_eV": float(barriers.mean()),
                "median_candidate_barrier_eV": float(np.median(barriers)),
                "max_candidate_barrier_eV": float(barriers.max()),
                "selected_barrier_eV": float(selected[0]["barrier_eV"]) if selected else "",
                "selected_move_class": selected[0]["move_class"] if selected else "",
                "selected_to_label": selected[0]["to_label"] if selected else "",
            }
        )
    return rows


def plot_candidate_barriers_vs_timestep(
    summary_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    first_event: dict[str, object],
    out_svg: Path,
    *,
    title: str,
    step_window: Optional[tuple[int, int]] = None,
    scatter_all_candidates: bool = False,
) -> None:
    if step_window is not None:
        lo, hi = step_window
        summary_rows = [row for row in summary_rows if lo <= int(row["step"]) <= hi]
        candidate_rows = [row for row in candidate_rows if lo <= int(row["step"]) <= hi]
    if not summary_rows:
        return
    width, height = 1050, 620
    left, right, top, bottom = 86, 210, 52, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    steps = np.asarray([int(row["step"]) for row in summary_rows], dtype=float)
    y_values = []
    for row in summary_rows:
        y_values.extend([float(row["min_candidate_barrier_eV"]), float(row["mean_candidate_barrier_eV"]), float(row["max_candidate_barrier_eV"])])
        if row.get("selected_barrier_eV", "") != "":
            y_values.append(float(row["selected_barrier_eV"]))
    if scatter_all_candidates:
        y_values.extend([float(row["barrier_eV"]) for row in candidate_rows])
    x_lo = float(steps.min())
    x_hi = float(steps.max()) if float(steps.max()) > x_lo else x_lo + 1.0
    y_lo = 0.0
    y_hi = max(float(max(y_values)) * 1.08, 0.01)
    first_step = int(first_event["step"])

    def sx(x: float) -> float:
        return left + (x - x_lo) * plot_w / (x_hi - x_lo)

    def sy(y: float) -> float:
        return top + plot_h - (y - y_lo) * plot_h / (y_hi - y_lo)

    def points(field: str) -> str:
        return " ".join(f'{sx(int(row["step"])):.1f},{sy(float(row[field])):.1f}' for row in summary_rows)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    for tick in np.linspace(x_lo, x_hi, 6):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 40}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick:.0f}</text>')
    for tick in np.linspace(y_lo, y_hi, 6):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.3g}</text>')
    if scatter_all_candidates:
        for row in candidate_rows:
            fill = region_color(row.get("to_label", ""))
            stroke = "#d62728" if int(row.get("candidate_selected", 0)) == 1 else "#666666"
            radius = 3.5 if int(row.get("candidate_selected", 0)) == 1 else 1.7
            lines.append(
                f'<circle cx="{sx(int(row["step"])):.1f}" cy="{sy(float(row["barrier_eV"])):.1f}" r="{radius}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="0.65" fill-opacity="0.75">'
                f'<title>step {int(row["step"])}; {html.escape(str(row.get("move_class", "")))}; '
                f'to {html.escape(str(row.get("to_label", "")))}; {float(row["barrier_eV"]):.6g} eV</title></circle>'
            )
    lines.append(f'<polyline points="{points("min_candidate_barrier_eV")}" fill="none" stroke="#2ca02c" stroke-width="1.3"/>')
    lines.append(f'<polyline points="{points("mean_candidate_barrier_eV")}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>')
    lines.append(f'<polyline points="{points("max_candidate_barrier_eV")}" fill="none" stroke="#9467bd" stroke-width="1.3"/>')
    selected_rows = [row for row in summary_rows if row.get("selected_barrier_eV", "") != ""]
    if selected_rows:
        selected_points = " ".join(f'{sx(int(row["step"])):.1f},{sy(float(row["selected_barrier_eV"])):.1f}' for row in selected_rows)
        lines.append(f'<polyline points="{selected_points}" fill="none" stroke="#d62728" stroke-width="1.4" stroke-opacity="0.8"/>')
        for row in selected_rows:
            lines.append(f'<circle cx="{sx(int(row["step"])):.1f}" cy="{sy(float(row["selected_barrier_eV"])):.1f}" r="2.7" fill="{region_color(row.get("selected_to_label", ""))}" stroke="#d62728" stroke-width="0.8"/>')
    if x_lo <= first_step <= x_hi:
        x = sx(first_step)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#d62728" stroke-width="2.0" stroke-dasharray="6 4"/>')
        lines.append(f'<text x="{x + 6:.1f}" y="{top + 16}" font-family="Arial, sans-serif" font-size="12" fill="#d62728">first GB entry step {first_step}</text>')
    legend_x = width - right + 24
    legend_y = top + 18
    for i, (label, color) in enumerate([("min candidate", "#2ca02c"), ("mean candidate", "#1f77b4"), ("max candidate", "#9467bd"), ("selected", "#d62728")]):
        y = legend_y + i * 20
        lines.append(f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 18}" y2="{y - 4}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{legend_x + 25}" y="{y}" font-family="Arial, sans-serif" font-size="12">{label}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">KMC step</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">candidate barrier (eV)</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def plot_first_gb_atom_barrier_vs_timestep(
    event_rows: list[dict[str, object]],
    first_event: dict[str, object],
    out_svg: Path,
    *,
    title: str,
    step_window: Optional[tuple[int, int]] = None,
) -> None:
    rows = [
        row for row in sorted(event_rows, key=lambda event: int(event["step"]))
        if finite_barrier(row.get("barrier_eV")) is not None
    ]
    if step_window is not None:
        lo, hi = step_window
        rows = [row for row in rows if lo <= int(row["step"]) <= hi]
    if not rows:
        return

    width, height = 1050, 610
    left, right, top, bottom = 86, 210, 52, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    steps = np.asarray([int(row["step"]) for row in rows], dtype=float)
    barriers = np.asarray([float(row["barrier_eV"]) for row in rows], dtype=float)
    x_lo = float(steps.min())
    x_hi = float(steps.max()) if float(steps.max()) > x_lo else x_lo + 1.0
    y_lo = 0.0
    y_hi = max(float(barriers.max()) * 1.08, 0.01)
    first_step = int(first_event["step"])

    def sx(x: float) -> float:
        return left + (x - x_lo) * plot_w / (x_hi - x_lo)

    def sy(y: float) -> float:
        return top + plot_h - (y - y_lo) * plot_h / (y_hi - y_lo)

    region_labels = ["bulk_grain_0", "bulk_grain_1", "transition", "grain_boundary"]
    class_colors = {
        "transition_entry_to_gb": "#d62728",
        "gb_exit_to_transition": "#9467bd",
        "grain_boundary_internal": "#1f77b4",
        "bulk_grain_0_entry_to_transition": "#ff7f0e",
        "bulk_grain_1_entry_to_transition": "#ff7f0e",
        "transition_exit_to_bulk_grain_0": "#2ca02c",
        "transition_exit_to_bulk_grain_1": "#2ca02c",
    }
    points = " ".join(f'{sx(int(row["step"])):.1f},{sy(float(row["barrier_eV"])):.1f}' for row in rows)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbbbbb"/>',
    ]
    for tick in np.linspace(x_lo, x_hi, 6):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - 40}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle">{tick:.0f}</text>')
    for tick in np.linspace(y_lo, y_hi, 6):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end">{tick:.3g}</text>')
    lines.append(f'<polyline points="{points}" fill="none" stroke="#555555" stroke-width="0.9" stroke-opacity="0.55"/>')
    if x_lo <= first_step <= x_hi:
        x = sx(first_step)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#d62728" stroke-width="2.0" stroke-dasharray="6 4"/>')
        lines.append(f'<text x="{x + 6:.1f}" y="{top + 16}" font-family="Arial, sans-serif" font-size="12" fill="#d62728">first GB entry step {first_step}</text>')
    for row in rows:
        move_class = str(row.get("move_class", ""))
        color = class_colors.get(move_class, "#333333")
        fill = region_color(row.get("to_label", ""))
        r = 4.2 if int(row["step"]) == first_step else 2.4
        lines.append(
            f'<circle cx="{sx(int(row["step"])):.1f}" cy="{sy(float(row["barrier_eV"])):.1f}" r="{r}" '
            f'fill="{fill}" stroke="{color}" stroke-width="0.9" fill-opacity="0.9">'
            f'<title>step {int(row["step"])}; {html.escape(move_class)}; to {html.escape(str(row.get("to_label", "")))}; {float(row["barrier_eV"]):.6g} eV</title></circle>'
        )
    legend_x = width - right + 24
    legend_y = top + 18
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" font-weight="700">fill: destination region</text>')
    for i, label in enumerate(region_labels):
        y = legend_y + 18 + i * 18
        lines.append(f'<circle cx="{legend_x + 7}" cy="{y - 4}" r="5" fill="{region_color(label)}" stroke="#555555" stroke-width="0.5"/>')
        lines.append(f'<text x="{legend_x + 20}" y="{y}" font-family="Arial, sans-serif" font-size="11">{html.escape(label)}</text>')
    y = legend_y + 18 + len(region_labels) * 18 + 14
    lines.append(f'<text x="{legend_x}" y="{y}" font-family="Arial, sans-serif" font-size="12" font-weight="700">stroke: selected move class</text>')
    for j, (move_class, color) in enumerate(class_colors.items()):
        yy = y + 18 + j * 17
        lines.append(f'<line x1="{legend_x}" y1="{yy - 4}" x2="{legend_x + 14}" y2="{yy - 4}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{legend_x + 20}" y="{yy}" font-family="Arial, sans-serif" font-size="10">{html.escape(move_class)}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" font-family="Arial, sans-serif" font-size="13" text-anchor="middle">KMC step</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 20 {top + plot_h / 2:.1f})">selected barrier (eV)</text>')
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="one-crystal-exp-master-cache-run", type=Path)
    parser.add_argument("--previous-run", type=Path, default=None, help="Optional original run directory to prepend.")
    parser.add_argument("--map-dir", type=Path, default=None, help="Directory containing sigma5 site map and region npz files.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--x-bins", type=int, default=25, help="Number of x-position bins for avg_barrier_vs_path_x_position.")
    parser.add_argument("--time-bins", type=int, default=25, help="Number of timestep bins for move_class_fraction_vs_time.")
    parser.add_argument("--path-window-steps", type=int, default=5000, help="Inclusive staggered timestep window size for x-position barrier maps.")
    parser.add_argument("--skip-allranks-split", action="store_true", help="Skip splitting rates_allranks.csv into per-H CSV files.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "postprocess")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir = (args.map_dir or (run_dir / "kmc_map_inputs")).resolve()

    keep_dirs = {
        "selected_atom_id_per_h",
        "allranks_atom_id_per_h",
        "avg_barrier_vs_path_x_position_by_timestep",
        "allranks_avg_barrier_vs_path_x_position_by_timestep",
        "directional_bias_per_h",
    }
    keep_names = {
        "avg_barrier_vs_path_x_position.csv",
        "avg_barrier_vs_path_x_position.svg",
        "max_selected_barrier_vs_path_x_displacement.csv",
        "max_selected_barrier_vs_path_x_displacement.svg",
        "allranks_avg_barrier_vs_path_x_position.csv",
        "allranks_avg_barrier_vs_path_x_position.svg",
        "directional_bias_by_atom_region.csv",
        "directional_bias_by_atom_region.svg",
        "directional_bias_master_average_by_region.csv",
        "directional_bias_master_average_by_region.svg",
        "move_class_fraction_vs_time.csv",
        "move_class_fraction_vs_time.svg",
    }
    cleanup_postprocess_dir(out_dir, keep_names=keep_names, keep_dirs=keep_dirs)

    inputs = discover_inputs(run_dir, args.previous_run)
    meta = load_site_metadata(map_dir)
    region_bands = x_region_bands(meta)
    selected_csvs = [
        path
        for inp in inputs
        for path in [
            _first_existing(inp.root / "diagnostics" / "kmc_selected_events.csv", inp.root / "kmc_selected_events.csv")
        ]
        if path is not None
    ]
    if selected_csvs:
        events = []
        for selected_csv in selected_csvs:
            events.extend(read_selected_events_csv(selected_csv))
        events.sort(key=lambda event: int(event["step"]))
    else:
        events = read_selected_events([inp.log for inp in inputs])

    totals = read_timestep_totals([inp.timestep_csv for inp in inputs])
    for event in events:
        annotate_event(event, meta)
        if event.get("total_time_s", "") == "" or not math.isfinite(float(event.get("total_time_s", float("nan")))):
            event["total_time_s"] = totals.get(int(event["step"]), "")
        if event.get("atom_id", "") == "":
            # Older logs do not carry atom IDs; fall back to trajectory only when needed.
            annotate_events_from_trajectory(
                events,
                [inp.trajectory for inp in inputs],
                meta,
                neighbor_cutoff_a=6.0,
                neighbor_k=8,
            )
            break

    annotate_barrier_changes(events)

    physics_classes = selected_physics_classes(events)
    time_class_rows = time_binned_move_class_summary(events, n_bins=args.time_bins, classes=physics_classes)
    write_csv(out_dir / "move_class_fraction_vs_time.csv", time_class_rows)
    plot_multi_line_summary(
        time_class_rows,
        x_field="mean_step",
        y_field="fraction_of_selected_moves",
        series_field="move_class",
        out_svg=out_dir / "move_class_fraction_vs_time.svg",
        title="Selected move-class fraction versus KMC step",
        x_label="KMC step",
        y_label="fraction of selected moves in bin",
        series_order=physics_classes,
    )

    path_points = all_h_path_barrier_points(events)
    box_lengths = meta["box_lengths"]
    lx = float(box_lengths[0])
    points_by_atom: dict[int, list[dict[str, object]]] = defaultdict(list)
    for point in path_points:
        points_by_atom[int(point["atom_id"])].append(point)
    for rows in points_by_atom.values():
        rows.sort(key=lambda row: int(row["post_timestep"]))
    all_atoms = sorted(points_by_atom)

    x_position_rows = fixed_x_barrier_summary(path_points, x_field="x_position_A", n_bins=args.x_bins, lx=lx)
    write_csv(out_dir / "avg_barrier_vs_path_x_position.csv", x_position_rows)
    plot_xy_summary(
        x_position_rows,
        x_field="mean_x_A",
        y_field="mean_barrier_eV",
        out_svg=out_dir / "avg_barrier_vs_path_x_position.svg",
        title="All H selected barriers averaged by x position",
        x_label="Mean post-move H x position (Angstrom)",
        y_label="Mean selected barrier (eV)",
        x_bounds=(0.0, lx),
        x_bands=region_bands,
    )

    x_window_dir = out_dir / "avg_barrier_vs_path_x_position_by_timestep"
    x_window_count = write_staggered_x_position_barrier_maps(
        path_points,
        x_window_dir,
        n_bins=args.x_bins,
        window_steps=args.path_window_steps,
        lx=lx,
        region_bands=region_bands,
    )

    displacement_values = [
        float(point["x_displacement_from_first_event_A"])
        for point in path_points
    ]
    displacement_bounds = (
        min(0.0, min(displacement_values) if displacement_values else 0.0),
        max(lx, max(displacement_values) if displacement_values else lx),
    )
    max_x_displacement_rows = binned_path_barrier_summary(
        path_points,
        x_field="x_displacement_from_first_event_A",
        n_bins=args.x_bins,
    )
    write_csv(out_dir / "max_selected_barrier_vs_path_x_displacement.csv", max_x_displacement_rows)
    plot_xy_summary(
        max_x_displacement_rows,
        x_field="mean_x_A",
        y_field="max_barrier_eV",
        out_svg=out_dir / "max_selected_barrier_vs_path_x_displacement.svg",
        title="Maximum selected barrier versus H x displacement",
        x_label="Mean H x displacement from first selected event (Angstrom)",
        y_label="Maximum selected barrier (eV)",
        x_bounds=displacement_bounds,
    )

    if allranks_x_position_outputs_exist(out_dir):
        allranks_window_count = len(list((out_dir / "allranks_avg_barrier_vs_path_x_position_by_timestep").glob("*.csv")))
    else:
        allranks_window_count = write_allranks_x_position_barrier_maps(
            allranks_input_files(inputs, out_dir),
            out_dir,
            out_dir / "allranks_avg_barrier_vs_path_x_position_by_timestep",
            max_step=max((int(point["step"]) for point in path_points), default=0),
            n_bins=args.x_bins,
            window_steps=args.path_window_steps,
            lx=lx,
            region_bands=region_bands,
        )
    directional_csv = out_dir / "directional_bias_by_atom_region.csv"
    directional_bias_rows = read_dict_csv(directional_csv)
    if not directional_bias_rows:
        directional_bias_rows = directional_bias_rows_from_allranks(
            allranks_input_files(inputs, out_dir),
            meta,
        )
        write_csv(directional_csv, directional_bias_rows)
    plot_directional_bias(
        directional_bias_rows,
        out_dir / "directional_bias_by_atom_region.svg",
        box_lengths,
    )
    master_bias_rows = average_directional_bias_by_region(directional_bias_rows)
    write_csv(out_dir / "directional_bias_master_average_by_region.csv", master_bias_rows)
    plot_master_directional_bias_star(
        master_bias_rows,
        out_dir / "directional_bias_master_average_by_region.svg",
    )
    per_h_bias_count = plot_per_atom_directional_bias_stars(
        directional_bias_rows,
        out_dir / "directional_bias_per_h",
    )

    selected_dir = out_dir / "selected_atom_id_per_h"
    write_per_h_sheets(events, selected_dir)

    split_counts: dict[int, int] = {}
    if not args.skip_allranks_split:
        split_counts = split_rates_allranks_by_atom_id(discover_rate_files(inputs), out_dir / "allranks_atom_id_per_h")

    assigned = sum(1 for e in events if e.get("atom_id") != "")
    print(f"Processed {len(events)} selected events")
    print(f"Assigned atom IDs for {assigned}/{len(events)} selected events")
    print(f"Wrote selected per-H sheets to {selected_dir}")
    print(f"Wrote {x_window_count} staggered x-position barrier maps to {x_window_dir}")
    print(f"Wrote {allranks_window_count} staggered all-ranks x-position barrier maps")
    print(f"Wrote directional bias study for {len(directional_bias_rows)} atom-region groups")
    print(f"Wrote {per_h_bias_count} per-H directional bias star plots")
    if args.skip_allranks_split:
        print("Skipped all-ranks per-H split")
    else:
        print(f"Wrote all-ranks per-H sheets for {len(split_counts)} atoms to {out_dir / 'allranks_atom_id_per_h'}")
    print(f"Kept requested plots/CSVs in {out_dir}")


if __name__ == "__main__":
    main()
