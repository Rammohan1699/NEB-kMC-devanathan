#!/usr/bin/env python3
"""Summarize selected KMC barriers by site-region zones."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np


def finite_float(value: object) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


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


def add(groups: dict[tuple[object, ...], list[float]], key: tuple[object, ...], value: float) -> None:
    groups[key].append(value)


def write_summary(path: Path, key_fields: list[str], groups: dict[tuple[object, ...], list[float]]) -> None:
    fields = key_fields + [
        "count",
        "mean_barrier_eV",
        "median_barrier_eV",
        "std_barrier_eV",
        "min_barrier_eV",
        "max_barrier_eV",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(groups.items()):
            arr = np.asarray(values, dtype=float)
            row = {field: key[idx] for idx, field in enumerate(key_fields)}
            row.update(
                {
                    "count": int(arr.size),
                    "mean_barrier_eV": float(arr.mean()),
                    "median_barrier_eV": float(median(values)),
                    "std_barrier_eV": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "min_barrier_eV": float(arr.min()),
                    "max_barrier_eV": float(arr.max()),
                }
            )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-events", type=Path, required=True)
    parser.add_argument("--region-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regions = np.load(args.region_file, allow_pickle=True)
    labels = np.asarray(regions["site_selection_label"], dtype=object)
    names = np.asarray(regions["site_region_name"], dtype=object)
    grains = np.asarray(regions["site_grain_id"], dtype=int)

    by_label_pair: dict[tuple[object, ...], list[float]] = defaultdict(list)
    by_region_pair: dict[tuple[object, ...], list[float]] = defaultdict(list)
    by_move_class: dict[tuple[object, ...], list[float]] = defaultdict(list)
    by_from_label: dict[tuple[object, ...], list[float]] = defaultdict(list)
    by_to_label: dict[tuple[object, ...], list[float]] = defaultdict(list)

    with args.selected_events.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            barrier = finite_float(row.get("barrier_eV"))
            if barrier is None:
                continue
            try:
                h_site = int(row["h_site"])
                n_site = int(row["n_site"])
            except Exception:
                continue
            from_label = str(labels[h_site])
            to_label = str(labels[n_site])
            from_region = str(names[h_site])
            to_region = str(names[n_site])
            from_grain = int(grains[h_site])
            to_grain = int(grains[n_site])
            move_class = classify_move(from_label, to_label, from_grain, to_grain)
            add(by_label_pair, (from_label, to_label), barrier)
            add(by_region_pair, (from_region, to_region, from_grain, to_grain), barrier)
            add(by_move_class, (move_class,), barrier)
            add(by_from_label, (from_label,), barrier)
            add(by_to_label, (to_label,), barrier)

    write_summary(args.out_dir / "selected_barrier_summary_by_zone_pair.csv", ["from_label", "to_label"], by_label_pair)
    write_summary(
        args.out_dir / "selected_barrier_summary_by_region_pair.csv",
        ["from_region", "to_region", "from_grain_id", "to_grain_id"],
        by_region_pair,
    )
    write_summary(args.out_dir / "selected_barrier_summary_by_move_class.csv", ["move_class"], by_move_class)
    write_summary(args.out_dir / "selected_barrier_summary_by_from_zone.csv", ["from_label"], by_from_label)
    write_summary(args.out_dir / "selected_barrier_summary_by_to_zone.csv", ["to_label"], by_to_label)
    print(f"Wrote zone summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
