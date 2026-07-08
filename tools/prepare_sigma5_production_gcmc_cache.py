#!/usr/bin/env python3
"""Convert one Sigma5 sweep cache into the production cache namespace."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kmc.gcmc_energy_cache import PersistentGCMCEnergyCache  # noqa: E402
from kmc.structures import (  # noqa: E402
    load_kmc_site_map,
    load_lammps_atomic_structure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-run", required=True, type=Path)
    parser.add_argument("--site-map", required=True, type=Path)
    parser.add_argument("--host", required=True, type=Path)
    parser.add_argument("--potential", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-fraction", type=float, default=0.01)
    parser.add_argument(
        "--initialization",
        choices=("empty", "target", "any"),
        default="target",
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Merge every matching worker cache instead of selecting one.",
    )
    return parser.parse_args()


def file_tag(path: Path) -> tuple[Any, ...]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def select_workers(
    sweep_run: Path,
    *,
    initialization: str,
    target_fraction: float,
    merge_all: bool,
) -> list[tuple[Path, dict[str, str]]]:
    candidates: list[tuple[float, Path, dict[str, str]]] = []
    for summary_path in sweep_run.expanduser().resolve().glob("workers/*/summary.csv"):
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            continue
        row = rows[0]
        if initialization != "any" and row.get("initialization") != initialization:
            continue
        cache_path = summary_path.parent / "gcmc_energy_cache.pkl"
        if not cache_path.is_file():
            continue
        distance = abs(float(row["mean_c_H_per_Fe"]) - target_fraction)
        candidates.append((distance, cache_path, row))
    if not candidates:
        raise FileNotFoundError(
            f"No matching worker cache found under {sweep_run}"
        )
    candidates.sort(key=lambda item: item[0])
    selected = candidates if merge_all else candidates[:1]
    return [(cache_path, row) for _distance, cache_path, row in selected]


def main() -> int:
    args = parse_args()
    for path in (args.site_map, args.host, args.potential):
        if not path.expanduser().is_file():
            raise FileNotFoundError(path)
    if args.output.expanduser().exists():
        raise FileExistsError(args.output)

    selected_workers = select_workers(
        args.sweep_run,
        initialization=args.initialization,
        target_fraction=args.target_fraction,
        merge_all=args.merge_all,
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = PersistentGCMCEnergyCache(output_path)

    potential_tag = (
        "EAM",
        file_tag(args.potential),
        "",
        "",
    )
    _sites, _site_types, map_box = load_kmc_site_map(args.site_map)
    host_atoms, _host_types, host_box = load_lammps_atomic_structure(args.host)
    box = np.asarray(map_box if map_box is not None else host_box, dtype=float)
    if not np.allclose(box, host_box, atol=1.0e-6):
        raise RuntimeError(
            f"Site-map box {box.tolist()} does not match host box {host_box.tolist()}"
        )
    host_numbers = np.asarray(host_atoms.get_atomic_numbers(), dtype=int)
    fe_count = int(np.count_nonzero(host_numbers == 26))
    host_tag = (
        "external_host",
        file_tag(args.site_map),
        file_tag(args.host),
        tuple(
            round(float(value), 12)
            for value in box
        ),
        fe_count,
    )

    converted = 0
    skipped = 0
    for source_path, source_row in selected_workers:
        source = PersistentGCMCEnergyCache(source_path)
        before = len(output.cache.store)
        for key, value in source.cache.store.items():
            if not isinstance(key, tuple) or len(key) != 11:
                skipped += 1
                continue
            if key[0] != "GCMC_SOURCE_INSERTION_DE_V2_HOST":
                skipped += 1
                continue
            converted_key = (
                *key[:6],
                potential_tag,
                host_tag,
                *key[8:],
            )
            output.cache.store[converted_key] = float(value)
            converted += 1
        print(
            f"Merged {source_path}: "
            f"mu={source_row['mu_eV']} replica={source_row['replica']} "
            f"initialization={source_row['initialization']} "
            f"mean_H_per_Fe={source_row['mean_c_H_per_Fe']} "
            f"new_unique_entries={len(output.cache.store) - before}"
        )
    output.save(full=True)

    print(
        f"Processed entries: {converted}; unique entries: {len(output)}; "
        f"skipped: {skipped}"
    )
    print(f"Wrote production cache: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
