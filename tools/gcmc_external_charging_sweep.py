#!/usr/bin/env python3
"""GCMC chemical-potential sweep for Sigma5 Devanathan controlled regions.

The energy model intentionally matches production:

* the external Sigma5 interstitial-site map and Fe host structure;
* either the legacy source interval [16.12518061159, 26.12518061159) A
  or a region label from sigma5_site_regions.npz;
* rigid local EAM insertion energies in an 8--10 A shell;
* site-flip grand-canonical acceptance at the requested temperature.

The parent process launches one isolated subprocess per (mu, replica), then
merges worker summaries. Workers alternate between empty and target-occupancy
starts so slow mixing or hysteresis is visible in the aggregate result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
KB_EV_PER_K = 8.617333262145e-5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kmc.gcmc_energy_cache import (  # noqa: E402
    PersistentGCMCEnergyCache,
    canonical_insertion_environment_key,
)
from kmc.lammps_factory import (  # noqa: E402
    LammpsCalculatorConfig,
    LammpsCalculatorFactory,
)
from kmc.structures import (  # noqa: E402
    assign_mass_and_type,
    build_local_neb_structures_shell_from_host,
    load_kmc_site_map,
    load_lammps_atomic_structure,
)


SUMMARY_FIELDS = [
    "mu_eV",
    "replica",
    "seed",
    "initialization",
    "initial_N_H",
    "temperature_K",
    "steps",
    "equil_steps",
    "sample_interval",
    "selection_mode",
    "region_labels",
    "source_x_min_A",
    "source_x_max_A",
    "source_sites",
    "source_Fe",
    "target_N_H",
    "target_c_H_per_Fe",
    "target_N_H_1pct",
    "mean_N_H",
    "std_N_H",
    "mean_c_H_per_Fe",
    "std_c_H_per_Fe",
    "first_half_mean_N_H",
    "second_half_mean_N_H",
    "half_drift_N_H",
    "final_N_H",
    "insert_attempts",
    "insert_accepts",
    "delete_attempts",
    "delete_accepts",
    "accepted_transitions",
    "cache_entries",
    "cache_hits",
    "cache_misses",
    "energy_evaluations",
    "wall_seconds",
]


@dataclass(frozen=True)
class SweepTask:
    index: int
    mu_eV: float
    replica: int
    seed: int
    initialization: str


@dataclass(frozen=True)
class SourceSelection:
    mode: str
    region_labels: tuple[str, ...]
    site_indices: np.ndarray
    fe_count: int
    x_min_a: float
    x_max_a: float


def normalize_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--mu-values" and i + 1 < len(argv):
            normalized.append(f"--mu-values={argv[i + 1]}")
            i += 2
            continue
        normalized.append(argv[i])
        i += 1
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep mu_H using the production Sigma5 external-map GCMC model."
    )
    parser.add_argument(
        "--mu-values",
        default="-1.75,-1.74,-1.73,-1.72,-1.71",
        help="Comma-separated chemical potentials in eV.",
    )
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--equil", type=int, default=60_000)
    parser.add_argument("--sample-interval", type=int, default=20)
    parser.add_argument("--trace-interval", type=int, default=100)
    parser.add_argument("--print-interval", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--source-x-min", type=float, default=16.12518061159)
    parser.add_argument("--source-width", type=float, default=10.0)
    parser.add_argument("--target-fraction", type=float, default=0.01)
    parser.add_argument(
        "--target-n-h",
        type=int,
        default=0,
        help="Absolute target H count for start/recommendation. Overrides --target-fraction when >0.",
    )
    parser.add_argument(
        "--region-file",
        default="",
        help="sigma5_site_regions.npz. When set, selects controlled sites by region label.",
    )
    parser.add_argument(
        "--region-labels",
        default="",
        help="Comma-separated site_selection_label values, for example bulk_grain_0.",
    )
    parser.add_argument("--bulk-x-min", type=float, default=None)
    parser.add_argument("--bulk-x-max", type=float, default=None)
    parser.add_argument("--sink-x-min", type=float, default=None)
    parser.add_argument("--bulk-sink-gap", type=float, default=2.8601)
    parser.add_argument("--lattice-a", type=float, default=2.856)
    parser.add_argument("--shell-inner", type=float, default=8.0)
    parser.add_argument("--shell-outer", type=float, default=10.0)
    parser.add_argument("--position-bin", type=float, default=0.1)
    parser.add_argument("--site-map", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--potential", default="")
    parser.add_argument(
        "--out-dir",
        default="runs/gcmc_sigma5_external_charging_sweep",
    )
    parser.add_argument(
        "--preload-run",
        default="",
        help=(
            "Completed sweep directory whose nearest same-initialization worker "
            "cache should seed each new worker."
        ),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--mu", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--replica", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--initialization", choices=("empty", "target"), default="empty")
    return parser.parse_args(normalize_argv(sys.argv[1:] if argv is None else argv))


def parse_mu_values(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--mu-values did not contain a numeric value")
    return values


def parse_region_labels(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def first_existing(candidates: Sequence[Path], label: str) -> Path:
    for path in candidates:
        expanded = path.expanduser()
        if expanded.is_file():
            return expanded.resolve()
    joined = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate {label}; checked:\n  {joined}")


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    map_path = (
        Path(args.site_map).expanduser().resolve()
        if args.site_map
        else first_existing(
            [
                ROOT / "kmc_map_inputs/sigma5_stage3_unified_sites.npz",
                ROOT / "tools/sigma5_stage3_unified_sites.npz",
                WORKSPACE_ROOT / "kmc_map_inputs/sigma5_stage3_unified_sites.npz",
            ],
            "Sigma5 site map",
        )
    )
    host_path = (
        Path(args.host).expanduser().resolve()
        if args.host
        else first_existing(
            [
                ROOT / "kmc_map_inputs/sigma5_210-20-20-5.lmp",
                WORKSPACE_ROOT / "kmc_map_inputs/sigma5_210-20-20-5.lmp",
                WORKSPACE_ROOT / "base-working-version/kmc_map_inputs/sigma5_210-20-20-5.lmp",
            ],
            "Sigma5 Fe host",
        )
    )
    potential_path = (
        Path(args.potential).expanduser().resolve()
        if args.potential
        else first_existing(
            [
                ROOT / "kmc/PotentialB3410-modified.fs",
                WORKSPACE_ROOT / "kmc/PotentialB3410-modified.fs",
            ],
            "Fe-H EAM potential",
        )
    )
    for path, label in (
        (map_path, "site map"),
        (host_path, "host"),
        (potential_path, "potential"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    return map_path, host_path, potential_path


def resolve_region_file(raw_path: str) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Region metadata file not found: {path}")
        return path
    return first_existing(
        [
            ROOT / "kmc_map_inputs/sigma5_site_regions.npz",
            WORKSPACE_ROOT / "kmc_map_inputs/sigma5_site_regions.npz",
            WORKSPACE_ROOT / "base-working-version/kmc_map_inputs/sigma5_site_regions.npz",
        ],
        "Sigma5 site-region metadata",
    )


def select_source_region(
    args: argparse.Namespace,
    *,
    sites: np.ndarray,
    host_fe: np.ndarray,
    box: np.ndarray,
) -> SourceSelection:
    labels = parse_region_labels(args.region_labels)
    use_region_file = bool(args.region_file or labels)
    if use_region_file:
        region_file = resolve_region_file(args.region_file)
        if not labels:
            labels = ("bulk_grain_0",)
        data = np.load(region_file, allow_pickle=True)
        if "site_selection_label" not in data.files:
            raise KeyError(f"{region_file} does not contain site_selection_label")
        if "nearest_fe_atom_index" not in data.files:
            raise KeyError(f"{region_file} does not contain nearest_fe_atom_index")
        site_labels = np.asarray(data["site_selection_label"], dtype=str)
        nearest_fe = np.asarray(data["nearest_fe_atom_index"], dtype=int)
        if len(site_labels) != len(sites) or len(nearest_fe) != len(sites):
            raise ValueError(
                f"Region metadata length {len(site_labels)} does not match site map length {len(sites)}"
            )
        x_min = float(args.bulk_x_min if args.bulk_x_min is not None else args.source_x_min)
        sink_x = float(args.sink_x_min if args.sink_x_min is not None else 0.8 * float(box[0]))
        x_max = float(
            args.bulk_x_max
            if args.bulk_x_max is not None
            else sink_x - float(args.bulk_sink_gap)
        )
        if x_max <= x_min:
            raise ValueError(f"Bulk x-window is empty: [{x_min}, {x_max}) A")
        label_mask = np.isin(site_labels, np.asarray(labels, dtype=str))
        x_mask = (sites[:, 0] >= x_min) & (sites[:, 0] < x_max)
        selected = np.flatnonzero(label_mask & x_mask).astype(int)
        selected_fe = nearest_fe[selected]
        selected_fe = selected_fe[selected_fe >= 0]
        fe_count = int(len(np.unique(selected_fe)))
        if len(selected) == 0 or fe_count == 0:
            raise RuntimeError(
                f"Region labels {labels!r} in x=[{x_min:g}, {x_max:g}) contain "
                f"{len(selected)} sites and {fe_count} Fe atoms"
            )
        return SourceSelection(
            mode="region",
            region_labels=labels,
            site_indices=selected,
            fe_count=fe_count,
            x_min_a=x_min,
            x_max_a=x_max,
        )

    source_max = args.source_x_min + args.source_width
    source_sites = np.flatnonzero(
        (sites[:, 0] >= args.source_x_min) & (sites[:, 0] < source_max)
    ).astype(int)
    source_fe_count = int(
        np.count_nonzero(
            (host_fe[:, 0] >= args.source_x_min) & (host_fe[:, 0] < source_max)
        )
    )
    if len(source_sites) == 0 or source_fe_count == 0:
        raise RuntimeError("The requested charging interval contains no sites or Fe atoms")
    return SourceSelection(
        mode="x_window",
        region_labels=(),
        site_indices=source_sites,
        fe_count=source_fe_count,
        x_min_a=float(args.source_x_min),
        x_max_a=float(source_max),
    )


def target_h_count(args: argparse.Namespace, fe_count: int, available_sites: int) -> int:
    target_h = (
        int(args.target_n_h)
        if int(args.target_n_h) > 0
        else int(round(float(args.target_fraction) * int(fe_count)))
    )
    if target_h < 0:
        raise ValueError(f"Target H count must be non-negative; got {target_h}")
    if target_h > available_sites:
        raise ValueError(
            f"Target H count {target_h} exceeds available controlled sites {available_sites}"
        )
    return target_h


def file_fingerprint(path: Path) -> tuple[str, int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.name, path.stat().st_size, digest.hexdigest()


def safe_mu(mu_eV: float) -> str:
    return (
        f"{mu_eV:+.6f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .rstrip("0")
        .rstrip("p")
    )


class ExternalChargingEnergy:
    def __init__(
        self,
        *,
        sites: np.ndarray,
        host_fe: np.ndarray,
        box: np.ndarray,
        lattice_a: float,
        shell_inner: float,
        shell_outer: float,
        position_bin: float,
        potential_path: Path,
        site_map_path: Path,
        host_path: Path,
        cache_path: Path,
    ) -> None:
        from scipy.spatial import cKDTree

        self.sites = np.asarray(sites, dtype=float)
        self.host_fe = np.mod(np.asarray(host_fe, dtype=float), box)
        self.box = np.asarray(box, dtype=float)
        self.lattice_a = float(lattice_a)
        self.shell_inner = float(shell_inner)
        self.shell_outer = float(shell_outer)
        self.position_bin = float(position_bin)
        self.host_tree = cKDTree(self.host_fe, boxsize=self.box)
        self.cache = PersistentGCMCEnergyCache(
            cache_path,
            autosave_interval=250,
        )
        self.energy_evaluations = 0
        self.potential_tag = ("EAM", file_fingerprint(potential_path))
        self.host_tag = (
            "external_sigma5_charging_sweep_v1",
            file_fingerprint(site_map_path),
            file_fingerprint(host_path),
            tuple(round(float(value), 12) for value in self.box),
            len(self.host_fe),
        )
        factory = LammpsCalculatorFactory(
            LammpsCalculatorConfig(
                potential="EAM",
                potential_eam_file=str(potential_path),
                lammps_files=str(potential_path),
            ),
            base_dir=str(potential_path.parent),
        )
        self.calculator = factory.create()

    def insertion_delta(self, lower_occupancy: set[int], trial_site: int) -> float:
        lower = np.fromiter(sorted(lower_occupancy), dtype=int)
        key = canonical_insertion_environment_key(
            sites=self.sites,
            lower_occupancy_indices=lower,
            trial_site=int(trial_site),
            box=self.box,
            lattice_a=self.lattice_a,
            shell_inner_a=self.shell_inner,
            shell_outer_a=self.shell_outer,
            position_bin_a=self.position_bin,
            potential_tag=self.potential_tag,
            host_geometry_tag=self.host_tag,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return float(cached)

        higher = np.append(lower, int(trial_site))
        h_positions = np.asarray(self.sites[higher], dtype=float)
        with_site, _same, mover_idx = build_local_neb_structures_shell_from_host(
            np.asarray(self.sites[int(trial_site)], dtype=float),
            np.asarray(self.sites[int(trial_site)], dtype=float),
            self.host_fe,
            self.box,
            host_fe_tree=self.host_tree,
            h_positions=h_positions,
            h_indices=higher,
            inner_radius_a=self.shell_inner,
            outer_radius_a=self.shell_outer,
        )
        without_site = with_site.copy()
        del without_site[int(mover_idx)]
        assign_mass_and_type(without_site)

        without_site.calc = self.calculator
        energy_without = float(without_site.get_potential_energy())
        with_site.calc = self.calculator
        energy_with = float(with_site.get_potential_energy())
        delta = energy_with - energy_without
        self.energy_evaluations += 1
        self.cache.set(key, delta)
        return float(delta)

    def close(self) -> None:
        self.cache.save(full=True)
        lmp = getattr(self.calculator, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


def accept(exponent: float, rng: np.random.Generator) -> bool:
    return exponent >= 0.0 or bool(rng.random() < math.exp(exponent))


def split_means(values: list[int]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    midpoint = max(1, len(array) // 2)
    first = array[:midpoint]
    second = array[midpoint:]
    if second.size == 0:
        second = first
    return float(first.mean()), float(second.mean())


def run_worker(args: argparse.Namespace) -> int:
    if args.mu is None:
        raise ValueError("--single-worker requires --mu")
    map_path, host_path, potential_path = resolve_inputs(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sites, _site_types, map_box = load_kmc_site_map(map_path)
    host_atoms, _host_types, host_box = load_lammps_atomic_structure(host_path)
    box = np.asarray(map_box if map_box is not None else host_box, dtype=float)
    if not np.allclose(box, host_box, atol=1.0e-6):
        raise RuntimeError(f"Map box {box.tolist()} does not match host box {host_box.tolist()}")
    sites = np.mod(np.asarray(sites, dtype=float), box)
    host_numbers = np.asarray(host_atoms.get_atomic_numbers(), dtype=int)
    host_fe = np.asarray(host_atoms.get_positions(), dtype=float)[host_numbers == 26]

    selection = select_source_region(args, sites=sites, host_fe=host_fe, box=box)
    source_sites = selection.site_indices
    source_fe_count = selection.fe_count
    target_h = target_h_count(args, source_fe_count, len(source_sites))
    one_percent_h = int(round(0.01 * source_fe_count))

    rng = np.random.default_rng(args.seed)
    occupied: set[int] = set()
    if args.initialization == "target" and target_h:
        occupied.update(
            int(value)
            for value in rng.choice(source_sites, size=target_h, replace=False)
        )
    initial_h = len(occupied)
    energy = ExternalChargingEnergy(
        sites=sites,
        host_fe=host_fe,
        box=box,
        lattice_a=args.lattice_a,
        shell_inner=args.shell_inner,
        shell_outer=args.shell_outer,
        position_bin=args.position_bin,
        potential_path=potential_path,
        site_map_path=map_path,
        host_path=host_path,
        cache_path=out_dir / "gcmc_energy_cache.pkl",
    )

    kbt = KB_EV_PER_K * args.temperature
    insert_attempts = insert_accepts = 0
    delete_attempts = delete_accepts = 0
    samples: list[int] = []
    trace_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    source_tuple = tuple(int(value) for value in source_sites)

    print(
        f"mu={args.mu:g} replica={args.replica} seed={args.seed} "
        f"initial={args.initialization} N0={initial_h} "
        f"mode={selection.mode} regions={'+'.join(selection.region_labels) or 'none'} "
        f"source_sites={len(source_sites)} source_Fe={source_fe_count} target={target_h}",
        flush=True,
    )
    try:
        for step in range(1, args.steps + 1):
            site = int(rng.choice(source_tuple))
            if site in occupied:
                delete_attempts += 1
                lower = set(occupied)
                lower.remove(site)
                insertion_delta = energy.insertion_delta(lower, site)
                delta_energy = -insertion_delta
                if accept(-(delta_energy + args.mu) / kbt, rng):
                    occupied.remove(site)
                    delete_accepts += 1
            else:
                insert_attempts += 1
                delta_energy = energy.insertion_delta(occupied, site)
                if accept(-(delta_energy - args.mu) / kbt, rng):
                    occupied.add(site)
                    insert_accepts += 1

            n_h = len(occupied)
            if step > args.equil and step % args.sample_interval == 0:
                samples.append(n_h)
            if step % args.trace_interval == 0:
                trace_rows.append(
                    {
                        "step": step,
                        "phase": "equilibration" if step <= args.equil else "production",
                        "N_H": n_h,
                        "c_H_per_Fe": n_h / source_fe_count,
                        "insert_accepts": insert_accepts,
                        "delete_accepts": delete_accepts,
                        "cache_entries": len(energy.cache),
                        "cache_hits": energy.cache.stats.hits,
                        "cache_misses": energy.cache.stats.misses,
                    }
                )
            if args.print_interval and step % args.print_interval == 0:
                print(
                    f"step={step}/{args.steps} N_H={n_h} "
                    f"c={n_h / source_fe_count:.8g} "
                    f"transitions={insert_accepts + delete_accepts} "
                    f"cache={len(energy.cache)} evals={energy.energy_evaluations} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    finally:
        energy.close()

    if not samples:
        raise RuntimeError("No production samples were collected")
    sample_array = np.asarray(samples, dtype=float)
    first_mean, second_mean = split_means(samples)
    wall_seconds = time.monotonic() - started
    summary = {
        "mu_eV": args.mu,
        "replica": args.replica,
        "seed": args.seed,
        "initialization": args.initialization,
        "initial_N_H": initial_h,
        "temperature_K": args.temperature,
        "steps": args.steps,
        "equil_steps": args.equil,
        "sample_interval": args.sample_interval,
        "selection_mode": selection.mode,
        "region_labels": "+".join(selection.region_labels),
        "source_x_min_A": selection.x_min_a,
        "source_x_max_A": selection.x_max_a,
        "source_sites": len(source_sites),
        "source_Fe": source_fe_count,
        "target_N_H": target_h,
        "target_c_H_per_Fe": target_h / source_fe_count,
        "target_N_H_1pct": one_percent_h,
        "mean_N_H": float(sample_array.mean()),
        "std_N_H": float(sample_array.std(ddof=0)),
        "mean_c_H_per_Fe": float(sample_array.mean() / source_fe_count),
        "std_c_H_per_Fe": float(sample_array.std(ddof=0) / source_fe_count),
        "first_half_mean_N_H": first_mean,
        "second_half_mean_N_H": second_mean,
        "half_drift_N_H": second_mean - first_mean,
        "final_N_H": len(occupied),
        "insert_attempts": insert_attempts,
        "insert_accepts": insert_accepts,
        "delete_attempts": delete_attempts,
        "delete_accepts": delete_accepts,
        "accepted_transitions": insert_accepts + delete_accepts,
        "cache_entries": len(energy.cache),
        "cache_hits": energy.cache.stats.hits,
        "cache_misses": energy.cache.stats.misses,
        "energy_evaluations": energy.energy_evaluations,
        "wall_seconds": wall_seconds,
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary)
    with (out_dir / "trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)
    with (out_dir / "final_occupied_sites.txt").open("w", encoding="utf-8") as handle:
        for site in sorted(occupied):
            handle.write(f"{site}\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def worker_command(
    args: argparse.Namespace,
    task: SweepTask,
    worker_dir: Path,
) -> tuple[str, ...]:
    map_path, host_path, potential_path = resolve_inputs(args)
    command = [
        str(args.python_bin),
        str(Path(__file__).resolve()),
        "--single-worker",
        "--mu",
        f"{task.mu_eV:.17g}",
        "--replica",
        str(task.replica),
        "--initialization",
        task.initialization,
        "--seed",
        str(task.seed),
        "--steps",
        str(args.steps),
        "--equil",
        str(args.equil),
        "--sample-interval",
        str(args.sample_interval),
        "--trace-interval",
        str(args.trace_interval),
        "--print-interval",
        str(args.print_interval),
        "--temperature",
        str(args.temperature),
        "--source-x-min",
        str(args.source_x_min),
        "--source-width",
        str(args.source_width),
        "--target-fraction",
        str(args.target_fraction),
        "--target-n-h",
        str(args.target_n_h),
        "--lattice-a",
        str(args.lattice_a),
        "--shell-inner",
        str(args.shell_inner),
        "--shell-outer",
        str(args.shell_outer),
        "--position-bin",
        str(args.position_bin),
        "--site-map",
        str(map_path),
        "--host",
        str(host_path),
        "--potential",
        str(potential_path),
        "--out-dir",
        str(worker_dir),
    ]
    if args.region_file:
        command.extend(["--region-file", str(Path(args.region_file).expanduser().resolve())])
    if args.region_labels:
        command.extend(["--region-labels", str(args.region_labels)])
    if args.bulk_x_min is not None:
        command.extend(["--bulk-x-min", str(args.bulk_x_min)])
    if args.bulk_x_max is not None:
        command.extend(["--bulk-x-max", str(args.bulk_x_max)])
    if args.sink_x_min is not None:
        command.extend(["--sink-x-min", str(args.sink_x_min)])
    command.extend(["--bulk-sink-gap", str(args.bulk_sink_gap)])
    return tuple(command)


def execute_task(
    args: argparse.Namespace,
    task: SweepTask,
    out_dir: Path,
) -> tuple[SweepTask, Path, int]:
    worker_dir = (
        out_dir
        / "workers"
        / f"{task.index:04d}_mu_{safe_mu(task.mu_eV)}_rep{task.replica}_{task.initialization}"
    )
    worker_dir.mkdir(parents=True, exist_ok=False)
    preload_source = nearest_preload_cache(args.preload_run, task)
    if preload_source is not None:
        shutil.copy2(preload_source, worker_dir / "gcmc_energy_cache.pkl")
    command = worker_command(args, task, worker_dir)
    with (worker_dir / "worker.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return task, worker_dir, result.returncode


def nearest_preload_cache(raw_run: str, task: SweepTask) -> Path | None:
    if not raw_run:
        return None
    run_dir = Path(raw_run).expanduser()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    candidates: list[tuple[float, Path]] = []
    for summary_path in run_dir.glob("workers/*/summary.csv"):
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0].get("initialization") != task.initialization:
            continue
        cache_path = summary_path.parent / "gcmc_energy_cache.pkl"
        if cache_path.is_file():
            candidates.append(
                (abs(float(rows[0]["mu_eV"]) - task.mu_eV), cache_path)
            )
    if not candidates:
        raise FileNotFoundError(
            f"No {task.initialization!r} worker cache found in preload run {run_dir}"
        )
    return min(candidates, key=lambda item: item[0])[1]


def read_worker_summary(worker_dir: Path) -> dict[str, str]:
    with (worker_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one summary row in {worker_dir}")
    return rows[0]


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["mu_eV"]), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for mu_eV in sorted(grouped):
        group = grouped[mu_eV]
        means_nh = np.asarray([float(row["mean_N_H"]) for row in group])
        means_c = np.asarray([float(row["mean_c_H_per_Fe"]) for row in group])
        drifts = np.asarray([float(row["half_drift_N_H"]) for row in group])
        target_n_h = float(group[0].get("target_N_H") or group[0]["target_N_H_1pct"])
        target_c = float(
            group[0].get("target_c_H_per_Fe")
            or (target_n_h / float(group[0]["source_Fe"]))
        )
        aggregate.append(
            {
                "mu_eV": mu_eV,
                "replicas": len(group),
                "target_N_H": target_n_h,
                "target_c_H_per_Fe": target_c,
                "mean_N_H": float(means_nh.mean()),
                "replica_std_mean_N_H": float(means_nh.std(ddof=0)),
                "min_replica_mean_N_H": float(means_nh.min()),
                "max_replica_mean_N_H": float(means_nh.max()),
                "mean_c_H_per_Fe": float(means_c.mean()),
                "replica_std_mean_c_H_per_Fe": float(means_c.std(ddof=0)),
                "mean_abs_half_drift_N_H": float(np.abs(drifts).mean()),
                "max_abs_half_drift_N_H": float(np.abs(drifts).max()),
                "accepted_transitions": sum(int(row["accepted_transitions"]) for row in group),
                "distance_from_target_N_H": abs(float(means_nh.mean()) - target_n_h),
                "distance_from_target_c_H_per_Fe": abs(float(means_c.mean()) - target_c),
                "distance_from_1pct": abs(float(means_c.mean()) - 0.01),
            }
        )
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_parent(args: argparse.Namespace) -> int:
    if args.workers < 1 or args.replicas < 1:
        raise ValueError("--workers and --replicas must be at least 1")
    if args.steps <= args.equil:
        raise ValueError("--steps must be greater than --equil")
    if min(args.sample_interval, args.trace_interval) < 1:
        raise ValueError("sample and trace intervals must be positive")
    map_path, host_path, potential_path = resolve_inputs(args)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = out_dir.resolve()
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory exists: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sites, _site_types, map_box = load_kmc_site_map(map_path)
    host_atoms, _host_types, host_box = load_lammps_atomic_structure(host_path)
    box = np.asarray(map_box if map_box is not None else host_box, dtype=float)
    if not np.allclose(box, host_box, atol=1.0e-6):
        raise RuntimeError(f"Map box {box.tolist()} does not match host box {host_box.tolist()}")
    sites = np.mod(np.asarray(sites, dtype=float), box)
    host_numbers = np.asarray(host_atoms.get_atomic_numbers(), dtype=int)
    host_fe = np.asarray(host_atoms.get_positions(), dtype=float)[host_numbers == 26]
    selection = select_source_region(args, sites=sites, host_fe=host_fe, box=box)
    target_h = target_h_count(args, selection.fe_count, len(selection.site_indices))
    selection_summary = {
        "selection_mode": selection.mode,
        "region_labels": list(selection.region_labels),
        "x_min_A": selection.x_min_a,
        "x_max_A": selection.x_max_a,
        "controlled_sites": int(len(selection.site_indices)),
        "controlled_Fe": int(selection.fe_count),
        "target_N_H": int(target_h),
        "target_c_H_per_Fe": float(target_h / selection.fe_count),
    }
    with (out_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_summary, handle, indent=2, sort_keys=True)

    tasks: list[SweepTask] = []
    for mu_index, mu_eV in enumerate(parse_mu_values(args.mu_values)):
        for replica in range(args.replicas):
            tasks.append(
                SweepTask(
                    index=len(tasks),
                    mu_eV=mu_eV,
                    replica=replica,
                    seed=args.seed + 1000 * mu_index + replica,
                    initialization="empty" if replica % 2 == 0 else "target",
                )
            )
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        config = vars(args).copy()
        config["tasks"] = [task.__dict__ for task in tasks]
        config["selection"] = selection_summary
        json.dump(config, handle, indent=2, sort_keys=True)

    print(
        f"Launching {len(tasks)} Sigma5 GCMC workers on "
        f"{min(args.workers, len(tasks))} local cores",
        flush=True,
    )
    print(
        f"Selection: mode={selection.mode} regions={'+'.join(selection.region_labels) or 'none'} "
        f"x=[{selection.x_min_a:g}, {selection.x_max_a:g}) A "
        f"sites={len(selection.site_indices)} Fe={selection.fe_count} "
        f"target={target_h} H ({target_h / selection.fe_count:.8g} H/Fe)",
        flush=True,
    )
    completed: list[tuple[SweepTask, Path, int]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {
            executor.submit(execute_task, args, task, out_dir): task for task in tasks
        }
        for future in as_completed(futures):
            task, worker_dir, returncode = future.result()
            completed.append((task, worker_dir, returncode))
            status = "ok" if returncode == 0 else f"failed({returncode})"
            print(
                f"[{len(completed)}/{len(tasks)}] mu={task.mu_eV:g} "
                f"replica={task.replica} initial={task.initialization} {status}",
                flush=True,
            )

    failures = [item for item in completed if item[2] != 0]
    if failures:
        for task, worker_dir, returncode in failures:
            print(
                f"ERROR mu={task.mu_eV:g} replica={task.replica} "
                f"returncode={returncode}; see {worker_dir / 'worker.log'}",
                file=sys.stderr,
            )
        return 1

    completed.sort(key=lambda item: item[0].index)
    rows = [read_worker_summary(worker_dir) for _task, worker_dir, _code in completed]
    write_csv(out_dir / "replica_summary.csv", rows)
    aggregate = aggregate_rows(rows)
    write_csv(out_dir / "mu_summary.csv", aggregate)
    best = min(aggregate, key=lambda row: float(row["distance_from_target_N_H"]))
    with (out_dir / "recommendation.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            f"closest_mu_eV={float(best['mu_eV']):.10g}\n"
            f"target_N_H={float(best['target_N_H']):.10g}\n"
            f"mean_c_H_per_Fe={float(best['mean_c_H_per_Fe']):.10g}\n"
            f"mean_N_H={float(best['mean_N_H']):.10g}\n"
            f"replica_std_mean_N_H={float(best['replica_std_mean_N_H']):.10g}\n"
            f"distance_from_target_N_H={float(best['distance_from_target_N_H']):.10g}\n"
            f"max_abs_half_drift_N_H={float(best['max_abs_half_drift_N_H']):.10g}\n"
        )
    print(f"Wrote {out_dir / 'mu_summary.csv'}", flush=True)
    print(
        f"Closest sampled point: mu={float(best['mu_eV']):g} eV, "
        f"mean_N_H={float(best['mean_N_H']):.8g}, "
        f"mean H/Fe={float(best['mean_c_H_per_Fe']):.8g}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.single_worker else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
