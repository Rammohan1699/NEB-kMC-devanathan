#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from development.lammps_only_neb.config import LammpsOnlyNEBConfig
from development.lammps_only_neb.geometry import (
    bcc_fe_positions,
    build_batch_from_sites,
    knn_sites,
    minimum_image_delta,
    tetrahedral_sites,
)
from development.lammps_only_neb.models import EnvironmentNEBBatch, HopResult
from development.lammps_only_neb.runner import load_lammps_class, run_batch_slots
from development.lammps_only_neb.scheduler import split_group_comm
from kmc.event_manager import KineticParameters, SelectedEvent, select_rejection_free_event


DEFAULT_LAMMPS_PYTHON = REPO_ROOT / "development" / "tests" / "native_lammps_neb_test"
DEFAULT_LAMMPS_LIB_DIR = Path("/Users/rtirunelveli/mylammps/build2")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small generated-box KMC run using grouped LAMMPS-only NEB batches.")
    parser.add_argument("--work-dir", type=Path, default=Path("development/lammps_only_neb_runs/small_kmc_5h_10step"))
    parser.add_argument("--potential", type=Path, default=REPO_ROOT / "kmc" / "PotentialB3410-modified.fs")
    parser.add_argument("--lammps-python", type=Path, default=DEFAULT_LAMMPS_PYTHON)
    parser.add_argument("--lammps-lib-dir", type=Path, default=DEFAULT_LAMMPS_LIB_DIR)
    parser.add_argument("--lattice-a", type=float, default=2.8601)
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--num-h", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--knn", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--shell-inner-radius", type=float, default=8.0)
    parser.add_argument("--shell-outer-radius", type=float, default=10.0)
    parser.add_argument("--shell-pad", type=float, default=2.0)
    parser.add_argument("--min-style", choices=("quickmin", "fire"), default="quickmin")
    parser.add_argument("--neb-steps", type=int, default=1000)
    parser.add_argument("--ftol", type=float, default=1.0e-5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--attempt-frequency", type=float, default=1.0e13)
    parser.add_argument("--debug-mode", action="store_true", help="Keep LAMMPS input, log, screen, and trajectory files.")
    return parser.parse_args(argv)


def env_key_for_h(
    h_site: int,
    valid_neighbors: list[int],
    occupied: set[int],
    *,
    sites: np.ndarray,
    box: np.ndarray,
    radius_a: float,
) -> tuple[Any, ...]:
    # First development key: group the six associated hops under one local H
    # environment. Only occupied H inside the shell radius are part of the
    # environment signature; sparse far-away H should not invalidate this batch.
    center = np.asarray(sites[int(h_site)], dtype=float)
    local_occ: list[tuple[int, tuple[float, float, float]]] = []
    for occ in sorted(map(int, occupied)):
        if occ == int(h_site):
            continue
        delta = minimum_image_delta(center, np.asarray(sites[occ], dtype=float), box)
        if float(np.linalg.norm(delta)) <= float(radius_a):
            local_occ.append((occ, tuple(float(x) for x in np.round(delta, 3))))
    return ("generated_shell_v2", int(h_site), tuple(map(int, valid_neighbors)), tuple(local_occ))


def valid_neighbors_for_h(h_site: int, neighbors: list[list[int]], occupied: set[int]) -> list[int]:
    return [int(n) for n in neighbors[int(h_site)][:6] if int(n) not in occupied and int(n) != int(h_site)]


def make_batches(
    *,
    h_indices: np.ndarray,
    sites: np.ndarray,
    neighbors: list[list[int]],
    fe_positions: np.ndarray,
    box: np.ndarray,
    lattice_a: float,
    env_cache: dict[Any, dict[int, float]],
    env_radius_a: float,
) -> tuple[list[EnvironmentNEBBatch], dict[tuple[int, int], float]]:
    occupied = set(map(int, h_indices))
    missing: list[EnvironmentNEBBatch] = []
    known_barriers: dict[tuple[int, int], float] = {}

    for h in map(int, h_indices):
        valid = valid_neighbors_for_h(h, neighbors, occupied)
        if not valid:
            continue
        env_key = env_key_for_h(
            h,
            valid,
            occupied,
            sites=sites,
            box=box,
            radius_a=env_radius_a,
        )
        cached = env_cache.get(env_key, {})
        for n in valid:
            if int(n) in cached:
                known_barriers[(h, int(n))] = float(cached[int(n)])
        missing_neighbors = [n for n in valid if int(n) not in cached]
        if missing_neighbors:
            missing.append(
                build_batch_from_sites(
                    env_key=env_key,
                    h_site=h,
                    sites=sites,
                    neighbor_indices=missing_neighbors,
                    fe_positions=fe_positions,
                    box=box,
                    lattice_a=lattice_a,
                )
            )
    return missing, known_barriers


def run_assigned_batches(
    batches: list[EnvironmentNEBBatch],
    *,
    step: int,
    group_id: int,
    n_groups: int,
    group_comm,
    cfg: LammpsOnlyNEBConfig,
    lammps_class,
) -> list[HopResult]:
    out: list[HopResult] = []
    for batch_idx, batch in enumerate(batches):
        if batch_idx % n_groups != group_id:
            continue
        batch_dir = cfg.work_dir / f"step_{step:04d}" / f"group_{group_id}" / f"batch_{batch_idx:04d}_h{batch.h_site}"
        out.extend(
            run_batch_slots(
                batch,
                list(range(len(batch.hops))),
                group_comm=group_comm,
                group_work_dir=batch_dir,
                cfg=cfg,
                lammps_class=lammps_class,
            )
        )
    return out


def select_event_from_barriers(
    *,
    h_indices: np.ndarray,
    neighbors: list[list[int]],
    occupied: set[int],
    barriers: dict[tuple[int, int], float],
    kinetics: KineticParameters,
) -> SelectedEvent | None:
    moves = []
    rates = []
    move_barriers = {}
    for h in map(int, h_indices):
        for n in valid_neighbors_for_h(h, neighbors, occupied):
            key = (int(h), int(n))
            if key not in barriers:
                continue
            barrier = float(barriers[key])
            rate = kinetics.rate_from_barrier(barrier)
            if np.isfinite(rate) and rate > 0.0:
                moves.append(key)
                rates.append(rate)
                move_barriers[key] = barrier
    ordered = sorted(zip(moves, rates), key=lambda item: (item[0][0], item[0][1]))
    return select_rejection_free_event(
        [move for move, _rate in ordered],
        [rate for _move, rate in ordered],
        barriers=move_barriers,
    )


def write_selected_events(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "step",
        "h_site",
        "n_site",
        "barrier_eV",
        "rate_hz",
        "dt_s",
        "sim_time_s",
        "neb_batches",
        "neb_jobs",
        "neb_wall_s",
        "step_wall_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        from mpi4py import MPI
    except Exception as exc:
        raise SystemExit(f"mpi4py is required: {exc}") from exc

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    group_id, n_groups, group_comm = split_group_comm(world, int(args.replicas))
    group_rank = group_comm.Get_rank()

    cfg = LammpsOnlyNEBConfig(
        replicas_per_neb=int(args.replicas),
        force_tolerance=float(args.ftol),
        max_steps=int(args.neb_steps),
        min_style=str(args.min_style),
        shell_inner_radius_a=float(args.shell_inner_radius),
        shell_outer_radius_a=float(args.shell_outer_radius),
        shell_pad_a=float(args.shell_pad),
        debug_mode=bool(args.debug_mode),
        potential_file=args.potential.resolve(),
        work_dir=args.work_dir.resolve(),
        lammps_python_path=args.lammps_python.resolve(),
        lammps_lib_dir=args.lammps_lib_dir.resolve(),
    ).normalized()

    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    box = np.array([args.nx * args.lattice_a, args.ny * args.lattice_a, args.nz * args.lattice_a], dtype=float)
    sites = tetrahedral_sites(args.nx, args.ny, args.nz, args.lattice_a)
    neighbors = knn_sites(sites, box, int(args.knn))
    fe_positions = bcc_fe_positions(args.nx, args.ny, args.nz, args.lattice_a)

    if rank == 0:
        h_indices = np.random.choice(len(sites), size=int(args.num_h), replace=False).astype(int)
        print(
            f"[rank 0] small KMC start: sites={len(sites)} Fe={len(fe_positions)} H={len(h_indices)} "
            f"steps={args.steps} groups={n_groups}",
            flush=True,
        )
    else:
        h_indices = None
    h_indices = world.bcast(h_indices, root=0)

    lammps_class = load_lammps_class(cfg)
    kinetics = KineticParameters(
        temperature_k=float(args.temperature_k),
        attempt_frequency_hz=float(args.attempt_frequency),
    )
    env_cache: dict[Any, dict[int, float]] = {}
    sim_time_s = 0.0
    selected_rows: list[dict[str, Any]] = []

    for step in range(int(args.steps)):
        step_t0 = time.perf_counter()
        occupied = set(map(int, h_indices))
        if rank == 0:
            batches, known = make_batches(
                h_indices=h_indices,
                sites=sites,
                neighbors=neighbors,
                fe_positions=fe_positions,
                box=box,
                lattice_a=float(args.lattice_a),
                env_cache=env_cache,
                env_radius_a=float(args.shell_outer_radius),
            )
            print(f"[rank 0] step {step}: missing env batches={len(batches)} cached_hops={len(known)}", flush=True)
        else:
            batches = None
            known = None
        batches = world.bcast(batches, root=0)
        known = world.bcast(known, root=0)
        world.Barrier()
        neb_t0 = time.perf_counter()
        local_results = run_assigned_batches(
            batches,
            step=step,
            group_id=group_id,
            n_groups=n_groups,
            group_comm=group_comm,
            cfg=cfg,
            lammps_class=lammps_class,
        )
        world.Barrier()
        neb_wall_s = time.perf_counter() - neb_t0
        gathered = world.gather(local_results if group_rank == 0 else [], root=0)

        if rank == 0:
            results = [result for part in gathered for result in part]
            for result in results:
                bucket = env_cache.setdefault(result.env_key, {})
                bucket[int(result.n_site)] = float(result.barrier_eV)
                known[(int(result.h_site), int(result.n_site))] = float(result.barrier_eV)
            selected = select_event_from_barriers(
                h_indices=h_indices,
                neighbors=neighbors,
                occupied=occupied,
                barriers=known,
                kinetics=kinetics,
            )
            if selected is None:
                print(f"[rank 0] step {step}: no valid event", flush=True)
                stop = True
            else:
                matches = np.where(h_indices == int(selected.h_site))[0]
                if len(matches) == 0:
                    raise RuntimeError(f"selected H site {selected.h_site} not occupied")
                h_indices = h_indices.copy()
                h_indices[int(matches[0])] = int(selected.n_site)
                sim_time_s += float(selected.dt_s)
                step_wall_s = time.perf_counter() - step_t0
                selected_rows.append(
                    {
                        "step": step,
                        "h_site": int(selected.h_site),
                        "n_site": int(selected.n_site),
                        "barrier_eV": float(selected.barrier_ev),
                        "rate_hz": float(selected.rate_hz),
                        "dt_s": float(selected.dt_s),
                        "sim_time_s": sim_time_s,
                        "neb_batches": len(batches),
                        "neb_jobs": len(results),
                        "neb_wall_s": neb_wall_s,
                        "step_wall_s": step_wall_s,
                    }
                )
                print(
                    f"[rank 0] step {step}: selected {selected.h_site}->{selected.n_site} "
                    f"barrier={selected.barrier_ev:.9g} eV neb_jobs={len(results)} "
                    f"neb_wall={neb_wall_s:.3f}s step_wall={step_wall_s:.3f}s",
                    flush=True,
                )
                stop = False
        else:
            stop = None
        stop = world.bcast(stop, root=0)
        h_indices = world.bcast(h_indices if rank == 0 else None, root=0)
        env_cache = world.bcast(env_cache if rank == 0 else None, root=0)
        if stop:
            break

    if rank == 0:
        write_selected_events(cfg.work_dir / "selected_events.csv", selected_rows)
        print(f"[rank 0] wrote {cfg.work_dir / 'selected_events.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
