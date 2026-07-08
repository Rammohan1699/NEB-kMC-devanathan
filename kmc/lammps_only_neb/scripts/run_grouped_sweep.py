#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from development.lammps_only_neb.config import LammpsOnlyNEBConfig
from development.lammps_only_neb.geometry import (
    bcc_fe_positions,
    build_batch_from_sites,
    knn_sites,
    tetrahedral_sites,
)
from development.lammps_only_neb.runner import load_lammps_class, run_batch_slots
from development.lammps_only_neb.scheduler import slots_for_group, split_group_comm


DEFAULT_LAMMPS_PYTHON = REPO_ROOT / "development" / "tests" / "native_lammps_neb_test"
DEFAULT_LAMMPS_LIB_DIR = Path("/Users/rtirunelveli/mylammps/build2")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one grouped LAMMPS-only NEB 6-neighbor sweep.")
    parser.add_argument("--work-dir", type=Path, default=Path("development/lammps_only_neb_runs/grouped_sweep"))
    parser.add_argument("--potential", type=Path, default=REPO_ROOT / "kmc" / "PotentialB3410-modified.fs")
    parser.add_argument("--lammps-python", type=Path, default=DEFAULT_LAMMPS_PYTHON)
    parser.add_argument("--lammps-lib-dir", type=Path, default=DEFAULT_LAMMPS_LIB_DIR)
    parser.add_argument("--lattice-a", type=float, default=2.8601)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=30)
    parser.add_argument("--nz", type=int, default=30)
    parser.add_argument("--knn", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--h-site", type=int, default=None)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--shell-inner-radius", type=float, default=8.0)
    parser.add_argument("--shell-outer-radius", type=float, default=10.0)
    parser.add_argument("--shell-pad", type=float, default=2.0)
    parser.add_argument("--min-style", choices=("quickmin", "fire"), default="quickmin")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ftol", type=float, default=1.0e-5)
    parser.add_argument("--debug-mode", action="store_true", help="Keep LAMMPS input, log, screen, and trajectory files.")
    return parser.parse_args(argv)


def write_summary(path: Path, rows) -> None:
    fields = [
        "slot",
        "h_site",
        "n_site",
        "barrier_eV",
        "ebf",
        "ebr",
        "wall_s",
        "atoms",
        "frozen",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in sorted(rows, key=lambda item: item.slot):
            data = asdict(result)
            writer.writerow({field: data.get(field, "") for field in fields})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        from mpi4py import MPI
    except Exception as exc:
        raise SystemExit(f"mpi4py is required: {exc}") from exc

    cfg = LammpsOnlyNEBConfig(
        replicas_per_neb=int(args.replicas),
        force_tolerance=float(args.ftol),
        max_steps=int(args.steps),
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

    world = MPI.COMM_WORLD
    world_rank = world.Get_rank()
    group_id, n_groups, group_comm = split_group_comm(world, cfg.replicas_per_neb)
    group_rank = group_comm.Get_rank()

    if world_rank == 0:
        print(
            f"[rank 0] grouped sweep: mpi_size={world.Get_size()}, groups={n_groups}, "
            f"replicas={cfg.replicas_per_neb}",
            flush=True,
        )

    box = np.array([args.nx * args.lattice_a, args.ny * args.lattice_a, args.nz * args.lattice_a], dtype=float)
    sites = tetrahedral_sites(args.nx, args.ny, args.nz, args.lattice_a)
    neighbors = knn_sites(sites, box, int(args.knn))
    rng = np.random.default_rng(args.seed)
    h_site = int(rng.integers(0, len(sites))) if args.h_site is None else int(args.h_site)
    fe_positions = bcc_fe_positions(args.nx, args.ny, args.nz, args.lattice_a)
    batch = build_batch_from_sites(
        env_key=("grouped_sweep", int(h_site)),
        h_site=h_site,
        sites=sites,
        neighbor_indices=neighbors[h_site][: int(args.knn)],
        fe_positions=fe_positions,
        box=box,
        lattice_a=args.lattice_a,
    )

    lammps_class = load_lammps_class(cfg)
    slots = slots_for_group(range(len(batch.hops)), group_id, n_groups)
    if group_rank == 0:
        print(f"[group {group_id}] assigned slots {slots}", flush=True)

    world.Barrier()
    t0 = time.perf_counter()
    local_results = run_batch_slots(
        batch,
        slots,
        group_comm=group_comm,
        group_work_dir=cfg.work_dir / f"group_{group_id}",
        cfg=cfg,
        lammps_class=lammps_class,
    )
    world.Barrier()
    sweep_wall_s = time.perf_counter() - t0

    gathered = world.gather(local_results if group_rank == 0 else [], root=0)
    if world_rank == 0:
        results = [row for part in gathered for row in part]
        write_summary(cfg.work_dir / "sweep_summary.csv", results)
        print("[rank 0] results:", flush=True)
        for result in sorted(results, key=lambda item: item.slot):
            print(
                f"  slot {result.slot}: n={result.n_site} barrier={result.barrier_eV:.9g} eV "
                f"wall={result.wall_s:.6f}s atoms={result.atoms} frozen={result.frozen}",
                flush=True,
            )
        print(f"[rank 0] total sweep wall: {sweep_wall_s:.6f} s", flush=True)
        print(f"[rank 0] summary: {cfg.work_dir / 'sweep_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
