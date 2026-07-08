from __future__ import annotations

import gc
import importlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .config import LammpsOnlyNEBConfig
from .geometry import build_shell_system
from .lammps_io import (
    choose_barrier,
    parse_last_barriers,
    write_endpoint_minimize_script,
    write_final_coords,
    write_final_coords_from_positions,
    write_input_script,
    write_lammps_data,
)
from .models import EnvironmentNEBBatch, HopResult


def fd_status() -> dict[str, int | str]:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    count: int | str = "unknown"
    for path in ("/proc/self/fd", "/dev/fd"):
        try:
            count = len(os.listdir(path))
            break
        except OSError:
            continue
    return {
        "open_fds": count,
        "soft_limit": int(soft) if soft >= 0 else "unlimited",
        "hard_limit": int(hard) if hard >= 0 else "unlimited",
    }


def load_lammps_class(cfg: LammpsOnlyNEBConfig):
    if cfg.lammps_python_path is not None and str(cfg.lammps_python_path) not in sys.path:
        sys.path.insert(0, str(cfg.lammps_python_path))
    if cfg.lammps_lib_dir is not None:
        old = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = f"{cfg.lammps_lib_dir}{':' + old if old else ''}"
    module = importlib.import_module("lammps")
    return getattr(module, "lammps")


def write_job_files(batch: EnvironmentNEBBatch, hop_slot: int, run_dir: Path, cfg: LammpsOnlyNEBConfig):
    hop = batch.hops[int(hop_slot)]
    system = build_shell_system(batch, hop, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_lammps_data(run_dir / "init.data", system)
    write_final_coords(run_dir / "coords.final", system)
    write_input_script(run_dir / "in.neb.lmp", system, cfg)
    metadata = {
        "env_key": str(batch.env_key),
        "h_site": int(hop.h_site),
        "n_site": int(hop.n_site),
        "slot": int(hop.slot),
        "atoms": int(len(system.positions)),
        "frozen": int(len(system.frozen_ids)),
        "shell_inner_radius_a": float(cfg.shell_inner_radius_a),
        "shell_outer_radius_a": float(cfg.shell_outer_radius_a),
        "endpoint_optimize": bool(cfg.endpoint_optimize),
        "endpoint_force_tolerance": float(cfg.endpoint_force_tolerance),
        "endpoint_max_steps": int(cfg.endpoint_max_steps),
        "fd_status_before_lammps": fd_status(),
    }
    (run_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return hop, system


def write_barrier_summary(path: Path, result: HopResult) -> None:
    summary = {
        "env_key": str(result.env_key),
        "h_site": int(result.h_site),
        "n_site": int(result.n_site),
        "slot": int(result.slot),
        "barrier_eV": float(result.barrier_eV),
        "ebf": None if result.ebf is None else float(result.ebf),
        "ebr": None if result.ebr is None else float(result.ebr),
        "wall_s": float(result.wall_s),
        "atoms": int(result.atoms),
        "frozen": int(result.frozen),
        "endpoint_opt_s": float(result.endpoint_opt_s),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def remove_job_scratch(run_dir: Path, *, stop_at: Path) -> None:
    shutil.rmtree(run_dir, ignore_errors=True)
    current = run_dir.parent
    stop_at = stop_at.resolve()
    while current != stop_at and stop_at in current.resolve().parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def run_lammps_job(run_dir: Path, cfg: LammpsOnlyNEBConfig, comm, lammps_class) -> None:
    screen_target = "screen.lammps" if cfg.debug_mode else "none"
    partition_screen_target = "screen.partition" if cfg.debug_mode else "none"
    partition_log_target = "log.partition" if cfg.debug_mode else "none"
    cmdargs = [
        "-echo",
        "none",
        "-partition",
        f"{cfg.replicas_per_neb}x1",
        "-in",
        "none",
        "-log",
        "log.lammps",
        "-plog",
        partition_log_target,
        "-screen",
        screen_target,
        "-pscreen",
        partition_screen_target,
    ]
    old_cwd = Path.cwd()
    os.chdir(run_dir)
    lmp = None
    try:
        lmp = lammps_class(cmdargs=cmdargs, comm=comm)
        lmp.file("in.neb.lmp")
    finally:
        if lmp is not None:
            try:
                # Repeated in-process NEB jobs can leave the active LAMMPS log
                # descriptor open on some builds unless logging is disabled
                # before clearing/closing the instance.
                lmp.command("log none")
            except Exception:
                pass
            try:
                lmp.command("clear")
            except Exception:
                pass
            try:
                lmp.close()
            except Exception:
                pass
            del lmp
            gc.collect()
        try:
            if cfg.debug_mode:
                (Path.cwd() / "fd_status_after_lammps.json").write_text(
                    json.dumps(fd_status(), indent=2),
                    encoding="utf-8",
                )
        finally:
            os.chdir(old_cwd)


def _final_endpoint_positions(system) -> np.ndarray:
    positions = np.asarray(system.positions, dtype=float).copy()
    for atom_id, pos in system.final_positions.items():
        positions[int(atom_id) - 1] = np.asarray(pos, dtype=float)
    return positions


def _gather_lammps_positions(lmp) -> np.ndarray:
    n_atoms = int(lmp.get_natoms())
    raw = lmp.gather_atoms("x", 1, 3)
    try:
        arr = np.ctypeslib.as_array(raw, shape=(n_atoms * 3,))
    except Exception:
        arr = np.asarray(raw, dtype=float)
    return np.asarray(arr, dtype=float).reshape((n_atoms, 3)).copy()


def run_endpoint_minimize_job(endpoint_dir: Path, cfg: LammpsOnlyNEBConfig, comm, lammps_class) -> np.ndarray:
    screen_target = "screen.lammps" if cfg.debug_mode else "none"
    cmdargs = [
        "-echo",
        "none",
        "-in",
        "none",
        "-log",
        "log.lammps",
        "-screen",
        screen_target,
    ]
    old_cwd = Path.cwd()
    os.chdir(endpoint_dir)
    lmp = None
    try:
        lmp = lammps_class(cmdargs=cmdargs, comm=comm)
        lmp.file("in.minimize.lmp")
        positions = _gather_lammps_positions(lmp)
        return positions
    finally:
        if lmp is not None:
            try:
                lmp.command("log none")
            except Exception:
                pass
            try:
                lmp.command("clear")
            except Exception:
                pass
            try:
                lmp.close()
            except Exception:
                pass
            del lmp
            gc.collect()
        os.chdir(old_cwd)


def run_endpoint_optimization_for_neb(
    run_dir: Path,
    system,
    *,
    cfg: LammpsOnlyNEBConfig,
    group_comm,
    lammps_class,
) -> float:
    group_rank = group_comm.Get_rank()
    if group_rank == 0:
        if system is None:
            raise RuntimeError("Endpoint optimization requires LammpsSystem on group rank 0")
        endpoints = {
            "initial": np.asarray(system.positions, dtype=float),
            "final": _final_endpoint_positions(system),
        }
        for label, positions in endpoints.items():
            endpoint_dir = run_dir / f"endpoint_opt_{label}"
            endpoint_dir.mkdir(parents=True, exist_ok=True)
            write_lammps_data(endpoint_dir / "endpoint.data", system, positions=positions)
            write_endpoint_minimize_script(endpoint_dir / "in.minimize.lmp", system, cfg)
    group_comm.Barrier()

    t0 = time.perf_counter()
    initial_min = run_endpoint_minimize_job(run_dir / "endpoint_opt_initial", cfg, group_comm, lammps_class)
    group_comm.Barrier()
    final_min = run_endpoint_minimize_job(run_dir / "endpoint_opt_final", cfg, group_comm, lammps_class)
    group_comm.Barrier()
    endpoint_opt_s = time.perf_counter() - t0

    if group_rank == 0:
        if not np.isfinite(initial_min).all() or not np.isfinite(final_min).all():
            raise RuntimeError("Endpoint minimization produced non-finite coordinates")
        if initial_min.shape != system.positions.shape or final_min.shape != system.positions.shape:
            raise RuntimeError(
                f"Endpoint minimization shape mismatch: initial={initial_min.shape}, "
                f"final={final_min.shape}, expected={system.positions.shape}"
            )
        write_lammps_data(run_dir / "init.data", system, positions=initial_min)
        write_final_coords_from_positions(run_dir / "coords.final", final_min)
        write_input_script(run_dir / "in.neb.lmp", system, cfg)
        if cfg.debug_mode:
            np.savetxt(run_dir / "endpoint_initial_minimized.xyz.txt", initial_min)
            np.savetxt(run_dir / "endpoint_final_minimized.xyz.txt", final_min)
    group_comm.Barrier()
    return endpoint_opt_s


def run_batch_slots(
    batch: EnvironmentNEBBatch,
    slots: list[int],
    *,
    group_comm,
    group_work_dir: Path,
    cfg: LammpsOnlyNEBConfig,
    lammps_class,
) -> list[HopResult]:
    results: list[HopResult] = []
    group_rank = group_comm.Get_rank()
    batch_cleanup_s = 0.0
    for slot in slots:
        run_dir = group_work_dir / f"neighbor_{int(slot):02d}"
        io_write_s = 0.0
        endpoint_opt_s = 0.0
        hop = None
        system = None
        if group_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_t0 = time.perf_counter()
            hop, system = write_job_files(batch, int(slot), run_dir, cfg)
            io_write_s = time.perf_counter() - write_t0
        group_comm.Barrier()
        if cfg.endpoint_optimize:
            endpoint_opt_s = run_endpoint_optimization_for_neb(
                run_dir,
                system,
                cfg=cfg,
                group_comm=group_comm,
                lammps_class=lammps_class,
            )
        t0 = time.perf_counter()
        run_lammps_job(run_dir, cfg, group_comm, lammps_class)
        group_comm.Barrier()
        lammps_run_s = time.perf_counter() - t0
        if group_rank == 0:
            parse_t0 = time.perf_counter()
            ebf, ebr = parse_last_barriers(run_dir / "log.lammps")
            io_parse_s = time.perf_counter() - parse_t0
            barrier = choose_barrier(ebf, ebr)
            result = HopResult(
                env_key=batch.env_key,
                hop_key=hop.hop_key,
                h_site=hop.h_site,
                n_site=hop.n_site,
                slot=hop.slot,
                barrier_eV=barrier,
                ebf=ebf,
                ebr=ebr,
                wall_s=io_write_s + endpoint_opt_s + lammps_run_s + io_parse_s,
                atoms=len(system.positions),
                frozen=len(system.frozen_ids),
                run_dir=run_dir,
                io_write_s=io_write_s,
                endpoint_opt_s=endpoint_opt_s,
                lammps_run_s=lammps_run_s,
                io_parse_s=io_parse_s,
            )
            if cfg.debug_mode:
                write_barrier_summary(run_dir / "barrier.json", result)
            results.append(result)
        group_comm.Barrier()
    if not cfg.debug_mode and group_rank == 0:
        cleanup_t0 = time.perf_counter()
        shutil.rmtree(group_work_dir, ignore_errors=True)
        batch_cleanup_s = time.perf_counter() - cleanup_t0
        if results:
            per_result_cleanup_s = batch_cleanup_s / float(len(results))
            updated_results: list[HopResult] = []
            for result in results:
                updated_results.append(
                    HopResult(
                        env_key=result.env_key,
                        hop_key=result.hop_key,
                        h_site=result.h_site,
                        n_site=result.n_site,
                        slot=result.slot,
                        barrier_eV=result.barrier_eV,
                        ebf=result.ebf,
                        ebr=result.ebr,
                        wall_s=result.wall_s + per_result_cleanup_s,
                        atoms=result.atoms,
                        frozen=result.frozen,
                        run_dir=result.run_dir,
                        io_write_s=result.io_write_s,
                        endpoint_opt_s=result.endpoint_opt_s,
                        lammps_run_s=result.lammps_run_s,
                        io_parse_s=result.io_parse_s,
                        io_cleanup_s=per_result_cleanup_s,
                    )
                )
            results = updated_results
    group_comm.Barrier()
    return results
