"""
Top-level MPI KMC/NEB simulation driver.

The driver owns the production simulation loop for hydrogen diffusion:
  - loads either generated BCC interstitial sites or an external site map
  - initializes occupied H sites and rank-local logging/timing state
  - builds candidate H hops from KNN site connectivity
  - reuses cached migration barriers when the local environment key matches
  - schedules cache-miss NEB calculations across MPI ranks
  - folds NEB barriers into rates and selects a rejection-free KMC event
  - updates H positions, trajectories, restart checkpoints, MSD data, and
    per-rank/global barrier caches

Large pieces that used to live in the monolithic ``cache_new.py`` have been
split into focused modules for configuration, MPI coordination, scheduling,
event handling, structure construction, LAMMPS calculator setup, and cache
management. This module ties those services together and preserves the
environment-driven run interface used by the launcher scripts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence
import csv
import faulthandler
import json
import os
import pickle
import re
import random
import signal
import time
import traceback
from itertools import count
from pathlib import Path

import numpy as np

try:
    from ase import Atoms
    from ase.io import write
    from ase.optimize import BFGS
except ImportError:  # Native LAMMPS-only generated runs do not need ASE.
    Atoms = Any  # type: ignore
    write = None  # type: ignore
    BFGS = None  # type: ignore

try:
    from .config import load_config
except Exception:
    load_config = None  # type: ignore

try:
    from .mpi_context import MPIContext
    from .logging_utils import RankLogger, StepTimer, TimingCSVWriter, log_step_timing
    from .lattice import generate_all_interstitial_sites, get_k_nearest_neighbors, pbc_diff
    from .environment import build_hydrogen_kdtree, make_env_key
    from .structures import (
        SupercellSpec,
        build_full_structure,
        build_full_structure_from_host,
        update_full_structure_h_atom,
        build_fe_3a_template,
        build_h_cell_index,
        load_kmc_site_map,
        load_lammps_atomic_structure,
        build_local_neb_structures_fast,
        build_local_neb_structures_shell,
        build_local_neb_structures_shell_from_host,
        assign_mass_and_type,
    )
    from .lammps_factory import LammpsCalculatorFactory, LammpsCalculatorConfig
    from .scheduler import SchedulerConfig, MPIJobScheduler, chunk_jobs
    from .event_manager import (
        KineticParameters,
        EventManagerConfig,
        build_candidate_events,
        fold_neb_results_into_events,
        merge_candidate_results,
        select_rejection_free_event,
        apply_selected_event_to_sites,
    )
    from .incremental_events import (
        IncrementalEventTable,
        affected_h_sites_for_particle_changes,
        affected_h_sites_for_move,
        max_neighbor_hop_distance,
    )
    from .devanathan import (
        DevanathanBoundary,
        DevanathanBoundaryConfig,
        DevanathanPulseBoundary,
        DevanathanUpdate,
        filter_fixed_x_neighbors,
        filter_x_direction_neighbors,
    )
    from .devanathan_gcmc import GCMCDevanathanBoundary, GCMCReservoirConfig
    from .gcmc_energy_cache import (
        PersistentGCMCEnergyCache,
        canonical_insertion_environment_key,
    )
    from .services.cache_manager import CacheManagerConfig, BarrierCacheManager, merge_rank_caches
    from .msd_tracker import MSDTracker
    from .lammps_only_neb.config import LammpsOnlyNEBConfig
    from .lammps_only_neb.geometry import bcc_fe_positions, build_batch_from_sites
    from .lammps_only_neb.models import EnvironmentNEBBatch
    from .lammps_only_neb.runner import load_lammps_class as load_native_lammps_class
    from .lammps_only_neb.runner import run_batch_slots as run_native_batch_slots
    from .lammps_only_neb.scheduler import split_group_comm
    try:
        from .engines.lammps_neb import LammpsNEB
    except Exception:
        LammpsNEB = None  # type: ignore
except ImportError:
    from mpi_context import MPIContext
    from logging_utils import RankLogger, StepTimer, TimingCSVWriter, log_step_timing
    from lattice import generate_all_interstitial_sites, get_k_nearest_neighbors, pbc_diff
    from environment import build_hydrogen_kdtree, make_env_key
    from structures import (
        SupercellSpec,
        build_full_structure,
        build_full_structure_from_host,
        update_full_structure_h_atom,
        build_fe_3a_template,
        build_h_cell_index,
        load_kmc_site_map,
        load_lammps_atomic_structure,
        build_local_neb_structures_fast,
        build_local_neb_structures_shell,
        build_local_neb_structures_shell_from_host,
        assign_mass_and_type,
    )
    from lammps_factory import LammpsCalculatorFactory, LammpsCalculatorConfig
    from scheduler import SchedulerConfig, MPIJobScheduler, chunk_jobs
    from event_manager import (
        KineticParameters,
        EventManagerConfig,
        build_candidate_events,
        fold_neb_results_into_events,
        merge_candidate_results,
        select_rejection_free_event,
        apply_selected_event_to_sites,
    )
    from incremental_events import (
        IncrementalEventTable,
        affected_h_sites_for_particle_changes,
        affected_h_sites_for_move,
        max_neighbor_hop_distance,
    )
    from devanathan import (
        DevanathanBoundary,
        DevanathanBoundaryConfig,
        DevanathanPulseBoundary,
        DevanathanUpdate,
        filter_fixed_x_neighbors,
        filter_x_direction_neighbors,
    )
    from devanathan_gcmc import GCMCDevanathanBoundary, GCMCReservoirConfig
    from gcmc_energy_cache import (
        PersistentGCMCEnergyCache,
        canonical_insertion_environment_key,
    )
    from services.cache_manager import CacheManagerConfig, BarrierCacheManager, merge_rank_caches
    from msd_tracker import MSDTracker
    from lammps_only_neb.config import LammpsOnlyNEBConfig
    from lammps_only_neb.geometry import bcc_fe_positions, build_batch_from_sites
    from lammps_only_neb.models import EnvironmentNEBBatch
    from lammps_only_neb.runner import load_lammps_class as load_native_lammps_class
    from lammps_only_neb.runner import run_batch_slots as run_native_batch_slots
    from lammps_only_neb.scheduler import split_group_comm
    try:
        from engines.lammps_neb import LammpsNEB
    except Exception:
        LammpsNEB = None  # type: ignore


_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]")
_STRUCTURE_SNAPSHOT_SEQ = count()


@dataclass
class NEBDiagnosticSummary:
    step: int
    rank: int
    neb_assigned: int = 0
    neb_ran: int = 0
    neb_batches: int = 0
    neb_wall_s: float = 0.0
    neb_avg_s: float = 0.0
    neb_struct_s: float = 0.0
    neb_setup_s: float = 0.0
    neb_preprocess_s: float = 0.0
    neb_preprocess_avg_s: float = 0.0
    neb_wait_s: float = 0.0
    neb_io_write_s: float = 0.0
    neb_lammps_run_s: float = 0.0
    neb_io_parse_s: float = 0.0
    neb_io_cleanup_s: float = 0.0
    neb_engine_total_s: float = 0.0
    neb_engine_avg_s: float = 0.0
    neb_total_with_preprocess_s: float = 0.0
    neb_total_with_preprocess_avg_s: float = 0.0
    neb_header_s: float = 0.0
    neb_endpoint_constraints_s: float = 0.0
    neb_endpoint_opt_s: float = 0.0
    neb_image_setup_s: float = 0.0
    neb_image_constraints_s: float = 0.0
    neb_interpolate_s: float = 0.0
    neb_optimize_s: float = 0.0
    neb_energy_eval_s: float = 0.0
    neb_failure_cleanup_s: float = 0.0
    neb_avg_atoms: float = 0.0
    neb_avg_fe: float = 0.0
    neb_avg_h: float = 0.0
    neb_avg_frozen: float = 0.0

    @classmethod
    def fields(cls) -> list[str]:
        return list(cls.__dataclass_fields__.keys())

    def row(self) -> dict[str, Any]:
        return asdict(self)


class DebugModeRankLogger(RankLogger):
    """Rank logger that creates log files only when debug mode is enabled.

    DEBUG_MODE=1 / DEBUG_LOGGING=1: write per-rank log files and mirror rank-0 stdout.
    DEBUG_MODE=0 / DEBUG_LOGGING=0: no log files; rank 0 still prints progress/errors.
    """

    def log(self, msg: str, step: Optional[int] = None, banner: bool = False) -> None:
        if self.rank != 0:
            return
        if self.debug:
            super().log(msg, step=step, banner=banner)
            return
        if self.mirror_root_to_stdout:
            print(msg)

    def __call__(self, msg: str, step: Optional[int] = None, banner: bool = False) -> None:
        self.log(msg, step=step, banner=banner)


def _sanitize_filename_component(value: Any) -> str:
    text = str(value)
    if not text:
        return ""
    return _FILENAME_SANITIZER.sub("_", text)


@dataclass
class DriverDefaults:
    lattice_a: float = 2.8601
    nx: int = 30
    ny: int = 30
    nz: int = 30
    num_h: int = 540
    steps: int = 10000

    knn_k: int = 6
    same_type_neighbors: bool = False

    env_radius_a: float = 5.0
    env_affect_radius_a: float = 6.0
    shell_inner_radius_a: float = 5.0
    shell_outer_radius_a: float = 6.0

    nimg: int = 3
    engine: str = "lammps_only"
    replicas_per_neb: int = 3
    native_spring_constant: float = 5.0
    native_force_tolerance: float = 1.0e-5
    native_max_steps: int = 1000
    native_climbing_steps: int = 0
    native_nevery: int = 50
    native_thermo_every: int = 50
    native_timestep: float = 0.01
    native_min_style: str = "quickmin"
    native_scratch_dir: str = "neb/lammps_only_scratch"
    native_debug_mode: bool = False
    native_endpoint_optimize: bool = False
    native_endpoint_min_style: str = "quickmin"
    native_endpoint_energy_tolerance: float = 0.0
    native_endpoint_force_tolerance: float = 0.05
    native_endpoint_max_steps: int = 200
    native_endpoint_max_evals: int = 2000
    native_endpoint_dmax: float = 0.1
    temperature_k: float = 300.0
    attempt_frequency_hz: float = 1.0e13

    cache_schema: str = ""
    job_assignment_mode: str = "cache_dedupe"
    barrier_merge_mode: str = "global"
    barrier_merge_interval: int = 100
    neb_min_batch: int = 64

    kmc_seed: Optional[int] = 42
    local_env_mode: str = "radial"
    shrinkwrap_pad_a: float = 2.0

    potential: str = "EAM"
    potential_eam_file: str = "PotentialB3410-modified.fs"
    potential_nnp_dir: str = ""
    lammps_engine: str = "eam_fs"
    lammps_pair_style: str = ""
    lammps_pair_coeff: str = ""
    lammps_files: str = "PotentialB3410-modified.fs"
    python_path: str = ""
    lib_dir: str = ""

    nnp_dir: str = ""
    nnp_elements: str = "Fe H"
    nnp_cutoff: float = 6.60
    nnp_showewsum: int = 1000
    nnp_maxew: int = 1000000
    nnp_cflength: float = 1.8897261328
    nnp_cfenergy: float = 0.0367493254

    optimize_endpoints: bool = True
    low_barrier_ev: float = 1.0e-3
    debug_logging: bool = True
    validation_enabled: bool = False
    validation_out: str = "neb/validation_dump"
    validation_overwrite: bool = True
    validation_ase_dump: bool = False
    validation_ase_dump_pre: bool = False
    validation_ase_dir: str = "ase"

    site_source: str = "generated"
    site_map_file: str = ""
    host_structure_file: str = ""
    host_fe_type: int = 1
    initial_h_region_file: str = ""
    initial_h_regions: str = ""


class KMCSimulation:
    def __init__(self, cfg: Any | None = None):
        self.cfg = cfg if cfg is not None else DriverDefaults()
        self.ctx = MPIContext.create()
        self._append_outputs = self._debug_flag("KMC_APPEND_OUTPUTS", False)

        debug = bool(self._cfg_value("debug_logging", "DEBUG_LOGGING", "DEBUG_MODE", default=True))
        self.log = DebugModeRankLogger(
            rank=self.ctx.rank,
            debug=debug,
            log_dir=str(self._logs_dir()),
            reset_on_start=not self._append_outputs,
        )
        if self._append_outputs:
            self.log(f"[Rank {self.ctx.rank}] Appending restart output to existing run directory")

        self.spec = self._make_supercell_spec()
        self.kinetics = self._make_kinetics()

        self.cache_mgr = self._make_cache_manager()
        self.scheduler = self._make_scheduler()

        self.lammps_factory = self._make_lammps_factory()
        self.gcmc_energy_cache = self._make_gcmc_energy_cache()
        self._gcmc_energy_cache_step_hits = 0
        self._gcmc_energy_cache_step_misses = 0
        self.calc_pool = None
        self.neb_engine = None
        self.native_lammps_class = None
        self.native_neb_group_id = 0
        self.native_neb_group_count = 1
        self.native_neb_group_comm = None
        if self._lammps_only_enabled():
            replicas = int(self._cfg_value("replicas_per_neb", "LAMMPS_NEB_REPLICAS", "NATIVE_NEB_REPLICAS", default=3))
            self.native_neb_group_id, self.native_neb_group_count, self.native_neb_group_comm = split_group_comm(
                self.ctx.comm,
                replicas,
            )

        self.sites: Optional[np.ndarray] = None
        self.site_types: Optional[np.ndarray] = None
        self.neighbors: Optional[list[list[int]]] = None
        self.h_indices: Optional[np.ndarray] = None
        self.full_structure = None
        self.fe_3a_template = None
        self.external_host_structure: Optional[Atoms] = None
        self.external_host_fe_positions: Optional[np.ndarray] = None
        self._external_host_fe_tree = None
        self._gcmc_calculator = None
        self._gcmc_energy_evaluations = 0
        self._gcmc_progress_header_written = False
        self.h_cell_index = {}
        self._env_key_h_positions: Optional[np.ndarray] = None
        self._env_key_h_tree = None
        self._env_key_h_signature: Optional[tuple[int, ...]] = None
        self._incremental_event_table: Optional[IncrementalEventTable] = None
        self._incremental_rebuild_h_sites: Optional[list[int]] = None
        self._incremental_remove_h_sites: Optional[list[int]] = None
        self._incremental_hop_radius_a: Optional[float] = None

        self.failed_barrier_keys: set[Any] = set()
        self.reported_failed_rate_keys: set[Any] = set()
        self.sim_time_s = 0.0
        self.h_unwrapped_positions: Optional[np.ndarray] = None
        self.h_initial_unwrapped_positions: Optional[np.ndarray] = None
        self.msd_tracker: Optional[MSDTracker] = None
        self.devanathan_boundary: Optional[DevanathanBoundary] = None
        self._pending_devanathan_state: Optional[Mapping[str, Any]] = None
        self._devanathan_header_written = False
        self.timing_writer = TimingCSVWriter(
            str(self._logs_dir() / f"timing_rank{self.ctx.rank}.csv"),
            enabled=debug and self.ctx.rank == 0,
            append_existing=self._append_outputs,
        )
        self._rates_header_written = self._append_output_exists(
            self._diagnostics_output_dir() / "rates_allranks.csv"
        )
        self._neb_diag_header_written = self._append_output_exists(
            self._neb_output_dir() / f"neb_diag_rank{self.ctx.rank}.csv"
        )
        self._neb_diag_all_header_written = self._append_output_exists(
            self._neb_output_dir() / "neb_diag_all.csv"
        )
        self._kmc_diag_initialized = self._append_output_exists(
            self._diagnostics_output_dir() / "kmc_diagnostics_rank0.log"
        )
        self._selected_events_header_written = self._append_output_exists(
            self._diagnostics_output_dir() / "kmc_selected_events.csv"
        )
        self._timestep_header_written = self._append_output_exists(
            self._diagnostics_output_dir() / "kmc_timestep_vs_step.csv"
        )
        self._only_h_traj_initialized = self._append_output_exists(
            self._trajectory_output_dir() / "H_trajectory_onlyH.lammpstrj"
        )
        self._devanathan_header_written = self._append_output_exists(
            self._diagnostics_output_dir() / "kmc_devanathan.csv"
        )
        self.restart_start_step = 0
        self._stack_dump_file = None

        seed_value = self._cfg_value("kmc_seed", "KMC_SEED", default=42)
        if seed_value in (None, "", "None"):
            seed_value = 42
        try:
            base_seed = int(seed_value)
        except (TypeError, ValueError):
            base_seed = abs(hash(str(seed_value))) & 0xFFFFFFFF
        rank_seed = (base_seed + self.ctx.rank) & 0xFFFFFFFF
        np.random.seed(rank_seed)
        random.seed(rank_seed)
        self.rng = random.Random(rank_seed)
        self._install_stack_dump_handler()

    def _append_output_exists(self, path: Path) -> bool:
        return bool(
            self._append_outputs
            and path.exists()
            and path.stat().st_size > 0
        )

    def _install_stack_dump_handler(self) -> None:
        raw = os.environ.get("KMC_STACK_DUMP_SIGNAL", "1")
        if str(raw).lower() in {"0", "false", "no", "off", ""}:
            return
        sig = getattr(signal, "SIGUSR1", None)
        if sig is None:
            return
        try:
            self._stack_dump_file = open(
                self._logs_dir() / f"stack_rank{self.ctx.rank}.log",
                "a",
                encoding="utf-8",
                buffering=1,
            )
            faulthandler.register(sig, file=self._stack_dump_file, all_threads=True, chain=False)
        except Exception as exc:
            try:
                self.log(f"[Rank {self.ctx.rank}] Could not install SIGUSR1 stack dump handler: {exc}")
            except Exception:
                pass

    def _heartbeat(self, step: int, phase: str, point: str) -> None:
        raw = os.environ.get("KMC_HEARTBEAT", os.environ.get("KMC_PHASE_HEARTBEAT", "0"))
        if str(raw).lower() in {"0", "false", "no", "off", ""}:
            return
        try:
            with (self._logs_dir() / f"heartbeat_rank{self.ctx.rank}.txt").open("w", encoding="utf-8") as fh:
                fh.write(f"time_epoch={time.time():.6f}\n")
                fh.write(f"rank={self.ctx.rank}\n")
                fh.write(f"step={int(step)}\n")
                fh.write(f"phase={str(phase)}\n")
                fh.write(f"point={str(point)}\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Config adapters
    # ------------------------------------------------------------------
    @staticmethod
    def _env_value(name: str, default=None):
        if name not in os.environ:
            return None
        raw = os.environ[name]
        if isinstance(default, bool):
            return str(raw).lower() not in {"0", "false", "no", "off"}
        if isinstance(default, int) and not isinstance(default, bool):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        return raw

    def _cfg_value(self, *names: str, default=None):
        for name in names:
            if name.isupper():
                value = self._env_value(name, default)
                if value is not None:
                    return value
        for name in names:
            if hasattr(self.cfg, name):
                return getattr(self.cfg, name)
            for section in ("size", "site_map", "validation", "diagnostics", "environment", "neb", "lammps", "cache", "restart", "paths"):
                nested = getattr(self.cfg, section, None)
                if nested is not None and hasattr(nested, name):
                    return getattr(nested, name)
        return default

    def _debug_flag(self, name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return str(raw).lower() not in {"0", "false", "no", "off"}

    def _run_subdir(self, *names: str, default: str) -> Path:
        raw = str(self._cfg_value(*names, default=default) or default).strip()
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _logs_dir(self) -> Path:
        return self._run_subdir("log_dir", "KMC_LOG_DIR", "LOG_DIR", default="logs")

    def _write_exception_log(self, label: str, exc: BaseException) -> None:
        try:
            path = self._logs_dir() / f"exception_rank{self.ctx.rank}.log"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} rank={self.ctx.rank} label={label} ---\n")
                fh.write(f"group_id={self.native_neb_group_id} group_count={self.native_neb_group_count}\n")
                fh.write(f"exception={type(exc).__name__}: {exc}\n")
                traceback.print_exc(file=fh)
        except Exception:
            pass

    def _neb_output_dir(self) -> Path:
        return self._run_subdir("neb_output_dir", "KMC_NEB_DIR", "NEB_OUTPUT_DIR", default="neb")

    def _diagnostics_output_dir(self) -> Path:
        return self._run_subdir("diagnostics_output_dir", "KMC_DIAGNOSTICS_DIR", default="diagnostics")

    def _trajectory_output_dir(self) -> Path:
        return self._run_subdir("trajectory_output_dir", "KMC_TRAJECTORY_DIR", default="trajectories")

    def _cache_output_dir(self) -> Path:
        return self._run_subdir("cache_output_dir", "KMC_CACHE_DIR", "CACHE_DIR", default="cache")

    def _dump_every_steps(self) -> int:
        return max(1, int(self._cfg_value("dump_every_steps", "DUMP_EVERY_STEPS", default=1)))

    def _should_periodic_dump(self, step: int) -> bool:
        interval = self._dump_every_steps()
        return step == 0 or ((step + 1) % interval == 0)

    def _write_rates_allranks_enabled(self) -> bool:
        return bool(self._cfg_value("WRITE_RATES_ALLRANKS", "RATES_ALLRANKS", default=True))

    def _lammps_only_enabled(self) -> bool:
        engine = str(self._cfg_value("engine", "NEB_ENGINE", "KMC_NEB_ENGINE", default="lammps_only") or "lammps_only")
        return engine.lower().strip() in {
            "lammps_only",
            "native_lammps",
            "lammps-neb-only",
            "lammps_only_endpoint_opt",
            "lammps_only_lmp_endpoint_opt",
            "native_lammps_endpoint_opt",
        }

    def _devanathan_enabled(self) -> bool:
        return bool(self._cfg_value("DEVANATHAN_ENABLED", default=False))

    @staticmethod
    def _normalize_devanathan_source_mode(raw: Any) -> str:
        mode = str(raw or "constant").strip().lower().replace("-", "_")
        if mode == "bulk_gcmc":
            return "gcmc_bulk"
        if mode in {"finite", "finite_pulse", "charged_pulse", "charging_pulse"}:
            return "pulse"
        return mode

    def _stop_when_no_h_remaining_enabled(self) -> bool:
        return bool(
            self._cfg_value(
                "KMC_STOP_WHEN_NO_H",
                "STOP_WHEN_NO_H",
                "DEVANATHAN_STOP_WHEN_EMPTY",
                default=False,
            )
        )

    def _make_devanathan_boundary(self) -> Optional[DevanathanBoundary]:
        if not self._devanathan_enabled():
            return None
        source_mode = self._normalize_devanathan_source_mode(
            self._cfg_value("DEVANATHAN_SOURCE_MODE", default="constant")
        )
        source_min = float(self._cfg_value("DEVANATHAN_SOURCE_X_MIN_A", default=0.0))
        source_layer = float(self._cfg_value("DEVANATHAN_SOURCE_LAYER_A", default=10.0))
        source_max = float(self._cfg_value("DEVANATHAN_SOURCE_X_MAX_A", default=source_min + source_layer))
        sink_x = float(self._cfg_value("DEVANATHAN_SINK_X_MIN_A", default=float(self.spec.box[0])))
        source_fraction = float(self._cfg_value("DEVANATHAN_SOURCE_FRACTION", default=0.01))
        region_label = "source"
        control_site_indices = None
        control_fe_count = None
        left_sink_x_max = float(
            self._cfg_value("DEVANATHAN_LEFT_SINK_X_MAX_A", default=0.0)
        )
        left_sink_x_min = float(
            self._cfg_value("DEVANATHAN_LEFT_SINK_X_MIN_A", default=0.0)
        )
        left_sink_site_indices = None
        left_sink_label = "left_sink"
        if source_mode == "gcmc_bulk":
            bulk_min = float(self._cfg_value("DEVANATHAN_BULK_X_MIN_A", default=source_min))
            bulk_gap = float(
                self._cfg_value(
                    "DEVANATHAN_BULK_SINK_GAP_A",
                    default=max(float(getattr(self.spec, "lattice_a", 0.0)), 1.0e-6),
                )
            )
            default_bulk_max = sink_x - bulk_gap
            if default_bulk_max <= bulk_min:
                default_bulk_max = source_max
            source_min = bulk_min
            source_max = float(self._cfg_value("DEVANATHAN_BULK_X_MAX_A", default=default_bulk_max))
            source_fraction = float(
                self._cfg_value(
                    "DEVANATHAN_BULK_TARGET_H_PER_FE",
                    "DEVANATHAN_BULK_TARGET_FRACTION",
                    default=0.001,
                )
            )
            region_label = "bulk"
            if self._site_source() == "external":
                region_raw = str(
                    self._cfg_value(
                        "DEVANATHAN_BULK_REGION",
                        "DEVANATHAN_BULK_REGIONS",
                        default="bulk_grain_0",
                    )
                    or ""
                ).strip()
                if region_raw:
                    if self.sites is None:
                        raise RuntimeError(
                            "External gcmc_bulk region selection requires sites to be loaded first"
                        )
                    region_specs = [
                        item.strip()
                        for item in region_raw.split(",")
                        if item.strip()
                    ]
                    control_site_indices, control_fe_count = (
                        self._bulk_gcmc_control_from_regions(
                            self.sites,
                            region_specs,
                            x_min_a=source_min,
                            x_max_a=source_max,
                        )
                    )
                    region_label = f"bulk:{'+'.join(region_specs)}"
            target_n_raw = self._cfg_value("DEVANATHAN_BULK_TARGET_N_H", default=None)
            if target_n_raw not in (None, "", "None"):
                target_n_h = int(round(float(target_n_raw)))
                if target_n_h < 0:
                    raise ValueError(
                        "DEVANATHAN_BULK_TARGET_N_H must be non-negative"
                    )
                basis_fe = control_fe_count
                if basis_fe is None:
                    fe_positions = self._native_fe_positions()
                    fe_x = np.asarray(fe_positions, dtype=float)[:, 0]
                    basis_fe = int(
                        np.count_nonzero((fe_x >= source_min) & (fe_x < source_max))
                    )
                if basis_fe <= 0:
                    raise RuntimeError(
                        "DEVANATHAN_BULK_TARGET_N_H requires a non-empty bulk Fe basis"
                )
                source_fraction = target_n_h / float(basis_fe)
                if self.ctx.rank == 0:
                    self.log(
                        "[Rank 0] DEVANATHAN_BULK_TARGET_N_H override: "
                        f"target={target_n_h}, Fe_basis={basis_fe}, "
                        f"H_per_Fe={source_fraction:.12g}"
                    )
        if source_mode == "pulse":
            pulse_n_raw = self._cfg_value(
                "DEVANATHAN_PULSE_N_H",
                "DEVANATHAN_CHARGING_ZONE_N_H",
                "DEVANATHAN_SOURCE_TARGET_N_H",
                "NUM_H",
                default=50,
            )
            pulse_initial_count = int(round(float(pulse_n_raw)))
            if pulse_initial_count < 0:
                raise ValueError("DEVANATHAN_PULSE_N_H must be non-negative")
            region_label = "pulse"
            if self._site_source() == "external":
                region_raw = str(
                    self._cfg_value(
                        "DEVANATHAN_PULSE_REGION",
                        "DEVANATHAN_CHARGING_ZONE_REGION",
                        "DEVANATHAN_SOURCE_REGION",
                        "KMC_INITIAL_H_REGIONS",
                        default="bulk_grain_0",
                    )
                    or ""
                ).strip()
                if region_raw:
                    if self.sites is None:
                        raise RuntimeError(
                            "External pulse region selection requires sites to be loaded first"
                        )
                    region_specs = [
                        item.strip()
                        for item in region_raw.split(",")
                        if item.strip()
                    ]
                    control_site_indices, control_fe_count = (
                        self._control_sites_from_regions_in_x_window(
                            self.sites,
                            region_specs,
                            x_min_a=source_min,
                            x_max_a=source_max,
                            label="charging pulse",
                        )
                    )
                    region_label = f"pulse:{'+'.join(region_specs)}"
        left_sink_region_raw = str(
            self._cfg_value(
                "DEVANATHAN_LEFT_SINK_REGION",
                "DEVANATHAN_LEFT_SINK_REGIONS",
                default="",
            )
            or ""
        ).strip()
        if left_sink_x_max > 0.0 and (left_sink_region_raw or left_sink_x_min > 0.0):
            if self.sites is None:
                raise RuntimeError("Region-restricted left sink requires sites to be loaded first")
            left_sink_specs = [
                item.strip()
                for item in left_sink_region_raw.split(",")
                if item.strip()
            ]
            left_sink_site_indices = self._site_indices_from_regions_in_x_window(
                self.sites,
                left_sink_specs,
                x_min_a=left_sink_x_min,
                x_max_a=left_sink_x_max,
                label="left sink",
            )
            region_text = "+".join(left_sink_specs) if left_sink_specs else "all"
            left_sink_label = f"left_sink:{region_text}"
        seed_raw = self._cfg_value("DEVANATHAN_SEED", default=self._cfg_value("kmc_seed", "KMC_SEED", default=42))
        seed = None if seed_raw in (None, "", "None") else int(seed_raw)
        cfg = DevanathanBoundaryConfig(
            source_x_min_a=source_min,
            source_x_max_a=source_max,
            source_target_fraction=source_fraction,
            left_sink_x_max_a=left_sink_x_max,
            sink_x_min_a=sink_x,
            cross_section_area_a2=float(self._cfg_value("DEVANATHAN_CROSS_SECTION_A2", default=0.0)),
            trim_source_excess=bool(self._cfg_value("DEVANATHAN_TRIM_SOURCE_EXCESS", default=True)),
            seed=seed,
        )
        if source_mode in {"gcmc", "gcmc_bulk"}:
            default_initialization_start = "target" if source_mode == "gcmc_bulk" else "empty"
            default_setpoint_mode = "target" if source_mode == "gcmc_bulk" else "mean"
            default_production_mode = "insert_delete" if source_mode == "gcmc_bulk" else "insert_only"
            default_target_tolerance = source_fraction * 0.5 if source_mode == "gcmc_bulk" else 0.005
            reservoir = GCMCReservoirConfig(
                chemical_potential_ev=float(
                    self._cfg_value("DEVANATHAN_GCMC_MU_EV", default=-1.75)
                ),
                temperature_k=float(
                    self._cfg_value(
                        "DEVANATHAN_GCMC_TEMPERATURE_K",
                        default=self._cfg_value("temperature_k", "TEMPERATURE_K", default=300.0),
                    )
                ),
                boltzmann_ev_per_k=float(
                    self._cfg_value("DEVANATHAN_GCMC_KB_EV_PER_K", default=8.617333262145e-5)
                ),
                attempts_per_adjustment=int(
                    self._cfg_value("DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT", default=1)
                ),
                initialization_start=str(
                    self._cfg_value(
                        "DEVANATHAN_GCMC_INITIALIZATION_START",
                        default=default_initialization_start,
                    )
                ).strip().lower(),
                maintenance_setpoint_mode=str(
                    self._cfg_value(
                        "DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE",
                        default=default_setpoint_mode,
                    )
                ).strip().lower(),
                production_maintenance_mode=str(
                    self._cfg_value(
                        "DEVANATHAN_GCMC_PRODUCTION_MODE",
                        "DEVANATHAN_GCMC_PRODUCTION_MAINTENANCE_MODE",
                        default=default_production_mode,
                    )
                ).strip().lower().replace("-", "_"),
                initialization_min_attempts=int(
                    self._cfg_value("DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS", default=15000)
                ),
                initialization_max_attempts=int(
                    self._cfg_value("DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS", default=50000)
                ),
                initialization_progress_interval=int(
                    self._cfg_value("DEVANATHAN_GCMC_PROGRESS_INTERVAL", default=100)
                ),
                convergence_window_attempts=int(
                    self._cfg_value("DEVANATHAN_GCMC_CONVERGENCE_WINDOW", default=7500)
                ),
                convergence_check_interval=int(
                    self._cfg_value("DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL", default=500)
                ),
                convergence_drift_tolerance_h=float(
                    self._cfg_value("DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H", default=2.0)
                ),
                target_fraction_tolerance=float(
                    self._cfg_value(
                        "DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE",
                        default=default_target_tolerance,
                    )
                ),
                convergence_required_checks=int(
                    self._cfg_value("DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS", default=2)
                ),
                accept_near_target_at_max=bool(
                    self._cfg_value(
                        "DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX",
                        default=False,
                    )
                ),
            )
            return GCMCDevanathanBoundary(
                cfg,
                reservoir,
                np.asarray(self.spec.box, dtype=float),
                self._evaluate_source_gcmc_delta_energy,
                self._report_gcmc_initialization_progress,
                region_label=region_label,
                control_site_indices=control_site_indices,
                control_fe_count=control_fe_count,
                left_sink_site_indices=left_sink_site_indices,
                left_sink_label=left_sink_label,
            )
        if source_mode == "pulse":
            return DevanathanPulseBoundary(
                cfg,
                np.asarray(self.spec.box, dtype=float),
                initial_count=pulse_initial_count,
                control_site_indices=control_site_indices,
                control_fe_count=control_fe_count,
                region_label=region_label,
                left_sink_site_indices=left_sink_site_indices,
                left_sink_label=left_sink_label,
            )
        if source_mode != "constant":
            raise ValueError(
                "DEVANATHAN_SOURCE_MODE must be 'constant', 'gcmc', 'gcmc_bulk', or 'pulse', "
                f"got {source_mode!r}"
            )
        return DevanathanBoundary(
            cfg,
            np.asarray(self.spec.box, dtype=float),
            left_sink_site_indices=left_sink_site_indices,
            left_sink_label=left_sink_label,
        )

    def _make_gcmc_energy_cache(self) -> Optional[PersistentGCMCEnergyCache]:
        if self.ctx.rank != 0:
            return None
        source_mode = self._normalize_devanathan_source_mode(
            self._cfg_value("DEVANATHAN_SOURCE_MODE", default="constant")
        )
        enabled = bool(
            self._cfg_value("DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED", default=True)
        )
        if source_mode not in {"gcmc", "gcmc_bulk"} or not enabled:
            return None
        default_name = (
            "gcmc_bulk_energy_cache.pkl"
            if source_mode == "gcmc_bulk"
            else "gcmc_source_energy_cache.pkl"
        )
        region_label = "bulk" if source_mode == "gcmc_bulk" else "source"
        default_path = self._cache_output_dir() / default_name
        path = Path(
            str(
                self._cfg_value(
                    "DEVANATHAN_GCMC_ENERGY_CACHE_FILE",
                    default=str(default_path),
                )
            )
        ).expanduser()
        cache = PersistentGCMCEnergyCache(
            path,
            autosave_interval=int(
                self._cfg_value(
                    "DEVANATHAN_GCMC_ENERGY_CACHE_SAVE_INTERVAL",
                    default=100,
                )
            ),
        )
        self.log(
            f"[Rank 0] Loaded GCMC {region_label} energy cache {path} "
            f"with {len(cache)} entries"
        )
        return cache

    def _gcmc_energy_potential_tag(self) -> tuple[Any, ...]:
        cfg = self.lammps_factory.cfg
        potential_file = Path(str(cfg.potential_eam_file)).expanduser()
        if not potential_file.is_absolute():
            potential_file = Path(self.lammps_factory.base_dir) / potential_file
        try:
            stat = potential_file.stat()
            file_tag: tuple[Any, ...] = (
                str(potential_file.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        except OSError:
            file_tag = (str(potential_file), None, None)
        return (
            str(cfg.potential).upper(),
            file_tag,
            str(cfg.lammps_pair_style),
            str(cfg.lammps_pair_coeff),
        )

    def _gcmc_host_geometry_tag(self) -> tuple[Any, ...] | None:
        if self.external_host_fe_positions is None:
            return None

        def file_tag(raw_path: Any) -> tuple[Any, ...]:
            path = Path(str(raw_path or "")).expanduser()
            try:
                stat = path.stat()
                return (
                    str(path.resolve()),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                )
            except OSError:
                return (str(path), None, None)

        return (
            "external_host",
            file_tag(
                self._cfg_value(
                    "site_map_file",
                    "KMC_SITE_MAP_FILE",
                    "SITE_MAP_FILE",
                    default="",
                )
            ),
            file_tag(
                self._cfg_value(
                    "host_structure_file",
                    "KMC_HOST_STRUCTURE_FILE",
                    "HOST_STRUCTURE_FILE",
                    default="",
                )
            ),
            tuple(round(float(value), 12) for value in np.asarray(self.spec.box, dtype=float)),
            int(len(self.external_host_fe_positions)),
        )

    def _evaluate_source_gcmc_delta_energy(
        self,
        h_indices: np.ndarray,
        site: int,
        kind: str,
    ) -> float:
        """Evaluate a local rigid-lattice E(after)-E(before) for a source move."""
        assert self.sites is not None
        site = int(site)
        current = np.asarray(h_indices, dtype=int)
        if kind == "insert":
            lower_indices = current
        elif kind == "delete":
            if site not in set(int(x) for x in current):
                raise ValueError(f"Cannot evaluate deletion of vacant site {site}")
            lower_indices = current[current != site]
        else:
            raise ValueError(f"Unknown GCMC move kind: {kind}")

        shell_inner = float(
            self._cfg_value(
                "DEVANATHAN_GCMC_SHELL_INNER_RADIUS_A",
                default=self._cfg_value("SHELL_INNER_RADIUS_A", default=8.0),
            )
        )
        shell_outer = float(
            self._cfg_value(
                "DEVANATHAN_GCMC_SHELL_OUTER_RADIUS_A",
                default=self._cfg_value("SHELL_OUTER_RADIUS_A", default=10.0),
            )
        )
        position_bin = float(
            self._cfg_value("DEVANATHAN_GCMC_ENERGY_CACHE_POS_BIN_A", default=0.1)
        )
        cache_key = canonical_insertion_environment_key(
            sites=np.asarray(self.sites, dtype=float),
            lower_occupancy_indices=lower_indices,
            trial_site=site,
            box=self.spec.box,
            lattice_a=self.spec.lattice_a,
            shell_inner_a=shell_inner,
            shell_outer_a=shell_outer,
            position_bin_a=position_bin,
            potential_tag=self._gcmc_energy_potential_tag(),
            host_geometry_tag=self._gcmc_host_geometry_tag(),
        )
        if self.gcmc_energy_cache is not None:
            cached_insertion_delta = self.gcmc_energy_cache.get(cache_key)
            if cached_insertion_delta is not None:
                self._gcmc_energy_cache_step_hits += 1
                # The cache stores E(with H) - E(without H). A deletion uses
                # the same lower-occupancy key with the opposite sign.
                return (
                    cached_insertion_delta
                    if kind == "insert"
                    else -cached_insertion_delta
                )
            self._gcmc_energy_cache_step_misses += 1

        higher_indices = np.append(lower_indices, site)
        trial_positions = np.asarray(self.sites[higher_indices], dtype=float)
        if self.external_host_fe_positions is not None:
            before_with_site, _same, mover_idx = build_local_neb_structures_shell_from_host(
                np.asarray(self.sites[site], dtype=float),
                np.asarray(self.sites[site], dtype=float),
                np.asarray(self.external_host_fe_positions, dtype=float),
                self.spec.box,
                host_fe_tree=self._external_host_fe_tree,
                h_positions=trial_positions,
                h_indices=higher_indices,
                inner_radius_a=shell_inner,
                outer_radius_a=shell_outer,
            )
        else:
            trial_cell_index = build_h_cell_index(
                trial_positions,
                higher_indices,
                self.spec,
            )
            before_with_site, _same, mover_idx = build_local_neb_structures_shell(
                np.asarray(self.sites[site], dtype=float),
                np.asarray(self.sites[site], dtype=float),
                self.spec,
                h_cell_index=trial_cell_index,
                inner_radius_a=shell_inner,
                outer_radius_a=shell_outer,
            )
        without_site = before_with_site.copy()
        del without_site[int(mover_idx)]
        assign_mass_and_type(without_site)

        if self._gcmc_calculator is None:
            self.log("[Rank 0] Creating reusable GCMC LAMMPS calculator")
            calculator_started = time.time()
            self._gcmc_calculator = self.lammps_factory.create()
            self.log(
                "[Rank 0] Reusable GCMC LAMMPS calculator ready in "
                f"{time.time() - calculator_started:.2f} s"
            )
        calc = self._gcmc_calculator
        evaluation_started = time.time()
        without_site.calc = calc
        energy_without = float(without_site.get_potential_energy())
        before_with_site.calc = calc
        energy_with = float(before_with_site.get_potential_energy())
        self._gcmc_energy_evaluations += 1
        if self._gcmc_energy_evaluations == 1:
            self.log(
                "[Rank 0] First uncached GCMC insertion energy completed in "
                f"{time.time() - evaluation_started:.2f} s "
                f"with {len(before_with_site)} local atoms"
            )
        insertion_delta = energy_with - energy_without
        if self.gcmc_energy_cache is not None:
            self.gcmc_energy_cache.set(cache_key, insertion_delta)
        return insertion_delta if kind == "insert" else -insertion_delta

    def _report_gcmc_initialization_progress(self, progress: Mapping[str, Any]) -> None:
        if self.ctx.rank != 0:
            return
        attempts = int(progress["attempts"])
        accepted = int(progress["accepted_moves"])
        current_h = int(progress["current_h"])
        current_fraction = float(progress["current_fraction_h_per_fe"])
        elapsed_s = float(progress["elapsed_s"])
        cache_entries = 0 if self.gcmc_energy_cache is None else len(self.gcmc_energy_cache)
        cache_hits = (
            0 if self.gcmc_energy_cache is None else int(self.gcmc_energy_cache.stats.hits)
        )
        cache_misses = (
            0 if self.gcmc_energy_cache is None else int(self.gcmc_energy_cache.stats.misses)
        )
        self.log(
            "[Rank 0] GCMC initialization progress: "
            f"attempts={attempts}, accepted={accepted}, "
            f"N_H={current_h}, H_per_Fe={current_fraction:.6g}, "
            f"cache_entries={cache_entries}, hits={cache_hits}, misses={cache_misses}, "
            f"uncached_evaluations={self._gcmc_energy_evaluations}, elapsed_s={elapsed_s:.1f}"
        )
        path = self._diagnostics_output_dir() / "gcmc_initialization_progress.csv"
        self._append_dict_csv(
            path,
            [
                "attempts",
                "accepted_moves",
                "N_H",
                "c_H_per_Fe",
                "target_H",
                "target_c_H_per_Fe",
                "cache_entries",
                "cache_hits",
                "cache_misses",
                "uncached_evaluations",
                "elapsed_s",
            ],
            [
                {
                    "attempts": attempts,
                    "accepted_moves": accepted,
                    "N_H": current_h,
                    "c_H_per_Fe": f"{current_fraction:.12g}",
                    "target_H": int(progress["target_h"]),
                    "target_c_H_per_Fe": f"{float(progress['target_fraction_h_per_fe']):.12g}",
                    "cache_entries": cache_entries,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "uncached_evaluations": self._gcmc_energy_evaluations,
                    "elapsed_s": f"{elapsed_s:.6f}",
                }
            ],
            write_header=not self._gcmc_progress_header_written,
            mode="w" if not self._gcmc_progress_header_written else "a",
        )
        self._gcmc_progress_header_written = True

    def _write_gcmc_initialization_diag(self, source_fe_atoms: int) -> None:
        if self.ctx.rank != 0 or self.devanathan_boundary is None:
            return
        history = tuple(
            int(value)
            for value in getattr(
                self.devanathan_boundary,
                "initialization_history",
                (),
            )
        )
        if not history:
            return
        path = self._diagnostics_output_dir() / "gcmc_initialization.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["attempt", "N_H", "c_H_per_Fe"])
            for attempt, n_h in enumerate(history, start=1):
                writer.writerow(
                    [
                        attempt,
                        n_h,
                        f"{n_h / source_fe_atoms:.12g}",
                    ]
                )

    def _append_devanathan_diag(self, step: int, update: Optional[DevanathanUpdate]) -> None:
        if self.ctx.rank != 0 or update is None:
            return
        path = self._diagnostics_output_dir() / "kmc_devanathan.csv"
        raw_proposals = tuple(getattr(update, "gcmc_proposals", ()))
        proposal_limit = max(
            0,
            int(self._cfg_value("DEVANATHAN_GCMC_DIAG_PROPOSAL_LIMIT", default=32)),
        )
        proposal_texts: list[str] = []
        for index, proposal in enumerate(raw_proposals):
            if index < proposal_limit or bool(proposal.accepted):
                proposal_texts.append(
                    f"{proposal.kind}:site={proposal.site}:"
                    f"dE={proposal.delta_energy_ev:.10g}:"
                    f"p={proposal.acceptance_probability:.10g}:"
                    f"u={proposal.random_value:.10g}:"
                    f"accepted={int(proposal.accepted)}"
                )
        if len(raw_proposals) > proposal_limit:
            accepted = sum(bool(proposal.accepted) for proposal in raw_proposals)
            omitted = max(0, len(raw_proposals) - len(proposal_texts))
            proposal_texts.append(
                f"truncated={omitted}:"
                f"total={len(raw_proposals)}:accepted={accepted}"
            )
        fields = [
            "step",
            "time_s",
            "control_region",
            "source_occupied",
            "source_sites",
            "source_fe_atoms",
            "source_target_h",
            "source_setpoint_h",
            "left_sink_occupied",
            "sink_occupied",
            "total_occupied",
            "inserted_this_step",
            "left_removed_this_step",
            "removed_this_step",
            "trimmed_this_step",
            "cumulative_inserted",
            "cumulative_left_removed",
            "cumulative_removed",
            "flux_per_a2_s",
            "inserted_sites",
            "deleted_left_sink_sites",
            "deleted_sink_sites",
            "trimmed_source_sites",
            "gcmc_mu_eV",
            "gcmc_attempts_this_step",
            "gcmc_accepts_this_step",
            "gcmc_cumulative_attempts",
            "gcmc_cumulative_accepts",
            "gcmc_energy_cache_hits_this_step",
            "gcmc_energy_cache_misses_this_step",
            "gcmc_energy_cache_cumulative_hits",
            "gcmc_energy_cache_cumulative_misses",
            "gcmc_energy_cache_entries",
            "gcmc_proposals",
        ]
        row = {
            "step": int(step),
            "time_s": f"{float(self.sim_time_s):.10g}",
            "control_region": str(
                getattr(self.devanathan_boundary, "region_label", "source")
            ),
            "source_occupied": int(update.source_occupied),
            "source_sites": int(update.source_sites),
            "source_fe_atoms": int(update.source_fe_atoms),
            "source_target_h": int(update.source_target_h),
            "source_setpoint_h": int(
                getattr(update, "maintenance_setpoint_h", update.source_target_h)
            ),
            "left_sink_occupied": int(getattr(update, "left_sink_occupied", 0)),
            "sink_occupied": int(update.sink_occupied),
            "total_occupied": int(update.total_occupied),
            "inserted_this_step": len(update.inserted_sites),
            "left_removed_this_step": len(
                getattr(update, "deleted_left_sink_sites", ())
            ),
            "removed_this_step": len(update.deleted_sink_sites),
            "trimmed_this_step": len(update.trimmed_source_sites),
            "cumulative_inserted": int(update.cumulative_inserted),
            "cumulative_left_removed": int(
                getattr(update, "cumulative_left_removed", 0)
            ),
            "cumulative_removed": int(update.cumulative_removed),
            "flux_per_a2_s": "" if update.flux_per_a2_s is None else f"{float(update.flux_per_a2_s):.10g}",
            "inserted_sites": " ".join(str(int(s)) for s in update.inserted_sites),
            "deleted_left_sink_sites": " ".join(
                str(int(s))
                for s in getattr(update, "deleted_left_sink_sites", ())
            ),
            "deleted_sink_sites": " ".join(str(int(s)) for s in update.deleted_sink_sites),
            "trimmed_source_sites": " ".join(str(int(s)) for s in update.trimmed_source_sites),
            "gcmc_mu_eV": (
                ""
                if not hasattr(self.devanathan_boundary, "reservoir")
                else f"{float(self.devanathan_boundary.reservoir.chemical_potential_ev):.10g}"
            ),
            "gcmc_attempts_this_step": len(getattr(update, "gcmc_proposals", ())),
            "gcmc_accepts_this_step": sum(
                bool(proposal.accepted) for proposal in getattr(update, "gcmc_proposals", ())
            ),
            "gcmc_cumulative_attempts": int(
                getattr(update, "cumulative_gcmc_attempts", 0)
            ),
            "gcmc_cumulative_accepts": int(
                getattr(update, "cumulative_gcmc_accepts", 0)
            ),
            "gcmc_energy_cache_hits_this_step": int(
                self._gcmc_energy_cache_step_hits
            ),
            "gcmc_energy_cache_misses_this_step": int(
                self._gcmc_energy_cache_step_misses
            ),
            "gcmc_energy_cache_cumulative_hits": (
                0
                if self.gcmc_energy_cache is None
                else int(self.gcmc_energy_cache.stats.hits)
            ),
            "gcmc_energy_cache_cumulative_misses": (
                0
                if self.gcmc_energy_cache is None
                else int(self.gcmc_energy_cache.stats.misses)
            ),
            "gcmc_energy_cache_entries": (
                0 if self.gcmc_energy_cache is None else len(self.gcmc_energy_cache)
            ),
            "gcmc_proposals": " | ".join(proposal_texts),
        }
        self._append_dict_csv(
            path,
            fields,
            [row],
            write_header=not self._devanathan_header_written,
            mode="w" if not self._devanathan_header_written else "a",
        )
        self._devanathan_header_written = True

    def _devanathan_particle_changed_sites(
        self,
        update: Optional[DevanathanUpdate],
    ) -> list[int]:
        if update is None:
            return []
        changed: list[int] = []
        changed.extend(int(site) for site in update.inserted_sites)
        changed.extend(int(site) for site in update.deleted_sink_sites)
        changed.extend(
            int(site)
            for site in getattr(update, "deleted_left_sink_sites", ())
        )
        changed.extend(int(site) for site in update.trimmed_source_sites)
        return changed

    def _apply_devanathan_boundary_rank0(
        self,
        step: int,
    ) -> Optional[DevanathanUpdate]:
        if self.ctx.rank != 0 or self.devanathan_boundary is None:
            return None
        assert self.sites is not None
        assert self.h_indices is not None
        assert self.h_unwrapped_positions is not None
        self._gcmc_energy_cache_step_hits = 0
        self._gcmc_energy_cache_step_misses = 0
        update = self.devanathan_boundary.apply(
            sites=self.sites,
            fe_positions=self._native_fe_positions(),
            h_indices=self.h_indices,
            h_unwrapped_positions=self.h_unwrapped_positions,
            elapsed_time_s=self.sim_time_s,
        )
        self.h_indices = update.h_indices
        self.h_unwrapped_positions = update.h_unwrapped_positions
        if update.particle_count_changed:
            region_label = str(getattr(self.devanathan_boundary, "region_label", "source"))
            self.log(
                f"[Rank 0] Devanathan step {step}: "
                f"+{len(update.inserted_sites)} {region_label} H, "
                f"-{len(update.trimmed_source_sites)} {region_label} H, "
                f"-{len(getattr(update, 'deleted_left_sink_sites', ()))} left-boundary H, "
                f"-{len(update.deleted_sink_sites)} sink H, "
                f"{region_label}={update.source_occupied}/"
                f"{getattr(update, 'maintenance_setpoint_h', update.source_target_h)} "
                f"H ({update.source_fe_atoms} Fe basis), "
                f"left_removed_total={getattr(update, 'cumulative_left_removed', 0)}, "
                f"sink_removed_total={update.cumulative_removed}"
            )
        return update

    def _sync_devanathan_update(
        self,
        update: Optional[DevanathanUpdate],
    ) -> Optional[DevanathanUpdate]:
        if self.devanathan_boundary is None:
            return update
        self.h_indices = np.asarray(
            self.ctx.comm.bcast(
                self.h_indices if self.ctx.rank == 0 else None,
                root=0,
            ),
            dtype=int,
        )
        self.h_unwrapped_positions = np.asarray(
            self.ctx.comm.bcast(
                self.h_unwrapped_positions if self.ctx.rank == 0 else None,
                root=0,
            ),
            dtype=float,
        )
        return self.ctx.comm.bcast(update, root=0)

    def _rebuild_full_structure_after_devanathan(
        self,
        update: Optional[DevanathanUpdate],
    ) -> None:
        if (
            self.full_structure is None
            or update is None
            or not update.particle_count_changed
        ):
            return
        try:
            if self.external_host_structure is not None:
                self.full_structure = build_full_structure_from_host(
                    self.external_host_structure,
                    self.sites,
                    self.h_indices,
                )
            else:
                self.full_structure = build_full_structure(
                    self.sites,
                    self.h_indices,
                    self.spec,
                )
        except Exception as exc:
            self.log(
                f"[Rank {self.ctx.rank}] WARNING: full_structure rebuild after "
                f"Devanathan update failed: {exc}"
            )

    def _try_devanathan_no_event_recovery(
        self,
        step: int,
        timer: StepTimer,
        *,
        incremental_enabled: bool = False,
    ) -> bool:
        if self.devanathan_boundary is None:
            return False

        t0 = time.time()
        update = self._apply_devanathan_boundary_rank0(step)
        update = self._sync_devanathan_update(update)
        self._rebuild_full_structure_after_devanathan(update)
        if incremental_enabled and update is not None:
            self._prepare_incremental_next_update(
                None,
                particle_changed_sites=self._devanathan_particle_changed_sites(update),
            )
        self._refresh_h_lookup()
        timer.add("state_update", time.time() - t0)

        t0 = time.time()
        if self.ctx.rank == 0:
            if self._should_periodic_dump(step):
                self._append_only_h_trajectory(step + 1)
            self._append_devanathan_diag(step, update)
            self._write_restart_checkpoint(step + 1)
        timer.add("dump_traj", time.time() - t0)

        t0 = time.time()
        if self.ctx.rank == 0 and self.gcmc_energy_cache is not None:
            self.gcmc_energy_cache.save()
        timer.add("cache_save", time.time() - t0)

        recovered = bool(update is not None and update.particle_count_changed)
        if self.ctx.rank == 0:
            if recovered:
                self.log(
                    f"[Rank 0] Step {step}: no KMC event; "
                    "Devanathan/GCMC changed occupancy, continuing"
                )
            else:
                self.log(
                    f"[Rank 0] Step {step}: no KMC event and "
                    "Devanathan/GCMC made no occupancy change"
                )
        return recovered

    def _native_lammps_config(self, step: Optional[int] = None) -> LammpsOnlyNEBConfig:
        scratch_root = Path(
            str(
                self._cfg_value(
                    "native_scratch_dir",
                    "LAMMPS_NEB_SCRATCH_DIR",
                    default=str(self._neb_output_dir() / "lammps_only_scratch"),
                )
                or str(self._neb_output_dir() / "lammps_only_scratch")
            )
        ).expanduser()
        if step is not None:
            scratch_root = scratch_root / f"step_{int(step):08d}"

        python_path_raw = str(self._cfg_value("python_path", "LAMMPS_PYTHON_PATH", "LAMMPS_PYTHON", default="") or "")
        lib_dir_raw = str(self._cfg_value("lib_dir", "LAMMPS_LIB_DIR", default="") or "")
        min_style = str(self._cfg_value("native_min_style", "LAMMPS_NEB_MIN_STYLE", default="quickmin") or "quickmin")
        engine = str(self._cfg_value("engine", "NEB_ENGINE", "KMC_NEB_ENGINE", default="lammps_only") or "lammps_only").lower().strip()
        endpoint_alias = engine in {
            "lammps_only_endpoint_opt",
            "lammps_only_lmp_endpoint_opt",
            "native_lammps_endpoint_opt",
        }
        endpoint_steps = int(
            self._cfg_value(
                "native_endpoint_max_steps",
                "LAMMPS_ENDPOINT_OPT_STEPS",
                "ENDPOINT_OPTIMIZER_STEPS",
                default=200,
            )
        )
        return LammpsOnlyNEBConfig(
            replicas_per_neb=int(self._cfg_value("replicas_per_neb", "LAMMPS_NEB_REPLICAS", "NATIVE_NEB_REPLICAS", default=3)),
            spring_constant=float(self._cfg_value("native_spring_constant", "LAMMPS_NEB_SPRING", default=5.0)),
            force_tolerance=float(self._cfg_value("native_force_tolerance", "LAMMPS_NEB_FTOL", default=1.0e-5)),
            max_steps=int(self._cfg_value("native_max_steps", "LAMMPS_NEB_STEPS", "NEB_LAMMPS_STEPS", default=1000)),
            climbing_steps=int(self._cfg_value("native_climbing_steps", "LAMMPS_NEB_CLIMBING_STEPS", default=0)),
            nevery=int(self._cfg_value("native_nevery", "LAMMPS_NEB_NEVERY", default=50)),
            thermo_every=int(self._cfg_value("native_thermo_every", "LAMMPS_NEB_THERMO_EVERY", default=50)),
            timestep=float(self._cfg_value("native_timestep", "LAMMPS_NEB_TIMESTEP", default=0.01)),
            min_style=min_style,
            debug_mode=bool(self._cfg_value("native_debug_mode", "LAMMPS_NEB_DEBUG_MODE", "NEB_DUMP", default=False)),
            endpoint_optimize=bool(
                self._cfg_value(
                    "native_endpoint_optimize",
                    "LAMMPS_ONLY_ENDPOINT_OPTIMIZE",
                    default=endpoint_alias,
                )
            ),
            endpoint_min_style=str(
                self._cfg_value(
                    "native_endpoint_min_style",
                    "LAMMPS_ENDPOINT_OPT_MIN_STYLE",
                    default=min_style,
                )
                or min_style
            ).lower(),
            endpoint_energy_tolerance=float(
                self._cfg_value(
                    "native_endpoint_energy_tolerance",
                    "LAMMPS_ENDPOINT_OPT_ETOL",
                    default=0.0,
                )
            ),
            endpoint_force_tolerance=float(
                self._cfg_value(
                    "native_endpoint_force_tolerance",
                    "LAMMPS_ENDPOINT_OPT_FTOL",
                    "NEB_OPTIMIZER_FMAX",
                    default=0.05,
                )
            ),
            endpoint_max_steps=endpoint_steps,
            endpoint_max_evals=int(
                self._cfg_value(
                    "native_endpoint_max_evals",
                    "LAMMPS_ENDPOINT_OPT_MAXEVAL",
                    default=max(1, endpoint_steps * 10),
                )
            ),
            endpoint_dmax=float(
                self._cfg_value(
                    "native_endpoint_dmax",
                    "LAMMPS_ENDPOINT_OPT_DMAX",
                    default=0.1,
                )
            ),
            shell_inner_radius_a=float(self._cfg_value("shell_inner_radius_a", "SHELL_INNER_RADIUS_A", default=8.0)),
            shell_outer_radius_a=float(self._cfg_value("shell_outer_radius_a", "SHELL_OUTER_RADIUS_A", default=10.0)),
            shell_pad_a=float(self._cfg_value("shrinkwrap_pad_a", "SHELL_PAD_A", "LAMMPS_NEB_SHELL_PAD_A", default=2.0)),
            potential_file=Path(str(self._cfg_value("potential_eam_file", "POTENTIAL_EAM_FILE", default="PotentialB3410-modified.fs"))).expanduser(),
            work_dir=scratch_root,
            lammps_python_path=Path(python_path_raw).expanduser() if python_path_raw else None,
            lammps_lib_dir=Path(lib_dir_raw).expanduser() if lib_dir_raw else None,
        ).normalized()

    def _validation_enabled(self) -> bool:
        return bool(self._cfg_value("validation_enabled", "enabled", "VALIDATION_MODE", default=False))

    def _validation_ase_dump(self) -> bool:
        return bool(self._cfg_value("validation_ase_dump", "ase_dump", "VALIDATION_ASE_DUMP", default=False))

    def _validation_ase_dump_pre(self) -> bool:
        return bool(self._cfg_value("validation_ase_dump_pre", "ase_dump_pre", "VALIDATION_ASE_DUMP_PRE", default=False))

    def _validation_overwrite(self) -> bool:
        return bool(self._cfg_value("validation_overwrite", "overwrite", "VALIDATION_OVERWRITE", default=True))

    def _validation_root(self) -> Path:
        default_out = str(self._neb_output_dir() / "validation_dump")
        configured = str(self._cfg_value("validation_out", "out_dir", "VALIDATION_OUT", default=default_out) or default_out)
        out = default_out if "VALIDATION_OUT" not in os.environ and configured in {"validation_dump", "neb/validation_dump"} else configured
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_only_enabled(self) -> bool:
        return bool(self._cfg_value("build_only", "BUILD_LOCAL_ENV_ONLY", default=False))

    def _incremental_events_enabled(self) -> bool:
        return bool(
            self._cfg_value(
                "incremental_events_enabled",
                "KMC_INCREMENTAL_EVENTS",
                "INCREMENTAL_EVENTS",
                default=False,
            )
        )

    def _incremental_impact_radius_a(self) -> float:
        configured = self._cfg_value(
            "incremental_impact_radius_a",
            "KMC_INCREMENTAL_IMPACT_RADIUS_A",
            "INCREMENTAL_IMPACT_RADIUS_A",
            default=None,
        )
        if configured not in (None, "", "None"):
            return float(configured)
        env_radius = float(self._cfg_value("env_radius_a", "radius_a", "ENV_RADIUS_A", default=5.0))
        hop_radius = self._incremental_hop_radius_a
        if hop_radius is None:
            assert self.sites is not None
            assert self.neighbors is not None
            hop_radius = max_neighbor_hop_distance(self.sites, self.neighbors, self.spec.box)
            self._incremental_hop_radius_a = float(hop_radius)
        return float(env_radius + hop_radius)

    def _mark_incremental_table_invalid(self) -> None:
        self._incremental_event_table = None
        self._incremental_rebuild_h_sites = None
        self._incremental_remove_h_sites = None

    def _prepare_incremental_next_update(
        self,
        selected,
        *,
        particle_changed_sites: Sequence[int] = (),
    ) -> None:
        if not self._incremental_events_enabled():
            return
        assert self.sites is not None
        assert self.h_indices is not None
        impact_radius = self._incremental_impact_radius_a()
        affected: set[int] = set()
        remove: set[int] = set()
        if selected is not None:
            affected.update(
                affected_h_sites_for_move(
                    sites=np.asarray(self.sites, dtype=float),
                    h_indices=self.h_indices,
                    box=self.spec.box,
                    old_site=int(selected.h_site),
                    new_site=int(selected.n_site),
                    impact_radius_a=impact_radius,
                )
            )
            remove.update((int(selected.h_site), int(selected.n_site)))
        changed = sorted({int(site) for site in particle_changed_sites})
        if changed:
            particle_affected = affected_h_sites_for_particle_changes(
                sites=np.asarray(self.sites, dtype=float),
                h_indices=self.h_indices,
                box=self.spec.box,
                changed_sites=changed,
                impact_radius_a=impact_radius,
            )
            affected.update(particle_affected)
            remove.update(changed)
        remove.update(affected)
        affected_sorted = sorted(affected)
        remove_sorted = sorted(remove)
        if not remove_sorted:
            self._incremental_rebuild_h_sites = None
            self._incremental_remove_h_sites = None
            return
        self._incremental_rebuild_h_sites = affected_sorted
        self._incremental_remove_h_sites = remove_sorted
        if self.ctx.rank == 0:
            self.log(
                f"[Rank 0] Incremental update prepared: rebuild_H={len(affected_sorted)}, "
                f"remove_sources={len(remove_sorted)}, particle_changes={len(changed)}, "
                f"impact_radius={impact_radius:.3f} A"
            )

    def _restart_enabled(self) -> bool:
        if "RESTART_MODE" in os.environ or "KMC_RESTART_MODE" in os.environ:
            return bool(self._cfg_value("RESTART_MODE", "KMC_RESTART_MODE", default=False))
        restart = getattr(self.cfg, "restart", None)
        if restart is not None and hasattr(restart, "enabled"):
            return bool(restart.enabled)
        return False

    def _restart_root(self) -> Path:
        if "RESTART_DIR" in os.environ or "KMC_RESTART_DIR" in os.environ:
            raw = str(self._cfg_value("RESTART_DIR", "KMC_RESTART_DIR", default=".") or ".")
        else:
            restart = getattr(self.cfg, "restart", None)
            raw = str(getattr(restart, "directory", ".") if restart is not None else ".")
        return Path(raw).expanduser()

    def _restart_step(self) -> int:
        if "RESTART_STEP" in os.environ or "KMC_RESTART_STEP" in os.environ:
            return max(0, int(self._cfg_value("RESTART_STEP", "KMC_RESTART_STEP", default=0)))
        restart = getattr(self.cfg, "restart", None)
        return max(0, int(getattr(restart, "step", 0) if restart is not None else 0))

    def _restart_strict(self) -> bool:
        if "RESTART_STRICT" in os.environ:
            return bool(self._cfg_value("RESTART_STRICT", default=True))
        restart = getattr(self.cfg, "restart", None)
        return bool(getattr(restart, "strict", True) if restart is not None else True)

    def _restart_file(self, attr_name: str, env_name: str, default_name: str) -> Path:
        root = self._restart_root()
        if env_name in os.environ:
            value = str(self._cfg_value(env_name, default=default_name) or default_name).strip()
        else:
            restart = getattr(self.cfg, "restart", None)
            value = str(getattr(restart, attr_name, default_name) if restart is not None else default_name).strip()
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        return path

    def _restart_trajectory_path(self) -> Path:
        path = self._restart_file("trajectory_file", "RESTART_TRAJECTORY_FILE", "H_trajectory_onlyH.lammpstrj")
        if path.exists():
            return path
        legacy = self._restart_root() / "H_only.lammpstrj"
        if legacy.exists():
            return legacy
        return path

    def _restart_timestep_path(self) -> Path:
        path = self._restart_file(
            "timestep_file",
            "RESTART_TIMESTEP_FILE",
            "diagnostics/kmc_timestep_vs_step.csv",
        )
        if path.exists():
            return path
        legacy = self._restart_root() / "kmc_timestep_vs_step.csv"
        if legacy.exists():
            return legacy
        return path

    def _checkpoint_interval(self) -> int:
        return max(
            0,
            int(
                self._cfg_value(
                    "restart_checkpoint_interval",
                    "RESTART_CHECKPOINT_INTERVAL",
                    "CHECKPOINT_INTERVAL",
                    default=1000,
                )
            ),
        )

    def _checkpoint_dir(self) -> Path:
        raw = str(
            self._cfg_value(
                "restart_checkpoint_dir",
                "RESTART_CHECKPOINT_DIR",
                "CHECKPOINT_DIR",
                default="checkpoints",
            )
            or "checkpoints"
        ).strip()
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _checkpoint_prefix(self) -> str:
        raw = str(
            self._cfg_value(
                "restart_checkpoint_prefix",
                "RESTART_CHECKPOINT_PREFIX",
                "CHECKPOINT_PREFIX",
                default="kmc_restart_checkpoint",
            )
            or "kmc_restart_checkpoint"
        ).strip()
        return raw or "kmc_restart_checkpoint"

    def _restart_checkpoint_path(self) -> Path:
        return self._restart_file("checkpoint_file", "RESTART_CHECKPOINT_FILE", "kmc_restart_checkpoint.pkl")

    def _restart_checkpoint_candidates(self) -> list[tuple[Path, bool]]:
        if "RESTART_CHECKPOINT_FILE" in os.environ or "KMC_RESTART_CHECKPOINT_FILE" in os.environ:
            return [(self._restart_checkpoint_path(), True)]
        root = self._restart_root()
        step = self._restart_step()
        default_name = "kmc_restart_checkpoint.pkl"
        checkpoint_root = root / "checkpoints"
        latest = checkpoint_root / default_name
        legacy_latest = root / default_name
        if step > 0:
            return [
                (checkpoint_root / f"kmc_restart_checkpoint_step{step}.pkl", False),
                (latest, False),
                (root / f"kmc_restart_checkpoint_step{step}.pkl", False),
                (legacy_latest, False),
            ]
        return [(latest, False), (legacy_latest, False)]

    def _ase_log_dir(self) -> Optional[Path]:
        if not bool(self._cfg_value("debug_logging", "DEBUG_LOGGING", "DEBUG_MODE", default=True)):
            return None
        default_log_dir = str(self._neb_output_dir() / "ase_logs")
        configured = str(self._cfg_value("ase_log_dir", "ASE_LOG_DIR", default=default_log_dir) or default_log_dir).strip()
        log_dir = default_log_dir if "ASE_LOG_DIR" not in os.environ and configured in {"ase_logs", "neb/ase_logs"} else configured
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _structure_snapshot_path(self, job_context: Dict[str, Any]) -> Optional[Path]:
        log_dir = self._ase_log_dir()
        if log_dir is None:
            return None
        parts = ["structure_snapshot"]
        for tag in ("step", "h", "n", "src_rank"):
            value = job_context.get(tag)
            if value is not None:
                sanitized = _sanitize_filename_component(value)
                if sanitized:
                    parts.append(f"{tag}{sanitized}")
        parts.append(f"seq{next(_STRUCTURE_SNAPSHOT_SEQ)}")
        return log_dir / ("_".join(parts) + ".lmp")

    def _ase_log_path(self, base_name: str, *, job_context: Optional[Dict[str, Any]] = None) -> str:
        log_dir = self._ase_log_dir()
        if log_dir is None:
            return os.devnull
        base_root = Path(base_name or "ase").stem or "ase"
        parts = [base_root]
        if job_context:
            for tag in ("step", "h", "n", "src_rank"):
                value = job_context.get(tag)
                if value is not None:
                    sanitized = _sanitize_filename_component(value)
                    if sanitized:
                        parts.append(f"{tag}{sanitized}")
        parts.append(f"rank{self.ctx.rank}")
        parts.append(f"seq{next(_STRUCTURE_SNAPSHOT_SEQ)}")
        return str(log_dir / ("_".join(parts) + ".log"))

    def _select_h_indices_for_build_only(self, sites: np.ndarray, num_h: int) -> np.ndarray:
        selector = str(self._cfg_value("build_selector", "BUILD_LOCAL_ENV_SELECTOR", default="random")).lower().strip()
        if selector == "random":
            return np.random.choice(len(sites), size=min(int(num_h), len(sites)), replace=False)
        if selector not in {"boundary", "cluster"}:
            selector = "random"
            return np.random.choice(len(sites), size=min(int(num_h), len(sites)), replace=False)

        sites = np.asarray(sites, dtype=float)
        num_h = min(int(num_h), len(sites))
        if num_h <= 0:
            return np.array([], dtype=int)
        box = np.asarray(self.spec.box, dtype=float)

        if selector == "boundary":
            margin = float(self._cfg_value("boundary_margin_a", "BOUNDARY_MARGIN_A", default=5.0))
            near_boundary = np.any((sites <= margin) | ((box - sites) <= margin), axis=1)
            candidate_idx = np.flatnonzero(near_boundary)
            if len(candidate_idx) < num_h:
                raise RuntimeError(
                    f"BUILD_LOCAL_ENV_SELECTOR='boundary' found only {len(candidate_idx)} candidate sites "
                    f"within {margin:.2f} A of a boundary; need {num_h}."
                )
            chosen = np.random.choice(candidate_idx, size=num_h, replace=False)
            return np.asarray(chosen, dtype=int)

        radius = float(self._cfg_value("cluster_radius_a", "CLUSTER_RADIUS_A", default=5.0))
        center_idx = int(np.random.randint(len(sites)))
        center = sites[center_idx]
        disp = sites - center
        disp -= box * np.round(disp / box)
        dist = np.linalg.norm(disp, axis=1)
        candidate_idx = np.flatnonzero(dist <= radius + 1e-12)
        if len(candidate_idx) < num_h:
            raise RuntimeError(
                f"BUILD_LOCAL_ENV_SELECTOR='cluster' found only {len(candidate_idx)} candidate sites "
                f"within {radius:.2f} A of the seed site; need {num_h}."
            )
        if len(candidate_idx) == num_h:
            return np.asarray(candidate_idx, dtype=int)
        others = candidate_idx[candidate_idx != center_idx]
        picked_others = np.random.choice(others, size=num_h - 1, replace=False)
        return np.asarray(np.concatenate(([center_idx], picked_others)), dtype=int)

    def _initial_h_region_file(self) -> str:
        return str(
            self._cfg_value(
                "initial_h_region_file",
                "KMC_INITIAL_H_REGION_FILE",
                "KMC_H_REGION_FILE",
                default="",
            )
            or ""
        ).strip()

    def _initial_h_region_specs(self) -> list[str]:
        raw = str(
            self._cfg_value(
                "initial_h_regions",
                "KMC_INITIAL_H_REGIONS",
                "KMC_INITIAL_H_REGION",
                "KMC_H_REGIONS",
                "KMC_H_REGION",
                default="",
            )
            or ""
        ).strip()
        if not raw:
            return []
        return [item.strip().lower() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _normalize_h_region_spec(spec: str) -> str:
        token = spec.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "gb": "grain_boundary",
            "gb_core": "grain_boundary",
            "gb_voronoi": "grain_boundary",
            "grain": "grain_boundary",
            "boundary": "grain_boundary",
            "grainboundary": "grain_boundary",
            "grain_boundary": "grain_boundary",
            "transition": "transition",
            "trans": "transition",
            "bulk": "bulk",
            "all_bulk": "bulk",
        }
        if token.startswith("bulk_") and token.removeprefix("bulk_").isdigit():
            return f"bulk_grain_{token.removeprefix('bulk_')}"
        if token.startswith("bulk") and token.removeprefix("bulk").isdigit():
            return f"bulk_grain_{token.removeprefix('bulk')}"
        return aliases.get(token, token)

    def _site_region_mask_from_specs(
        self,
        sites: np.ndarray,
        region_specs: Sequence[str],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], Path]:
        region_file = self._initial_h_region_file()
        if not region_file:
            raise RuntimeError(
                "KMC site-region selection requires KMC_INITIAL_H_REGION_FILE "
                "(or KMC_H_REGION_FILE), for example kmc_map_inputs/sigma5_site_regions.npz."
            )

        path = Path(region_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"KMC site-region file not found: {path}")

        n_sites = len(sites)
        with np.load(path, allow_pickle=True) as data:
            if "site_selection_label" not in data.files and "site_region_name" not in data.files:
                raise ValueError(
                    f"{path} must contain site_selection_label or site_region_name arrays "
                    "from tools/prepare_kmc_region_inputs.py"
                )

            selection_label = (
                np.asarray(data["site_selection_label"]).astype(str)
                if "site_selection_label" in data.files
                else np.asarray(data["site_region_name"]).astype(str)
            )
            region_name = (
                np.asarray(data["site_region_name"]).astype(str)
                if "site_region_name" in data.files
                else np.asarray([""] * n_sites, dtype=str)
            )
            grain_id = (
                np.asarray(data["site_grain_id"], dtype=int)
                if "site_grain_id" in data.files
                else np.full(n_sites, -1, dtype=int)
            )
            nearest_fe_atom_index = (
                np.asarray(data["nearest_fe_atom_index"], dtype=int)
                if "nearest_fe_atom_index" in data.files
                else np.full(n_sites, -1, dtype=int)
            )
        if len(selection_label) != n_sites:
            raise ValueError(
                f"{path} labels {len(selection_label)} sites, but the active KMC map has {n_sites} sites."
            )
        if len(region_name) != n_sites or len(grain_id) != n_sites:
            raise ValueError(
                f"{path} region arrays must have the same length as the active KMC map "
                f"({n_sites} sites)."
            )
        if len(nearest_fe_atom_index) != n_sites:
            nearest_fe_atom_index = np.full(n_sites, -1, dtype=int)

        mask = np.zeros(n_sites, dtype=bool)
        normalized_specs = [self._normalize_h_region_spec(spec) for spec in region_specs]
        labels_lower = np.char.lower(selection_label.astype(str))
        names_lower = np.char.lower(region_name.astype(str))
        for spec in normalized_specs:
            if spec == "bulk":
                mask |= names_lower == "bulk"
                mask |= np.char.startswith(labels_lower, "bulk_grain_")
            elif spec.startswith("bulk_grain_"):
                mask |= labels_lower == spec
            elif spec.startswith("grain_") and spec.removeprefix("grain_").isdigit():
                mask |= (names_lower == "bulk") & (grain_id == int(spec.removeprefix("grain_")))
            elif spec in {"transition", "grain_boundary"}:
                mask |= names_lower == spec
                mask |= labels_lower == spec
            else:
                mask |= labels_lower == spec
                mask |= names_lower == spec

        arrays = {
            "selection_label": selection_label,
            "region_name": region_name,
            "grain_id": grain_id,
            "nearest_fe_atom_index": nearest_fe_atom_index,
        }
        return mask, arrays, path

    def _select_h_indices_from_regions(self, sites: np.ndarray, num_h: int, region_specs: Sequence[str]) -> np.ndarray:
        mask, arrays, path = self._site_region_mask_from_specs(sites, region_specs)
        selection_label = arrays["selection_label"]
        candidate_idx = np.flatnonzero(mask)
        num_h = int(num_h)
        if num_h <= 0:
            return np.array([], dtype=int)
        if len(candidate_idx) < num_h:
            available = ", ".join(
                f"{label}:{count}" for label, count in zip(*np.unique(selection_label.astype(str), return_counts=True))
            )
            raise RuntimeError(
                f"KMC initial H regions {region_specs!r} provide only {len(candidate_idx)} sites; "
                f"NUM_H={num_h}. Available labels: {available}"
            )
        chosen = np.random.choice(candidate_idx, size=num_h, replace=False)
        return np.asarray(chosen, dtype=int)

    def _control_sites_from_regions_in_x_window(
        self,
        sites: np.ndarray,
        region_specs: Sequence[str],
        *,
        x_min_a: float,
        x_max_a: float,
        label: str,
    ) -> tuple[tuple[int, ...], int | None]:
        mask, arrays, path = self._site_region_mask_from_specs(sites, region_specs)
        positions = np.asarray(sites, dtype=float)
        x = positions[:, 0]
        mask &= (x >= float(x_min_a)) & (x < float(x_max_a))
        control_indices = tuple(int(site) for site in np.flatnonzero(mask))
        if not control_indices:
            raise RuntimeError(
                f"External {label} regions {list(region_specs)!r} from {path} "
                f"provide no sites in x=[{float(x_min_a):g}, {float(x_max_a):g}) A."
            )

        nearest_fe = np.asarray(arrays["nearest_fe_atom_index"], dtype=int)
        fe_indices = nearest_fe[np.asarray(control_indices, dtype=int)]
        fe_indices = fe_indices[fe_indices >= 0]
        fe_count = int(len(np.unique(fe_indices))) if fe_indices.size else None
        if fe_count is not None and fe_count <= 0:
            fe_count = None
        if self.ctx.rank == 0:
            fe_text = "unknown" if fe_count is None else str(fe_count)
            self.log(
                f"[Rank 0] External {label} regions {list(region_specs)!r}: "
                f"sites={len(control_indices)}, Fe_basis={fe_text}, "
                f"x=[{float(x_min_a):g}, {float(x_max_a):g}) A, region_file={path}"
            )
        return control_indices, fe_count

    def _bulk_gcmc_control_from_regions(
        self,
        sites: np.ndarray,
        region_specs: Sequence[str],
        *,
        x_min_a: float,
        x_max_a: float,
    ) -> tuple[tuple[int, ...], int | None]:
        mask, arrays, path = self._site_region_mask_from_specs(sites, region_specs)
        positions = np.asarray(sites, dtype=float)
        x = positions[:, 0]
        x_mask = (x >= float(x_min_a)) & (x < float(x_max_a))
        mask &= x_mask
        control_indices = tuple(int(site) for site in np.flatnonzero(mask))
        if not control_indices:
            raise RuntimeError(
                f"External gcmc_bulk regions {list(region_specs)!r} from {path} "
                f"provide no sites in x=[{float(x_min_a):g}, {float(x_max_a):g}) A."
            )

        nearest_fe = np.asarray(arrays["nearest_fe_atom_index"], dtype=int)
        fe_indices = nearest_fe[np.asarray(control_indices, dtype=int)]
        fe_indices = fe_indices[fe_indices >= 0]
        fe_count = int(len(np.unique(fe_indices))) if fe_indices.size else None
        if fe_count is not None and fe_count <= 0:
            fe_count = None
        if self.ctx.rank == 0:
            fe_text = "unknown" if fe_count is None else str(fe_count)
            self.log(
                f"[Rank 0] External gcmc_bulk control regions {list(region_specs)!r}: "
                f"sites={len(control_indices)}, Fe_basis={fe_text}, "
                f"x=[{float(x_min_a):g}, {float(x_max_a):g}) A, region_file={path}"
            )
        return control_indices, fe_count

    def _site_indices_from_regions_in_x_window(
        self,
        sites: np.ndarray,
        region_specs: Sequence[str],
        *,
        x_min_a: float,
        x_max_a: float,
        label: str,
    ) -> tuple[int, ...]:
        positions = np.asarray(sites, dtype=float)
        x = positions[:, 0]
        mask = (x >= float(x_min_a)) & (x < float(x_max_a))
        path: Path | str = "<all sites>"
        if region_specs:
            region_mask, _arrays, path = self._site_region_mask_from_specs(sites, region_specs)
            mask &= region_mask
        indices = tuple(int(site) for site in np.flatnonzero(mask))
        if not indices:
            region_text = list(region_specs) if region_specs else ["all"]
            raise RuntimeError(
                f"External {label} regions {region_text!r} from {path} provide no sites "
                f"in x=[{float(x_min_a):g}, {float(x_max_a):g}) A."
            )
        if self.ctx.rank == 0:
            region_text = list(region_specs) if region_specs else ["all"]
            self.log(
                f"[Rank 0] External {label} regions {region_text!r}: "
                f"sites={len(indices)}, x=[{float(x_min_a):g}, {float(x_max_a):g}) A, "
                f"region_file={path}"
            )
        return indices

    @staticmethod
    def _tag_neb_frame(atoms: Atoms, mover_idx: Optional[int]) -> Atoms:
        tagged = atoms.copy()
        if bool(np.any(tagged.pbc)):
            tagged.wrap()
        n_atoms = len(tagged)
        types = np.ones(n_atoms, dtype=int)
        masses = np.ones(n_atoms, dtype=float)
        ids = np.arange(1, n_atoms + 1, dtype=int)
        atomic_numbers = tagged.get_atomic_numbers()
        for idx, z in enumerate(atomic_numbers):
            if int(z) == 26:
                types[idx] = 1
                masses[idx] = 55.845
            else:
                masses[idx] = 1.008
                types[idx] = 2 if mover_idx is not None and idx == mover_idx else 3
        tagged.set_array("type", types)
        tagged.set_array("mass", masses)
        tagged.set_array("id", ids)
        return tagged

    def _write_neb_traj(self, frames: Sequence[Atoms], path: Path, mover_idx: Optional[int]) -> None:
        if not frames:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        traj_frames = []
        for atoms in frames:
            tagged = self._tag_neb_frame(atoms, mover_idx)
            if mover_idx is not None and 0 <= mover_idx < len(tagged):
                order = [mover_idx] + [i for i in range(len(tagged)) if i != mover_idx]
                tagged = tagged[order]
            traj_frames.append(tagged)
        write(str(path), traj_frames, format="traj")

    def dump_validation_neb(
        self,
        *,
        h_site: int,
        n_site: int,
        step: int,
        rank: int,
        initial: Atoms,
        final: Atoms,
        barrier_ev: float,
        diag: Optional[Dict[str, Any]],
        timing: Dict[str, float],
        local_mode: str,
    ) -> None:
        if not self._validation_enabled():
            return

        out_dir = (
            self._validation_root()
            / f"rank{rank}_step{int(step)}_h{int(h_site)}"
            / str(self._cfg_value("validation_ase_dir", "ase_dir", "VALIDATION_ASE_DIR", default="ase"))
            / str(int(n_site))
        )
        legacy_h_dir = self._validation_root() / f"rank{rank}_step{int(step)}_h{int(h_site)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        legacy_h_dir.mkdir(parents=True, exist_ok=True)
        overwrite = self._validation_overwrite()

        def should_write(path: Path) -> bool:
            return overwrite or not path.exists()

        mover_idx = None if not diag else diag.get("mover_idx")

        initial_path = out_dir / "initial.lmp"
        final_path = out_dir / "final.lmp"
        if should_write(initial_path):
            self._write_lammps_atomic(initial_path, initial)
        if should_write(final_path):
            self._write_lammps_atomic(final_path, final)

        legacy_initial_path = legacy_h_dir / "initial.data"
        if should_write(legacy_initial_path):
            self._write_lammps_atomic(
                legacy_initial_path,
                initial,
                mover_idx=mover_idx,
                mark_neighbor_h=True,
            )

        legacy_neighbors_path = legacy_h_dir / "neighbors.csv"
        target_pos = np.asarray(final.positions[int(mover_idx)], dtype=float) if mover_idx is not None else np.zeros(3)
        neighbor_rows: dict[int, list[str]] = {}
        if legacy_neighbors_path.exists():
            try:
                with legacy_neighbors_path.open("r", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        existing_n = row.get("n_site")
                        if existing_n is None or str(existing_n).strip() == "":
                            continue
                        neighbor_rows[int(existing_n)] = [
                            str(int(existing_n)),
                            str(row.get("x", "")),
                            str(row.get("y", "")),
                            str(row.get("z", "")),
                        ]
            except Exception:
                neighbor_rows = {}
        if overwrite or int(n_site) not in neighbor_rows:
            neighbor_rows[int(n_site)] = [
                str(int(n_site)),
                f"{float(target_pos[0]):.10f}",
                f"{float(target_pos[1]):.10f}",
                f"{float(target_pos[2]):.10f}",
            ]
        with legacy_neighbors_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["n_site", "x", "y", "z"])
            for key in sorted(neighbor_rows):
                writer.writerow(neighbor_rows[key])

        if self._validation_ase_dump() and diag:
            post_frames = diag.get("images_post") or []
            post_path = out_dir / "dump.ase.post"
            if post_frames and should_write(post_path):
                self._write_neb_traj(post_frames, post_path, mover_idx)

            if self._validation_ase_dump_pre():
                pre_frames = diag.get("images_pre") or []
                pre_path = out_dir / "dump.ase.pre"
                if pre_frames and should_write(pre_path):
                    self._write_neb_traj(pre_frames, pre_path, mover_idx)

        post_frame_count = len((diag or {}).get("images_post") or []) if self._validation_ase_dump() else 0
        pre_frame_count = len((diag or {}).get("images_pre") or []) if self._validation_ase_dump_pre() else 0
        meta = {
            "step": int(step),
            "rank": int(rank),
            "h_site": int(h_site),
            "n_site": int(n_site),
            "local_mode": str(local_mode),
            "barrier_eV": float(barrier_ev) if np.isfinite(barrier_ev) else None,
            "mover_idx": None if mover_idx is None else int(mover_idx),
            "diagnostics_collected": bool(diag),
            "ase_neb_post_frames": post_frame_count,
            "ase_neb_pre_frames": pre_frame_count,
            "timings": dict(timing or {}),
            "files": {
                "initial_lammps": initial_path.name,
                "final_lammps": final_path.name,
                "ase_post": "dump.ase.post" if self._validation_ase_dump() else None,
                "ase_pre": "dump.ase.pre" if self._validation_ase_dump_pre() else None,
            },
        }
        meta_path = out_dir / "meta.json"
        if should_write(meta_path):
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, sort_keys=True)

        summary_path = self._validation_root() / f"neb_validation_summary_rank{int(rank)}.csv"
        write_header = not summary_path.exists()
        with summary_path.open("a", newline="", encoding="utf-8") as fh:
            fields = [
                "step",
                "rank",
                "h_site",
                "n_site",
                "local_mode",
                "barrier_eV",
                "diagnostics_collected",
                "ase_neb_post_frames",
                "ase_neb_pre_frames",
                "total_s",
                "optimizer_s",
                "energy_eval_s",
                "out_dir",
            ]
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "step": int(step),
                    "rank": int(rank),
                    "h_site": int(h_site),
                    "n_site": int(n_site),
                    "local_mode": str(local_mode),
                    "barrier_eV": "" if not np.isfinite(barrier_ev) else f"{float(barrier_ev):.12g}",
                    "diagnostics_collected": int(bool(diag)),
                    "ase_neb_post_frames": meta["ase_neb_post_frames"],
                    "ase_neb_pre_frames": meta["ase_neb_pre_frames"],
                    "total_s": f"{float((timing or {}).get('total_s', 0.0)):.6f}",
                    "optimizer_s": f"{float((timing or {}).get('optimizer_s', 0.0)):.6f}",
                    "energy_eval_s": f"{float((timing or {}).get('energy_eval_s', 0.0)):.6f}",
                    "out_dir": str(out_dir),
                }
            )

    def run_endpoint_opt(
        self,
        target_atoms: Atoms,
        logfile: str,
        rank: int,
        calc: Any,
        job_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if len(target_atoms) == 0:
            self.log(f"[Rank {rank}] ERROR: Empty Atoms object passed to run_opt")
            return False
        target_atoms.set_array("id", np.arange(1, len(target_atoms) + 1))
        if "type" not in target_atoms.arrays or "mass" not in target_atoms.arrays:
            assign_mass_and_type(target_atoms)
        if target_atoms.calc is None:
            target_atoms.calc = calc
        ase_opt_log = self._ase_log_path(logfile or "opt", job_context=job_context)
        if ase_opt_log != os.devnull:
            self.log(f"[Rank {rank}] ASE optimization log -> {ase_opt_log}")
        try:
            _ = target_atoms.get_forces()
            _ = target_atoms.get_potential_energy()
            opt = BFGS(target_atoms, logfile=ase_opt_log)
            opt.run(
                fmax=float(self._cfg_value("optimizer_fmax", "NEB_OPTIMIZER_FMAX", default=0.05)),
                steps=int(self._cfg_value("endpoint_optimizer_steps", "ENDPOINT_OPTIMIZER_STEPS", default=200)),
            )
            return bool(np.isfinite(target_atoms.get_positions()).all())
        except Exception as exc:
            self.log(f"[Rank {rank}] Optimization error: {exc}")
            return False

    @staticmethod
    def _write_lammps_atomic(
        path: Path,
        atoms: Atoms,
        *,
        mover_idx: Optional[int] = None,
        mark_neighbor_h: bool = False,
    ) -> None:
        positions = np.asarray(atoms.get_positions(), dtype=float)
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        cell = np.asarray(atoms.cell.array, dtype=float)
        lengths = np.diag(cell) if cell.shape == (3, 3) else np.asarray(atoms.cell.lengths(), dtype=float)
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        lo = np.minimum(np.zeros(3), mins) - 1.0e-6
        hi = np.maximum(lengths, maxs) + 1.0e-6
        types = np.asarray(atoms.arrays.get("type", np.ones(len(atoms), dtype=int)), dtype=int)
        effective_types = types.copy()
        effective_types[atomic_numbers == 26] = 1
        n_atom_types = max(2, int(np.max(effective_types)) if len(effective_types) else 2)

        with path.open("w", encoding="utf-8") as fh:
            fh.write("LAMMPS data file via kmc refactor structure snapshot\n\n")
            fh.write(f"{len(atoms)} atoms\n")
            fh.write(f"{n_atom_types} atom types\n\n")
            fh.write(f"{float(lo[0]):.10f} {float(hi[0]):.10f} xlo xhi\n")
            fh.write(f"{float(lo[1]):.10f} {float(hi[1]):.10f} ylo yhi\n")
            fh.write(f"{float(lo[2]):.10f} {float(hi[2]):.10f} zlo zhi\n\n")
            fh.write("Masses\n\n")
            fh.write("1 55.845\n")
            fh.write("2 1.008\n")
            if n_atom_types >= 3:
                fh.write("3 1.008\n")
            if n_atom_types >= 4:
                fh.write("4 1.008\n")
            fh.write("\n")
            fh.write("Atoms # atomic\n\n")
            for idx0, (atype, z, pos) in enumerate(zip(types, atomic_numbers, positions)):
                if z == 26:
                    atype = 1
                comment = ""
                if mark_neighbor_h and int(z) == 1 and mover_idx is not None and idx0 != int(mover_idx):
                    comment = " # neighbor"
                fh.write(
                    f"{idx0 + 1} {int(atype)} "
                    f"{pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}{comment}\n"
                )

    @staticmethod
    def _write_atoms_xyz(path: Path, atoms: Atoms) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"{len(atoms)}\n")
            fh.write("local environment build\n")
            for atom in atoms:
                x, y, z = atom.position
                fh.write(f"{atom.symbol} {float(x):.10f} {float(y):.10f} {float(z):.10f}\n")

    def _nearby_h_count(self, h_site: int, radius_a: float) -> int:
        assert self.sites is not None
        assert self.h_indices is not None
        center = np.asarray(self.sites[int(h_site)], dtype=float)
        count = 0
        for other_site in map(int, self.h_indices):
            if other_site == int(h_site):
                continue
            diff = pbc_diff(center, self.sites[other_site], self.spec.box)
            if float(np.linalg.norm(diff)) <= float(radius_a) + 1.0e-12:
                count += 1
        return count

    def dump_local_env_build(
        self,
        out_dir: Path,
        initial_atoms: Atoms,
        final_atoms: Atoms,
        mover_idx: int,
        *,
        mode: str,
        h_index: int,
        n_index: int,
        env_key: Any = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_atoms_xyz(out_dir / "initial.xyz", initial_atoms)
        self._write_atoms_xyz(out_dir / "final.xyz", final_atoms)

        freeze_mask = np.asarray(
            initial_atoms.arrays.get("freeze_mask", np.zeros(len(initial_atoms), dtype=bool)),
            dtype=bool,
        )
        shell_distance = np.asarray(
            initial_atoms.arrays.get("shell_distance", np.full(len(initial_atoms), np.nan)),
            dtype=float,
        )
        with (out_dir / "atom_roles.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["atom_index", "symbol", "role", "frozen", "distance_A", "x", "y", "z"])
            for idx, atom in enumerate(initial_atoms):
                if idx == int(mover_idx):
                    role = "mover"
                elif idx < len(freeze_mask) and bool(freeze_mask[idx]):
                    role = "shell_frozen"
                else:
                    role = "inner_movable"
                x, y, z = atom.position
                distance = ""
                if idx < len(shell_distance) and np.isfinite(shell_distance[idx]):
                    distance = float(shell_distance[idx])
                writer.writerow(
                    [
                        int(idx),
                        atom.symbol,
                        role,
                        int(idx < len(freeze_mask) and bool(freeze_mask[idx])),
                        distance,
                        float(x),
                        float(y),
                        float(z),
                    ]
                )

        meta = {
            "mode": str(mode),
            "h_site": int(h_index),
            "n_site": int(n_index),
            "mover_idx": int(mover_idx),
            "num_atoms": int(len(initial_atoms)),
            "num_fe": int(sum(1 for s in initial_atoms.get_chemical_symbols() if s == "Fe")),
            "num_h": int(sum(1 for s in initial_atoms.get_chemical_symbols() if s == "H")),
            "num_neighbor_h": max(
                0,
                int(sum(1 for s in initial_atoms.get_chemical_symbols() if s == "H")) - 1,
            ),
            "num_frozen": int(np.sum(freeze_mask)),
            "cell_A": [float(v) for v in np.diag(np.asarray(initial_atoms.cell, dtype=float))],
            "env_key": str(env_key) if env_key is not None else None,
        }
        with (out_dir / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def build_and_dump_local_envs(self) -> int:
        assert self.sites is not None
        assert self.neighbors is not None
        assert self.h_indices is not None

        out_dir = Path(str(self._cfg_value("build_out", "BUILD_LOCAL_ENV_OUT", default="local_env_builds")))
        count = int(self._cfg_value("build_count", "BUILD_LOCAL_ENV_COUNT", default=10))
        mode = str(self._cfg_value("local_env_mode", "mode", "LOCAL_ENV_MODE", default="wrapped")).lower()
        if mode not in {"wrapped", "radial", "shell"}:
            mode = "wrapped"
        out_dir.mkdir(parents=True, exist_ok=True)

        occupied = set(map(int, self.h_indices))
        if mode == "shell":
            score_radius = float(self._cfg_value("shell_outer_radius_a", "SHELL_OUTER_RADIUS_A", default=6.0))
        elif mode == "radial":
            score_radius = float(self._cfg_value("env_radius_a", "radius_a", "ENV_RADIUS_A", default=5.0))
        else:
            score_radius = 3.0 * float(self.spec.lattice_a)

        candidates: list[tuple[int, int, int]] = []
        for h_site in map(int, self.h_indices):
            if h_site >= len(self.neighbors):
                continue
            nearby_h = self._nearby_h_count(h_site, score_radius)
            for n_site in map(int, self.neighbors[h_site]):
                if n_site in occupied:
                    continue
                candidates.append((-nearby_h, int(h_site), int(n_site)))
        candidates.sort()

        built = 0
        for neg_nearby_h, h_site, n_site in candidates:
            env_key = self.make_event_key(h_site, n_site)
            try:
                initial, final, mover_idx, actual_mode = self._build_local_neb_endpoints(h_site, n_site, mode)
            except Exception as exc:
                self.log(f"[Rank {self.ctx.rank}] BUILD_LOCAL_ENV_ONLY skip h={h_site}->n={n_site}: {exc}")
                continue

            nearby_h = -int(neg_nearby_h)
            case_dir = out_dir / f"env_{built:02d}_nearH{nearby_h}_h{h_site}_n{n_site}"
            self.dump_local_env_build(
                case_dir,
                initial,
                final,
                int(mover_idx),
                mode=actual_mode,
                h_index=h_site,
                n_index=n_site,
                env_key=env_key,
            )
            built += 1
            if built >= count:
                return built
        return built

    def dump_neb_structure_snapshot(
        self,
        initial_atoms: Atoms,
        final_atoms: Atoms,
        mover_idx: Optional[int],
        job_context: Dict[str, Any],
        step: int,
    ) -> None:
        if initial_atoms is None or final_atoms is None or mover_idx is None:
            return
        if not self._debug_flag("STRUCTURE_LOG_ALL", False):
            return
        try:
            if not (0 <= mover_idx < len(final_atoms)):
                return
            snapshot = initial_atoms.copy()
            target_pos = np.asarray(final_atoms.positions[mover_idx], dtype=float)
            snapshot += Atoms("H", positions=[target_pos], cell=snapshot.cell, pbc=bool(np.any(snapshot.pbc)))
            if bool(np.any(snapshot.pbc)):
                snapshot.wrap()
            assign_mass_and_type(snapshot)

            types = np.array(snapshot.arrays.get("type", []), dtype=int)
            masses = snapshot.arrays.get("mass").copy()
            init_pos = np.asarray(initial_atoms.positions[mover_idx], dtype=float)
            h_indices = [i for i, z in enumerate(snapshot.get_atomic_numbers()) if z == 1]
            if h_indices:
                init_idx = min(h_indices, key=lambda idx: np.linalg.norm(snapshot.positions[idx] - init_pos))
                final_idx = min(h_indices, key=lambda idx: np.linalg.norm(snapshot.positions[idx] - target_pos))
                for idx in h_indices:
                    if masses.shape and idx < len(masses):
                        masses[idx] = 1.008
                    types[idx] = 3
                if init_idx < len(types):
                    types[init_idx] = 2
                if final_idx < len(types):
                    types[final_idx] = 4
                    masses[final_idx] = 1.008
            snapshot.set_array("type", types)
            snapshot.set_array("mass", masses)

            path = self._structure_snapshot_path(job_context)
            if path is None:
                return
            self._write_lammps_atomic(path, snapshot)
            self.log(f"[Rank {self.ctx.rank}] Structure snapshot written to {path}")
        except Exception as exc:
            self.log(f"[Rank {self.ctx.rank}] Structure snapshot failed: {exc}")

    @staticmethod
    def _append_dict_csv(
        path: Path,
        fields: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        *,
        write_header: bool,
        mode: str = "a",
    ) -> None:
        if not rows:
            return
        with path.open(mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_rank_neb_diag(self, summary: NEBDiagnosticSummary) -> None:
        path = self._neb_output_dir() / f"neb_diag_rank{self.ctx.rank}.csv"
        first_write = not self._neb_diag_header_written
        self._append_dict_csv(
            path,
            NEBDiagnosticSummary.fields(),
            [summary.row()],
            write_header=first_write,
            mode="w" if first_write else "a",
        )
        self._neb_diag_header_written = True

    def _write_all_neb_diag(self, summaries: Sequence[NEBDiagnosticSummary]) -> None:
        if self.ctx.rank != 0:
            return
        path = self._neb_output_dir() / "neb_diag_all.csv"
        first_write = not self._neb_diag_all_header_written
        rows = [summary.row() for summary in summaries]
        self._append_dict_csv(
            path,
            NEBDiagnosticSummary.fields(),
            rows,
            write_header=first_write,
            mode="w" if first_write else "a",
        )
        self._neb_diag_all_header_written = True

    def _event_geometry_row(self, h_site: int, n_site: int) -> dict[str, str]:
        if self.sites is None:
            return {
                "from_x_A": "",
                "from_y_A": "",
                "from_z_A": "",
                "to_x_A": "",
                "to_y_A": "",
                "to_z_A": "",
                "dx_A": "",
                "dy_A": "",
                "dz_A": "",
                "hop_distance_A": "",
            }

        start = np.asarray(self.sites[int(h_site)], dtype=float)
        end = np.asarray(self.sites[int(n_site)], dtype=float)
        delta = pbc_diff(start, end, self.spec.box)
        return {
            "from_x_A": f"{float(start[0]):.8f}",
            "from_y_A": f"{float(start[1]):.8f}",
            "from_z_A": f"{float(start[2]):.8f}",
            "to_x_A": f"{float(end[0]):.8f}",
            "to_y_A": f"{float(end[1]):.8f}",
            "to_z_A": f"{float(end[2]):.8f}",
            "dx_A": f"{float(delta[0]):.8f}",
            "dy_A": f"{float(delta[1]):.8f}",
            "dz_A": f"{float(delta[2]):.8f}",
            "hop_distance_A": f"{float(np.linalg.norm(delta)):.8f}",
        }

    def _h_site_to_atom_id(self) -> dict[int, int]:
        if self.h_indices is None:
            return {}
        return {int(site): int(idx) + 1 for idx, site in enumerate(self.h_indices)}

    def _write_rates_allranks(self, gathered_rate_rows: Sequence[Sequence[Any]], selected: Any = None) -> None:
        if self.ctx.rank != 0:
            return
        if not self._write_rates_allranks_enabled():
            return
        rows = []
        site_to_atom_id = self._h_site_to_atom_id()
        for part in gathered_rate_rows:
            for item in part:
                raw = item.as_tuple() if hasattr(item, "as_tuple") else tuple(item)
                if len(raw) < 9:
                    continue
                step, rank, h_site, n_site, barrier_ev, rate_hz, source, env_kind, status = raw[:9]
                h_site_i = int(h_site)
                n_site_i = int(n_site)
                geom = self._event_geometry_row(h_site_i, n_site_i)
                is_selected = (
                    selected is not None
                    and h_site_i == int(selected.h_site)
                    and n_site_i == int(selected.n_site)
                )
                rows.append(
                    [
                        int(step),
                        int(rank),
                        site_to_atom_id.get(h_site_i, ""),
                        h_site_i,
                        n_site_i,
                        geom["from_x_A"],
                        geom["from_y_A"],
                        geom["from_z_A"],
                        geom["to_x_A"],
                        geom["to_y_A"],
                        geom["to_z_A"],
                        geom["dx_A"],
                        geom["dy_A"],
                        geom["dz_A"],
                        geom["hop_distance_A"],
                        barrier_ev,
                        rate_hz,
                        source,
                        env_kind,
                        status,
                        int(is_selected),
                    ]
                )
        if not rows:
            return
        path = self._diagnostics_output_dir() / "rates_allranks.csv"
        fields = [
            "step",
            "rank",
            "atom_id",
            "h_site",
            "n_site",
            "from_x_A",
            "from_y_A",
            "from_z_A",
            "to_x_A",
            "to_y_A",
            "to_z_A",
            "dx_A",
            "dy_A",
            "dz_A",
            "hop_distance_A",
            "barrier_eV",
            "rate_Hz",
            "source",
            "env_kind",
            "status",
            "selected",
        ]
        first_write = not self._rates_header_written
        with path.open("w" if first_write else "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if first_write:
                writer.writerow(fields)
            writer.writerows(rows)
        self._rates_header_written = True

    def _init_kmc_diag(self) -> None:
        if self.ctx.rank != 0 or self._kmc_diag_initialized:
            return
        path = self._diagnostics_output_dir() / "kmc_diagnostics_rank0.log"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("KMC Diagnostics Log (Rank 0)\n")
            fh.write("Step | atom_id | H move | barrier (eV) | rate (Hz) | dt (s) | Total t (s)\n")
        self._kmc_diag_initialized = True

    def _append_kmc_diag(self, step: int, selected: Any, moved_slot: Any = None) -> None:
        if self.ctx.rank != 0 or selected is None:
            return
        self._init_kmc_diag()
        atom_id = "" if moved_slot is None else int(moved_slot) + 1
        barrier = "" if selected.barrier_ev is None else f"{float(selected.barrier_ev):.8f}"
        path = self._diagnostics_output_dir() / "kmc_diagnostics_rank0.log"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{int(step)} | atom_id {atom_id} | H site {int(selected.h_site)} -> {int(selected.n_site)} "
                f"(aggregated 1 step(s)) | barrier = {barrier} eV | "
                f"rate = {float(selected.rate_hz):.6e} Hz | dt = {float(selected.dt_s):.6e} s | "
                f"t_total = {self.sim_time_s:.6e} s\n"
            )

    def _append_selected_event_csv(self, step: int, selected: Any, moved_slot: Any = None) -> None:
        if self.ctx.rank != 0 or selected is None:
            return
        path = self._diagnostics_output_dir() / "kmc_selected_events.csv"
        fields = [
            "step",
            "atom_id",
            "h_site",
            "n_site",
            "from_x_A",
            "from_y_A",
            "from_z_A",
            "to_x_A",
            "to_y_A",
            "to_z_A",
            "dx_A",
            "dy_A",
            "dz_A",
            "hop_distance_A",
            "barrier_eV",
            "rate_Hz",
            "total_rate_Hz",
            "dt_s",
            "total_time_s",
            "random_index_value",
            "random_time_value",
        ]
        geom = self._event_geometry_row(int(selected.h_site), int(selected.n_site))
        row = {
            "step": int(step),
            "atom_id": "" if moved_slot is None else int(moved_slot) + 1,
            "h_site": int(selected.h_site),
            "n_site": int(selected.n_site),
            **geom,
            "barrier_eV": "" if selected.barrier_ev is None else f"{float(selected.barrier_ev):.10g}",
            "rate_Hz": f"{float(selected.rate_hz):.10g}",
            "total_rate_Hz": f"{float(selected.total_rate_hz):.10g}",
            "dt_s": f"{float(selected.dt_s):.10g}",
            "total_time_s": f"{float(self.sim_time_s):.10g}",
            "random_index_value": f"{float(selected.random_index_value):.10g}",
            "random_time_value": f"{float(selected.random_time_value):.10g}",
        }
        first_write = not self._selected_events_header_written
        self._append_dict_csv(
            path,
            fields,
            [row],
            write_header=first_write,
            mode="w" if first_write else "a",
        )
        self._selected_events_header_written = True

    def _append_kmc_timestep(self, step: int, selected: Any) -> None:
        if self.ctx.rank != 0 or selected is None:
            return
        path = self._diagnostics_output_dir() / "kmc_timestep_vs_step.csv"
        first_write = not self._timestep_header_written
        with path.open("w" if first_write else "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if first_write:
                writer.writerow(["step", "timestep", "total"])
            writer.writerow([int(step), f"{float(selected.dt_s):.6e}", f"{self.sim_time_s:.6e}"])
        self._timestep_header_written = True

    def _append_only_h_trajectory(self, step: int) -> None:
        if self.ctx.rank != 0 or self.sites is None or self.h_indices is None:
            return
        path = self._trajectory_output_dir() / "H_trajectory_onlyH.lammpstrj"
        mode = "a" if self._only_h_traj_initialized else "w"
        positions = (
            np.asarray(self.h_unwrapped_positions, dtype=float)
            if self.h_unwrapped_positions is not None
            else np.asarray(self.sites[self.h_indices], dtype=float)
        )
        box = np.asarray(self.spec.box, dtype=float)
        with path.open(mode, encoding="utf-8") as fh:
            fh.write("ITEM: TIMESTEP\n")
            fh.write(f"{int(step)}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n")
            fh.write(f"{len(positions)}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            fh.write(f"0.0 {box[0]:.10f}\n0.0 {box[1]:.10f}\n0.0 {box[2]:.10f}\n")
            fh.write("ITEM: ATOMS id type x y z site\n")
            for idx, (site, pos) in enumerate(zip(self.h_indices, positions), start=1):
                fh.write(
                    f"{idx} 2 {float(pos[0]):.10f} {float(pos[1]):.10f} "
                    f"{float(pos[2]):.10f} {int(site)}\n"
                )
        self._only_h_traj_initialized = True

    def _checkpoint_payload(self, step: int) -> dict[str, Any]:
        assert self.h_indices is not None
        assert self.h_unwrapped_positions is not None
        devanathan_state = None
        state_writer = getattr(self.devanathan_boundary, "state_dict", None)
        if callable(state_writer):
            devanathan_state = state_writer()
        return {
            "version": 2,
            "step": int(step),
            "sim_time_s": float(self.sim_time_s),
            "h_indices": np.asarray(self.h_indices, dtype=int),
            "h_unwrapped_positions": np.asarray(self.h_unwrapped_positions, dtype=float),
            "cache_schema": str(self.cache_mgr.cfg.schema),
            "num_h": int(len(self.h_indices)),
            "box": np.asarray(self.spec.box, dtype=float),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "devanathan_state": devanathan_state,
            "site_source": self._site_source(),
            "site_map_file": str(self._cfg_value("site_map_file", "KMC_SITE_MAP_FILE", "SITE_MAP_FILE", default="") or ""),
            "host_structure_file": str(
                self._cfg_value("host_structure_file", "KMC_HOST_STRUCTURE_FILE", "HOST_STRUCTURE_FILE", default="")
                or ""
            ),
        }

    def _write_restart_checkpoint(self, step: int, *, force: bool = False) -> None:
        if self.ctx.rank != 0 or self.h_indices is None or self.h_unwrapped_positions is None:
            return
        interval = self._checkpoint_interval()
        if not force and (interval <= 0 or int(step) <= 0 or int(step) % interval != 0):
            return

        out_dir = self._checkpoint_dir()
        prefix = self._checkpoint_prefix()
        payload = self._checkpoint_payload(step)
        latest_path = out_dir / f"{prefix}.pkl"
        step_path = out_dir / f"{prefix}_step{int(step)}.pkl"
        tmp_latest = latest_path.with_suffix(latest_path.suffix + ".tmp")
        tmp_step = step_path.with_suffix(step_path.suffix + ".tmp")
        try:
            with tmp_step.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_step, step_path)
            with tmp_latest.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_latest, latest_path)
            self.log(f"[Rank 0] Wrote restart checkpoint for step {int(step)} to {step_path}")
        except Exception as exc:
            self.log(f"[Rank 0] WARNING: failed to write restart checkpoint for step {int(step)}: {exc}")
            for tmp in (tmp_step, tmp_latest):
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

    def _load_restart_checkpoint(self, path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Restart checkpoint {path} did not contain a mapping")
        return dict(payload)

    def _apply_restart_checkpoint(
        self,
        payload: Mapping[str, Any],
        path: Path,
        *,
        explicit: bool = False,
    ) -> bool:
        assert self.sites is not None
        checkpoint_step = int(payload.get("step", -1))
        requested_step = self._restart_step()
        if requested_step and checkpoint_step != requested_step:
            msg = (
                f"Restart checkpoint {path} is for step {checkpoint_step}, "
                f"but RESTART_STEP={requested_step}."
            )
            if explicit and self._restart_strict():
                raise ValueError(msg)
            self.log(f"[Rank {self.ctx.rank}] WARNING: {msg} Falling back to trajectory restart.")
            return False

        cache_schema = str(payload.get("cache_schema", ""))
        if cache_schema and cache_schema != str(self.cache_mgr.cfg.schema):
            msg = (
                f"Restart checkpoint {path} cache schema {cache_schema!r} does not match "
                f"current schema {self.cache_mgr.cfg.schema!r}."
            )
            if self._restart_strict():
                raise ValueError(msg)
            self.log(f"[Rank {self.ctx.rank}] WARNING: {msg}")

        h_indices = np.asarray(payload.get("h_indices"), dtype=int)
        h_unwrapped = np.asarray(payload.get("h_unwrapped_positions"), dtype=float)
        if h_indices.ndim != 1 or h_unwrapped.shape != (len(h_indices), 3):
            raise ValueError(f"Restart checkpoint {path} has invalid H state shapes")
        if np.any(h_indices < 0) or np.any(h_indices >= len(self.sites)):
            raise ValueError(f"Restart checkpoint {path} contains site indices outside the current site map")
        if len(np.unique(h_indices)) != len(h_indices):
            raise ValueError(f"Restart checkpoint {path} contains duplicate occupied H site indices")

        self.h_indices = h_indices
        self.h_unwrapped_positions = h_unwrapped
        self.h_initial_unwrapped_positions = h_unwrapped.copy()
        self.sim_time_s = float(payload.get("sim_time_s", 0.0))
        self.restart_start_step = checkpoint_step
        raw_devanathan_state = payload.get("devanathan_state")
        self._pending_devanathan_state = (
            dict(raw_devanathan_state)
            if isinstance(raw_devanathan_state, Mapping)
            else None
        )

        py_state = payload.get("python_random_state")
        np_state = payload.get("numpy_random_state")
        if py_state is not None:
            random.setstate(py_state)
        if np_state is not None:
            np.random.set_state(np_state)

        if self.ctx.rank == 0:
            self.log(
                f"[Rank 0] Restart loaded checkpoint {path}: "
                f"step={checkpoint_step}, H={len(h_indices)}, t={self.sim_time_s:.6e} s"
            )
        return True

    def _restore_devanathan_state(self) -> None:
        if self._pending_devanathan_state is None or self.devanathan_boundary is None:
            return
        state_loader = getattr(self.devanathan_boundary, "load_state_dict", None)
        if not callable(state_loader):
            self.log(
                "[Rank 0] WARNING: restart checkpoint contains Devanathan state "
                "but the configured boundary does not support restoring it"
            )
            return
        state_loader(self._pending_devanathan_state)
        self._pending_devanathan_state = None
        self.log("[Rank 0] Restored Devanathan/GCMC controller state from checkpoint")

    def _read_restart_h_frame(self, path: Path, target_step: int) -> tuple[np.ndarray, np.ndarray, int]:
        if not path.exists():
            raise FileNotFoundError(f"Restart trajectory not found: {path}")

        found_step: Optional[int] = None
        found_sites: Optional[np.ndarray] = None
        found_positions: Optional[np.ndarray] = None

        with path.open("r", encoding="utf-8") as fh:
            while True:
                line = fh.readline()
                if not line:
                    break
                if line.strip() != "ITEM: TIMESTEP":
                    continue

                step_line = fh.readline()
                if not step_line:
                    break
                frame_step = int(step_line.strip())

                if fh.readline().strip() != "ITEM: NUMBER OF ATOMS":
                    raise ValueError(f"Malformed restart trajectory near timestep {frame_step}: missing atom count")
                n_atoms = int(fh.readline().strip())

                if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                    raise ValueError(f"Malformed restart trajectory near timestep {frame_step}: missing box bounds")
                for _ in range(3):
                    fh.readline()

                atom_header = fh.readline().strip()
                if not atom_header.startswith("ITEM: ATOMS"):
                    raise ValueError(f"Malformed restart trajectory near timestep {frame_step}: missing atom header")
                columns = atom_header.split()[2:]
                col_index = {name: idx for idx, name in enumerate(columns)}
                required = {"id", "x", "y", "z", "site"}
                missing = sorted(required.difference(col_index))
                if missing:
                    raise ValueError(
                        f"Restart trajectory frame {frame_step} is missing required atom columns: {missing}"
                    )

                rows: list[list[str]] = []
                for _ in range(n_atoms):
                    rows.append(fh.readline().split())

                if frame_step != int(target_step):
                    continue

                parsed: list[tuple[int, int, np.ndarray]] = []
                for row in rows:
                    atom_id = int(row[col_index["id"]])
                    site = int(row[col_index["site"]])
                    pos = np.asarray(
                        [
                            float(row[col_index["x"]]),
                            float(row[col_index["y"]]),
                            float(row[col_index["z"]]),
                        ],
                        dtype=float,
                    )
                    parsed.append((atom_id, site, pos))
                parsed.sort(key=lambda item: item[0])
                found_step = frame_step
                found_sites = np.asarray([site for _atom_id, site, _pos in parsed], dtype=int)
                found_positions = np.asarray([pos for _atom_id, _site, pos in parsed], dtype=float)
                break

        if found_sites is None or found_positions is None or found_step is None:
            raise ValueError(f"Restart trajectory {path} does not contain TIMESTEP {int(target_step)}")
        return found_sites, found_positions, found_step

    def _read_restart_time(self, path: Path, restart_step: int) -> float:
        if restart_step <= 0:
            return 0.0
        if not path.exists():
            diag_path = self._restart_file("diagnostics_file", "RESTART_DIAGNOSTICS_FILE", "kmc_diagnostics_rank0.log")
            diag_time = self._read_restart_time_from_diagnostics(diag_path, restart_step)
            if diag_time is not None:
                return diag_time
            if self._restart_strict():
                raise FileNotFoundError(
                    f"Restart timestep CSV not found and diagnostics fallback failed: {path}, {diag_path}"
                )
            self.log(f"[Rank {self.ctx.rank}] Restart time files not found; continuing with t=0: {path}")
            return 0.0

        target_kmc_step = int(restart_step) - 1
        last_total = 0.0
        with path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    row_step = int(row.get("step", ""))
                except Exception:
                    continue
                total_raw = row.get("total", "")
                if total_raw not in (None, ""):
                    try:
                        last_total = float(total_raw)
                    except Exception:
                        pass
                if row_step == target_kmc_step:
                    return last_total
        diag_path = self._restart_file("diagnostics_file", "RESTART_DIAGNOSTICS_FILE", "kmc_diagnostics_rank0.log")
        diag_time = self._read_restart_time_from_diagnostics(diag_path, restart_step)
        if diag_time is not None:
            return diag_time
        if self._restart_strict():
            raise ValueError(
                f"Restart timestep CSV {path} has no row for completed KMC step {target_kmc_step} "
                f"and diagnostics fallback failed: {diag_path}"
            )
        return last_total

    def _read_restart_time_from_diagnostics(self, path: Path, restart_step: int) -> Optional[float]:
        target_kmc_step = int(restart_step) - 1
        if target_kmc_step < 0 or not path.exists():
            return None
        pattern = re.compile(r"^\s*(?P<step>\d+)\s*\|.*t_total\s*=\s*(?P<total>[0-9eE+\-.]+)")
        last_total: Optional[float] = None
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                match = pattern.search(line)
                if not match:
                    continue
                try:
                    row_step = int(match.group("step"))
                    total = float(match.group("total"))
                except Exception:
                    continue
                last_total = total
                if row_step == target_kmc_step:
                    return total
        if not self._restart_strict():
            return last_total
        return None

    def _apply_restart_state(self) -> None:
        if not self._restart_enabled():
            return
        assert self.sites is not None

        for checkpoint_path, explicit_checkpoint in self._restart_checkpoint_candidates():
            checkpoint = self._load_restart_checkpoint(checkpoint_path)
            if checkpoint is not None and self._apply_restart_checkpoint(
                checkpoint,
                checkpoint_path,
                explicit=explicit_checkpoint,
            ):
                return

        step = self._restart_step()
        traj_path = self._restart_trajectory_path()
        time_path = self._restart_timestep_path()

        h_indices, h_unwrapped, frame_step = self._read_restart_h_frame(traj_path, step)
        if np.any(h_indices < 0) or np.any(h_indices >= len(self.sites)):
            bad = h_indices[(h_indices < 0) | (h_indices >= len(self.sites))]
            raise ValueError(
                f"Restart frame {frame_step} contains site indices outside the current site map; "
                f"first bad entries: {bad[:10].tolist()}"
            )
        if len(np.unique(h_indices)) != len(h_indices):
            raise ValueError(f"Restart frame {frame_step} contains duplicate occupied H site indices")

        expected_h = int(self._cfg_value("num_h", "NUM_H", default=len(h_indices)))
        if len(h_indices) != expected_h:
            msg = (
                f"Restart frame {frame_step} contains {len(h_indices)} H atoms but NUM_H={expected_h}. "
                "Use the same NUM_H as the original run or set NUM_H to the frame count."
            )
            if self._restart_strict():
                raise ValueError(msg)
            self.log(f"[Rank {self.ctx.rank}] WARNING: {msg}")

        self.h_indices = h_indices
        self.h_unwrapped_positions = h_unwrapped
        self.h_initial_unwrapped_positions = h_unwrapped.copy()
        self.sim_time_s = self._read_restart_time(time_path, frame_step)
        self.restart_start_step = int(frame_step)
        if self.ctx.rank == 0:
            self.log(
                f"[Rank 0] Restart loaded state from {traj_path}: "
                f"frame={frame_step}, H={len(h_indices)}, t={self.sim_time_s:.6e} s"
            )

    def _preload_restart_cache(self) -> None:
        if not self._restart_enabled() or not self.scheduler.cfg.use_cache_for_jobs:
            return

        root = self._restart_root()
        cache_roots = [root / "cache", root]
        cfg = self.cache_mgr.cfg
        loaded: dict[Any, Any] = {}
        try:
            from .services.cache import BarrierCache  # type: ignore
        except Exception:
            try:
                from services.cache import BarrierCache  # type: ignore
            except Exception:
                from cache import BarrierCache  # type: ignore

        for cache_root in cache_roots:
            merged_path = cache_root / f"barrier_cache_{cfg.schema}.pkl"
            if merged_path.exists():
                try:
                    loaded.update(dict(BarrierCache(str(merged_path))))
                except Exception as exc:
                    if self._restart_strict():
                        raise RuntimeError(f"Failed to load restart merged barrier cache {merged_path}: {exc}") from exc
                    self.log(f"[Rank {self.ctx.rank}] WARNING: failed to load restart merged cache {merged_path}: {exc}")
                break
            for rank_idx in range(int(self.ctx.size)):
                rank_path = cache_root / f"barrier_cache_rank{rank_idx}_{cfg.schema}.pkl"
                delta_path = Path(str(rank_path) + ".delta.pkl")
                if not rank_path.exists() and not delta_path.exists():
                    continue
                try:
                    loaded.update(dict(BarrierCache(str(rank_path))))
                except Exception as exc:
                    if self._restart_strict():
                        raise RuntimeError(f"Failed to load restart rank cache {rank_path}: {exc}") from exc
                    self.log(f"[Rank {self.ctx.rank}] WARNING: failed to load restart rank cache {rank_path}: {exc}")
            if loaded:
                break

        if loaded:
            self.cache_mgr.cache.store.update(loaded)
            if self.ctx.rank == 0:
                self.log(f"[Rank 0] Restart preloaded {len(loaded)} barrier cache entries from {root}")
        elif self._restart_strict():
            raise FileNotFoundError(
                f"No restart barrier cache found in {root} for schema {cfg.schema}. "
                "Expected cache/barrier_cache_<schema>.pkl or "
                "cache/barrier_cache_rank<RANK>_<schema>.pkl[.delta.pkl] "
                "(legacy top-level cache filenames are also accepted)."
            )
        elif self.ctx.rank == 0:
            self.log(f"[Rank 0] WARNING: no restart barrier cache found in {root} for schema {cfg.schema}")

    def _site_source(self) -> str:
        raw = str(self._cfg_value("site_source", "source", "KMC_SITE_SOURCE", default="generated") or "generated")
        source = raw.lower().strip()
        if source in {"map", "file", "external_map"}:
            source = "external"
        site_map = str(self._cfg_value("site_map_file", "KMC_SITE_MAP_FILE", "SITE_MAP_FILE", default="") or "").strip()
        host_file = str(
            self._cfg_value("host_structure_file", "KMC_HOST_STRUCTURE_FILE", "HOST_STRUCTURE_FILE", default="") or ""
        ).strip()
        if source == "generated" and (site_map or host_file):
            source = "external"
        if source not in {"generated", "external"}:
            raise ValueError(f"Unsupported KMC_SITE_SOURCE={raw!r}; expected 'generated' or 'external'.")
        return source

    def _external_map_paths(self) -> tuple[Path, Path]:
        site_map = str(self._cfg_value("site_map_file", "KMC_SITE_MAP_FILE", "SITE_MAP_FILE", default="") or "").strip()
        host_file = str(
            self._cfg_value("host_structure_file", "KMC_HOST_STRUCTURE_FILE", "HOST_STRUCTURE_FILE", default="") or ""
        ).strip()
        if not site_map or not host_file:
            raise ValueError(
                "External KMC map mode requires KMC_SITE_MAP_FILE and KMC_HOST_STRUCTURE_FILE "
                "(or SITE_MAP_FILE and HOST_STRUCTURE_FILE)."
            )
        return Path(site_map).expanduser(), Path(host_file).expanduser()

    def _make_supercell_spec(self) -> SupercellSpec:
        return SupercellSpec(
            lattice_a=float(self._cfg_value("lattice_a", "a", "LATTICE_A", default=2.8601)),
            nx=int(self._cfg_value("nx", "NX", default=30)),
            ny=int(self._cfg_value("ny", "NY", default=30)),
            nz=int(self._cfg_value("nz", "NZ", default=30)),
        )

    def _make_kinetics(self) -> KineticParameters:
        return KineticParameters(
            temperature_k=float(self._cfg_value("temperature_k", "T", default=300.0)),
            attempt_frequency_hz=float(self._cfg_value("attempt_frequency_hz", "nu", default=1.0e13)),
        )

    def _make_cache_manager(self) -> BarrierCacheManager:
        default_schema = self._default_cache_schema()
        cache_cfg = CacheManagerConfig(
            schema=str(self._cfg_value("cache_schema", "schema", "CACHE_SCHEMA", default=default_schema) or default_schema),
            directory=str(self._cache_output_dir()),
            merge_mode=str(self._cfg_value("barrier_merge_mode", "merge_mode", "BARRIER_MERGE_MODE", default="global")).lower(),
            merge_interval=int(self._cfg_value("barrier_merge_interval", "merge_interval", "BARRIER_MERGE_INTERVAL", default=100)),
            enabled=True,
            preload_file=str(
                self._cfg_value(
                    "preload_file",
                    "BARRIER_CACHE_FILE",
                    "KMC_BARRIER_CACHE_FILE",
                    "MASTER_BARRIER_CACHE_FILE",
                    default="",
                )
                or ""
            ),
        )
        return BarrierCacheManager.from_config(
            cache_cfg,
            self.ctx.rank,
            logger=self.log,
        )

    def _default_cache_schema(self) -> str:
        mode = str(self._cfg_value("local_env_mode", "mode", "LOCAL_ENV_MODE", default="radial")).lower()
        if self._site_source() == "external":
            mode = "shell"
        if mode == "shell":
            inner = float(self._cfg_value("shell_inner_radius_a", "SHELL_INNER_RADIUS_A", default=6.0))
            outer = float(self._cfg_value("shell_outer_radius_a", "SHELL_OUTER_RADIUS_A", default=8.0))
            # Keep the historical schema prefix so existing cache files remain
            # readable across the refactor-to-production transition.
            schema = f"scaffold_v4_shell_IR{inner}_OR{outer}"
            if self._site_source() == "external":
                site_map = str(self._cfg_value("site_map_file", "KMC_SITE_MAP_FILE", "SITE_MAP_FILE", default="") or "")
                map_name = Path(site_map).expanduser().stem or "external_map"
                schema = f"{schema}_map_{map_name}"
            return schema
        if mode == "wrapped":
            return "scaffold_v4_wrapped"
        return "scaffold_v4_radial_pbc"

    def _make_scheduler(self) -> MPIJobScheduler:
        sched_cfg = SchedulerConfig(
            assignment_mode=str(self._cfg_value("job_assignment_mode", "JOB_ASSIGNMENT_MODE", default="cache_dedupe")).lower(),
            merge_mode=str(self._cfg_value("barrier_merge_mode", "merge_mode", "BARRIER_MERGE_MODE", default="global")).lower(),
            min_batch=int(self._cfg_value("neb_min_batch", "min_batch", "NEB_MIN_BATCH", default=64)),
        )
        return MPIJobScheduler(
            self.ctx,
            sched_cfg,
            logger=self.log,
            debug=bool(self._cfg_value("debug_logging", "DEBUG_LOGGING", "DEBUG_MODE", default=True)),
        )

    def _make_lammps_factory(self) -> LammpsCalculatorFactory:
        package_dir = Path(__file__).resolve().parent
        lmp_cfg = LammpsCalculatorConfig(
            potential=str(self._cfg_value("potential", "POTENTIAL", default="EAM")),
            potential_eam_file=str(self._cfg_value("potential_eam_file", "POTENTIAL_EAM_FILE", default="PotentialB3410-modified.fs")),
            potential_nnp_dir=str(self._cfg_value("potential_nnp_dir", "POTENTIAL_NNP_DIR", default="")),
            lammps_engine=str(self._cfg_value("lammps_engine", "engine", "LAMMPS_ENGINE", default="eam_fs")),
            lammps_pair_style=str(self._cfg_value("lammps_pair_style", "pair_style", "LAMMPS_PAIR_STYLE", default="")),
            lammps_pair_coeff=str(self._cfg_value("lammps_pair_coeff", "pair_coeff", "LAMMPS_PAIR_COEFF", default="")),
            lammps_files=self._cfg_value("lammps_files", "files", "files_raw", "LAMMPS_FILES", default="PotentialB3410-modified.fs"),
            nnp_dir=str(self._cfg_value("nnp_dir", "NNP_DIR", default="")),
            nnp_elements=str(self._cfg_value("nnp_elements", "NNP_ELEMENTS", default="Fe H")),
            nnp_cutoff=float(self._cfg_value("nnp_cutoff", "NNP_CUTOFF", default=6.60)),
            nnp_showewsum=int(self._cfg_value("nnp_showewsum", "NNP_SHOWEWSUM", default=1000)),
            nnp_maxew=int(self._cfg_value("nnp_maxew", "NNP_MAXEW", default=1000000)),
            nnp_cflength=float(self._cfg_value("nnp_cflength", "NNP_CFLENGTH", default=1.8897261328)),
            nnp_cfenergy=float(self._cfg_value("nnp_cfenergy", "NNP_CFENERGY", default=0.0367493254)),
        )
        return LammpsCalculatorFactory(
            lmp_cfg,
            base_dir=str(package_dir),
            logger=lambda msg: self.log(f"[Rank {self.ctx.rank}] {msg}"),
        )

    def _ensure_neb_engine(self):
        if self.neb_engine is not None:
            return self.neb_engine
        if LammpsNEB is None:
            raise RuntimeError("ASE-based LammpsNEB is unavailable; install ASE or run with NEB_ENGINE=lammps_only")

        nimg = max(3, int(self._cfg_value("nimg", "n_images", "NIMG", default=3)))
        self.log(f"[Rank {self.ctx.rank}] Initializing {nimg} LAMMPS calculator(s) for NEB")
        self.calc_pool = self.lammps_factory.create_pool(nimg)

        helpers = {
            "assign_mass_and_type": assign_mass_and_type,
            "log": self.log,
            "make_calculator": self.lammps_factory.create,
            "run_opt": self.run_endpoint_opt,
            "optimize_endpoints": bool(self._cfg_value("optimize_endpoints", "OPTIMIZE_ENDPOINTS", default=True)),
            "optimizer_kwargs": {
                "fmax": float(self._cfg_value("optimizer_fmax", "NEB_OPTIMIZER_FMAX", default=0.05)),
                "steps": int(self._cfg_value("optimizer_steps", "NEB_OPTIMIZER_STEPS", default=300)),
            },
        }
        self.neb_engine = LammpsNEB(self.calc_pool, helpers=helpers)
        self.log(f"[Rank {self.ctx.rank}] NEB engine ready with {nimg} image calculator(s)")
        return self.neb_engine

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        rank = self.ctx.rank
        site_source = self._site_source()

        if rank == 0:
            t0 = time.time()
            if site_source == "external":
                site_map_path, host_path = self._external_map_paths()
                sites, site_types, site_box = load_kmc_site_map(site_map_path)
                host_atoms, host_types, host_box = load_lammps_atomic_structure(
                    host_path,
                    fe_type=int(self._cfg_value("host_fe_type", "KMC_HOST_FE_TYPE", default=1)),
                )
                box = np.asarray(site_box if site_box is not None else host_box, dtype=float)
                if not np.allclose(box, host_box, atol=1.0e-6):
                    raise RuntimeError(
                        f"Site-map box {box.tolist()} does not match host box {host_box.tolist()}"
                    )
                self.spec = SupercellSpec(
                    lattice_a=float(self._cfg_value("lattice_a", "a", "LATTICE_A", default=2.8601)),
                    nx=int(self._cfg_value("nx", "NX", default=30)),
                    ny=int(self._cfg_value("ny", "NY", default=30)),
                    nz=int(self._cfg_value("nz", "NZ", default=30)),
                    box_override=tuple(float(v) for v in box),
                )
                self.sites = np.mod(np.asarray(sites, dtype=float), self.spec.box)
                self.site_types = site_types
                self.external_host_structure = host_atoms
                host_numbers = np.asarray(host_atoms.get_atomic_numbers(), dtype=int)
                self.external_host_fe_positions = np.mod(
                    np.asarray(host_atoms.get_positions(), dtype=float)[host_numbers == 26],
                    self.spec.box,
                )
                from scipy.spatial import cKDTree

                self._external_host_fe_tree = cKDTree(
                    self.external_host_fe_positions,
                    boxsize=np.asarray(self.spec.box, dtype=float),
                )
                self.log(
                    f"[Rank 0] Loaded external KMC map: {len(self.sites)} sites from {site_map_path}, "
                    f"{len(self.external_host_fe_positions)} Fe host atoms from {host_path} "
                    f"in {time.time() - t0:.2f} s"
                )
                del host_types
            else:
                site_pack = generate_all_interstitial_sites(
                    self.spec.nx,
                    self.spec.ny,
                    self.spec.nz,
                    self.spec.lattice_a,
                )
                self.sites = site_pack["tetra"]
                self.site_types = None
                self.external_host_structure = None
                self.external_host_fe_positions = None
                self._external_host_fe_tree = None
                self.log(
                    f"[Rank 0] Generated {len(self.sites)} tetrahedral sites "
                    f"in {time.time() - t0:.2f} s"
                )

            t0 = time.time()
            knn_k = int(self._cfg_value("knn_k", "KNN_K", default=6))
            same_type = bool(self._cfg_value("same_type_neighbors", "SAME_TYPE", default=False))
            self.neighbors = get_k_nearest_neighbors(
                self.sites,
                self.spec.box,
                k=knn_k,
                same_type=same_type,
                per_cell=12,
            )
            if self._devanathan_enabled():
                self.neighbors = filter_fixed_x_neighbors(self.neighbors, self.sites, self.spec.box)
                self.log("[Rank 0] Devanathan mode: filtered x-wrap neighbor links for fixed x boundaries")
                hop_x_direction = str(
                    self._cfg_value(
                        "DEVANATHAN_HOP_X_DIRECTION",
                        "KMC_HOP_X_DIRECTION",
                        default="all",
                    )
                    or "all"
                ).strip().lower()
                if bool(
                    self._cfg_value(
                        "DEVANATHAN_RIGHT_DIRECTION_ONLY",
                        "KMC_ONLY_RIGHT_DIRECTION_HOPS",
                        default=False,
                    )
                ):
                    hop_x_direction = "right"
                if hop_x_direction not in {"", "all", "both", "bidirectional", "none"}:
                    before_links = sum(len(row) for row in self.neighbors)
                    self.neighbors = filter_x_direction_neighbors(
                        self.neighbors,
                        self.sites,
                        hop_x_direction,
                    )
                    after_links = sum(len(row) for row in self.neighbors)
                    self.log(
                        f"[Rank 0] Devanathan mode: kept {after_links}/{before_links} "
                        f"KNN links for KMC_HOP_X_DIRECTION={hop_x_direction!r}"
                    )
            self.log(
                f"[Rank 0] Built KNN neighbors with K={knn_k}, same_type={same_type} "
                f"in {time.time() - t0:.2f} s"
            )

            num_h = int(self._cfg_value("num_h", "NUM_H", default=5))
            initial_h_regions = self._initial_h_region_specs()
            self.devanathan_boundary = self._make_devanathan_boundary()
            if self.devanathan_boundary is not None and not self._restart_enabled():
                fe_positions = self._native_fe_positions()
                self.h_indices = self.devanathan_boundary.initialize_source(self.sites, fe_positions)
                source_fe_atoms = self.devanathan_boundary.source_fe_count(fe_positions)
                source_target = self.devanathan_boundary.target_source_count(self.sites, fe_positions)
                self._write_gcmc_initialization_diag(source_fe_atoms)
                if self.gcmc_energy_cache is not None:
                    self.gcmc_energy_cache.save()
                initialization_summary = getattr(
                    self.devanathan_boundary,
                    "initialization_summary",
                    None,
                )
                region_label = str(
                    getattr(self.devanathan_boundary, "region_label", "source")
                )
                self.log(
                    f"[Rank 0] Devanathan {region_label} initialized with "
                    f"{len(self.h_indices)} H from {source_fe_atoms} Fe atoms "
                    f"in x=[{self.devanathan_boundary.config.source_x_min_a:g}, "
                    f"{self.devanathan_boundary.config.source_x_max_a:g}) A "
                    f"({self.devanathan_boundary.config.source_target_fraction:g} fraction; "
                    f"target={source_target}, available_interstitials={len(self.devanathan_boundary.source_sites(self.sites))})."
                )
                if initialization_summary is not None:
                    initialization_status = (
                        "converged"
                        if initialization_summary.converged
                        else "accepted near target at maximum attempts"
                    )
                    self.log(
                        f"[Rank 0] GCMC initialization {initialization_status}: "
                        f"attempts={initialization_summary.attempts}, "
                        f"accepted={initialization_summary.accepted_moves}, "
                        f"window_mean_H={initialization_summary.mean_h:.6g}, "
                        f"window_H_per_Fe={initialization_summary.mean_fraction_h_per_fe:.6g}, "
                        f"half_drift_H={initialization_summary.half_drift_h:.6g}, "
                        f"maintenance_setpoint={initialization_summary.maintenance_setpoint_h}"
                    )
                if self.gcmc_energy_cache is not None:
                    self.log(
                        f"[Rank 0] GCMC {region_label} energy cache after initialization: "
                        f"entries={len(self.gcmc_energy_cache)}, "
                        f"hits={self.gcmc_energy_cache.stats.hits}, "
                        f"misses={self.gcmc_energy_cache.stats.misses}"
                    )
            elif initial_h_regions:
                self.h_indices = self._select_h_indices_from_regions(self.sites, num_h, initial_h_regions)
                self.log(
                    f"[Rank 0] Initial H region selector {initial_h_regions!r} "
                    f"chose {len(self.h_indices)} sites from {self._initial_h_region_file()}."
                )
            elif self._build_only_enabled() and str(
                self._cfg_value("build_selector", "BUILD_LOCAL_ENV_SELECTOR", default="random")
            ).lower().strip() != "random":
                self.h_indices = self._select_h_indices_for_build_only(self.sites, num_h)
                self.log(
                    f"[Rank 0] Build-only H selector "
                    f"'{self._cfg_value('build_selector', 'BUILD_LOCAL_ENV_SELECTOR', default='random')}' "
                    f"chose {len(self.h_indices)} sites."
                )
            else:
                self.h_indices = np.random.choice(len(self.sites), size=num_h, replace=False)

            if self._restart_enabled():
                self._apply_restart_state()
        else:
            self.sites = None
            self.site_types = None
            self.neighbors = None
            self.h_indices = None
            self.h_unwrapped_positions = None
            self.external_host_structure = None
            self.external_host_fe_positions = None
            self._external_host_fe_tree = None

        self.spec = self.ctx.comm.bcast(self.spec, root=0)
        self.sites = self.ctx.comm.bcast(self.sites, root=0)
        self.site_types = self.ctx.comm.bcast(self.site_types, root=0)
        self.neighbors = self.ctx.comm.bcast(self.neighbors, root=0)
        self.h_indices = self.ctx.comm.bcast(self.h_indices, root=0)
        self.h_unwrapped_positions = self.ctx.comm.bcast(self.h_unwrapped_positions, root=0)
        self.sim_time_s = float(self.ctx.comm.bcast(self.sim_time_s if rank == 0 else None, root=0))
        self.restart_start_step = int(self.ctx.comm.bcast(self.restart_start_step if rank == 0 else 0, root=0))
        self.external_host_structure = self.ctx.comm.bcast(self.external_host_structure, root=0)
        self.external_host_fe_positions = self.ctx.comm.bcast(self.external_host_fe_positions, root=0)
        if rank != 0:
            self.devanathan_boundary = self._make_devanathan_boundary()
        self._restore_devanathan_state()

        if self._lammps_only_enabled():
            self.full_structure = None
            self.fe_3a_template = None
        elif self.external_host_structure is not None:
            self.full_structure = build_full_structure_from_host(
                self.external_host_structure,
                self.sites,
                self.h_indices,
            )
            self.fe_3a_template = None
        else:
            self.full_structure = build_full_structure(self.sites, self.h_indices, self.spec)
            self.fe_3a_template = build_fe_3a_template(self.spec.lattice_a)
        h_positions = np.asarray(self.sites[self.h_indices], dtype=float)
        if self.h_unwrapped_positions is None:
            self.h_unwrapped_positions = h_positions.copy()
        self.h_initial_unwrapped_positions = np.asarray(self.h_unwrapped_positions, dtype=float).copy()

        if rank == 0:
            self._init_kmc_diag()
            if self.devanathan_boundary is not None:
                self.msd_tracker = None
                self.log("[Rank 0] Devanathan mode: disabled MSD tracker because H count can change.")
            else:
                self.msd_tracker = MSDTracker(
                    initial_indices=self.h_indices,
                    box=self.spec.box,
                    initial_positions=self.h_initial_unwrapped_positions,
                    use_unwrapped_positions=True,
                    log_interval=int(self._cfg_value("msd_log_interval", "MSD_LOG_INTERVAL", default=1000)),
                    log_file=str(self._cfg_value("msd_log_file", "MSD_LOG_FILE", default="msd_vs_time.csv")),
                    plot_file=str(self._cfg_value("msd_plot_file", "MSD_PLOT_FILE", default="msd_plot.png")),
                    enabled=True,
                )
                self.msd_tracker.record(
                    self.restart_start_step,
                    self.sim_time_s,
                    current_positions=self.h_unwrapped_positions,
                    force=True,
                )

        self._refresh_h_lookup()
        self._preload_restart_cache()

        self.log(f"[Rank {rank}] Initialization complete; local H count={len(self.local_h_sites())}")

    def _refresh_h_lookup(self) -> None:
        assert self.sites is not None
        assert self.h_indices is not None
        h_positions = np.asarray(self.sites[self.h_indices], dtype=float)
        self.h_cell_index = build_h_cell_index(h_positions, self.h_indices, self.spec)
        self._env_key_h_positions = None
        self._env_key_h_tree = None
        self._env_key_h_signature = None

    def _prepare_env_key_cache(self) -> None:
        assert self.sites is not None
        assert self.h_indices is not None
        signature = tuple(int(x) for x in self.h_indices)
        if self._env_key_h_signature == signature and self._env_key_h_tree is not None:
            return
        h_positions = np.asarray(self.sites[self.h_indices], dtype=float)
        self._env_key_h_positions = h_positions
        self._env_key_h_tree = build_hydrogen_kdtree(h_positions, box=self.spec.box)
        self._env_key_h_signature = signature

    def local_h_sites(self) -> np.ndarray:
        assert self.h_indices is not None
        return np.asarray(self.h_indices[self.ctx.rank::self.ctx.size], dtype=int)

    # ------------------------------------------------------------------
    # Environment key adapter
    # ------------------------------------------------------------------
    def make_event_key(self, h_site: int, n_site: int) -> Any:
        assert self.sites is not None
        assert self.h_indices is not None

        self._prepare_env_key_cache()
        assert self._env_key_h_positions is not None
        radius = float(self._cfg_value("env_radius_a", "radius_a", "ENV_RADIUS_A", default=5.0))
        pos_bin_a = float(self._cfg_value("pos_bin_a", "POS_BIN_A", default=0.10))
        hop_bin_a = float(self._cfg_value("hop_bin_a", "HOP_BIN_A", default=0.02))
        env_key_mode = str(self._cfg_value("env_key_mode", "ENV_KEY_MODE", default="env_plus_dir"))
        return make_env_key(
            self.sites[int(h_site)],
            self.sites[int(n_site)],
            self._env_key_h_positions,
            h_tree=self._env_key_h_tree,
            box=self.spec.box,
            radius_A=radius,
            pos_bin_A=pos_bin_a,
            hop_bin_A=hop_bin_a,
            mode=env_key_mode,
        )

    def radial_neighbor_positions(self, h_site: int) -> list[tuple[np.ndarray, int]]:
        assert self.sites is not None
        assert self.h_indices is not None

        rel_initial = np.asarray(self.sites[int(h_site)], dtype=float)
        h_positions = np.asarray(self.sites[self.h_indices], dtype=float)
        radius = float(self._cfg_value("env_radius_a", "radius_a", "ENV_RADIUS_A", default=5.0))
        neighbors: list[tuple[np.ndarray, int]] = []

        for site_idx, pos in zip(self.h_indices, h_positions):
            diff = pbc_diff(rel_initial, pos, self.spec.box)
            if np.allclose(diff, 0.0, atol=1e-6):
                continue
            if float(np.linalg.norm(diff)) <= radius:
                neighbors.append((rel_initial + diff, int(site_idx)))
        return neighbors

    def _build_local_neb_endpoints(
        self,
        h_site: int,
        n_site: int,
        requested_mode: str,
    ) -> tuple[Atoms, Atoms, int, str]:
        assert self.sites is not None
        assert self.h_indices is not None

        rel_initial = np.asarray(self.sites[int(h_site)], dtype=float)
        rel_final = np.asarray(self.sites[int(n_site)], dtype=float)
        mode = requested_mode.lower().strip()

        if self.external_host_fe_positions is not None:
            h_positions = np.asarray(self.sites[self.h_indices], dtype=float)
            initial, final, mover_idx = build_local_neb_structures_shell_from_host(
                rel_initial,
                rel_final,
                self.external_host_fe_positions,
                self.spec.box,
                h_positions=h_positions,
                h_indices=self.h_indices,
                inner_radius_a=float(self._cfg_value("shell_inner_radius_a", "SHELL_INNER_RADIUS_A", default=6.0)),
                outer_radius_a=float(self._cfg_value("shell_outer_radius_a", "SHELL_OUTER_RADIUS_A", default=8.0)),
                shrinkwrap_pad_a=float(self._cfg_value("shrinkwrap_pad_a", default=2.0)),
            )
            return initial, final, mover_idx, "external_shell"

        if mode == "shell":
            initial, final, mover_idx = build_local_neb_structures_shell(
                rel_initial,
                rel_final,
                self.spec,
                h_cell_index=self.h_cell_index,
                inner_radius_a=float(self._cfg_value("shell_inner_radius_a", "SHELL_INNER_RADIUS_A", default=6.0)),
                outer_radius_a=float(self._cfg_value("shell_outer_radius_a", "SHELL_OUTER_RADIUS_A", default=8.0)),
                shrinkwrap_pad_a=float(self._cfg_value("shrinkwrap_pad_a", default=2.0)),
            )
            return initial, final, mover_idx, "shell"

        assert self.fe_3a_template is not None
        neighbor_positions = self.radial_neighbor_positions(int(h_site)) if mode == "radial" else None
        initial, final, mover_idx = build_local_neb_structures_fast(
            rel_initial,
            rel_final,
            self.spec,
            self.fe_3a_template,
            mode=mode,
            neighbor_positions=neighbor_positions,
            h_cell_index=self.h_cell_index,
            shrinkwrap_pad_a=float(self._cfg_value("shrinkwrap_pad_a", default=2.0)),
        )
        return initial, final, mover_idx, mode

    # ------------------------------------------------------------------
    # NEB execution
    # ------------------------------------------------------------------
    def run_neb_jobs(self, my_jobs: Sequence[tuple[Any, tuple[int, int, int]]], step: int):
        results_local = []
        diag_summary = NEBDiagnosticSummary(step=int(step), rank=int(self.ctx.rank), neb_assigned=len(my_jobs))
        if not my_jobs:
            self._write_rank_neb_diag(diag_summary)
            return results_local, diag_summary

        assert self.sites is not None

        try:
            neb_engine = self._ensure_neb_engine()
        except Exception as exc:
            self.log(f"[Rank {self.ctx.rank}] ERROR initializing NEB engine: {exc}")
            results_local = [(key, int(src_rank), float("inf")) for key, (src_rank, _h, _n) in my_jobs]
            self._write_rank_neb_diag(diag_summary)
            return results_local, diag_summary

        local_mode = str(self._cfg_value("local_env_mode", "mode", "LOCAL_ENV_MODE", default="radial")).lower()
        if local_mode not in {"wrapped", "radial", "shell"}:
            self.log(f"[Rank {self.ctx.rank}] Local mode '{local_mode}' not wired; using radial.")
            local_mode = "radial"
        if self.external_host_fe_positions is not None and local_mode != "shell":
            self.log(
                f"[Rank {self.ctx.rank}] External host map uses explicit Fe shell environments; "
                f"ignoring LOCAL_ENV_MODE='{local_mode}'."
            )

        chunks = chunk_jobs(my_jobs, int(self._cfg_value("neb_min_batch", "min_batch", default=64)))
        diag_summary.neb_batches = len(chunks)
        wall_start = time.time()
        atom_counts: list[int] = []
        fe_counts: list[int] = []
        h_counts: list[int] = []
        frozen_counts: list[int] = []

        for chunk in chunks:
            self.log(f"[Rank {self.ctx.rank}] Running NEB chunk with {len(chunk)} job(s)")
            for job_idx, (key, (src_rank, h_site, n_site)) in enumerate(chunk, start=1):
                barrier_ev = float("inf")
                try:
                    struct_t0 = time.time()
                    initial, final, mover_idx, actual_mode = self._build_local_neb_endpoints(
                        int(h_site),
                        int(n_site),
                        local_mode,
                    )
                    diag_summary.neb_struct_s += time.time() - struct_t0

                    job_context = {
                        "step": int(step),
                        "h": int(h_site),
                        "n": int(n_site),
                        "src_rank": int(src_rank),
                        "key": str(key),
                    }
                    self.dump_neb_structure_snapshot(initial, final, mover_idx, job_context, step)
                    atom_counts.append(len(initial))
                    fe_counts.append(sum(1 for z in initial.get_atomic_numbers() if int(z) == 26))
                    h_counts.append(sum(1 for z in initial.get_atomic_numbers() if int(z) == 1))
                    frozen_counts.append(int(np.sum(initial.arrays.get("freeze_mask", np.zeros(len(initial), dtype=bool)))))

                    timing: dict[str, float] = {}
                    collect_validation_diag = self._validation_enabled() and self._validation_ase_dump()
                    neb_result = neb_engine.barrier(
                        initial,
                        final,
                        rank=self.ctx.rank,
                        return_images=False,
                        collect_diagnostics=collect_validation_diag,
                        mover_idx=mover_idx,
                        auto_detect_mover=False,
                        neb_dump_mode=False,
                        expected_h_count=sum(1 for z in initial.get_atomic_numbers() if z == 1),
                        job_context=job_context,
                        timing_out=timing,
                    )
                    diag = None
                    if collect_validation_diag and isinstance(neb_result, tuple) and len(neb_result) == 2:
                        barrier_ev, diag = neb_result
                        if isinstance(diag, dict) and "timings" in diag:
                            timing.update(diag.get("timings") or {})
                    else:
                        barrier_ev = neb_result

                    if barrier_ev is None or not np.isfinite(barrier_ev):
                        barrier_ev = float("inf")

                    self.dump_validation_neb(
                        h_site=int(h_site),
                        n_site=int(n_site),
                        step=int(step),
                        rank=int(src_rank),
                        initial=initial,
                        final=final,
                        barrier_ev=float(barrier_ev),
                        diag=diag if isinstance(diag, dict) else None,
                        timing=timing,
                        local_mode=actual_mode,
                    )
                    diag_summary.neb_ran += 1
                    diag_summary.neb_engine_total_s += float(timing.get("total_s", 0.0))
                    diag_summary.neb_header_s += float(timing.get("header_s", 0.0))
                    diag_summary.neb_endpoint_constraints_s += float(timing.get("endpoint_constraints_s", 0.0))
                    diag_summary.neb_endpoint_opt_s += float(timing.get("endpoint_opt_s", 0.0))
                    diag_summary.neb_image_setup_s += float(timing.get("image_setup_s", 0.0))
                    diag_summary.neb_image_constraints_s += float(timing.get("image_constraints_s", 0.0))
                    diag_summary.neb_interpolate_s += float(timing.get("interpolate_s", 0.0))
                    diag_summary.neb_optimize_s += float(timing.get("optimizer_s", 0.0))
                    diag_summary.neb_energy_eval_s += float(timing.get("energy_eval_s", 0.0))
                    diag_summary.neb_failure_cleanup_s += float(timing.get("failure_cleanup_s", 0.0))

                    self.log(
                        f"[Rank {self.ctx.rank}] NEB job {job_idx}/{len(chunk)} "
                        f"h={h_site}->n={n_site} barrier={barrier_ev:.6f} eV "
                        f"wall={timing.get('total_s', 0.0):.3f}s"
                    )

                except Exception as exc:
                    self.log(f"[Rank {self.ctx.rank}] NEB job failed for h={h_site}->n={n_site}: {exc}")
                    barrier_ev = float("inf")

                results_local.append((key, int(src_rank), float(barrier_ev)))

        diag_summary.neb_wall_s = time.time() - wall_start
        if diag_summary.neb_ran:
            diag_summary.neb_avg_s = diag_summary.neb_wall_s / diag_summary.neb_ran
            diag_summary.neb_engine_avg_s = diag_summary.neb_engine_total_s / diag_summary.neb_ran
        diag_summary.neb_setup_s = diag_summary.neb_struct_s
        diag_summary.neb_preprocess_s = diag_summary.neb_struct_s + diag_summary.neb_setup_s
        if diag_summary.neb_ran:
            diag_summary.neb_preprocess_avg_s = diag_summary.neb_preprocess_s / diag_summary.neb_ran
        diag_summary.neb_total_with_preprocess_s = diag_summary.neb_engine_total_s + diag_summary.neb_preprocess_s
        if diag_summary.neb_ran:
            diag_summary.neb_total_with_preprocess_avg_s = diag_summary.neb_total_with_preprocess_s / diag_summary.neb_ran
            diag_summary.neb_avg_atoms = float(np.mean(atom_counts)) if atom_counts else 0.0
            diag_summary.neb_avg_fe = float(np.mean(fe_counts)) if fe_counts else 0.0
            diag_summary.neb_avg_h = float(np.mean(h_counts)) if h_counts else 0.0
            diag_summary.neb_avg_frozen = float(np.mean(frozen_counts)) if frozen_counts else 0.0
        self._write_rank_neb_diag(diag_summary)
        return results_local, diag_summary

    def _native_fe_positions(self) -> np.ndarray:
        if self.external_host_fe_positions is not None:
            return np.asarray(self.external_host_fe_positions, dtype=float)
        return bcc_fe_positions(
            self.spec.nx,
            self.spec.ny,
            self.spec.nz,
            self.spec.lattice_a,
        )

    def _build_native_batches(
        self,
        miss_jobs: Sequence[tuple[Any, tuple[int, int, int]]],
    ) -> list[EnvironmentNEBBatch]:
        assert self.sites is not None
        assert self.neighbors is not None

        by_h: dict[int, list[tuple[Any, int]]] = {}
        for key, (_src_rank, h_site, n_site) in miss_jobs:
            by_h.setdefault(int(h_site), []).append((key, int(n_site)))

        fe_positions = self._native_fe_positions()
        batches: list[EnvironmentNEBBatch] = []
        for h_site in sorted(by_h):
            key_and_neighbors = by_h[h_site]
            neighbor_indices = [n_site for _key, n_site in key_and_neighbors]
            env_key = (
                "lammps_only_env_batch",
                self.make_event_key(h_site, neighbor_indices[0]) if neighbor_indices else h_site,
                tuple(sorted(int(n) for n in neighbor_indices)),
            )
            batch = build_batch_from_sites(
                env_key=env_key,
                h_site=h_site,
                sites=np.asarray(self.sites, dtype=float),
                neighbor_indices=neighbor_indices,
                fe_positions=fe_positions,
                box=np.asarray(self.spec.box, dtype=float),
                lattice_a=float(self.spec.lattice_a),
            )
            patched_hops = tuple(
                replace(hop, hop_key=key, slot=slot)
                for slot, (hop, (key, _n_site)) in enumerate(zip(batch.hops, key_and_neighbors))
            )
            batches.append(replace(batch, hops=patched_hops))
        return batches

    def _build_native_batches_for_group(
        self,
        miss_moves: Sequence[tuple[int, int]],
        *,
        group_id: int,
        n_groups: int,
    ) -> tuple[list[EnvironmentNEBBatch], int]:
        """Build only this NEB group's environment batches from compact moves."""

        assert self.sites is not None
        assert self.neighbors is not None

        by_h: dict[int, list[int]] = {}
        for h_site, n_site in miss_moves:
            by_h.setdefault(int(h_site), []).append(int(n_site))

        fe_positions = self._native_fe_positions()
        batches: list[EnvironmentNEBBatch] = []
        total_batches = 0
        for env_idx, h_site in enumerate(sorted(by_h)):
            neighbor_indices = by_h[h_site]
            if env_idx % int(n_groups) != int(group_id):
                total_batches += 1
                continue
            env_key = (
                "lammps_only_env_batch",
                self.make_event_key(h_site, neighbor_indices[0]) if neighbor_indices else h_site,
                tuple(sorted(int(n) for n in neighbor_indices)),
            )
            batch = build_batch_from_sites(
                env_key=env_key,
                h_site=h_site,
                sites=np.asarray(self.sites, dtype=float),
                neighbor_indices=neighbor_indices,
                fe_positions=fe_positions,
                box=np.asarray(self.spec.box, dtype=float),
                lattice_a=float(self.spec.lattice_a),
            )
            patched_hops = tuple(
                replace(hop, hop_key=self.make_event_key(h_site, n_site), slot=slot)
                for slot, (hop, n_site) in enumerate(zip(batch.hops, neighbor_indices))
            )
            batches.append(replace(batch, hops=patched_hops))
            total_batches += 1
        return batches, total_batches

    def _run_native_assigned_batches(
        self,
        batches: Sequence[EnvironmentNEBBatch],
        *,
        step: int,
        cfg: LammpsOnlyNEBConfig,
        diag_summary: NEBDiagnosticSummary,
    ):
        assert self.native_neb_group_comm is not None
        if self.native_lammps_class is None:
            self.native_lammps_class = load_native_lammps_class(cfg)

        group_rank = self.native_neb_group_comm.Get_rank()
        local_results = []
        for batch_idx, batch in enumerate(batches):
            diag_summary.neb_batches += 1
            group_work_dir = (
                cfg.work_dir
                / f"group_{int(self.native_neb_group_id):04d}"
                / f"batch_{int(batch_idx):08d}_h{int(batch.h_site)}"
            )
            results = run_native_batch_slots(
                batch,
                list(range(len(batch.hops))),
                group_comm=self.native_neb_group_comm,
                group_work_dir=group_work_dir,
                cfg=cfg,
                lammps_class=self.native_lammps_class,
            )
            if group_rank == 0:
                local_results.extend(results)
                diag_summary.neb_ran += len(results)
                diag_summary.neb_io_write_s += sum(float(r.io_write_s) for r in results)
                diag_summary.neb_endpoint_opt_s += sum(float(r.endpoint_opt_s) for r in results)
                diag_summary.neb_lammps_run_s += sum(float(r.lammps_run_s) for r in results)
                diag_summary.neb_io_parse_s += sum(float(r.io_parse_s) for r in results)
                diag_summary.neb_io_cleanup_s += sum(float(r.io_cleanup_s) for r in results)
                diag_summary.neb_engine_total_s += sum(float(r.endpoint_opt_s) + float(r.lammps_run_s) for r in results)
        return local_results if group_rank == 0 else []

    def step_once_lammps_only(self, step: int) -> bool:
        assert self.sites is not None
        assert self.neighbors is not None
        assert self.h_indices is not None

        step_wall_t0 = time.time()
        timer = StepTimer()
        rank = self.ctx.rank
        selected = None
        moved_slot = None

        occupied = set(map(int, self.h_indices))
        event_cfg = EventManagerConfig(
            env_radius_a=float(self._cfg_value("env_radius_a", "radius_a", default=5.0)),
            use_cache_for_jobs=self.scheduler.cfg.use_cache_for_jobs,
            cache_only=(self.scheduler.cfg.assignment_mode == "cache_only"),
            low_barrier_ev=float(self._cfg_value("low_barrier_ev", default=1.0e-3)),
            rank=rank,
        )

        incremental_enabled = self._incremental_events_enabled()
        if rank == 0:
            incremental_active = bool(
                incremental_enabled
                and self._incremental_event_table is not None
                and self._incremental_rebuild_h_sites is not None
                and self._incremental_remove_h_sites is not None
            )
        else:
            incremental_active = None
        incremental_active = bool(self.ctx.comm.bcast(incremental_active, root=0))

        cache_events = None
        miss_moves = None
        if incremental_active:
            t0 = time.time()
            if rank == 0:
                assert self._incremental_rebuild_h_sites is not None
                self._prepare_env_key_cache()
                cache_events = build_candidate_events(
                    step=step,
                    rank=0,
                    local_h_sites=self._incremental_rebuild_h_sites,
                    neighbors=self.neighbors,
                    occupied_sites=occupied,
                    cache=self.cache_mgr.cache,
                    make_key=self.make_event_key,
                    kinetics=self.kinetics,
                    cfg=event_cfg,
                    failed_keys=self.failed_barrier_keys,
                    reported_failed_keys=self.reported_failed_rate_keys,
                )
                miss_moves = [
                    (int(h_site), int(n_site))
                    for _key, (_src_rank, h_site, n_site) in cache_events.miss_jobs
                ]
                total_batches = len({int(h_site) for h_site, _n_site in miss_moves})
                self.log(
                    f"[Rank 0] Incremental candidate summary: rebuild_H={len(self._incremental_rebuild_h_sites)}, "
                    f"cache_hits={cache_events.cache_hits}, cache_misses={cache_events.cache_misses}, "
                    f"env_batches={total_batches}, missing_hops={len(cache_events.miss_jobs)}, "
                    f"valid_cache_events={len(cache_events.valid_moves)}"
                )
            timer.add("move_loop", time.time() - t0)
        else:
            local_h = self.local_h_sites()
            t0 = time.time()
            self._prepare_env_key_cache()
            local_cache_events = build_candidate_events(
                step=step,
                rank=rank,
                local_h_sites=local_h,
                neighbors=self.neighbors,
                occupied_sites=occupied,
                cache=self.cache_mgr.cache,
                make_key=self.make_event_key,
                kinetics=self.kinetics,
                cfg=event_cfg,
                failed_keys=self.failed_barrier_keys,
                reported_failed_keys=self.reported_failed_rate_keys,
            )
            timer.add("move_loop", time.time() - t0)

            t0 = time.time()
            gathered_cache_events = self.ctx.comm.gather(local_cache_events, root=0)
            timer.add("result_comm", time.time() - t0)

            if rank == 0:
                assert gathered_cache_events is not None
                t0 = time.time()
                cache_events = merge_candidate_results(*gathered_cache_events)
                miss_moves = [(int(h_site), int(n_site)) for _key, (_src_rank, h_site, n_site) in cache_events.miss_jobs]
                total_batches = len({int(h_site) for h_site, _n_site in miss_moves})
                timer.add("job_schedule", time.time() - t0)
                self.log(
                    f"[Rank 0] LAMMPS-only candidate summary: H={len(self.h_indices)}, "
                    f"cache_hits={cache_events.cache_hits}, cache_misses={cache_events.cache_misses}, "
                    f"env_batches={total_batches}, missing_hops={len(cache_events.miss_jobs)}, "
                    f"valid_cache_events={len(cache_events.valid_moves)}"
                )

        t0 = time.time()
        try:
            miss_moves = self.ctx.comm.bcast(miss_moves, root=0)
            batches, _total_batches = self._build_native_batches_for_group(
                miss_moves,
                group_id=int(self.native_neb_group_id),
                n_groups=int(self.native_neb_group_count),
            )
        except BaseException as exc:
            self._write_exception_log("lammps_only_broadcast_or_group_build", exc)
            raise
        timer.add("job_comm", time.time() - t0)
        cfg = self._native_lammps_config(step)
        diag_summary = NEBDiagnosticSummary(
            step=int(step),
            rank=int(rank),
            neb_assigned=sum(len(batch.hops) for batch in batches),
        )

        self.ctx.comm.Barrier()
        t0 = time.time()
        try:
            native_results = self._run_native_assigned_batches(
                batches,
                step=step,
                cfg=cfg,
                diag_summary=diag_summary,
            )
            self.ctx.comm.Barrier()
        except BaseException as exc:
            self._write_exception_log("lammps_only_native_run", exc)
            raise
        diag_summary.neb_wall_s = time.time() - t0
        if diag_summary.neb_ran:
            diag_summary.neb_avg_s = diag_summary.neb_wall_s / diag_summary.neb_ran
            diag_summary.neb_engine_avg_s = diag_summary.neb_engine_total_s / diag_summary.neb_ran
        timer.add("barrier_neb", diag_summary.neb_wall_s)
        timer.add("neb_io_write", diag_summary.neb_io_write_s)
        timer.add("neb_endpoint_opt", diag_summary.neb_endpoint_opt_s)
        timer.add("neb_lammps_run", diag_summary.neb_lammps_run_s)
        timer.add("neb_io_parse", diag_summary.neb_io_parse_s)
        timer.add("neb_io_cleanup", diag_summary.neb_io_cleanup_s)

        t0 = time.time()
        gathered_results = self.ctx.comm.gather(native_results, root=0)
        gathered_neb_diag = self.ctx.comm.gather(diag_summary, root=0)
        timer.add("result_comm", time.time() - t0)

        if rank == 0:
            flat_results = [result for part in gathered_results for result in part]
            fold_results = {result.hop_key: float(result.barrier_eV) for result in flat_results}
            failed_keys = {key for key, value in fold_results.items() if not np.isfinite(float(value))}
        else:
            fold_results = None
            failed_keys = None

        t0 = time.time()
        fold_results = self.ctx.comm.bcast(fold_results, root=0)
        failed_keys = self.ctx.comm.bcast(failed_keys, root=0)
        timer.add("result_comm", time.time() - t0)

        self.failed_barrier_keys.update(failed_keys)
        if self.scheduler.cfg.use_cache_for_jobs:
            t0 = time.time()
            n_cached = self.cache_mgr.update_finite(fold_results)
            timer.add("cache_update", time.time() - t0)
            if rank == 0 and n_cached:
                self.log(f"[Rank 0] Cached {n_cached} finite LAMMPS-only NEB result(s)")

        if rank == 0:
            assert cache_events is not None
            selection_t0 = time.time()
            neb_events = fold_neb_results_into_events(
                step=step,
                rank=None,
                local_miss_jobs=cache_events.miss_jobs,
                fold_results=fold_results,
                kinetics=self.kinetics,
                failed_keys=self.failed_barrier_keys,
                reported_failed_keys=self.reported_failed_rate_keys,
            )
            local_events = merge_candidate_results(cache_events, neb_events)
            if incremental_enabled:
                if incremental_active:
                    assert self._incremental_event_table is not None
                    assert self._incremental_remove_h_sites is not None
                    stats = self._incremental_event_table.update_source_h_sites(
                        self._incremental_remove_h_sites,
                        local_events,
                    )
                    self.log(
                        f"[Rank 0] Incremental table update: removed_events={stats.removed_events}, "
                        f"added_events={stats.added_events}, rebuilt_H={stats.rebuilt_h_sites}, "
                        f"table_events={len(self._incremental_event_table.moves)}"
                    )
                else:
                    self._incremental_event_table = IncrementalEventTable.from_candidate_result(local_events)
                    self.log(
                        f"[Rank 0] Incremental table initialized: "
                        f"table_events={len(self._incremental_event_table.moves)}"
                    )
            timer.add("selection_cpu", time.time() - selection_t0)
            self._write_all_neb_diag(gathered_neb_diag)
            t0 = time.time()
            self._write_rates_allranks([local_events.rate_rows], selected=None)
            timer.add("rate_io", time.time() - t0)

            t0 = time.time()
            if incremental_enabled:
                assert self._incremental_event_table is not None
                selected = self._incremental_event_table.select()
            else:
                ordered_events = sorted(
                    zip(local_events.valid_moves, local_events.rates),
                    key=lambda item: (int(item[0][0]), int(item[0][1])),
                )
                all_moves = [move for move, _rate in ordered_events]
                all_rates = [rate for _move, rate in ordered_events]
                selected = select_rejection_free_event(
                    all_moves,
                    all_rates,
                    barriers=local_events.barriers,
                )
            if selected is not None:
                new_h_indices, moved_slot = apply_selected_event_to_sites(self.h_indices, selected)
                self.log(
                    f"[Rank 0] Step {step}: selected H {selected.h_site}->{selected.n_site}, "
                    f"barrier={selected.barrier_ev}, rate={selected.rate_hz:.6e} Hz, "
                    f"dt={selected.dt_s:.6e} s"
                )
            else:
                new_h_indices = None
                self.log(f"[Rank 0] Step {step}: no valid KMC event available")
            timer.add("selection_cpu", time.time() - t0)
        else:
            new_h_indices = None

        t0 = time.time()
        selected = self.ctx.comm.bcast(selected, root=0)
        timer.add("selection_comm", time.time() - t0)
        if selected is None:
            recovered = self._try_devanathan_no_event_recovery(
                step,
                timer,
                incremental_enabled=incremental_enabled,
            )
            wall_total = time.time() - step_wall_t0
            self.timing_writer.append(step, wall_total, timer.snapshot())
            log_step_timing(self.log, step, timer.snapshot(), total_wall=wall_total)
            return recovered

        t0 = time.time()
        self.h_indices = np.asarray(self.ctx.comm.bcast(new_h_indices, root=0), dtype=int)
        moved_slot = self.ctx.comm.bcast(moved_slot if rank == 0 else None, root=0)
        timer.add("state_comm", time.time() - t0)

        t0 = time.time()
        if moved_slot is not None and self.h_unwrapped_positions is not None:
            delta = pbc_diff(self.sites[int(selected.h_site)], self.sites[int(selected.n_site)], self.spec.box)
            self.h_unwrapped_positions[int(moved_slot)] += delta
        self.sim_time_s += float(selected.dt_s)
        if self.full_structure is not None and moved_slot is not None:
            try:
                update_full_structure_h_atom(
                    self.full_structure,
                    self.sites,
                    int(moved_slot),
                    int(selected.n_site),
                )
            except Exception as exc:
                self.log(f"[Rank {rank}] WARNING: full_structure H update failed: {exc}")

        devanathan_update = None
        if rank == 0 and self.devanathan_boundary is not None:
            assert self.h_unwrapped_positions is not None
            self._gcmc_energy_cache_step_hits = 0
            self._gcmc_energy_cache_step_misses = 0
            devanathan_update = self.devanathan_boundary.apply(
                sites=self.sites,
                fe_positions=self._native_fe_positions(),
                h_indices=self.h_indices,
                h_unwrapped_positions=self.h_unwrapped_positions,
                elapsed_time_s=self.sim_time_s,
            )
            self.h_indices = devanathan_update.h_indices
            self.h_unwrapped_positions = devanathan_update.h_unwrapped_positions
            if devanathan_update.particle_count_changed:
                region_label = str(
                    getattr(self.devanathan_boundary, "region_label", "source")
                )
                self.log(
                    f"[Rank 0] Devanathan step {step}: "
                    f"+{len(devanathan_update.inserted_sites)} {region_label} H, "
                    f"-{len(devanathan_update.trimmed_source_sites)} {region_label} H, "
                    f"-{len(getattr(devanathan_update, 'deleted_left_sink_sites', ()))} left-boundary H, "
                    f"-{len(devanathan_update.deleted_sink_sites)} sink H, "
                    f"{region_label}={devanathan_update.source_occupied}/"
                    f"{getattr(devanathan_update, 'maintenance_setpoint_h', devanathan_update.source_target_h)} "
                    f"H ({devanathan_update.source_fe_atoms} Fe basis), "
                    f"left_removed_total={getattr(devanathan_update, 'cumulative_left_removed', 0)}, "
                    f"sink_removed_total={devanathan_update.cumulative_removed}"
                )

        if self.devanathan_boundary is not None:
            self.h_indices = np.asarray(self.ctx.comm.bcast(self.h_indices if rank == 0 else None, root=0), dtype=int)
            self.h_unwrapped_positions = np.asarray(
                self.ctx.comm.bcast(self.h_unwrapped_positions if rank == 0 else None, root=0),
                dtype=float,
            )
            devanathan_update = self.ctx.comm.bcast(devanathan_update, root=0)
        if incremental_enabled:
            particle_changed_sites: list[int] = []
            if devanathan_update is not None:
                particle_changed_sites.extend(
                    int(site) for site in devanathan_update.inserted_sites
                )
                particle_changed_sites.extend(
                    int(site) for site in devanathan_update.deleted_sink_sites
                )
                particle_changed_sites.extend(
                    int(site)
                    for site in getattr(
                        devanathan_update,
                        "deleted_left_sink_sites",
                        (),
                    )
                )
                particle_changed_sites.extend(
                    int(site) for site in devanathan_update.trimmed_source_sites
                )
            self._prepare_incremental_next_update(
                selected,
                particle_changed_sites=particle_changed_sites,
            )
        self._refresh_h_lookup()
        timer.add("state_update", time.time() - t0)

        t0 = time.time()
        if self.msd_tracker is not None and self.h_unwrapped_positions is not None:
            self.msd_tracker.record(step + 1, self.sim_time_s, current_positions=self.h_unwrapped_positions)
        if self._should_periodic_dump(step):
            self._append_only_h_trajectory(step + 1)
        self._append_kmc_diag(step, selected, moved_slot)
        self._append_selected_event_csv(step, selected, moved_slot)
        self._append_devanathan_diag(step, devanathan_update)
        self._append_kmc_timestep(step, selected)
        self._write_restart_checkpoint(step + 1)
        timer.add("dump_traj", time.time() - t0)

        if rank == 0:
            t0 = time.time()
            self.cache_mgr.save_delta()
            if self.gcmc_energy_cache is not None:
                self.gcmc_energy_cache.save()
            timer.add("cache_save", time.time() - t0)
        wall_total = time.time() - step_wall_t0
        self.timing_writer.append(step, wall_total, timer.snapshot())
        log_step_timing(self.log, step, timer.snapshot(), total_wall=wall_total)
        return True

    # ------------------------------------------------------------------
    # One KMC step
    # ------------------------------------------------------------------
    def step_once(self, step: int) -> bool:
        assert self.sites is not None
        assert self.neighbors is not None
        assert self.h_indices is not None

        step_wall_t0 = time.time()
        timer = StepTimer()
        rank = self.ctx.rank
        occupied = set(map(int, self.h_indices))
        local_h = self.local_h_sites()

        event_cfg = EventManagerConfig(
            env_radius_a=float(self._cfg_value("env_radius_a", "radius_a", default=5.0)),
            use_cache_for_jobs=self.scheduler.cfg.use_cache_for_jobs,
            cache_only=(self.scheduler.cfg.assignment_mode == "cache_only"),
            low_barrier_ev=float(self._cfg_value("low_barrier_ev", default=1.0e-3)),
            rank=rank,
        )

        t0 = time.time()
        self._prepare_env_key_cache()
        cache_events = build_candidate_events(
            step=step,
            rank=rank,
            local_h_sites=local_h,
            neighbors=self.neighbors,
            occupied_sites=occupied,
            cache=self.cache_mgr.cache,
            make_key=self.make_event_key,
            kinetics=self.kinetics,
            cfg=event_cfg,
            failed_keys=self.failed_barrier_keys,
            reported_failed_keys=self.reported_failed_rate_keys,
        )
        timer.add("move_loop", time.time() - t0)

        self.log(
            f"[Rank {rank}] Candidate summary: local_H={len(local_h)}, "
            f"cache_hits={cache_events.cache_hits}, cache_misses={cache_events.cache_misses}, "
            f"miss_jobs={len(cache_events.miss_jobs)}, valid_cache_events={len(cache_events.valid_moves)}, "
            f"rate_rows={len(cache_events.rate_rows)}"
        )

        t0 = time.time()
        my_jobs, key_request_ranks = self.scheduler.gather_and_scatter_miss_jobs(
            cache_events.miss_jobs,
            step=step,
        )
        timer.add("job_comm", time.time() - t0)

        t0 = time.time()
        results_local, neb_diag = self.run_neb_jobs(my_jobs, step)
        timer.add("barrier_neb", time.time() - t0)
        timer.add("neb_struct", neb_diag.neb_struct_s)

        t0 = time.time()
        if self.scheduler.cfg.merge_mode == "global":
            fold_results = self.scheduler.collect_results_global(results_local, step=step)
            failed_keys = {k for k, v in fold_results.items() if not np.isfinite(v)}
        else:
            fold_results, failed_keys = self.scheduler.collect_results_local(
                results_local,
                step=step,
                key_request_ranks=key_request_ranks,
            )
        timer.add("result_comm", time.time() - t0)

        self.failed_barrier_keys.update(failed_keys)

        if self.scheduler.cfg.use_cache_for_jobs:
            t0 = time.time()
            n_cached = self.cache_mgr.update_finite(fold_results)
            timer.add("cache_update", time.time() - t0)
            if n_cached:
                self.log(f"[Rank {rank}] Cached {n_cached} finite NEB result(s)")

            # In mixed mode, local ranks normally receive only the barriers they
            # requested. Periodically allgather finite cache entries so future
            # steps can benefit from work computed on other ranks without doing
            # a global result broadcast every step.
            if self.scheduler.cfg.merge_mode == "mixed" and self.cache_mgr.should_mixed_share(step):
                self.ctx.phase_guard(step, "POSTNEB_MIXED_CACHE_SYNC")
                shared_parts = self.ctx.comm.allgather(list(self.cache_mgr.cache.store.items()))
                merged_shared: Dict[Any, Any] = {}
                for part in shared_parts:
                    for key, value in part:
                        if key not in merged_shared and np.isfinite(float(value)):
                            merged_shared[key] = float(value)
                n_shared = self.cache_mgr.update_finite(merged_shared)
                if n_shared:
                    self.log(f"[Rank {rank}] Mixed cache sync merged {n_shared} finite entry/entries")

        t0 = time.time()
        neb_events = fold_neb_results_into_events(
            step=step,
            rank=rank,
            local_miss_jobs=cache_events.miss_jobs,
            fold_results=fold_results,
            kinetics=self.kinetics,
            failed_keys=self.failed_barrier_keys,
            reported_failed_keys=self.reported_failed_rate_keys,
        )

        local_events = merge_candidate_results(cache_events, neb_events)
        timer.add("selection_cpu", time.time() - t0)

        # Guard KMC-selection collectives explicitly so a failure before or after
        # NEB result exchange cannot silently cross these gather calls.
        t0 = time.time()
        self._heartbeat(step, "KMC_GATHER_NEB_DIAG", "before_gather")
        gathered_neb_diag = self.ctx.comm.gather(neb_diag, root=0)
        self._heartbeat(step, "KMC_GATHER_NEB_DIAG", "after_gather")
        if rank == 0:
            self._write_all_neb_diag(gathered_neb_diag)

        self.ctx.phase_guard(step, "KMC_GATHER_RATE_ROWS")
        self._heartbeat(step, "KMC_GATHER_RATE_ROWS", "before_gather")
        gathered_rate_rows = self.ctx.comm.gather(local_events.rate_rows, root=0)
        self._heartbeat(step, "KMC_GATHER_RATE_ROWS", "after_gather")
        timer.add("rate_comm", time.time() - t0)

        t0 = time.time()
        self.ctx.phase_guard(step, "KMC_GATHER_MOVES")
        self._heartbeat(step, "KMC_GATHER_MOVES", "before_gather")
        gathered_moves = self.ctx.comm.gather(local_events.valid_moves, root=0)
        self._heartbeat(step, "KMC_GATHER_MOVES", "after_gather")
        self.ctx.phase_guard(step, "KMC_GATHER_RATES")
        self._heartbeat(step, "KMC_GATHER_RATES", "before_gather")
        gathered_rates = self.ctx.comm.gather(local_events.rates, root=0)
        self._heartbeat(step, "KMC_GATHER_RATES", "after_gather")
        self.ctx.phase_guard(step, "KMC_GATHER_BARRIERS")
        self._heartbeat(step, "KMC_GATHER_BARRIERS", "before_gather")
        gathered_barriers = self.ctx.comm.gather(local_events.barriers, root=0)
        self._heartbeat(step, "KMC_GATHER_BARRIERS", "after_gather")
        timer.add("selection_comm", time.time() - t0)

        selected = None
        if rank == 0:
            all_moves = [move for part in gathered_moves for move in part]
            all_rates = [rate for part in gathered_rates for rate in part]
            all_barriers: Dict[tuple[int, int], float] = {}
            for part in gathered_barriers:
                all_barriers.update(part)

            # Keep event selection independent of whether a barrier came from
            # the cache path or the just-computed NEB path.  Those paths append
            # events in different local order, and Python's weighted selector is
            # order-sensitive for a fixed RNG draw.
            ordered_events = sorted(
                zip(all_moves, all_rates),
                key=lambda item: (int(item[0][0]), int(item[0][1])),
            )
            all_moves = [move for move, _rate in ordered_events]
            all_rates = [rate for _move, rate in ordered_events]

            selected = select_rejection_free_event(
                all_moves,
                all_rates,
                barriers=all_barriers,
            )

            if selected is None:
                self.log(f"[Rank 0] Step {step}: no valid KMC event available")
            else:
                self.log(
                    f"[Rank 0] Step {step}: selected H {selected.h_site}->{selected.n_site}, "
                    f"barrier={selected.barrier_ev}, rate={selected.rate_hz:.6e} Hz, "
                    f"dt={selected.dt_s:.6e} s"
                )

        self._write_rates_allranks(gathered_rate_rows or [], selected=selected)

        self.ctx.phase_guard(step, "KMC_BCAST_SELECTED")
        t0 = time.time()
        self._heartbeat(step, "KMC_BCAST_SELECTED", "before_bcast")
        selected = self.ctx.comm.bcast(selected, root=0)
        self._heartbeat(step, "KMC_BCAST_SELECTED", "after_bcast")
        timer.add("selection_comm", time.time() - t0)
        if selected is None:
            recovered = self._try_devanathan_no_event_recovery(step, timer)
            wall_total = time.time() - step_wall_t0
            self.timing_writer.append(step, wall_total, timer.snapshot())
            log_step_timing(self.log, step, timer.snapshot(), total_wall=wall_total)
            return recovered

        # Apply the accepted move once on rank 0 and broadcast the authoritative
        # occupancy vector. This prevents rank-dependent local_H divergence on
        # later steps.
        moved_slot = None
        new_h_indices = None
        if rank == 0:
            new_h_indices, moved_slot = apply_selected_event_to_sites(self.h_indices, selected)

        self.ctx.phase_guard(step, "KMC_BCAST_STATE")
        t0 = time.time()
        self._heartbeat(step, "KMC_BCAST_STATE_H_INDICES", "before_bcast")
        self.h_indices = np.asarray(self.ctx.comm.bcast(new_h_indices, root=0), dtype=int)
        self._heartbeat(step, "KMC_BCAST_STATE_H_INDICES", "after_bcast")
        self._heartbeat(step, "KMC_BCAST_STATE_MOVED_SLOT", "before_bcast")
        moved_slot = self.ctx.comm.bcast(moved_slot, root=0)
        self._heartbeat(step, "KMC_BCAST_STATE_MOVED_SLOT", "after_bcast")
        timer.add("state_comm", time.time() - t0)

        t0 = time.time()
        if moved_slot is not None and self.h_unwrapped_positions is not None:
            delta = pbc_diff(self.sites[int(selected.h_site)], self.sites[int(selected.n_site)], self.spec.box)
            self.h_unwrapped_positions[int(moved_slot)] += delta
        self.sim_time_s += float(selected.dt_s)

        if self.full_structure is not None and moved_slot is not None:
            try:
                update_full_structure_h_atom(
                    self.full_structure,
                    self.sites,
                    int(moved_slot),
                    int(selected.n_site),
                )
            except Exception as exc:
                self.log(f"[Rank {rank}] WARNING: full_structure H update failed: {exc}")

        devanathan_update = None
        if rank == 0 and self.devanathan_boundary is not None:
            assert self.h_unwrapped_positions is not None
            self._gcmc_energy_cache_step_hits = 0
            self._gcmc_energy_cache_step_misses = 0
            devanathan_update = self.devanathan_boundary.apply(
                sites=self.sites,
                fe_positions=self._native_fe_positions(),
                h_indices=self.h_indices,
                h_unwrapped_positions=self.h_unwrapped_positions,
                elapsed_time_s=self.sim_time_s,
            )
            self.h_indices = devanathan_update.h_indices
            self.h_unwrapped_positions = devanathan_update.h_unwrapped_positions
            if devanathan_update.particle_count_changed:
                region_label = str(
                    getattr(self.devanathan_boundary, "region_label", "source")
                )
                self.log(
                    f"[Rank 0] Devanathan step {step}: "
                    f"+{len(devanathan_update.inserted_sites)} {region_label} H, "
                    f"-{len(devanathan_update.trimmed_source_sites)} {region_label} H, "
                    f"-{len(getattr(devanathan_update, 'deleted_left_sink_sites', ()))} left-boundary H, "
                    f"-{len(devanathan_update.deleted_sink_sites)} sink H, "
                    f"{region_label}={devanathan_update.source_occupied}/"
                    f"{getattr(devanathan_update, 'maintenance_setpoint_h', devanathan_update.source_target_h)} "
                    f"H ({devanathan_update.source_fe_atoms} Fe basis), "
                    f"left_removed_total={getattr(devanathan_update, 'cumulative_left_removed', 0)}, "
                    f"sink_removed_total={devanathan_update.cumulative_removed}"
                )

        if self.devanathan_boundary is not None:
            self.h_indices = np.asarray(
                self.ctx.comm.bcast(self.h_indices if rank == 0 else None, root=0),
                dtype=int,
            )
            self.h_unwrapped_positions = np.asarray(
                self.ctx.comm.bcast(
                    self.h_unwrapped_positions if rank == 0 else None,
                    root=0,
                ),
                dtype=float,
            )
            devanathan_update = self.ctx.comm.bcast(devanathan_update, root=0)
            if (
                self.full_structure is not None
                and devanathan_update is not None
                and devanathan_update.particle_count_changed
            ):
                try:
                    if self.external_host_structure is not None:
                        self.full_structure = build_full_structure_from_host(
                            self.external_host_structure,
                            self.sites,
                            self.h_indices,
                        )
                    else:
                        self.full_structure = build_full_structure(
                            self.sites,
                            self.h_indices,
                            self.spec,
                        )
                except Exception as exc:
                    self.log(
                        f"[Rank {rank}] WARNING: full_structure rebuild after "
                        f"Devanathan update failed: {exc}"
                    )

        self._refresh_h_lookup()
        timer.add("state_update", time.time() - t0)
        should_dump = self._should_periodic_dump(step)
        if self.msd_tracker is not None and self.h_unwrapped_positions is not None:
            t0 = time.time()
            self.msd_tracker.record(
                step + 1,
                self.sim_time_s,
                current_positions=self.h_unwrapped_positions,
            )
            timer.add("dump_diag", time.time() - t0)
        t0 = time.time()
        if should_dump:
            self._append_only_h_trajectory(step + 1)
        self._append_kmc_diag(step, selected, moved_slot)
        self._append_selected_event_csv(step, selected, moved_slot)
        self._append_devanathan_diag(step, devanathan_update)
        self._append_kmc_timestep(step, selected)
        self._write_restart_checkpoint(step + 1)
        timer.add("dump_traj", time.time() - t0)

        t0 = time.time()
        self.cache_mgr.save_delta()
        if self.gcmc_energy_cache is not None:
            self.gcmc_energy_cache.save()
        timer.add("cache_save", time.time() - t0)
        wall_total = time.time() - step_wall_t0
        self.timing_writer.append(step, wall_total, timer.snapshot())
        log_step_timing(self.log, step, timer.snapshot(), total_wall=wall_total)
        return True

    def run(self) -> None:
        self.initialize()
        if self._restart_enabled() and self.ctx.rank == 0 and int(self.restart_start_step) > 0:
            # Preserve the loaded restart state in the new run outputs before
            # the first accepted move advances the trajectory to step + 1.
            if not self._only_h_traj_initialized:
                self._append_only_h_trajectory(int(self.restart_start_step))
            elif self._append_outputs:
                self.log(
                    f"[Rank 0] Existing trajectory retained at restart step "
                    f"{int(self.restart_start_step)}"
                )
            self._write_restart_checkpoint(int(self.restart_start_step), force=True)
        if self._build_only_enabled():
            built = 0
            if self.ctx.rank == 0:
                built = self.build_and_dump_local_envs()
                self.log(
                    f"[Rank 0] Local environment build-only mode complete: "
                    f"built {built}/{int(self._cfg_value('build_count', 'BUILD_LOCAL_ENV_COUNT', default=10))} "
                    f"'{self._cfg_value('local_env_mode', 'mode', 'LOCAL_ENV_MODE', default='wrapped')}' "
                    f"environments in {self._cfg_value('build_out', 'BUILD_LOCAL_ENV_OUT', default='local_env_builds')}"
                )
                if built < int(self._cfg_value("build_count", "BUILD_LOCAL_ENV_COUNT", default=10)):
                    self.log(
                        f"[Rank 0] Requested {int(self._cfg_value('build_count', 'BUILD_LOCAL_ENV_COUNT', default=10))} "
                        f"environments but only built {built}.",
                        step=-1,
                    )
            self.ctx.comm.bcast(built if self.ctx.rank == 0 else None, root=0)
            return
        steps = int(self._cfg_value("steps", "STEPS", default=100))
        t0 = time.time()
        completed_steps = int(self.restart_start_step)

        start_step = int(self.restart_start_step)
        if start_step >= steps:
            if self.ctx.rank == 0:
                self.log(
                    f"[Rank 0] Restart frame is step {start_step}, which is >= STEPS={steps}; "
                    "no additional KMC steps requested."
                )
        for step in range(start_step, steps):
            if self._lammps_only_enabled():
                keep_going = self.step_once_lammps_only(step)
            else:
                keep_going = self.step_once(step)
            if not keep_going:
                break
            completed_steps = step + 1
            if (
                self._stop_when_no_h_remaining_enabled()
                and self.h_indices is not None
                and len(self.h_indices) == 0
            ):
                if self.ctx.rank == 0:
                    self.log(
                        f"[Rank 0] Step {completed_steps}: no H atoms remain; "
                        "stopping because KMC_STOP_WHEN_NO_H=1"
                    )
                break

        if (not self._lammps_only_enabled()) or self.ctx.rank == 0:
            self.cache_mgr.save_delta()
        if self.ctx.rank == 0 and self.gcmc_energy_cache is not None:
            self.gcmc_energy_cache.save(full=True)
        self.ctx.comm.Barrier()
        if self.ctx.rank == 0:
            try:
                merge_size = 1 if self._lammps_only_enabled() else self.ctx.size
                merge_rank_caches(
                    self.cache_mgr.cfg,
                    merge_size,
                    logger=self.log,
                )
            except Exception as exc:
                self.log(f"[Rank 0] Final barrier cache merge failed: {exc}")
            if self.msd_tracker is not None:
                try:
                    self.msd_tracker.record(
                        completed_steps,
                        self.sim_time_s,
                        current_positions=self.h_unwrapped_positions,
                        force=True,
                    )
                    self.msd_tracker.plot()
                except Exception as exc:
                    self.log(f"[Rank 0] MSD post-processing failed: {exc}")
            self._write_restart_checkpoint(completed_steps, force=True)
            self.log(f"[Rank 0] KMC/NEB simulation complete in {time.time() - t0:.2f} s")


def build_simulation_from_env():
    """Build a simulation from the same environment-driven config as the legacy driver."""
    ctx = MPIContext.create()
    if load_config is None:
        return KMCSimulation(DriverDefaults())
    cfg = load_config(mpi_size=ctx.size, rank=ctx.rank)
    return KMCSimulation(cfg)
