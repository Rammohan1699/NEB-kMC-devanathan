#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for the refactored KMC + NEB implementation.

This file intentionally has no MPI, ASE, LAMMPS, scipy, or project-service imports.
It only reads environment variables, validates/normalizes them, and builds a
single immutable SimulationConfig object that the rest of the code can receive.

Usage
-----
    from config import load_config

    cfg = load_config()
    print(cfg.cache.schema)
    print(cfg.lammps.potential)

For a quick sanity check:
    python config.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


# -----------------------------------------------------------------------------
# Environment parsing helpers
# -----------------------------------------------------------------------------
_FALSES = {"0", "false", "False", "FALSE", "no", "No", "NO", "off", "OFF"}
_TRUES = {"1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "ON"}


class ConfigError(ValueError):
    """Raised when configuration is internally inconsistent."""


def _env(mapping: Mapping[str, str], name: str, default: Any = None) -> Any:
    value = mapping.get(name)
    return default if value is None else value


def env_str(mapping: Mapping[str, str], name: str, default: Any = None) -> Any:
    return _env(mapping, name, default)


def env_int(mapping: Mapping[str, str], name: str, default: Any = None) -> int:
    value = _env(mapping, name, default)
    try:
        return int(value)
    except Exception as exc:
        raise ConfigError(f"Environment variable {name}={value!r} is not an int") from exc


def env_float(mapping: Mapping[str, str], name: str, default: Any = None) -> float:
    value = _env(mapping, name, default)
    try:
        return float(value)
    except Exception as exc:
        raise ConfigError(f"Environment variable {name}={value!r} is not a float") from exc


def env_bool(mapping: Mapping[str, str], name: str, default: Any = False) -> bool:
    value = mapping.get(name)
    if value is None:
        value = default
    if isinstance(value, bool):
        return value
    text = str(value)
    if text in _FALSES:
        return False
    if text in _TRUES:
        return True
    # Preserve the legacy behavior: any non-false string is truthy.
    return True


def env_optional_bool(mapping: Mapping[str, str], name: str) -> Optional[bool]:
    value = mapping.get(name)
    if value is None:
        return None
    return str(value) not in _FALSES


def _abs_path(path: str | os.PathLike[str], *, base_dir: Path) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return str(p)


def _parse_csv_paths(raw: str | None, *, base_dir: Path) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(_abs_path(item, base_dir=base_dir))
    return out


# -----------------------------------------------------------------------------
# Section configs
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PathsConfig:
    """Project paths derived from the location of this config module."""

    script_dir: str
    default_eam_file: str
    default_nnp_dir: str


@dataclass(frozen=True)
class SimulationSizeConfig:
    """Basic lattice, KMC, and physical constants."""

    lattice_a: float = 2.8601
    nx: int = 30
    ny: int = 30
    nz: int = 30
    cutoff: int = 2
    num_h: int = 540
    steps: int = 10000
    k_b_ev_per_k: float = 8.617e-5
    temperature_k: float = 300.0
    attempt_frequency_hz: float = 1e13
    use_octahedral_voids: bool = False
    kmc_seed: Optional[str] = None

    @property
    def box_lengths(self) -> tuple[float, float, float]:
        return (
            self.lattice_a * self.nx,
            self.lattice_a * self.ny,
            self.lattice_a * self.nz,
        )


@dataclass(frozen=True)
class SiteMapConfig:
    """Controls whether KMC uses generated bulk sites or an external map."""

    source: str = "generated"  # generated | external
    site_map_file: str = ""
    host_structure_file: str = ""
    host_fe_type: int = 1
    initial_h_region_file: str = ""
    initial_h_regions: str = ""


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = False
    out_dir: str = "neb/validation_dump"
    file_format: str = "xyz"
    use_ase_writer: bool = False
    overwrite: bool = True
    ase_dump: bool = False
    ase_dump_pre: bool = False
    ase_dir: str = "ase"


@dataclass(frozen=True)
class DiagnosticsConfig:
    debug_logging: bool = True
    ase_log_dir: str = "neb/ase_logs"
    dump_every_steps: int = 1
    low_barrier_ev: float = 1e-3
    clump_eps_a: float = 0.05
    msd_log_interval: int = 1000
    msd_log_file: str = "msd_vs_time.csv"
    msd_plot_file: str = "msd_plot.png"
    neb_diag: bool = True
    neb_job_trace: bool = False
    neb_dump: bool = False
    neb_failure_capture: bool = False
    neb_failure_dir: str = "neb/neb_failure_capture"
    restart_checkpoint_interval: int = 1000
    restart_checkpoint_dir: str = "checkpoints"
    restart_checkpoint_prefix: str = "kmc_restart_checkpoint"


@dataclass(frozen=True)
class LocalEnvironmentConfig:
    mode: str = "shell"  # radial | shell | wrapped
    env_key_mode: str = "env_plus_dir"  # env_plus_dir | env_only
    radius_a: float = 5.0
    pos_bin_a: float = 0.10
    hop_bin_a: float = 0.02
    affect_radius_a: float = 6.0
    shell_inner_radius_a: float = 6.0
    shell_outer_radius_a: float = 8.0
    build_only: bool = False
    build_count: int = 10
    build_out: str = "local_env_builds"
    build_selector: str = "random"  # random | boundary | cluster
    boundary_margin_a: float = 5.0
    cluster_radius_a: float = 5.0

    @property
    def mode_schema(self) -> str:
        if self.mode == "shell":
            return f"mode:shell_IR{self.shell_inner_radius_a}_OR{self.shell_outer_radius_a}"
        if self.mode == "radial":
            return "mode:radial"
        return "mode:wrapped"


@dataclass(frozen=True)
class NEBConfig:
    n_images: int = 3
    engine: str = "lammps_only"  # lammps_only | ase_lammps
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
    min_batch: int = 64
    optimize_endpoints: bool = True
    # These two mirror the legacy FIX_SUBLATTICE_* overrides passed into LammpsNEB.
    fix_sublattice_h: Optional[bool] = None
    fix_sublattice_fe: Optional[bool] = None
    # These two mirror the older FIX_OTHER_H/FIX_FE behavior in neb_engine.py/lammps_neb.py.
    fix_other_h: bool = False
    fix_fe: bool = False
    optimizer_fmax: float = 0.05
    optimizer_steps: int = 300
    endpoint_optimizer_steps: int = 200

    @property
    def fix_sublattice_overrides(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        if self.fix_sublattice_h is not None:
            out["H"] = self.fix_sublattice_h
        if self.fix_sublattice_fe is not None:
            out["Fe"] = self.fix_sublattice_fe
        return out


@dataclass(frozen=True)
class LammpsConfig:
    potential: str = "EAM"  # EAM | NNP
    potential_eam_file: str = ""
    potential_nnp_dir: str = ""
    engine: str = "eam_fs"
    pair_style: str = ""
    pair_coeff: str = ""
    files_raw: str = ""
    files: tuple[str, ...] = ()
    nnp_dir: str = ""
    nnp_elements: str = "Fe H"
    nnp_cutoff: float = 6.60
    nnp_showewsum: int = 1000
    nnp_maxew: int = 1_000_000
    nnp_cflength: float = 1.8897261328
    nnp_cfenergy: float = 0.0367493254
    python_path: str = ""
    lib_dir: str = ""


@dataclass(frozen=True)
class CacheConfig:
    schema: str
    rank_cache_template: str
    merged_cache_file: str
    preload_file: str = ""
    merge_mode: str = "global"  # global | local | mixed
    merge_interval: int = 100
    job_assignment_mode: str = "cache_dedupe"  # all_jobs | cache_no_dedupe | cache_dedupe | cache_only
    use_cache_for_jobs: bool = True
    dedupe_jobs: bool = True


@dataclass(frozen=True)
class RestartConfig:
    enabled: bool = False
    directory: str = "."
    step: int = 0
    trajectory_file: str = "H_trajectory_onlyH.lammpstrj"
    timestep_file: str = "diagnostics/kmc_timestep_vs_step.csv"
    diagnostics_file: str = "kmc_diagnostics_rank0.log"
    checkpoint_file: str = "kmc_restart_checkpoint.pkl"
    strict: bool = True


@dataclass(frozen=True)
class SimulationConfig:
    paths: PathsConfig
    size: SimulationSizeConfig
    site_map: SiteMapConfig
    validation: ValidationConfig
    diagnostics: DiagnosticsConfig
    environment: LocalEnvironmentConfig
    neb: NEBConfig
    lammps: LammpsConfig
    cache: CacheConfig
    restart: RestartConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
def load_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    script_dir: str | os.PathLike[str] | None = None,
    mpi_size: int = 1,
    rank: int = 0,
) -> SimulationConfig:
    """
    Load, normalize, and validate simulation configuration.

    Parameters
    ----------
    env
        Environment mapping. Defaults to os.environ. Supplying a dict is useful
        for tests.
    script_dir
        Base directory used for relative potential/cache paths. Defaults to the
        directory containing this config.py file.
    mpi_size, rank
        Optional MPI information. We keep MPI out of this module, but these
        values allow us to apply legacy multi-rank safety adjustments.
    """

    e = os.environ if env is None else env
    base_dir = Path(script_dir).resolve() if script_dir is not None else Path(__file__).resolve().parent

    default_eam_file = _abs_path("PotentialB3410-modified.fs", base_dir=base_dir)
    default_nnp_dir = "/Users/rtirunelveli/LAMMPS/Input/NNP/POTENTIAL"
    paths = PathsConfig(
        script_dir=str(base_dir),
        default_eam_file=default_eam_file,
        default_nnp_dir=default_nnp_dir,
    )

    size = SimulationSizeConfig(
        lattice_a=env_float(e, "LATTICE_A", 2.8601),
        nx=env_int(e, "NX", 30),
        ny=env_int(e, "NY", 30),
        nz=env_int(e, "NZ", 30),
        cutoff=env_int(e, "CUTOFF", 2),
        num_h=env_int(e, "NUM_H", 540),
        steps=env_int(e, "STEPS", 100),
        k_b_ev_per_k=env_float(e, "K_B_EV_PER_K", 8.617e-5),
        temperature_k=env_float(e, "TEMPERATURE_K", env_float(e, "T", 300.0)),
        attempt_frequency_hz=env_float(e, "ATTEMPT_FREQUENCY_HZ", env_float(e, "NU", 1e13)),
        use_octahedral_voids=env_bool(e, "USE_OCTAHEDRAL_VOIDS", False),
        kmc_seed=env_str(e, "KMC_SEED", None),
    )

    site_source = str(env_str(e, "KMC_SITE_SOURCE", "generated") or "generated").lower().strip()
    if site_source in {"map", "file", "external_map"}:
        site_source = "external"
    site_map_file = str(env_str(e, "KMC_SITE_MAP_FILE", env_str(e, "SITE_MAP_FILE", "")) or "").strip()
    host_structure_file = str(
        env_str(e, "KMC_HOST_STRUCTURE_FILE", env_str(e, "HOST_STRUCTURE_FILE", "")) or ""
    ).strip()
    if site_source == "generated" and (site_map_file or host_structure_file):
        site_source = "external"
    if site_source not in {"generated", "external"}:
        raise ConfigError(f"KMC_SITE_SOURCE={site_source!r} must be 'generated' or 'external'")
    site_map = SiteMapConfig(
        source=site_source,
        site_map_file=site_map_file,
        host_structure_file=host_structure_file,
        host_fe_type=env_int(e, "KMC_HOST_FE_TYPE", 1),
        initial_h_region_file=str(
            env_str(e, "KMC_INITIAL_H_REGION_FILE", env_str(e, "KMC_H_REGION_FILE", "")) or ""
        ).strip(),
        initial_h_regions=str(
            env_str(
                e,
                "KMC_INITIAL_H_REGIONS",
                env_str(e, "KMC_INITIAL_H_REGION", env_str(e, "KMC_H_REGIONS", env_str(e, "KMC_H_REGION", ""))),
            )
            or ""
        ).strip(),
    )

    validation_format = str(env_str(e, "VALIDATION_FORMAT", "xyz")).lower().strip()
    if validation_format not in {"xyz", "lammps"}:
        validation_format = "xyz"

    validation = ValidationConfig(
        enabled=env_bool(e, "VALIDATION_MODE", False),
        out_dir=str(env_str(e, "VALIDATION_OUT", "neb/validation_dump")),
        file_format=validation_format,
        use_ase_writer=env_bool(e, "VALIDATION_USE_ASE", False),
        overwrite=env_bool(e, "VALIDATION_OVERWRITE", True),
        ase_dump=env_bool(e, "VALIDATION_ASE_DUMP", False),
        ase_dump_pre=env_bool(e, "VALIDATION_ASE_DUMP_PRE", False),
        ase_dir=str(env_str(e, "VALIDATION_ASE_DIR", "ase")),
    )

    diagnostics = DiagnosticsConfig(
        debug_logging=env_bool(e, "DEBUG_MODE", True),
        ase_log_dir=str(env_str(e, "ASE_LOG_DIR", "neb/ase_logs")),
        dump_every_steps=max(1, env_int(e, "DUMP_EVERY_STEPS", 1)),
        low_barrier_ev=env_float(e, "LOW_BARRIER_EV", 1e-3),
        clump_eps_a=env_float(e, "CLUMP_EPS_A", 0.05),
        msd_log_interval=env_int(e, "MSD_LOG_INTERVAL", 1000),
        msd_log_file=str(env_str(e, "MSD_LOG_FILE", "msd_vs_time.csv")),
        msd_plot_file=str(env_str(e, "MSD_PLOT_FILE", "msd_plot.png")),
        neb_diag=env_bool(e, "NEB_DIAG", True),
        neb_job_trace=env_bool(e, "NEB_JOB_TRACE", False),
        neb_dump=env_bool(e, "NEB_DUMP", False),
        neb_failure_capture=env_bool(e, "NEB_FAILURE_CAPTURE", False),
        neb_failure_dir=str(env_str(e, "NEB_FAILURE_DIR", "neb/neb_failure_capture")),
        restart_checkpoint_interval=max(
            0,
            env_int(e, "RESTART_CHECKPOINT_INTERVAL", env_int(e, "CHECKPOINT_INTERVAL", 1000)),
        ),
        restart_checkpoint_dir=str(
            env_str(e, "RESTART_CHECKPOINT_DIR", env_str(e, "CHECKPOINT_DIR", "checkpoints")) or "checkpoints"
        ).strip(),
        restart_checkpoint_prefix=str(
            env_str(e, "RESTART_CHECKPOINT_PREFIX", env_str(e, "CHECKPOINT_PREFIX", "kmc_restart_checkpoint"))
            or "kmc_restart_checkpoint"
        ).strip(),
    )

    # Legacy behavior: DEBUG_MODE=0 disables detailed NEB diagnostics/traces.
    if not diagnostics.debug_logging:
        diagnostics = DiagnosticsConfig(
            **{**asdict(diagnostics), "neb_diag": False, "neb_job_trace": False}
        )

    # Legacy safety: validation ASE band dumps were disabled for multi-rank runs.
    if validation.enabled and mpi_size > 1:
        validation = ValidationConfig(
            **{**asdict(validation), "ase_dump": False, "ase_dump_pre": False}
        )

    local_mode = str(env_str(e, "LOCAL_ENV_MODE", "radial")).lower().strip()
    if local_mode not in {"wrapped", "radial", "shell"}:
        local_mode = "radial"
    if site_map.source == "external":
        local_mode = "shell"

    build_selector = str(env_str(e, "BUILD_LOCAL_ENV_SELECTOR", "random")).lower().strip()
    if build_selector not in {"random", "boundary", "cluster"}:
        build_selector = "random"

    shell_inner = max(0.0, env_float(e, "SHELL_INNER_RADIUS_A", 5.0))
    shell_outer = max(shell_inner, env_float(e, "SHELL_OUTER_RADIUS_A", 6.0))

    environment = LocalEnvironmentConfig(
        mode=local_mode,
        env_key_mode=str(env_str(e, "ENV_KEY_MODE", "env_plus_dir")),
        radius_a=env_float(e, "ENV_RADIUS_A", 5.0),
        pos_bin_a=env_float(e, "POS_BIN_A", 0.10),
        hop_bin_a=env_float(e, "HOP_BIN_A", 0.02),
        affect_radius_a=env_float(e, "ENV_AFFECT_RADIUS_A", 6.0),
        shell_inner_radius_a=shell_inner,
        shell_outer_radius_a=shell_outer,
        build_only=env_bool(e, "BUILD_LOCAL_ENV_ONLY", False),
        build_count=max(1, env_int(e, "BUILD_LOCAL_ENV_COUNT", 10)),
        build_out=str(env_str(e, "BUILD_LOCAL_ENV_OUT", "local_env_builds") or "local_env_builds"),
        build_selector=build_selector,
        boundary_margin_a=max(0.0, env_float(e, "BOUNDARY_MARGIN_A", 5.0)),
        cluster_radius_a=max(0.0, env_float(e, "CLUSTER_RADIUS_A", 5.0)),
    )

    neb_engine = str(env_str(e, "NEB_ENGINE", env_str(e, "KMC_NEB_ENGINE", "lammps_only")) or "lammps_only").lower()
    native_endpoint_alias = neb_engine in {
        "lammps_only_endpoint_opt",
        "lammps_only_lmp_endpoint_opt",
        "native_lammps_endpoint_opt",
    }
    native_endpoint_steps = env_int(e, "LAMMPS_ENDPOINT_OPT_STEPS", env_int(e, "ENDPOINT_OPTIMIZER_STEPS", 200))

    neb = NEBConfig(
        n_images=max(3, env_int(e, "NIMG", 3)),
        engine=neb_engine,
        replicas_per_neb=max(2, env_int(e, "LAMMPS_NEB_REPLICAS", env_int(e, "NATIVE_NEB_REPLICAS", 3))),
        native_spring_constant=env_float(e, "LAMMPS_NEB_SPRING", 5.0),
        native_force_tolerance=env_float(e, "LAMMPS_NEB_FTOL", 1.0e-5),
        native_max_steps=env_int(e, "LAMMPS_NEB_STEPS", env_int(e, "NEB_LAMMPS_STEPS", 1000)),
        native_climbing_steps=env_int(e, "LAMMPS_NEB_CLIMBING_STEPS", 0),
        native_nevery=env_int(e, "LAMMPS_NEB_NEVERY", 50),
        native_thermo_every=env_int(e, "LAMMPS_NEB_THERMO_EVERY", 50),
        native_timestep=env_float(e, "LAMMPS_NEB_TIMESTEP", 0.01),
        native_min_style=str(env_str(e, "LAMMPS_NEB_MIN_STYLE", "quickmin") or "quickmin").lower(),
        native_scratch_dir=str(env_str(e, "LAMMPS_NEB_SCRATCH_DIR", "neb/lammps_only_scratch") or "neb/lammps_only_scratch"),
        native_debug_mode=env_bool(e, "LAMMPS_NEB_DEBUG_MODE", env_bool(e, "NEB_DUMP", False)),
        native_endpoint_optimize=env_bool(e, "LAMMPS_ONLY_ENDPOINT_OPTIMIZE", native_endpoint_alias),
        native_endpoint_min_style=str(
            env_str(
                e,
                "LAMMPS_ENDPOINT_OPT_MIN_STYLE",
                str(env_str(e, "LAMMPS_NEB_MIN_STYLE", "quickmin") or "quickmin"),
            )
            or "quickmin"
        ).lower(),
        native_endpoint_energy_tolerance=env_float(e, "LAMMPS_ENDPOINT_OPT_ETOL", 0.0),
        native_endpoint_force_tolerance=env_float(e, "LAMMPS_ENDPOINT_OPT_FTOL", env_float(e, "NEB_OPTIMIZER_FMAX", 0.05)),
        native_endpoint_max_steps=native_endpoint_steps,
        native_endpoint_max_evals=env_int(e, "LAMMPS_ENDPOINT_OPT_MAXEVAL", max(1, native_endpoint_steps * 10)),
        native_endpoint_dmax=env_float(e, "LAMMPS_ENDPOINT_OPT_DMAX", 0.1),
        min_batch=env_int(e, "NEB_MIN_BATCH", 64),
        optimize_endpoints=env_bool(e, "OPTIMIZE_ENDPOINTS", True),
        fix_sublattice_h=env_optional_bool(e, "FIX_SUBLATTICE_H"),
        fix_sublattice_fe=env_optional_bool(e, "FIX_SUBLATTICE_FE"),
        fix_other_h=env_bool(e, "FIX_OTHER_H", False),
        fix_fe=env_bool(e, "FIX_FE", False),
        optimizer_fmax=env_float(e, "NEB_OPTIMIZER_FMAX", 0.05),
        optimizer_steps=env_int(e, "NEB_OPTIMIZER_STEPS", 300),
        endpoint_optimizer_steps=env_int(e, "ENDPOINT_OPTIMIZER_STEPS", 200),
    )

    lammps_engine = str(env_str(e, "LAMMPS_ENGINE", "eam_fs") or "eam_fs").strip().lower()
    potential = str(env_str(e, "POTENTIAL", "EAM") or "EAM").strip().upper()
    if "POTENTIAL" not in e and lammps_engine in {"n2p2", "hdnnp", "many_body", "manybody"}:
        potential = "NNP"
    if potential not in {"EAM", "NNP"}:
        if rank == 0:
            print(f"[Rank 0] Unsupported POTENTIAL={potential!r}. Falling back to POTENTIAL='EAM'.")
        potential = "EAM"

    potential_eam_file = _abs_path(str(env_str(e, "POTENTIAL_EAM_FILE", default_eam_file)), base_dir=base_dir)
    potential_nnp_dir = _abs_path(str(env_str(e, "POTENTIAL_NNP_DIR", default_nnp_dir)), base_dir=base_dir)
    nnp_dir = str(env_str(e, "NNP_DIR", potential_nnp_dir))
    if "NNP_DIR" not in e:
        nnp_dir = potential_nnp_dir

    lammps_files_raw = str(env_str(e, "LAMMPS_FILES", default_eam_file))
    lammps_files = _parse_csv_paths(lammps_files_raw, base_dir=base_dir)
    if potential == "EAM" and potential_eam_file not in lammps_files:
        lammps_files.append(potential_eam_file)

    lammps = LammpsConfig(
        potential=potential,
        potential_eam_file=potential_eam_file,
        potential_nnp_dir=potential_nnp_dir,
        engine=lammps_engine,
        pair_style=str(env_str(e, "LAMMPS_PAIR_STYLE", "") or "").strip(),
        pair_coeff=str(env_str(e, "LAMMPS_PAIR_COEFF", "") or "").strip(),
        files_raw=lammps_files_raw,
        files=tuple(lammps_files),
        nnp_dir=nnp_dir,
        nnp_elements=str(env_str(e, "NNP_ELEMENTS", "Fe H") or "Fe H").strip(),
        nnp_cutoff=env_float(e, "NNP_CUTOFF", 6.60),
        nnp_showewsum=env_int(e, "NNP_SHOWEWSUM", 1000),
        nnp_maxew=env_int(e, "NNP_MAXEW", 1_000_000),
        nnp_cflength=env_float(e, "NNP_CFLENGTH", 1.8897261328),
        nnp_cfenergy=env_float(e, "NNP_CFENERGY", 0.0367493254),
        python_path=str(env_str(e, "LAMMPS_PYTHON_PATH", env_str(e, "LAMMPS_PYTHON", "")) or "").strip(),
        lib_dir=str(env_str(e, "LAMMPS_LIB_DIR", "") or "").strip(),
    )

    merge_mode = str(env_str(e, "BARRIER_MERGE_MODE", "global") or "global").strip().lower()
    if merge_mode not in {"global", "local", "mixed"}:
        merge_mode = "global"

    job_assignment_mode = str(env_str(e, "JOB_ASSIGNMENT_MODE", "cache_dedupe") or "cache_dedupe").strip().lower()
    if job_assignment_mode not in {"all_jobs", "cache_no_dedupe", "cache_dedupe", "cache_only"}:
        job_assignment_mode = "cache_dedupe"

    cache_schema = (
        f"v4_envkey:{environment.mode_schema}:{environment.env_key_mode}:"
        f"R{environment.radius_a}_PB{environment.pos_bin_a}_HB{environment.hop_bin_a}"
    )
    if site_map.source == "external":
        map_name = Path(site_map.site_map_file).expanduser().stem or "external_map"
        cache_schema = f"{cache_schema}:map:{map_name}"
    rank_cache_template = f"barrier_cache_rank{{rank}}_{cache_schema}.pkl"
    merged_cache_file = f"barrier_cache_{cache_schema}.pkl"

    cache = CacheConfig(
        schema=cache_schema,
        rank_cache_template=rank_cache_template,
        merged_cache_file=merged_cache_file,
        preload_file=str(
            env_str(
                e,
                "BARRIER_CACHE_FILE",
                env_str(e, "KMC_BARRIER_CACHE_FILE", env_str(e, "MASTER_BARRIER_CACHE_FILE", "")),
            )
            or ""
        ).strip(),
        merge_mode=merge_mode,
        merge_interval=max(1, env_int(e, "BARRIER_MERGE_INTERVAL", 100)),
        job_assignment_mode=job_assignment_mode,
        use_cache_for_jobs=(job_assignment_mode != "all_jobs"),
        dedupe_jobs=(job_assignment_mode == "cache_dedupe"),
    )

    restart_enabled = env_bool(e, "RESTART_MODE", env_bool(e, "KMC_RESTART_MODE", False))
    restart_dir = str(env_str(e, "RESTART_DIR", env_str(e, "KMC_RESTART_DIR", ".")) or ".").strip()
    restart_step_default = env_int(e, "KMC_RESTART_STEP", 0) if "KMC_RESTART_STEP" in e else 0
    restart_step = env_int(e, "RESTART_STEP", restart_step_default)
    restart = RestartConfig(
        enabled=restart_enabled,
        directory=restart_dir,
        step=max(0, restart_step),
        trajectory_file=str(
            env_str(
                e,
                "RESTART_TRAJECTORY_FILE",
                env_str(e, "KMC_RESTART_TRAJECTORY_FILE", "H_trajectory_onlyH.lammpstrj"),
            )
            or "H_trajectory_onlyH.lammpstrj"
        ).strip(),
        timestep_file=str(
            env_str(
                e,
                "RESTART_TIMESTEP_FILE",
                env_str(e, "KMC_RESTART_TIMESTEP_FILE", "diagnostics/kmc_timestep_vs_step.csv"),
            )
            or "diagnostics/kmc_timestep_vs_step.csv"
        ).strip(),
        diagnostics_file=str(
            env_str(
                e,
                "RESTART_DIAGNOSTICS_FILE",
                env_str(e, "KMC_RESTART_DIAGNOSTICS_FILE", "kmc_diagnostics_rank0.log"),
            )
            or "kmc_diagnostics_rank0.log"
        ).strip(),
        checkpoint_file=str(
            env_str(
                e,
                "RESTART_CHECKPOINT_FILE",
                env_str(e, "KMC_RESTART_CHECKPOINT_FILE", "kmc_restart_checkpoint.pkl"),
            )
            or "kmc_restart_checkpoint.pkl"
        ).strip(),
        strict=env_bool(e, "RESTART_STRICT", True),
    )

    return SimulationConfig(
        paths=paths,
        size=size,
        site_map=site_map,
        validation=validation,
        diagnostics=diagnostics,
        environment=environment,
        neb=neb,
        lammps=lammps,
        cache=cache,
        restart=restart,
    )


def config_summary(cfg: SimulationConfig) -> str:
    """Compact human-readable summary for rank-0 startup logging."""
    return "\n".join(
        [
            f"KMC: {cfg.size.nx}x{cfg.size.ny}x{cfg.size.nz}, num_H={cfg.size.num_h}, steps={cfg.size.steps}",
            f"Site map: source={cfg.site_map.source}, map={cfg.site_map.site_map_file or '<generated>'}",
            f"Environment: mode={cfg.environment.mode}, schema={cfg.cache.schema}",
            f"NEB: n_images={cfg.neb.n_images}, optimize_endpoints={cfg.neb.optimize_endpoints}, "
            f"native_endpoint_optimize={cfg.neb.native_endpoint_optimize}",
            f"LAMMPS: potential={cfg.lammps.potential}, engine={cfg.lammps.engine}",
            f"Cache: mode={cfg.cache.merge_mode}, job_assignment={cfg.cache.job_assignment_mode}",
            f"Diagnostics: debug={cfg.diagnostics.debug_logging}, neb_diag={cfg.diagnostics.neb_diag}, neb_dump={cfg.diagnostics.neb_dump}",
            f"Restart: enabled={cfg.restart.enabled}, dir={cfg.restart.directory}, step={cfg.restart.step}",
        ]
    )


if __name__ == "__main__":
    cfg = load_config()
    print(config_summary(cfg))
    print("\nFull resolved config:")
    print(cfg.to_json())
