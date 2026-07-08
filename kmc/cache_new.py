# MPI-based KMC with NEB for H diffusion in BCC Fe
# Coupled with LAMMPSlib to dynamically compute migration barriers

import json
import hashlib
import os
import re
from dataclasses import dataclass
from mpi4py import MPI # type: ignore
import numpy as np
from scipy.spatial import cKDTree
import random
import time
import csv
from ase import Atoms # type: ignore
from ase.build import make_supercell # type: ignore
from ase.optimize import BFGS # type: ignore
#from ase.mep import NEB # type: ignore # legacy
from ase.calculators.lammpslib import LAMMPSlib # type: ignore
from ase.io import write, read # type: ignore
from ase.io.lammpsdata import write_lammps_data # type: ignore
from collections import defaultdict
import pickle
import threading
import sys
from typing import Any, Dict, List, Optional
from itertools import count

try:
    from .neighbor_closeness import average_distance, compute_neighbor_distances
except (ImportError, ValueError):
    from neighbor_closeness import average_distance, compute_neighbor_distances


def _log_friendly(value):
    """Recursively convert numpy scalars/arrays for cleaner logging."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _log_friendly(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_log_friendly(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(_log_friendly(v) for v in value)
    return value
# -----------------------------------------------------------------------------
# Environment helpers shared by configuration inputs
# -----------------------------------------------------------------------------
_FALSES = {"0", "false", "False", "FALSE", "no", "No", "NO", "off", "OFF"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EAM_FILE = os.path.join(SCRIPT_DIR, "PotentialB3410-modified.fs")
DEFAULT_NNP_DIR = "/Users/rtirunelveli/LAMMPS/Input/NNP/POTENTIAL"

def _env_value(name: str, default: Any = None) -> Any:
    val = os.getenv(name)
    return default if val is None else val

def env_str(name: str, default: Any = None) -> Any:
    return _env_value(name, default)

def env_int(name: str, default: Any = None) -> int:
    val = _env_value(name, default)
    return int(val)

def env_float(name: str, default: Any = None) -> float:
    val = _env_value(name, default)
    return float(val)

def env_bool(name: str, default: Any = False) -> bool:
    val = os.getenv(name)
    if val is None:
        val = default
    if isinstance(val, bool):
        return val
    return str(val) not in _FALSES

def _env_flag_optional(name: str) -> Optional[bool]:
    val = os.getenv(name)
    if val is None:
        return None
    return val not in _FALSES

# Support both package and script execution modes
try:
    from .services.cache import BarrierCache  # type: ignore
    from .engines.lammps_neb import LammpsNEB  # type: ignore
except (ImportError, ValueError):
    from services.cache import BarrierCache
    from engines.lammps_neb import LammpsNEB
try:
    from .msd_tracker import MSDTracker  # type: ignore
except (ImportError, ValueError):
    from msd_tracker import MSDTracker  # type: ignore

# -----------------------------------------------------------------------------
# Inputs & global tags (environment-driven configuration)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvInputs:
    neb_dump: bool = env_bool("NEB_DUMP")
    validation_mode: bool = env_bool("VALIDATION_MODE")
    validation_out: str = env_str("VALIDATION_OUT", "validation_dump")
    validation_format: str = env_str("VALIDATION_FORMAT", "xyz").lower()
    validation_use_ase: bool = env_bool("VALIDATION_USE_ASE")
    validation_overwrite: bool = env_bool("VALIDATION_OVERWRITE", True)
    validation_ase_dump: bool = env_bool("VALIDATION_ASE_DUMP")
    validation_ase_dump_pre: bool = env_bool("VALIDATION_ASE_DUMP_PRE")
    validation_ase_dir: str = env_str("VALIDATION_ASE_DIR", "ase")
    ase_log_dir: str = env_str("ASE_LOG_DIR", "ase_logs")
    local_env_mode: str = env_str("LOCAL_ENV_MODE", "wrapped").lower()
    shell_inner_radius_a: float = env_float("SHELL_INNER_RADIUS_A", 5.0)
    shell_outer_radius_a: float = env_float("SHELL_OUTER_RADIUS_A", 6.0)
    build_local_env_only: bool = env_bool("BUILD_LOCAL_ENV_ONLY")
    build_local_env_count: int = env_int("BUILD_LOCAL_ENV_COUNT", 10)
    build_local_env_out: str = env_str("BUILD_LOCAL_ENV_OUT", "local_env_builds")
    build_local_env_selector: str = env_str("BUILD_LOCAL_ENV_SELECTOR", "random").lower()
    boundary_margin_a: float = env_float("BOUNDARY_MARGIN_A", 5.0)
    cluster_radius_a: float = env_float("CLUSTER_RADIUS_A", 5.0)
    env_key_mode: str = env_str("ENV_KEY_MODE", "env_plus_dir")
    env_radius_a: float = env_float("ENV_RADIUS_A", 5.0)
    pos_bin_a: float = env_float("POS_BIN_A", 0.10)
    hop_bin_a: float = env_float("HOP_BIN_A", 0.02)
    env_affect_radius_a: float = env_float("ENV_AFFECT_RADIUS_A", 6.0)
    barrier_merge_mode: str = env_str("BARRIER_MERGE_MODE", "global").lower()
    barrier_merge_interval: int = env_int("BARRIER_MERGE_INTERVAL", 100)
    neb_min_batch: int = env_int("NEB_MIN_BATCH", 64)
    neb_diag: bool = env_bool("NEB_DIAG", True)
    neb_job_trace: bool = env_bool("NEB_JOB_TRACE")
    neb_failure_capture: bool = env_bool("NEB_FAILURE_CAPTURE")
    neb_failure_dir: str = env_str("NEB_FAILURE_DIR", "neb_failure_capture")
    debug_logging: bool = env_bool("DEBUG_MODE", True)
    msd_log_interval: int = env_int("MSD_LOG_INTERVAL", 1000)
    msd_log_file: str = env_str("MSD_LOG_FILE", "msd_vs_time.csv")
    msd_plot_file: str = env_str("MSD_PLOT_FILE", "msd_plot.png")
    kmc_seed: Optional[str] = env_str("KMC_SEED", None)
    optimize_endpoints: bool = env_bool("OPTIMIZE_ENDPOINTS", True)
    use_octahedral_voids: bool = env_bool("USE_OCTAHEDRAL_VOIDS")
    job_assignment_mode: str = env_str("JOB_ASSIGNMENT_MODE", "cache_dedupe").lower()
    clump_eps_a: float = env_float("CLUMP_EPS_A", 0.05)
    fix_sublattice_h: Optional[bool] = _env_flag_optional("FIX_SUBLATTICE_H")
    fix_sublattice_fe: Optional[bool] = _env_flag_optional("FIX_SUBLATTICE_FE")
    num_h: int = env_int("NUM_H", 540)
    nimg: int = env_int("NIMG", 3)
    low_barrier_ev: float = env_float("LOW_BARRIER_EV", 1e-3)
    dump_every_steps: int = env_int("DUMP_EVERY_STEPS", 1)
    potential: str = env_str("POTENTIAL", "EAM")
    potential_eam_file: str = env_str("POTENTIAL_EAM_FILE", DEFAULT_EAM_FILE)
    potential_nnp_dir: str = env_str("POTENTIAL_NNP_DIR", DEFAULT_NNP_DIR)
    lammps_engine: str = env_str("LAMMPS_ENGINE", "eam_fs").lower()
    lammps_pair_style: str = env_str("LAMMPS_PAIR_STYLE", "")
    lammps_pair_coeff: str = env_str("LAMMPS_PAIR_COEFF", "")
    lammps_files: str = env_str("LAMMPS_FILES", DEFAULT_EAM_FILE)
    nnp_dir: str = env_str("NNP_DIR", DEFAULT_NNP_DIR)
    nnp_elements: str = env_str("NNP_ELEMENTS", "Fe H")
    nnp_cutoff: float = env_float("NNP_CUTOFF", 6.60)
    nnp_showewsum: int = env_int("NNP_SHOWEWSUM", 1000)
    nnp_maxew: int = env_int("NNP_MAXEW", 1000000)
    nnp_cflength: float = env_float("NNP_CFLENGTH", 1.8897261328)
    nnp_cfenergy: float = env_float("NNP_CFENERGY", 0.0367493254)

INPUTS = EnvInputs()

# -----------------------------------------------------------------------------
# MPI initialization and runtime configuration
# -----------------------------------------------------------------------------
# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
os.environ["ASE_RANK"] = str(rank)

# DEBUG / modes
KMC_SEED = 42

# --- Validation, logging, and cache inputs ---
NEB_DUMP = INPUTS.neb_dump
VALIDATION_MODE = INPUTS.validation_mode
VALIDATION_OUT = INPUTS.validation_out
# 'xyz' is fast and robust; set to 'lammps' if you explicitly want initial.data
VALIDATION_FORMAT = INPUTS.validation_format
VALIDATION_USE_ASE = INPUTS.validation_use_ase
VALIDATION_OVERWRITE = INPUTS.validation_overwrite
VALIDATION_ASE_DUMP = INPUTS.validation_ase_dump
VALIDATION_ASE_DUMP_PRE = INPUTS.validation_ase_dump_pre
VALIDATION_ASE_DIR = INPUTS.validation_ase_dir
ASE_LOG_DIR = INPUTS.ase_log_dir
DUMP_EVERY_STEPS = max(1, int(INPUTS.dump_every_steps))
POTENTIAL = (INPUTS.potential or "EAM").strip().upper()
POTENTIAL_EAM_FILE = (INPUTS.potential_eam_file or DEFAULT_EAM_FILE).strip()
POTENTIAL_NNP_DIR = (INPUTS.potential_nnp_dir or DEFAULT_NNP_DIR).strip()
LAMMPS_ENGINE = (INPUTS.lammps_engine or "eam_fs").strip().lower()
LAMMPS_PAIR_STYLE = (INPUTS.lammps_pair_style or "").strip()
LAMMPS_PAIR_COEFF = (INPUTS.lammps_pair_coeff or "").strip()
LAMMPS_FILES = INPUTS.lammps_files
NNP_DIR = INPUTS.nnp_dir
NNP_ELEMENTS = (INPUTS.nnp_elements or "Fe H").strip()
NNP_CUTOFF = INPUTS.nnp_cutoff
NNP_SHOWEWSUM = int(INPUTS.nnp_showewsum)
NNP_MAXEW = int(INPUTS.nnp_maxew)
NNP_CFLENGTH = INPUTS.nnp_cflength
NNP_CFENERGY = INPUTS.nnp_cfenergy

if POTENTIAL_EAM_FILE and not os.path.isabs(POTENTIAL_EAM_FILE):
    POTENTIAL_EAM_FILE = os.path.join(SCRIPT_DIR, POTENTIAL_EAM_FILE)
if POTENTIAL_NNP_DIR and not os.path.isabs(POTENTIAL_NNP_DIR):
    POTENTIAL_NNP_DIR = os.path.join(SCRIPT_DIR, POTENTIAL_NNP_DIR)

if "POTENTIAL" not in os.environ and LAMMPS_ENGINE in {"n2p2", "hdnnp", "many_body", "manybody"}:
    POTENTIAL = "NNP"

if POTENTIAL not in {"EAM", "NNP"}:
    if rank == 0:
        print(f"[Rank 0] Unsupported POTENTIAL='{POTENTIAL}'. Falling back to POTENTIAL='EAM'.")
    POTENTIAL = "EAM"

# Keep NNP_DIR aligned with POTENTIAL_NNP_DIR unless explicitly overridden.
if "NNP_DIR" not in os.environ:
    NNP_DIR = POTENTIAL_NNP_DIR

VALIDATION_MODE = bool(comm.bcast(int(VALIDATION_MODE) if rank == 0 else None, root=0))
DUMP_EVERY_STEPS = int(comm.bcast(DUMP_EVERY_STEPS if rank == 0 else None, root=0))

# Validation ASE band dumps are expensive and have repeatedly destabilized
# collectives in multi-rank validation runs. Keep them disabled there.
if VALIDATION_MODE and size > 1:
    if rank == 0 and (VALIDATION_ASE_DUMP or VALIDATION_ASE_DUMP_PRE):
        print("[Rank 0] Forcing VALIDATION_ASE_DUMP=0 and VALIDATION_ASE_DUMP_PRE=0 for multi-rank validation.")
    VALIDATION_ASE_DUMP = False
    VALIDATION_ASE_DUMP_PRE = False

ENV_KEY_MODE = INPUTS.env_key_mode  # "env_plus_dir" | "env_only"
ENV_RADIUS_A = INPUTS.env_radius_a  # neighbor search radius (Å)
POS_BIN_A = INPUTS.pos_bin_a        # quantization for neighbor positions (Å)
HOP_BIN_A = INPUTS.hop_bin_a        # quantization for hop vector (Å)

# --- Stage-1.5 controls (affect radius & NEB batching) ---
# Use a slightly larger "affect" radius for deciding which H need recomputation.
# This is intentionally decoupled from ENV_RADIUS_A to simplify reasoning.
ENV_AFFECT_RADIUS_A = INPUTS.env_affect_radius_a
BARRIER_MERGE_MODE = (INPUTS.barrier_merge_mode or "global").strip().lower()
BARRIER_MERGE_INTERVAL = max(1, int(INPUTS.barrier_merge_interval))
# Minimum NEB batch size to avoid fragmented scheduling (tune per hardware).
NEB_MIN_BATCH = INPUTS.neb_min_batch

if BARRIER_MERGE_MODE not in {"global", "local", "mixed"}:
    BARRIER_MERGE_MODE = "global"

# --- NEB diagnostics toggle (enabled by default) ---
NEB_DIAG = INPUTS.neb_diag
NEB_DIAG_ALL_PATH = "neb_diag_all.csv"  # aggregated output written by rank 0
# Per-job trace (off by default; can be large)
NEB_JOB_TRACE = INPUTS.neb_job_trace
# Dump NEB failure context (initial/final + metadata) when enabled.
NEB_FAILURE_CAPTURE = INPUTS.neb_failure_capture
NEB_FAILURE_DIR = INPUTS.neb_failure_dir

# Optional overrides so H and Fe can be toggled independently.
FIX_SUBLATTICE_H = INPUTS.fix_sublattice_h
FIX_SUBLATTICE_FE = INPUTS.fix_sublattice_fe

# --- Master debug toggle (controls rate/timing outputs & aux logs) ---
DEBUG_LOGGING = INPUTS.debug_logging
if not DEBUG_LOGGING:
    NEB_DIAG = False
    NEB_JOB_TRACE = False

MSD_LOG_INTERVAL = INPUTS.msd_log_interval
MSD_LOG_FILE = INPUTS.msd_log_file
MSD_PLOT_FILE = INPUTS.msd_plot_file
LOCAL_ENV_MODE = (INPUTS.local_env_mode or "wrapped").strip().lower()
if LOCAL_ENV_MODE not in {"wrapped", "radial", "shell"}:
    LOCAL_ENV_MODE = "wrapped"
SHELL_INNER_RADIUS_A = max(0.0, float(INPUTS.shell_inner_radius_a))
SHELL_OUTER_RADIUS_A = max(SHELL_INNER_RADIUS_A, float(INPUTS.shell_outer_radius_a))
if LOCAL_ENV_MODE == "shell":
    _env_mode_schema = (
        f"mode:shell_IR{SHELL_INNER_RADIUS_A}_OR{SHELL_OUTER_RADIUS_A}"
    )
elif LOCAL_ENV_MODE == "radial":
    _env_mode_schema = "mode:radial"
else:
    _env_mode_schema = "mode:wrapped"
CACHE_SCHEMA = (
    f"v4_envkey:{_env_mode_schema}:{ENV_KEY_MODE}:"
    f"R{ENV_RADIUS_A}_PB{POS_BIN_A}_HB{HOP_BIN_A}"
)
BUILD_LOCAL_ENV_ONLY = bool(INPUTS.build_local_env_only)
BUILD_LOCAL_ENV_COUNT = max(1, int(INPUTS.build_local_env_count))
BUILD_LOCAL_ENV_OUT = (INPUTS.build_local_env_out or "local_env_builds").strip() or "local_env_builds"
BUILD_LOCAL_ENV_SELECTOR = (INPUTS.build_local_env_selector or "random").strip().lower()
if BUILD_LOCAL_ENV_SELECTOR not in {"random", "boundary", "cluster"}:
    BUILD_LOCAL_ENV_SELECTOR = "random"
BOUNDARY_MARGIN_A = max(0.0, float(INPUTS.boundary_margin_a))
CLUSTER_RADIUS_A = max(0.0, float(INPUTS.cluster_radius_a))

H_positions: Optional[np.ndarray] = None
H_indices: Optional[np.ndarray] = None
H_unwrapped_positions: Optional[np.ndarray] = None
H_initial_unwrapped_positions: Optional[np.ndarray] = None
H_cell_index: Optional[Dict[tuple[int, int, int], List[tuple[np.ndarray, Optional[int]]]]] = None

_MPI_SEQ_COUNTER = count()
_ASE_LOG_SEQ = count()
_STRUCT_TRAJ_SEQ = count()
_STRUCT_SNAP_SEQ = count()
_LMP_ENGINE_LOGGED = False

_env_seed = INPUTS.kmc_seed
if _env_seed is not None:
    KMC_SEED = _env_seed
    
_PHASE_COUNTER = 0


def _build_h_cell_index(
    positions: Optional[np.ndarray],
    indices: Optional[np.ndarray],
    lattice_a: float,
) -> Dict[tuple[int, int, int], List[tuple[np.ndarray, Optional[int]]]]:
    cell_index: Dict[tuple[int, int, int], List[tuple[np.ndarray, Optional[int]]]] = defaultdict(list)
    if positions is None or len(positions) == 0:
        return cell_index

    pos_array = np.asarray(positions, dtype=float)
    idx_array = None if indices is None else np.asarray(indices, dtype=int)
    for idx_occ, pos_abs in enumerate(pos_array):
        if pos_abs.shape != (3,) or not np.isfinite(pos_abs).all():
            continue
        cx, cy, cz = np.floor(pos_abs / lattice_a + 1e-12).astype(int)
        key = (int(cx % nx), int(cy % ny), int(cz % nz))
        neighbor_idx = None
        if idx_array is not None and idx_occ < len(idx_array):
            neighbor_idx = int(idx_array[idx_occ])
        cell_index[key].append((np.array(pos_abs, dtype=float), neighbor_idx))
    return cell_index

def phase_guard(comm, rank, step, name):
    """
    Ensures all ranks enter collectives in the exact same order.
    If collectives get crossed, we abort with a durable breadcrumb file.
    """
    global _PHASE_COUNTER

    local_step = int(step)
    local_name = str(name)
    if rank == 0:
        _PHASE_COUNTER += 1
    comm.Barrier()
    if rank == 0 and DEBUG_LOGGING:
        try:
            log(f"[Rank {rank}] phase_guard seq={_PHASE_COUNTER}, step={local_step}, name={local_name}")
        except Exception:
            pass

OPTIMIZE_ENDPOINTS = INPUTS.optimize_endpoints
USE_OCTAHEDRAL_VOIDS = INPUTS.use_octahedral_voids

# Job assignment modes:
#   * `all_jobs`: run every NEB candidate (skips cache lookups and deduplication).
#   * `cache_no_dedupe`: honor cache hits but send all cache misses (no global dedupe).
#   * `cache_dedupe`: current behavior (default) – cache + dedupe.
JOB_ASSIGNMENT_MODE = INPUTS.job_assignment_mode
JOB_ASSIGNMENT_MODES = {
    "all_jobs": "Run every (h→n) NEB each step without caching or job deduplication.",
    "cache_no_dedupe": "Use the barrier cache but assign every cache miss (no dedup).",
    "cache_dedupe": "Use the barrier cache and deduplicate NEB jobs globally (default).",
}
if JOB_ASSIGNMENT_MODE not in JOB_ASSIGNMENT_MODES:
    JOB_ASSIGNMENT_MODE = "cache_dedupe"

if rank == 0:
    if BARRIER_MERGE_MODE == "global":
        merge_mode_desc = "broadcast merged results to all ranks every step"
    elif BARRIER_MERGE_MODE == "local":
        merge_mode_desc = "return results only to source ranks"
    else:
        merge_mode_desc = (
            f"return results to source ranks and globally share accumulated cache entries "
            f"every {BARRIER_MERGE_INTERVAL} step(s)"
        )
    print(
        f"[Rank 0] Barrier merge mode: {BARRIER_MERGE_MODE} "
        f"({merge_mode_desc})"
    )
    
# --- Diagnostics: absolute clumping threshold (Å) ---
CLUMP_EPS_A = INPUTS.clump_eps_a  # 0.05 Å by default
    
# -----------------------------------------------------------------------------
# Environment key construction and random seed helpers
# -----------------------------------------------------------------------------
# <editable:envkey>
def make_env_key(rel_initial, rel_final, H_tree, H_positions, box, *,
                 radius_A, pos_bin_A, hop_bin_A, mode="env_plus_dir"):
    """
    Build a cache key invariant to translation and atom ordering.

    - rel_initial, rel_final: absolute positions (Å) of moving H sites
    - H_tree: cKDTree built on H_positions (absolute)
    - H_positions: list/array of absolute H positions (Å)
    - box: np.array([Lx, Ly, Lz]) (Å)
    - radius_A: neighbor cutoff around rel_initial (Å)
    - pos_bin_A: bin size for neighbor displacement quantization (Å)
    - hop_bin_A: bin size for hop vector quantization (Å)
    - mode: "env_plus_dir" (include hop direction) or "env_only"

    Returns: a hashable tuple key
    """
    rel_initial = _ensure_numeric_coords(rel_initial, "rel_initial in env key")
    rel_final = _ensure_numeric_coords(rel_final, "rel_final in env key")
    # neighbors around initial H
    nbr_idx = H_tree.query_ball_point(rel_initial, r=radius_A)
    # Build displacement set excluding self
    disp = []
    for i in nbr_idx:
        neighbor_pos = _ensure_numeric_coords(H_positions[i], f"H_positions[{i}] in env key")
        d = pbc_diff(rel_initial, neighbor_pos, box)
        if np.allclose(d, 0.0, atol=0.2):
            continue  # skip central H
        q = tuple(np.round(d / pos_bin_A).astype(int))
        disp.append(q)

    # sort to be permutation invariant
    disp_sorted = tuple(sorted(disp))

    if mode == "env_only":
        return ("ENVONLY", pos_bin_A, radius_A, disp_sorted)

    # also encode hop vector (MIC) quantized
    hop = pbc_diff(rel_initial, rel_final, box)
    hop_q = tuple(np.round(hop / hop_bin_A).astype(int))
    return ("ENV+DIR", pos_bin_A, radius_A, hop_bin_A, disp_sorted, hop_q)

# --- Fast-path builders to avoid recomputing neighbor env for each hop ---
def _compute_disp_sorted(rel_initial, H_tree, H_positions, box, *, radius_A, pos_bin_A):
    """Return the sorted, quantized neighbor displacement tuple for rel_initial."""
    rel_initial = _ensure_numeric_coords(rel_initial, "rel_initial for disp sorted")
    nbr_idx = H_tree.query_ball_point(rel_initial, r=radius_A)
    disp = []
    for i in nbr_idx:
        neighbor_pos = _ensure_numeric_coords(H_positions[i], f"H_positions[{i}] for disp sorted")
        d = pbc_diff(rel_initial, neighbor_pos, box)
        if np.allclose(d, 0.0, atol=1e-8):
            continue
        disp.append(tuple(np.round(d / pos_bin_A).astype(int)))
    return tuple(sorted(disp))

def make_env_key_from_disp(rel_initial, rel_final, box, *, disp_sorted, hop_bin_A, mode="env_plus_dir"):
    """Build env key using precomputed neighbor displacement (disp_sorted)."""
    if mode == "env_only":
        # Keep identical structure/fields to existing keys
        return ("ENVONLY", POS_BIN_A, ENV_RADIUS_A, disp_sorted)
    hop = pbc_diff(rel_initial, rel_final, box)
    hop_q = tuple(np.round(hop / hop_bin_A).astype(int))
    return ("ENV+DIR", POS_BIN_A, ENV_RADIUS_A, hop_bin_A, disp_sorted, hop_q)
# </editable:envkey>

def _init_seeds(user_seed, rank):
    if user_seed in (None, "", "None"):
        # Per-run, per-rank random seed from OS entropy + time + rank
        entropy = (int.from_bytes(os.urandom(8), "little")
                   ^ time.time_ns()
                   ^ (rank * 0x9E3779B97F4A7C15)) & 0xFFFFFFFF
        np.random.seed(entropy)
        random.seed(entropy)
        if rank == 0:
            print(f"[Seed] No KMC_SEED set -> random seeding (rank 0 seed = {entropy})")
    else:
        # Deterministic: base + rank
        try:
            base = int(user_seed)
        except ValueError:
            base = abs(hash(str(user_seed))) & 0xFFFFFFFF
        seed = (base + rank) & 0xFFFFFFFF
        np.random.seed(seed)
        random.seed(seed)
        if rank == 0:
            print(f"[Seed] Using deterministic KMC_SEED={base} (rank seed = base + rank)")

_init_seeds(KMC_SEED, rank)
# -----------------------------------------------------------------------------
# Global simulation constants
# -----------------------------------------------------------------------------
# Constants
a = 2.8601  # lattice parameter
nx, ny, nz = 30, 30, 30
0  # super-cell size
cutoff = 2  # used in neighbour creation
num_H = 540 # number of hydrogen in the supercell
steps = 100 # kMC steps
k_B = 8.617e-5  # eV/K
T = 300  # Temperature in K
nu = 1e13  # Attempt frequency in Hz
NIMG = INPUTS.nimg
num_H = INPUTS.num_h

# --- Low-barrier diagnostics config ---
LOW_BARRIER_EV = INPUTS.low_barrier_ev  # threshold to trigger dump

# -----------------------------------------------------------------------------
# Distributed job utilities and general helpers
# -----------------------------------------------------------------------------
def dedupe_jobs(job_list):
    """job_list: list of (key, payload). key is hashable."""
    seen = set()
    out = []
    for k, p in job_list:
        if k not in seen:
            seen.add(k); out.append((k, p))
    return out

def assign_jobs_round_robin(all_jobs, n_ranks):
    """Return dict: rank -> list[(key,payload), ...]."""
    assigns = {r: [] for r in range(n_ranks)}
    for i, job in enumerate(all_jobs):
        assigns[i % n_ranks].append(job)
    return assigns

def _vec_norm(v): 
    return float(np.linalg.norm(v))

def dump_neb_images_lammpstrj(images, basename):
    """Write a NEB image list to basename+'.traj' (ASE trajectory)."""
    if not images:
        return None
    out = f"{basename}.traj"
    write_neb_lammpstrj(images, out)
    return out

def compare_endpoints_report(initial_pre, final_pre, initial_post, final_post,
                             mover_idx, cell, rank, step, h, n, b, rate):
    """
    Simple logic checks for endpoint equality / displacement pre vs post NEB.
    Logs a compact report and returns file paths to pre/post ASE trajectory bands.
    """
    # Moving-H positions
    pi_pre = initial_pre[mover_idx].position
    pf_pre = final_pre[mover_idx].position
    pi_post = initial_post[mover_idx].position
    pf_post = final_post[mover_idx].position

    # Displacements under PBC of the 3a cell used in NEB
    dr_pre  = pbc_diff(pi_pre,  pf_pre,  cell)
    dr_post = pbc_diff(pi_post, pf_post, cell)

    # Did pre/post endpoints move internally during BFGS?
    di_ip = pbc_diff(pi_pre,  pi_post,  cell)   # initial (pre) -> initial (post)
    di_fp = pbc_diff(pf_pre,  pf_post,  cell)   # final (pre)   -> final (post)

    log(f"[Rank {rank}] LOW-BARRIER DIAG step={step} move {h}->{n}: "
        f"barrier={b:.6e} eV, rate={rate:.6e} Hz", step)
    log(f"[Rank {rank}]   |Δr_pre|  = {_vec_norm(dr_pre):.3e} Å ; "
        f"|Δr_post| = {_vec_norm(dr_post):.3e} Å", step)
    log(f"[Rank {rank}]   |Δinitial(pre→post)| = {_vec_norm(di_ip):.3e} Å ; "
        f"|Δfinal(pre→post)|    = {_vec_norm(di_fp):.3e} Å", step)

    # Return nothing; file writes are handled by the caller

# -----------------------------------------------------------------------------
# Logging and timing helpers
# -----------------------------------------------------------------------------
# <editable:logging>
# Logging function
def log(msg, step=None, banner=False):  # Log function that primarily builds rank logs.
    log_filename = f"log_rank{rank}.txt"
    if DEBUG_LOGGING or rank == 0:
        with open(log_filename, "a") as f:
            if banner and step is not None:
                f.write(f"\n========== STEP {step} ==========\n")  # First step print
            elif banner:
                f.write("\n" + "=" * 40 + "\n")  # Consecutive step partition
            f.write(msg + "\n")  # writes the message in the logfile
    if rank == 0:    
        print(msg)  # prints the same msg in terminal

def _ensure_numeric_coords(value, label="atom coordinates", step=None):
    """
    Convert an input to a float ndarray; exit if conversion fails (non-numeric data).
    """
    try:
        coords = np.asarray(value, dtype=float)
    except Exception as exc:
        log(f"[Rank {rank}] ERROR: {label} contains non-numeric atom coordinates: {exc}", step)
        sys.exit(1)
    if not np.isfinite(coords).all():
        invalid = coords[~np.isfinite(coords)]
        invalid_list = invalid.tolist()
        log(
            f"[Rank {rank}] ERROR: {label} contains non-finite atom coordinates {invalid_list}; "
            f"raw value={_log_friendly(value)}",
            step,
        )
        sys.exit(1)
    return coords


def _neighbor_entry_to_pos(entry, context):
    """
    Normalize a neighbor-position entry (array / tuple / dict) to (numpy coords, index).
    """
    if entry is None:
        return None, None
    idx = None
    pos = entry
    if isinstance(entry, dict):
        pos = entry.get("pos")
        idx = entry.get("index")
    elif isinstance(entry, (list, tuple)):
        if not entry:
            return None, None
        pos = entry[0]
        if len(entry) > 1:
            idx = entry[1]
    arr = _ensure_numeric_coords(pos, context)
    return arr, idx


TIMING_BUCKET_FIELDS = [
    "build_struct",
    "build_H_tree",
    "move_loop",
    "batch_extract_env",
    "cache_lookup",
    "barrier_neb",
    "selection_comm",
    "selection_cpu",
    "logging",
    "extract_env",
    "validation",
    "job_comm",
    "job_schedule",
    "neb_struct",
    "result_comm",
    "result_merge",
    "move_reduce",
    "cache_update",
    "cache_save",
    "rate_comm",
    "rate_io",
    "state_comm",
    "state_update",
    "dump_diag",
    "dump_traj",
    "timing_append",
]


def log_step_timing(step, rank, times_dict, total_wall=None):
    """
    times_dict: dict with keys below (seconds)
    total_wall: optional wall time for the step; if None, sum(times_dict.values()) is used
    """
    order = TIMING_BUCKET_FIELDS
    colw = 14  # column width to mimic your barrier table look

    # make sure all keys exist
    for k in order:
        times_dict.setdefault(k, 0.0)

    total = total_wall if total_wall is not None else sum(times_dict[k] for k in order)
    # compute row strings
    header = "Stage:       " + " ".join(f"{k:<{colw}}" for k in order)
    times  = "Time (s):    " + " ".join(f"{times_dict[k]:<{colw}.2f}" for k in order)
    perc   = "Percent:     " + " ".join(
        f"{(times_dict[k]/total*100 if total>0 else 0):<{colw}.1f}%" for k in order
    )

    log(f"========== STEP {step} TIMING ==========", banner=True)
    log(f"Total: {total:.2f} s")
    log(header)
    log(times)
    log(perc)

    # (optional) show any unaccounted time vs wall clock
    accounted = sum(times_dict[k] for k in order)
    if total_wall is not None:
        other = total_wall - accounted
        if abs(other) > 1e-3:
            log(f"Unaccounted/other: {other:.2f} s")
# </editable:logging>

# -----------------------------------------------------------------------------
# Barrier cache management helpers
# -----------------------------------------------------------------------------
# <editable:cache>
# Barrier cache per rank
cache_filename = f"barrier_cache_rank{rank}_{CACHE_SCHEMA}.pkl"
merged_fname  = f"barrier_cache_{CACHE_SCHEMA}.pkl"
preloaded_cache = None
if os.path.exists(merged_fname):
    with open(merged_fname, "rb") as f:
        preloaded_cache = pickle.load(f)
    log(f"[Rank {rank}] Loaded merged barrier cache ({CACHE_SCHEMA}) with {len(preloaded_cache)} entries")
cache = BarrierCache(cache_filename, initial_store=preloaded_cache)
# Lazy NEB engine reference; instantiated on first use with the active calc_pool.
_neb_engine: Optional[LammpsNEB] = None
# </editable:cache>

# -----------------------------------------------------------------------------
# Lattice and structural utilities
# -----------------------------------------------------------------------------
# Site generation
def get_tetrahedral_sites(a,):  # mulitply lattice parameter with all possible tetrahedral sites in bcc
    return a * np.array(
        [
            [0.25, 0.5, 0.0],
            [0.75, 0.5, 0.0],
            [0.5, 0.25, 0.0],
            [0.5, 0.75, 0.0],
            [0.25, 0.0, 0.5],
            [0.75, 0.0, 0.5],
            [0.5, 0.0, 0.25],
            [0.5, 0.0, 0.75],
            [0.0, 0.25, 0.5],
            [0.0, 0.75, 0.5],
            [0.0, 0.5, 0.25],
            [0.0, 0.5, 0.75],
        ]
    )

def get_octahedral_sites(
    a,
):  # mulitply lattice parameter with all possible octahedral sites in bcc
    return a * np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

def pbc_diff(pos_i, pos_j, box):
    delta = pos_j - pos_i  # Computes the naive difference between the two positions
    return delta - box * np.round(
        delta / box
    )  # gives us the shortest vector distance, np.round(delta / box) gives us the orientation

def dump_neb_failure_debug(h, n, initial, final, rank, step):
    try:
        fname_prefix = f"debug_rank{rank}_step{step}_h{h}_n{n}"
        write(f"{fname_prefix}_initial.xyz", initial)
        write(f"{fname_prefix}_final.xyz", final)
        log(f"[Rank {rank}] Wrote debug NEB XYZ files for move {h}->{n}", step)
    except Exception as e:
        log(f"[Rank {rank}] Failed to write debug structures: {e}", step)

def generate_interstitial_sites(a, nx, ny, nz):
    # Vectorized generation of tetrahedral sites (octahedral sites optional)
    tet = get_tetrahedral_sites(a)
    if USE_OCTAHEDRAL_VOIDS:
        octa = get_octahedral_sites(a)
        unit = np.vstack([tet, octa])
    else:
        unit = tet

    grid = np.indices((nx, ny, nz)).reshape(3, -1).T * a
    all_sites = (unit[:, np.newaxis, :] + grid).reshape(-1, 3)
    return all_sites

def get_orthogonal_bcc_fe_supercell(a, nx, ny, nz):
    basis = Atoms(
        "Fe2", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[a, a, a], pbc=True
    )
    P = np.diag([nx, ny, nz])
    supercell = make_supercell(basis, P)
    supercell.set_cell([a * nx, a * ny, a * nz])
    supercell.set_pbc(True)
    return supercell

def build_full_structure(sites, H_indices):
    valid_indices = [i for i in H_indices if i < len(sites)]
    Fe_atoms = get_orthogonal_bcc_fe_supercell(a, nx, ny, nz)
    H_positions = sites[valid_indices]
    H_atoms = Atoms("H" * len(H_positions), positions=H_positions)
    return Fe_atoms + H_atoms


def update_full_structure_h_atom(full_structure, sites, h_slot, new_site_idx):
    """
    Update the moved H atom in-place inside the cached full structure.

    `build_full_structure()` appends all H atoms after the Fe lattice in the
    same order as `H_indices`, so the slot index in H_indices maps directly to
    the H atom offset in the combined Atoms object.
    """
    if full_structure is None:
        return
    h_slot = int(h_slot)
    new_site_idx = int(new_site_idx)
    h_count = int(sum(1 for s in full_structure.get_chemical_symbols() if s == "H"))
    h_offset = len(full_structure) - h_count
    atom_idx = h_offset + h_slot
    if atom_idx < 0 or atom_idx >= len(full_structure):
        raise IndexError(
            f"H atom slot {h_slot} maps to invalid atom index {atom_idx} "
            f"(len={len(full_structure)}, h_offset={h_offset})"
        )
    full_structure[atom_idx].position = _ensure_numeric_coords(
        sites[new_site_idx], f"site[{new_site_idx}] for full_structure update"
    )


def select_h_indices_for_test_mode(
    sites,
    num_h,
    *,
    selector,
    box,
    boundary_margin_a,
    cluster_radius_a,
):
    sites = np.asarray(sites, dtype=float)
    total_sites = len(sites)
    if total_sites == 0:
        return np.array([], dtype=int)
    num_h = min(int(num_h), total_sites)
    if num_h <= 0:
        return np.array([], dtype=int)

    if selector == "random":
        return np.random.choice(total_sites, size=num_h, replace=False)

    box = np.asarray(box, dtype=float)
    if selector == "boundary":
        near_boundary = np.any((sites <= boundary_margin_a) | ((box - sites) <= boundary_margin_a), axis=1)
        candidate_idx = np.flatnonzero(near_boundary)
        if len(candidate_idx) < num_h:
            raise RuntimeError(
                f"BUILD_LOCAL_ENV_SELECTOR='boundary' found only {len(candidate_idx)} candidate sites "
                f"within {boundary_margin_a:.2f} A of a boundary; need {num_h}."
            )
        chosen = np.random.choice(candidate_idx, size=num_h, replace=False)
        return np.asarray(chosen, dtype=int)

    if selector == "cluster":
        center_idx = int(np.random.randint(total_sites))
        center = sites[center_idx]
        disp = sites - center
        disp -= box * np.round(disp / box)
        dist = np.linalg.norm(disp, axis=1)
        candidate_idx = np.flatnonzero(dist <= cluster_radius_a + 1e-12)
        if len(candidate_idx) < num_h:
            raise RuntimeError(
                f"BUILD_LOCAL_ENV_SELECTOR='cluster' found only {len(candidate_idx)} candidate sites "
                f"within {cluster_radius_a:.2f} A of the seed site; need {num_h}."
            )
        if center_idx not in candidate_idx:
            candidate_idx = np.concatenate(([center_idx], candidate_idx))
            candidate_idx = np.unique(candidate_idx)
        if len(candidate_idx) == num_h:
            return np.asarray(candidate_idx, dtype=int)
        others = candidate_idx[candidate_idx != center_idx]
        picked_others = np.random.choice(others, size=num_h - 1, replace=False)
        chosen = np.concatenate(([center_idx], picked_others))
        return np.asarray(chosen, dtype=int)

    raise RuntimeError(f"Unknown BUILD_LOCAL_ENV_SELECTOR='{selector}'")

def assign_mass_and_type(atoms):
    type_map = {"Fe": 1, "H": 2}
    mass_map = {"Fe": 55.845, "H": 1.008}
    symbols = atoms.get_chemical_symbols()
    types = np.array([type_map[s] for s in symbols])
    ids = np.arange(1, len(symbols) + 1)
    masses = np.array([mass_map[s] for s in symbols])

    atoms.set_atomic_numbers([26 if s == "Fe" else 1 for s in symbols])
    atoms.set_array("mass", masses)
    atoms.set_array("id", ids)
    atoms.set_array("type", types)

def print_atoms_info(atoms, tag=""):
    log(f"--- {tag} ATOMS INFO ---")

    # True order of arrays
    for key in atoms.arrays:
        log(f"{key.capitalize()} array: {atoms.arrays[key]}")

# --- Validation helpers ------------------------------------------------------
def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]")

def _sanitize_filename_component(value):
    text = str(value)
    if not text:
        return ""
    return _FILENAME_SANITIZER.sub("_", text)

def _ase_log_dir():
    if not DEBUG_LOGGING:
        return None
    log_dir = (ASE_LOG_DIR or "ase_logs").strip()
    if not log_dir:
        log_dir = "ase_logs"
    _ensure_dir(log_dir)
    return log_dir

def _ase_log_path(base_name=None, *, rank=None, job_context=None):
    dir_path = _ase_log_dir()
    if not dir_path:
        return None
    base_root = os.path.splitext(os.path.basename(base_name or "ase"))[0]
    if not base_root:
        base_root = "ase"
    parts = [base_root]
    if job_context:
        for tag in ("step", "h", "n", "src_rank"):
            value = job_context.get(tag)
            if value is not None:
                sanitized = _sanitize_filename_component(value)
                if sanitized:
                    parts.append(f"{tag}{sanitized}")
    if rank is not None:
        parts.append(f"rank{rank}")
    seq = next(_ASE_LOG_SEQ)
    parts.append(f"seq{seq}")
    filename = "_".join(part for part in parts if part)
    if not filename.endswith(".log"):
        filename = f"{filename}.log"
    return os.path.join(dir_path, filename)


def _structure_traj_path(*, job_context=None):
    dir_path = _ase_log_dir()
    if not dir_path:
        return None
    parts = ["structure"]
    if job_context:
        for tag in ("step", "h", "n", "src_rank"):
            value = job_context.get(tag)
            if value is not None:
                sanitized = _sanitize_filename_component(value)
                if sanitized:
                    parts.append(f"{tag}{sanitized}")
    seq = next(_STRUCT_TRAJ_SEQ)
    parts.append(f"seq{seq}")
    filename = "_".join(part for part in parts if part)
    if not filename.endswith(".lammpstrj"):
        filename = f"{filename}.lammpstrj"
    return os.path.join(dir_path, filename)


def _structure_snapshot_path(*, job_context=None):
    dir_path = _ase_log_dir()
    if not dir_path:
        return None
    parts = ["structure_snapshot"]
    if job_context:
        for tag in ("step", "h", "n", "src_rank"):
            value = job_context.get(tag)
            if value is not None:
                sanitized = _sanitize_filename_component(value)
                if sanitized:
                    parts.append(f"{tag}{sanitized}")
    seq = next(_STRUCT_SNAP_SEQ)
    parts.append(f"seq{seq}")
    filename = "_".join(part for part in parts if part)
    if not filename.endswith(".lmp"):
        filename = f"{filename}.lmp"
    return os.path.join(dir_path, filename)


def _tag_neb_frame(atoms: Atoms, mover_idx: Optional[int]) -> Atoms:
    tagged = atoms.copy()
    tagged.wrap()
    n_atoms = len(tagged)
    types = np.ones(n_atoms, dtype=int)
    masses = np.ones(n_atoms, dtype=float)
    ids = np.arange(1, n_atoms + 1, dtype=int)
    Z = tagged.get_atomic_numbers()
    for idx in range(n_atoms):
        Z_val = Z[idx]
        if Z_val == 26:
            types[idx] = 1
            masses[idx] = 55.845
        else:
            masses[idx] = 1.008
            if mover_idx is not None and idx == mover_idx:
                types[idx] = 2
            else:
                types[idx] = 3
    tagged.set_array("type", types)
    tagged.set_array("mass", masses)
    tagged.set_array("id", ids)
    return tagged


def dump_neb_final_trajectory(
    diag: Dict[str, Any],
    job_context: Dict[str, Any],
    rank: int,
    step: int,
):
    if not diag:
        return
    images_post = diag.get("images_post")
    mover_idx = diag.get("mover_idx")
    if not images_post:
        return
    frames: List[Atoms] = []
    for atoms in images_post:
        if atoms is None or len(atoms) == 0:
            continue
        tagged = _tag_neb_frame(atoms, mover_idx)
        if mover_idx is not None and 0 <= mover_idx < len(tagged):
            order = [mover_idx] + [i for i in range(len(tagged)) if i != mover_idx]
            tagged = tagged[order]
        frames.append(tagged)
    if not frames:
        return
    path = _structure_traj_path(job_context=job_context)
    if path is None:
        return
    try:
        write(path, frames, format="lammpstrj")
        if DEBUG_LOGGING:
            log(f"[Rank {rank}] Structure trajectory written to {path}", step)
    except Exception as exc:
        log(f"[Rank {rank}] failed to write structure trajectory: {exc}", step)


def dump_neb_structure_snapshot(
    initial_atoms,
    final_atoms,
    mover_idx,
    job_context,
    rank,
    step,
):
    if initial_atoms is None or final_atoms is None or mover_idx is None:
        return
    try:
        if not (0 <= mover_idx < len(final_atoms)):
            return
        snapshot = initial_atoms.copy()
        target_pos = final_atoms.positions[mover_idx]
        snapshot += Atoms("H", positions=[target_pos], cell=snapshot.cell, pbc=True)
        snapshot.wrap()
        assign_mass_and_type(snapshot)
        types = np.array(snapshot.arrays.get("type", []), dtype=int)
        masses = snapshot.arrays.get("mass").copy()
        h_mask = snapshot.get_atomic_numbers() == 1
        init_pos = initial_atoms.positions[mover_idx]
        final_pos = target_pos
        snapshot_h_idxs = [i for i, Z in enumerate(snapshot.get_atomic_numbers()) if Z == 1]
        if snapshot_h_idxs:
            init_idx = min(snapshot_h_idxs, key=lambda idx: np.linalg.norm(snapshot.positions[idx] - init_pos))
            final_idx = min(snapshot_h_idxs, key=lambda idx: np.linalg.norm(snapshot.positions[idx] - final_pos))
            for idx in snapshot_h_idxs:
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
        path = _structure_snapshot_path(job_context=job_context)
        if path is None:
            return
        _manual_lammps_atomic_write(path, snapshot)
        if DEBUG_LOGGING:
            log(f"[Rank {rank}] Structure snapshot written to {path}", step)
    except Exception as exc:
        log(f"[Rank {rank}] Structure snapshot failed: {exc}", step)

def _manual_xyz_write(path, atoms):
    """Last resort XYZ writer (single frame)."""
    from io import StringIO
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    buf = StringIO()
    buf.write(f"{len(symbols)}\n")
    buf.write("validation dump\n")
    for s, (x, y, z) in zip(symbols, positions):
        buf.write(f"{s} {x:.10f} {y:.10f} {z:.10f}\n")
    data = buf.getvalue().encode("utf-8")
    import tempfile
    d = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmpname = tmp.name
    os.replace(tmpname, path)

def _write_atoms_xyz(path, atoms):
    try:
        write(path, atoms)
    except Exception:
        try:
            _manual_xyz_write(path, atoms)
        except Exception as exc:
            log(f"[Rank {rank}] Failed to write XYZ for {path}: {exc}")

def _short_hash(value, length=8):
    try:
        text = str(value).encode("utf-8", errors="ignore")
        digest = hashlib.sha1(text).hexdigest()
        return digest[:length]
    except Exception:
        return "hashfail"

def handle_neb_failure(payload):
    if not NEB_FAILURE_CAPTURE:
        return
    rank_value = payload.get("rank")
    job_context = payload.get("job_context") or {}
    step_value = job_context.get("step")
    h_value = job_context.get("h")
    n_value = job_context.get("n")
    key_value = job_context.get("key")
    safe_key = _short_hash(key_value) if key_value is not None else None
    parts = [f"rank{rank_value if rank_value is not None else rank}"]
    if step_value is not None:
        parts.append(f"step{step_value}")
    if h_value is not None and n_value is not None:
        parts.append(f"h{h_value}_n{n_value}")
    if safe_key is not None:
        parts.append(f"key{safe_key}")
    out_dir = os.path.join(NEB_FAILURE_DIR, "_".join(parts))
    _ensure_dir(out_dir)
    images = payload.get("images") or []
    context = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exception": payload.get("exception"),
        "mover_idx": payload.get("mover_idx"),
        "job_context": {k: _log_friendly(v) for k, v in job_context.items()},
        "images_captured": len(images),
    }
    context_path = os.path.join(out_dir, "context.json")
    try:
        with open(context_path, "w") as ctxf:
            json.dump(context, ctxf, indent=2)
    except Exception as exc:
        log(f"[Rank {rank_value if rank_value is not None else rank}] Failed to write NEB failure metadata: {exc}", step_value)
    for label in ("initial", "final"):
        atoms = payload.get(label)
        if isinstance(atoms, Atoms):
            atom_path = os.path.join(out_dir, f"{label}.xyz")
            _write_atoms_xyz(atom_path, atoms)
    for idx, img in enumerate(images):
        path = os.path.join(out_dir, f"neb_image_{idx:02d}.xyz")
        if isinstance(img, Atoms):
            _write_atoms_xyz(path, img)
    log(f"[Rank {rank_value if rank_value is not None else rank}] Captured NEB failure context in {out_dir}", step_value)

def _manual_lammps_atomic_write(path, atoms):
    """
    Minimal LAMMPS `atomic` style writer used as a last-resort fallback when ASE fails.
    Assumes an orthorhombic simulation cell (true for the 3a validation boxes).
    """
    symbols = atoms.get_chemical_symbols()
    if len(symbols) == 0:
        raise ValueError("Cannot write LAMMPS data for zero-atom configuration")

    positions = atoms.get_positions()
    cell = np.array(atoms.get_cell(), dtype=float)
    # Only support diagonal cells for the fallback path.
    off_diag = cell - np.diag(np.diagonal(cell))
    if np.linalg.norm(off_diag) > 1e-8:
        raise ValueError("Manual LAMMPS writer only supports orthorhombic cells")
    lx, ly, lz = np.diagonal(cell)

    type_array = atoms.arrays.get("type")
    if type_array is not None:
        types = np.asarray(type_array, dtype=int)
    else:
        type_lookup = {}
        types = []
        next_type = 1
        for s in symbols:
            if s not in type_lookup:
                type_lookup[s] = next_type
                next_type += 1
            types.append(type_lookup[s])
        types = np.asarray(types, dtype=int)

    ids_array = atoms.arrays.get("id")
    if ids_array is not None:
        ids = np.asarray(ids_array, dtype=int)
    else:
        ids = np.arange(1, len(symbols) + 1, dtype=int)

    masses = atoms.get_masses()
    type_masses = {}
    for t, m in zip(types, masses):
        type_masses.setdefault(int(t), float(m))

    unique_types = sorted(type_masses.keys())
    lines = []
    lines.append("(manual lammps-data fallback)")
    lines.append("")
    lines.append(f"{len(symbols)} atoms")
    lines.append(f"{len(unique_types)} atom types")
    lines.append("")
    lines.append(f"0.0 {lx:.16f}  xlo xhi")
    lines.append(f"0.0 {ly:.16f}  ylo yhi")
    lines.append(f"0.0 {lz:.16f}  zlo zhi")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    for t in unique_types:
        lines.append(f"{t:5d} {type_masses[t]:.10f}")
    lines.append("")
    lines.append("Atoms # atomic")
    lines.append("")
    neighbor_idx_array = atoms.arrays.get("neighbor_index")
    for atom_idx, (atom_id, atom_type, pos) in enumerate(zip(ids, types, positions)):
        x, y, z = pos
        comment = ""
        if neighbor_idx_array is not None:
            neighbor_val = int(neighbor_idx_array[atom_idx])
            if neighbor_val >= 0:
                comment = f" # neighbor {neighbor_val}"
        lines.append(f"{int(atom_id):6d} {int(atom_type):3d} {x:23.15f} {y:23.15f} {z:23.15f}{comment}")
    data = "\n".join(lines) + "\n"

    import tempfile
    d = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=d, delete=False, mode="w", encoding="ascii") as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmpname = tmp.name
    os.replace(tmpname, path)

def _atomic_ase_write(path, atoms, fmt, **kwargs):
    """Write via ASE using a temp file + fsync for robustness."""
    import tempfile
    d = os.path.dirname(path) or "."
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=d, delete=False) as tf:
            tmp = tf.name
        write(tmp, atoms, format=fmt, **kwargs)
        size = os.path.getsize(tmp)
        if size == 0:
            raise IOError(f"ASE writer produced empty output for {path}")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        raise
    else:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def extract_local_box_initial(rel_initial, a):
    """
    Deterministic builder for a 3a×3a×3a BCC Fe box around rel_initial.
    Generates 54 Fe (27 cells × 2 Fe) plus the moving H at its wrapped local
    coordinate, without referencing the global supercell. The returned box is
    non-periodic with vacuum around the 3a cluster.
    """
    rel_initial = _ensure_numeric_coords(rel_initial, "rel_initial for local box")
    L = 3.0 * a

    # Origin of the 3×3×3 conventional-cell block (cells ci-1..ci+1 etc.).
    ci, cj, ck = np.floor(rel_initial / a).astype(int)
    origin = np.array([ci - 1, cj - 1, ck - 1], float) * a

    # BCC basis (corner + body)
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.5]], float)
    pos = []
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                cell_origin = origin + np.array([dx, dy, dz], float) * a
                for b in basis:
                    pos.append((cell_origin + b * a - origin) % L)
    pos = np.array(pos, float)

    # H in local coords ([0,3a))
    h_local = (rel_initial - origin) % L

    initial = Atoms(symbols=["Fe"] * len(pos), positions=pos, cell=[L, L, L], pbc=False)
    initial += Atoms("H", positions=[h_local], cell=initial.cell, pbc=False)
    assign_mass_and_type(initial)
    freeze_mask = _build_3a_corner_freeze_mask(initial, a)
    _set_nonperiodic_local_box(initial, initial.copy())
    initial.set_array("freeze_mask", np.asarray(freeze_mask, dtype=bool))
    return initial

def _to_validation_local_coords(rel_initial, coords):
    """
    Convert absolute supercell coordinates to the 3a-local validation box.
    """
    rel_initial = _ensure_numeric_coords(rel_initial, "rel_initial for validation coords")
    L = 3.0 * a
    ci, cj, ck = np.floor(rel_initial / a).astype(int)
    origin = np.array([ci - 1, cj - 1, ck - 1], float) * a
    local = []
    for pos in coords:
        vec, _ = _neighbor_entry_to_pos(pos, "validation coordinate")
        local.append(((vec - origin) % L))
    return local


def _collect_validation_neighbor_h_positions(rel_initial, h_index):
    """
    Gather absolute positions of other H atoms near rel_initial using global H positions.
    """
    global H_positions, H_indices
    if H_positions is None or len(H_positions) == 0:
        return []
    rel_initial = _ensure_numeric_coords(rel_initial, f"rel_initial for H{h_index} neighbor collection")
    box = np.array([a * nx, a * ny, a * nz], float)
    collected = []
    for idx, pos_abs in enumerate(H_positions):
        idx_h = H_indices[idx]
        if idx_h == h_index:
            continue
        pos_abs = _ensure_numeric_coords(pos_abs, f"H_positions[{idx}] for neighbor collection")
        diff = pbc_diff(rel_initial, pos_abs, box)
        if np.linalg.norm(diff) <= ENV_RADIUS_A + 1e-6:
            wrapped = rel_initial + diff
            collected.append((wrapped, int(idx_h)))
    return collected


def _merge_validation_neighbor_hpos(store, h_index, positions):
    """
    Merge absolute neighbor H positions (global coords) into per-H store.
    Ensures uniqueness (rounded to 1e-8) and filters invalid entries.
    """
    if not positions:
        return
    cleaned = []
    for pos in positions:
        arr, idx = _neighbor_entry_to_pos(pos, f"neighbor position for H{h_index}")
        if arr is None or arr.shape != (3,):
            continue
        cleaned.append((arr, idx))
    if not cleaned:
        return
    existing = store.setdefault(h_index, [])
    existing.extend(cleaned)
    unique = []
    seen = set()
    for arr in existing:
        key = tuple(np.round(arr[0] if isinstance(arr, tuple) else arr, 8))
        if key in seen:
            continue
        seen.add(key)
        unique.append(arr)
    store[h_index] = unique


def dump_validation_initial_and_neighbors(
    out_dir,
    initial_atoms,
    h_index,
    neighbors_xyz,
    neighbors_idx,
    rel_initial=None,
    neighbor_h_positions=None,
):
    """
    Writes:
      - initial.data       (LAMMPS 'data' via ASE)
      - neighbors.txt      (plain x y z per line, header with count)
      - neighbors.csv      (n_site,x,y,z)
      - meta.json          (small metadata blob)
    """
    _ensure_dir(out_dir)
    lammps_path = os.path.join(out_dir, "initial.data")
    xyz_path = os.path.join(out_dir, "initial.xyz")
    try:
        t0 = time.time()
        if VALIDATION_FORMAT == "lammps":
            log(f"[VALID][rank{rank}] writing initial.data for h={h_index} -> {lammps_path}", -1)
            if VALIDATION_USE_ASE:
                _atomic_ase_write(lammps_path, initial_atoms, "lammps-data", atom_style="atomic")
            else:
                _manual_lammps_atomic_write(lammps_path, initial_atoms)
            log(f"[VALID] wrote initial.data in {time.time() - t0:.3f}s", -1)
        else:
            log(f"[VALID][rank{rank}] writing initial.xyz for h={h_index} -> {xyz_path}", -1)
            _atomic_ase_write(xyz_path, initial_atoms, "xyz")
            log(f"[VALID] wrote initial.xyz in {time.time() - t0:.3f}s", -1)
    except Exception as _e:
        if VALIDATION_FORMAT == "lammps":
            log(f"[VALID] initial.data write failed for h={h_index}: {_e}; trying manual LAMMPS", -1)
            try:
                _manual_lammps_atomic_write(lammps_path, initial_atoms)
                log(f"[VALID] manual LAMMPS write succeeded for h={h_index}", -1)
            except Exception as _fallback_exc:
                log(f"[VALID] manual LAMMPS fallback failed for h={h_index}: {_fallback_exc}; trying manual XYZ", -1)
                try:
                    _manual_xyz_write(xyz_path, initial_atoms)
                    log(f"[VALID] manual XYZ write succeeded for h={h_index}", -1)
                except Exception as _xyz_exc:
                    log(f"[VALID] manual XYZ write failed for h={h_index}: {_xyz_exc}", -1)
        else:
            log(f"[VALID] initial write failed for h={h_index}: {_e}; trying manual XYZ", -1)
            try:
                _manual_xyz_write(xyz_path, initial_atoms)
                log(f"[VALID] manual XYZ write succeeded for h={h_index}", -1)
            except Exception as _e2:
                log(f"[VALID] manual XYZ write failed for h={h_index}: {_e2}", -1)
    if rel_initial is not None:
        neighbors_xyz_local = _to_validation_local_coords(rel_initial, neighbors_xyz)
    else:
        neighbors_xyz_local = neighbors_xyz

    if rel_initial is not None and (not neighbor_h_positions):
        neighbor_h_positions = _collect_validation_neighbor_h_positions(rel_initial, h_index)
    neighbor_h_positions = neighbor_h_positions or []
    if rel_initial is not None and neighbor_h_positions:
        neighbor_h_local = _to_validation_local_coords(rel_initial, neighbor_h_positions)
        # Deduplicate (avoid inserting moving H twice)
        existing_positions = initial_atoms.get_positions(wrap=True)
        existing_keys = {tuple(np.round(p, 8)) for p in existing_positions}
        new_h = []
        for pos in neighbor_h_local:
            key = tuple(np.round(pos, 8))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_h.append(pos)
        if new_h:
            initial_atoms += Atoms("H" * len(new_h), positions=new_h, cell=initial_atoms.cell, pbc=True)
            assign_mass_and_type(initial_atoms)

    with open(os.path.join(out_dir, "neighbors.txt"), "w") as f:
        f.write(f"{len(neighbors_xyz_local)}\n")
        for xyz in neighbors_xyz_local:
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")
    with open(os.path.join(out_dir, "neighbors.csv"), "w") as f:
        f.write("n_site,x,y,z\n")
        for n, xyz in zip(neighbors_idx, neighbors_xyz_local):
            f.write(f"{n},{xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f}\n")
    try:
        import json
        meta = {"h_site": int(h_index), "neighbors": list(map(int, neighbors_idx))}
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def dump_validation_ase_neb(
    h_index,
    n_index,
    step,
    rank,
    diag,
    barrier_ev=None,
):
    """
    Write ASE NEB band dumps under the validation folder.
    Outputs:
      - dump.ase.post (optimized ASE trajectory)
      - dump.ase.pre  (optional; pre-optimization ASE trajectory)
    """
    if not diag:
        return
    out_dir = os.path.join(
        VALIDATION_OUT,
        f"rank{rank}_step{step}_h{h_index}",
        VALIDATION_ASE_DIR,
        str(n_index),
    )
    _ensure_dir(out_dir)
    mover_idx = diag.get("mover_idx")

    post_frames = diag.get("images_post") or []
    post_path = os.path.join(out_dir, "dump.ase.post")
    if post_frames and (VALIDATION_OVERWRITE or not os.path.exists(post_path)):
        write_neb_lammpstrj(post_frames, post_path, mover_idx=mover_idx, rank=rank)

    if VALIDATION_ASE_DUMP_PRE:
        pre_frames = diag.get("images_pre") or []
        pre_path = os.path.join(out_dir, "dump.ase.pre")
        if pre_frames and (VALIDATION_OVERWRITE or not os.path.exists(pre_path)):
            write_neb_lammpstrj(pre_frames, pre_path, mover_idx=mover_idx, rank=rank)

    if barrier_ev is not None:
        meta_path = os.path.join(out_dir, "meta.json")
        if VALIDATION_OVERWRITE or not os.path.exists(meta_path):
            try:
                import json
                meta = {
                    "h_site": int(h_index),
                    "n_site": int(n_index),
                    "barrier_eV": float(barrier_ev) if np.isfinite(barrier_ev) else None,
                    "rank": int(rank),
                    "step": int(step),
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass


def dump_local_env_build(
    out_dir,
    initial_atoms,
    final_atoms,
    mover_idx,
    *,
    mode,
    h_index,
    n_index,
    env_key=None,
):
    _ensure_dir(out_dir)
    initial_path = os.path.join(out_dir, "initial.xyz")
    final_path = os.path.join(out_dir, "final.xyz")
    _write_atoms_xyz(initial_path, initial_atoms)
    _write_atoms_xyz(final_path, final_atoms)

    freeze_mask = np.asarray(initial_atoms.arrays.get("freeze_mask", np.zeros(len(initial_atoms), dtype=bool)), dtype=bool)
    shell_distance = np.asarray(initial_atoms.arrays.get("shell_distance", np.full(len(initial_atoms), np.nan)), dtype=float)
    role_path = os.path.join(out_dir, "atom_roles.csv")
    with open(role_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atom_index", "symbol", "role", "frozen", "distance_A", "x", "y", "z"])
        for idx, atom in enumerate(initial_atoms):
            if idx == mover_idx:
                role = "mover"
            elif freeze_mask[idx]:
                role = "shell_frozen"
            else:
                role = "inner_movable"
            x, y, z = atom.position
            writer.writerow([
                idx,
                atom.symbol,
                role,
                int(bool(freeze_mask[idx])),
                float(shell_distance[idx]) if np.isfinite(shell_distance[idx]) else "",
                float(x),
                float(y),
                float(z),
            ])

    meta = {
        "mode": mode,
        "h_site": int(h_index),
        "n_site": int(n_index),
        "mover_idx": int(mover_idx),
        "num_atoms": int(len(initial_atoms)),
        "num_fe": int(sum(1 for s in initial_atoms.get_chemical_symbols() if s == "Fe")),
        "num_h": int(sum(1 for s in initial_atoms.get_chemical_symbols() if s == "H")),
        "num_frozen": int(np.sum(freeze_mask)),
        "cell_A": [float(v) for v in np.diag(np.asarray(initial_atoms.cell, dtype=float))],
        "env_key": str(env_key) if env_key is not None else None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_and_dump_local_envs(
    full_structure,
    sites,
    neighbors,
    h_indices,
    *,
    count,
    out_dir,
    rank,
):
    global H_positions, H_indices, H_cell_index
    box = np.array([a * nx, a * ny, a * nz], dtype=float)
    H_positions = np.asarray(sites[h_indices], dtype=float)
    H_indices = np.asarray(h_indices, dtype=int)
    H_cell_index = _build_h_cell_index(H_positions, H_indices, a)
    H_tree = cKDTree(H_positions, boxsize=box)

    built = 0
    _ensure_dir(out_dir)
    for h in map(int, h_indices):
        if h >= len(neighbors):
            continue
        for n in neighbors[h]:
            rel_initial, rel_final, env_key, neighbor_pos = extract_local_environment(
                full_structure,
                h,
                int(n),
                sites,
                a,
                rank,
                step=-1,
                radius=ENV_RADIUS_A,
                H_tree=H_tree,
                H_positions=H_positions,
            )
            if rel_initial is None or rel_final is None:
                continue
            initial, final, mover_idx = build_local_neb_structures(
                full_structure,
                rel_initial,
                rel_final,
                a,
                rank,
                step=-1,
                neighbor_positions=neighbor_pos,
            )
            if initial is None or final is None or mover_idx is None:
                continue
            case_dir = os.path.join(out_dir, f"env_{built:02d}_h{int(h)}_n{int(n)}")
            dump_local_env_build(
                case_dir,
                initial,
                final,
                mover_idx,
                mode=LOCAL_ENV_MODE,
                h_index=h,
                n_index=n,
                env_key=env_key,
            )
            built += 1
            if built >= count:
                return built
    return built


# ----------------------------------------------------------------------
# Defensive helper: normalize MPI-gathered arrays/lists to dicts
# ----------------------------------------------------------------------
def _as_dict(obj):
    """Convert numpy arrays or list-of-pairs into a proper dict."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.ndarray):
        try:
            return {k: v for k, v in obj.tolist()}
        except Exception:
            return {}
    if isinstance(obj, (list, tuple)):
        try:
            return dict(obj)
        except Exception:
            return {}
    return {}

# <editable:envextract>
def extract_local_environment(
    full_structure,
    h_index,
    target_index,
    sites,
    a,
    rank,
    step,
    radius=5.0,
    H_tree=None,
    H_positions=None,
):
    if h_index >= len(sites) or target_index >= len(sites):
        log(f"[Rank {rank}] Site index out of bounds: {h_index}->{target_index}", step)
        return None, None, None, None  # no key returned yet

    rel_initial = _ensure_numeric_coords(sites[h_index], f"site[{h_index}] initial")
    rel_final = _ensure_numeric_coords(sites[target_index], f"site[{target_index}] final")

    assert H_positions is not None and H_tree is not None
    global H_indices
    # === 2. Return the geometry key info for cache checking ===
    box = np.array([a*nx, a*ny, a*nz], dtype=float)
    env_key = make_env_key(
        rel_initial, rel_final, H_tree, H_positions, box,
        radius_A=ENV_RADIUS_A, pos_bin_A=POS_BIN_A, hop_bin_A=HOP_BIN_A,
        mode=ENV_KEY_MODE
    )

    nbr_idx = H_tree.query_ball_point(rel_initial, r=radius)
    neighbor_positions = []
    for idx in nbr_idx:
        pos = _ensure_numeric_coords(H_positions[idx], f"H_positions[{idx}] near site[{h_index}]")
        # skip the moving H itself
        if np.allclose(pbc_diff(rel_initial, pos, box), 0.0, atol=1e-6):
            continue
        diff = pbc_diff(rel_initial, pos, box)
        actual_idx = None
        if H_indices is not None and idx < len(H_indices):
            actual_idx = int(H_indices[idx])
        neighbor_positions.append((np.array(rel_initial + diff, dtype=float), actual_idx))

    return rel_initial, rel_final, env_key, neighbor_positions
# </editable:envextract>

def batch_extract_local_environment(
    h_index,
    target_indices,
    sites,
    a,
    rank,
    step,
    radius=5.0,
    H_tree=None,
    H_positions=None,
):
    """
    Compute env once for H=h_index and build keys for all targets (neighbors).
    Returns:
      rel_initial, finals_by_n (dict n->rel_final), keys_by_n (dict n->key), neighbor_positions
    """
    if h_index >= len(sites):
        log(f"[Rank {rank}] Site index out of bounds for batch h={h_index}", step)
        return None, {}, {}, None
    # absolute coords in the supercell
    box = np.array([a*nx, a*ny, a*nz], float)
    rel_initial = _ensure_numeric_coords(sites[h_index], f"site[{h_index}] initial batch")

    if H_positions is None:
        H_positions = _ensure_numeric_coords(sites, "sites array for batch")
    else:
        H_positions = _ensure_numeric_coords(H_positions, "H_positions array for batch")

    if H_tree is None:
        if len(H_positions) == 0:
            H_tree = cKDTree(np.zeros((1, 3)))
        else:
            H_tree = cKDTree(H_positions, boxsize=box)

    # One-time neighbor fingerprint at the initial site
    disp_sorted = _compute_disp_sorted(
        rel_initial, H_tree, H_positions, box, radius_A=ENV_RADIUS_A, pos_bin_A=POS_BIN_A
    )

    # Neighbor cloud around initial (for building local NEB boxes)
    nbr_idx = H_tree.query_ball_point(rel_initial, r=radius)
    neighbor_positions = []
    global H_indices
    for idx in nbr_idx:
        pos = _ensure_numeric_coords(H_positions[idx], f"H_positions[{idx}] in batch")
        if np.allclose(pbc_diff(rel_initial, pos, box), 0.0, atol=1e-6):
            continue
        diff = pbc_diff(rel_initial, pos, box)
        actual_idx = None
        if H_indices is not None and idx < len(H_indices):
            actual_idx = int(H_indices[idx])
        neighbor_positions.append((np.array(rel_initial + diff, dtype=float), actual_idx))

    finals_by_n, keys_by_n = {}, {}
    for n in target_indices:
        if n >= len(sites):
            continue
        rel_final = _ensure_numeric_coords(sites[n], f"site[{n}] final batch")
        key = make_env_key_from_disp(
            rel_initial, rel_final, box, disp_sorted=disp_sorted,
            hop_bin_A=HOP_BIN_A, mode=ENV_KEY_MODE
        )
        finals_by_n[n] = rel_final
        keys_by_n[n] = key
    return rel_initial, finals_by_n, keys_by_n, neighbor_positions

# -----------------------------------------------------------------------------
# NEB structure builders and calculator factories
# -----------------------------------------------------------------------------
def build_neb_structures(full_structure, rel_initial, rel_final, a, rank, step):
    """
    Deterministic 3a×3a×3a extractor around the interstitial's parent cell.

    - Identify parent cell (i,j,k) = floor(rel_initial/a).
    - For each atom, compute its conventional-cell index (cx,cy,cz)=floor(pos/a)
      and the wrapped index deltas (dx,dy,dz) relative to (i,j,k) in {-1,0,1}.
    - Keep atoms with |dx|,|dy|,|dz| ≤ 1.
    - Local coordinate in [0,3a) is constructed from its *in-cell* coordinate
      r = pos − (cx,cy,cz)*a plus the (dx,dy,dz) offset:
          r_local = (dx+1, dy+1, dz+1)*a + r
      This guarantees 0 ≤ r_local < 3a per axis, no big-cell wrapping needed.
    - Final H endpoint is placed via full-cell MIC vector.
    """
    rel_initial = _ensure_numeric_coords(rel_initial, "rel_initial for extractor")
    rel_final = _ensure_numeric_coords(rel_final, "rel_final for extractor")
    # parent cell of interstitial
    i, j, k = np.floor(rel_initial / a).astype(int)
    Lx, Ly, Lz = a*nx, a*ny, a*nz
    full_box = np.array([Lx, Ly, Lz], float)

    def wrap_cell_delta(c, c0, n):
        d = c - c0
        d -= n * np.round(d / n)
        return int(d)

    loc_pos = []
    loc_sym = []
    loc_abs_for_match = []  # for matching moving H robustly

    for idx, atom in enumerate(full_structure):
        pos = _ensure_numeric_coords(atom.position, f"{atom.symbol}[{idx}] position in extractor")
        # Conventional cell index of this atom
        cx, cy, cz = np.floor(pos / a + 1e-12).astype(int)

        dx = wrap_cell_delta(cx, i, nx)
        dy = wrap_cell_delta(cy, j, ny)
        dz = wrap_cell_delta(cz, k, nz)

        if (abs(dx) <= 1) and (abs(dy) <= 1) and (abs(dz) <= 1):
            # in-cell coordinate 0..a
            r = pos - np.array([cx*a, cy*a, cz*a], float)
            # map into 3a box deterministically
            r_local = np.array([dx+1, dy+1, dz+1], float) * a + r
            loc_pos.append(r_local)
            loc_sym.append(atom.symbol)
            loc_abs_for_match.append(pos)

    if not loc_pos:
        log(f"[Rank {rank}] extractor kept 0 atoms (unexpected).", step)
        return None, None, None

    initial = Atoms(symbols=loc_sym, positions=np.array(loc_pos),
                    cell=[3*a, 3*a, 3*a], pbc=True)
    final = initial.copy()
    assign_mass_and_type(initial)
    assign_mass_and_type(final)
    # Sanity: exactly 54 Fe (27 cells × 2 Fe/cell)
    nFe = sum(1 for s in loc_sym if s == "Fe")
    if nFe != 54:
        log(f"[Rank {rank}] extractor Fe count = {nFe} (expected 54). Aborting NEB for this hop.", step)
        return None, None, None

    # Identify moving H by nearest to the global rel_initial under full-cell MIC
    h_local_index, mind = None, float("inf")
    for idx, (s, abspos) in enumerate(zip(loc_sym, loc_abs_for_match)):
        if s != "H":
            continue
        d = np.linalg.norm(pbc_diff(rel_initial, abspos, full_box))
        if d < mind:
            mind = d
            h_local_index = idx
    if h_local_index is None or mind > 0.25:  # conservative
        log(f"[Rank {rank}] moving-H match failed (Δ={mind:.3f} Å).", step)
        return None, None, None

    # Place final H endpoint using full-cell MIC
    d_global = pbc_diff(rel_initial, rel_final, full_box)
    final[h_local_index].position = initial[h_local_index].position + d_global
    initial.wrap(); final.wrap()

    # Final guard: everything must be finite and inside [0,3a)
    for tag, atoms in (("initial", initial), ("final", final)):
        P = atoms.get_positions()
        if not np.isfinite(P).all():
            log(f"[Rank {rank}] {tag} contains non-finite coords; skipping.", step)
            return None, None, None
        if (P < -1e-9).any() or (P > 3*a + 1e-9).any():
            log(f"[Rank {rank}] {tag} coords outside 3a box; skipping.", step)
            return None, None, None

    return initial, final, h_local_index


def build_neb_structures_shell(
    full_structure,
    rel_initial,
    rel_final,
    a,
    rank,
    step,
    inner_radius=None,
    outer_radius=None,
):
    """
    Build a shell-cropped local environment from a wrapped local cell window.

    - Atoms with r <= inner_radius are movable.
    - Atoms with inner_radius < r <= outer_radius are kept but frozen.
    - Atoms with r > outer_radius are deleted.
    - Candidate atoms are gathered from wrapped neighboring conventional cells,
      then recentered using MIC relative to the moving H.
    - The local structure is written into a compact cubic cell to avoid
      scanning or simulating the full supercell for each hop.
    """
    del full_structure  # shell mode no longer scans the full structure
    rel_initial = _ensure_numeric_coords(rel_initial, "rel_initial for shell extractor")
    rel_final = _ensure_numeric_coords(rel_final, "rel_final for shell extractor")
    inner = SHELL_INNER_RADIUS_A if inner_radius is None else float(inner_radius)
    outer = SHELL_OUTER_RADIUS_A if outer_radius is None else float(outer_radius)
    inner = max(0.0, inner)
    outer = max(inner, outer)
    full_box = np.array([a * nx, a * ny, a * nz], dtype=float)
    i, j, k = np.floor(rel_initial / a + 1e-12).astype(int)
    cell_pad = max(1, int(np.ceil(outer / a)) + 1)
    d_global = pbc_diff(rel_initial, rel_final, full_box)
    local_pad = max(a, float(np.linalg.norm(d_global))) + 0.5
    local_half = outer + local_pad
    local_box = np.array([2.0 * local_half, 2.0 * local_half, 2.0 * local_half], dtype=float)
    local_center = np.array([local_half, local_half, local_half], dtype=float)

    loc_pos = []
    loc_sym = []
    freeze_mask = []
    dist_from_center = []
    mover_local_key = None
    mover_dist = float("inf")
    seen_fe = set()
    seen_h = set()

    def _append_atom(symbol, pos_local, dist, *, is_mover=False, dedupe_key=None):
        nonlocal mover_local_key, mover_dist
        if dedupe_key is not None:
            if symbol == "Fe":
                if dedupe_key in seen_fe:
                    return
                seen_fe.add(dedupe_key)
            else:
                if dedupe_key in seen_h:
                    return
                seen_h.add(dedupe_key)

        loc_pos.append(np.asarray(pos_local, dtype=float))
        loc_sym.append(symbol)
        dist_from_center.append(float(dist))
        freeze_mask.append((dist > inner) and (dist <= outer))
        idx_local = len(loc_pos) - 1
        if is_mover and dist < mover_dist:
            mover_dist = float(dist)
            mover_local_key = idx_local

    for sx in range(-cell_pad, cell_pad + 1):
        cx = (i + sx) % nx
        x0 = cx * a
        for sy in range(-cell_pad, cell_pad + 1):
            cy = (j + sy) % ny
            y0 = cy * a
            for sz in range(-cell_pad, cell_pad + 1):
                cz = (k + sz) % nz
                z0 = cz * a
                base = np.array([x0, y0, z0], dtype=float)
                for basis_idx, basis in enumerate((
                    np.array([0.0, 0.0, 0.0], dtype=float),
                    0.5 * a * np.array([1.0, 1.0, 1.0], dtype=float),
                )):
                    pos_abs = base + basis
                    diff = pbc_diff(rel_initial, pos_abs, full_box)
                    dist = float(np.linalg.norm(diff))
                    if dist > outer + 1e-8:
                        continue
                    _append_atom(
                        "Fe",
                        local_center + diff,
                        dist,
                        dedupe_key=(cx, cy, cz, basis_idx),
                    )

    global H_cell_index
    occupied_h_by_cell = H_cell_index or {}
    for sx in range(-cell_pad, cell_pad + 1):
        cx = (i + sx) % nx
        for sy in range(-cell_pad, cell_pad + 1):
            cy = (j + sy) % ny
            for sz in range(-cell_pad, cell_pad + 1):
                cz = (k + sz) % nz
                for pos_abs, neighbor_idx in occupied_h_by_cell.get((cx, cy, cz), ()):
                    pos_abs = np.asarray(pos_abs, dtype=float)
                    if pos_abs.shape != (3,) or not np.isfinite(pos_abs).all():
                        continue
                    diff = pbc_diff(rel_initial, pos_abs, full_box)
                    dist = float(np.linalg.norm(diff))
                    if dist > outer + 1e-8:
                        continue
                    dedupe_key = (
                        "idx",
                        int(neighbor_idx),
                    ) if neighbor_idx is not None else (
                        "pos",
                        tuple(np.round(pos_abs, 8)),
                    )
                    _append_atom(
                        "H",
                        local_center + diff,
                        dist,
                        is_mover=(dist <= 0.25),
                        dedupe_key=dedupe_key,
                    )

    if not loc_pos:
        log(f"[Rank {rank}] shell extractor kept 0 atoms.", step)
        return None, None, None

    initial = Atoms(
        symbols=loc_sym,
        positions=np.array(loc_pos, dtype=float),
        cell=local_box,
        pbc=False,
    )
    final = initial.copy()
    assign_mass_and_type(initial)
    assign_mass_and_type(final)

    h_local_index = mover_local_key
    if h_local_index is None or mover_dist > 0.25:
        log(f"[Rank {rank}] shell extractor moving-H match failed (Δ={mover_dist:.3f} Å).", step)
        return None, None, None

    final[h_local_index].position = initial[h_local_index].position + d_global
    _set_nonperiodic_local_box(initial, final)

    freeze_mask_arr = np.asarray(freeze_mask, dtype=bool)
    freeze_mask_arr[h_local_index] = False
    initial.set_array("freeze_mask", freeze_mask_arr.copy())
    final.set_array("freeze_mask", freeze_mask_arr.copy())
    initial.set_array("shell_distance", np.asarray(dist_from_center, dtype=float))
    final.set_array("shell_distance", np.asarray(dist_from_center, dtype=float))

    return initial, final, h_local_index

def get_lammps_calculator():
    global _LMP_ENGINE_LOGGED
    def _parse_lammps_files(raw):
        if not raw:
            return []
        if isinstance(raw, (list, tuple)):
            vals = [str(x).strip() for x in raw if str(x).strip()]
        else:
            vals = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
        return [v if os.path.isabs(v) else os.path.join(SCRIPT_DIR, v) for v in vals]

    if LAMMPS_PAIR_STYLE and LAMMPS_PAIR_COEFF:
        lmpcmds = [LAMMPS_PAIR_STYLE, LAMMPS_PAIR_COEFF]
        files = _parse_lammps_files(LAMMPS_FILES)
        potential_label = "CUSTOM"
    elif POTENTIAL == "NNP":
        lmpcmds = [
            (
                f'pair_style hdnnp {NNP_CUTOFF:.2f} dir "{NNP_DIR}" '
                f"showew no showewsum {NNP_SHOWEWSUM} resetew yes maxew {NNP_MAXEW} "
                f"cflength {NNP_CFLENGTH} cfenergy {NNP_CFENERGY}"
            ),
            f"pair_coeff * * {NNP_ELEMENTS}",
        ]
        files = []
        potential_label = "NNP"
    else:
        lmpcmds = [f"pair_style eam/fs", f"pair_coeff * * {POTENTIAL_EAM_FILE} Fe H"]
        files = _parse_lammps_files(LAMMPS_FILES)
        potential_label = "EAM"
        if POTENTIAL_EAM_FILE not in files:
            files.append(POTENTIAL_EAM_FILE)

    if rank == 0 and not _LMP_ENGINE_LOGGED:
        log(f"[Rank 0] POTENTIAL='{potential_label}' using commands: {lmpcmds}")
        _LMP_ENGINE_LOGGED = True
    calc = LAMMPSlib(
        lmpcmds=lmpcmds,
        atom_types={"Fe": 1, "H": 2},
        files=files,
        keep_alive=True,
        comm=MPI.COMM_SELF,  # <-- each rank gets its own independent LAMMPS instance
    )
    return calc

def _finite_or_none(atoms, tag, rank, step):
    if atoms is None: return False
    P = atoms.get_positions()
    if not np.isfinite(P).all():
        log(f"[Rank {rank}] {tag}: non-finite coords; skip.", step)
        return False
    return True

# <editable:neighbors>
def get_k_nearest_neighbors(sites, a, nx, ny, nz, k=8, same_type=False):
    """
    Return a list-of-lists: for each site i, the indices of its k geometric
    nearest neighbors under PBC. No arbitrary distance cuts.
    If same_type=True, we prefer neighbors with the same interstitial type
    (tet/tet or oct/oct); if not enough exist, we backfill with nearest others.
    """
    box = np.array([a*nx, a*ny, a*nz], dtype=float)
    tree = cKDTree(sites, boxsize=box)

    # k+1 because the nearest “neighbor” is self at distance 0
    dists, idxs = tree.query(sites, k=k+1, workers=-1)
    dists = dists[:, 1:]
    idxs  = idxs[:, 1:]

    if not same_type:
        return [list(row) for row in idxs]

    # same_type mode: keep tet<->tet and oct<->oct first
    # interstitial basis repeats every 18 sites (12 tet, 6 oct)
    per_cell = 18
    def is_tet(i):  return (i % per_cell) < 12
    def is_same(i,j): return is_tet(i) == is_tet(j)

    N = len(sites)
    out = []
    for i in range(N):
        nbrs = list(idxs[i])
        # take same-type first
        same = [j for j in nbrs if is_same(i, j)]
        if len(same) >= k:
            out.append(same[:k])
        else:
            # backfill with nearest others
            other = [j for j in nbrs if not is_same(i, j)]
            out.append((same + other)[:k])
    return out
# </editable:neighbors>

def run_opt(target_atoms, logfile, rank, calc, job_context=None):
    if len(target_atoms) == 0:
        log(f"[Rank {rank}] ERROR: Empty Atoms object passed to run_opt")
        return False

    # Reassign ID explicitly and validate required arrays
    ids = np.arange(1, len(target_atoms) + 1)
    target_atoms.set_array("id", ids)

    if "type" not in target_atoms.arrays:
        raise RuntimeError(f"[Rank {rank}] Missing 'type' array in atoms")
    if "mass" not in target_atoms.arrays:
        raise RuntimeError(f"[Rank {rank}] Missing 'mass' array in atoms")

    if target_atoms.calc is None:
        target_atoms.calc = calc

    ase_opt_log = _ase_log_path(logfile or "opt", rank=rank, job_context=job_context)
    if DEBUG_LOGGING and ase_opt_log:
        log(f"[Rank {rank}] ASE optimization log -> {ase_opt_log}")
    ase_opt_log_target = ase_opt_log if ase_opt_log else os.devnull

    try:
        _ = target_atoms.get_forces()
        energy = target_atoms.get_potential_energy()
        opt = BFGS(target_atoms, logfile=ase_opt_log_target)
        opt.run(fmax=0.05, steps=200)
        positions = target_atoms.get_positions()
        if not np.isfinite(positions).all():
            log(f"[Rank {rank}] Optimization produced non-finite coordinates.", -1)
            return False
        return True
    except Exception as e:
        log(f"[Rank {rank}] Optimization error: {e}")
        return False

# <editable:neb_core>
# NEB core logic now lives in engines.lammps_neb.LammpsNEB.
# </editable:neb_core>

def barrier(initial_atoms, final_atoms, calc_pool, rank,
            return_images=False, collect_diagnostics=False, mover_idx=None, job_context=None, **kwargs):
    """
    Public shim that lazily constructs a LammpsNEB engine and forwards calls
    to the core barrier routine. Keeps existing call sites stable.
    """
    global _neb_engine
    if _neb_engine is None:
        fix_sublattice_overrides = {}
        if FIX_SUBLATTICE_H is not None:
            fix_sublattice_overrides["H"] = FIX_SUBLATTICE_H
        if FIX_SUBLATTICE_FE is not None:
            fix_sublattice_overrides["Fe"] = FIX_SUBLATTICE_FE

        helpers = {
            "assign_mass_and_type": assign_mass_and_type,
            "log": log,
            # "dump": lambda payload: my_dump_sink(payload),  # optional structured dump hook
            "run_opt": run_opt,
            "make_calculator": get_lammps_calculator,
            "optimize_endpoints": OPTIMIZE_ENDPOINTS,
            "failure_handler": handle_neb_failure,
            "fix_sublattice": fix_sublattice_overrides,
        }
        _neb_engine = LammpsNEB(calc_pool, helpers=helpers)
        try:
            log(f"[Rank {rank}] NEB engine instantiated (images={max(3, len(calc_pool))}).")
        except Exception:
            pass

    neb_dump_mode = False
    try:
        neb_dump_mode = bool(NEB_DUMP)  # type: ignore[name-defined]
    except Exception:
        neb_dump_mode = False

    try:
        expected_h_count = sum(1 for Z in initial_atoms.get_atomic_numbers() if Z == 1)
    except Exception:
        expected_h_count = None

    optimizer_log_path = _ase_log_path("opt_mid", rank=rank, job_context=job_context)
    if DEBUG_LOGGING and optimizer_log_path:
        log(f"[Rank {rank}] ASE midpoint optimizer log -> {optimizer_log_path}")

    return _neb_engine.barrier(
        initial_atoms,
        final_atoms,
        rank=rank,
        return_images=return_images,
        collect_diagnostics=collect_diagnostics,
        mover_idx=mover_idx,
        auto_detect_mover=True,
        neb_dump_mode=neb_dump_mode,
        expected_h_count=expected_h_count,
        job_context=job_context,
        optimizer_log_path=optimizer_log_path,
        **kwargs,
    )

def write_neb_lammpstrj(frames, filepath, mover_idx=None, rank=None):
    if not frames:
        return
    traj_frames = []
    for atoms in frames:
        atoms_copy = atoms.copy()
        atoms_copy.wrap()

        if mover_idx is not None and 0 <= mover_idx < len(atoms_copy):
            order = [mover_idx] + [i for i in range(len(atoms_copy)) if i != mover_idx]
            atoms_copy = atoms_copy[order]

        traj_frames.append(atoms_copy)

    try:
        write(filepath, traj_frames, format="traj")
    except Exception as exc:
        if rank is not None:
            log(f"[Rank {rank}] ASE NEB trajectory write failed ({filepath}): {exc}")
        else:
            log(f"ASE NEB trajectory write failed ({filepath}): {exc}")

def _path_len(seq, cell):
    return float(sum(np.linalg.norm(pbc_diff(seq[i], seq[i+1], cell))
                     for i in range(len(seq)-1)))

def compute_prepost_metrics(diag, a):
    """
    Returns: drms_A, dmax_A, mid_A, Lpre_A, Lpost_A, deltaL_A
    """
    mover = diag["mover_idx"]
    pre  = diag["images_pre"]
    post = diag["images_post"]
    cell3 = np.array([3*a, 3*a, 3*a], dtype=float)

    pre_pos  = [frm[mover].position for frm in pre]
    post_pos = [frm[mover].position for frm in post]

    # Compare only interior images to avoid trivial zeros at endpoints
    if len(pre_pos) >= 3:
        idxs = range(1, len(pre_pos)-1)
    else:
        idxs = range(len(pre_pos))

    devs = [np.linalg.norm(pbc_diff(pre_pos[i], post_pos[i], cell3)) for i in idxs]
    drms = float(np.sqrt(np.mean(np.square(devs)))) if devs else 0.0
    dmax = float(np.max(devs)) if devs else 0.0
    mid  = float(np.linalg.norm(pbc_diff(pre_pos[len(pre_pos)//2],
                                         post_pos[len(post_pos)//2], cell3)))

    Lpre  = _path_len(pre_pos,  cell3)
    Lpost = _path_len(post_pos, cell3)
    dL    = Lpost - Lpre
    return drms, dmax, mid, Lpre, Lpost, dL

def spacing_abs(pre_frames, post_frames, mover_idx, a):
    """
    Purely absolute diagnostics (Å) that reveal clumping:
    - path length, end-to-end, interior span, segment stats, stuck count
    Also returns pre↔post interior RMS (drms_A) and deltaL_A.
    """
    cell3 = np.array([3*a, 3*a, 3*a], dtype=float)

    def pos(frames):
        return [frm[mover_idx].position for frm in frames]

    def stats(seq):
        n = len(seq)
        if n < 2:
            return dict(path_A=0.0, e2e_A=0.0, span_A=0.0,
                        meanseg_A=0.0, maxseg_A=0.0, minseg_A=0.0,
                        stuck_ct=0, seg=[], n=n)
        seg = [float(np.linalg.norm(pbc_diff(seq[i+1], seq[i], cell3))) for i in range(n-1)]
        path_A = float(sum(seg))
        e2e_A  = float(np.linalg.norm(pbc_diff(seq[-1], seq[0], cell3)))
        interior = range(1, n-1) if n >= 3 else range(0)
        span_A = max((float(np.linalg.norm(pbc_diff(seq[i], seq[0], cell3))) for i in interior), default=0.0)
        meanseg_A = float(np.mean(seg)) if seg else 0.0
        maxseg_A  = float(np.max(seg))  if seg else 0.0
        minseg_A  = float(np.min(seg))  if seg else 0.0
        stuck_ct  = sum(1 for i in interior
                        if np.linalg.norm(pbc_diff(seq[i], seq[0], cell3)) <= CLUMP_EPS_A)
        return dict(path_A=path_A, e2e_A=e2e_A, span_A=span_A,
                    meanseg_A=meanseg_A, maxseg_A=maxseg_A, minseg_A=minseg_A,
                    stuck_ct=stuck_ct, seg=seg, n=n)

    pre  = pos(pre_frames)
    post = pos(post_frames)

    s_pre  = stats(pre)
    s_post = stats(post)

    # pre↔post interior RMS deviation in Å
    if min(len(pre), len(post)) >= 3:
        idxs = range(1, min(len(pre), len(post)) - 1)
        devs = [float(np.linalg.norm(pbc_diff(pre[i], post[i], cell3))) for i in idxs]
        drms = float(np.sqrt(np.mean(np.square(devs)))) if devs else 0.0
    else:
        drms = 0.0

    deltaL = s_post["path_A"] - s_pre["path_A"]

    pre_clump = int(
        (s_pre["span_A"]   < CLUMP_EPS_A) or
        (s_pre["e2e_A"]    < CLUMP_EPS_A) or
        (s_pre["maxseg_A"] < CLUMP_EPS_A)
    )

    return {
        # PRE (absolute Å)
        "pre_len_A": s_pre["path_A"],
        "pre_span_A": s_pre["span_A"],
        "pre_e2e_A": s_pre["e2e_A"],
        "pre_meanseg_A": s_pre["meanseg_A"],
        "pre_maxseg_A": s_pre["maxseg_A"],
        "pre_minseg_A": s_pre["minseg_A"],
        "pre_stuck_ct": s_pre["stuck_ct"],
        "pre_clump": pre_clump,
        # POST (absolute Å)
        "post_len_A": s_post["path_A"],
        "post_span_A": s_post["span_A"],
        "post_e2e_A": s_post["e2e_A"],
        "post_meanseg_A": s_post["meanseg_A"],
        "post_maxseg_A": s_post["maxseg_A"],
        "post_minseg_A": s_post["minseg_A"],
        # PRE↔POST
        "drms_A": drms,
        "deltaL_A": deltaL,
    }

def merge_barrier_caches():
    comm.Barrier()
    if rank == 0:
        merged_cache = {}
        for r in range(size):
            fname = f"barrier_cache_rank{r}_{CACHE_SCHEMA}.pkl"  # <-- schema-aware
            delta_fname = f"{fname}.delta.pkl"
            if (not os.path.exists(fname)) and (not os.path.exists(delta_fname)):
                log(f"[Rank 0] Note: {fname} (and delta) not found; skipping")
                continue
            try:
                rank_cache = BarrierCache(fname)
                data = dict(rank_cache)
                log(f"[Rank 0] Merging {len(data)} entries from {fname}")
                merged_cache.update(data)
            except Exception as e:
                log(f"[Rank 0] Error loading {fname}: {e}")

        merged_fname = f"barrier_cache_{CACHE_SCHEMA}.pkl"
        try:
            with open(merged_fname, "wb") as f:
                pickle.dump(merged_cache, f)
            log(f"[Rank 0] Wrote merged barrier cache with {len(merged_cache)} entries to {merged_fname}")
        except Exception as e:
            log(f"[Rank 0] ERROR writing merged cache: {e}")

# === Fast 3a Fe template & NEB endpoints (no full supercell) ===
FE_3A_TEMPLATE = None  # set in main()
TEMPLATE_SHRINKWRAP_PAD_A = 2.0


def build_fe_3a_template(a):
    """Build a non-periodic 3a×3a×3a bcc Fe template (54 Fe) once; reuse for all hops."""
    cell = np.array([3*a, 3*a, 3*a], float)
    pos = []
    # 3x3x3 conventional cells; 2 Fe per cell (corner + body)
    for I in range(3):
        for J in range(3):
            for K in range(3):
                base = np.array([I, J, K], float) * a
                pos.append(base + np.array([0.0, 0.0, 0.0]))            # corner
                pos.append(base + 0.5*a*np.array([1.0, 1.0, 1.0]))      # body
    Fe = Atoms('Fe' * len(pos), positions=np.array(pos), cell=cell, pbc=False)
    assign_mass_and_type(Fe)
    return Fe


def _build_3a_corner_freeze_mask(atoms, a):
    """
    Freeze only one extreme corner Fe atom of the canonical 3a template.
    """
    positions = np.asarray(atoms.get_positions(), dtype=float)
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    tol = 1e-8
    anchor = np.array([0.0, 0.0, 0.0], dtype=float)
    is_anchor = np.all(np.isclose(positions, anchor[None, :], atol=tol), axis=1)
    return np.logical_and(atomic_numbers == 26, is_anchor)


def _set_nonperiodic_local_box(initial, final, padding=TEMPLATE_SHRINKWRAP_PAD_A):
    """
    Convert a local NEB pair to a non-periodic box sized to contain both states.

    This prevents the template-mode local environments from interacting across
    periodic images while still giving LAMMPS enough vacuum to shrink-wrap.
    """
    combined = np.vstack((initial.get_positions(), final.get_positions()))
    mins = combined.min(axis=0) - float(padding)
    maxs = combined.max(axis=0) + float(padding)
    cell = np.maximum(maxs - mins, 1.0)

    for atoms in (initial, final):
        atoms.positions[:] = atoms.get_positions() - mins
        atoms.set_cell(cell)
        atoms.set_pbc(False)


# <editable:neb_endpoints_fast>
def _build_neb_structures_fast_impl(
    rel_initial,
    rel_final,
    a,
    rank,
    step,
    neighbor_positions=None,
    *,
    include_window_occupancy,
):
    """
    Fast endpoint builder over the 3a Fe template.

    `include_window_occupancy=True` uses the wrapped 3x3x3 cell occupancy map.

    `include_window_occupancy=False` preserves the KD-tree radial-neighbor path
    and only inserts the H supplied via `neighbor_positions`.
    """
    global FE_3A_TEMPLATE
    if FE_3A_TEMPLATE is None:
        raise RuntimeError("FE_3A_TEMPLATE not initialized")

    rel_initial = _ensure_numeric_coords(rel_initial, "rel_initial for fast extractor")
    rel_final = _ensure_numeric_coords(rel_final, "rel_final for fast extractor")
    if not (np.isfinite(rel_initial).all() and np.isfinite(rel_final).all()):
        log(f"[Rank {rank}] fast extractor received non-finite endpoints; skipping.", step)
        return None, None, None

    # Parent cell coords of initial site and its in-cell coordinate [0,a)
    i, j, k = np.floor(rel_initial / a).astype(int)
    r0 = rel_initial - np.array([i, j, k], float) * a
    off = np.array([1.0, 1.0, 1.0]) * a      # center the parent cell inside the 3a box

    # Full-box MIC displacement initial->final
    full_box = np.array([a*nx, a*ny, a*nz], float)
    d_global = pbc_diff(rel_initial, rel_final, full_box)

    # Create endpoints
    initial = FE_3A_TEMPLATE.copy()
    final   = FE_3A_TEMPLATE.copy()

    pos0 = off + r0
    pos1 = pos0 + d_global

    initial += Atoms('H', positions=[pos0], cell=initial.cell, pbc=False)
    final   += Atoms('H', positions=[pos1], cell=final.cell,   pbc=False)
    mover_idx = len(initial) - 1  # index of the moving H before adding neighbors

    def wrap_cell_delta(c, c0, n):
        d = c - c0
        d -= n * np.round(d / n)
        return int(d)

    neighbor_entries = []
    seen_neighbor_keys = set()

    def _register_neighbor(pos_abs, neighbor_idx):
        pos_abs = np.asarray(pos_abs, dtype=float)
        if pos_abs.shape != (3,) or not np.isfinite(pos_abs).all():
            return
        if np.allclose(pbc_diff(rel_initial, pos_abs, full_box), 0.0, atol=1e-6):
            return
        key = tuple(np.round(pos_abs, 8))
        if key in seen_neighbor_keys:
            return
        seen_neighbor_keys.add(key)
        neighbor_entries.append((pos_abs.copy(), neighbor_idx))

    if include_window_occupancy:
        global H_cell_index
        occupied_h_by_cell = H_cell_index or {}
        for sx in (-1, 0, 1):
            cx = (i + sx) % nx
            for sy in (-1, 0, 1):
                cy = (j + sy) % ny
                for sz in (-1, 0, 1):
                    cz = (k + sz) % nz
                    for pos_abs, neighbor_idx in occupied_h_by_cell.get((cx, cy, cz), ()):
                        _register_neighbor(pos_abs, neighbor_idx)
    else:
        for entry in (neighbor_positions or []):
            pos_abs, neighbor_idx = _neighbor_entry_to_pos(
                entry, f"neighbor position for fast extractor (step {step})"
            )
            if pos_abs is None:
                continue
            _register_neighbor(pos_abs, neighbor_idx)

    neighbor_meta = []
    for pos_abs, neighbor_idx in neighbor_entries:
        cx, cy, cz = np.floor(pos_abs / a + 1e-12).astype(int)
        dx = wrap_cell_delta(cx, i, nx)
        dy = wrap_cell_delta(cy, j, ny)
        dz = wrap_cell_delta(cz, k, nz)

        if (abs(dx) > 1) or (abs(dy) > 1) or (abs(dz) > 1):
            log(f"[Rank {rank}] Neighbor H outside 3a extractor: dx={dx}, dy={dy}, dz={dz}", step)
            continue

        r = pos_abs - np.array([cx*a, cy*a, cz*a], float)
        r_local = np.array([dx+1, dy+1, dz+1], float) * a + r

        initial += Atoms('H', positions=[r_local], cell=initial.cell, pbc=False)
        init_idx = len(initial) - 1
        final   += Atoms('H', positions=[r_local], cell=final.cell,   pbc=False)
        final_idx = len(final) - 1
        neighbor_meta.append((init_idx, final_idx, neighbor_idx))

    freeze_mask = _build_3a_corner_freeze_mask(initial, a)
    if 0 <= mover_idx < len(freeze_mask):
        freeze_mask[mover_idx] = False
    _set_nonperiodic_local_box(initial, final)

    # Ensure arrays exist for endpoint optimization
    assign_mass_and_type(initial)
    assign_mass_and_type(final)
    initial.set_array("freeze_mask", np.asarray(freeze_mask, dtype=bool))
    final.set_array("freeze_mask", np.asarray(freeze_mask, dtype=bool))
    neighbor_idx_init = np.full(len(initial), -1, dtype=int)
    neighbor_idx_final = np.full(len(final), -1, dtype=int)
    for init_idx, final_idx, neighbor_idx in neighbor_meta:
        if neighbor_idx is None:
            continue
        neighbor_idx_init[init_idx] = int(neighbor_idx)
        neighbor_idx_final[final_idx] = int(neighbor_idx)
    initial.set_array("neighbor_index", neighbor_idx_init)
    final.set_array("neighbor_index", neighbor_idx_final)
    # quick sanity: Fe count should be 54
    nFe = sum(1 for s in initial.get_chemical_symbols() if s == "Fe")
    if nFe != 54:
        log(f"[Rank {rank}] fast extractor Fe count = {nFe} (expected 54).", step)
        return None, None, None

    return initial, final, mover_idx


def build_neb_structures_fast_wrapped(rel_initial, rel_final, a, rank, step, neighbor_positions=None):
    return _build_neb_structures_fast_impl(
        rel_initial,
        rel_final,
        a,
        rank,
        step,
        neighbor_positions=neighbor_positions,
        include_window_occupancy=True,
    )


def build_neb_structures_fast_radial(rel_initial, rel_final, a, rank, step, neighbor_positions=None):
    return _build_neb_structures_fast_impl(
        rel_initial,
        rel_final,
        a,
        rank,
        step,
        neighbor_positions=neighbor_positions,
        include_window_occupancy=False,
    )
# </editable:neb_endpoints_fast>


def build_local_neb_structures(
    full_structure,
    rel_initial,
    rel_final,
    a,
    rank,
    step,
    neighbor_positions=None,
):
    if LOCAL_ENV_MODE == "shell":
        return build_neb_structures_shell(
            full_structure,
            rel_initial,
            rel_final,
            a,
            rank,
            step,
        )
    if LOCAL_ENV_MODE == "radial":
        return build_neb_structures_fast_radial(
            rel_initial,
            rel_final,
            a,
            rank,
            step,
            neighbor_positions=neighbor_positions,
        )
    return build_neb_structures_fast_wrapped(
        rel_initial,
        rel_final,
        a,
        rank,
        step,
        neighbor_positions=neighbor_positions,
    )

if NEB_DUMP:
    steps = 1

# -----------------------------------------------------------------------------
# Main KMC/NEB simulation driver
# -----------------------------------------------------------------------------
def main():
    global H_positions, H_indices, H_unwrapped_positions, H_initial_unwrapped_positions, H_cell_index, VALIDATION_MODE
    prog_wall_t0 = time.time()
    # Rank-0 init timing accumulator (human-readable; CSV remains authoritative)
    init_times = {
        "gen_sites": 0.0,
        "build_knn": 0.0,
        "save_knn": 0.0,
        "bcast_neighbors": 0.0,
        "calc_pool": 0.0,
    }
    # --------------- Timing CSV (per rank) -------------------
    perstep_bucket_count = len(TIMING_BUCKET_FIELDS)
    timing_csv = None
    if DEBUG_LOGGING:
        timing_csv = f"timing_rank{rank}.csv"
        with open(timing_csv, "w") as f:
            f.write(",".join([
                "step",
                "wall_total",
                *TIMING_BUCKET_FIELDS,
                "components_sum",
                "unaccounted"
            ]) + "\n")
    init_row_written = False
    # ----------------------------------------------------------- 
    # ---- Global rate table CSV (aggregated from all ranks) ----
    rate_csv = None
    if VALIDATION_MODE:
        rate_csv = "validation_rates_allranks.csv"
    elif DEBUG_LOGGING:
        rate_csv = "rates_allranks.csv"
    if rate_csv and rank == 0:
        with open(rate_csv, "w") as f:
            f.write(",".join([
                "step","rank","h_site","n_site",
                "barrier_eV","rate_Hz","source","env_kind","status"
            ]) + "\n")
    # -----------------------------------------------------------    
    
    step_start = time.time()
    step_wall_times = []
    global sites
    traj_file = "H_trajectory_onlyH.lammpstrj"
    t_traj = 0.0
    kmc_diag_file = f"kmc_diagnostics_rank{rank}.log"
    if rank == 0:
        kmc_diag_file = "kmc_diagnostics_rank0.log"
        with open(kmc_diag_file, "w") as f:
            f.write("KMC Diagnostics Log (Rank 0)\n")
            f.write("Step | H move | Δt (s) | Total t (s)\n")
        diag_window_start_step = None
        diag_window_end_step = None
        diag_window_dt_sum = 0.0
        diag_window_count = 0
        diag_window_last_move = (-1, -1)
        diag_window_has_nonfinite_dt = False
    else:
        kmc_diag_file = None  # Unused by non-root ranks

    if DEBUG_LOGGING or rank == 0:
        with open(f"log_rank{rank}.txt", "w") as f:
            f.write(f"[Rank {rank}] Starting new KMC-NEB simulation log\n")
    if rank == 0:
        t0 = time.time()
        sites = generate_interstitial_sites(a, nx, ny, nz)
        dt_sites = time.time() - t0
        init_times["gen_sites"] = dt_sites
        mode_label = "tet+octa" if USE_OCTAHEDRAL_VOIDS else "tet-only"
        log(
            f"[Rank {rank}] Generated {len(sites)} interstitial sites in {dt_sites:.2f} s "
            f"({mode_label})"
        )
        log(
            f"[Rank {rank}] Local environment mode: {LOCAL_ENV_MODE} "
            f"(inner={SHELL_INNER_RADIUS_A:.2f} A, outer={SHELL_OUTER_RADIUS_A:.2f} A, "
            f"cache_schema={CACHE_SCHEMA})"
        )
        log(f"[Rank {rank}] Job assignment mode: {JOB_ASSIGNMENT_MODE} "
            f"({JOB_ASSIGNMENT_MODES[JOB_ASSIGNMENT_MODE]})")
    else:
        sites = None
    sites = comm.bcast(sites, root=0)
    global FE_3A_TEMPLATE
    FE_3A_TEMPLATE = build_fe_3a_template(a)
    num_sites = len(sites)
    box = [a * nx, a * ny, a * nz]
    box_array = np.array(box, dtype=float)
    # --- Robust neighbor list filename (include method + K) ---
    KNN_K = 6                # <- set the number of neighbors you want
    SAME_TYPE = False        # <- set True if you want tet↔tet, oct↔oct only

    neighbors_file = f"neighbor_list_knnK{KNN_K}_sametype{int(SAME_TYPE)}_nx{nx}_ny{ny}_nz{nz}.pkl"
    pending_neighbor_sync = 0.0

    if rank == 0:
        if os.path.exists(neighbors_file):
            t_load = time.time()
            with open(neighbors_file, "rb") as f:
                neighbors = pickle.load(f)
            load_dt = time.time() - t_load
            init_times["build_knn"] = load_dt
            log(f"[Rank {rank}] Loaded KNN neighbors from {neighbors_file} in {load_dt:.2f} s")
        else:
            t1 = time.time()
            neighbors = get_k_nearest_neighbors(sites, a, nx, ny, nz,
                                                k=KNN_K, same_type=SAME_TYPE)
            build_dt = time.time() - t1
            init_times["build_knn"] = build_dt
            log(f"[Rank {rank}] Built KNN neighbors (K={KNN_K}, same_type={SAME_TYPE}) "
                f"in {build_dt:.2f} s")
            t_save = time.time()
            with open(neighbors_file, "wb") as f:
                pickle.dump(neighbors, f)
            save_dt = time.time() - t_save
            init_times["save_knn"] = save_dt
            log(f"[Rank {rank}] Saved KNN neighbors to {neighbors_file} "
                f"in {save_dt:.2f} s")
    else:
        neighbors = None

    t0_neighbor_sync = time.time()
    neighbors = comm.bcast(neighbors, root=0)
    neighbor_sync_time = time.time() - t0_neighbor_sync
    if rank == 0:
        init_times["bcast_neighbors"] = neighbor_sync_time
        log(f"[Rank {rank}] Broadcasted neighbor list in {neighbor_sync_time:.2f} s")

    # Log neighbor statistics only once
    if rank == 0:
        avg_neighbors = sum(len(nbrs) for nbrs in neighbors) / len(neighbors)
    else:
        avg_neighbors = None
    avg_neighbors = comm.bcast(avg_neighbors, root=0)
    log(f"[Rank {rank}] Average number of neighbors per site: {avg_neighbors:.2f}")
    pending_neighbor_sync = neighbor_sync_time

    if rank == 0:
        if BUILD_LOCAL_ENV_ONLY and BUILD_LOCAL_ENV_SELECTOR != "random":
            H_indices = select_h_indices_for_test_mode(
                sites,
                num_H,
                selector=BUILD_LOCAL_ENV_SELECTOR,
                box=box_array,
                boundary_margin_a=BOUNDARY_MARGIN_A,
                cluster_radius_a=CLUSTER_RADIUS_A,
            )
            log(
                f"[Rank 0] Build-only H selector '{BUILD_LOCAL_ENV_SELECTOR}' chose {len(H_indices)} sites "
                f"(boundary_margin={BOUNDARY_MARGIN_A:.2f} A, cluster_radius={CLUSTER_RADIUS_A:.2f} A)."
            )
        else:
            H_indices = np.random.choice(num_sites, size=num_H, replace=False)
    else:
        H_indices = None

    H_indices = comm.bcast(H_indices, root=0)
    if rank == 0:
        H_unwrapped_positions = np.asarray(sites[H_indices], dtype=float).copy()
        H_initial_unwrapped_positions = H_unwrapped_positions.copy()
    else:
        H_unwrapped_positions = None
        H_initial_unwrapped_positions = None
    full_structure = build_full_structure(sites, H_indices)
    local_H = H_indices[rank::size]
    if BUILD_LOCAL_ENV_ONLY:
        built = 0
        if rank == 0:
            built = build_and_dump_local_envs(
                full_structure,
                sites,
                neighbors,
                H_indices,
                count=BUILD_LOCAL_ENV_COUNT,
                out_dir=BUILD_LOCAL_ENV_OUT,
                rank=rank,
            )
            log(
                f"[Rank 0] Local environment build-only mode complete: "
                f"built {built}/{BUILD_LOCAL_ENV_COUNT} '{LOCAL_ENV_MODE}' environments in {BUILD_LOCAL_ENV_OUT}"
            )
        built = comm.bcast(built if rank == 0 else None, root=0)
        if rank == 0 and built < BUILD_LOCAL_ENV_COUNT:
            log(
                f"[Rank 0] Requested {BUILD_LOCAL_ENV_COUNT} environments but only built {built}.",
                step=-1,
            )
        return
    if rank == 0:
        Lx, Ly, Lz = a*nx, a*ny, a*nz

        def _append_h_frame(tstep_value, h_list):
            h_pos = sites[h_list]
            with open(traj_file, "a") as f:
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{int(tstep_value)}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{len(h_pos)}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write(f"0.0 {Lx:.6f}\n")
                f.write(f"0.0 {Ly:.6f}\n")
                f.write(f"0.0 {Lz:.6f}\n")
                f.write("ITEM: ATOMS id type x y z\n")
                for i, pos in enumerate(h_pos, start=1):
                    f.write(f"{i} 2 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        # Reset trajectory output and write initial frame (step 0).
        open(traj_file, "w").close()
        t0_traj = time.time()
        _append_h_frame(0, H_indices)
        t_traj += time.time() - t0_traj
    msd_tracker = MSDTracker(
        initial_indices=H_indices.copy(),
        box=box_array,
        initial_positions=H_initial_unwrapped_positions,
        use_unwrapped_positions=(rank == 0),
        log_interval=MSD_LOG_INTERVAL,
        log_file=MSD_LOG_FILE,
        plot_file=MSD_PLOT_FILE,
        enabled=(rank == 0),
    )
    if rank == 0 and H_unwrapped_positions is not None:
        msd_tracker.record(
            step=0,
            sim_time=0.0,
            current_positions=H_unwrapped_positions,
            force=True,
        )
    # Log local H assignment before KMC loop
    log(f"[Rank {rank}] Assigned {len(local_H)} local H atoms")
    log(
        f"[Rank {rank}] Approx. NEB calculations per KMC step: {len(local_H) * avg_neighbors:.0f}"
    )
    if rank == 0:
        log(f"[Rank 0] Trajectory + KMC diagnostics dump interval: every {DUMP_EVERY_STEPS} step(s).")
    sim_time = 0.0
    _t0_calc_pool = time.time()
    calc_pool = [get_lammps_calculator() for _ in range(NIMG)]
    _dt_calc_pool = time.time() - _t0_calc_pool
    if rank == 0:
        init_times["calc_pool"] = _dt_calc_pool
        log(f"[Rank {rank}] Initialized {NIMG} LAMMPS calculators in {_dt_calc_pool:.2f} s")
        _subtotal = (
            init_times["gen_sites"]
            + init_times["build_knn"]
            + init_times["save_knn"]
            + init_times["bcast_neighbors"]
            + init_times["calc_pool"]
        )
        log(
            f"[Rank {rank}] INIT subtotal (listed components): {_subtotal:.2f} s "
            f"[gen={init_times['gen_sites']:.2f}, knn={init_times['build_knn']:.2f}, "
            f"save={init_times['save_knn']:.2f}, bcast={init_times['bcast_neighbors']:.2f}, "
            f"calc_pool={init_times['calc_pool']:.2f}]"
        )
    # Reset the lazy NEB engine so it captures the fresh calculator pool for this run.
    global _neb_engine
    _neb_engine = None
    step_start = time.time()

    # --- NEB diagnostics helpers (per-rank CSV output) ---
    neb_diag_path = f"neb_diag_rank{rank}.csv"
    neb_jobs_path = f"neb_jobs_rank{rank}.csv"
    neb_diag_rows = []  # stash rows locally for rank-0 aggregation at the end
    neb_job_rows = []   # optional per-job trace
    neb_neighbor_distance_path = f"neb_neighbor_distances_rank{rank}.csv"
    distance_columns = []
    for idx in range(KNN_K):
        distance_columns.extend([f"neighbor_{idx}", f"distance_{idx}"])
    distance_header = [
        "step",
        "rank",
        "h_site",
        "neb_target",
        *distance_columns,
        "avg_neighbor_distance",
    ]
    _distance_header_written = False

    def _ensure_distance_header():
        nonlocal _distance_header_written
        if _distance_header_written:
            return
        if not os.path.exists(neb_neighbor_distance_path):
            with open(neb_neighbor_distance_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(distance_header)
        _distance_header_written = True

    def _append_distance_row(row):
        _ensure_distance_header()
        with open(neb_neighbor_distance_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _neb_diag_write_header_if_needed(current_step: int) -> None:
        if not NEB_DIAG or current_step != 0:
            return
        with open(neb_diag_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "rank",
                "neb_assigned",
                "neb_ran",
                "neb_batches",
                "neb_wall_s",
                "neb_avg_s",
                "neb_struct_s",
                "neb_setup_s",
                "neb_preprocess_s",
                "neb_preprocess_avg_s",
                "neb_wait_s",
                "neb_engine_total_s",
                "neb_engine_avg_s",
                "neb_total_with_preprocess_s",
                "neb_total_with_preprocess_avg_s",
                "neb_header_s",
                "neb_endpoint_constraints_s",
                "neb_endpoint_opt_s",
                "neb_image_setup_s",
                "neb_image_constraints_s",
                "neb_interpolate_s",
                "neb_optimize_s",
                "neb_energy_eval_s",
                "neb_failure_cleanup_s",
                "neb_avg_atoms",
                "neb_avg_fe",
                "neb_avg_h",
                "neb_avg_frozen",
            ])
        if NEB_JOB_TRACE:
            with open(neb_jobs_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "rank",
                    "key",
                    "t_struct_s",
                    "t_setup_s",
                    "t_solver_s",
                    "t_engine_total_s",
                    "t_header_s",
                    "t_endpoint_constraints_s",
                    "t_endpoint_opt_s",
                    "t_image_setup_s",
                    "t_image_constraints_s",
                    "t_interpolate_s",
                    "t_optimize_s",
                    "t_energy_eval_s",
                    "t_failure_cleanup_s",
                    "barrier_eV",
                ])

    def _neb_diag_append(current_step: int,
                         assigned: int,
                         ran: int,
                         batches: int,
                         neb_wall: float,
                         struct_wall: float,
                         setup_wall: float,
                         wait_wall: float,
                         engine_total_wall: float,
                         engine_stage_sums: Dict[str, float],
                         atom_stats_sums: Dict[str, float]) -> None:
        if not NEB_DIAG:
            return
        avg = (neb_wall / ran) if ran > 0 else 0.0
        preprocess_wall = struct_wall + setup_wall
        preprocess_avg = (preprocess_wall / ran) if ran > 0 else 0.0
        engine_avg = (engine_total_wall / ran) if ran > 0 else 0.0
        total_with_preprocess = neb_wall + preprocess_wall
        total_with_preprocess_avg = (total_with_preprocess / ran) if ran > 0 else 0.0
        avg_atoms = (atom_stats_sums.get("atoms", 0.0) / ran) if ran > 0 else 0.0
        avg_fe = (atom_stats_sums.get("fe", 0.0) / ran) if ran > 0 else 0.0
        avg_h = (atom_stats_sums.get("h", 0.0) / ran) if ran > 0 else 0.0
        avg_frozen = (atom_stats_sums.get("frozen", 0.0) / ran) if ran > 0 else 0.0
        neb_diag_rows.append([
            int(current_step),
            int(rank),
            int(assigned),
            int(ran),
            int(batches),
            float(neb_wall),
            float(avg),
            float(struct_wall),
            float(setup_wall),
            float(preprocess_wall),
            float(preprocess_avg),
            float(wait_wall),
            float(engine_total_wall),
            float(engine_avg),
            float(total_with_preprocess),
            float(total_with_preprocess_avg),
            float(engine_stage_sums.get("header_s", 0.0)),
            float(engine_stage_sums.get("endpoint_constraints_s", 0.0)),
            float(engine_stage_sums.get("endpoint_opt_s", 0.0)),
            float(engine_stage_sums.get("image_setup_s", 0.0)),
            float(engine_stage_sums.get("image_constraints_s", 0.0)),
            float(engine_stage_sums.get("interpolate_s", 0.0)),
            float(engine_stage_sums.get("optimizer_s", 0.0)),
            float(engine_stage_sums.get("energy_eval_s", 0.0)),
            float(engine_stage_sums.get("failure_cleanup_s", 0.0)),
            float(avg_atoms),
            float(avg_fe),
            float(avg_h),
            float(avg_frozen),
        ])
        with open(neb_diag_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(current_step),
                int(rank),
                int(assigned),
                int(ran),
                int(batches),
                float(neb_wall),
                float(avg),
                float(struct_wall),
                float(setup_wall),
                float(preprocess_wall),
                float(preprocess_avg),
                float(wait_wall),
                float(engine_total_wall),
                float(engine_avg),
                float(total_with_preprocess),
                float(total_with_preprocess_avg),
                float(engine_stage_sums.get("header_s", 0.0)),
                float(engine_stage_sums.get("endpoint_constraints_s", 0.0)),
                float(engine_stage_sums.get("endpoint_opt_s", 0.0)),
                float(engine_stage_sums.get("image_setup_s", 0.0)),
                float(engine_stage_sums.get("image_constraints_s", 0.0)),
                float(engine_stage_sums.get("interpolate_s", 0.0)),
                float(engine_stage_sums.get("optimizer_s", 0.0)),
                float(engine_stage_sums.get("energy_eval_s", 0.0)),
                float(engine_stage_sums.get("failure_cleanup_s", 0.0)),
                float(avg_atoms),
                float(avg_fe),
                float(avg_h),
                float(avg_frozen),
            ])

    def _should_periodic_dump(current_step: int) -> bool:
        """Dump every N steps and always include the final step."""
        return ((current_step + 1) % DUMP_EVERY_STEPS == 0) or (current_step == steps - 1)

    # Validation outputs are staged in-memory during the KMC loop and flushed after the loop.
    # This decouples heavy filesystem writes from step-level MPI collectives.
    staged_validation = {}  # (step, h) -> record dict

    def _stage_validation_record(step_idx, h_idx, rel_init, cand_ns, neigh_xyz, neighbor_hp, barrier_items=None):
        key = (int(step_idx), int(h_idx))
        rec = staged_validation.get(key)
        if rec is None:
            rec = {
                "rel_initial": None,
                "neighbors_idx": [],
                "neighbors_xyz": [],
                "neighbor_h_positions": [],
                "barriers": {},
            }
            staged_validation[key] = rec

        if rel_init is not None and rec["rel_initial"] is None:
            rec["rel_initial"] = _ensure_numeric_coords(rel_init, f"staged rel_initial for h={h_idx}").copy()

        if cand_ns:
            rec["neighbors_idx"] = [int(v) for v in cand_ns]
        if neigh_xyz:
            rec["neighbors_xyz"] = [
                _ensure_numeric_coords(v, f"staged neighbors_xyz for h={h_idx}").copy()
                for v in neigh_xyz
            ]

        if neighbor_hp:
            merged = []
            seen = set()
            for entry in rec["neighbor_h_positions"]:
                arr, idx = _neighbor_entry_to_pos(entry, f"staged neighbor H existing for h={h_idx}")
                if arr is None:
                    continue
                k = tuple(np.round(arr, 8))
                if k in seen:
                    continue
                seen.add(k)
                merged.append((arr.copy(), idx))
            for entry in neighbor_hp:
                arr, idx = _neighbor_entry_to_pos(entry, f"staged neighbor H new for h={h_idx}")
                if arr is None:
                    continue
                k = tuple(np.round(arr, 8))
                if k in seen:
                    continue
                seen.add(k)
                merged.append((arr.copy(), idx))
            rec["neighbor_h_positions"] = merged

        if barrier_items:
            bmap = rec["barriers"]
            for n_j, b_j in barrier_items:
                bmap[int(n_j)] = float(b_j)

    # --- Local (per-rank) rate reuse state across steps ---
    # Reuse never leaves the rank that owns a given h.
    prev_rates_local = {}   # h -> list[(n, barrier_eV, rate_Hz)]
    affected_h_next = None  # (rank-local) set of h to recompute next step
    last_move_endpoints = None  # (h_old, h_new) for diagnostics/metrics
    pending_mixed_share = {}  # key -> barrier value accumulated since last global share
    failed_barrier_keys = set()  # env keys that already produced a non-finite barrier this run
    reported_failed_rate_keys = set()  # rate CSV dedupe for repeated failed envs on this rank
    rate_status_totals = defaultdict(int)
    rate_rows_total = 0
    use_cache_for_jobs = JOB_ASSIGNMENT_MODE != "all_jobs"
    dedupe_jobs_enabled = JOB_ASSIGNMENT_MODE == "cache_dedupe"
    for step in range(steps):
        # On first loop entry, write an INIT row capturing pre-loop wall time.
        if DEBUG_LOGGING and not init_row_written:
            init_wall = time.time() - prog_wall_t0
            zeros = ",".join(["0.000000"] * perstep_bucket_count)
            with open(timing_csv, "a") as f:
                f.write(f"-1,{init_wall:.6f},{zeros},0.000000,{init_wall:.6f}\n")
            init_row_written = True
        # Per-step timers
        t0_step = time.time()
        t_build_struct = 0.0
        t_build_tree = 0.0
        t_move_loop = 0.0
        t_batch_extract_env = 0.0
        t_cache_lookup = 0.0
        t_barrier_neb = 0.0
        t_selection_comm = 0.0
        t_logging = 0.0
        t_extract_env = 0.0
        t_validation = 0.0
        t_job_comm = 0.0
        t_job_schedule = 0.0
        t_neb_struct = 0.0
        t_result_comm = 0.0
        t_result_merge = 0.0
        t_move_reduce = 0.0
        t_cache_update = 0.0
        t_cache_save = 0.0
        t_rate_comm = 0.0
        t_rate_io = 0.0
        t_state_comm = 0.0
        t_state_update = 0.0
        t_dump_diag = 0.0
        t_dump_traj = 0.0
        t_neighbor_sync = pending_neighbor_sync
        pending_neighbor_sync = 0.0
        t_timing_append = 0.0

        step_hits = 0
        step_misses = 0
        step_end = time.time()
        step_duration = step_end - step_start
        step_wall_times.append(step_duration)  # <-- this line was missing
        step_start = time.time()  # reset for next step
        log(f"[Rank {rank}] Step {step} time: {step_duration:.2f} s", step)
        log(f"Starting KMC Step {step}", step=step, banner=True)
        _neb_diag_write_header_if_needed(step)
        H_set = set(H_indices)
        local_moves = []
        local_rates = []

        h_move_log = {}  # {h: [(n, barrier_string), ...]}
        t0 = time.time()
        H_positions = sites[H_indices]  # (num_H, 3) — direct, zero overhead
        H_cell_index = _build_h_cell_index(H_positions, H_indices, a)
        t_build_struct += time.time() - t0  # keep timing; now very small
        t0 = time.time()
        H_tree = cKDTree(H_positions, boxsize=np.array([a*nx, a*ny, a*nz], float))
        t_build_tree += time.time() - t0
        # -------------------------- NEB DUMP MODE --------------------------
        if NEB_DUMP:
            dumped = 0
            H_set = set(H_indices)

            # absolute, de-normalized rows
            low_rows  = []  # (h,n,barrier,pre_len,pre_span,pre_e2e,pre_mean,pre_max,pre_min,pre_stuck,pre_clump,
                            #  post_len,post_span,post_e2e,post_mean,post_max,post_min,drms,deltaL)
            norm_rows = []

            def _abs_metrics(frames, mover_idx):
                """Absolute path metrics that flag clumping without normalization."""
                cell3 = np.array([3*a, 3*a, 3*a], dtype=float)
                pos   = [frm[mover_idx].position for frm in frames]
                if len(pos) < 2:
                    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1)
                seg = [float(np.linalg.norm(pbc_diff(pos[i+1], pos[i], cell3)))
                       for i in range(len(pos)-1)]
                total = float(sum(seg))
                e2e   = float(np.linalg.norm(pbc_diff(pos[-1], pos[0], cell3)))
                span  = float(max((np.linalg.norm(pbc_diff(p, pos[0], cell3))
                                   for p in pos[1:-1]), default=0.0))
                mean  = float(np.mean(seg)) if seg else 0.0
                smax  = float(np.max(seg))  if seg else 0.0
                smin  = float(np.min(seg))  if seg else 0.0

                # clump heuristics: tiny hop + all interior near start, or uniform-tiny segments
                eps   = max(1e-3, 0.05*e2e)
                stuck = int(sum(1 for p in pos[1:-1]
                                if np.linalg.norm(pbc_diff(p, pos[0], cell3)) <= eps))
                uniform_tiny = (smin > 0 and (smax/smin) < 1.05 and e2e < 0.05)
                clump = int((e2e < 0.02 and stuck >= max(1, len(pos)-2)) or uniform_tiny)
                return total, span, e2e, mean, smax, smin, stuck, clump

            for h in local_H:
                for n in neighbors[h]:
                    if n in H_set or n == h:
                        continue

                    # Define initial/final and build 3x3x3 local boxes
                    rel_initial, rel_final, env_key, neighbor_pos = extract_local_environment(
                        None, h, n, sites, a, rank, step,
                        radius=ENV_RADIUS_A, H_tree=H_tree, H_positions=H_positions,
                    )
                    initial, final, hloc = build_local_neb_structures(
                        full_structure,
                        rel_initial,
                        rel_final,
                        a,
                        rank,
                        step,
                        neighbor_positions=neighbor_pos,
                    )
                    if initial is None or final is None:
                        log(f"[Rank {rank}] NEB_DUMP skip: could not build local structures for {h}->{n}", step)
                        continue

                    # Run NEB with diagnostics (gives images_pre & images_post)
                    bar, diag = barrier(initial, final, calc_pool, rank,
                    return_images=False, collect_diagnostics=True, mover_idx=hloc)

                    # Absolute metrics on PRE/POST bands
                    pre_len,  pre_span,  pre_e2e,  pre_mean,  pre_max,  pre_min,  pre_stuck,  pre_clump  = _abs_metrics(diag["images_pre"],  diag["mover_idx"])
                    post_len, post_span, post_e2e, post_mean, post_max, post_min, _,          _           = _abs_metrics(diag["images_post"], diag["mover_idx"])

                    # Interior RMS deviation pre↔post
                    cell3 = np.array([3*a, 3*a, 3*a], dtype=float)
                    pre_pos  = [frm[diag["mover_idx"]].position for frm in diag["images_pre"]]
                    post_pos = [frm[diag["mover_idx"]].position for frm in diag["images_post"]]
                    if len(pre_pos) == len(post_pos) and len(pre_pos) >= 3:
                        idxs = range(1, len(pre_pos)-1)
                    else:
                        idxs = range(min(len(pre_pos), len(post_pos)))
                    devs = [np.linalg.norm(pbc_diff(pre_pos[i], post_pos[i], cell3)) for i in idxs] if idxs else []
                    drms = float(np.sqrt(np.mean(np.square(devs)))) if devs else 0.0
                    dL   = float(post_len - pre_len)

                    # Write PRE/POST trajectories; moving H is id=1
                    pre_path  = f"neb_pre_rank{rank}_step{step}_h{h}_n{n}.traj"
                    post_path = f"neb_post_rank{rank}_step{step}_h{h}_n{n}.traj"
                    write_neb_lammpstrj(
                        diag["images_pre"],
                        pre_path,
                        mover_idx=diag["mover_idx"],
                        rank=rank,
                    )
                    write_neb_lammpstrj(
                        diag["images_post"],
                        post_path,
                        mover_idx=diag["mover_idx"],
                        rank=rank,
                    )
                    dumped += 1

                    # Sanity: compare full-cell vs 3a-cell hop length
                    D_global = float(np.linalg.norm(
                        pbc_diff(rel_initial, rel_final, np.array([a*nx, a*ny, a*nz], dtype=float))
                    ))
                    D_3a = float(np.linalg.norm(
                        pbc_diff(diag["final_post"][diag["mover_idx"]].position,
                                 diag["init_post"][diag["mover_idx"]].position, cell3)
                    ))

                    # Classify by barrier
                    is_low = np.isfinite(bar) and (bar < LOW_BARRIER_EV)
                    row = (h, n, float(bar),
                           pre_len,  pre_span,  pre_e2e,  pre_mean,  pre_max,  pre_min,  pre_stuck, pre_clump,
                           post_len, post_span, post_e2e, post_mean, post_max, post_min,
                           drms, dL)

                    if is_low:
                        low_rows.append(row);  tag = "LOW";  which = "neb_dump_low.txt"
                    else:
                        norm_rows.append(row); tag = "NORM"; which = "neb_dump_norm.txt"

                    # Clear, compact log per move
                    log(
                        f"[Rank {rank}] NEB_DUMP {tag} h={h} n={n} | barrier={bar:.6f} eV | "
                        f"D_global={D_global:.4f} Å vs D_3a={D_3a:.4f} Å | "
                        f"pre_e2e={pre_e2e:.4f} Å pre_len={pre_len:.4f} Å pre_clump={pre_clump} | "
                        f"drms={drms:.4f} Å ΔL={dL:.4f} Å | "
                        f"PRE={pre_path} POST={post_path} -> {which}",
                        step
                    )

            # Gather and write single, sorted CSVs on rank 0
            all_low  = comm.gather(low_rows,  root=0)
            all_norm = comm.gather(norm_rows, root=0)

            if rank == 0:
                low_flat  = [r for part in all_low  for r in part]
                norm_flat = [r for part in all_norm for r in part]
                low_flat.sort(key=lambda r: r[2])   # sort by barrier
                norm_flat.sort(key=lambda r: r[2])

                header = ("h,n,barrier_eV,"
                          "pre_len_A,pre_span_A,pre_e2e_A,pre_meanseg_A,pre_maxseg_A,pre_minseg_A,pre_stuck_ct,pre_clump,"
                          "post_len_A,post_span_A,post_e2e_A,post_meanseg_A,post_maxseg_A,post_minseg_A,"
                          "drms_A,deltaL_A\n")

                with open("neb_dump_low.txt", "w") as f:
                    f.write(header)
                    for r in low_flat:
                        f.write(",".join([
                            str(r[0]), str(r[1]),
                            f"{r[2]:.6f}",
                            f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}",
                            f"{r[6]:.6f}", f"{r[7]:.6f}", f"{r[8]:.6f}",
                            str(r[9]), str(r[10]),
                            f"{r[11]:.6f}", f"{r[12]:.6f}", f"{r[13]:.6f}",
                            f"{r[14]:.6f}", f"{r[15]:.6f}", f"{r[16]:.6f}",
                            f"{r[17]:.6f}", f"{r[18]:.6f}"
                        ]) + "\n")

                with open("neb_dump_norm.txt", "w") as f:
                    f.write(header)
                    for r in norm_flat:
                        f.write(",".join([
                            str(r[0]), str(r[1]),
                            f"{r[2]:.6f}",
                            f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}",
                            f"{r[6]:.6f}", f"{r[7]:.6f}", f"{r[8]:.6f}",
                            str(r[9]), str(r[10]),
                            f"{r[11]:.6f}", f"{r[12]:.6f}", f"{r[13]:.6f}",
                            f"{r[14]:.6f}", f"{r[15]:.6f}", f"{r[16]:.6f}",
                            f"{r[17]:.6f}", f"{r[18]:.6f}"
                        ]) + "\n")

                log(f"[Rank 0] NEB_DUMP summary: {len(low_flat)} LOW rows -> neb_dump_low.txt, "
                    f"{len(norm_flat)} NORM rows -> neb_dump_norm.txt", step)

            log(f"[Rank {rank}] NEB_DUMP complete: wrote {dumped*2} bands (pre+post) for {len(local_H)} local H.", step)

            # finish NEB_DUMP in a single step
            comm.Barrier()
            if rank == 0:
                log("[Rank 0] NEB_DUMP: one-step dump done; exiting.", step)
            # Totals for NEB_DUMP early-exit
            wall_total = time.time() - prog_wall_t0
            if rank == 0:
                log(f"[Rank 0] TOTAL KMC SIMULATED TIME: {sim_time:.6e} s", step)
                log(f"[Rank 0] TOTAL WALL TIME: {wall_total:.2f} s", step)
            try:
                MPI.Finalize()
            except Exception as e:
                log(f"[Rank {rank}] MPI Finalize failed: {e}", step)
            return
        # ----------------------- END NEB DUMP MODE ------------------------

        t0_loop = time.time()
        local_rate_rows = []   # rows: (step, rank, h, n, barrier, rate, source, env_kind, status)
        validation_barriers = {}
        validation_neighbors_map = {}
        validation_neighbor_hpos = {}
        # Accumulate fresh rows this step; merge into prev_rates_local after selection.
        step_prev_local = {}   # h -> list[(n, barrier_eV, rate_Hz)]
        local_miss_jobs = []  # [(key, (src_rank, h, n)), ...]

        # --- NEB diagnostics (per-step accumulators) ---
        neb_diag_assigned = 0
        neb_diag_ran = 0
        neb_diag_batches = 0
        neb_diag_neb_wall = 0.0
        neb_diag_struct_wall = 0.0
        neb_diag_setup_wall = 0.0
        neb_diag_engine_total_wall = 0.0
        neb_diag_engine_stage_sums = defaultdict(float)
        neb_diag_atom_stats_sums = defaultdict(float)
        # we'll compute neb_wait_s after an allreduce each step
        neb_diag_wait_wall = 0.0
        for h in local_H:
            if h not in H_indices:
                log(f"[Rank {rank}] WARNING: Skipping H index {h} not in H_indices")
                continue
            move_entries = []
            # --- Fast path: reuse last-step rates if this H is outside the affected region ---
            if (affected_h_next is not None) and (h not in affected_h_next) and (h in prev_rates_local):
                for (n_prev, b_prev, r_prev) in prev_rates_local.get(h, []):
                    if not (np.isfinite(r_prev) and r_prev > 0.0):
                        continue
                    if n_prev in H_set:
                        # Skip moves whose target is now occupied.
                        continue
                    local_moves.append((h, n_prev))
                    local_rates.append(r_prev)
                    move_entries.append((n_prev, "R"))
                    barrier_val = float(b_prev) if np.isfinite(b_prev) else float("inf")
                    local_rate_rows.append((step, rank, h, n_prev, barrier_val, float(r_prev), "reuse", "reuse", "ok"))
                    step_prev_local.setdefault(h, []).append((n_prev, float(b_prev), float(r_prev)))
                h_move_log[h] = move_entries
                continue
            cand_ns = [n for n in neighbors[h] if n not in H_set and n != h]
            if VALIDATION_MODE and h not in validation_neighbors_map:
                validation_neighbors_map[h] = (list(cand_ns), [sites[idx] for idx in cand_ns])
            t0 = time.time()
            rel_initial_b, finals_by_n_b, keys_by_n_b, neighbor_pos_b = batch_extract_local_environment(
                h, cand_ns, sites, a, rank, step,
                radius=ENV_RADIUS_A, H_tree=H_tree, H_positions=H_positions
            )
            t_batch_extract_env += time.time() - t0
            if VALIDATION_MODE:
                _merge_validation_neighbor_hpos(validation_neighbor_hpos, h, neighbor_pos_b)
            for n in neighbors[h]:
                if n in H_set or n == h:
                    continue
                rel_initial = None
                rel_final = None
                neighbor_pos = None
                try:
                    t0 = time.time()
                    if rel_initial_b is not None and n in finals_by_n_b:
                        rel_initial = rel_initial_b
                        rel_final = finals_by_n_b[n]
                        env_key = keys_by_n_b[n]
                        neighbor_pos = neighbor_pos_b
                    else:
                        rel_initial, rel_final, env_key, neighbor_pos = extract_local_environment(
                            None, h, n, sites, a, rank, step,
                            radius=ENV_RADIUS_A, H_tree=H_tree, H_positions=H_positions,
                        )
                        if VALIDATION_MODE:
                            _merge_validation_neighbor_hpos(validation_neighbor_hpos, h, neighbor_pos)
                    t_extract_env += time.time() - t0
                    key = env_key
                    if key in failed_barrier_keys:
                        move_entries.append((n, "F"))
                        if key not in reported_failed_rate_keys:
                            local_rate_rows.append((step, rank, h, n, float("inf"), float("nan"), "failed_env", "failed_env", "fail"))
                            reported_failed_rate_keys.add(key)
                        continue
                    if use_cache_for_jobs:
                        t0_lookup = time.time()
                        if key in cache:
                            b = cache[key]
                            step_hits += 1
                            t_cache_lookup += time.time() - t0_lookup

                            if not np.isfinite(b):
                                move_entries.append((n, "C"))
                                if key not in reported_failed_rate_keys:
                                    local_rate_rows.append((step, rank, h, n, float("nan"), float("nan"), "cache", "cache", "bad_cached"))
                                    reported_failed_rate_keys.add(key)
                                log(f"[Rank {rank}] Cached barrier is not finite for move {h}->{n}", step)
                                failed_barrier_keys.add(key)
                                continue

                            source = "cache"
                            env_kind = "cache"
                            rate = nu * np.exp(-b / (k_B * T))
                            local_moves.append((h, n))
                            local_rates.append(rate)
                            move_entries.append((n, "C"))
                            barrier_val = float(b) if np.isfinite(b) else float("inf")
                            local_rate_rows.append((step, rank, h, n, barrier_val, float(rate), source, env_kind, "ok"))
                            step_prev_local.setdefault(h, []).append((n, float(barrier_val), float(rate)))
                            continue
                        step_misses += 1
                        t_cache_lookup += time.time() - t0_lookup
                    else:
                        step_misses += 1
                    local_miss_jobs.append((key, (rank, h, n)))
                except Exception as e:
                    move_entries.append((n, "error"))
                    log(f"[Rank {rank}] NEB error for move {h}->{n}: {e}", step)
                    try:
                        initial, final, hloc = build_local_neb_structures(
                            full_structure,
                            rel_initial,
                            rel_final,
                            a,
                            rank,
                            step,
                            neighbor_positions=neighbor_pos,
                        )
                        dump_neb_failure_debug(h, n, initial, final, rank, step)
                    except Exception:
                        pass
                    # >>> rate-csv: exception case
                    local_rate_rows.append((step, rank, h, n, float("nan"), float("nan"), "neb", "unique", "error"))
            h_move_log[h] = move_entries
        # >>>>>> NEW: GLOBAL NEB JOB POOL (collect -> dedupe -> assign -> compute -> merge -> broadcast)

        # --- Validation: stage initial geometry metadata once per H with misses on this rank
        if VALIDATION_MODE and local_miss_jobs:
            validation_seq = next(_MPI_SEQ_COUNTER)
            t0_validation = time.time()
            try:
                if DEBUG_LOGGING:
                    log(f"[Rank {rank}] Validation stage (pre-NEB): start (seq={validation_seq}).", step)
                seen_h = set()
                for _key_m, (_src_rank_m, h_m, _n_m) in local_miss_jobs:
                    if h_m in seen_h:
                        continue
                    seen_h.add(h_m)
                    rel_initial_m = _ensure_numeric_coords(sites[h_m], f"site[{h_m}] initial validation")
                    cand_ns, neigh_xyz = validation_neighbors_map.get(h_m, ([], []))
                    neighbor_h = validation_neighbor_hpos.get(h_m, [])
                    if not neighbor_h:
                        neighbor_h = _collect_validation_neighbor_h_positions(rel_initial_m, h_m)
                    _stage_validation_record(
                        step,
                        h_m,
                        rel_initial_m,
                        cand_ns,
                        neigh_xyz,
                        neighbor_h,
                        barrier_items=None,
                    )
                if DEBUG_LOGGING:
                    log(f"[Rank {rank}] Validation stage (pre-NEB): end (seq={validation_seq}).", step)
            except Exception as _e:
                log(f"[Rank {rank}] Pre-NEB validation staging failed: {_e}", step)
            finally:
                t_validation += time.time() - t0_validation

        # 1) gather all miss jobs on root
        miss_gather_seq = next(_MPI_SEQ_COUNTER)
        if DEBUG_LOGGING:
            try:
                log(f"[Rank {rank}] Miss-job gather: start (seq={miss_gather_seq}), count={len(local_miss_jobs)}.", step)
            except Exception:
                pass
        t0_comm = time.time()
        phase_guard(comm, rank, step, "MISS_GATHER")
        try:
            all_miss_jobs = comm.gather(local_miss_jobs, root=0)
        except Exception as _miss_gather_exc:
            if DEBUG_LOGGING:
                try:
                    log(f"[Rank {rank}] ERROR miss-job gather (seq={miss_gather_seq}): {_miss_gather_exc}", step)
                except Exception:
                    pass
            all_miss_jobs = [] if rank == 0 else None
        t_job_comm += time.time() - t0_comm
        if DEBUG_LOGGING:
            try:
                if rank == 0:
                    count_parts = len(all_miss_jobs) if all_miss_jobs is not None else 0
                    log(f"[Rank 0] Miss-job gather: received from {count_parts} rank(s) (seq={miss_gather_seq}).", step)
                else:
                    log(f"[Rank {rank}] Miss-job gather: returned (seq={miss_gather_seq}).", step)
            except Exception:
                pass

        key_request_ranks = None
        if rank == 0:
            t0_sched = time.time()
            flat_jobs = [jp for part in all_miss_jobs for jp in part]          # [(key,payload), ...]
            key_request_ranks = {}
            for key_req, (src_rank_req, _h_req, _n_req) in flat_jobs:
                key_request_ranks.setdefault(key_req, set()).add(int(src_rank_req))
            if dedupe_jobs_enabled:
                global_jobs = dedupe_jobs(flat_jobs)
            else:
                global_jobs = flat_jobs
            if not global_jobs:
                assignments = {r: [] for r in range(size)}
            else:
                assignments = assign_jobs_round_robin(global_jobs, size)
            scatter_payload = [assignments[r] for r in range(size)]
            t_job_schedule += time.time() - t0_sched
        else:
            scatter_payload = None

        # 2) scatter assignments: each rank gets its chunk of [(key, payload), ...]
        t0_comm = time.time()
        scatter_seq = next(_MPI_SEQ_COUNTER)
        if DEBUG_LOGGING:
            try:
                if rank == 0:
                    sizes = [len(payload) for payload in scatter_payload]
                    log(f"[Rank 0] Miss-job scatter: payload sizes={sizes} (seq={scatter_seq}).", step)
                else:
                    log(f"[Rank {rank}] Miss-job scatter: awaiting payload (seq={scatter_seq}).", step)
            except Exception:
                pass
        phase_guard(comm, rank, step, "MISS_SCATTER")
        try:
            my_jobs = comm.scatter(scatter_payload, root=0)
        except Exception as _scatter_exc:
            if DEBUG_LOGGING:
                try:
                    log(f"[Rank {rank}] ERROR miss-job scatter (seq={scatter_seq}): {_scatter_exc}", step)
                except Exception:
                    pass
            my_jobs = []
        t_job_comm += time.time() - t0_comm
        if DEBUG_LOGGING:
            try:
                log(f"[Rank {rank}] Miss-job scatter: received {len(my_jobs)} job(s) (seq={scatter_seq}).", step)
            except Exception:
                pass
        # DIAG: show how many NEB jobs each rank will run this step
        try:
            log(f"[Rank {rank}] NEB: assigned {len(my_jobs)} job(s) this step.", step)
        except Exception:
            pass

        # 3) compute assigned jobs on this rank (batch locally to avoid fragmentation)
        results_local = []   # [(key, src_rank, barrier_value), ...]
        if my_jobs:
            neb_diag_assigned = len(my_jobs)
            if len(my_jobs) <= NEB_MIN_BATCH:
                job_chunks = [my_jobs]
            else:
                job_chunks = [my_jobs[i:i + NEB_MIN_BATCH] for i in range(0, len(my_jobs), NEB_MIN_BATCH)]
            neb_diag_batches = len(job_chunks)
            for chunk_idx, chunk in enumerate(job_chunks, 1):
                if DEBUG_LOGGING:
                    try:
                        log(f"[Rank {rank}] NEB: running chunk {chunk_idx}/{len(job_chunks)} with {len(chunk)} job(s).", step)
                    except Exception:
                        pass
                # Rebuild local NEB structures from compact (h,n) jobs to avoid large MPI payloads.
                job_h_to_ns = {}
                for _key_j, (_src_rank_j, h_j, n_j) in chunk:
                    job_h_to_ns.setdefault(h_j, set()).add(n_j)
                job_env_by_h = {}
                for h_j, ns_j in job_h_to_ns.items():
                    t0 = time.time()
                    rel_i_b, finals_b, _keys_b, neighbor_pos_b = batch_extract_local_environment(
                        h_j, list(ns_j), sites, a, rank, step,
                        radius=ENV_RADIUS_A, H_tree=H_tree, H_positions=H_positions
                    )
                    t_batch_extract_env += time.time() - t0
                    job_env_by_h[h_j] = (rel_i_b, finals_b, neighbor_pos_b)

                for idx_job, (key, (src_rank_j, h_j, n_j)) in enumerate(chunk, 1):
                    key_for_log = _log_friendly(key)
                    job_wall_start = time.time()
                    dt_struct = 0.0
                    dt_barrier = 0.0
                    dt_setup = 0.0
                    bval = float("inf")
                    if DEBUG_LOGGING:
                        try:
                            log(f"[Rank {rank}] NEB: starting job {idx_job}/{len(chunk)} (key={key_for_log}, h={h_j}, n={n_j}).", step)
                        except Exception:
                            pass
                    rel_i_j = None
                    rel_f_j = None
                    neighbor_pos_j = None
                    rel_i_b, finals_b, neighbor_pos_b = job_env_by_h.get(h_j, (None, {}, None))
                    if rel_i_b is not None and n_j in finals_b:
                        rel_i_j = rel_i_b
                        rel_f_j = finals_b[n_j]
                        neighbor_pos_j = neighbor_pos_b
                    else:
                        try:
                            t0 = time.time()
                            rel_i_j, rel_f_j, _tmp_key, neighbor_pos_j = extract_local_environment(
                                None, h_j, n_j, sites, a, rank, step,
                                radius=ENV_RADIUS_A, H_tree=H_tree, H_positions=H_positions,
                            )
                            t_extract_env += time.time() - t0
                        except Exception:
                            rel_i_j, rel_f_j, neighbor_pos_j = None, None, None
                    # Record neighbor distances for diagnostics before running the NEB
                    neighbor_distances = []
                    avg_distance_value = 0.0
                    neighbor_ids_for_distance = []
                    try:
                        neighbor_ids_for_distance = neighbors[h_j][:KNN_K]
                        center_pos = _ensure_numeric_coords(
                            sites[h_j], f"site[{h_j}] center for neighbor diagnostics"
                        )
                        neighbor_positions = [
                            _ensure_numeric_coords(sites[idx], f"site[{idx}] neighbor for diagnostics")
                            for idx in neighbor_ids_for_distance
                        ]
                        neighbor_distances = compute_neighbor_distances(center_pos, neighbor_positions, box_array)
                        avg_distance_value = average_distance(neighbor_distances)
                    except Exception as exc:
                        log(f"[Rank {rank}] Neighbor distance diagnostics failed for h={h_j}: {exc}", step)
                    distance_row = [step, rank, h_j, n_j]
                    for idx in range(KNN_K):
                        if idx < len(neighbor_ids_for_distance):
                            neighbor_id = neighbor_ids_for_distance[idx]
                            value = neighbor_distances[idx] if idx < len(neighbor_distances) else 0.0
                            distance_row.append(int(neighbor_id))
                            distance_row.append(f"{value:.6f}")
                        else:
                            distance_row.extend(["", ""])
                    distance_row.append(f"{avg_distance_value:.6f}")
                    try:
                        _append_distance_row(distance_row)
                    except Exception as exc:
                        log(f"[Rank {rank}] Failed to write neighbor distances row for h={h_j}, n={n_j}: {exc}", step)
                    t0_setup = time.time()
                    try:
                        t0_struct = time.time()
                        initial, final, hloc = build_local_neb_structures(
                            full_structure,
                            rel_i_j,
                            rel_f_j,
                            a,
                            rank,
                            step,
                            neighbor_positions=neighbor_pos_j,
                        )
                        if initial is None or final is None:
                            if DEBUG_LOGGING and LOCAL_ENV_MODE != "shell":
                                log(
                                    f"[Rank {rank}] fast NEB builder returned None for hop {h_j}->{n_j}; "
                                    "falling back to legacy extractor.", step
                                )
                            if LOCAL_ENV_MODE != "shell":
                                initial, final, hloc = build_neb_structures(
                                    full_structure, rel_i_j, rel_f_j, a, rank, step
                                )
                                if initial is None or final is None:
                                    log(
                                        f"[Rank {rank}] Legacy NEB builder also failed for {h_j}->{n_j}; "
                                        "skipping NEB (barrier=inf).",
                                        step,
                                    )
                        job_context = {
                            "step": step,
                            "h": h_j,
                            "n": n_j,
                            "key": key_for_log,
                            "src_rank": src_rank_j,
                            "rel_initial": _log_friendly(rel_i_j),
                            "rel_final": _log_friendly(rel_f_j),
                            "neighbor_positions": _log_friendly(neighbor_pos_j),
                        }
                        dump_neb_structure_snapshot(initial, final, hloc, job_context, rank, step)
                        dt_struct = time.time() - t0_struct
                        t_neb_struct += dt_struct
                        neb_diag_struct_wall += dt_struct
                        # --- Setup overhead before calling barrier() ---
                        # Includes endpoint building and NEB object preparation
                        dt_setup = time.time() - t0_setup
                        neb_diag_setup_wall += dt_setup
                        if initial is not None and final is not None:
                            atomic_numbers = initial.get_atomic_numbers()
                            freeze_mask = np.asarray(
                                initial.arrays.get("freeze_mask", np.zeros(len(initial), dtype=bool)),
                                dtype=bool,
                            )
                            neb_diag_atom_stats_sums["atoms"] += float(len(initial))
                            neb_diag_atom_stats_sums["fe"] += float(np.count_nonzero(atomic_numbers == 26))
                            neb_diag_atom_stats_sums["h"] += float(np.count_nonzero(atomic_numbers == 1))
                            neb_diag_atom_stats_sums["frozen"] += float(np.count_nonzero(freeze_mask))
                        dt_barrier = 0.0
                        timing_breakdown = {}
                        if initial is None or final is None:
                            bval = float("inf")
                        else:
                            t0_barrier = time.time()
                            diag = None
                            if VALIDATION_MODE and VALIDATION_ASE_DUMP:
                                result = barrier(
                                    initial,
                                    final,
                                    calc_pool,
                                    rank,
                                    return_images=False,
                                    collect_diagnostics=True,
                                    job_context=job_context,
                                    timing_out=timing_breakdown,
                                )
                                if isinstance(result, tuple) and len(result) == 2:
                                    bval, diag = result
                                else:
                                    bval = result
                                if diag and isinstance(diag, dict) and "timings" in diag:
                                    timing_breakdown.update(diag.get("timings") or {})
                                if diag:
                                    dump_validation_ase_neb(
                                        h_index=h_j,
                                        n_index=n_j,
                                        step=step,
                                        rank=src_rank_j,
                                        diag=diag,
                                        barrier_ev=bval,
                                    )
                                    dump_neb_final_trajectory(diag, job_context, rank, step)
                            else:
                                bval = barrier(
                                    initial,
                                    final,
                                    calc_pool,
                                    rank,
                                    return_images=False,
                                    collect_diagnostics=False,
                                    job_context=job_context,
                                    timing_out=timing_breakdown,
                                )
                            dt_barrier = time.time() - t0_barrier
                            t_barrier_neb += dt_barrier
                            neb_diag_neb_wall += dt_barrier
                            neb_diag_engine_total_wall += float(timing_breakdown.get("total_s", 0.0))
                            for timing_name, timing_value in timing_breakdown.items():
                                if timing_name == "total_s":
                                    continue
                                neb_diag_engine_stage_sums[timing_name] += float(timing_value)
                            neb_diag_ran += 1
                            if bval is None or not np.isfinite(bval):
                                bval = float("inf")
                        # optional per-job trace
                        if NEB_JOB_TRACE:
                            neb_job_rows.append([
                                int(step),
                                int(rank),
                                str(key),
                                float(dt_struct),
                                float(dt_setup),
                                float(dt_barrier if initial is not None and final is not None else 0.0),
                                float(timing_breakdown.get("total_s", 0.0)),
                                float(timing_breakdown.get("header_s", 0.0)),
                                float(timing_breakdown.get("endpoint_constraints_s", 0.0)),
                                float(timing_breakdown.get("endpoint_opt_s", 0.0)),
                                float(timing_breakdown.get("image_setup_s", 0.0)),
                                float(timing_breakdown.get("image_constraints_s", 0.0)),
                                float(timing_breakdown.get("interpolate_s", 0.0)),
                                float(timing_breakdown.get("optimizer_s", 0.0)),
                                float(timing_breakdown.get("energy_eval_s", 0.0)),
                                float(timing_breakdown.get("failure_cleanup_s", 0.0)),
                                float(bval if np.isfinite(bval) else 1e9),
                            ])
                    except Exception as exc:
                        log(
                            f"[Rank {rank}] NEB job failed before producing a finite barrier "
                            f"(h={h_j}, n={n_j}, key={key_for_log}): {exc}",
                            step,
                        )
                        bval = float("inf")
                    job_wall = time.time() - job_wall_start
                    barrier_str = f"{bval:.6f}" if np.isfinite(bval) else "inf"
                    if DEBUG_LOGGING:
                        try:
                            log(
                                f"[Rank {rank}] NEB: finished job {idx_job}/{len(chunk)} "
                                f"(key={key_for_log}, h={h_j}, n={n_j}) "
                                f"→ barrier={barrier_str} eV, wall={job_wall:.3f}s "
                                f"(struct={dt_struct:.3f}s, setup={dt_setup:.3f}s, neb={dt_barrier:.3f}s, "
                                f"endpoint_opt={timing_breakdown.get('endpoint_opt_s', 0.0):.3f}s, "
                                f"interpolate={timing_breakdown.get('interpolate_s', 0.0):.3f}s, "
                                f"optimize={timing_breakdown.get('optimizer_s', 0.0):.3f}s, "
                                f"energy={timing_breakdown.get('energy_eval_s', 0.0):.3f}s).",
                                step,
                            )
                        except Exception:
                            pass
                    results_local.append((key, int(src_rank_j), float(bval)))

        # 4) collect results according to barrier merge mode
        if BARRIER_MERGE_MODE == "global":
            t0_comm = time.time()
            allgather_seq = next(_MPI_SEQ_COUNTER)
            if DEBUG_LOGGING:
                try:
                    local_ser_len = len(pickle.dumps(results_local, protocol=4))
                except Exception:
                    local_ser_len = -1
                try:
                    log(
                        f"[Rank {rank}] Post-NEB allgather: contributing {len(results_local)} result(s), "
                        f"pickle_bytes={local_ser_len} (seq={allgather_seq}).",
                        step,
                    )
                except Exception:
                    pass
            phase_guard(comm, rank, step, "POSTNEB_ALLGATHER_RESULTS")
            try:
                all_results = comm.allgather(results_local)
            except Exception as _allgather_exc:
                if DEBUG_LOGGING:
                    try:
                        log(
                            f"[Rank {rank}] ERROR Post-NEB allgather (seq={allgather_seq}): {_allgather_exc}",
                            step,
                        )
                    except Exception:
                        pass
                all_results = []
            t_result_comm += time.time() - t0_comm

            t0_merge = time.time()
            merged = {}
            for part in all_results:
                for k, _src_rank_result, v in part:
                    if k not in merged:  # first result wins (we deduped already)
                        merged[k] = v
            merged = _as_dict(merged)
            fold_results = merged
            failed_barrier_keys.update(k for k, v in merged.items() if not np.isfinite(v))
            t_result_merge += time.time() - t0_merge

            if DEBUG_LOGGING:
                try:
                    per_rank_counts = [len(p) if p is not None else -1 for p in (all_results or [])]
                    log(
                        f"[Rank {rank}] Post-NEB allgather complete (seq={allgather_seq}, "
                        f"recv_counts={per_rank_counts}, merged={len(merged)}).",
                        step,
                    )
                except Exception:
                    pass

            if use_cache_for_jobs:
                t0_cache = time.time()
                if merged:
                    cache.update((k, v) for k, v in merged.items() if np.isfinite(v))
                t_cache_update += time.time() - t0_cache
                try:
                    log(
                        f"[Rank {rank}] Post-NEB cache update: cached {len(merged)} entries "
                        f"(local misses={len(local_miss_jobs)}, merge_mode={BARRIER_MERGE_MODE}).",
                        step,
                    )
                except Exception:
                    pass
            else:
                if DEBUG_LOGGING:
                    try:
                        log(
                            f"[Rank {rank}] Post-NEB cache update skipped (mode={JOB_ASSIGNMENT_MODE}).",
                            step,
                        )
                    except Exception:
                        pass
        else:
            t0_comm = time.time()
            result_gather_seq = next(_MPI_SEQ_COUNTER)
            if DEBUG_LOGGING:
                try:
                    local_ser_len = len(pickle.dumps(results_local, protocol=4))
                except Exception:
                    local_ser_len = -1
                try:
                    log(
                        f"[Rank {rank}] Post-NEB gather(local): contributing {len(results_local)} result(s), "
                        f"pickle_bytes={local_ser_len} (seq={result_gather_seq}).",
                        step,
                    )
                except Exception:
                    pass
            phase_guard(comm, rank, step, "POSTNEB_GATHER_RESULTS_LOCAL")
            try:
                gathered_results = comm.gather(results_local, root=0)
            except Exception as _gather_exc:
                if DEBUG_LOGGING:
                    try:
                        log(
                            f"[Rank {rank}] ERROR Post-NEB gather(local) (seq={result_gather_seq}): {_gather_exc}",
                            step,
                        )
                    except Exception:
                        pass
                gathered_results = [] if rank == 0 else None
            t_result_comm += time.time() - t0_comm

            t0_merge = time.time()
            if rank == 0:
                results_by_src_rank = {r: {} for r in range(size)}
                failed_keys_global = set()
                for part in gathered_results:
                    for k, src_rank_result, v in part:
                        if not np.isfinite(v):
                            failed_keys_global.add(k)
                        target_ranks = key_request_ranks.get(k, {int(src_rank_result)}) if key_request_ranks else {int(src_rank_result)}
                        for target_rank in target_ranks:
                            rank_bucket = results_by_src_rank.setdefault(int(target_rank), {})
                            if k not in rank_bucket:
                                rank_bucket[k] = v
                scatter_results = [results_by_src_rank[r] for r in range(size)]
            else:
                scatter_results = None
                failed_keys_global = None
            t_result_merge += time.time() - t0_merge

            t0_comm = time.time()
            result_scatter_seq = next(_MPI_SEQ_COUNTER)
            if DEBUG_LOGGING:
                try:
                    if rank == 0:
                        sizes = [len(bucket) for bucket in scatter_results]
                        log(
                            f"[Rank 0] Post-NEB scatter(local): payload sizes={sizes} (seq={result_scatter_seq}).",
                            step,
                        )
                    else:
                        log(f"[Rank {rank}] Post-NEB scatter(local): awaiting payload (seq={result_scatter_seq}).", step)
                except Exception:
                    pass
            phase_guard(comm, rank, step, "POSTNEB_SCATTER_RESULTS_LOCAL")
            try:
                fold_results = comm.scatter(scatter_results, root=0)
            except Exception as _scatter_exc:
                if DEBUG_LOGGING:
                    try:
                        log(
                            f"[Rank {rank}] ERROR Post-NEB scatter(local) (seq={result_scatter_seq}): {_scatter_exc}",
                            step,
                        )
                    except Exception:
                        pass
                fold_results = {}
            t_result_comm += time.time() - t0_comm

            try:
                failed_keys_global = comm.bcast(failed_keys_global, root=0)
            except Exception:
                failed_keys_global = []
            failed_barrier_keys.update(failed_keys_global or [])

            if DEBUG_LOGGING:
                try:
                    recv_counts = [len(part) if part is not None else -1 for part in (gathered_results or [])] if rank == 0 else []
                    if rank == 0:
                        log(
                            f"[Rank 0] Post-NEB gather(local) complete (seq={result_gather_seq}, "
                            f"recv_counts={recv_counts}).",
                            step,
                        )
                    log(
                        f"[Rank {rank}] Post-NEB scatter(local) complete (seq={result_scatter_seq}, "
                        f"received={len(fold_results)}).",
                        step,
                    )
                except Exception:
                    pass

            if use_cache_for_jobs:
                t0_cache = time.time()
                if fold_results:
                    cache.update((k, v) for k, v in fold_results.items() if np.isfinite(v))
                    if BARRIER_MERGE_MODE == "mixed":
                        pending_mixed_share.update((k, v) for k, v in fold_results.items() if np.isfinite(v))
                t_cache_update += time.time() - t0_cache
                try:
                    log(
                        f"[Rank {rank}] Post-NEB cache update: cached {len(fold_results)} entries "
                        f"(local misses={len(local_miss_jobs)}, merge_mode={BARRIER_MERGE_MODE}).",
                        step,
                    )
                except Exception:
                    pass
            else:
                if DEBUG_LOGGING:
                    try:
                        log(
                            f"[Rank {rank}] Post-NEB cache update skipped (mode={JOB_ASSIGNMENT_MODE}).",
                            step,
                        )
                    except Exception:
                        pass

            if BARRIER_MERGE_MODE == "mixed" and use_cache_for_jobs:
                sync_due = ((step + 1) % BARRIER_MERGE_INTERVAL == 0) or ((step + 1) == steps)
                if sync_due:
                    t0_comm = time.time()
                    mixed_sync_seq = next(_MPI_SEQ_COUNTER)
                    if DEBUG_LOGGING:
                        try:
                            local_ser_len = len(pickle.dumps(list(pending_mixed_share.items()), protocol=4))
                        except Exception:
                            local_ser_len = -1
                        try:
                            log(
                                f"[Rank {rank}] Mixed cache sync: contributing {len(pending_mixed_share)} pending entry(s), "
                                f"pickle_bytes={local_ser_len} (seq={mixed_sync_seq}).",
                                step,
                            )
                        except Exception:
                            pass
                    phase_guard(comm, rank, step, "POSTNEB_MIXED_CACHE_SYNC")
                    try:
                        shared_parts = comm.allgather(list(pending_mixed_share.items()))
                    except Exception as _mixed_sync_exc:
                        if DEBUG_LOGGING:
                            try:
                                log(
                                    f"[Rank {rank}] ERROR Mixed cache sync (seq={mixed_sync_seq}): {_mixed_sync_exc}",
                                    step,
                                )
                            except Exception:
                                pass
                        shared_parts = []
                    t_result_comm += time.time() - t0_comm

                    t0_merge = time.time()
                    merged_shared = {}
                    for part in shared_parts:
                        for k, v in part:
                            if k not in merged_shared:
                                merged_shared[k] = v
                    t_result_merge += time.time() - t0_merge

                    t0_cache = time.time()
                    if merged_shared:
                        cache.update((k, v) for k, v in merged_shared.items() if np.isfinite(v))
                    pending_mixed_share.clear()
                    t_cache_update += time.time() - t0_cache

                    if DEBUG_LOGGING:
                        try:
                            per_rank_counts = [len(p) if p is not None else -1 for p in (shared_parts or [])]
                            log(
                                f"[Rank {rank}] Mixed cache sync complete (seq={mixed_sync_seq}, "
                                f"recv_counts={per_rank_counts}, shared={len(merged_shared)}).",
                                step,
                            )
                        except Exception:
                            pass
                else:
                    if DEBUG_LOGGING:
                        try:
                            log(
                                f"[Rank {rank}] Mixed cache sync deferred: pending={len(pending_mixed_share)}, "
                                f"next_sync_step={((step // BARRIER_MERGE_INTERVAL) + 1) * BARRIER_MERGE_INTERVAL - 1}.",
                                step,
                            )
                        except Exception:
                            pass

        # 6) fold my OWN misses into local logs/moves/rates
        #    (so move selection and NEB table include them)
        if local_miss_jobs:
            for (key, (_src_rank_m, h_m, n_m)) in local_miss_jobs:
                b = fold_results.get(key, float('inf'))
                if np.isfinite(b):
                    source = "neb_pool"
                    rate = nu * np.exp(-b / (k_B * T))
                    local_moves.append((h_m, n_m))
                    local_rates.append(rate)
                    # append into the same list we created earlier for this H
                    if h_m not in h_move_log:
                        h_move_log[h_m] = []
                    h_move_log[h_m].append((n_m, f"{b:.2f}"))
                    env_kind = "unique"
                    local_rate_rows.append((step, rank, h_m, n_m, float(b), float(rate), source, env_kind, "ok"))
                    if VALIDATION_MODE:
                        validation_barriers.setdefault(h_m, []).append((n_m, float(b)))
                    step_prev_local.setdefault(h_m, []).append((n_m, float(b), float(rate)))
                else:
                    if h_m not in h_move_log:
                        h_move_log[h_m] = []
                    h_move_log[h_m].append((n_m, "inf"))
                    local_rate_rows.append((step, rank, h_m, n_m, float("inf"), float("nan"), "neb_pool", "unique", "fail"))
        if DEBUG_LOGGING:
            try:
                log(f"[Rank {rank}] Post-NEB miss folding done (local_moves={len(local_moves)}, local_rates={len(local_rates)}).", step)
            except Exception:
                pass

        total_attempts = step_hits + step_misses
        hit_rate = (step_hits / total_attempts * 100) if total_attempts > 0 else 0.0
        log(
            f"[Rank {rank}] Step {step} barrier cache hits: {step_hits}, misses: {step_misses}, hit rate: {hit_rate:.1f}%",
            step,
        )
        # --- Global stats via MPI reduce
        if DEBUG_LOGGING:
            try:
                log(f"[Rank {rank}] Entering global barrier reduce.", step)
            except Exception:
                pass
        t0_reduce = time.time()
        global_hits = comm.reduce(step_hits, op=MPI.SUM, root=0)
        global_misses = comm.reduce(step_misses, op=MPI.SUM, root=0)
        t_move_reduce += time.time() - t0_reduce

        if rank == 0:
            global_total = global_hits + global_misses
            global_hit_rate = (
                (global_hits / global_total * 100) if global_total > 0 else 0.0
            )
            log(
                f"[Rank 0] Step {step} GLOBAL barrier cache hits: {global_hits}, misses: {global_misses}, hit rate: {global_hit_rate:.1f}%",
                step,
            )
        else:
            if DEBUG_LOGGING:
                try:
                    log(f"[Rank {rank}] Global barrier reduce complete.", step)
                except Exception:
                    pass
        t_move_loop += time.time() - t0_loop
        t_move_loop -= (
            t_batch_extract_env +
            t_extract_env +
            t_cache_lookup +
            t_barrier_neb +
            t_validation +
            t_neb_struct +
            t_job_comm +
            t_job_schedule +
            t_result_comm +
            t_result_merge +
            t_move_reduce +
            t_cache_update
        )
        if t_move_loop < 0: t_move_loop = 0.0
        if DEBUG_LOGGING:
            t0 = time.time()
            # === Log formatted table ===
            log(f"========== STEP {step} NEB Table ==========", banner=True)
            for h_idx, entries in h_move_log.items():
                log(f"H atom [{h_idx}] moves attempted:")

                padded_nbrs = [f"{n:<6}" for n, _ in entries] + [""] * (14 - len(entries))
                padded_barr = [f"{b:<6}" for _, b in entries] + [""] * (14 - len(entries))

                log("Nbrs:   " + " ".join(padded_nbrs))
                log("Barrier:" + " ".join(padded_barr))
                log("")  # spacer between atoms
            t_logging += time.time() - t0
        # ---- Gather and append rate rows (all ranks -> one CSV) ----
        if rate_csv:
            t0_comm = time.time()
            if DEBUG_LOGGING:
                try:
                    log(f"[Rank {rank}] Rate gather: sending {len(local_rate_rows)} row(s).", step)
                except Exception:
                    pass
            rate_gather_seq = None
            if VALIDATION_MODE:
                rate_gather_seq = next(_MPI_SEQ_COUNTER)
                try:
                    log(
                        f"[Rank {rank}] Validation rate gather: entering "
                        f"(seq={rate_gather_seq}, rows={len(local_rate_rows)}).",
                        step,
                    )
                except Exception:
                    pass
            try:
                all_rate_rows = comm.gather(local_rate_rows, root=0)
            except Exception as _rate_gather_exc:
                if VALIDATION_MODE:
                    try:
                        log(
                            f"[Rank {rank}] ERROR: validation rate gather failed "
                            f"(seq={rate_gather_seq}): {_rate_gather_exc}",
                            step,
                        )
                    except Exception:
                        pass
                log(f"[Rank {rank}] ERROR: rate row gather failed: {_rate_gather_exc}", step)
                raise
            else:
                if VALIDATION_MODE:
                    try:
                        if rank == 0:
                            recv_counts = [
                                len(rows) if rows is not None else -1
                                for rows in (all_rate_rows or [])
                            ]
                            log(
                                f"[Rank 0] Validation rate gather: exited "
                                f"(seq={rate_gather_seq}, recv_counts={recv_counts}).",
                                step,
                            )
                        else:
                            recv_state = "present" if all_rate_rows is not None else "None"
                            log(
                                f"[Rank {rank}] Validation rate gather: exited "
                                f"(seq={rate_gather_seq}, received={recv_state}, "
                                f"rows_sent={len(local_rate_rows)}).",
                                step,
                            )
                    except Exception:
                        pass
            t_rate_comm += time.time() - t0_comm
            if rank == 0:
                t0_io = time.time()
                per_rank_counts = [len(rows) if rows is not None else -1 for rows in all_rate_rows or []]
                row_total = sum(max(cnt, 0) for cnt in per_rank_counts)
                log(f"[Rank 0] Rate gather received per-rank counts={per_rank_counts} (total={row_total}).", step)
                try:
                    with open(rate_csv, "a") as f:
                        for rows in all_rate_rows:
                            for (stp, rnk, hi, nj, bar_ev, rate_hz, src, envk, status) in rows:
                                rate_rows_total += 1
                                rate_status_totals[str(status)] += 1
                                f.write(f"{stp},{rnk},{hi},{nj},"
                                        f"{bar_ev if np.isfinite(bar_ev) else 'nan'},"
                                        f"{rate_hz if np.isfinite(rate_hz) else 'nan'},"
                                        f"{src},{envk},{status}\n")
                except Exception as _rate_io_exc:
                    log(f"[Rank 0] ERROR writing rate CSV: {_rate_io_exc}", step)
                    raise
                else:
                    log(f"[Rank 0] Rate CSV append complete ({row_total} rows).", step)
                t_rate_io += time.time() - t0_io
        if VALIDATION_MODE:
            t0_validation = time.time()
            try:
                validation_seq = next(_MPI_SEQ_COUNTER)
                if DEBUG_LOGGING:
                    log(f"[Rank {rank}] Validation barrier stage: start (seq={validation_seq}).", step)
                for h_key, items in validation_barriers.items():
                    rel_init = _ensure_numeric_coords(sites[h_key], f"site[{h_key}] staged validation rel_init")
                    neighbor_hp = validation_neighbor_hpos.get(h_key, [])
                    if not neighbor_hp:
                        neighbor_hp = _collect_validation_neighbor_h_positions(rel_init, h_key)
                    cand_ns, neigh_xyz = validation_neighbors_map.get(h_key, ([], []))
                    _stage_validation_record(
                        step,
                        h_key,
                        rel_init,
                        cand_ns,
                        neigh_xyz,
                        neighbor_hp,
                        barrier_items=items,
                    )
                if DEBUG_LOGGING:
                    log(f"[Rank {rank}] Validation barrier stage: end (seq={validation_seq}).", step)
            except Exception as _e:
                log(f"[Rank {rank}] Validation barrier staging failed (seq={locals().get('validation_seq', 'NA')}): {_e}", step)
            finally:
                t_validation += time.time() - t0_validation
        # ------------------------------------------------------------
        # ================= CENTRALIZED GLOBAL MOVE SELECTION ===================
        # Gather local moves and rates from all ranks to rank 0
        local_move_data = list(zip(local_moves, local_rates))
        t0_comm = time.time()
        all_move_data = comm.gather(local_move_data, root=0)
        t_selection_comm += time.time() - t0_comm
        # Use the local broadcasted copy (already synchronized)
        global_H_indices = H_indices.copy()  # on all ranks; used only by rank 0
        t_selection_cpu = 0.0
        selected_move = None
        if rank == 0:
            t0_selcpu = time.time()
            try:
                global_moves = []
                global_rates = []
                for move_data in all_move_data:
                    for move, rate in move_data:
                        if np.isfinite(rate) and rate > 0.0:
                            global_moves.append(move)
                            global_rates.append(rate)
                # --- FIX 2: Sanity check before selecting move ---
                global_H_set = set(global_H_indices)
                invalid_moves = [(h, n) for (h, n) in global_moves if h not in global_H_set]
                if invalid_moves:
                    log(f"[Rank 0] WARNING: {len(invalid_moves)} invalid moves found (h not in H_indices):", step)
                    for h, n in invalid_moves:
                        log(f"  Invalid move: {h} -> {n} (h not in global_H_indices)", step)
                    # Optional: filter out invalid moves before selecting
                    filtered = [((h, n), r) for (h, n), r in zip(global_moves, global_rates) if h in global_H_set]
                    global_moves = [m for (m, r) in filtered]
                    global_rates = [r for (m, r) in filtered]

                if not global_moves:
                    dt = float('inf')
                    log("[Rank 0] No valid global moves found", step)
                    selected_move = (-1, -1)
                else:
                    total_rate = sum(global_rates)
                    dt = np.random.exponential(1.0 / total_rate)
                    (h, n) = random.choices(global_moves, weights=global_rates, k=1)[0]

                    log(f"[Rank 0] Selected move: {h} -> {n}", step)
                    log(f"[Rank 0] KMC time advanced by {dt:.3e} s", step)

                    h_in_list = np.isin(global_H_indices, h)
                    if np.any(h_in_list):
                        idx = np.where(h_in_list)[0][0]
                        global_H_indices[idx] = n
                        if H_unwrapped_positions is not None:
                            delta = pbc_diff(sites[h], sites[n], box_array)
                            H_unwrapped_positions[idx] += delta
                        selected_move = (h, n)
                        sim_time += dt
                    else:
                        log(f"[Rank 0] ERROR: H index {h} not found in global_H_indices during update.", step)
                        selected_move = (-1, -1)
                        dt = float("inf")
            except Exception as _sel_exc:
                log(f"[Rank 0] ERROR during move selection: {_sel_exc}", step)
                selected_move = (-1, -1)
                dt = float("inf")
            t_selection_cpu = time.time() - t0_selcpu

            if kmc_diag_file:
                if diag_window_start_step is None:
                    diag_window_start_step = step
                diag_window_end_step = step
                diag_window_count += 1
                diag_window_last_move = selected_move
                if np.isfinite(dt):
                    diag_window_dt_sum += float(dt)
                else:
                    diag_window_has_nonfinite_dt = True

                if _should_periodic_dump(step):
                    t0_dump_diag = time.time()
                    if (
                        diag_window_start_step is None
                        or diag_window_end_step is None
                        or diag_window_count <= 0
                    ):
                        step_label = str(step)
                    elif diag_window_start_step == diag_window_end_step:
                        step_label = str(diag_window_start_step)
                    else:
                        step_label = f"{diag_window_start_step}-{diag_window_end_step}"
                    hm, nm = diag_window_last_move
                    if diag_window_has_nonfinite_dt:
                        dt_label = "inf"
                    else:
                        dt_label = f"{diag_window_dt_sum:.5e}"
                    with open(kmc_diag_file, "a") as f:
                        f.write(
                            f"{step_label} | H index {hm} -> {nm} "
                            f"(aggregated {diag_window_count} step(s)) | "
                            f"Δt = {dt_label} s | t_total = {sim_time:.5e} s\n"
                        )
                    t_dump_diag += time.time() - t0_dump_diag
                    diag_window_start_step = None
                    diag_window_end_step = None
                    diag_window_dt_sum = 0.0
                    diag_window_count = 0
                    diag_window_last_move = (-1, -1)
                    diag_window_has_nonfinite_dt = False
        else:
            dt = None
            selected_move = None
            t_selection_cpu = 0
        # Broadcast only selected move + dt + sim_time, then apply move locally on each rank.
        t0_comm = time.time()
        if rank == 0:
            h0, h1 = (-1, -1)
            if selected_move is not None and len(selected_move) == 2:
                h0, h1 = int(selected_move[0]), int(selected_move[1])
            header = np.array(
                [float(h0), float(h1), float(dt) if dt is not None else float("inf"), float(sim_time)],
                dtype=np.float64,
            )
        else:
            header = np.zeros(4, dtype=np.float64)

        phase_guard(comm, rank, step, "POSTSEL_BCAST_HEADER_BUF")
        comm.Bcast(header, root=0)
        t_state_comm += time.time() - t0_comm

        selected_move = (int(round(header[0])), int(round(header[1])))
        dt = float(header[2])
        sim_time = float(header[3])
        t0_state_update = time.time()
        if selected_move != (-1, -1):
            h_old, h_new = selected_move
            match = np.where(H_indices == h_old)[0]
            if match.size > 0:
                moved_slot = int(match[0])
                H_indices[moved_slot] = h_new
                try:
                    update_full_structure_h_atom(full_structure, sites, moved_slot, h_new)
                except Exception as exc:
                    log(
                        f"[Rank {rank}] WARNING: full_structure in-place update failed for slot {moved_slot} "
                        f"({h_old}->{h_new}): {exc}",
                        step,
                    )
            else:
                log(f"[Rank {rank}] WARNING: selected move source {h_old} not found in local H_indices.", step)

        # Defensive consistency check to fail fast instead of hanging later collectives.
        invalid_mask = (H_indices < 0) | (H_indices >= len(sites))
        if np.any(invalid_mask):
            bad = H_indices[invalid_mask][:8].tolist()
            log(
                f"[Rank {rank}] ERROR: Invalid H_indices after post-selection update "
                f"(sample={bad}, total_bad={int(np.count_nonzero(invalid_mask))}). Aborting.",
                step,
            )
            try:
                comm.Abort(2)
            except Exception:
                raise RuntimeError("Invalid H_indices after post-selection update")
        local_H = H_indices[rank::size]

        if NEB_DIAG:
            # Avoid a post-selection allreduce to keep the collective sequence minimal.
            neb_diag_wait_wall = 0.0
            _neb_diag_append(
                step,
                neb_diag_assigned,
                neb_diag_ran,
                neb_diag_batches,
                neb_diag_neb_wall,
                neb_diag_struct_wall,
                neb_diag_setup_wall,
                neb_diag_wait_wall,
                neb_diag_engine_total_wall,
                dict(neb_diag_engine_stage_sums),
                dict(neb_diag_atom_stats_sums),
            )
            if NEB_JOB_TRACE and neb_job_rows:
                with open(neb_jobs_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(neb_job_rows)
                neb_job_rows.clear()

        # === Build affected set for next step (rank-local, no global broadcast) ===
        # Within ENV_AFFECT_RADIUS_A of old OR new site (PBC-aware).
        # Each rank computes this only for its local_H.
        if selected_move is not None and selected_move != (-1, -1):
            h_old, h_new = selected_move
            pos_old = _ensure_numeric_coords(sites[h_old], f"site[{h_old}] old move position")
            pos_new = _ensure_numeric_coords(sites[h_new], f"site[{h_new}] new move position")
            _box = np.array([a * nx, a * ny, a * nz], dtype=float)
            affected_local = set()
            Rcut = float(ENV_AFFECT_RADIUS_A)
            Rcut2 = Rcut * Rcut
            for hh in local_H:
                p = _ensure_numeric_coords(sites[hh], f"site[{hh}] for affected radius")
                d2_old = np.dot(d := pbc_diff(p, pos_old, _box), d)
                d2_new = np.dot(d := pbc_diff(p, pos_new, _box), d)
                if (d2_old <= Rcut2) or (d2_new <= Rcut2):
                    affected_local.add(hh)
            affected_h_next = affected_local
        else:
            affected_h_next = None
        last_move_endpoints = selected_move

        # === Merge this step's fresh rates into the persistent local reuse table ===
        if step_prev_local:
            for hh, rows in step_prev_local.items():
                if rows:
                    prev_rates_local[hh] = rows
                else:
                    prev_rates_local.pop(hh, None)
        
        if rank != 0:
            assert all(h < len(sites) for h in H_indices), f"[Rank {rank}] Invalid H index in H_indices"
            local_H_check = H_indices[rank::size]
            for h in local_H_check:
                if h not in H_indices:
                    log(f"[Rank {rank}] WARNING: Local H {h} not in H_indices after broadcast", step)
        t_state_update += time.time() - t0_state_update
        if rank == 0 and _should_periodic_dump(step):
            t0_traj = time.time()
            _append_h_frame(step + 1, H_indices)
            dt_traj = time.time() - t0_traj
            t_traj += dt_traj
            t_dump_traj += dt_traj
        if rank == 0 and H_unwrapped_positions is not None:
            msd_tracker.record(
                step=step + 1,
                sim_time=sim_time,
                current_positions=H_unwrapped_positions,
            )
        # ======================================================================
        # All ranks save their local barrier cache independently (append-only)
        t0_save = time.time()
        dirty = getattr(cache, "dirty_count", None)
        should_save = (dirty is None) or (dirty > 0)
        if should_save:
            cache.save()
            suffix = ""
            if isinstance(dirty, int) and dirty > 0:
                suffix = f" (+{dirty} new)"
            log(f"[Rank {rank}] Saved local barrier cache with {len(cache)} entries{suffix}")
        t_cache_save += time.time() - t0_save 

        # Optional consistency gather removed in validation/high-load runs to avoid extra collectives.
        if DEBUG_LOGGING:
            # Append per-step timings to rank-specific CSV (per-step buckets only).
            # We exclude any init-only costs (e.g., neighbor_sync) from the per-step CSV.
            placeholder = f"{0.0:14.6f}"
            # Order must mirror the CSV header following wall_total, excluding timing_append
            # until its write cost has been measured.
            perstep_metrics = {
                "build_struct": t_build_struct,
                "build_H_tree": t_build_tree,
                "move_loop": t_move_loop,
                "batch_extract_env": t_batch_extract_env,
                "cache_lookup": t_cache_lookup,
                "barrier_neb": t_barrier_neb,
                "selection_comm": t_selection_comm,
                "selection_cpu": t_selection_cpu,
                "logging": t_logging,
                "extract_env": t_extract_env,
                "validation": t_validation,
                "job_comm": t_job_comm,
                "job_schedule": t_job_schedule,
                "neb_struct": t_neb_struct,
                "result_comm": t_result_comm,
                "result_merge": t_result_merge,
                "move_reduce": t_move_reduce,
                "cache_update": t_cache_update,
                "cache_save": t_cache_save,
                "rate_comm": t_rate_comm,
                "rate_io": t_rate_io,
                "state_comm": t_state_comm,
                "state_update": t_state_update,
                "dump_diag": t_dump_diag,
                "dump_traj": t_dump_traj,
            }
            _perstep_values = [perstep_metrics[field] for field in TIMING_BUCKET_FIELDS if field != "timing_append"]
            metric_fields = ",".join([f"{v:.6f}" for v in _perstep_values])
            with open(timing_csv, "r+") as f:
                f.seek(0, os.SEEK_END)
                f.write(f"{step},")
                wall_placeholder_pos = f.tell()
                f.write(placeholder)
                f.write(",")
                f.write(metric_fields)
                f.write(",")
                timing_placeholder_pos = f.tell()
                t0_append = time.time()
                f.write(placeholder)  # timing_append placeholder; we'll overwrite
                append_elapsed = time.time() - t0_append
                t_timing_append += append_elapsed
                t_step_total_wall = time.time() - t0_step
                f.seek(timing_placeholder_pos)
                f.write(f"{t_timing_append:14.6f}")
                f.seek(wall_placeholder_pos)
                f.write(f"{t_step_total_wall:14.6f}")
                components_sum = sum(_perstep_values) + t_timing_append
                unaccounted = t_step_total_wall - components_sum
                f.seek(0, os.SEEK_END)
                f.write(f",{components_sum:.6f},{unaccounted:.6f}\n")

            # build the dict in the same order as the helper’s columns
            _step_times = {
                "build_struct":   t_build_struct,
                "build_H_tree":   t_build_tree,
                "move_loop":      t_move_loop,
                "batch_extract_env": t_batch_extract_env,
                "cache_lookup":   t_cache_lookup,
                "barrier_neb":    t_barrier_neb,
                "selection_comm": t_selection_comm,
                "selection_cpu":  t_selection_cpu,
                "logging":        t_logging,
                "extract_env":    t_extract_env,
                "validation":     t_validation,
                "job_comm":       t_job_comm,
                "job_schedule":   t_job_schedule,
                "neb_struct":     t_neb_struct,
                "result_comm":    t_result_comm,
                "result_merge":   t_result_merge,
                "move_reduce":    t_move_reduce,
                "cache_update":   t_cache_update,
                "cache_save":     t_cache_save,
                "rate_comm":      t_rate_comm,
                "rate_io":        t_rate_io,
                "state_comm":     t_state_comm,
                "state_update":   t_state_update,
                "dump_diag":      t_dump_diag,
                "dump_traj":      t_dump_traj,
                "timing_append":  t_timing_append,
            }
            log_step_timing(step, rank, _step_times, total_wall=t_step_total_wall)

    if DEBUG_LOGGING and not init_row_written:
        init_wall = time.time() - prog_wall_t0
        zeros = ",".join(["0.000000"] * perstep_bucket_count)
        with open(timing_csv, "a") as f:
            f.write(f"-1,{init_wall:.6f},{zeros},0.000000,{init_wall:.6f}\n")
        init_row_written = True

    if VALIDATION_MODE and staged_validation:
        flush_seq = next(_MPI_SEQ_COUNTER)
        try:
            if DEBUG_LOGGING:
                log(f"[Rank {rank}] Validation flush (post-loop): start (seq={flush_seq}, records={len(staged_validation)}).")
            for (stp, h_key), rec in sorted(staged_validation.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                out_dir = os.path.join(VALIDATION_OUT, f"rank{rank}_step{stp}_h{h_key}")
                _ensure_dir(out_dir)

                rel_init = rec.get("rel_initial")
                if rel_init is None:
                    rel_init = _ensure_numeric_coords(sites[h_key], f"site[{h_key}] flush rel_init")

                neighbors_idx = rec.get("neighbors_idx", []) or []
                neighbors_xyz = rec.get("neighbors_xyz", []) or []
                neighbor_hp = rec.get("neighbor_h_positions", []) or []
                if not neighbor_hp:
                    neighbor_hp = _collect_validation_neighbor_h_positions(rel_init, h_key)

                has_initial = (
                    os.path.exists(os.path.join(out_dir, "initial.data")) or
                    os.path.exists(os.path.join(out_dir, "initial.xyz"))
                )
                if (not has_initial) or VALIDATION_OVERWRITE:
                    init_atoms, _, _ = build_local_neb_structures(
                        full_structure,
                        rel_init,
                        rel_init,
                        a,
                        rank,
                        stp,
                        neighbor_positions=neighbor_hp,
                    )
                    if init_atoms is None:
                        init_atoms = extract_local_box_initial(rel_init, a)
                    dump_validation_initial_and_neighbors(
                        out_dir,
                        init_atoms,
                        h_index=h_key,
                        neighbors_xyz=neighbors_xyz,
                        neighbors_idx=neighbors_idx,
                        rel_initial=rel_init,
                        neighbor_h_positions=neighbor_hp,
                    )

                barriers = rec.get("barriers", {}) or {}
                if barriers:
                    with open(os.path.join(out_dir, "barriers.csv"), "w") as f:
                        f.write("n_site,barrier_eV\n")
                        for n_j, b_j in sorted(barriers.items()):
                            f.write(f"{int(n_j)},{float(b_j):.6f}\n")
            if DEBUG_LOGGING:
                log(f"[Rank {rank}] Validation flush (post-loop): end (seq={flush_seq}).")
        except Exception as _flush_exc:
            log(f"[Rank {rank}] Validation flush failed (seq={flush_seq}): {_flush_exc}")

    postloop_t0 = time.time()

    if NEB_DIAG:
        try:
            gathered = comm.gather(neb_diag_rows, root=0)
        except Exception:
            gathered = None
        if rank == 0 and gathered is not None:
            with open(NEB_DIAG_ALL_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "rank",
                    "neb_assigned",
                    "neb_ran",
                    "neb_batches",
                    "neb_wall_s",
                    "neb_avg_s",
                    "neb_struct_s",
                    "neb_setup_s",
                    "neb_preprocess_s",
                    "neb_preprocess_avg_s",
                    "neb_wait_s",
                    "neb_engine_total_s",
                    "neb_engine_avg_s",
                    "neb_total_with_preprocess_s",
                    "neb_total_with_preprocess_avg_s",
                    "neb_header_s",
                    "neb_endpoint_constraints_s",
                    "neb_endpoint_opt_s",
                    "neb_image_setup_s",
                    "neb_image_constraints_s",
                    "neb_interpolate_s",
                    "neb_optimize_s",
                    "neb_energy_eval_s",
                    "neb_failure_cleanup_s",
                    "neb_avg_atoms",
                    "neb_avg_fe",
                    "neb_avg_h",
                    "neb_avg_frozen",
                ])
                for part in gathered:
                    if not part:
                        continue
                    for row in sorted(part, key=lambda r: (r[0], r[1])):
                        writer.writerow(row)
    #Timer for merge barrier function
    t0 = time.time(); merge_barrier_caches(); t_merge = time.time()-t0
    t_msd  = 0.0
    if rank == 0:
        log("[Rank 0] H-only trajectory streamed to H_trajectory_onlyH.lammpstrj")
    # Timer Start for MSD
    t0 = time.time(); 
    # Compute MSD
    if rank == 0:
        assert (
            H_unwrapped_positions is not None and H_initial_unwrapped_positions is not None
        ), "Unwrapped positions must be available on rank 0"
        delta = H_unwrapped_positions - H_initial_unwrapped_positions
        displacements = np.einsum("ij,ij->i", delta, delta)
        msd = float(np.mean(displacements)) if displacements.size else 0.0
        # Convert MSD to diffusivity in m^2/s using Einstein relation (3D)
        if sim_time > 0:
            D_m2s = (msd * 1e-20) / (6.0 * sim_time)  # msd in Å^2 -> m^2, then divide by 6t
            log(f"[Rank 0] Diffusivity (Einstein, 3D): {D_m2s:.6e} m^2 s^-1 "
                f"(MSD={msd:.4f} Å^2 over t={sim_time:.6e} s)")
        else:
            log("[Rank 0] Diffusivity not computed because total KMC time is zero.")
        t_msd = time.time()-t0 if rank==0 else 0.0
    # Symmetric consistency check across all ranks
    #Timer for allgather function
    t0 = time.time(); all_H = comm.allgather(H_indices); t_allgather = time.time() - t0
    if rank == 0:
        base = all_H[0]
        mism = [i for i, hlist in enumerate(all_H) if not np.array_equal(hlist, base)]
        if len(mism) >= 1:  # >1 because rank 0 equals itself
            log(f"[Rank 0] WARNING: H_indices mismatch on ranks {mism}", step)
    def start_barrier_watchdog(comm, rank, timeout=60):
        # Arm only on rank 0 to avoid N watchdogs
        if rank != 0:
            return None
        def kill_if_stuck():
            try:
                log(f"[Rank {rank}] Barrier timeout after {timeout} seconds. Aborting MPI.")
            except Exception:
                pass
            try:
                comm.Abort(1)  # collective abort; safer than sys.exit in a thread
            except Exception:
                os._exit(1)    # last resort
        t = threading.Timer(timeout, kill_if_stuck)
        t.daemon = True
        t.start()
        return t
    #Timer Start for watchdog function
    t0 = time.time() 
    log(f"[Rank {rank}] Reached final barrier.")
    _watchdog = start_barrier_watchdog(comm, rank, timeout=60)

    try:
        comm.Barrier()  # Sync all ranks
        if _watchdog:
            _watchdog.cancel()
        if rank == 0:
            log("[Rank 0] Simulation complete. Exiting cleanly.")
    except Exception as e:
        log(f"[Rank {rank}] ERROR during final barrier: {e}")
    t_barrier = time.time()-t0
    log(f"Merging barriers:{t_merge:.6f}s")
    log(f"Writing trajectory:{t_traj:.6f}s")
    log(f"MSD calculation:{t_msd:.6f}s")
    log(f"All gather comm:{t_allgather:.6f}s")
    log(f"Watchdog:{t_barrier:.6f}s")
    # Totals: KMC simulated time and wall time
    wall_total = time.time() - prog_wall_t0
    if rank == 0:
        log(f"[Rank 0] TOTAL KMC SIMULATED TIME: {sim_time:.6e} s")
        log(f"[Rank 0] TOTAL WALL TIME: {wall_total:.2f} s")
        successful_jobs = int(rate_status_totals.get("ok", 0))
        failed_jobs = int(rate_rows_total - successful_jobs)
        status_parts = ", ".join(
            f"{status}={count}"
            for status, count in sorted(rate_status_totals.items())
        ) or "none"
        log(
            f"[Rank 0] RATE SUMMARY: total_jobs={rate_rows_total}, "
            f"successful={successful_jobs}, failed={failed_jobs}",
        )
        log(f"[Rank 0] RATE STATUS BREAKDOWN: {status_parts}")
        if H_unwrapped_positions is not None:
            msd_tracker.record(
                step=steps,
                sim_time=sim_time,
                current_positions=H_unwrapped_positions,
                force=True,
            )
        plot_success = msd_tracker.plot()
        log(f"[Rank 0] MSD log stored in {MSD_LOG_FILE}")
        if plot_success:
            log(f"[Rank 0] MSD plot stored in {MSD_PLOT_FILE}")
        else:
            log(f"[Rank 0] MSD plot not created (matplotlib unavailable or no data).")
    # Append a FINAL row capturing epilogue wall time after the per-step loop.
    try:
        finalize_wall = time.time() - postloop_t0
    except NameError:
        finalize_wall = 0.0
    if DEBUG_LOGGING:
        zeros = ",".join(["0.000000"] * perstep_bucket_count)
        try:
            final_step_index = steps
        except NameError:
            final_step_index = -2
        with open(timing_csv, "a") as f:
            f.write(
                f"{final_step_index},{finalize_wall:.6f},{zeros},0.000000,{finalize_wall:.6f}\n"
            )
    # Final check before shutdown
    try:
        MPI.Finalize()
    except Exception as e:
        log(f"[Rank {rank}] MPI Finalize failed: {e}")

    return
    
if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
