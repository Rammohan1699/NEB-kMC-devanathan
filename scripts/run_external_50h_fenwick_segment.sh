#!/usr/bin/env bash
set -euo pipefail

# Run one fixed-population Sigma5 external-map KMC segment with incremental
# event updates and Fenwick-tree event selection.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${KMC_PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PACKAGE_DIR"

if [[ -d "$PACKAGE_DIR/kmc_map_inputs" ]]; then
  DEFAULT_MAP_ROOT="$PACKAGE_DIR/kmc_map_inputs"
elif [[ -d "$PACKAGE_DIR/../../kmc_map_inputs" ]]; then
  DEFAULT_MAP_ROOT="$(cd "$PACKAGE_DIR/../../kmc_map_inputs" && pwd)"
elif [[ -d "/scratch/rtirunelveli/LAMMPS/kmc_cluster_lammps_only/kmc_map_inputs" ]]; then
  DEFAULT_MAP_ROOT="/scratch/rtirunelveli/LAMMPS/kmc_cluster_lammps_only/kmc_map_inputs"
else
  DEFAULT_MAP_ROOT="$PACKAGE_DIR/kmc_map_inputs"
fi

MAP_ROOT="${KMC_MAP_ROOT:-$DEFAULT_MAP_ROOT}"
MAP_FILE="${KMC_SITE_MAP_FILE:-$MAP_ROOT/sigma5_stage3_unified_sites.npz}"
HOST_FILE="${KMC_HOST_STRUCTURE_FILE:-$MAP_ROOT/sigma5_210-20-20-5.lmp}"
REGION_FILE="${KMC_INITIAL_H_REGION_FILE:-$MAP_ROOT/sigma5_site_regions.npz}"
POTENTIAL_FILE="${POTENTIAL_EAM_FILE:-$PACKAGE_DIR/kmc/PotentialB3410-modified.fs}"

RUN_ROOT="${RUN_ROOT:-$PACKAGE_DIR/runs/external_50H_fenwick_segment_$(date +%Y%m%d_%H%M%S)}"
MPI_RANKS="${MPI_RANKS:-${NP:-18}}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
STEPS="${STEPS:-50000}"
NUM_H="${NUM_H:-50}"
KMC_SEED="${KMC_SEED:-5001}"
KMC_INITIAL_H_REGIONS="${KMC_INITIAL_H_REGIONS:-bulk_grain_0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if (( MPI_RANKS <= 0 )); then
  echo "MPI_RANKS must be positive; got $MPI_RANKS" >&2
  exit 2
fi
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS=$MPI_RANKS must be divisible by LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS" >&2
  exit 2
fi
if (( STEPS <= 0 )); then
  echo "STEPS must be positive; got $STEPS" >&2
  exit 2
fi
if (( NUM_H != 50 )); then
  echo "This production launcher requires NUM_H=50; got $NUM_H" >&2
  exit 2
fi

for path in "$MAP_FILE" "$HOST_FILE" "$REGION_FILE" "$POTENTIAL_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing or empty: $path" >&2
    exit 2
  fi
done

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_ROOT" >&2
  exit 2
fi

export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ROOT
export MPI_RANKS
export NP="$MPI_RANKS"
export STEPS
export NUM_H
export KMC_SEED

export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE="$MAP_FILE"
export KMC_HOST_STRUCTURE_FILE="$HOST_FILE"
export KMC_HOST_FE_TYPE="${KMC_HOST_FE_TYPE:-1}"
export KMC_INITIAL_H_REGION_FILE="$REGION_FILE"
export KMC_INITIAL_H_REGIONS

# This is a fixed-population diffusion campaign. The Devanathan codebase is
# used for its incremental/Fenwick implementation, but source/sink particle
# insertion and removal are intentionally disabled.
export DEVANATHAN_ENABLED=0
export KMC_INCREMENTAL_EVENTS=1
export KMC_INCREMENTAL_IMPACT_RADIUS_A="${KMC_INCREMENTAL_IMPACT_RADIUS_A:-15.0}"

export KNN_K="${KNN_K:-6}"
export LOCAL_ENV_MODE=shell
export ENV_KEY_MODE="${ENV_KEY_MODE:-env_plus_dir}"
export ENV_RADIUS_A="${ENV_RADIUS_A:-5.0}"
export POS_BIN_A="${POS_BIN_A:-0.10}"
export HOP_BIN_A="${HOP_BIN_A:-0.02}"
export SHELL_INNER_RADIUS_A="${SHELL_INNER_RADIUS_A:-8.0}"
export SHELL_OUTER_RADIUS_A="${SHELL_OUTER_RADIUS_A:-10.0}"

export POTENTIAL="${POTENTIAL:-EAM}"
export POTENTIAL_EAM_FILE="$POTENTIAL_FILE"
export LAMMPS_FILES="${LAMMPS_FILES:-$POTENTIAL_FILE}"
export NEB_ENGINE=lammps_only
export NIMG="${NIMG:-3}"
export LAMMPS_NEB_REPLICAS
export LAMMPS_NEB_STEPS="${LAMMPS_NEB_STEPS:-1000}"
export LAMMPS_NEB_FTOL="${LAMMPS_NEB_FTOL:-1e-5}"
export LAMMPS_NEB_MIN_STYLE="${LAMMPS_NEB_MIN_STYLE:-quickmin}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"

export CACHE_SCHEMA="${CACHE_SCHEMA:-v4_lammps_only:mode:shell_IR${SHELL_INNER_RADIUS_A}_OR${SHELL_OUTER_RADIUS_A}:env_plus_dir:R5.0_PB0.1_HB0.02:map:sigma5_stage3_unified_sites}"
export BARRIER_CACHE_FILE="${BARRIER_CACHE_FILE:-}"
export JOB_ASSIGNMENT_MODE="${JOB_ASSIGNMENT_MODE:-cache_dedupe}"
export BARRIER_MERGE_MODE="${BARRIER_MERGE_MODE:-global}"
export BARRIER_MERGE_INTERVAL="${BARRIER_MERGE_INTERVAL:-100}"

export RESTART_MODE="${RESTART_MODE:-0}"
export RESTART_DIR="${RESTART_DIR:-.}"
export RESTART_STEP="${RESTART_STEP:-0}"
export RESTART_CHECKPOINT_FILE="${RESTART_CHECKPOINT_FILE:-}"
export RESTART_STRICT="${RESTART_STRICT:-1}"
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-10000}"

export DEBUG_MODE="${DEBUG_MODE:-0}"
export DEBUG_LOGGING="${DEBUG_LOGGING:-0}"
export WRITE_RATES_ALLRANKS="${WRITE_RATES_ALLRANKS:-0}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-10000}"
export MSD_LOG_INTERVAL="${MSD_LOG_INTERVAL:-10000}"

export KMC_LOG_DIR="$RUN_ROOT/logs"
export KMC_DIAGNOSTICS_DIR="$RUN_ROOT/diagnostics"
export KMC_TRAJECTORY_DIR="$RUN_ROOT/trajectories"
export KMC_CACHE_DIR="$RUN_ROOT/cache"
export RESTART_CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
export LAMMPS_NEB_SCRATCH_DIR="$RUN_ROOT/neb/lammps_only_scratch"

if [[ -n "${LAMMPS_PYTHON_PATH:-}" ]]; then
  export PYTHONPATH="$LAMMPS_PYTHON_PATH:$PYTHONPATH"
fi
if [[ -n "${LAMMPS_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$LAMMPS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export DYLD_LIBRARY_PATH="$LAMMPS_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

echo "Fixed-50H Sigma5 external-map Fenwick segment"
echo "  package=$PACKAGE_DIR"
echo "  output=$RUN_ROOT"
echo "  target_step=$STEPS restart_step=$RESTART_STEP restart_mode=$RESTART_MODE"
echo "  seed=$KMC_SEED H=$NUM_H regions=$KMC_INITIAL_H_REGIONS"
echo "  ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS groups=$((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "  map=$KMC_SITE_MAP_FILE"
echo "  region_file=$KMC_INITIAL_H_REGION_FILE"
echo "  incremental_events=$KMC_INCREMENTAL_EVENTS impact_radius_A=$KMC_INCREMENTAL_IMPACT_RADIUS_A"
echo "  cache_preload=${BARRIER_CACHE_FILE:-fresh}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p \
  "$KMC_LOG_DIR" \
  "$KMC_DIAGNOSTICS_DIR" \
  "$KMC_TRAJECTORY_DIR" \
  "$KMC_CACHE_DIR" \
  "$RESTART_CHECKPOINT_DIR" \
  "$LAMMPS_NEB_SCRATCH_DIR"

{
  printf 'PACKAGE_DIR=%s\n' "$PACKAGE_DIR"
  printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
  printf 'STEPS=%s\n' "$STEPS"
  printf 'NUM_H=%s\n' "$NUM_H"
  printf 'KMC_SEED=%s\n' "$KMC_SEED"
  printf 'KMC_INITIAL_H_REGIONS=%s\n' "$KMC_INITIAL_H_REGIONS"
  printf 'MPI_RANKS=%s\n' "$MPI_RANKS"
  printf 'LAMMPS_NEB_REPLICAS=%s\n' "$LAMMPS_NEB_REPLICAS"
  printf 'KMC_INCREMENTAL_EVENTS=%s\n' "$KMC_INCREMENTAL_EVENTS"
  printf 'KMC_INCREMENTAL_IMPACT_RADIUS_A=%s\n' "$KMC_INCREMENTAL_IMPACT_RADIUS_A"
  printf 'RESTART_MODE=%s\n' "$RESTART_MODE"
  printf 'RESTART_STEP=%s\n' "$RESTART_STEP"
  printf 'RESTART_DIR=%s\n' "$RESTART_DIR"
  printf 'RESTART_CHECKPOINT_FILE=%s\n' "$RESTART_CHECKPOINT_FILE"
  printf 'CACHE_SCHEMA=%s\n' "$CACHE_SCHEMA"
  printf 'BARRIER_CACHE_FILE=%s\n' "$BARRIER_CACHE_FILE"
  printf 'KMC_SITE_MAP_FILE=%s\n' "$KMC_SITE_MAP_FILE"
  printf 'KMC_HOST_STRUCTURE_FILE=%s\n' "$KMC_HOST_STRUCTURE_FILE"
  printf 'KMC_INITIAL_H_REGION_FILE=%s\n' "$KMC_INITIAL_H_REGION_FILE"
} > "$RUN_ROOT/run_config_snapshot.env"

"$PYTHON_BIN" - <<'PY'
import importlib.util

missing = [
    name
    for name in ("ase", "mpi4py", "lammps", "numpy", "scipy")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
print("Python dependency check OK")
PY

MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
"$MPIRUN_BIN" -np "$MPI_RANKS" "$PYTHON_BIN" -m kmc.main
