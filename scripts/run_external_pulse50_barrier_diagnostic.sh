#!/usr/bin/env bash
set -euo pipefail

# Finite 50-H charging-zone pulse for Sigma5 external-map barrier diagnostics.
# The pulse is initialized in the left charging zone, KMC hops are restricted
# to +x, and atoms are removed only when they reach the right sink.

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

if [[ -z "${PYTHON_BIN:-}" && -x "/Users/rtirunelveli/kMC_code/python_kmc/bin/python" ]]; then
  PYTHON_BIN="/Users/rtirunelveli/kMC_code/python_kmc/bin/python"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PACKAGE_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PACKAGE_DIR/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find a Python interpreter. Set PYTHON_BIN." >&2
    exit 1
  fi
fi

ENGINE_MODE="${NEB_ENGINE_MODE:-${NEB_MODE:-lammps_only}}"
case "${ENGINE_MODE}" in
  lammps_only|native_lammps|lammps-neb-only)
    ENGINE_TAG="lammps_only"
    ENGINE_VALUE="lammps_only"
    ;;
  lammps_only_endpoint_opt|lammps_only_lmp_endpoint_opt|native_lammps_endpoint_opt)
    ENGINE_TAG="lammps_only_lmp_endpoint_opt"
    ENGINE_VALUE="lammps_only_endpoint_opt"
    ;;
  ase|ase_neb|ase_lammps|lammpslib)
    ENGINE_TAG="ase_neb"
    ENGINE_VALUE="ase_lammps"
    ;;
  *)
    echo "NEB_ENGINE_MODE must be lammps_only, lammps_only_endpoint_opt, or ase_neb; got ${ENGINE_MODE}" >&2
    exit 2
    ;;
esac

MPI_RANKS="${MPI_RANKS:-${NP:-6}}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
STEPS="${STEPS:-50000}"
NUM_H="${NUM_H:-50}"
KMC_SEED="${KMC_SEED:-91050}"
DEVANATHAN_SEED="${DEVANATHAN_SEED:-$KMC_SEED}"
PULSE_REGION="${DEVANATHAN_PULSE_REGION:-${DEVANATHAN_CHARGING_ZONE_REGION:-bulk_grain_0}}"
SOURCE_X_MIN_A="${DEVANATHAN_SOURCE_X_MIN_A:-16.12518061159}"
SOURCE_X_MAX_A="${DEVANATHAN_SOURCE_X_MAX_A:-26.12518061159}"
SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-204.19906933748}"
RUN_ROOT="${RUN_ROOT:-$PACKAGE_DIR/runs/external_pulse50_barrier_${ENGINE_TAG}_$(date +%Y%m%d_%H%M%S)}"

if (( MPI_RANKS <= 0 )); then
  echo "MPI_RANKS must be positive; got $MPI_RANKS" >&2
  exit 2
fi
if [[ "$ENGINE_VALUE" == lammps_only* && $((MPI_RANKS % LAMMPS_NEB_REPLICAS)) -ne 0 ]]; then
  echo "MPI_RANKS=$MPI_RANKS must be divisible by LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS" >&2
  exit 2
fi
if (( STEPS <= 0 )); then
  echo "STEPS must be positive; got $STEPS" >&2
  exit 2
fi
if (( NUM_H != 50 )); then
  echo "This diagnostic launcher expects NUM_H=50; got $NUM_H" >&2
  exit 2
fi

for path in "$MAP_FILE" "$HOST_FILE" "$REGION_FILE" "$POTENTIAL_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing or empty: $path" >&2
    exit 2
  fi
done

if [[ -e "$RUN_ROOT" && "${ALLOW_EXISTING_RUN_ROOT:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_ROOT" >&2
  exit 2
fi

SOURCE_LAYER_A="$("$PYTHON_BIN" -c 'import sys; print(float(sys.argv[2]) - float(sys.argv[1]))' "$SOURCE_X_MIN_A" "$SOURCE_X_MAX_A")"
map_tag="$(basename "$MAP_FILE" .npz)"
region_tag="${PULSE_REGION//[^A-Za-z0-9_]/_}"

export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ROOT
export MPI_RANKS
export NP="$MPI_RANKS"
export STEPS
export NUM_H
export KMC_SEED
export DEVANATHAN_SEED

export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE="$MAP_FILE"
export KMC_HOST_STRUCTURE_FILE="$HOST_FILE"
export KMC_HOST_FE_TYPE="${KMC_HOST_FE_TYPE:-1}"
export KMC_INITIAL_H_REGION_FILE="$REGION_FILE"
export KMC_INITIAL_H_REGIONS="$PULSE_REGION"

export DEVANATHAN_ENABLED=1
export DEVANATHAN_SOURCE_MODE=pulse
export DEVANATHAN_PULSE_N_H="$NUM_H"
export DEVANATHAN_PULSE_REGION="$PULSE_REGION"
export DEVANATHAN_SOURCE_X_MIN_A="$SOURCE_X_MIN_A"
export DEVANATHAN_SOURCE_X_MAX_A="$SOURCE_X_MAX_A"
export DEVANATHAN_SOURCE_LAYER_A="$SOURCE_LAYER_A"
export DEVANATHAN_SOURCE_FRACTION="${DEVANATHAN_SOURCE_FRACTION:-0.0}"
export DEVANATHAN_SINK_X_MIN_A="$SINK_X_MIN_A"
export DEVANATHAN_LEFT_SINK_X_MAX_A="${DEVANATHAN_LEFT_SINK_X_MAX_A:-0}"
export DEVANATHAN_TRIM_SOURCE_EXCESS=0
export KMC_HOP_X_DIRECTION=right
export DEVANATHAN_RIGHT_DIRECTION_ONLY=1
export KMC_STOP_WHEN_NO_H=1

export KMC_INCREMENTAL_EVENTS="${KMC_INCREMENTAL_EVENTS:-0}"
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
export NEB_ENGINE="$ENGINE_VALUE"
export NIMG="${NIMG:-3}"
export LAMMPS_NEB_REPLICAS
export LAMMPS_NEB_STEPS="${LAMMPS_NEB_STEPS:-1000}"
export LAMMPS_NEB_FTOL="${LAMMPS_NEB_FTOL:-1e-5}"
export LAMMPS_NEB_MIN_STYLE="${LAMMPS_NEB_MIN_STYLE:-quickmin}"
# Keep the compact rank/rate diagnostics by default, but do not retain every
# native LAMMPS NEB scratch directory unless explicitly requested.
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"
export LAMMPS_NEB_MIN_OPEN_FILES="${LAMMPS_NEB_MIN_OPEN_FILES:-8192}"

if [[ "$ENGINE_VALUE" == lammps_only* ]]; then
  current_open_files="$(ulimit -n)"
  if [[ "$current_open_files" != "unlimited" && "$current_open_files" -lt "$LAMMPS_NEB_MIN_OPEN_FILES" ]]; then
    if ! ulimit -n "$LAMMPS_NEB_MIN_OPEN_FILES" 2>/dev/null; then
      echo "WARNING: could not raise open-file limit from ${current_open_files} to ${LAMMPS_NEB_MIN_OPEN_FILES}; current limit is $(ulimit -n)" >&2
    fi
  fi
  export LAMMPS_NEB_OPEN_FILE_LIMIT="$(ulimit -n)"
fi

export OPTIMIZE_ENDPOINTS="${OPTIMIZE_ENDPOINTS:-1}"
export NEB_OPTIMIZER_FMAX="${NEB_OPTIMIZER_FMAX:-0.05}"
export NEB_OPTIMIZER_STEPS="${NEB_OPTIMIZER_STEPS:-300}"
export ENDPOINT_OPTIMIZER_STEPS="${ENDPOINT_OPTIMIZER_STEPS:-200}"
if [[ "$ENGINE_VALUE" == "lammps_only_endpoint_opt" ]]; then
  export LAMMPS_ONLY_ENDPOINT_OPTIMIZE="${LAMMPS_ONLY_ENDPOINT_OPTIMIZE:-1}"
else
  export LAMMPS_ONLY_ENDPOINT_OPTIMIZE="${LAMMPS_ONLY_ENDPOINT_OPTIMIZE:-0}"
fi
export LAMMPS_ENDPOINT_OPT_MIN_STYLE="${LAMMPS_ENDPOINT_OPT_MIN_STYLE:-$LAMMPS_NEB_MIN_STYLE}"
export LAMMPS_ENDPOINT_OPT_ETOL="${LAMMPS_ENDPOINT_OPT_ETOL:-0.0}"
export LAMMPS_ENDPOINT_OPT_FTOL="${LAMMPS_ENDPOINT_OPT_FTOL:-$NEB_OPTIMIZER_FMAX}"
export LAMMPS_ENDPOINT_OPT_STEPS="${LAMMPS_ENDPOINT_OPT_STEPS:-$ENDPOINT_OPTIMIZER_STEPS}"
export LAMMPS_ENDPOINT_OPT_MAXEVAL="${LAMMPS_ENDPOINT_OPT_MAXEVAL:-$((ENDPOINT_OPTIMIZER_STEPS * 10))}"
export LAMMPS_ENDPOINT_OPT_DMAX="${LAMMPS_ENDPOINT_OPT_DMAX:-0.1}"

export CACHE_SCHEMA="${CACHE_SCHEMA:-barrier_discrepancy:pulse50:${ENGINE_TAG}:external:${map_tag}:region${region_tag}:x${SOURCE_X_MIN_A}_to_${SOURCE_X_MAX_A}:sink${SINK_X_MIN_A}:right_only:shell_IR${SHELL_INNER_RADIUS_A}_OR${SHELL_OUTER_RADIUS_A}}"
export BARRIER_CACHE_FILE="${BARRIER_CACHE_FILE:-}"
export JOB_ASSIGNMENT_MODE="${JOB_ASSIGNMENT_MODE:-cache_dedupe}"
if [[ "$ENGINE_TAG" == "ase_neb" ]]; then
  export BARRIER_MERGE_MODE="${BARRIER_MERGE_MODE:-local}"
else
  export BARRIER_MERGE_MODE="${BARRIER_MERGE_MODE:-global}"
fi
export BARRIER_MERGE_INTERVAL="${BARRIER_MERGE_INTERVAL:-100}"
export NEB_MIN_BATCH="${NEB_MIN_BATCH:-64}"

export RESTART_MODE="${RESTART_MODE:-0}"
export RESTART_DIR="${RESTART_DIR:-.}"
export RESTART_STEP="${RESTART_STEP:-0}"
export RESTART_CHECKPOINT_FILE="${RESTART_CHECKPOINT_FILE:-}"
export RESTART_STRICT="${RESTART_STRICT:-1}"
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-1000}"

export DEBUG_MODE="${DEBUG_MODE:-1}"
export DEBUG_LOGGING="${DEBUG_LOGGING:-1}"
export WRITE_RATES_ALLRANKS="${WRITE_RATES_ALLRANKS:-1}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-1}"
export MSD_LOG_INTERVAL="${MSD_LOG_INTERVAL:-1000}"
export VALIDATION_MODE="${VALIDATION_MODE:-1}"
export VALIDATION_ASE_DUMP="${VALIDATION_ASE_DUMP:-1}"
export VALIDATION_ASE_DUMP_PRE="${VALIDATION_ASE_DUMP_PRE:-1}"
export VALIDATION_OUT="${VALIDATION_OUT:-$RUN_ROOT/neb/validation_dump}"

export KMC_LOG_DIR="$RUN_ROOT/logs"
export KMC_DIAGNOSTICS_DIR="$RUN_ROOT/diagnostics"
export KMC_TRAJECTORY_DIR="$RUN_ROOT/trajectories"
export KMC_CACHE_DIR="$RUN_ROOT/cache"
export KMC_NEB_DIR="$RUN_ROOT/neb"
export RESTART_CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
export LAMMPS_NEB_SCRATCH_DIR="$RUN_ROOT/neb/lammps_only_scratch"

if [[ -z "${LAMMPS_LIB_DIR:-}" && -d "/Users/rtirunelveli/lammps-29Aug2024/build" ]]; then
  export LAMMPS_LIB_DIR="/Users/rtirunelveli/lammps-29Aug2024/build"
fi
if [[ -n "${LAMMPS_PYTHON_PATH:-}" ]]; then
  export PYTHONPATH="$LAMMPS_PYTHON_PATH:$PYTHONPATH"
fi
if [[ -n "${LAMMPS_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$LAMMPS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export DYLD_LIBRARY_PATH="$LAMMPS_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "Sigma5 external-map 50H pulse barrier diagnostic"
echo "  package=$PACKAGE_DIR"
echo "  output=$RUN_ROOT"
echo "  engine=${ENGINE_TAG} (${NEB_ENGINE})"
echo "  steps=$STEPS stop_when_empty=$KMC_STOP_WHEN_NO_H"
echo "  seed=$KMC_SEED H=$NUM_H pulse_region=$PULSE_REGION"
echo "  charging_x=[${SOURCE_X_MIN_A}, ${SOURCE_X_MAX_A}) A"
echo "  right_sink=x>=${SINK_X_MIN_A} A"
echo "  hop_x_direction=$KMC_HOP_X_DIRECTION"
echo "  ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
if [[ "$ENGINE_VALUE" == lammps_only* ]]; then
  echo "  open_file_limit=$LAMMPS_NEB_OPEN_FILE_LIMIT"
fi
echo "  optimize_endpoints=$OPTIMIZE_ENDPOINTS"
echo "  lammps_only_endpoint_optimize=$LAMMPS_ONLY_ENDPOINT_OPTIMIZE"
echo "  barrier_merge_mode=$BARRIER_MERGE_MODE"
echo "  map=$KMC_SITE_MAP_FILE"
echo "  region_file=$KMC_INITIAL_H_REGION_FILE"
echo "  cache_schema=$CACHE_SCHEMA"
echo "  diagnostics=$KMC_DIAGNOSTICS_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p \
  "$KMC_LOG_DIR" \
  "$KMC_DIAGNOSTICS_DIR" \
  "$KMC_TRAJECTORY_DIR" \
  "$KMC_CACHE_DIR" \
  "$KMC_NEB_DIR" \
  "$RESTART_CHECKPOINT_DIR" \
  "$LAMMPS_NEB_SCRATCH_DIR"

{
  printf 'PACKAGE_DIR=%s\n' "$PACKAGE_DIR"
  printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
  printf 'NEB_ENGINE_MODE=%s\n' "$ENGINE_TAG"
  printf 'NEB_ENGINE=%s\n' "$NEB_ENGINE"
  printf 'STEPS=%s\n' "$STEPS"
  printf 'NUM_H=%s\n' "$NUM_H"
  printf 'KMC_SEED=%s\n' "$KMC_SEED"
  printf 'DEVANATHAN_SEED=%s\n' "$DEVANATHAN_SEED"
  printf 'DEVANATHAN_SOURCE_MODE=%s\n' "$DEVANATHAN_SOURCE_MODE"
  printf 'DEVANATHAN_PULSE_REGION=%s\n' "$DEVANATHAN_PULSE_REGION"
  printf 'DEVANATHAN_SOURCE_X_MIN_A=%s\n' "$DEVANATHAN_SOURCE_X_MIN_A"
  printf 'DEVANATHAN_SOURCE_X_MAX_A=%s\n' "$DEVANATHAN_SOURCE_X_MAX_A"
  printf 'DEVANATHAN_SINK_X_MIN_A=%s\n' "$DEVANATHAN_SINK_X_MIN_A"
  printf 'KMC_HOP_X_DIRECTION=%s\n' "$KMC_HOP_X_DIRECTION"
  printf 'KMC_STOP_WHEN_NO_H=%s\n' "$KMC_STOP_WHEN_NO_H"
  printf 'MPI_RANKS=%s\n' "$MPI_RANKS"
  printf 'LAMMPS_NEB_REPLICAS=%s\n' "$LAMMPS_NEB_REPLICAS"
  printf 'LAMMPS_NEB_STEPS=%s\n' "$LAMMPS_NEB_STEPS"
  printf 'LAMMPS_NEB_FTOL=%s\n' "$LAMMPS_NEB_FTOL"
  printf 'LAMMPS_NEB_DEBUG_MODE=%s\n' "$LAMMPS_NEB_DEBUG_MODE"
  printf 'LAMMPS_NEB_OPEN_FILE_LIMIT=%s\n' "${LAMMPS_NEB_OPEN_FILE_LIMIT:-}"
  printf 'OPTIMIZE_ENDPOINTS=%s\n' "$OPTIMIZE_ENDPOINTS"
  printf 'NEB_OPTIMIZER_FMAX=%s\n' "$NEB_OPTIMIZER_FMAX"
  printf 'NEB_OPTIMIZER_STEPS=%s\n' "$NEB_OPTIMIZER_STEPS"
  printf 'ENDPOINT_OPTIMIZER_STEPS=%s\n' "$ENDPOINT_OPTIMIZER_STEPS"
  printf 'LAMMPS_ONLY_ENDPOINT_OPTIMIZE=%s\n' "$LAMMPS_ONLY_ENDPOINT_OPTIMIZE"
  printf 'LAMMPS_ENDPOINT_OPT_MIN_STYLE=%s\n' "$LAMMPS_ENDPOINT_OPT_MIN_STYLE"
  printf 'LAMMPS_ENDPOINT_OPT_ETOL=%s\n' "$LAMMPS_ENDPOINT_OPT_ETOL"
  printf 'LAMMPS_ENDPOINT_OPT_FTOL=%s\n' "$LAMMPS_ENDPOINT_OPT_FTOL"
  printf 'LAMMPS_ENDPOINT_OPT_STEPS=%s\n' "$LAMMPS_ENDPOINT_OPT_STEPS"
  printf 'LAMMPS_ENDPOINT_OPT_MAXEVAL=%s\n' "$LAMMPS_ENDPOINT_OPT_MAXEVAL"
  printf 'LAMMPS_ENDPOINT_OPT_DMAX=%s\n' "$LAMMPS_ENDPOINT_OPT_DMAX"
  printf 'BARRIER_MERGE_MODE=%s\n' "$BARRIER_MERGE_MODE"
  printf 'SHELL_INNER_RADIUS_A=%s\n' "$SHELL_INNER_RADIUS_A"
  printf 'SHELL_OUTER_RADIUS_A=%s\n' "$SHELL_OUTER_RADIUS_A"
  printf 'CACHE_SCHEMA=%s\n' "$CACHE_SCHEMA"
  printf 'BARRIER_CACHE_FILE=%s\n' "$BARRIER_CACHE_FILE"
  printf 'KMC_SITE_MAP_FILE=%s\n' "$KMC_SITE_MAP_FILE"
  printf 'KMC_HOST_STRUCTURE_FILE=%s\n' "$KMC_HOST_STRUCTURE_FILE"
  printf 'KMC_INITIAL_H_REGION_FILE=%s\n' "$KMC_INITIAL_H_REGION_FILE"
  printf 'WRITE_RATES_ALLRANKS=%s\n' "$WRITE_RATES_ALLRANKS"
  printf 'DEBUG_MODE=%s\n' "$DEBUG_MODE"
  printf 'VALIDATION_MODE=%s\n' "$VALIDATION_MODE"
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
if [[ "$MPI_RANKS" == "1" ]]; then
  "$PYTHON_BIN" -m kmc.main
else
  "$MPIRUN_BIN" -np "$MPI_RANKS" "$PYTHON_BIN" -m kmc.main
fi
