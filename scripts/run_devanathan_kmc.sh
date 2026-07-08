#!/usr/bin/env bash
set -euo pipefail

# Practice launcher for the full base KMC driver with Devanathan boundaries.
# Run this from practice-version/devanathan-kmc-base.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="${DEVANATHAN_PRACTICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PRACTICE_ROOT"

if ! bash -n "${BASH_SOURCE[0]}"; then
  echo "Launcher syntax validation failed before starting KMC." >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null || printf '%s' "$PYTHON_BIN")"
elif [[ -x "${PRACTICE_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PRACTICE_ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-../../development/gcmc_kmc_engine/.venv/bin/python}"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Could not find an executable Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

MPI_RANKS="${MPI_RANKS:-6}"
STEPS="${STEPS:-10000}"

LATTICE_A="${LATTICE_A:-2.8601}"
NX="${NX:-90}"
NY="${NY:-20}"
NZ="${NZ:-20}"
SOURCE_X_MIN_A="${DEVANATHAN_SOURCE_X_MIN_A:-0}"
SOURCE_LAYER_A="${DEVANATHAN_SOURCE_LAYER_A:-10}"
SOURCE_FRACTION="${DEVANATHAN_SOURCE_FRACTION:-0.01}"
SINK_LAYER_A="${DEVANATHAN_SINK_LAYER_A:-10}"
SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-$("$PYTHON_BIN" -c "print(0.8 * ${LATTICE_A} * ${NX})")}"

RUN_ROOT="${RUN_ROOT:-runs/devanathan_generated_${NX}x${NY}x${NZ}_${STEPS}}"

export PYTHONPATH="${PRACTICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NEB_ENGINE=lammps_only
export KMC_SITE_SOURCE="${KMC_SITE_SOURCE:-generated}"
export DEVANATHAN_ENABLED=1
export DEVANATHAN_SOURCE_X_MIN_A="$SOURCE_X_MIN_A"
export DEVANATHAN_SOURCE_LAYER_A="$SOURCE_LAYER_A"
export DEVANATHAN_SOURCE_FRACTION="$SOURCE_FRACTION"
export DEVANATHAN_SINK_X_MIN_A="$SINK_X_MIN_A"
export STEPS="$STEPS"
export LATTICE_A="$LATTICE_A"
export NX="$NX"
export NY="$NY"
export NZ="$NZ"
export KNN_K="${KNN_K:-6}"
export LOCAL_ENV_MODE="${LOCAL_ENV_MODE:-shell}"
export ENV_KEY_MODE="${ENV_KEY_MODE:-env_plus_dir}"
export ENV_RADIUS_A="${ENV_RADIUS_A:-5.0}"
export POS_BIN_A="${POS_BIN_A:-0.10}"
export HOP_BIN_A="${HOP_BIN_A:-0.02}"
export SHELL_INNER_RADIUS_A="${SHELL_INNER_RADIUS_A:-8.0}"
export SHELL_OUTER_RADIUS_A="${SHELL_OUTER_RADIUS_A:-10.0}"
export POTENTIAL="${POTENTIAL:-EAM}"
export POTENTIAL_EAM_FILE="${POTENTIAL_EAM_FILE:-${PRACTICE_ROOT}/kmc/PotentialB3410-modified.fs}"
export LAMMPS_FILES="${LAMMPS_FILES:-$POTENTIAL_EAM_FILE}"
export NIMG="${NIMG:-3}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export LAMMPS_NEB_STEPS="${LAMMPS_NEB_STEPS:-1000}"
export LAMMPS_NEB_FTOL="${LAMMPS_NEB_FTOL:-1e-5}"
export LAMMPS_NEB_MIN_STYLE="${LAMMPS_NEB_MIN_STYLE:-quickmin}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"
export JOB_ASSIGNMENT_MODE="${JOB_ASSIGNMENT_MODE:-cache_dedupe}"
export BARRIER_MERGE_MODE="${BARRIER_MERGE_MODE:-global}"
export BARRIER_MERGE_INTERVAL="${BARRIER_MERGE_INTERVAL:-100}"
export WRITE_RATES_ALLRANKS="${WRITE_RATES_ALLRANKS:-0}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-1}"
export DEBUG_MODE="${DEBUG_MODE:-0}"
export KMC_LOG_DIR="${KMC_LOG_DIR:-$RUN_ROOT/logs}"
export KMC_DIAGNOSTICS_DIR="${KMC_DIAGNOSTICS_DIR:-$RUN_ROOT/diagnostics}"
export KMC_TRAJECTORY_DIR="${KMC_TRAJECTORY_DIR:-$RUN_ROOT/trajectories}"
export KMC_CACHE_DIR="${KMC_CACHE_DIR:-$RUN_ROOT/cache}"
export RESTART_CHECKPOINT_DIR="${RESTART_CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
export LAMMPS_NEB_SCRATCH_DIR="${LAMMPS_NEB_SCRATCH_DIR:-$RUN_ROOT/neb/lammps_only_scratch}"
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:generated:${NX}x${NY}x${NZ}:mode:shell_IR${SHELL_INNER_RADIUS_A}_OR${SHELL_OUTER_RADIUS_A}:source${SOURCE_LAYER_A}A_fe_frac${SOURCE_FRACTION}:sink0.8Lx}"

if [[ -z "${LAMMPS_LIB_DIR:-}" && -d "/Users/rtirunelveli/lammps-29Aug2024/build" ]]; then
  export LAMMPS_LIB_DIR="/Users/rtirunelveli/lammps-29Aug2024/build"
fi
if [[ -d "${PRACTICE_ROOT}/lammps_python_source/lammps" ]]; then
  DEFAULT_LAMMPS_SOURCE_PYTHON_PATH="${PRACTICE_ROOT}/lammps_python_source"
else
  DEFAULT_LAMMPS_SOURCE_PYTHON_PATH=""
fi
LAMMPS_SOURCE_PYTHON_PATH="${LAMMPS_SOURCE_PYTHON_PATH:-$DEFAULT_LAMMPS_SOURCE_PYTHON_PATH}"
LAMMPS_RUNTIME_PYTHON_PATH="${LAMMPS_RUNTIME_PYTHON_PATH:-$RUN_ROOT/lammps_python}"
if [[ -n "$LAMMPS_SOURCE_PYTHON_PATH" && -d "${LAMMPS_SOURCE_PYTHON_PATH}/lammps" ]]; then
  mkdir -p "$LAMMPS_RUNTIME_PYTHON_PATH"
  rm -rf "${LAMMPS_RUNTIME_PYTHON_PATH}/lammps"
  cp -R "${LAMMPS_SOURCE_PYTHON_PATH}/lammps" "${LAMMPS_RUNTIME_PYTHON_PATH}/"
  if [[ -n "${LAMMPS_LIB_DIR:-}" && -f "${LAMMPS_LIB_DIR}/liblammps.dylib" ]]; then
    ln -sf "${LAMMPS_LIB_DIR}/liblammps.dylib" "${LAMMPS_RUNTIME_PYTHON_PATH}/lammps/liblammps.dylib"
  fi
  export LAMMPS_PYTHON_PATH="${LAMMPS_RUNTIME_PYTHON_PATH}"
  export PYTHONPATH="${LAMMPS_PYTHON_PATH}:${PYTHONPATH}"
fi
if [[ -n "${LAMMPS_LIB_DIR:-}" && -d "${LAMMPS_LIB_DIR}" ]]; then
  export DYLD_LIBRARY_PATH="${LAMMPS_LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${LAMMPS_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "Running practice Devanathan KMC"
echo "  root=$PRACTICE_ROOT"
echo "  python=$PYTHON_BIN"
echo "  engine=$NEB_ENGINE steps=$STEPS mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  generated_cell=${NX}x${NY}x${NZ} lattice_a=${LATTICE_A}"
echo "  source_x_min=${SOURCE_X_MIN_A} A source_width=${SOURCE_LAYER_A} A fraction=${SOURCE_FRACTION}"
echo "  sink_x_min=${SINK_X_MIN_A} A"
echo "  diagnostics=${KMC_DIAGNOSTICS_DIR}"
echo "  trajectories=${KMC_TRAJECTORY_DIR}"

if [[ "$MPI_RANKS" == "1" ]]; then
  "$PYTHON_BIN" -m kmc.main
else
  mpirun -np "$MPI_RANKS" "$PYTHON_BIN" -m kmc.main
fi
