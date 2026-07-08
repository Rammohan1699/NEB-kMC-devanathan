#!/usr/bin/env bash
set -euo pipefail

# Practice launcher for Devanathan KMC using the external sigma5 lattice map.
# Run this from practice-version/devanathan-kmc-base.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -d "$PRACTICE_ROOT/kmc_map_inputs" ]]; then
  MAP_ROOT="$PRACTICE_ROOT/kmc_map_inputs"
else
  MAP_ROOT="$(cd "$PRACTICE_ROOT/../.." && pwd)/kmc_map_inputs"
fi
cd "$PRACTICE_ROOT"

if [[ -z "${PYTHON_BIN:-}" && -x "${PRACTICE_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PRACTICE_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-/Users/rtirunelveli/kMC_code/python_kmc/bin/python}"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-../../development/gcmc_kmc_engine/.venv/bin/python}"
fi

MPI_RANKS="${MPI_RANKS:-6}"
STEPS="${STEPS:-10000}"

MAP_FILE="${KMC_SITE_MAP_FILE:-${MAP_ROOT}/sigma5_stage3_unified_sites.npz}"
HOST_FILE="${KMC_HOST_STRUCTURE_FILE:-${MAP_ROOT}/sigma5_210-20-20-5.lmp}"

SOURCE_X_MIN_A="${DEVANATHAN_SOURCE_X_MIN_A:-0}"
SOURCE_LAYER_A="${DEVANATHAN_SOURCE_LAYER_A:-10}"
SOURCE_FRACTION="${DEVANATHAN_SOURCE_FRACTION:-0.01}"
SINK_LAYER_A="${DEVANATHAN_SINK_LAYER_A:-10}"
SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-$("$PYTHON_BIN" - "$MAP_FILE" "$SINK_LAYER_A" <<'PY'
import sys
import numpy as np
data = np.load(sys.argv[1])
box = np.asarray(data["box_lengths"], dtype=float)
print(0.8 * float(box[0]))
PY
)}"

RUN_ROOT="${RUN_ROOT:-runs/devanathan_external_map_10000}"

export PYTHONPATH="${PRACTICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NEB_ENGINE=lammps_only
export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE="$MAP_FILE"
export KMC_HOST_STRUCTURE_FILE="$HOST_FILE"
export DEVANATHAN_ENABLED=1
export DEVANATHAN_SOURCE_X_MIN_A="$SOURCE_X_MIN_A"
export DEVANATHAN_SOURCE_LAYER_A="$SOURCE_LAYER_A"
export DEVANATHAN_SOURCE_FRACTION="$SOURCE_FRACTION"
export DEVANATHAN_SINK_X_MIN_A="$SINK_X_MIN_A"
export STEPS="$STEPS"
export KNN_K="${KNN_K:-6}"
export LOCAL_ENV_MODE=shell
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
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:external:$(basename "$MAP_FILE" .npz):source${SOURCE_LAYER_A}A_fe_frac${SOURCE_FRACTION}:sink0.8Lx}"

export LAMMPS_LIB_DIR="${LAMMPS_LIB_DIR:-/Users/rtirunelveli/lammps-29Aug2024/build}"
if [[ -d "${PRACTICE_ROOT}/lammps_python_source/lammps" ]]; then
  DEFAULT_LAMMPS_SOURCE_PYTHON_PATH="${PRACTICE_ROOT}/lammps_python_source"
else
  DEFAULT_LAMMPS_SOURCE_PYTHON_PATH="../../development/tests/native_lammps_neb_test"
fi
LAMMPS_SOURCE_PYTHON_PATH="${LAMMPS_SOURCE_PYTHON_PATH:-$DEFAULT_LAMMPS_SOURCE_PYTHON_PATH}"
LAMMPS_RUNTIME_PYTHON_PATH="${LAMMPS_RUNTIME_PYTHON_PATH:-$RUN_ROOT/lammps_python}"
if [[ -d "${LAMMPS_SOURCE_PYTHON_PATH}/lammps" ]]; then
  mkdir -p "$LAMMPS_RUNTIME_PYTHON_PATH"
  rm -rf "${LAMMPS_RUNTIME_PYTHON_PATH}/lammps"
  cp -R "${LAMMPS_SOURCE_PYTHON_PATH}/lammps" "${LAMMPS_RUNTIME_PYTHON_PATH}/"
  ln -sf "${LAMMPS_LIB_DIR}/liblammps.dylib" "${LAMMPS_RUNTIME_PYTHON_PATH}/lammps/liblammps.dylib"
  export LAMMPS_PYTHON_PATH="${LAMMPS_RUNTIME_PYTHON_PATH}"
  export PYTHONPATH="${LAMMPS_PYTHON_PATH}:${PYTHONPATH}"
fi
export DYLD_LIBRARY_PATH="${LAMMPS_LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

echo "Running practice Devanathan external-map KMC"
echo "  root=$PRACTICE_ROOT"
echo "  python=$PYTHON_BIN"
echo "  engine=$NEB_ENGINE steps=$STEPS mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  map=$KMC_SITE_MAP_FILE"
echo "  host=$KMC_HOST_STRUCTURE_FILE"
echo "  source=[${SOURCE_X_MIN_A}, ${SOURCE_LAYER_A}) A fraction=${SOURCE_FRACTION}"
echo "  sink_x_min=${SINK_X_MIN_A} A"
echo "  diagnostics=${KMC_DIAGNOSTICS_DIR}"
echo "  trajectories=${KMC_TRAJECTORY_DIR}"

if [[ "$MPI_RANKS" == "1" ]]; then
  "$PYTHON_BIN" -m kmc.main
else
  mpirun -np "$MPI_RANKS" "$PYTHON_BIN" -m kmc.main
fi
