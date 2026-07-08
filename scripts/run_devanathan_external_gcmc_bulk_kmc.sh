#!/usr/bin/env bash
set -euo pipefail

# External-map Sigma5 Devanathan run with a bulk GCMC-controlled reservoir in
# one grain. The controlled sites are selected from sigma5_site_regions.npz
# labels such as bulk_grain_0, then clipped to the x-window before the sink.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="${DEVANATHAN_PRACTICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ -d "$PRACTICE_ROOT/kmc_map_inputs" ]]; then
  MAP_ROOT="$PRACTICE_ROOT/kmc_map_inputs"
else
  MAP_ROOT="$(cd "$PRACTICE_ROOT/../.." && pwd)/kmc_map_inputs"
fi
cd "$PRACTICE_ROOT"

if [[ -z "${PYTHON_BIN:-}" && -x "/Users/rtirunelveli/kMC_code/python_kmc/bin/python" ]]; then
  export PYTHON_BIN="/Users/rtirunelveli/kMC_code/python_kmc/bin/python"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PRACTICE_ROOT}/.venv/bin/python" ]]; then
    export PYTHON_BIN="${PRACTICE_ROOT}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find a Python interpreter. Set PYTHON_BIN." >&2
    exit 1
  fi
fi

MAP_FILE="${KMC_SITE_MAP_FILE:-${MAP_ROOT}/sigma5_stage3_unified_sites.npz}"
HOST_FILE="${KMC_HOST_STRUCTURE_FILE:-${MAP_ROOT}/sigma5_210-20-20-5.lmp}"
REGION_FILE="${KMC_INITIAL_H_REGION_FILE:-${MAP_ROOT}/sigma5_site_regions.npz}"
POTENTIAL_FILE="${POTENTIAL_EAM_FILE:-${PRACTICE_ROOT}/kmc/PotentialB3410-modified.fs}"

for path in "$MAP_FILE" "$HOST_FILE" "$REGION_FILE" "$POTENTIAL_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing or empty: $path" >&2
    exit 2
  fi
done

export STEPS="${STEPS:-1000}"
export MPI_RANKS="${MPI_RANKS:-6}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export NIMG="${NIMG:-3}"

if (( LAMMPS_NEB_REPLICAS < 3 )); then
  echo "LAMMPS-only NEB production runs require LAMMPS_NEB_REPLICAS >= 3." >&2
  exit 2
fi
if (( MPI_RANKS < LAMMPS_NEB_REPLICAS || MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS must be a positive multiple of LAMMPS_NEB_REPLICAS for LAMMPS-only NEB." >&2
  echo "Got MPI_RANKS=$MPI_RANKS and LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS." >&2
  exit 2
fi

export DEVANATHAN_BULK_REGION="${DEVANATHAN_BULK_REGION:-bulk_grain_0}"
export DEVANATHAN_SOURCE_X_MIN_A="${DEVANATHAN_SOURCE_X_MIN_A:-0}"
export DEVANATHAN_BULK_X_MIN_A="${DEVANATHAN_BULK_X_MIN_A:-$DEVANATHAN_SOURCE_X_MIN_A}"
export DEVANATHAN_BULK_TARGET_H_PER_FE="${DEVANATHAN_BULK_TARGET_H_PER_FE:-0.001}"
export DEVANATHAN_SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-$("$PYTHON_BIN" -c 'import sys, numpy as np; box=np.asarray(np.load(sys.argv[1])["box_lengths"], dtype=float); print(0.8 * float(box[0]))' "$MAP_FILE")}"
export DEVANATHAN_BULK_SINK_GAP_A="${DEVANATHAN_BULK_SINK_GAP_A:-2.8601}"
export DEVANATHAN_BULK_X_MAX_A="${DEVANATHAN_BULK_X_MAX_A:-$("$PYTHON_BIN" -c 'import os; print(float(os.environ["DEVANATHAN_SINK_X_MIN_A"]) - float(os.environ["DEVANATHAN_BULK_SINK_GAP_A"]))')}"
export DEVANATHAN_SOURCE_X_MAX_A="$DEVANATHAN_BULK_X_MAX_A"
export DEVANATHAN_SOURCE_LAYER_A="${DEVANATHAN_SOURCE_LAYER_A:-$("$PYTHON_BIN" -c 'import os; print(float(os.environ["DEVANATHAN_BULK_X_MAX_A"]) - float(os.environ["DEVANATHAN_BULK_X_MIN_A"]))')}"
export DEVANATHAN_SOURCE_FRACTION="$DEVANATHAN_BULK_TARGET_H_PER_FE"

export RUN_ROOT="${RUN_ROOT:-runs/devanathan_external_gcmc_bulk_${DEVANATHAN_BULK_REGION}_$(date +%Y%m%d_%H%M%S)}"
target_tag="${DEVANATHAN_BULK_TARGET_H_PER_FE//./p}"
map_tag="$(basename "$MAP_FILE" .npz)"
region_tag="${DEVANATHAN_BULK_REGION//[^A-Za-z0-9_]/_}"

export PYTHONPATH="${PRACTICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NEB_ENGINE=lammps_only
export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE="$MAP_FILE"
export KMC_HOST_STRUCTURE_FILE="$HOST_FILE"
export KMC_HOST_FE_TYPE="${KMC_HOST_FE_TYPE:-1}"
export KMC_INITIAL_H_REGION_FILE="$REGION_FILE"
export DEVANATHAN_ENABLED=1
export DEVANATHAN_SOURCE_MODE=gcmc_bulk
export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}"
export DEVANATHAN_GCMC_TEMPERATURE_K="${DEVANATHAN_GCMC_TEMPERATURE_K:-300}"
export DEVANATHAN_GCMC_INITIALIZATION_START="${DEVANATHAN_GCMC_INITIALIZATION_START:-target}"
export DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE="${DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE:-target}"
export DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"
export DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT="${DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT:-1}"
export DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS:-2000}"
export DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS:-10000}"
export DEVANATHAN_GCMC_CONVERGENCE_WINDOW="${DEVANATHAN_GCMC_CONVERGENCE_WINDOW:-1000}"
export DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL="${DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL:-250}"
export DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H="${DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H:-2.0}"
export DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE="${DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE:-0.0005}"
export DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS="${DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS:-1}"
export DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX="${DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX:-1}"
export DEVANATHAN_GCMC_PROGRESS_INTERVAL="${DEVANATHAN_GCMC_PROGRESS_INTERVAL:-250}"
export DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED="${DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED:-1}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-$RUN_ROOT/cache/gcmc_external_bulk_energy_cache.pkl}"

export KMC_INCREMENTAL_EVENTS="${KMC_INCREMENTAL_EVENTS:-1}"
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
export LAMMPS_NEB_STEPS="${LAMMPS_NEB_STEPS:-1000}"
export LAMMPS_NEB_FTOL="${LAMMPS_NEB_FTOL:-1e-5}"
export LAMMPS_NEB_MIN_STYLE="${LAMMPS_NEB_MIN_STYLE:-quickmin}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"
export JOB_ASSIGNMENT_MODE="${JOB_ASSIGNMENT_MODE:-cache_dedupe}"
export BARRIER_MERGE_MODE="${BARRIER_MERGE_MODE:-global}"
export BARRIER_MERGE_INTERVAL="${BARRIER_MERGE_INTERVAL:-100}"
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:gcmc_bulk:lammps_only:external:${map_tag}:region${region_tag}:target${target_tag}:sink${DEVANATHAN_SINK_X_MIN_A}}"

export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-100}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-100}"
export DEBUG_MODE="${DEBUG_MODE:-0}"
export DEBUG_LOGGING="${DEBUG_LOGGING:-0}"
export WRITE_RATES_ALLRANKS="${WRITE_RATES_ALLRANKS:-0}"
export KMC_LOG_DIR="${KMC_LOG_DIR:-$RUN_ROOT/logs}"
export KMC_DIAGNOSTICS_DIR="${KMC_DIAGNOSTICS_DIR:-$RUN_ROOT/diagnostics}"
export KMC_TRAJECTORY_DIR="${KMC_TRAJECTORY_DIR:-$RUN_ROOT/trajectories}"
export KMC_CACHE_DIR="${KMC_CACHE_DIR:-$RUN_ROOT/cache}"
export RESTART_CHECKPOINT_DIR="${RESTART_CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
export LAMMPS_NEB_SCRATCH_DIR="${LAMMPS_NEB_SCRATCH_DIR:-$RUN_ROOT/neb/lammps_only_scratch}"

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
export LD_LIBRARY_PATH="${LAMMPS_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,sm}"

echo "Running external-map GCMC-bulk Devanathan production"
echo "  root=$PRACTICE_ROOT"
echo "  python=$PYTHON_BIN"
echo "  steps=$STEPS mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  map=$KMC_SITE_MAP_FILE"
echo "  host=$KMC_HOST_STRUCTURE_FILE"
echo "  region_file=$KMC_INITIAL_H_REGION_FILE"
echo "  bulk_region=$DEVANATHAN_BULK_REGION"
echo "  bulk_x=[${DEVANATHAN_BULK_X_MIN_A}, ${DEVANATHAN_BULK_X_MAX_A}) A"
echo "  bulk_target_H_per_Fe=$DEVANATHAN_BULK_TARGET_H_PER_FE"
if [[ -n "${DEVANATHAN_BULK_TARGET_N_H:-}" ]]; then
  echo "  bulk_target_N_H=$DEVANATHAN_BULK_TARGET_N_H"
fi
echo "  sink_x_min=$DEVANATHAN_SINK_X_MIN_A A"
echo "  reservoir_mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  production_mode=$DEVANATHAN_GCMC_PRODUCTION_MODE"
echo "  output=$RUN_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

if [[ "$MPI_RANKS" == "1" ]]; then
  "$PYTHON_BIN" -m kmc.main
else
  mpirun -np "$MPI_RANKS" "$PYTHON_BIN" -m kmc.main
fi
