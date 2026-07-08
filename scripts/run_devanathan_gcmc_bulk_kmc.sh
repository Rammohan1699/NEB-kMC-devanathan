#!/usr/bin/env bash
set -euo pipefail

# Local LAMMPS-only Devanathan run with a bulk GCMC-controlled reservoir.
# The controlled region keeps the same left edge as the normal charging zone,
# but extends to just before the sink by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="${DEVANATHAN_PRACTICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
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

export LATTICE_A="${LATTICE_A:-2.8601}"
export NX="${NX:-90}"
export NY="${NY:-20}"
export NZ="${NZ:-20}"
export STEPS="${STEPS:-1000}"
export MPI_RANKS="${MPI_RANKS:-6}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export NIMG="${NIMG:-3}"

if (( LAMMPS_NEB_REPLICAS < 3 )); then
  echo "LAMMPS-only NEB production runs require LAMMPS_NEB_REPLICAS >= 3." >&2
  exit 1
fi
if (( MPI_RANKS < LAMMPS_NEB_REPLICAS || MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS must be a positive multiple of LAMMPS_NEB_REPLICAS for LAMMPS-only NEB." >&2
  echo "Got MPI_RANKS=$MPI_RANKS and LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS." >&2
  exit 1
fi

export DEVANATHAN_SOURCE_X_MIN_A="${DEVANATHAN_SOURCE_X_MIN_A:-0}"
export DEVANATHAN_BULK_X_MIN_A="${DEVANATHAN_BULK_X_MIN_A:-$DEVANATHAN_SOURCE_X_MIN_A}"
export DEVANATHAN_BULK_TARGET_H_PER_FE="${DEVANATHAN_BULK_TARGET_H_PER_FE:-0.001}"
export DEVANATHAN_SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-$("$PYTHON_BIN" -c 'import os; print(0.8 * float(os.environ["LATTICE_A"]) * int(os.environ["NX"]))')}"
export DEVANATHAN_BULK_SINK_GAP_A="${DEVANATHAN_BULK_SINK_GAP_A:-$LATTICE_A}"
export DEVANATHAN_BULK_X_MAX_A="${DEVANATHAN_BULK_X_MAX_A:-$("$PYTHON_BIN" -c 'import os; print(float(os.environ["DEVANATHAN_SINK_X_MIN_A"]) - float(os.environ["DEVANATHAN_BULK_SINK_GAP_A"]))')}"
export DEVANATHAN_SOURCE_X_MAX_A="$DEVANATHAN_BULK_X_MAX_A"
export DEVANATHAN_SOURCE_LAYER_A="${DEVANATHAN_SOURCE_LAYER_A:-$("$PYTHON_BIN" -c 'import os; print(float(os.environ["DEVANATHAN_BULK_X_MAX_A"]) - float(os.environ["DEVANATHAN_BULK_X_MIN_A"]))')}"
export DEVANATHAN_SOURCE_FRACTION="$DEVANATHAN_BULK_TARGET_H_PER_FE"

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

target_tag="${DEVANATHAN_BULK_TARGET_H_PER_FE//./p}"
export RUN_ROOT="${RUN_ROOT:-runs/devanathan_gcmc_bulk_target${target_tag}_$(date +%Y%m%d_%H%M%S)}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-$RUN_ROOT/cache/gcmc_bulk_energy_cache.pkl}"
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:gcmc_bulk:lammps_only:generated:${NX}x${NY}x${NZ}:bulk${DEVANATHAN_BULK_X_MIN_A}_to_${DEVANATHAN_BULK_X_MAX_A}:target${DEVANATHAN_BULK_TARGET_H_PER_FE}:sink${DEVANATHAN_SINK_X_MIN_A}}"

export KMC_INCREMENTAL_EVENTS="${KMC_INCREMENTAL_EVENTS:-1}"
export KMC_INCREMENTAL_IMPACT_RADIUS_A="${KMC_INCREMENTAL_IMPACT_RADIUS_A:-15.0}"
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-100}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-100}"
export DEBUG_MODE="${DEBUG_MODE:-0}"
export DEBUG_LOGGING="${DEBUG_LOGGING:-0}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,sm}"

echo "Running GCMC-bulk Devanathan production"
echo "  root=$PRACTICE_ROOT"
echo "  python=$PYTHON_BIN"
echo "  steps=$STEPS mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  bulk_x=[${DEVANATHAN_BULK_X_MIN_A}, ${DEVANATHAN_BULK_X_MAX_A}) A"
echo "  bulk_target_H_per_Fe=$DEVANATHAN_BULK_TARGET_H_PER_FE"
echo "  sink_x_min=$DEVANATHAN_SINK_X_MIN_A A"
echo "  reservoir_mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  production_mode=$DEVANATHAN_GCMC_PRODUCTION_MODE"
echo "  gcmc_energy_cache=$DEVANATHAN_GCMC_ENERGY_CACHE_FILE"
echo "  output=$RUN_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

base_launcher="${DEVANATHAN_BASE_LAUNCHER:-$SCRIPT_DIR/run_devanathan_kmc.sh}"
if [[ ! -s "$base_launcher" ]]; then
  echo "ERROR: Missing base launcher: $base_launcher" >&2
  exit 1
fi
bash "$base_launcher"
