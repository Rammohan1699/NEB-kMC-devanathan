#!/usr/bin/env bash
set -euo pipefail

# Bulk Devanathan run with a GCMC-backed constant source.
#   1. equilibrate the 0-10 A charging zone with GCMC at mu_H=-1.75 eV
#   2. maintain the production source by GCMC insertions below setpoint and
#      GCMC deletions above setpoint
#   3. preload the prior migration-barrier and GCMC source-energy caches

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="${DEVANATHAN_PRACTICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PRACTICE_ROOT"

CACHE_SCHEMA_DEFAULT="devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx"
CACHE_SCHEMA="${CACHE_SCHEMA:-$CACHE_SCHEMA_DEFAULT}"

OLD_MASTER_CACHE="runs/devanathan_generated_90x20x20_sink0p8_accumulated_cache_through_7000000.pkl"
GCMC_FINAL_CACHE="runs/devanathan_gcmc_fenwick_mu_m1p75_7m_serial/segment_14_step6500000_to_7000000/cache/barrier_cache_${CACHE_SCHEMA}.pkl"
if [[ -n "${BARRIER_CACHE_FILE:-}" ]]; then
  MASTER_CACHE="$BARRIER_CACHE_FILE"
elif [[ "${PREFER_GCMC_MIGRATION_CACHE:-1}" == "1" && -s "$GCMC_FINAL_CACHE" ]]; then
  MASTER_CACHE="$GCMC_FINAL_CACHE"
else
  MASTER_CACHE="$OLD_MASTER_CACHE"
fi

GCMC_ENERGY_CACHE_DEFAULT="runs/gcmc_source_energy_cache_eam_shell_IR8_OR10.pkl"
GCMC_ENERGY_CACHE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-$GCMC_ENERGY_CACHE_DEFAULT}"
RUN_ROOT="${RUN_ROOT:-runs/devanathan_gcmc_balanced_mu_m1p75_7m_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -s "$MASTER_CACHE" ]]; then
  echo "Missing migration-barrier preload cache: $MASTER_CACHE" >&2
  exit 1
fi
if [[ ! -s "$GCMC_ENERGY_CACHE" ]]; then
  echo "Missing GCMC source-energy cache: $GCMC_ENERGY_CACHE" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing output: $RUN_ROOT" >&2
  exit 1
fi

export RUN_ROOT
export STEPS="${STEPS:-7000000}"
export MPI_RANKS="${MPI_RANKS:-18}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export NIMG="${NIMG:-3}"
export NX="${NX:-90}"
export NY="${NY:-20}"
export NZ="${NZ:-20}"

export DEVANATHAN_SOURCE_X_MIN_A="${DEVANATHAN_SOURCE_X_MIN_A:-0}"
export DEVANATHAN_SOURCE_LAYER_A="${DEVANATHAN_SOURCE_LAYER_A:-10}"
export DEVANATHAN_SOURCE_FRACTION="${DEVANATHAN_SOURCE_FRACTION:-0.01}"
export DEVANATHAN_SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-205.9272}"
export DEVANATHAN_SOURCE_MODE=gcmc
export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}"
export DEVANATHAN_GCMC_TEMPERATURE_K="${DEVANATHAN_GCMC_TEMPERATURE_K:-300}"
export DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT="${DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT:-1}"
export DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"

export DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS:-15000}"
export DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS:-50000}"
export DEVANATHAN_GCMC_CONVERGENCE_WINDOW="${DEVANATHAN_GCMC_CONVERGENCE_WINDOW:-7500}"
export DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL="${DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL:-500}"
export DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H="${DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H:-2.0}"
export DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE="${DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE:-0.005}"
export DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS="${DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS:-2}"
export DEVANATHAN_GCMC_SHELL_INNER_RADIUS_A="${DEVANATHAN_GCMC_SHELL_INNER_RADIUS_A:-8.0}"
export DEVANATHAN_GCMC_SHELL_OUTER_RADIUS_A="${DEVANATHAN_GCMC_SHELL_OUTER_RADIUS_A:-10.0}"
export DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED="${DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED:-1}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="$GCMC_ENERGY_CACHE"
export DEVANATHAN_GCMC_ENERGY_CACHE_POS_BIN_A="${DEVANATHAN_GCMC_ENERGY_CACHE_POS_BIN_A:-0.1}"
export DEVANATHAN_GCMC_ENERGY_CACHE_SAVE_INTERVAL="${DEVANATHAN_GCMC_ENERGY_CACHE_SAVE_INTERVAL:-100}"

export KMC_INCREMENTAL_EVENTS=1
export KMC_INCREMENTAL_IMPACT_RADIUS_A="${KMC_INCREMENTAL_IMPACT_RADIUS_A:-15.0}"
export BARRIER_CACHE_FILE="$MASTER_CACHE"
export CACHE_SCHEMA
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-1000}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-1000}"
export DEBUG_MODE="${DEBUG_MODE:-0}"
export DEBUG_LOGGING="${DEBUG_LOGGING:-0}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"

if [[ "${RESTART_MODE:-0}" == "1" ]]; then
  echo "Restarting GCMC-balanced Devanathan production"
  echo "  restart_step=${RESTART_STEP:-unknown} restart_dir=${RESTART_DIR:-unknown}"
else
  echo "Running fresh GCMC-balanced Devanathan production"
fi
echo "  steps=$STEPS mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  source_H_per_Fe_target=$DEVANATHAN_SOURCE_FRACTION"
echo "  reservoir_mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  production_mode=$DEVANATHAN_GCMC_PRODUCTION_MODE"
echo "  job_assignment_mode=${JOB_ASSIGNMENT_MODE:-cache_dedupe}"
echo "  migration_barrier_cache=$MASTER_CACHE"
echo "  gcmc_source_energy_cache=$GCMC_ENERGY_CACHE"
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
