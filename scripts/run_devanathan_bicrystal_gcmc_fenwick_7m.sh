#!/usr/bin/env bash
set -euo pipefail

# Sigma5 bicrystal Devanathan production using the same GCMC initialization,
# incremental event updates, and Fenwick selection as the bulk 7M campaign.
#
# The left periodic-end GB is excluded from the physical cell. The charging
# zone begins after one complete original-bulk hop shell; H entering that shell
# is removed at the left absorbing boundary. The right sink is the complete
# bulk lattice plane nearest 0.8 Lx.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="${DEVANATHAN_PRACTICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ -d "$PRACTICE_ROOT/kmc_map_inputs" ]]; then
  MAP_ROOT="$PRACTICE_ROOT/kmc_map_inputs"
else
  MAP_ROOT="$(cd "$PRACTICE_ROOT/../.." && pwd)/kmc_map_inputs"
fi
cd "$PRACTICE_ROOT"

MAP_FILE="${KMC_SITE_MAP_FILE:-${MAP_ROOT}/sigma5_stage3_unified_sites.npz}"
HOST_FILE="${KMC_HOST_STRUCTURE_FILE:-${MAP_ROOT}/sigma5_210-20-20-5.lmp}"
RUN_ROOT="${RUN_ROOT:-runs/devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p75_7m_$(date +%Y%m%d_%H%M%S)}"
MIGRATION_CACHE="${BARRIER_CACHE_FILE:-${INITIAL_CACHE:-}}"
GCMC_ENERGY_CACHE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_source_energy_cache_sigma5_bicrystal_eam_shell_IR8_OR10.pkl}"

if [[ ! -s "$MAP_FILE" ]]; then
  echo "Missing Sigma5 site map: $MAP_FILE" >&2
  exit 1
fi
if [[ ! -s "$HOST_FILE" ]]; then
  echo "Missing Sigma5 host structure: $HOST_FILE" >&2
  exit 1
fi
if [[ -n "$MIGRATION_CACHE" && ! -s "$MIGRATION_CACHE" ]]; then
  echo "Requested migration-barrier preload cache is missing: $MIGRATION_CACHE" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" && "${ALLOW_EXISTING_RUN_ROOT:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing output: $RUN_ROOT" >&2
  exit 1
fi

export RUN_ROOT
export STEPS="${STEPS:-7000000}"
export MPI_RANKS="${MPI_RANKS:-18}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export NIMG="${NIMG:-3}"

export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE="$MAP_FILE"
export KMC_HOST_STRUCTURE_FILE="$HOST_FILE"
export LATTICE_A="${LATTICE_A:-2.856}"
export LOCAL_ENV_MODE=shell

# Map-derived boundaries for sigma5_stage3_unified_sites.npz:
#   left end non-bulk sites end at x=13.41104130 A
#   generated type-2 sites end at x=14.68828333 A
#   one complete original-bulk hop shell is retained below x=16.12518061 A
#   the first charging plane is x=16.28483587 A
#   the complete plane nearest 0.8 Lx is x=204.35872459 A
export DEVANATHAN_SOURCE_X_MIN_A="${DEVANATHAN_SOURCE_X_MIN_A:-16.12518061159}"
export DEVANATHAN_SOURCE_LAYER_A="${DEVANATHAN_SOURCE_LAYER_A:-10}"
export DEVANATHAN_LEFT_SINK_X_MAX_A="${DEVANATHAN_LEFT_SINK_X_MAX_A:-$DEVANATHAN_SOURCE_X_MIN_A}"
export DEVANATHAN_SOURCE_FRACTION="${DEVANATHAN_SOURCE_FRACTION:-0.01}"
export DEVANATHAN_SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-204.19906933748}"

export DEVANATHAN_SOURCE_MODE=gcmc
export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}"
export DEVANATHAN_GCMC_TEMPERATURE_K="${DEVANATHAN_GCMC_TEMPERATURE_K:-300}"
export DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT="${DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT:-1}"
export DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS:-15000}"
export DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS:-50000}"
export DEVANATHAN_GCMC_PROGRESS_INTERVAL="${DEVANATHAN_GCMC_PROGRESS_INTERVAL:-100}"
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
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v3_bicrystal_left_sink:lammps_only:external:sigma5_stage3_unified_sites:mode:shell_IR8.0_OR10.0:source16.125_to_26.125:gcmc_mu-1.75:sink204.199}"
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-1000}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-1000}"
export DEBUG_MODE="${DEBUG_MODE:-0}"
export DEBUG_LOGGING="${DEBUG_LOGGING:-0}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"

if [[ -n "$MIGRATION_CACHE" ]]; then
  export BARRIER_CACHE_FILE="$MIGRATION_CACHE"
else
  unset BARRIER_CACHE_FILE
fi

if [[ "${RESTART_MODE:-0}" == "1" ]]; then
  echo "Restarting Sigma5 bicrystal GCMC + Fenwick Devanathan production"
  echo "  restart_step=${RESTART_STEP:-unknown} restart_dir=${RESTART_DIR:-unknown}"
else
  echo "Running fresh Sigma5 bicrystal GCMC + Fenwick Devanathan production"
fi
echo "  steps=$STEPS mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  map=$KMC_SITE_MAP_FILE"
echo "  host=$KMC_HOST_STRUCTURE_FILE"
echo "  left_absorber=x<$DEVANATHAN_LEFT_SINK_X_MAX_A A"
echo "  charging_zone_start=$DEVANATHAN_SOURCE_X_MIN_A A width=$DEVANATHAN_SOURCE_LAYER_A A"
echo "  right_sink=x>=$DEVANATHAN_SINK_X_MIN_A A"
echo "  reservoir_mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  migration_barrier_cache=${MIGRATION_CACHE:-fresh}"
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
