#!/usr/bin/env bash
set -euo pipefail

# Short compatibility validation from the old 7M checkpoint. Production uses
# run_devanathan_gcmc_fenwick_7m.sh to start from a fresh GCMC-equilibrated
# charging zone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRACTICE_ROOT"

SOURCE_RUN="${SOURCE_RUN:-runs/devanathan_generated_90x20x20_sink0p8_7000000_restart_from_6500000_serial}"
START_STEP="${START_STEP:-7000000}"
ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-10}"
TARGET_STEPS=$((START_STEP + ADDITIONAL_STEPS))
CHECKPOINT="checkpoints/kmc_restart_checkpoint_step${START_STEP}.pkl"
CHECKPOINT_FILE="${RESTART_CHECKPOINT_OVERRIDE:-$SOURCE_RUN/$CHECKPOINT}"
CACHE="${BARRIER_CACHE_FILE:-runs/devanathan_generated_90x20x20_sink0p8_accumulated_cache_through_7000000.pkl}"
RUN_ROOT="${RUN_ROOT:-runs/devanathan_gcmc_validation_from_7000000_${ADDITIONAL_STEPS}steps}"

if [[ ! -s "$CHECKPOINT_FILE" ]]; then
  echo "Missing restart checkpoint: $CHECKPOINT_FILE" >&2
  exit 1
fi
if [[ ! -s "$CACHE" ]]; then
  echo "Missing accumulated 7M barrier cache: $CACHE" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing output: $RUN_ROOT" >&2
  exit 1
fi

export SOURCE_RUN RUN_ROOT
export STEPS="$TARGET_STEPS"
export MPI_RANKS="${MPI_RANKS:-3}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export NIMG="${NIMG:-3}"
export NX=90 NY=20 NZ=20
export DEVANATHAN_SOURCE_X_MIN_A=0
export DEVANATHAN_SOURCE_LAYER_A=10
export DEVANATHAN_SOURCE_FRACTION="${DEVANATHAN_SOURCE_FRACTION:-0.01}"
export DEVANATHAN_SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-205.9272}"
export DEVANATHAN_SOURCE_MODE=gcmc
export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}"
export DEVANATHAN_GCMC_TEMPERATURE_K="${DEVANATHAN_GCMC_TEMPERATURE_K:-300}"
export DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT="${DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT:-1}"
export DEVANATHAN_GCMC_SHELL_INNER_RADIUS_A="${DEVANATHAN_GCMC_SHELL_INNER_RADIUS_A:-8.0}"
export DEVANATHAN_GCMC_SHELL_OUTER_RADIUS_A="${DEVANATHAN_GCMC_SHELL_OUTER_RADIUS_A:-10.0}"
export BARRIER_CACHE_FILE="$CACHE"
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx}"
export RESTART_MODE=1
export RESTART_DIR="$SOURCE_RUN"
export RESTART_STEP="$START_STEP"
export RESTART_CHECKPOINT_FILE="$CHECKPOINT_FILE"
export RESTART_STRICT=1
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-1}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-1}"
export DEBUG_MODE="${DEBUG_MODE:-0}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"

echo "Running GCMC-controlled Devanathan validation"
echo "  source_checkpoint=$CHECKPOINT_FILE"
echo "  barrier_cache=$CACHE"
echo "  steps=$START_STEP->$TARGET_STEPS"
echo "  target_H_per_Fe=$DEVANATHAN_SOURCE_FRACTION"
echo "  reservoir_mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  output=$RUN_ROOT"
echo "  note=compatibility continuation; use run_devanathan_gcmc_fenwick_7m.sh for fresh production"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

bash scripts/run_devanathan_kmc.sh
