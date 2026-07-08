#!/usr/bin/env bash
set -euo pipefail

# Run consecutive Devanathan restart segments, merging newly calculated NEBs
# into the accumulated preload cache after each successful segment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRACTICE_ROOT"

START_RUN="${START_RUN:-runs/devanathan_generated_90x20x20_sink0p8_3000000_restart_from_2500000_manual}"
START_STEP="${START_STEP:-3000000}"
FINAL_STEP="${FINAL_STEP:-6000000}"
SEGMENT_STEPS="${SEGMENT_STEPS:-500000}"
MPI_RANKS="${MPI_RANKS:-3}"
INITIAL_CACHE="${INITIAL_CACHE:-runs/devanathan_generated_90x20x20_sink0p8_accumulated_cache_through_2500000.pkl}"
CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx}"
PYTHON_BIN="${PYTHON_BIN:-/Users/rtirunelveli/kMC_code/python_kmc/bin/python}"

if (( START_STEP >= FINAL_STEP )); then
  echo "START_STEP must be less than FINAL_STEP" >&2
  exit 1
fi
if (( (FINAL_STEP - START_STEP) % SEGMENT_STEPS != 0 )); then
  echo "Step range must be divisible by SEGMENT_STEPS=$SEGMENT_STEPS" >&2
  exit 1
fi
if [[ ! -s "$START_RUN/checkpoints/kmc_restart_checkpoint_step${START_STEP}.pkl" ]]; then
  echo "Missing starting checkpoint at step $START_STEP in $START_RUN" >&2
  exit 1
fi
if [[ ! -s "$INITIAL_CACHE" ]]; then
  echo "Missing initial accumulated cache: $INITIAL_CACHE" >&2
  exit 1
fi

current_run="$START_RUN"
current_step="$START_STEP"
current_cache="runs/devanathan_generated_90x20x20_sink0p8_accumulated_cache_through_${START_STEP}.pkl"

if [[ "$INITIAL_CACHE" == "$current_cache" ]]; then
  echo "Using existing accumulated cache through step $START_STEP"
else
  echo "Preparing accumulated cache through step $START_STEP"
  "$PYTHON_BIN" tools/merge_devanathan_caches.py \
    --base "$INITIAL_CACHE" \
    --run "$START_RUN" \
    --out "$current_cache" \
    --schema "$CACHE_SCHEMA" \
    --ranks "$MPI_RANKS"
fi

while (( current_step < FINAL_STEP )); do
  next_step=$((current_step + SEGMENT_STEPS))
  next_run="runs/devanathan_generated_90x20x20_sink0p8_${next_step}_restart_from_${current_step}_serial"
  next_cache="runs/devanathan_generated_90x20x20_sink0p8_accumulated_cache_through_${next_step}.pkl"

  if [[ -e "$next_run" ]]; then
    echo "Refusing to overwrite existing output: $next_run" >&2
    exit 1
  fi

  echo "Starting serial segment: $current_step -> $next_step"
  SOURCE_RUN="$current_run" \
  RUN_ROOT="$next_run" \
  STEPS="$next_step" \
  MPI_RANKS="$MPI_RANKS" \
  LAMMPS_NEB_REPLICAS=3 \
  NIMG=3 \
  DUMP_EVERY_STEPS=1000 \
  RESTART_CHECKPOINT_INTERVAL=1000 \
  BARRIER_CACHE_FILE="$current_cache" \
  DEBUG_MODE=0 \
  LAMMPS_NEB_DEBUG_MODE=0 \
  bash scripts/run_devanathan_restart.sh

  checkpoint="$next_run/checkpoints/kmc_restart_checkpoint_step${next_step}.pkl"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Segment ended without required checkpoint: $checkpoint" >&2
    exit 1
  fi

  "$PYTHON_BIN" tools/merge_devanathan_caches.py \
    --base "$current_cache" \
    --run "$next_run" \
    --out "$next_cache" \
    --schema "$CACHE_SCHEMA" \
    --ranks "$MPI_RANKS"

  current_run="$next_run"
  current_step="$next_step"
  current_cache="$next_cache"
  echo "Completed serial segment through step $current_step"
done

echo "All serial Devanathan restarts completed through step $FINAL_STEP"
