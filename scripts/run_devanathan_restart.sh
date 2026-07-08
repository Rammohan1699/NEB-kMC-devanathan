#!/usr/bin/env bash
set -euo pipefail

# Restart the interrupted 90x20x20 Devanathan run from its newest checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRACTICE_ROOT"

SOURCE_RUN="${SOURCE_RUN:-runs/devanathan_generated_90x20x20_sink0p8_2000000}"
TARGET_STEPS="${STEPS:-2000000}"

if [[ ! -d "$SOURCE_RUN/checkpoints" ]]; then
  echo "Restart checkpoint directory not found: $SOURCE_RUN/checkpoints" >&2
  exit 1
fi

latest_step="$({
  find "$SOURCE_RUN/checkpoints" -maxdepth 1 -type f \
    -name 'kmc_restart_checkpoint_step*.pkl' -print
} | sed -E 's/.*_step([0-9]+)\.pkl/\1/' | sort -n | tail -1)"

if [[ -z "$latest_step" ]]; then
  echo "No step-specific restart checkpoint found in $SOURCE_RUN/checkpoints" >&2
  exit 1
fi

checkpoint_rel="checkpoints/kmc_restart_checkpoint_step${latest_step}.pkl"
checkpoint_path="$SOURCE_RUN/$checkpoint_rel"
if [[ ! -s "$checkpoint_path" ]]; then
  echo "Restart checkpoint is missing or empty: $checkpoint_path" >&2
  exit 1
fi
if (( latest_step >= TARGET_STEPS )); then
  echo "Checkpoint step $latest_step has already reached target STEPS=$TARGET_STEPS" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-runs/devanathan_generated_90x20x20_sink0p8_2000000_restart_from_${latest_step}}"

export SOURCE_RUN
export RUN_ROOT
export STEPS="$TARGET_STEPS"
export MPI_RANKS="${MPI_RANKS:-3}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export NIMG="${NIMG:-3}"
export NX=90
export NY=20
export NZ=20
export DEVANATHAN_SOURCE_X_MIN_A=0
export DEVANATHAN_SOURCE_LAYER_A=10
export DEVANATHAN_SOURCE_FRACTION=0.01
export DEVANATHAN_SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-205.9272}"
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx}"

export RESTART_MODE=1
export RESTART_DIR="$SOURCE_RUN"
export RESTART_STEP="$latest_step"
export RESTART_CHECKPOINT_FILE="$checkpoint_rel"
export RESTART_STRICT=1
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-1000}"

# Keep diagnostic structures off by default and use fewer MPI ranks than the
# interrupted six-rank run to reduce per-rank cache and LAMMPS memory pressure.
export DEBUG_MODE="${DEBUG_MODE:-0}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"

echo "Restarting practice Devanathan KMC"
echo "  source_run=$SOURCE_RUN"
echo "  checkpoint=$checkpoint_path"
echo "  restart_step=$latest_step target_steps=$TARGET_STEPS"
echo "  output_run=$RUN_ROOT"
echo "  mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

bash scripts/run_devanathan_kmc.sh
