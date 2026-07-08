#!/usr/bin/env bash
set -euo pipefail

# Fresh incremental/Fenwick continuation from step 7,000,000 to 7,100,000.
# Usage:
#   bash scripts/run_devanathan_fenwick_100k_from_7000000.sh
#   RUN_ROOT=/absolute/or/relative/output/path bash scripts/run_devanathan_fenwick_100k_from_7000000.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CODE_ROOT="$PRACTICE_ROOT/runs/devanathan_incremental_ab_100k_from_7000000_combined_cache/worktrees/incremental_events"
SOURCE_RUN="$PRACTICE_ROOT/runs/devanathan_generated_90x20x20_sink0p8_7000000_restart_from_6500000_serial"
CACHE_SCHEMA="devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx"
PRELOAD_CACHE="$PRACTICE_ROOT/runs/devanathan_incremental_ab_100k_from_7000000_combined_cache/preload_cache/barrier_cache_${CACHE_SCHEMA}.pkl"
SOURCE_CHECKPOINT="$SOURCE_RUN/checkpoints/kmc_restart_checkpoint_step7000000.pkl"

if [[ ! -d "$CODE_ROOT" ]]; then
  echo "Fenwick code snapshot not found: $CODE_ROOT" >&2
  exit 1
fi
if [[ ! -s "$SOURCE_CHECKPOINT" ]]; then
  echo "Source checkpoint not found: $SOURCE_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -s "$PRELOAD_CACHE" ]]; then
  echo "Combined preload cache not found: $PRELOAD_CACHE" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-$PRACTICE_ROOT/runs/devanathan_fenwick_100k_from_7000000_$(date +%Y%m%d_%H%M%S)}"
case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$PRACTICE_ROOT/$RUN_ROOT" ;;
esac

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"

echo "Starting Devanathan incremental/Fenwick continuation"
echo "  source_step=7000000 target_step=7100000"
echo "  mpi_ranks=6 replicas=3 impact_radius=15.0 A"
echo "  source_checkpoint=$SOURCE_CHECKPOINT"
echo "  preload_cache=$PRELOAD_CACHE"
echo "  output=$RUN_ROOT"

cd "$CODE_ROOT"
SOURCE_RUN="$SOURCE_RUN" \
RUN_ROOT="$RUN_ROOT" \
STEPS=7100000 \
MPI_RANKS=6 \
LAMMPS_NEB_REPLICAS=3 \
NIMG=3 \
CACHE_SCHEMA="$CACHE_SCHEMA" \
BARRIER_CACHE_FILE="$PRELOAD_CACHE" \
DEBUG_MODE=1 \
DEBUG_LOGGING=1 \
DUMP_EVERY_STEPS=1000 \
RESTART_CHECKPOINT_INTERVAL=1000 \
KMC_INCREMENTAL_EVENTS=1 \
KMC_INCREMENTAL_IMPACT_RADIUS_A=15.0 \
bash scripts/run_devanathan_restart.sh 2>&1 | tee "$RUN_ROOT/run_stdout.log"
