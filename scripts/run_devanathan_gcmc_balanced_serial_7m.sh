#!/usr/bin/env bash
set -euo pipefail

# Run the bulk GCMC-balanced Devanathan calculation as independent serial
# segments. Each segment restarts from the prior checkpoint and preloads the
# prior merged migration-barrier cache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRACTICE_ROOT"

FINAL_STEP="${FINAL_STEP:-7000000}"
SEGMENT_STEPS="${SEGMENT_STEPS:-500000}"
MPI_RANKS="${MPI_RANKS:-18}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx}"
OLD_MASTER_CACHE="runs/devanathan_generated_90x20x20_sink0p8_accumulated_cache_through_7000000.pkl"
GCMC_FINAL_CACHE="runs/devanathan_gcmc_fenwick_mu_m1p75_7m_serial/segment_14_step6500000_to_7000000/cache/barrier_cache_${CACHE_SCHEMA}.pkl"
if [[ -n "${INITIAL_CACHE:-}" ]]; then
  INITIAL_CACHE="$INITIAL_CACHE"
elif [[ "${PREFER_GCMC_MIGRATION_CACHE:-1}" == "1" && -s "$GCMC_FINAL_CACHE" ]]; then
  INITIAL_CACHE="$GCMC_FINAL_CACHE"
else
  INITIAL_CACHE="$OLD_MASTER_CACHE"
fi
GCMC_ENERGY_CACHE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_source_energy_cache_eam_shell_IR8_OR10.pkl}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-runs/devanathan_gcmc_balanced_mu_m1p75_7m_serial_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

if (( FINAL_STEP <= 0 )); then
  echo "FINAL_STEP must be positive" >&2
  exit 1
fi
if (( SEGMENT_STEPS <= 0 )); then
  echo "SEGMENT_STEPS must be positive" >&2
  exit 1
fi
if (( FINAL_STEP % SEGMENT_STEPS != 0 )); then
  echo "FINAL_STEP=$FINAL_STEP must be divisible by SEGMENT_STEPS=$SEGMENT_STEPS" >&2
  exit 1
fi
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS=$MPI_RANKS must be divisible by LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS" >&2
  exit 1
fi
if [[ ! -s "$INITIAL_CACHE" ]]; then
  echo "Missing initial migration-barrier cache: $INITIAL_CACHE" >&2
  exit 1
fi
if [[ ! -s "$GCMC_ENERGY_CACHE" ]]; then
  echo "Missing GCMC source-energy cache: $GCMC_ENERGY_CACHE" >&2
  exit 1
fi
if [[ -e "$CAMPAIGN_ROOT" && "$RESUME" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "Campaign root already exists; set RESUME=1 to continue: $CAMPAIGN_ROOT" >&2
  exit 1
fi

segment_count=$((FINAL_STEP / SEGMENT_STEPS))
manifest="$CAMPAIGN_ROOT/serial_manifest.tsv"

echo "GCMC-balanced segmented Devanathan campaign"
echo "  final_step=$FINAL_STEP segment_steps=$SEGMENT_STEPS segments=$segment_count"
echo "  mpi_ranks=$MPI_RANKS replicas_per_neb=$LAMMPS_NEB_REPLICAS neb_groups=$((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "  initial_migration_cache=$INITIAL_CACHE"
echo "  shared_gcmc_energy_cache=$GCMC_ENERGY_CACHE"
echo "  production_mode=${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"
echo "  job_assignment_mode=${JOB_ASSIGNMENT_MODE:-cache_dedupe}"
echo "  campaign_root=$CAMPAIGN_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  start_step=0
  for ((segment_index = 1; segment_index <= segment_count; segment_index++)); do
    end_step=$((start_step + SEGMENT_STEPS))
    printf "  segment_%02d: %d -> %d\n" "$segment_index" "$start_step" "$end_step"
    start_step="$end_step"
  done
  exit 0
fi

mkdir -p "$CAMPAIGN_ROOT"
if [[ ! -e "$manifest" ]]; then
  printf "segment\tstart_step\tend_step\trun_dir\tcheckpoint\tmigration_cache\tstatus\n" > "$manifest"
fi

launcher_snapshot="$CAMPAIGN_ROOT/launcher_snapshot"
mkdir -p "$launcher_snapshot"
balanced_source="$SCRIPT_DIR/run_devanathan_gcmc_balanced_7m.sh"
base_source="$SCRIPT_DIR/run_devanathan_kmc.sh"
if [[ ! -s "$balanced_source" || ! -s "$base_source" ]]; then
  echo "ERROR: Missing launcher file(s) in $SCRIPT_DIR:" >&2
  [[ -s "$balanced_source" ]] || echo "  $balanced_source" >&2
  [[ -s "$base_source" ]] || echo "  $base_source" >&2
  echo "Restage the Hyperion package with the complete scripts directory." >&2
  exit 1
fi
if [[ ! -e "$launcher_snapshot/run_devanathan_gcmc_balanced_7m.sh" ]]; then
  cp -p "$balanced_source" "$launcher_snapshot/"
fi
if [[ ! -e "$launcher_snapshot/run_devanathan_kmc.sh" ]]; then
  cp -p "$base_source" "$launcher_snapshot/"
fi
bash -n \
  "$launcher_snapshot/run_devanathan_gcmc_balanced_7m.sh" \
  "$launcher_snapshot/run_devanathan_kmc.sh"
balanced_launcher="$launcher_snapshot/run_devanathan_gcmc_balanced_7m.sh"
base_launcher="$launcher_snapshot/run_devanathan_kmc.sh"

current_step=0
current_run=""
current_cache="$INITIAL_CACHE"

for ((segment_index = 1; segment_index <= segment_count; segment_index++)); do
  next_step=$((current_step + SEGMENT_STEPS))
  segment_name="$(
    printf "segment_%02d_step%07d_to_%07d" \
      "$segment_index" "$current_step" "$next_step"
  )"
  next_run="$CAMPAIGN_ROOT/$segment_name"
  checkpoint="$next_run/checkpoints/kmc_restart_checkpoint_step${next_step}.pkl"
  merged_cache="$next_run/cache/barrier_cache_${CACHE_SCHEMA}.pkl"

  if [[ -s "$checkpoint" && -s "$merged_cache" ]]; then
    if [[ "$RESUME" != "1" ]]; then
      echo "Completed segment already exists; set RESUME=1: $next_run" >&2
      exit 1
    fi
    echo "Reusing completed segment $segment_index: $current_step -> $next_step"
    if ! awk -F '\t' -v segment="$segment_index" '$1 == segment { found=1 } END { exit !found }' "$manifest"; then
      printf "%d\t%d\t%d\t%s\t%s\t%s\t%s\n" \
        "$segment_index" "$current_step" "$next_step" "$next_run" \
        "$checkpoint" "$merged_cache" "recovered_completed" >> "$manifest"
    fi
    current_run="$next_run"
    current_cache="$merged_cache"
    current_step="$next_step"
    continue
  fi

  if [[ -e "$next_run" ]]; then
    echo "Incomplete segment directory exists; refusing to overwrite: $next_run" >&2
    exit 1
  fi

  echo "Starting segment $segment_index/$segment_count: $current_step -> $next_step"
  if (( current_step == 0 )); then
    env \
      RUN_ROOT="$next_run" \
      STEPS="$next_step" \
      MPI_RANKS="$MPI_RANKS" \
      LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
      BARRIER_CACHE_FILE="$current_cache" \
      CACHE_SCHEMA="$CACHE_SCHEMA" \
      DEVANATHAN_GCMC_ENERGY_CACHE_FILE="$GCMC_ENERGY_CACHE" \
      DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}" \
      DEVANATHAN_PRACTICE_ROOT="$PRACTICE_ROOT" \
      DEVANATHAN_BASE_LAUNCHER="$base_launcher" \
      RESTART_MODE=0 \
      bash "$balanced_launcher"
  else
    previous_checkpoint="$current_run/checkpoints/kmc_restart_checkpoint_step${current_step}.pkl"
    if [[ ! -s "$previous_checkpoint" ]]; then
      echo "Missing previous segment checkpoint: $previous_checkpoint" >&2
      exit 1
    fi
    env \
      RUN_ROOT="$next_run" \
      STEPS="$next_step" \
      MPI_RANKS="$MPI_RANKS" \
      LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
      BARRIER_CACHE_FILE="$current_cache" \
      CACHE_SCHEMA="$CACHE_SCHEMA" \
      DEVANATHAN_GCMC_ENERGY_CACHE_FILE="$GCMC_ENERGY_CACHE" \
      DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}" \
      DEVANATHAN_PRACTICE_ROOT="$PRACTICE_ROOT" \
      DEVANATHAN_BASE_LAUNCHER="$base_launcher" \
      RESTART_MODE=1 \
      RESTART_DIR="$current_run" \
      RESTART_STEP="$current_step" \
      RESTART_CHECKPOINT_FILE="checkpoints/kmc_restart_checkpoint_step${current_step}.pkl" \
      RESTART_STRICT=1 \
      bash "$balanced_launcher"
  fi

  if [[ ! -s "$checkpoint" ]]; then
    echo "Segment ended without required checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -s "$merged_cache" ]]; then
    echo "Segment ended without required merged migration cache: $merged_cache" >&2
    exit 1
  fi

  printf "%d\t%d\t%d\t%s\t%s\t%s\t%s\n" \
    "$segment_index" "$current_step" "$next_step" "$next_run" \
    "$checkpoint" "$merged_cache" "completed" >> "$manifest"

  current_run="$next_run"
  current_cache="$merged_cache"
  current_step="$next_step"
  echo "Completed segment $segment_index through step $current_step"
done

echo "All $segment_count serial segments completed through step $FINAL_STEP"
echo "Final run: $current_run"
echo "Final checkpoint: $current_run/checkpoints/kmc_restart_checkpoint_step${FINAL_STEP}.pkl"
echo "Final migration cache: $current_cache"
echo "Shared GCMC energy cache: $GCMC_ENERGY_CACHE"
