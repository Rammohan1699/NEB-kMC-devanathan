#!/usr/bin/env bash
set -euo pipefail

# Run a fixed-50H campaign as consecutive checkpoint/restart segments. Each
# segment is serial with respect to the next; the NEB work inside each segment
# remains MPI-parallel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${KMC_PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PACKAGE_DIR"

absolute_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PACKAGE_DIR" "$path"
  fi
}

FINAL_STEP="${FINAL_STEP:-2000000}"
SEGMENT_STEPS="${SEGMENT_STEPS:-50000}"
MPI_RANKS="${MPI_RANKS:-18}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
NUM_H="${NUM_H:-50}"
KMC_SEED="${KMC_SEED:-5001}"
KMC_INITIAL_H_REGIONS="${KMC_INITIAL_H_REGIONS:-bulk_grain_0}"
CACHE_SCHEMA="${CACHE_SCHEMA:-v4_lammps_only:mode:shell_IR8.0_OR10.0:env_plus_dir:R5.0_PB0.1_HB0.02:map:sigma5_stage3_unified_sites}"
INITIAL_CACHE="${INITIAL_CACHE:-}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-runs/external_one_grain_50H_fenwick_seed${KMC_SEED}_2m_50k}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

CAMPAIGN_ROOT="$(absolute_path "$CAMPAIGN_ROOT")"
if [[ -n "$INITIAL_CACHE" ]]; then
  INITIAL_CACHE="$(absolute_path "$INITIAL_CACHE")"
fi

if (( FINAL_STEP <= 0 )); then
  echo "FINAL_STEP must be positive; got $FINAL_STEP" >&2
  exit 2
fi
if (( SEGMENT_STEPS <= 0 )); then
  echo "SEGMENT_STEPS must be positive; got $SEGMENT_STEPS" >&2
  exit 2
fi
if (( FINAL_STEP % SEGMENT_STEPS != 0 )); then
  echo "FINAL_STEP=$FINAL_STEP must be divisible by SEGMENT_STEPS=$SEGMENT_STEPS" >&2
  exit 2
fi
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS=$MPI_RANKS must be divisible by LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS" >&2
  exit 2
fi
if (( NUM_H != 50 )); then
  echo "This production campaign requires NUM_H=50; got $NUM_H" >&2
  exit 2
fi
if [[ -n "$INITIAL_CACHE" && ! -s "$INITIAL_CACHE" ]]; then
  echo "Initial barrier cache is missing or empty: $INITIAL_CACHE" >&2
  exit 2
fi
if [[ -e "$CAMPAIGN_ROOT" && "$RESUME" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "Campaign root already exists; set RESUME=1 to continue: $CAMPAIGN_ROOT" >&2
  exit 2
fi

segment_count=$((FINAL_STEP / SEGMENT_STEPS))
manifest="$CAMPAIGN_ROOT/serial_manifest.tsv"

echo "Fixed-50H Sigma5 external-map Fenwick campaign"
echo "  campaign=$CAMPAIGN_ROOT"
echo "  final_step=$FINAL_STEP segment_steps=$SEGMENT_STEPS segments=$segment_count"
echo "  seed=$KMC_SEED H=$NUM_H regions=$KMC_INITIAL_H_REGIONS"
echo "  ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS groups=$((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "  initial_cache=${INITIAL_CACHE:-fresh}"

if [[ "$DRY_RUN" == "1" ]]; then
  start_step=0
  for ((segment_index = 1; segment_index <= segment_count; segment_index++)); do
    end_step=$((start_step + SEGMENT_STEPS))
    printf '  segment_%02d: %d -> %d\n' "$segment_index" "$start_step" "$end_step"
    start_step="$end_step"
  done
  exit 0
fi

mkdir -p "$CAMPAIGN_ROOT"
if [[ ! -e "$manifest" ]]; then
  printf 'segment\tstart_step\tend_step\trun_dir\tcheckpoint\tbarrier_cache\tstatus\n' > "$manifest"
fi

launcher_snapshot="$CAMPAIGN_ROOT/launcher_snapshot"
mkdir -p "$launcher_snapshot"
if [[ ! -e "$launcher_snapshot/run_external_50h_fenwick_segment.sh" ]]; then
  cp scripts/run_external_50h_fenwick_segment.sh "$launcher_snapshot/"
fi
bash -n "$launcher_snapshot/run_external_50h_fenwick_segment.sh"
segment_launcher="$launcher_snapshot/run_external_50h_fenwick_segment.sh"

current_step=0
current_run=""
current_checkpoint=""
current_cache="$INITIAL_CACHE"

for ((segment_index = 1; segment_index <= segment_count; segment_index++)); do
  next_step=$((current_step + SEGMENT_STEPS))
  segment_name="$(
    printf 'segment_%02d_step%07d_to_%07d' \
      "$segment_index" "$current_step" "$next_step"
  )"
  next_run="$CAMPAIGN_ROOT/$segment_name"
  checkpoint="$next_run/checkpoints/kmc_restart_checkpoint_step${next_step}.pkl"
  merged_cache="$next_run/cache/barrier_cache_${CACHE_SCHEMA}.pkl"

  if [[ -s "$checkpoint" && -s "$merged_cache" ]]; then
    if [[ "$RESUME" != "1" ]]; then
      echo "Completed segment already exists; set RESUME=1: $next_run" >&2
      exit 2
    fi
    echo "Reusing completed segment $segment_index: $current_step -> $next_step"
    if ! awk -F '\t' -v segment="$segment_index" \
      '$1 == segment { found=1 } END { exit !found }' "$manifest"; then
      printf '%d\t%d\t%d\t%s\t%s\t%s\t%s\n' \
        "$segment_index" "$current_step" "$next_step" "$next_run" \
        "$checkpoint" "$merged_cache" "recovered_completed" >> "$manifest"
    fi
    current_run="$next_run"
    current_checkpoint="$checkpoint"
    current_cache="$merged_cache"
    current_step="$next_step"
    continue
  fi

  if [[ -e "$next_run" ]]; then
    echo "Incomplete segment directory exists; refusing to overwrite: $next_run" >&2
    echo "Recover it explicitly or move it aside before resubmitting." >&2
    exit 2
  fi

  echo "Starting segment $segment_index/$segment_count: $current_step -> $next_step"
  if (( current_step == 0 )); then
    env \
      KMC_PACKAGE_DIR="$PACKAGE_DIR" \
      RUN_ROOT="$next_run" \
      STEPS="$next_step" \
      NUM_H="$NUM_H" \
      KMC_SEED="$KMC_SEED" \
      KMC_INITIAL_H_REGIONS="$KMC_INITIAL_H_REGIONS" \
      MPI_RANKS="$MPI_RANKS" \
      LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
      CACHE_SCHEMA="$CACHE_SCHEMA" \
      BARRIER_CACHE_FILE="$current_cache" \
      RESTART_MODE=0 \
      bash "$segment_launcher"
  else
    if [[ ! -s "$current_checkpoint" ]]; then
      echo "Previous segment checkpoint is missing: $current_checkpoint" >&2
      exit 2
    fi
    if [[ ! -s "$current_cache" ]]; then
      echo "Previous segment barrier cache is missing: $current_cache" >&2
      exit 2
    fi
    env \
      KMC_PACKAGE_DIR="$PACKAGE_DIR" \
      RUN_ROOT="$next_run" \
      STEPS="$next_step" \
      NUM_H="$NUM_H" \
      KMC_SEED="$KMC_SEED" \
      KMC_INITIAL_H_REGIONS="$KMC_INITIAL_H_REGIONS" \
      MPI_RANKS="$MPI_RANKS" \
      LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
      CACHE_SCHEMA="$CACHE_SCHEMA" \
      BARRIER_CACHE_FILE="$current_cache" \
      RESTART_MODE=1 \
      RESTART_DIR="$current_run" \
      RESTART_STEP="$current_step" \
      RESTART_CHECKPOINT_FILE="$current_checkpoint" \
      RESTART_STRICT=1 \
      bash "$segment_launcher"
  fi

  if [[ ! -s "$checkpoint" ]]; then
    echo "Segment ended without required checkpoint: $checkpoint" >&2
    exit 2
  fi
  if [[ ! -s "$merged_cache" ]]; then
    echo "Segment ended without required merged barrier cache: $merged_cache" >&2
    exit 2
  fi

  printf '%d\t%d\t%d\t%s\t%s\t%s\t%s\n' \
    "$segment_index" "$current_step" "$next_step" "$next_run" \
    "$checkpoint" "$merged_cache" "completed" >> "$manifest"

  current_run="$next_run"
  current_checkpoint="$checkpoint"
  current_cache="$merged_cache"
  current_step="$next_step"
  echo "Completed segment $segment_index through step $current_step"
done

echo "Completed all $segment_count segments through step $FINAL_STEP"
echo "Final run: $current_run"
echo "Final checkpoint: $current_checkpoint"
echo "Final barrier cache: $current_cache"
