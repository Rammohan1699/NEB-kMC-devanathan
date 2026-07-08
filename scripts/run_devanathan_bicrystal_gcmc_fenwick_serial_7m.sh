#!/usr/bin/env bash
set -euo pipefail

# Run the Sigma5 bicrystal GCMC/Fenwick calculation as independent serial
# processes. Each segment restarts from the preceding checkpoint and merged
# migration-barrier cache, matching the bulk 7M campaign structure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRACTICE_ROOT"

absolute_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s\n" "$path"
  else
    printf "%s/%s\n" "$PRACTICE_ROOT" "$path"
  fi
}

FINAL_STEP="${FINAL_STEP:-7000000}"
SEGMENT_STEPS="${SEGMENT_STEPS:-500000}"
MPI_RANKS="${MPI_RANKS:-18}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
INITIAL_CACHE="${INITIAL_CACHE:-}"
CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v3_bicrystal_left_sink:lammps_only:external:sigma5_stage3_unified_sites:mode:shell_IR8.0_OR10.0:source16.125_to_26.125:gcmc_mu-1.75:sink204.199}"
GCMC_ENERGY_CACHE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_source_energy_cache_sigma5_bicrystal_eam_shell_IR8_OR10.pkl}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-runs/devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p75_7m_serial_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"
DRY_RUN="${DRY_RUN:-0}"
BOOTSTRAP_STEP="${BOOTSTRAP_STEP:-0}"
BOOTSTRAP_RUN="${BOOTSTRAP_RUN:-}"
BOOTSTRAP_CHECKPOINT_FILE="${BOOTSTRAP_CHECKPOINT_FILE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if (( FINAL_STEP <= 0 )); then
  echo "FINAL_STEP must be positive" >&2
  exit 1
fi
if (( SEGMENT_STEPS <= 0 )); then
  echo "SEGMENT_STEPS must be positive" >&2
  exit 1
fi
if (( BOOTSTRAP_STEP < 0 || BOOTSTRAP_STEP >= FINAL_STEP )); then
  echo "BOOTSTRAP_STEP=$BOOTSTRAP_STEP must satisfy 0 <= step < FINAL_STEP=$FINAL_STEP" >&2
  exit 1
fi
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS=$MPI_RANKS must be divisible by LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS" >&2
  exit 1
fi
if [[ -n "$INITIAL_CACHE" && ! -s "$INITIAL_CACHE" ]]; then
  echo "Requested initial migration-barrier cache is missing: $INITIAL_CACHE" >&2
  exit 1
fi
if (( BOOTSTRAP_STEP > 0 )); then
  if [[ -z "$BOOTSTRAP_RUN" || ! -d "$BOOTSTRAP_RUN" ]]; then
    echo "BOOTSTRAP_RUN must be an existing run directory when BOOTSTRAP_STEP>0" >&2
    exit 1
  fi
  if [[ -z "$BOOTSTRAP_CHECKPOINT_FILE" ]]; then
    BOOTSTRAP_CHECKPOINT_FILE="$BOOTSTRAP_RUN/checkpoints/kmc_restart_checkpoint_step${BOOTSTRAP_STEP}.pkl"
  fi
  BOOTSTRAP_CHECKPOINT_FILE="$(absolute_path "$BOOTSTRAP_CHECKPOINT_FILE")"
  if [[ ! -s "$BOOTSTRAP_CHECKPOINT_FILE" ]]; then
    echo "Bootstrap checkpoint is missing: $BOOTSTRAP_CHECKPOINT_FILE" >&2
    exit 1
  fi
  if [[ -z "$INITIAL_CACHE" ]]; then
    echo "INITIAL_CACHE is required when bootstrapping from step $BOOTSTRAP_STEP" >&2
    exit 1
  fi
fi
if [[ -e "$CAMPAIGN_ROOT" && "$RESUME" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "Campaign root already exists; set RESUME=1 to continue: $CAMPAIGN_ROOT" >&2
  exit 1
fi

remaining_steps=$((FINAL_STEP - BOOTSTRAP_STEP))
segment_count=$(((remaining_steps + SEGMENT_STEPS - 1) / SEGMENT_STEPS))
segment_index_offset=$((BOOTSTRAP_STEP / SEGMENT_STEPS))
manifest="$CAMPAIGN_ROOT/serial_manifest.tsv"

echo "Sigma5 bicrystal GCMC + Fenwick segmented Devanathan campaign"
echo "  final_step=$FINAL_STEP segment_steps=$SEGMENT_STEPS segments=$segment_count"
echo "  mpi_ranks=$MPI_RANKS replicas_per_neb=$LAMMPS_NEB_REPLICAS neb_groups=$((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "  initial_migration_cache=${INITIAL_CACHE:-fresh}"
echo "  shared_gcmc_energy_cache=$GCMC_ENERGY_CACHE"
echo "  campaign_root=$CAMPAIGN_ROOT"
echo "  bootstrap_step=$BOOTSTRAP_STEP"
if (( BOOTSTRAP_STEP > 0 )); then
  echo "  bootstrap_run=$BOOTSTRAP_RUN"
  echo "  bootstrap_checkpoint=$BOOTSTRAP_CHECKPOINT_FILE"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  start_step="$BOOTSTRAP_STEP"
  for ((local_index = 1; local_index <= segment_count; local_index++)); do
    segment_index=$((segment_index_offset + local_index))
    end_step=$((start_step + SEGMENT_STEPS))
    if (( end_step > FINAL_STEP )); then
      end_step="$FINAL_STEP"
    fi
    printf "  segment_%02d: %d -> %d\n" "$segment_index" "$start_step" "$end_step"
    start_step="$end_step"
  done
  exit 0
fi

mkdir -p "$CAMPAIGN_ROOT"
if [[ ! -e "$manifest" ]]; then
  printf "segment\tstart_step\tend_step\trun_dir\tcheckpoint\tmigration_cache\tstatus\n" > "$manifest"
fi
if (( BOOTSTRAP_STEP > 0 )); then
  if ! awk -F '\t' -v end="$BOOTSTRAP_STEP" '$3 == end { found=1 } END { exit !found }' "$manifest"; then
    printf "%d\t%d\t%d\t%s\t%s\t%s\t%s\n" \
      "$segment_index_offset" "$BOOTSTRAP_STEP" "$BOOTSTRAP_STEP" \
      "$BOOTSTRAP_RUN" "$BOOTSTRAP_CHECKPOINT_FILE" "$INITIAL_CACHE" \
      "bootstrap_external" >> "$manifest"
  fi
fi

launcher_snapshot="$CAMPAIGN_ROOT/launcher_snapshot"
mkdir -p "$launcher_snapshot"
bicrystal_source="$SCRIPT_DIR/run_devanathan_bicrystal_gcmc_fenwick_7m.sh"
base_source="$SCRIPT_DIR/run_devanathan_kmc.sh"
if [[ ! -s "$bicrystal_source" || ! -s "$base_source" ]]; then
  echo "ERROR: Missing launcher file(s) in $SCRIPT_DIR:" >&2
  [[ -s "$bicrystal_source" ]] || echo "  $bicrystal_source" >&2
  [[ -s "$base_source" ]] || echo "  $base_source" >&2
  echo "Restage the Hyperion package with the complete scripts directory." >&2
  exit 1
fi
if [[ ! -e "$launcher_snapshot/run_devanathan_bicrystal_gcmc_fenwick_7m.sh" ]]; then
  cp -p "$bicrystal_source" "$launcher_snapshot/"
fi
if [[ ! -e "$launcher_snapshot/run_devanathan_kmc.sh" ]]; then
  cp -p "$base_source" "$launcher_snapshot/"
fi
bash -n \
  "$launcher_snapshot/run_devanathan_bicrystal_gcmc_fenwick_7m.sh" \
  "$launcher_snapshot/run_devanathan_kmc.sh"
bicrystal_launcher="$launcher_snapshot/run_devanathan_bicrystal_gcmc_fenwick_7m.sh"
base_launcher="$launcher_snapshot/run_devanathan_kmc.sh"

current_step="$BOOTSTRAP_STEP"
current_run="$BOOTSTRAP_RUN"
current_checkpoint="$BOOTSTRAP_CHECKPOINT_FILE"
current_cache="$INITIAL_CACHE"

for ((local_index = 1; local_index <= segment_count; local_index++)); do
  segment_index=$((segment_index_offset + local_index))
  next_step=$((current_step + SEGMENT_STEPS))
  if (( next_step > FINAL_STEP )); then
    next_step="$FINAL_STEP"
  fi
  segment_name="$(
    printf "segment_%02d_step%07d_to_%07d" \
      "$segment_index" "$current_step" "$next_step"
  )"
  next_run="$CAMPAIGN_ROOT/$segment_name"
  checkpoint="$next_run/checkpoints/kmc_restart_checkpoint_step${next_step}.pkl"
  checkpoint_abs="$(absolute_path "$checkpoint")"
  merged_cache="$next_run/cache/barrier_cache_${CACHE_SCHEMA}.pkl"

  if [[ -s "$checkpoint_abs" && -s "$merged_cache" ]]; then
    if [[ "$RESUME" != "1" ]]; then
      echo "Completed segment already exists; set RESUME=1: $next_run" >&2
      exit 1
    fi
    echo "Reusing completed segment $segment_index: $current_step -> $next_step"
    if ! awk -F '\t' -v segment="$segment_index" '$1 == segment { found=1 } END { exit !found }' "$manifest"; then
      printf "%d\t%d\t%d\t%s\t%s\t%s\t%s\n" \
        "$segment_index" "$current_step" "$next_step" "$next_run" \
        "$checkpoint_abs" "$merged_cache" "recovered_completed" >> "$manifest"
    fi
    current_run="$next_run"
    current_checkpoint="$checkpoint_abs"
    current_cache="$merged_cache"
    current_step="$next_step"
    continue
  fi

  if [[ -e "$next_run" ]]; then
    if [[ "$RESUME" != "1" || "$RESUME_INCOMPLETE" != "1" ]]; then
      echo "Incomplete segment directory exists; set RESUME=1 RESUME_INCOMPLETE=1: $next_run" >&2
      exit 1
    fi

    latest_info="$(
      "$PYTHON_BIN" - "$next_run/checkpoints" <<'PY'
import glob
import os
import pickle
import sys

best = None
for path in glob.glob(os.path.join(sys.argv[1], "kmc_restart_checkpoint_step*.pkl")):
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        step = int(payload["step"])
    except Exception:
        continue
    if best is None or step > best[0]:
        best = (step, os.path.abspath(path))
if best is None:
    raise SystemExit("No readable step checkpoint found for incomplete segment")
print(f"{best[0]}\t{best[1]}")
PY
    )"
    restart_step="${latest_info%%$'\t'*}"
    restart_checkpoint="${latest_info#*$'\t'}"
    if (( restart_step < current_step || restart_step >= next_step )); then
      echo "Latest incomplete-segment checkpoint step $restart_step is outside [$current_step, $next_step)" >&2
      exit 1
    fi

    echo "Resuming incomplete segment $segment_index in place: $restart_step -> $next_step"
    "$PYTHON_BIN" tools/prepare_in_place_restart.py \
      --run-dir "$next_run" \
      --checkpoint "$restart_checkpoint"

    env \
      RUN_ROOT="$next_run" \
      STEPS="$next_step" \
      MPI_RANKS="$MPI_RANKS" \
      LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
      BARRIER_CACHE_FILE="$current_cache" \
      CACHE_SCHEMA="$CACHE_SCHEMA" \
      DEVANATHAN_GCMC_ENERGY_CACHE_FILE="$GCMC_ENERGY_CACHE" \
      DEVANATHAN_PRACTICE_ROOT="$PRACTICE_ROOT" \
      DEVANATHAN_BASE_LAUNCHER="$SCRIPT_DIR/run_devanathan_kmc.sh" \
      RESTART_MODE=1 \
      RESTART_DIR="$next_run" \
      RESTART_STEP="$restart_step" \
      RESTART_CHECKPOINT_FILE="$restart_checkpoint" \
      RESTART_STRICT=1 \
      KMC_APPEND_OUTPUTS=1 \
      ALLOW_EXISTING_RUN_ROOT=1 \
      bash "$SCRIPT_DIR/run_devanathan_bicrystal_gcmc_fenwick_7m.sh"
  else
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
        DEVANATHAN_PRACTICE_ROOT="$PRACTICE_ROOT" \
        DEVANATHAN_BASE_LAUNCHER="$base_launcher" \
        RESTART_MODE=0 \
        bash "$bicrystal_launcher"
    else
      if [[ ! -s "$current_checkpoint" ]]; then
        echo "Missing previous segment checkpoint: $current_checkpoint" >&2
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
        DEVANATHAN_PRACTICE_ROOT="$PRACTICE_ROOT" \
        DEVANATHAN_BASE_LAUNCHER="$base_launcher" \
        RESTART_MODE=1 \
        RESTART_DIR="$current_run" \
        RESTART_STEP="$current_step" \
        RESTART_CHECKPOINT_FILE="$current_checkpoint" \
        RESTART_STRICT=1 \
        bash "$bicrystal_launcher"
    fi
  fi

  if [[ ! -s "$checkpoint_abs" ]]; then
    echo "Segment ended without required checkpoint: $checkpoint_abs" >&2
    exit 1
  fi
  if [[ ! -s "$merged_cache" ]]; then
    echo "Segment ended without required merged migration cache: $merged_cache" >&2
    exit 1
  fi

  printf "%d\t%d\t%d\t%s\t%s\t%s\t%s\n" \
    "$segment_index" "$current_step" "$next_step" "$next_run" \
    "$checkpoint_abs" "$merged_cache" "completed" >> "$manifest"

  current_run="$next_run"
  current_checkpoint="$checkpoint_abs"
  current_cache="$merged_cache"
  current_step="$next_step"
  echo "Completed segment $segment_index through step $current_step"
done

echo "All $segment_count serial segments completed through step $FINAL_STEP"
echo "Final run: $current_run"
echo "Final checkpoint: $current_checkpoint"
echo "Final migration cache: $current_cache"
echo "Shared GCMC energy cache: $GCMC_ENERGY_CACHE"
