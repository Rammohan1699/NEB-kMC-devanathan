#!/usr/bin/env bash
set -euo pipefail

# Run the external-map grain-bulk GCMC Devanathan calculation as consecutive
# checkpoint/restart segments. The segments are serial; each segment's
# LAMMPS-only NEB work remains MPI-parallel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="${DEVANATHAN_PRACTICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PRACTICE_ROOT"

if [[ -z "${PYTHON_BIN:-}" && -x "/Users/rtirunelveli/kMC_code/python_kmc/bin/python" ]]; then
  export PYTHON_BIN="/Users/rtirunelveli/kMC_code/python_kmc/bin/python"
elif [[ -z "${PYTHON_BIN:-}" && -x "${PRACTICE_ROOT}/.venv/bin/python" ]]; then
  export PYTHON_BIN="${PRACTICE_ROOT}/.venv/bin/python"
fi

absolute_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PRACTICE_ROOT" "$path"
  fi
}

FINAL_STEP="${FINAL_STEP:-7000000}"
SEGMENT_STEPS="${SEGMENT_STEPS:-50000}"
MPI_RANKS="${MPI_RANKS:-6}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
DEVANATHAN_BULK_REGION="${DEVANATHAN_BULK_REGION:-bulk_grain_0}"
DEVANATHAN_BULK_TARGET_H_PER_FE="${DEVANATHAN_BULK_TARGET_H_PER_FE:-0.001}"
DEVANATHAN_BULK_TARGET_N_H="${DEVANATHAN_BULK_TARGET_N_H:-}"
INITIAL_CACHE="${INITIAL_CACHE:-${BARRIER_CACHE_FILE:-}}"
GCMC_ENERGY_CACHE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_external_bulk_energy_cache_${DEVANATHAN_BULK_REGION}.pkl}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-runs/devanathan_external_gcmc_bulk_${DEVANATHAN_BULK_REGION}_7m_50k_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

CAMPAIGN_ROOT="$(absolute_path "$CAMPAIGN_ROOT")"
GCMC_ENERGY_CACHE="$(absolute_path "$GCMC_ENERGY_CACHE")"
if [[ -n "$INITIAL_CACHE" ]]; then
  INITIAL_CACHE="$(absolute_path "$INITIAL_CACHE")"
fi

if [[ -d "$PRACTICE_ROOT/kmc_map_inputs" ]]; then
  MAP_ROOT="$PRACTICE_ROOT/kmc_map_inputs"
else
  MAP_ROOT="$(cd "$PRACTICE_ROOT/../.." && pwd)/kmc_map_inputs"
fi
MAP_FILE="${KMC_SITE_MAP_FILE:-${MAP_ROOT}/sigma5_stage3_unified_sites.npz}"
HOST_FILE="${KMC_HOST_STRUCTURE_FILE:-${MAP_ROOT}/sigma5_210-20-20-5.lmp}"
REGION_FILE="${KMC_INITIAL_H_REGION_FILE:-${MAP_ROOT}/sigma5_site_regions.npz}"
POTENTIAL_FILE="${POTENTIAL_EAM_FILE:-${PRACTICE_ROOT}/kmc/PotentialB3410-modified.fs}"

for path in "$MAP_FILE" "$HOST_FILE" "$REGION_FILE" "$POTENTIAL_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing or empty: $path" >&2
    exit 2
  fi
done
if [[ -n "$INITIAL_CACHE" && ! -s "$INITIAL_CACHE" ]]; then
  echo "Initial barrier cache is missing or empty: $INITIAL_CACHE" >&2
  exit 2
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
if (( LAMMPS_NEB_REPLICAS < 3 )); then
  echo "LAMMPS_NEB_REPLICAS must be at least 3; got $LAMMPS_NEB_REPLICAS" >&2
  exit 2
fi
if (( MPI_RANKS < LAMMPS_NEB_REPLICAS || MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS=$MPI_RANKS must be a positive multiple of LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS" >&2
  exit 2
fi
if [[ -e "$CAMPAIGN_ROOT" && "$RESUME" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "Campaign root already exists; set RESUME=1 to continue: $CAMPAIGN_ROOT" >&2
  exit 2
fi

PYTHON_FOR_SCHEMA="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_FOR_SCHEMA" >/dev/null 2>&1; then
  PYTHON_FOR_SCHEMA="${PYTHON_BIN_FALLBACK:-python}"
fi
DEVANATHAN_SINK_X_MIN_A="${DEVANATHAN_SINK_X_MIN_A:-$("$PYTHON_FOR_SCHEMA" -c 'import sys, numpy as np; box=np.asarray(np.load(sys.argv[1])["box_lengths"], dtype=float); print(0.8 * float(box[0]))' "$MAP_FILE")}"
if [[ -n "$DEVANATHAN_BULK_TARGET_N_H" ]]; then
  target_tag="nH${DEVANATHAN_BULK_TARGET_N_H//./p}"
else
  target_tag="frac${DEVANATHAN_BULK_TARGET_H_PER_FE//./p}"
fi
map_tag="$(basename "$MAP_FILE" .npz)"
region_tag="${DEVANATHAN_BULK_REGION//[^A-Za-z0-9_]/_}"
CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:gcmc_bulk:lammps_only:external:${map_tag}:region${region_tag}:target${target_tag}:sink${DEVANATHAN_SINK_X_MIN_A}}"

segment_count=$((FINAL_STEP / SEGMENT_STEPS))
manifest="$CAMPAIGN_ROOT/serial_manifest.tsv"

echo "External-map GCMC-bulk segmented Devanathan campaign"
echo "  campaign=$CAMPAIGN_ROOT"
echo "  final_step=$FINAL_STEP segment_steps=$SEGMENT_STEPS segments=$segment_count"
echo "  bulk_region=$DEVANATHAN_BULK_REGION target_H_per_Fe=$DEVANATHAN_BULK_TARGET_H_PER_FE"
if [[ -n "$DEVANATHAN_BULK_TARGET_N_H" ]]; then
  echo "  target_N_H=$DEVANATHAN_BULK_TARGET_N_H"
fi
echo "  reservoir_mu_eV=${DEVANATHAN_GCMC_MU_EV:--1.75}"
echo "  ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS groups=$((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "  map=$MAP_FILE"
echo "  region_file=$REGION_FILE"
echo "  initial_cache=${INITIAL_CACHE:-fresh}"
echo "  shared_gcmc_energy_cache=$GCMC_ENERGY_CACHE"
echo "  cache_schema=$CACHE_SCHEMA"

if [[ "$DRY_RUN" == "1" ]]; then
  start_step=0
  for ((segment_index = 1; segment_index <= segment_count; segment_index++)); do
    end_step=$((start_step + SEGMENT_STEPS))
    printf '  segment_%03d: %d -> %d\n' "$segment_index" "$start_step" "$end_step"
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
segment_source="$SCRIPT_DIR/run_devanathan_external_gcmc_bulk_kmc.sh"
if [[ ! -s "$segment_source" ]]; then
  echo "Missing segment launcher: $segment_source" >&2
  exit 2
fi
if [[ ! -e "$launcher_snapshot/run_devanathan_external_gcmc_bulk_kmc.sh" ]]; then
  cp -p "$segment_source" "$launcher_snapshot/"
fi
bash -n "$launcher_snapshot/run_devanathan_external_gcmc_bulk_kmc.sh"
segment_launcher="$launcher_snapshot/run_devanathan_external_gcmc_bulk_kmc.sh"

current_step=0
current_run=""
current_checkpoint=""
current_cache="$INITIAL_CACHE"

for ((segment_index = 1; segment_index <= segment_count; segment_index++)); do
  next_step=$((current_step + SEGMENT_STEPS))
  segment_name="$(
    printf 'segment_%03d_step%07d_to_%07d' \
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
    if ! awk -F '\t' -v segment="$segment_index" '$1 == segment { found=1 } END { exit !found }' "$manifest"; then
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
    echo "Move it aside or rerun with RESUME=1 after completing/recovering it." >&2
    exit 2
  fi

  echo "Starting segment $segment_index/$segment_count: $current_step -> $next_step"
  if (( current_step == 0 )); then
    env \
      DEVANATHAN_PRACTICE_ROOT="$PRACTICE_ROOT" \
      RUN_ROOT="$next_run" \
      STEPS="$next_step" \
      MPI_RANKS="$MPI_RANKS" \
      LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
      CACHE_SCHEMA="$CACHE_SCHEMA" \
      BARRIER_CACHE_FILE="$current_cache" \
      DEVANATHAN_GCMC_ENERGY_CACHE_FILE="$GCMC_ENERGY_CACHE" \
      DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}" \
      DEVANATHAN_BULK_REGION="$DEVANATHAN_BULK_REGION" \
      DEVANATHAN_BULK_TARGET_H_PER_FE="$DEVANATHAN_BULK_TARGET_H_PER_FE" \
      DEVANATHAN_BULK_TARGET_N_H="$DEVANATHAN_BULK_TARGET_N_H" \
      KMC_SITE_MAP_FILE="$MAP_FILE" \
      KMC_HOST_STRUCTURE_FILE="$HOST_FILE" \
      KMC_INITIAL_H_REGION_FILE="$REGION_FILE" \
      POTENTIAL_EAM_FILE="$POTENTIAL_FILE" \
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
      DEVANATHAN_PRACTICE_ROOT="$PRACTICE_ROOT" \
      RUN_ROOT="$next_run" \
      STEPS="$next_step" \
      MPI_RANKS="$MPI_RANKS" \
      LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
      CACHE_SCHEMA="$CACHE_SCHEMA" \
      BARRIER_CACHE_FILE="$current_cache" \
      DEVANATHAN_GCMC_ENERGY_CACHE_FILE="$GCMC_ENERGY_CACHE" \
      DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}" \
      DEVANATHAN_BULK_REGION="$DEVANATHAN_BULK_REGION" \
      DEVANATHAN_BULK_TARGET_H_PER_FE="$DEVANATHAN_BULK_TARGET_H_PER_FE" \
      DEVANATHAN_BULK_TARGET_N_H="$DEVANATHAN_BULK_TARGET_N_H" \
      KMC_SITE_MAP_FILE="$MAP_FILE" \
      KMC_HOST_STRUCTURE_FILE="$HOST_FILE" \
      KMC_INITIAL_H_REGION_FILE="$REGION_FILE" \
      POTENTIAL_EAM_FILE="$POTENTIAL_FILE" \
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
echo "Shared GCMC energy cache: $GCMC_ENERGY_CACHE"
