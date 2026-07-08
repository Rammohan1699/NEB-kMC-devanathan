#!/usr/bin/env bash
set -euo pipefail

# Local Sigma5 external Devanathan run with GCMC insertions and deletions.
# This starts a fresh trajectory from step 0 and GCMC initialization. The
# stopped insertion-only campaign is used only as a migration-cache preload.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-/Users/rtirunelveli/kMC_code/python_kmc/bin/python}"
export MPI_RANKS="${MPI_RANKS:-9}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export FINAL_STEP="${FINAL_STEP:-7000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-10000}"

export KMC_SITE_MAP_FILE="${KMC_SITE_MAP_FILE:-$WORKSPACE_ROOT/kmc_map_inputs/sigma5_stage3_unified_sites.npz}"
export KMC_HOST_STRUCTURE_FILE="${KMC_HOST_STRUCTURE_FILE:-$WORKSPACE_ROOT/kmc_map_inputs/sigma5_210-20-20-5.lmp}"
export POTENTIAL_EAM_FILE="${POTENTIAL_EAM_FILE:-$ROOT/kmc/PotentialB3410-modified.fs}"

export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.7215}"
export DEVANATHAN_GCMC_INITIALIZATION_START="${DEVANATHAN_GCMC_INITIALIZATION_START:-target}"
export DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE="${DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE:-target}"
export DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"
export DEVANATHAN_GCMC_PRODUCTION_MAINTENANCE_MODE="${DEVANATHAN_GCMC_PRODUCTION_MAINTENANCE_MODE:-$DEVANATHAN_GCMC_PRODUCTION_MODE}"
export DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS:-60000}"
export DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS:-120000}"
export DEVANATHAN_GCMC_CONVERGENCE_WINDOW="${DEVANATHAN_GCMC_CONVERGENCE_WINDOW:-30000}"
export DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL="${DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL:-1000}"
export DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H="${DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H:-4.0}"
export DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE="${DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE:-0.0025}"
export DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS="${DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS:-1}"
export DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX="${DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX:-1}"
export DEVANATHAN_GCMC_PROGRESS_INTERVAL="${DEVANATHAN_GCMC_PROGRESS_INTERVAL:-1000}"
export DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED="${DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED:-1}"
export DEVANATHAN_GCMC_ENERGY_CACHE_SAVE_INTERVAL="${DEVANATHAN_GCMC_ENERGY_CACHE_SAVE_INTERVAL:-100}"
export DEVANATHAN_SEED="${DEVANATHAN_SEED:-20260624}"

export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v4_bicrystal_left_sink:lammps_only:external:sigma5_stage3_unified_sites:mode:shell_IR8.0_OR10.0:source16.125_to_26.125:gcmc_mu-1.7215:sink204.199}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_source_energy_cache_sigma5_bicrystal_local_production_merged.pkl}"
SOURCE_CAMPAIGN="${SOURCE_CAMPAIGN:-runs/devanathan_sigma5_local_7m_10k_from_445k}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$SOURCE_CAMPAIGN/serial_manifest.tsv}"

export LAMMPS_LIB_DIR="${LAMMPS_LIB_DIR:-/Users/rtirunelveli/lammps-29Aug2024/build}"
export DYLD_LIBRARY_PATH="${LAMMPS_LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,sm

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Local KMC Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
for path in "$KMC_SITE_MAP_FILE" "$KMC_HOST_STRUCTURE_FILE" "$POTENTIAL_EAM_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required local input is missing: $path" >&2
    exit 2
  fi
done
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "MPI_RANKS=$MPI_RANKS must be divisible by LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS" >&2
  exit 2
fi
case "$DEVANATHAN_GCMC_PRODUCTION_MODE" in
  insert_delete|balanced) ;;
  *)
    echo "This launcher must use deletion-capable production mode; got DEVANATHAN_GCMC_PRODUCTION_MODE=$DEVANATHAN_GCMC_PRODUCTION_MODE" >&2
    exit 2
    ;;
esac
USER_INITIAL_CACHE="${INITIAL_CACHE:-}"
if [[ -z "$USER_INITIAL_CACHE" ]]; then
  if [[ ! -s "$SOURCE_MANIFEST" ]]; then
    echo "Stopped source campaign manifest is missing: $SOURCE_MANIFEST" >&2
    exit 2
  fi

  latest_info="$(
    "$PYTHON_BIN" - "$SOURCE_MANIFEST" "$ROOT" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
root = Path(sys.argv[2])
valid_statuses = {"completed", "recovered_completed", "bootstrap_external"}

def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path

candidates = []
with manifest.open(newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        status = (row.get("status") or "").strip()
        if status not in valid_statuses:
            continue
        try:
            segment = int(row["segment"])
            end_step = int(row["end_step"])
        except Exception:
            continue
        run_dir = resolve(row.get("run_dir") or "")
        migration_cache = resolve(row.get("migration_cache") or "")
        if not run_dir.is_dir():
            continue
        if not migration_cache.is_file():
            continue
        candidates.append((end_step, segment, run_dir, migration_cache, status))

if not candidates:
    raise SystemExit(f"No completed usable rows found in {manifest}")

end_step, segment, run_dir, migration_cache, status = sorted(candidates)[-1]
print(
    f"{segment}\t{end_step}\t{run_dir}\t{migration_cache}\t{status}"
)
PY
  )"
  IFS=$'\t' read -r DETECTED_SEGMENT DETECTED_STEP DETECTED_RUN DETECTED_CACHE DETECTED_STATUS <<< "$latest_info"
  export INITIAL_CACHE="$DETECTED_CACHE"
else
  DETECTED_SEGMENT="${DETECTED_SEGMENT:-user}"
  DETECTED_STEP="${DETECTED_STEP:-unknown}"
  DETECTED_RUN="${DETECTED_RUN:-user_provided_initial_cache}"
  DETECTED_STATUS="${DETECTED_STATUS:-user_initial_cache}"
  export INITIAL_CACHE="$USER_INITIAL_CACHE"
fi

export BOOTSTRAP_STEP=0
unset BOOTSTRAP_RUN
unset BOOTSTRAP_CHECKPOINT_FILE
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-runs/devanathan_sigma5_local_7m_10k_balanced_fresh_gcmc}"

if [[ ! -s "$INITIAL_CACHE" ]]; then
  echo "Initial migration-barrier cache is missing: $INITIAL_CACHE" >&2
  exit 2
fi

if [[ -z "$USER_INITIAL_CACHE" && "${MERGE_PARTIAL_MIGRATION_CACHE:-1}" == "1" && "$DETECTED_SEGMENT" =~ ^[0-9]+$ && "$DETECTED_STEP" =~ ^[0-9]+$ ]]; then
  partial_end=$((DETECTED_STEP + SEGMENT_STEPS))
  if (( partial_end > FINAL_STEP )); then
    partial_end="$FINAL_STEP"
  fi
  printf -v partial_default "%s/segment_%02d_step%07d_to_%07d" \
    "$SOURCE_CAMPAIGN" "$((DETECTED_SEGMENT + 1))" "$DETECTED_STEP" "$partial_end"
  PARTIAL_MIGRATION_RUN="${PARTIAL_MIGRATION_RUN:-$partial_default}"
  partial_rank0="$PARTIAL_MIGRATION_RUN/cache/barrier_cache_rank0_${CACHE_SCHEMA}.pkl"
  printf -v DETECTED_LABEL "%07d" "$DETECTED_STEP"
  printf -v COMBINED_INITIAL_CACHE_DEFAULT \
    "runs/devanathan_sigma5_local_7m_balanced_preloads/barrier_cache_%s_through_step%s_plus_partial.pkl" \
    "$CACHE_SCHEMA" "$DETECTED_LABEL"
  COMBINED_INITIAL_CACHE="${COMBINED_INITIAL_CACHE:-$COMBINED_INITIAL_CACHE_DEFAULT}"

  if [[ -s "$partial_rank0" || -s "${partial_rank0}.delta.pkl" ]]; then
    if [[ -s "$COMBINED_INITIAL_CACHE" ]]; then
      export INITIAL_CACHE="$COMBINED_INITIAL_CACHE"
    elif [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN: would merge partial migration cache from $PARTIAL_MIGRATION_RUN"
    else
      echo "Merging partial stopped-run migration cache into preload"
      "$PYTHON_BIN" tools/merge_devanathan_caches.py \
        --base "$INITIAL_CACHE" \
        --run "$PARTIAL_MIGRATION_RUN" \
        --out "$COMBINED_INITIAL_CACHE" \
        --schema "$CACHE_SCHEMA" \
        --ranks "$MPI_RANKS"
      export INITIAL_CACHE="$COMBINED_INITIAL_CACHE"
    fi
  fi
fi

if [[ ! -s "$DEVANATHAN_GCMC_ENERGY_CACHE_FILE" ]]; then
  SWEEP_RUN="${SIGMA5_GCMC_SWEEP_RUN:-runs/gcmc_sigma5_external_charging_mu_m1p723_validation_20260621}"
  if [[ -d "$SWEEP_RUN/workers" ]]; then
    echo "Preparing production-compatible GCMC cache from $SWEEP_RUN"
    "$PYTHON_BIN" tools/prepare_sigma5_production_gcmc_cache.py \
      --sweep-run "$SWEEP_RUN" \
      --site-map "$KMC_SITE_MAP_FILE" \
      --host "$KMC_HOST_STRUCTURE_FILE" \
      --potential "$POTENTIAL_EAM_FILE" \
      --output "$DEVANATHAN_GCMC_ENERGY_CACHE_FILE" \
      --initialization any \
      --merge-all
  else
    echo "No local sweep cache found; GCMC initialization will build a fresh cache."
  fi
fi

echo "Local Sigma5 7M balanced serial campaign"
echo "  campaign=$CAMPAIGN_ROOT"
echo "  physical_start=fresh_step0_gcmc_initialization"
echo "  source_campaign=$SOURCE_CAMPAIGN"
echo "  detected_source_status=$DETECTED_STATUS"
echo "  detected_cache_step=$DETECTED_STEP"
echo "  production_mode=$DEVANATHAN_GCMC_PRODUCTION_MODE"
echo "  mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS groups=$((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "  gcmc_start=$DEVANATHAN_GCMC_INITIALIZATION_START"
echo "  gcmc_attempts=$DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS..$DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS"
echo "  gcmc_cache=$DEVANATHAN_GCMC_ENERGY_CACHE_FILE"
echo "  migration_cache=$INITIAL_CACHE"
echo "  segment_steps=$SEGMENT_STEPS"
echo "  bootstrap_step=0"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Preparation-only mode complete."
  exit 0
fi

bash scripts/run_devanathan_bicrystal_gcmc_fenwick_serial_7m.sh
