#!/usr/bin/env bash
set -euo pipefail

# Local counterpart of the Hyperion 18-rank segmented Sigma5 campaign.
# Segments run sequentially; each segment uses MPI-parallel grouped NEBs.

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
export DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS:-60000}"
export DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS="${DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS:-120000}"
export DEVANATHAN_GCMC_CONVERGENCE_WINDOW="${DEVANATHAN_GCMC_CONVERGENCE_WINDOW:-30000}"
export DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL="${DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL:-1000}"
export DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H="${DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H:-4.0}"
export DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE="${DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE:-0.0025}"
export DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS="${DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS:-1}"
export DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX="${DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX:-1}"
export DEVANATHAN_GCMC_PROGRESS_INTERVAL="${DEVANATHAN_GCMC_PROGRESS_INTERVAL:-1000}"
export DEVANATHAN_SEED="${DEVANATHAN_SEED:-20260624}"

export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v4_bicrystal_left_sink:lammps_only:external:sigma5_stage3_unified_sites:mode:shell_IR8.0_OR10.0:source16.125_to_26.125:gcmc_mu-1.7215:sink204.199}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_source_energy_cache_sigma5_bicrystal_local_production_merged.pkl}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-runs/devanathan_sigma5_local_7m_10k_from_445k}"

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

RECOVERY_RUN="${RECOVERY_RUN:-runs/devanathan_sigma5_local_7m_50k_from_250k/segment_09_step0400000_to_0450000}"
RECOVERY_STEP="${RECOVERY_STEP:-445000}"
RECOVERY_CHECKPOINT="${RECOVERY_CHECKPOINT:-$RECOVERY_RUN/checkpoints/kmc_restart_checkpoint_step${RECOVERY_STEP}.pkl}"
RECOVERY_RANK_CACHE="${RECOVERY_RANK_CACHE:-$RECOVERY_RUN/cache/barrier_cache_rank0_${CACHE_SCHEMA}.pkl}"
RECOVERY_BASE_CACHE="${RECOVERY_BASE_CACHE:-runs/devanathan_sigma5_local_7m_50k_from_250k/segment_08_step0350000_to_0400000/cache/barrier_cache_${CACHE_SCHEMA}.pkl}"
RECOVERY_COMPACT_CACHE="${RECOVERY_COMPACT_CACHE:-runs/devanathan_sigma5_local_7m_recovered_barrier_cache_for_step445000_restart.pkl}"
USE_RECOVERY="${USE_RECOVERY:-${USE_100K_RECOVERY:-1}}"

if [[ "$USE_RECOVERY" == "1" ]]; then
  if [[ ! -s "$RECOVERY_CHECKPOINT" ]]; then
    echo "Recovery checkpoint is missing: $RECOVERY_CHECKPOINT" >&2
    exit 2
  fi
  "$PYTHON_BIN" - "$RECOVERY_CHECKPOINT" "$RECOVERY_STEP" <<'PY'
import pickle
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
with open(path, "rb") as handle:
    checkpoint = pickle.load(handle)
actual = int(checkpoint.get("step", -1))
if actual != expected:
    raise SystemExit(
        f"Recovery checkpoint step mismatch: expected {expected}, found {actual}"
    )
print(
    f"Recovery checkpoint OK: step={actual} "
    f"H={checkpoint.get('num_h')} path={path}"
)
PY

  if [[ ! -s "$RECOVERY_COMPACT_CACHE" ]]; then
    if [[ -s "$RECOVERY_BASE_CACHE" ]]; then
      echo "Merging completed and interrupted migration caches for recovery"
      "$PYTHON_BIN" tools/merge_devanathan_caches.py \
        --base "$RECOVERY_BASE_CACHE" \
        --run "$RECOVERY_RUN" \
        --out "$RECOVERY_COMPACT_CACHE" \
        --schema "$CACHE_SCHEMA" \
        --ranks "$MPI_RANKS"
    else
      echo "Completed base cache is unavailable; compacting interrupted rank cache"
      "$PYTHON_BIN" tools/compact_barrier_cache.py \
        --source "$RECOVERY_RANK_CACHE" \
        --output "$RECOVERY_COMPACT_CACHE"
    fi
  fi
  export BOOTSTRAP_STEP="$RECOVERY_STEP"
  export BOOTSTRAP_RUN="$RECOVERY_RUN"
  export BOOTSTRAP_CHECKPOINT_FILE="$RECOVERY_CHECKPOINT"
  export INITIAL_CACHE="$RECOVERY_COMPACT_CACHE"
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

echo "Local Sigma5 7M serial campaign"
echo "  campaign=$CAMPAIGN_ROOT"
echo "  mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS groups=$((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "  gcmc_start=$DEVANATHAN_GCMC_INITIALIZATION_START"
echo "  gcmc_attempts=$DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS..$DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS"
echo "  gcmc_cache=$DEVANATHAN_GCMC_ENERGY_CACHE_FILE"
echo "  segment_steps=$SEGMENT_STEPS"
echo "  bootstrap_step=${BOOTSTRAP_STEP:-0}"

if [[ "${INITIALIZATION_ONLY:-0}" == "1" ]]; then
  export RUN_ROOT="${RUN_ROOT:-runs/devanathan_sigma5_gcmc_initialization_smoke_$(date +%Y%m%d_%H%M%S)}"
  export STEPS=0
  export MPI_RANKS=1
  export LAMMPS_NEB_REPLICAS=1
  export OMPI_MCA_btl=self
  bash scripts/run_devanathan_bicrystal_gcmc_fenwick_7m.sh
  exit
fi

bash scripts/run_devanathan_bicrystal_gcmc_fenwick_serial_7m.sh
