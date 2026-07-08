#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=dev_sigma5_mu17215_7m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --mem=200gb
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm_logs/dev_sigma5_mu17215_7m-%j.out
#SBATCH --error=slurm_logs/dev_sigma5_mu17215_7m-%j.err

set -euo pipefail

module purge
module load LAMMPS/23Jun2022-foss-2022a-kokkos
source ~/kmc_env/bin/activate

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PACKAGE_DIR="${KMC_PACKAGE_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PACKAGE_DIR" || ! -d "$PACKAGE_DIR/kmc" || ! -d "$PACKAGE_DIR/scripts" ]]; then
  echo "ERROR: Could not locate the Devanathan package directory." >&2
  echo "Submit from the package root or export KMC_PACKAGE_DIR." >&2
  exit 2
fi
cd "$PACKAGE_DIR"

MAP_FILE="${KMC_SITE_MAP_FILE:-$PACKAGE_DIR/kmc_map_inputs/sigma5_stage3_unified_sites.npz}"
HOST_FILE="${KMC_HOST_STRUCTURE_FILE:-$PACKAGE_DIR/kmc_map_inputs/sigma5_210-20-20-5.lmp}"
POTENTIAL_FILE="${POTENTIAL_EAM_FILE:-$PACKAGE_DIR/kmc/PotentialB3410-modified.fs}"
for path in "$MAP_FILE" "$HOST_FILE" "$POTENTIAL_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: Required input is missing: $path" >&2
    exit 2
  fi
done

mkdir -p slurm_logs runs

export PYTHON_BIN="${PYTHON_BIN:-python}"
export MPI_RANKS="${SLURM_NTASKS:-18}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export FINAL_STEP="${FINAL_STEP:-7000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-50000}"

if (( MPI_RANKS != 18 )); then
  echo "ERROR: This batch script requires exactly 18 Slurm tasks; got $MPI_RANKS." >&2
  exit 2
fi
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "ERROR: MPI_RANKS=$MPI_RANKS must be divisible by replicas=$LAMMPS_NEB_REPLICAS." >&2
  exit 2
fi

export KMC_SITE_MAP_FILE="$MAP_FILE"
export KMC_HOST_STRUCTURE_FILE="$HOST_FILE"
export POTENTIAL_EAM_FILE="$POTENTIAL_FILE"

# Sigma5 charging-zone calibration from the external-map local sweep.
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
export DEVANATHAN_SEED="${DEVANATHAN_SEED:-20260624}"

case "$DEVANATHAN_GCMC_PRODUCTION_MODE" in
  insert_delete|balanced) ;;
  *)
    echo "ERROR: This recovery must use insertion/deletion-capable GCMC production; got DEVANATHAN_GCMC_PRODUCTION_MODE=$DEVANATHAN_GCMC_PRODUCTION_MODE" >&2
    exit 2
    ;;
esac

# Keep this cache on Hyperion. Its external-host keys include Hyperion paths.
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_source_energy_cache_sigma5_bicrystal_eam_shell_IR8_OR10.pkl}"
export CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v4_bicrystal_left_sink:lammps_only:external:sigma5_stage3_unified_sites:mode:shell_IR8.0_OR10.0:source16.125_to_26.125:gcmc_mu-1.7215:sink204.199}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$PACKAGE_DIR/runs/devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p7215_7m_serial_${SLURM_JOB_ID}}"

"$PYTHON_BIN" - <<'PY'
import importlib.util

missing = [
    name
    for name in ("ase", "mpi4py", "lammps", "numpy", "scipy")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
print("Python dependency check OK")
PY

# To recover an interrupted segment, submit with RECOVERY_RUN pointing to
# that segment directory. The physical state is restored exactly at
# RECOVERY_STEP. Cache entries computed after that checkpoint remain reusable:
# they only avoid repeated barrier calculations and do not alter KMC state.
RECOVERY_RUN="${RECOVERY_RUN:-}"
RECOVERY_STEP="${RECOVERY_STEP:-100000}"
if [[ -n "$RECOVERY_RUN" ]]; then
  if [[ "$RECOVERY_RUN" != /* ]]; then
    RECOVERY_RUN="$PACKAGE_DIR/$RECOVERY_RUN"
  fi
  RECOVERY_CHECKPOINT="${RECOVERY_CHECKPOINT:-$RECOVERY_RUN/checkpoints/kmc_restart_checkpoint_step${RECOVERY_STEP}.pkl}"
  RECOVERY_RANK_CACHE="${RECOVERY_RANK_CACHE:-$RECOVERY_RUN/cache/barrier_cache_rank0_${CACHE_SCHEMA}.pkl}"
  RECOVERY_CAMPAIGN_NAME="$(basename "$(dirname "$RECOVERY_RUN")")"
  RECOVERY_COMPACT_CACHE="${RECOVERY_COMPACT_CACHE:-$PACKAGE_DIR/runs/${RECOVERY_CAMPAIGN_NAME}_barrier_cache_through_step${RECOVERY_STEP}.pkl}"
  RECOVERY_BASE_CACHE="${RECOVERY_BASE_CACHE:-}"

  # Find the newest completed cache preceding the interrupted segment when the
  # caller does not provide it explicitly.
  if [[ -z "$RECOVERY_BASE_CACHE" ]]; then
    RECOVERY_MANIFEST="$(dirname "$RECOVERY_RUN")/serial_manifest.tsv"
    if [[ -s "$RECOVERY_MANIFEST" ]]; then
      RECOVERY_BASE_CACHE="$(
        awk -F '\t' -v step="$RECOVERY_STEP" '
          $3 <= step && ($7 == "completed" || $7 == "recovered_completed") {
            if ($3 > best) {
              best = $3
              cache = $6
            }
          }
          END { print cache }
        ' "$RECOVERY_MANIFEST"
      )"
    fi
  fi

  if [[ ! -s "$RECOVERY_CHECKPOINT" ]]; then
    echo "ERROR: Recovery checkpoint is missing: $RECOVERY_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -s "$RECOVERY_RANK_CACHE" && ! -s "${RECOVERY_RANK_CACHE}.delta.pkl" ]]; then
    echo "ERROR: Recovery migration cache is missing." >&2
    echo "Checked: $RECOVERY_RANK_CACHE" >&2
    echo "Checked: ${RECOVERY_RANK_CACHE}.delta.pkl" >&2
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
print(f"Recovery checkpoint validation OK: step={actual} path={path}")
PY

  if [[ ! -s "$RECOVERY_COMPACT_CACHE" ]]; then
    if [[ -n "$RECOVERY_BASE_CACHE" && -s "$RECOVERY_BASE_CACHE" ]]; then
      echo "Merging completed and interrupted migration caches"
      "$PYTHON_BIN" tools/merge_devanathan_caches.py \
        --base "$RECOVERY_BASE_CACHE" \
        --run "$RECOVERY_RUN" \
        --out "$RECOVERY_COMPACT_CACHE" \
        --schema "$CACHE_SCHEMA" \
        --ranks "$MPI_RANKS"
    else
      echo "WARNING: No completed base cache found; compacting interrupted rank cache only." >&2
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

echo "-----------------------------------------------------"
echo "Sigma5 calibrated Devanathan GCMC/Fenwick serial run"
echo "START_TIME = $(date +'%Y-%m-%d %H:%M:%S %s')"
echo "HOST       = $(hostname)"
echo "PACKAGE    = $PACKAGE_DIR"
echo "CAMPAIGN   = $CAMPAIGN_ROOT"
echo "FINAL_STEP = $FINAL_STEP"
echo "SEGMENT    = $SEGMENT_STEPS"
echo "MPI_RANKS  = $MPI_RANKS"
echo "REPLICAS   = $LAMMPS_NEB_REPLICAS"
echo "NEB_GROUPS = $((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "MU_EV      = $DEVANATHAN_GCMC_MU_EV"
echo "GCMC_START = $DEVANATHAN_GCMC_INITIALIZATION_START"
echo "GCMC_PROD  = $DEVANATHAN_GCMC_PRODUCTION_MODE"
echo "GCMC_STEPS = $DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS..$DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS"
echo "GCMC_CACHE = $DEVANATHAN_GCMC_ENERGY_CACHE_FILE"
echo "PYTHON     = $(command -v "$PYTHON_BIN")"
echo "MPIRUN     = $(command -v mpirun)"
echo "INITIAL_CACHE = ${INITIAL_CACHE:-fresh}"
echo "RECOVERY_RUN  = ${RECOVERY_RUN:-none}"
echo "RECOVERY_BASE = ${RECOVERY_BASE_CACHE:-none}"
echo "BOOTSTRAP_STEP = ${BOOTSTRAP_STEP:-0}"
echo "-----------------------------------------------------"

START_TIME=$(date +%s)
if bash scripts/run_devanathan_bicrystal_gcmc_fenwick_serial_7m.sh; then
  STATUS=0
else
  STATUS=$?
fi
END_TIME=$(date +%s)

echo "-----------------------------------------------------"
echo "END_TIME = $(date +'%Y-%m-%d %H:%M:%S %s')"
echo "RUN_TIME seconds = $((END_TIME - START_TIME))"
echo "EXIT_STATUS = $STATUS"
echo "-----------------------------------------------------"

exit "$STATUS"
