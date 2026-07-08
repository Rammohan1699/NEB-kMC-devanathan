#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=ext50h_fenwick
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --mem=100gb
#SBATCH --time=14-00:00:00
#SBATCH --array=0-9%10
#SBATCH --output=slurm_logs/ext50h_fenwick-%A_%a.out
#SBATCH --error=slurm_logs/ext50h_fenwick-%A_%a.err

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
  echo "ERROR: Could not locate the Devanathan/Fenwick package directory." >&2
  echo "Submit from the package root or export KMC_PACKAGE_DIR." >&2
  exit 2
fi
cd "$PACKAGE_DIR"
mkdir -p slurm_logs runs

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID > 9 )); then
  echo "ERROR: Array task must be in 0..9; got $TASK_ID" >&2
  exit 2
fi

H_LAYOUT="${H_LAYOUT:-one_grain}"
case "$H_LAYOUT" in
  one|one_grain|one-grain)
    LAYOUT_TAG="one_grain"
    INITIAL_REGIONS="bulk_grain_0"
    DEFAULT_SEED_BASE=5001
    ;;
  both|both_grains|both-grains)
    LAYOUT_TAG="both_grains"
    INITIAL_REGIONS="bulk_grain_0,bulk_grain_1"
    DEFAULT_SEED_BASE=6001
    ;;
  *)
    echo "ERROR: H_LAYOUT must be one_grain or both_grains; got '$H_LAYOUT'" >&2
    exit 2
    ;;
esac

MPI_RANKS="${SLURM_NTASKS:-18}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
if (( MPI_RANKS != 18 )); then
  echo "ERROR: This batch script requires exactly 18 Slurm tasks; got $MPI_RANKS" >&2
  exit 2
fi
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "ERROR: MPI_RANKS=$MPI_RANKS must be divisible by replicas=$LAMMPS_NEB_REPLICAS" >&2
  exit 2
fi

if [[ -d "$PACKAGE_DIR/kmc_map_inputs" ]]; then
  DEFAULT_MAP_ROOT="$PACKAGE_DIR/kmc_map_inputs"
elif [[ -d "/scratch/rtirunelveli/LAMMPS/kmc_cluster_lammps_only/kmc_map_inputs" ]]; then
  DEFAULT_MAP_ROOT="/scratch/rtirunelveli/LAMMPS/kmc_cluster_lammps_only/kmc_map_inputs"
else
  DEFAULT_MAP_ROOT="$PACKAGE_DIR/kmc_map_inputs"
fi
MAP_ROOT="${KMC_MAP_ROOT:-$DEFAULT_MAP_ROOT}"
export KMC_SITE_MAP_FILE="${KMC_SITE_MAP_FILE:-$MAP_ROOT/sigma5_stage3_unified_sites.npz}"
export KMC_HOST_STRUCTURE_FILE="${KMC_HOST_STRUCTURE_FILE:-$MAP_ROOT/sigma5_210-20-20-5.lmp}"
export KMC_INITIAL_H_REGION_FILE="${KMC_INITIAL_H_REGION_FILE:-$MAP_ROOT/sigma5_site_regions.npz}"
export POTENTIAL_EAM_FILE="${POTENTIAL_EAM_FILE:-$PACKAGE_DIR/kmc/PotentialB3410-modified.fs}"
for path in \
  "$KMC_SITE_MAP_FILE" \
  "$KMC_HOST_STRUCTURE_FILE" \
  "$KMC_INITIAL_H_REGION_FILE" \
  "$POTENTIAL_EAM_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: Required input is missing or empty: $path" >&2
    exit 2
  fi
done

SEED_BASE="${KMC_SEED_BASE:-$DEFAULT_SEED_BASE}"
export KMC_SEED=$((SEED_BASE + TASK_ID))
export KMC_INITIAL_H_REGIONS="$INITIAL_REGIONS"
export NUM_H=50
export MPI_RANKS
export LAMMPS_NEB_REPLICAS
export FINAL_STEP="${FINAL_STEP:-2000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-50000}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
CAMPAIGN_BASE="${CAMPAIGN_BASE:-$PACKAGE_DIR/runs/external_${LAYOUT_TAG}_50H_fenwick_2m_50k_array_${ARRAY_JOB_ID}}"
export CAMPAIGN_ROOT="${TASK_CAMPAIGN_ROOT:-$CAMPAIGN_BASE/${LAYOUT_TAG}_seed${KMC_SEED}_task${TASK_ID}}"

echo "-----------------------------------------------------"
echo "External-map fixed-50H Fenwick array task"
echo "START_TIME = $(date +'%Y-%m-%d %H:%M:%S %s')"
echo "HOST       = $(hostname)"
echo "PACKAGE    = $PACKAGE_DIR"
echo "ARRAY_JOB  = $ARRAY_JOB_ID"
echo "ARRAY_TASK = $TASK_ID"
echo "LAYOUT     = $LAYOUT_TAG"
echo "REGIONS    = $KMC_INITIAL_H_REGIONS"
echo "SEED_BASE  = $SEED_BASE"
echo "SEED       = $KMC_SEED"
echo "FINAL_STEP = $FINAL_STEP"
echo "SEGMENT    = $SEGMENT_STEPS"
echo "MPI_RANKS  = $MPI_RANKS"
echo "REPLICAS   = $LAMMPS_NEB_REPLICAS"
echo "NEB_GROUPS = $((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "CAMPAIGN   = $CAMPAIGN_ROOT"
echo "MAP_ROOT   = $MAP_ROOT"
echo "INITIAL_CACHE = ${INITIAL_CACHE:-fresh}"
echo "RESUME     = ${RESUME:-0}"
echo "-----------------------------------------------------"

START_TIME=$(date +%s)
if bash scripts/run_external_50h_fenwick_segmented.sh; then
  STATUS=0
else
  STATUS=$?
fi
END_TIME=$(date +%s)

mkdir -p "$CAMPAIGN_ROOT"
{
  printf 'array_job=%s\n' "$ARRAY_JOB_ID"
  printf 'array_task=%s\n' "$TASK_ID"
  printf 'layout=%s\n' "$LAYOUT_TAG"
  printf 'regions=%s\n' "$KMC_INITIAL_H_REGIONS"
  printf 'seed=%s\n' "$KMC_SEED"
  printf 'status=%s\n' "$STATUS"
  printf 'start_epoch=%s\n' "$START_TIME"
  printf 'end_epoch=%s\n' "$END_TIME"
  printf 'runtime_seconds=%s\n' "$((END_TIME - START_TIME))"
} > "$CAMPAIGN_ROOT/array_task_status.env"

echo "-----------------------------------------------------"
echo "END_TIME = $(date +'%Y-%m-%d %H:%M:%S %s')"
echo "RUN_TIME seconds = $((END_TIME - START_TIME))"
echo "EXIT_STATUS = $STATUS"
echo "-----------------------------------------------------"

exit "$STATUS"
