#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=ext_gcmc_50h_7m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --mem=200gb
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm_logs/ext_gcmc_50h_7m-%j.out
#SBATCH --error=slurm_logs/ext_gcmc_50h_7m-%j.err

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
mkdir -p slurm_logs runs

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

export PYTHON_BIN="${PYTHON_BIN:-python}"
export MPI_RANKS="${SLURM_NTASKS:-18}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export FINAL_STEP="${FINAL_STEP:-7000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-50000}"
export GCMC_BULK_FIT_SWEEP_RUN="${GCMC_BULK_FIT_SWEEP_RUN:-$PACKAGE_DIR/runs/gcmc_sigma5_external_bulk_bulk_grain_0_target50_sweep_20260626_220038}"
export DEVANATHAN_BULK_REGION="${DEVANATHAN_BULK_REGION:-bulk_grain_0}"
export DEVANATHAN_BULK_TARGET_N_H="${DEVANATHAN_BULK_TARGET_N_H:-50}"
export DEVANATHAN_BULK_TARGET_H_PER_FE="${DEVANATHAN_BULK_TARGET_H_PER_FE:-0.0031177901103697698}"
export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}"
export DEVANATHAN_GCMC_INITIALIZATION_START="${DEVANATHAN_GCMC_INITIALIZATION_START:-target}"
export DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE="${DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE:-target}"
export DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"
export DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE="${DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE:-0.0002}"
export DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX="${DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX:-1}"
export CACHE_SCHEMA="${CACHE_SCHEMA:-}"

if (( MPI_RANKS != 18 )); then
  echo "ERROR: This batch script requires exactly 18 Slurm tasks; got $MPI_RANKS." >&2
  exit 2
fi
if (( LAMMPS_NEB_REPLICAS != 3 )); then
  echo "ERROR: This batch script expects LAMMPS_NEB_REPLICAS=3; got $LAMMPS_NEB_REPLICAS." >&2
  exit 2
fi

mu_tag="${DEVANATHAN_GCMC_MU_EV/-/m}"
mu_tag="${mu_tag//./p}"
region_tag="${DEVANATHAN_BULK_REGION//[^A-Za-z0-9_]/_}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$PACKAGE_DIR/runs/devanathan_external_gcmc_bulk_${region_tag}_${DEVANATHAN_BULK_TARGET_N_H}H_mu_${mu_tag}_7m_50k_${SLURM_JOB_ID}}"

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

echo "-----------------------------------------------------"
echo "External-map fitted 50-H GCMC-bulk Devanathan serial run"
echo "START_TIME = $(date +'%Y-%m-%d %H:%M:%S %s')"
echo "HOST       = $(hostname)"
echo "PACKAGE    = $PACKAGE_DIR"
echo "CAMPAIGN   = $CAMPAIGN_ROOT"
echo "SWEEP_FIT  = $GCMC_BULK_FIT_SWEEP_RUN"
echo "FINAL_STEP = $FINAL_STEP"
echo "SEGMENT    = $SEGMENT_STEPS"
echo "MPI_RANKS  = $MPI_RANKS"
echo "REPLICAS   = $LAMMPS_NEB_REPLICAS"
echo "NEB_GROUPS = $((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "BULK_REGION = $DEVANATHAN_BULK_REGION"
echo "TARGET_N_H = $DEVANATHAN_BULK_TARGET_N_H"
echo "TARGET_H_PER_FE = $DEVANATHAN_BULK_TARGET_H_PER_FE"
echo "MU_EV      = $DEVANATHAN_GCMC_MU_EV"
echo "MAP        = $KMC_SITE_MAP_FILE"
echo "REGIONS    = $KMC_INITIAL_H_REGION_FILE"
echo "PYTHON     = $(command -v "$PYTHON_BIN")"
echo "MPIRUN     = $(command -v mpirun)"
echo "RESUME     = ${RESUME:-0}"
echo "-----------------------------------------------------"

START_TIME=$(date +%s)
if bash scripts/run_devanathan_external_gcmc_bulk_50h_fitted_serial_7m.sh; then
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
