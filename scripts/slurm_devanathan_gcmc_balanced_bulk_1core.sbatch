#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=dev_bulk_gcmc_bal1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100gb
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm_logs/dev_bulk_gcmc_bal1-%j.out
#SBATCH --error=slurm_logs/dev_bulk_gcmc_bal1-%j.err

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

POTENTIAL_FILE="${POTENTIAL_EAM_FILE:-$PACKAGE_DIR/kmc/PotentialB3410-modified.fs}"
if [[ ! -s "$POTENTIAL_FILE" ]]; then
  echo "ERROR: Missing potential file: $POTENTIAL_FILE" >&2
  exit 2
fi

mkdir -p slurm_logs runs

export PYTHON_BIN="${PYTHON_BIN:-python}"
export MPI_RANKS="${SLURM_NTASKS:-1}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-1}"
export FINAL_STEP="${FINAL_STEP:-7000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-500000}"
export JOB_ASSIGNMENT_MODE="${JOB_ASSIGNMENT_MODE:-cache_only}"
export DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"
export PREFER_GCMC_MIGRATION_CACHE="${PREFER_GCMC_MIGRATION_CACHE:-1}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_source_energy_cache_eam_shell_IR8_OR10.pkl}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$PACKAGE_DIR/runs/devanathan_bulk_gcmc_balanced_1core_cacheonly_${SLURM_JOB_ID}}"

if (( MPI_RANKS != 1 )); then
  echo "ERROR: This batch script requires exactly 1 Slurm task; got $MPI_RANKS." >&2
  exit 2
fi
if [[ "$JOB_ASSIGNMENT_MODE" != "cache_only" ]]; then
  echo "ERROR: The 1-core benchmark is intended for JOB_ASSIGNMENT_MODE=cache_only." >&2
  echo "A true native-LAMMPS NEB miss requires multiple replicas." >&2
  exit 2
fi
if [[ ! -s "$DEVANATHAN_GCMC_ENERGY_CACHE_FILE" ]]; then
  echo "ERROR: Missing GCMC source-energy cache: $DEVANATHAN_GCMC_ENERGY_CACHE_FILE" >&2
  exit 2
fi

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
echo "Bulk Devanathan GCMC-balanced serial run, 1 task"
echo "This is a strict cache-only benchmark. Barrier misses will not run NEB."
echo "START_TIME = $(date +'%Y-%m-%d %H:%M:%S %s')"
echo "HOST       = $(hostname)"
echo "PACKAGE    = $PACKAGE_DIR"
echo "CAMPAIGN   = $CAMPAIGN_ROOT"
echo "FINAL_STEP = $FINAL_STEP"
echo "SEGMENT    = $SEGMENT_STEPS"
echo "MPI_RANKS  = $MPI_RANKS"
echo "REPLICAS   = $LAMMPS_NEB_REPLICAS"
echo "MODE       = $DEVANATHAN_GCMC_PRODUCTION_MODE"
echo "JOBS       = $JOB_ASSIGNMENT_MODE"
echo "GCMC_CACHE = $DEVANATHAN_GCMC_ENERGY_CACHE_FILE"
echo "INITIAL_CACHE = ${INITIAL_CACHE:-auto}"
echo "PYTHON     = $(command -v "$PYTHON_BIN")"
echo "MPIRUN     = $(command -v mpirun || true)"
echo "-----------------------------------------------------"

START_TIME=$(date +%s)
if bash scripts/run_devanathan_gcmc_balanced_serial_7m.sh; then
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
