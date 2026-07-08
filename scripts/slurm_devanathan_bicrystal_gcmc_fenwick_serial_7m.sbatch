#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=dev_sigma5_gcmc_7m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --mem=100gb
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm_logs/dev_sigma5_gcmc_7m-%j.out
#SBATCH --error=slurm_logs/dev_sigma5_gcmc_7m-%j.err

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
  echo "Submit from the package root or set KMC_PACKAGE_DIR explicitly." >&2
  exit 2
fi
cd "$PACKAGE_DIR"

if [[ ! -f "kmc_map_inputs/sigma5_stage3_unified_sites.npz" ]]; then
  echo "ERROR: Missing kmc_map_inputs/sigma5_stage3_unified_sites.npz" >&2
  exit 2
fi
if [[ ! -f "kmc_map_inputs/sigma5_210-20-20-5.lmp" ]]; then
  echo "ERROR: Missing kmc_map_inputs/sigma5_210-20-20-5.lmp" >&2
  exit 2
fi

mkdir -p slurm_logs runs

export PYTHON_BIN="${PYTHON_BIN:-python}"
export MPI_RANKS="${SLURM_NTASKS:-18}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
export FINAL_STEP="${FINAL_STEP:-7000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-500000}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${PACKAGE_DIR}/runs/devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p75_7m_serial_${SLURM_JOB_ID}}"

if (( MPI_RANKS != 18 )); then
  echo "ERROR: This batch script requires exactly 18 Slurm tasks; got $MPI_RANKS." >&2
  exit 2
fi
if (( MPI_RANKS % LAMMPS_NEB_REPLICAS != 0 )); then
  echo "ERROR: MPI_RANKS=$MPI_RANKS must be divisible by LAMMPS_NEB_REPLICAS=$LAMMPS_NEB_REPLICAS." >&2
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
echo "Sigma5 bicrystal Devanathan GCMC/Fenwick serial run"
echo "START_TIME = $(date +'%Y-%m-%d %H:%M:%S %s')"
echo "HOST       = $(hostname)"
echo "PACKAGE    = $PACKAGE_DIR"
echo "CAMPAIGN   = $CAMPAIGN_ROOT"
echo "FINAL_STEP = $FINAL_STEP"
echo "SEGMENT    = $SEGMENT_STEPS"
echo "MPI_RANKS  = $MPI_RANKS"
echo "REPLICAS   = $LAMMPS_NEB_REPLICAS"
echo "NEB_GROUPS = $((MPI_RANKS / LAMMPS_NEB_REPLICAS))"
echo "PYTHON     = $(command -v "$PYTHON_BIN")"
echo "MPIRUN     = $(command -v mpirun)"
echo "INITIAL_CACHE = ${INITIAL_CACHE:-fresh}"
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
