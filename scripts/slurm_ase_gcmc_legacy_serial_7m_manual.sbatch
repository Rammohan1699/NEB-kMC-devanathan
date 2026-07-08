#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=ase_gcmc_manual_7m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --mem=200gb
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm_logs/ase_gcmc_manual_7m-%j.out
#SBATCH --error=slurm_logs/ase_gcmc_manual_7m-%j.err

set -euo pipefail

PACKAGE_DIR="${KMC_PACKAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PACKAGE_DIR"
mkdir -p slurm_logs runs

if [[ "${LOAD_HYPERION_KMC_ENV:-1}" == "1" ]]; then
  module purge
  module load "${LAMMPS_MODULE:-LAMMPS/23Jun2022-foss-2022a-kokkos}"
  source ~/kmc_env/bin/activate
fi

export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,tcp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export MPI_RANKS="${SLURM_NTASKS:-18}"
export FINAL_STEP="${FINAL_STEP:-7000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-50000}"
export CACHE_SCHEMA="${CACHE_SCHEMA:-v4_envkey:mode:shell_IR5.0_OR6.0:env_plus_dir:R5.0_PB0.1_HB0.02:map:sigma5_stage3_unified_sites}"
export INITIAL_CACHE="${INITIAL_CACHE:-$PACKAGE_DIR/cache_bank/legacy_one_crystal_seed43_2m_ir5_or6.pkl}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-$PACKAGE_DIR/runs/gcmc_bulk_ase_legacy_energy_cache.pkl}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$PACKAGE_DIR/runs/devanathan_external_gcmc_bulk_ase_legacy_7m_50k_${SLURM_JOB_ID}}"

export DEVANATHAN_BULK_REGION="${DEVANATHAN_BULK_REGION:-bulk_grain_0}"
export DEVANATHAN_BULK_TARGET_N_H="${DEVANATHAN_BULK_TARGET_N_H:-50}"
export DEVANATHAN_BULK_TARGET_H_PER_FE="${DEVANATHAN_BULK_TARGET_H_PER_FE:-0.00311779011036977}"
export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:--1.75}"
export DEVANATHAN_GCMC_INITIALIZATION_START="${DEVANATHAN_GCMC_INITIALIZATION_START:-target}"
export DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE="${DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE:-target}"
export DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"
export DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT="${DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT:-8192}"
export DEVANATHAN_GCMC_DIAG_PROPOSAL_LIMIT="${DEVANATHAN_GCMC_DIAG_PROPOSAL_LIMIT:-32}"
export KMC_HOP_X_DIRECTION="${KMC_HOP_X_DIRECTION:-right}"

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("ase", "mpi4py", "lammps", "numpy", "scipy")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
print("Python dependency check OK")
PY

echo "PACKAGE=$PACKAGE_DIR"
echo "CAMPAIGN=$CAMPAIGN_ROOT"
echo "PYTHON=$(command -v "$PYTHON_BIN")"
echo "MPIRUN=$(command -v mpirun)"
echo "MPI_RANKS=$MPI_RANKS FINAL_STEP=$FINAL_STEP SEGMENT_STEPS=$SEGMENT_STEPS"
echo "INITIAL_CACHE=$INITIAL_CACHE"

if ! command -v mpirun >/dev/null 2>&1; then
  echo "ERROR: mpirun is not available. The Hyperion LAMMPS/OpenMPI module did not load." >&2
  exit 127
fi

bash scripts/run_devanathan_external_gcmc_bulk_ase_legacy_serial_7m.sh
