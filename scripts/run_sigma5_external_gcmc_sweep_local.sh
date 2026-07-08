#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Users/rtirunelveli/kMC_code/python_kmc/bin/python}"
WORKERS="${WORKERS:-10}"
MU_VALUES="${MU_VALUES:--1.75,-1.74,-1.73,-1.72,-1.71}"
REPLICAS="${REPLICAS:-2}"
STEPS="${STEPS:-120000}"
EQUIL="${EQUIL:-60000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-20}"
TRACE_INTERVAL="${TRACE_INTERVAL:-100}"
PRINT_INTERVAL="${PRINT_INTERVAL:-10000}"
RUN_ROOT="${RUN_ROOT:-runs/gcmc_sigma5_external_charging_sweep_$(date +%Y%m%d_%H%M%S)}"
PRELOAD_RUN="${PRELOAD_RUN:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

export LAMMPS_LIB_DIR="${LAMMPS_LIB_DIR:-/Users/rtirunelveli/lammps-29Aug2024/build}"
export DYLD_LIBRARY_PATH="${LAMMPS_LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self

echo "Sigma5 external charging-zone GCMC sweep"
echo "  workers=$WORKERS"
echo "  mu_values=$MU_VALUES"
echo "  replicas=$REPLICAS"
echo "  steps=$STEPS equil=$EQUIL"
echo "  output=$RUN_ROOT"
echo "  preload_run=${PRELOAD_RUN:-none}"

command=(
  "$PYTHON_BIN" tools/gcmc_external_charging_sweep.py
  --python-bin "$PYTHON_BIN"
  --workers "$WORKERS"
  --mu-values="$MU_VALUES"
  --replicas "$REPLICAS"
  --steps "$STEPS"
  --equil "$EQUIL"
  --sample-interval "$SAMPLE_INTERVAL"
  --trace-interval "$TRACE_INTERVAL"
  --print-interval "$PRINT_INTERVAL"
  --site-map "$WORKSPACE_ROOT/kmc_map_inputs/sigma5_stage3_unified_sites.npz"
  --host "$WORKSPACE_ROOT/kmc_map_inputs/sigma5_210-20-20-5.lmp"
  --potential "$ROOT/kmc/PotentialB3410-modified.fs"
  --out-dir "$RUN_ROOT"
)
if [[ -n "$PRELOAD_RUN" ]]; then
  command+=(--preload-run "$PRELOAD_RUN")
fi
"${command[@]}"
