#!/usr/bin/env bash
set -euo pipefail

# Fixed-mu GCMC calibration for filling one Sigma5 external-map bulk grain.
# This is GCMC only: no KMC propagation and no NEB jobs are launched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Users/rtirunelveli/kMC_code/python_kmc/bin/python}"
WORKERS="${WORKERS:-6}"
MU_VALUES="${MU_VALUES:--1.78,-1.77,-1.76,-1.75,-1.74,-1.73}"
REPLICAS="${REPLICAS:-2}"
STEPS="${STEPS:-160000}"
EQUIL="${EQUIL:-80000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-20}"
TRACE_INTERVAL="${TRACE_INTERVAL:-100}"
PRINT_INTERVAL="${PRINT_INTERVAL:-10000}"
DEVANATHAN_BULK_REGION="${DEVANATHAN_BULK_REGION:-bulk_grain_0}"
DEVANATHAN_BULK_X_MIN_A="${DEVANATHAN_BULK_X_MIN_A:-0}"
DEVANATHAN_BULK_SINK_GAP_A="${DEVANATHAN_BULK_SINK_GAP_A:-2.8601}"
TARGET_N_H="${TARGET_N_H:-50}"
RUN_ROOT="${RUN_ROOT:-runs/gcmc_sigma5_external_bulk_${DEVANATHAN_BULK_REGION}_target${TARGET_N_H}_sweep_$(date +%Y%m%d_%H%M%S)}"
PRELOAD_RUN="${PRELOAD_RUN:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ -d "$ROOT/kmc_map_inputs" ]]; then
  MAP_ROOT="$ROOT/kmc_map_inputs"
else
  MAP_ROOT="$WORKSPACE_ROOT/kmc_map_inputs"
fi

MAP_FILE="${KMC_SITE_MAP_FILE:-$MAP_ROOT/sigma5_stage3_unified_sites.npz}"
HOST_FILE="${KMC_HOST_STRUCTURE_FILE:-$MAP_ROOT/sigma5_210-20-20-5.lmp}"
REGION_FILE="${KMC_INITIAL_H_REGION_FILE:-$MAP_ROOT/sigma5_site_regions.npz}"
POTENTIAL_FILE="${POTENTIAL_EAM_FILE:-$ROOT/kmc/PotentialB3410-modified.fs}"

for path in "$MAP_FILE" "$HOST_FILE" "$REGION_FILE" "$POTENTIAL_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing or empty: $path" >&2
    exit 2
  fi
done

export LAMMPS_LIB_DIR="${LAMMPS_LIB_DIR:-/Users/rtirunelveli/lammps-29Aug2024/build}"
export DYLD_LIBRARY_PATH="${LAMMPS_LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self}"

echo "Sigma5 external bulk-grain GCMC mu sweep"
echo "  workers=$WORKERS"
echo "  mu_values=$MU_VALUES"
echo "  replicas=$REPLICAS"
echo "  steps=$STEPS equil=$EQUIL"
echo "  region=$DEVANATHAN_BULK_REGION target_N_H=$TARGET_N_H"
echo "  bulk_x_min=$DEVANATHAN_BULK_X_MIN_A bulk_sink_gap=$DEVANATHAN_BULK_SINK_GAP_A"
echo "  map=$MAP_FILE"
echo "  region_file=$REGION_FILE"
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
  --target-n-h "$TARGET_N_H"
  --region-file "$REGION_FILE"
  --region-labels "$DEVANATHAN_BULK_REGION"
  --bulk-x-min "$DEVANATHAN_BULK_X_MIN_A"
  --bulk-sink-gap "$DEVANATHAN_BULK_SINK_GAP_A"
  --site-map "$MAP_FILE"
  --host "$HOST_FILE"
  --potential "$POTENTIAL_FILE"
  --out-dir "$RUN_ROOT"
)

if [[ -n "${DEVANATHAN_SINK_X_MIN_A:-}" ]]; then
  command+=(--sink-x-min "$DEVANATHAN_SINK_X_MIN_A")
fi
if [[ -n "${DEVANATHAN_BULK_X_MAX_A:-}" ]]; then
  command+=(--bulk-x-max "$DEVANATHAN_BULK_X_MAX_A")
fi
if [[ -n "$PRELOAD_RUN" ]]; then
  command+=(--preload-run "$PRELOAD_RUN")
fi

"${command[@]}"
