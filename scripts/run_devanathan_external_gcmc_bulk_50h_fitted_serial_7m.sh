#!/usr/bin/env bash
set -euo pipefail

# Segmented 7M/50k external-map Devanathan production using the fitted
# 50-H bulk-GCMC initialization and maintenance target.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="${DEVANATHAN_PRACTICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PRACTICE_ROOT"

if [[ -z "${PYTHON_BIN:-}" && -x "/Users/rtirunelveli/kMC_code/python_kmc/bin/python" ]]; then
  export PYTHON_BIN="/Users/rtirunelveli/kMC_code/python_kmc/bin/python"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PRACTICE_ROOT}/.venv/bin/python" ]]; then
    export PYTHON_BIN="${PRACTICE_ROOT}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find a Python interpreter. Set PYTHON_BIN." >&2
    exit 1
  fi
fi

SWEEP_RUN_DEFAULT="runs/gcmc_sigma5_external_bulk_bulk_grain_0_target50_sweep_20260626_220038"
SWEEP_RUN="${GCMC_BULK_FIT_SWEEP_RUN:-$SWEEP_RUN_DEFAULT}"
if [[ "$SWEEP_RUN" != /* ]]; then
  SWEEP_RUN="$PRACTICE_ROOT/$SWEEP_RUN"
fi

recommendation="$SWEEP_RUN/recommendation.txt"
selection="$SWEEP_RUN/selection_summary.json"
if [[ ! -s "$recommendation" ]]; then
  echo "Missing fitted GCMC recommendation: $recommendation" >&2
  exit 2
fi
if [[ ! -s "$selection" ]]; then
  echo "Missing fitted GCMC selection summary: $selection" >&2
  exit 2
fi

read_fit_value() {
  local key="$1"
  local path="$2"
  awk -F= -v key="$key" '$1 == key { print $2; found=1 } END { exit found ? 0 : 1 }' "$path"
}

fit_mu="$(read_fit_value closest_mu_eV "$recommendation")"
fit_target_n_h="$(read_fit_value target_N_H "$recommendation")"
fit_target_h_per_fe="$("$PYTHON_BIN" -c 'import json, sys; data=json.load(open(sys.argv[1])); print("{:.17g}".format(float(data["target_c_H_per_Fe"])))' "$selection")"
fit_region="$("$PYTHON_BIN" -c 'import json, sys; data=json.load(open(sys.argv[1])); print(",".join(str(x) for x in data["region_labels"]))' "$selection")"

export DEVANATHAN_BULK_REGION="${DEVANATHAN_BULK_REGION:-$fit_region}"
export DEVANATHAN_BULK_TARGET_N_H="${DEVANATHAN_BULK_TARGET_N_H:-$fit_target_n_h}"
export DEVANATHAN_BULK_TARGET_H_PER_FE="${DEVANATHAN_BULK_TARGET_H_PER_FE:-$fit_target_h_per_fe}"
export DEVANATHAN_GCMC_MU_EV="${DEVANATHAN_GCMC_MU_EV:-$fit_mu}"
export DEVANATHAN_GCMC_INITIALIZATION_START="${DEVANATHAN_GCMC_INITIALIZATION_START:-target}"
export DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE="${DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE:-target}"
export DEVANATHAN_GCMC_PRODUCTION_MODE="${DEVANATHAN_GCMC_PRODUCTION_MODE:-insert_delete}"
export DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX="${DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX:-1}"
export DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE="${DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE:-0.0002}"

export FINAL_STEP="${FINAL_STEP:-7000000}"
export SEGMENT_STEPS="${SEGMENT_STEPS:-50000}"
export MPI_RANKS="${MPI_RANKS:-6}"
export LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"

target_tag="${DEVANATHAN_BULK_TARGET_N_H//./p}"
mu_tag="${DEVANATHAN_GCMC_MU_EV/-/m}"
mu_tag="${mu_tag//./p}"
region_tag="${DEVANATHAN_BULK_REGION//[^A-Za-z0-9_]/_}"
export DEVANATHAN_GCMC_ENERGY_CACHE_FILE="${DEVANATHAN_GCMC_ENERGY_CACHE_FILE:-runs/gcmc_external_bulk_energy_cache_${region_tag}_${target_tag}H_mu_${mu_tag}.pkl}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-runs/devanathan_external_gcmc_bulk_${region_tag}_${target_tag}H_mu_${mu_tag}_7m_50k_$(date +%Y%m%d_%H%M%S)}"

echo "External-map fitted 50-H bulk-GCMC segmented local run"
echo "  sweep_fit=$SWEEP_RUN"
echo "  fitted_mu_eV=$DEVANATHAN_GCMC_MU_EV"
echo "  target_N_H=$DEVANATHAN_BULK_TARGET_N_H"
echo "  target_H_per_Fe=$DEVANATHAN_BULK_TARGET_H_PER_FE"
echo "  bulk_region=$DEVANATHAN_BULK_REGION"
echo "  final_step=$FINAL_STEP segment_steps=$SEGMENT_STEPS"
echo "  mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  campaign=$CAMPAIGN_ROOT"

bash "$SCRIPT_DIR/run_devanathan_external_gcmc_bulk_serial_7m.sh"
