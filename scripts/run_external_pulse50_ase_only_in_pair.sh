#!/usr/bin/env bash
set -euo pipefail

# Run only the ASE/LAMMPSlib side of an existing paired Sigma5 50-H pulse
# diagnostic campaign. This is for the case where lammps_only has already
# completed and only ase_neb needs to be rerun.
#
# Usage:
#   bash scripts/run_external_pulse50_ase_only_in_pair.sh [CAMPAIGN_ROOT]
#
# Useful overrides:
#   CAMPAIGN_ROOT=...          Existing external_pulse50_barrier_pair_* folder.
#   ASE_RUN_ROOT=...           Exact ASE output folder to use.
#   ASE_RUN_NAME=ase_neb_fast  ASE subfolder name inside CAMPAIGN_ROOT.
#   ALLOW_EXISTING_RUN_ROOT=1  Reuse an existing ASE_RUN_ROOT.
#   SKIP_COMPARE=1            Do not run comparison after ASE completes.
#   SHELL_INNER_RADIUS_A=8.0   Override the legacy 5/6 A extraction shell.
#   SHELL_OUTER_RADIUS_A=10.0  Override the legacy 5/6 A extraction shell.
#   OPTIMIZE_ENDPOINTS=0       Disable endpoint optimization.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${KMC_PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CALL_DIR="$(pwd)"
cd "$PACKAGE_DIR"

usage() {
  sed -n '2,18p' "$0" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage
  exit 2
fi

resolve_dir() {
  local path="$1"
  if [[ -d "$path" ]]; then
    (cd "$path" && pwd)
  elif [[ -d "$CALL_DIR/$path" ]]; then
    (cd "$CALL_DIR/$path" && pwd)
  else
    (cd "$path" && pwd)
  fi
}

find_latest_campaign() {
  local -a candidates=()
  local d

  if [[ -d "$CALL_DIR/lammps_only" && "$(basename "$CALL_DIR")" == external_pulse50_barrier_pair_* ]]; then
    candidates+=("$CALL_DIR")
  fi

  shopt -s nullglob
  for d in "$PACKAGE_DIR"/runs/external_pulse50_barrier_pair_*; do
    [[ -d "$d/lammps_only" ]] || continue
    candidates+=("$d")
  done
  shopt -u nullglob

  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${candidates[@]}" | sort | tail -n 1
}

if [[ -n "${1:-}" ]]; then
  CAMPAIGN_ROOT="$1"
elif [[ -z "${CAMPAIGN_ROOT:-}" ]]; then
  if ! CAMPAIGN_ROOT="$(find_latest_campaign)"; then
    echo "No external_pulse50_barrier_pair_* campaign with lammps_only was found." >&2
    echo "Pass CAMPAIGN_ROOT explicitly." >&2
    exit 2
  fi
fi
CAMPAIGN_ROOT="$(resolve_dir "$CAMPAIGN_ROOT")"

LMP_RUN_ROOT="${LMP_RUN_ROOT:-$CAMPAIGN_ROOT/lammps_only}"
LMP_RUN_ROOT="$(resolve_dir "$LMP_RUN_ROOT")"
LMP_SNAPSHOT="$LMP_RUN_ROOT/run_config_snapshot.env"
if [[ ! -s "$LMP_SNAPSHOT" ]]; then
  echo "Missing lammps_only snapshot: $LMP_SNAPSHOT" >&2
  exit 2
fi

ASE_RUN_ROOT_WAS_SET=0
ASE_RUN_NAME_WAS_SET=0
[[ -n "${ASE_RUN_ROOT:-}" ]] && ASE_RUN_ROOT_WAS_SET=1
[[ -n "${ASE_RUN_NAME:-}" ]] && ASE_RUN_NAME_WAS_SET=1

if [[ "$ASE_RUN_ROOT_WAS_SET" == "1" ]]; then
  ASE_RUN_ROOT="$ASE_RUN_ROOT"
else
  ASE_RUN_NAME="${ASE_RUN_NAME:-ase_neb}"
  ASE_RUN_ROOT="$CAMPAIGN_ROOT/$ASE_RUN_NAME"
fi

if [[ "$ASE_RUN_ROOT_WAS_SET" == "0" && "$ASE_RUN_NAME_WAS_SET" == "0" && -e "$ASE_RUN_ROOT" && "${ALLOW_EXISTING_RUN_ROOT:-0}" != "1" ]]; then
  ASE_RUN_ROOT="$CAMPAIGN_ROOT/ase_neb_retry_$(date +%Y%m%d_%H%M%S)"
fi

ASE_RUN_PARENT="$(dirname "$ASE_RUN_ROOT")"
mkdir -p "$ASE_RUN_PARENT"
ASE_RUN_ROOT="$(cd "$ASE_RUN_PARENT" && pwd)/$(basename "$ASE_RUN_ROOT")"
ASE_RUN_BASENAME="$(basename "$ASE_RUN_ROOT")"

if [[ -z "${PYTHON_BIN:-}" && -x "/Users/rtirunelveli/kMC_code/python_kmc/bin/python" ]]; then
  PYTHON_BIN="/Users/rtirunelveli/kMC_code/python_kmc/bin/python"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PACKAGE_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PACKAGE_DIR/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find a Python interpreter. Set PYTHON_BIN." >&2
    exit 1
  fi
fi
export PYTHON_BIN

snapshot_value() {
  local key="$1"
  awk -v key="$key" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "$LMP_SNAPSHOT"
}

export_if_unset_from_snapshot() {
  local key="$1"
  local value
  if [[ -n "${!key:-}" ]]; then
    return
  fi
  value="$(snapshot_value "$key")"
  if [[ -n "$value" ]]; then
    export "$key=$value"
  fi
}

for key in \
  STEPS \
  NUM_H \
  KMC_SEED \
  DEVANATHAN_SEED \
  DEVANATHAN_PULSE_REGION \
  DEVANATHAN_SOURCE_X_MIN_A \
  DEVANATHAN_SOURCE_X_MAX_A \
  DEVANATHAN_SINK_X_MIN_A \
  KMC_SITE_MAP_FILE \
  KMC_HOST_STRUCTURE_FILE \
  KMC_INITIAL_H_REGION_FILE \
  MPI_RANKS \
  LAMMPS_NEB_REPLICAS \
  LAMMPS_NEB_STEPS \
  LAMMPS_NEB_FTOL \
  WRITE_RATES_ALLRANKS \
  DEBUG_MODE \
  VALIDATION_MODE
do
  export_if_unset_from_snapshot "$key"
done

# Defaults chosen for the ASE retry path: use the legacy extraction shell and
# endpoint optimization used in the current barrier-comparison workflow.
export ENV_KEY_MODE="${ENV_KEY_MODE:-env_plus_dir}"
export ENV_RADIUS_A="${ENV_RADIUS_A:-5.0}"
export POS_BIN_A="${POS_BIN_A:-0.10}"
export HOP_BIN_A="${HOP_BIN_A:-0.02}"
export SHELL_INNER_RADIUS_A="${SHELL_INNER_RADIUS_A:-5.0}"
export SHELL_OUTER_RADIUS_A="${SHELL_OUTER_RADIUS_A:-6.0}"
export OPTIMIZE_ENDPOINTS="${OPTIMIZE_ENDPOINTS:-1}"
export BARRIER_MERGE_MODE="${BARRIER_MERGE_MODE:-local}"
export VALIDATION_ASE_DUMP="${VALIDATION_ASE_DUMP:-0}"
export VALIDATION_ASE_DUMP_PRE="${VALIDATION_ASE_DUMP_PRE:-0}"

echo "ASE-only Sigma5 50H pulse diagnostic"
echo "  package=$PACKAGE_DIR"
echo "  campaign=$CAMPAIGN_ROOT"
echo "  lammps_only_run=$LMP_RUN_ROOT"
echo "  ase_neb_run=$ASE_RUN_ROOT"
echo "  ranks=${MPI_RANKS:-${NP:-6}}"
echo "  shell_inner=$SHELL_INNER_RADIUS_A shell_outer=$SHELL_OUTER_RADIUS_A"
echo "  optimize_endpoints=$OPTIMIZE_ENDPOINTS"
echo "  barrier_merge_mode=$BARRIER_MERGE_MODE"

NEB_ENGINE_MODE=ase_neb RUN_ROOT="$ASE_RUN_ROOT" \
  bash "$SCRIPT_DIR/run_external_pulse50_barrier_diagnostic.sh"

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_COMPARE:-0}" == "1" ]]; then
  exit 0
fi

if [[ "$ASE_RUN_BASENAME" == "ase_neb" ]]; then
  ANALYSIS_OUT="${ANALYSIS_OUT:-$CAMPAIGN_ROOT/analysis}"
else
  ANALYSIS_OUT="${ANALYSIS_OUT:-$CAMPAIGN_ROOT/analysis_${ASE_RUN_BASENAME}}"
fi

"$PYTHON_BIN" "$PACKAGE_DIR/tools/compare_barrier_discrepancy_runs.py" \
  --lammps-run "$LMP_RUN_ROOT" \
  --ase-run "$ASE_RUN_ROOT" \
  --out-dir "$ANALYSIS_OUT"

echo "ASE-only run complete"
echo "  ase_neb_run=$ASE_RUN_ROOT"
echo "  analysis=$ANALYSIS_OUT"
