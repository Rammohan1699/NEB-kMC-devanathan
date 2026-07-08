#!/usr/bin/env bash
set -euo pipefail

# Run only the native LAMMPS-NEB side of an existing paired Sigma5 50-H pulse
# diagnostic campaign, but use the shell/extraction settings from the ASE run.
#
# Usage:
#   bash scripts/run_external_pulse50_lammps_only_ase_shell_in_pair.sh [CAMPAIGN_ROOT]
#
# Useful overrides:
#   CAMPAIGN_ROOT=...                         Existing external_pulse50_barrier_pair_* folder.
#   LMP_ASE_SHELL_RUN_NAME=lammps_only_5_6    New subfolder name inside CAMPAIGN_ROOT.
#   LMP_ASE_SHELL_RUN_ROOT=...                Exact output folder to use.
#   LMP_ASE_SHELL_NEB_ENGINE_MODE=...         Native engine mode to run.
#   ASE_RUN_ROOT=...                          ASE run to read shell settings from.
#   BASE_LMP_RUN_ROOT=...                     Existing lammps_only run to read pulse settings from.
#   ALLOW_EXISTING_RUN_ROOT=1                 Reuse an existing output folder.
#   SKIP_COMPARE=1                            Do not run comparison after LAMMPS completes.

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

BASE_LMP_RUN_ROOT="${BASE_LMP_RUN_ROOT:-$CAMPAIGN_ROOT/lammps_only}"
BASE_LMP_RUN_ROOT="$(resolve_dir "$BASE_LMP_RUN_ROOT")"
BASE_LMP_SNAPSHOT="$BASE_LMP_RUN_ROOT/run_config_snapshot.env"
if [[ ! -s "$BASE_LMP_SNAPSHOT" ]]; then
  echo "Missing base lammps_only snapshot: $BASE_LMP_SNAPSHOT" >&2
  exit 2
fi

ASE_RUN_ROOT="${ASE_RUN_ROOT:-$CAMPAIGN_ROOT/ase_neb}"
ASE_RUN_ROOT="$(resolve_dir "$ASE_RUN_ROOT")"
ASE_SNAPSHOT="$ASE_RUN_ROOT/run_config_snapshot.env"

LMP_ASE_SHELL_RUN_ROOT_WAS_SET=0
[[ -n "${LMP_ASE_SHELL_RUN_ROOT:-}" ]] && LMP_ASE_SHELL_RUN_ROOT_WAS_SET=1

if [[ "$LMP_ASE_SHELL_RUN_ROOT_WAS_SET" == "1" ]]; then
  if [[ "$LMP_ASE_SHELL_RUN_ROOT" != /* ]]; then
    LMP_ASE_SHELL_RUN_ROOT="$CALL_DIR/$LMP_ASE_SHELL_RUN_ROOT"
  fi
else
  LMP_ASE_SHELL_RUN_NAME="${LMP_ASE_SHELL_RUN_NAME:-lammps_only_ase_shell}"
  LMP_ASE_SHELL_RUN_ROOT="$CAMPAIGN_ROOT/$LMP_ASE_SHELL_RUN_NAME"
fi

LMP_ASE_SHELL_RUN_PARENT="$(dirname "$LMP_ASE_SHELL_RUN_ROOT")"
mkdir -p "$LMP_ASE_SHELL_RUN_PARENT"
LMP_ASE_SHELL_RUN_ROOT="$(cd "$LMP_ASE_SHELL_RUN_PARENT" && pwd)/$(basename "$LMP_ASE_SHELL_RUN_ROOT")"
LMP_ASE_SHELL_RUN_BASENAME="$(basename "$LMP_ASE_SHELL_RUN_ROOT")"
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
  local file="$1"
  local key="$2"
  awk -v key="$key" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "$file"
}

export_if_unset_from_snapshot() {
  local file="$1"
  local key="$2"
  local value
  if [[ -n "${!key:-}" || ! -s "$file" ]]; then
    return
  fi
  value="$(snapshot_value "$file" "$key")"
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
  LAMMPS_NEB_DEBUG_MODE \
  WRITE_RATES_ALLRANKS \
  DEBUG_MODE \
  VALIDATION_MODE
do
  export_if_unset_from_snapshot "$BASE_LMP_SNAPSHOT" "$key"
done

for key in \
  SHELL_INNER_RADIUS_A \
  SHELL_OUTER_RADIUS_A \
  OPTIMIZE_ENDPOINTS \
  NEB_OPTIMIZER_FMAX \
  NEB_OPTIMIZER_STEPS \
  ENDPOINT_OPTIMIZER_STEPS
do
  export_if_unset_from_snapshot "$ASE_SNAPSHOT" "$key"
done

# ASE-shell comparison defaults. The merge mode stays global because this is
# still the native LAMMPS-only engine path.
export ENV_KEY_MODE="${ENV_KEY_MODE:-env_plus_dir}"
export ENV_RADIUS_A="${ENV_RADIUS_A:-5.0}"
export POS_BIN_A="${POS_BIN_A:-0.10}"
export HOP_BIN_A="${HOP_BIN_A:-0.02}"
export SHELL_INNER_RADIUS_A="${SHELL_INNER_RADIUS_A:-5.0}"
export SHELL_OUTER_RADIUS_A="${SHELL_OUTER_RADIUS_A:-6.0}"
export OPTIMIZE_ENDPOINTS="${OPTIMIZE_ENDPOINTS:-1}"
export BARRIER_MERGE_MODE="${BARRIER_MERGE_MODE:-global}"
LMP_ASE_SHELL_NEB_ENGINE_MODE="${LMP_ASE_SHELL_NEB_ENGINE_MODE:-lammps_only}"

echo "LAMMPS-only Sigma5 50H pulse diagnostic using ASE shell settings"
echo "  package=$PACKAGE_DIR"
echo "  campaign=$CAMPAIGN_ROOT"
echo "  base_lammps_only_run=$BASE_LMP_RUN_ROOT"
echo "  ase_settings_run=$ASE_RUN_ROOT"
echo "  lammps_only_ase_shell_run=$LMP_ASE_SHELL_RUN_ROOT"
echo "  ranks=${MPI_RANKS:-${NP:-6}} replicas=${LAMMPS_NEB_REPLICAS:-3}"
echo "  shell_inner=$SHELL_INNER_RADIUS_A shell_outer=$SHELL_OUTER_RADIUS_A"
echo "  optimize_endpoints=$OPTIMIZE_ENDPOINTS"
echo "  neb_engine_mode=$LMP_ASE_SHELL_NEB_ENGINE_MODE"
echo "  barrier_merge_mode=$BARRIER_MERGE_MODE"

NEB_ENGINE_MODE="$LMP_ASE_SHELL_NEB_ENGINE_MODE" RUN_ROOT="$LMP_ASE_SHELL_RUN_ROOT" \
  bash "$SCRIPT_DIR/run_external_pulse50_barrier_diagnostic.sh"

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_COMPARE:-0}" == "1" ]]; then
  exit 0
fi

if [[ -s "$ASE_RUN_ROOT/diagnostics/rates_allranks.csv" ]]; then
  ANALYSIS_OUT="${ANALYSIS_OUT:-$CAMPAIGN_ROOT/analysis_${LMP_ASE_SHELL_RUN_BASENAME}_vs_${ASE_RUN_BASENAME}}"
  "$PYTHON_BIN" "$PACKAGE_DIR/tools/compare_barrier_discrepancy_runs.py" \
    --lammps-run "$LMP_ASE_SHELL_RUN_ROOT" \
    --ase-run "$ASE_RUN_ROOT" \
    --out-dir "$ANALYSIS_OUT"
  echo "LAMMPS-only ASE-shell run complete"
  echo "  lammps_only_ase_shell_run=$LMP_ASE_SHELL_RUN_ROOT"
  echo "  analysis=$ANALYSIS_OUT"
else
  echo "LAMMPS-only ASE-shell run complete"
  echo "  lammps_only_ase_shell_run=$LMP_ASE_SHELL_RUN_ROOT"
  echo "  comparison skipped because ASE diagnostics were not found under $ASE_RUN_ROOT" >&2
fi
