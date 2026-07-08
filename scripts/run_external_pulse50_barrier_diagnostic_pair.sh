#!/usr/bin/env bash
set -euo pipefail

# Run the same finite 50-H pulse with native LAMMPS NEB and ASE/LAMMPSlib NEB,
# then summarize barrier and selected-event differences.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${KMC_PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PACKAGE_DIR"

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

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$PACKAGE_DIR/runs/external_pulse50_barrier_pair_$(date +%Y%m%d_%H%M%S)}"
LMP_RUN_ROOT="${LMP_RUN_ROOT:-$CAMPAIGN_ROOT/lammps_only}"
ASE_RUN_ROOT="${ASE_RUN_ROOT:-$CAMPAIGN_ROOT/ase_neb}"

echo "Paired Sigma5 50H pulse barrier diagnostic"
echo "  package=$PACKAGE_DIR"
echo "  campaign=$CAMPAIGN_ROOT"
echo "  lammps_only_run=$LMP_RUN_ROOT"
echo "  ase_neb_run=$ASE_RUN_ROOT"

NEB_ENGINE_MODE=lammps_only RUN_ROOT="$LMP_RUN_ROOT" \
  bash "$SCRIPT_DIR/run_external_pulse50_barrier_diagnostic.sh"

NEB_ENGINE_MODE=ase_neb RUN_ROOT="$ASE_RUN_ROOT" \
  bash "$SCRIPT_DIR/run_external_pulse50_barrier_diagnostic.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"$PYTHON_BIN" "$PACKAGE_DIR/tools/compare_barrier_discrepancy_runs.py" \
  --lammps-run "$LMP_RUN_ROOT" \
  --ase-run "$ASE_RUN_ROOT" \
  --out-dir "$CAMPAIGN_ROOT/analysis"
