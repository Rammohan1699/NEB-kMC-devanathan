#!/usr/bin/env bash
set -euo pipefail

# Run a native LAMMPS-only pulse rerun in an existing paired campaign using
# ASE legacy shell settings plus native LAMMPS endpoint minimization before NEB.
#
# Usage:
#   bash scripts/run_external_pulse50_lammps_only_lmp_endpoint_opt_in_pair.sh [CAMPAIGN_ROOT]
#
# Useful overrides:
#   LMP_ENDPOINT_OPT_RUN_NAME=...        New subfolder name inside CAMPAIGN_ROOT.
#   LMP_ENDPOINT_OPT_RUN_ROOT=...        Exact output folder to use.
#   LAMMPS_ENDPOINT_OPT_FTOL=0.05        Endpoint minimization force tolerance.
#   LAMMPS_ENDPOINT_OPT_STEPS=200        Endpoint minimization max iterations.
#   LAMMPS_ENDPOINT_OPT_MIN_STYLE=fire   Endpoint minimization style.
#   SKIP_COMPARE=1                      Do not compare against ase_neb afterward.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LMP_ASE_SHELL_NEB_ENGINE_MODE="lammps_only_endpoint_opt"
export LAMMPS_ONLY_ENDPOINT_OPTIMIZE="${LAMMPS_ONLY_ENDPOINT_OPTIMIZE:-1}"
export LMP_ASE_SHELL_RUN_NAME="${LMP_ENDPOINT_OPT_RUN_NAME:-lammps_only_lmp_endpoint_opt_ase_shell}"
if [[ -n "${LMP_ENDPOINT_OPT_RUN_ROOT:-}" ]]; then
  export LMP_ASE_SHELL_RUN_ROOT="$LMP_ENDPOINT_OPT_RUN_ROOT"
fi

bash "$SCRIPT_DIR/run_external_pulse50_lammps_only_ase_shell_in_pair.sh" "$@"
