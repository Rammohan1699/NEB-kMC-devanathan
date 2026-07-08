#!/usr/bin/env bash
set -euo pipefail

# End-to-end validation helper for the refactored KMC driver.
#
# Default behavior:
#   1. Run python -m kmc.main with VALIDATION_MODE=1 and ASE NEB dumps.
#   2. Generate native LAMMPS NEB job folders using helper/lammps_neb.sh.
#   3. Build a comparison CSV using helper/extractor.py.
#
# Native LAMMPS jobs are generated but not executed unless RUN_LAMMPS=1.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/Users/rtirunelveli/kMC_code/python_kmc/bin/python}"
VALIDATION_OUT="${VALIDATION_OUT:-validation_dump}"
CACHE_SCHEMA="${CACHE_SCHEMA:-validation_refactor}"
STEPS="${STEPS:-1}"
LOCAL_ENV_MODE="${LOCAL_ENV_MODE:-radial}"

POTFILE="${POTFILE:-${ROOT_DIR}/kmc/PotentialB3410-modified.fs}"
LMP="${LMP:-/Users/rtirunelveli/mylammps/build2/lmp}"
MPIRUN="${MPIRUN:-mpirun}"
NP="${NP:-3}"
PARTITION="${PARTITION:-3x1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

RUN_KMC="${RUN_KMC:-1}"
GENERATE_LAMMPS="${GENERATE_LAMMPS:-1}"
RUN_LAMMPS="${RUN_LAMMPS:-0}"
EXTRACT_COMPARE="${EXTRACT_COMPARE:-1}"

VALIDATION_ASE_DUMP="${VALIDATION_ASE_DUMP:-1}"
VALIDATION_ASE_DUMP_PRE="${VALIDATION_ASE_DUMP_PRE:-0}"
VALIDATION_OVERWRITE="${VALIDATION_OVERWRITE:-1}"

NSTEPS1="${NSTEPS1:-10000}"
NSTEPS2="${NSTEPS2:-0}"
F1="${F1:-1.0e-6}"
KS="${KS:-5.0}"
DUMP="${DUMP:-50}"
SKIP_OCCUPIED="${SKIP_OCCUPIED:-1}"
MAX_NEI="${MAX_NEI:-0}"

COMPARE_OUT="${COMPARE_OUT:-${VALIDATION_OUT}/comparison.csv}"
ASE_BARRIERS="${ASE_BARRIERS:-${VALIDATION_OUT}/neb_validation_summary_rank0.csv}"
RATES="${RATES:-validation_rates_allranks.csv}"

echo "[validation] root=${ROOT_DIR}"
echo "[validation] python=${PYTHON_BIN}"
echo "[validation] out=${VALIDATION_OUT}"
echo "[validation] mode=${LOCAL_ENV_MODE}, steps=${STEPS}"

if [[ "${RUN_KMC}" == "1" ]]; then
  echo "[validation] running refactored KMC validation pass"
  VALIDATION_MODE=1 \
  VALIDATION_ASE_DUMP="${VALIDATION_ASE_DUMP}" \
  VALIDATION_ASE_DUMP_PRE="${VALIDATION_ASE_DUMP_PRE}" \
  VALIDATION_OVERWRITE="${VALIDATION_OVERWRITE}" \
  VALIDATION_OUT="${VALIDATION_OUT}" \
  CACHE_SCHEMA="${CACHE_SCHEMA}" \
  STEPS="${STEPS}" \
  LOCAL_ENV_MODE="${LOCAL_ENV_MODE}" \
  "${PYTHON_BIN}" -m kmc.main
else
  echo "[validation] RUN_KMC=0, skipping refactored KMC validation pass"
fi

if [[ "${GENERATE_LAMMPS}" == "1" ]]; then
  echo "[validation] generating native LAMMPS NEB jobs"
  POTFILE="${POTFILE}" \
  RUN=0 \
  NSTEPS1="${NSTEPS1}" \
  NSTEPS2="${NSTEPS2}" \
  F1="${F1}" \
  KS="${KS}" \
  DUMP="${DUMP}" \
  SKIP_OCCUPIED="${SKIP_OCCUPIED}" \
  MAX_NEI="${MAX_NEI}" \
  ./helper/lammps_neb.sh "${VALIDATION_OUT}" "${LMP}"
else
  echo "[validation] GENERATE_LAMMPS=0, skipping LAMMPS job generation"
fi

if [[ "${RUN_LAMMPS}" == "1" ]]; then
  echo "[validation] running native LAMMPS NEB jobs"
  ROOT="${VALIDATION_OUT}" \
  MPIRUN="${MPIRUN}" \
  LMP="${LMP}" \
  NP="${NP}" \
  PARTITION="${PARTITION}" \
  MAX_PARALLEL="${MAX_PARALLEL}" \
  SKIP_DONE="${SKIP_DONE:-1}" \
  FRESH_LOG="${FRESH_LOG:-0}" \
  ./helper/run_lammps_jobs.sh
else
  echo "[validation] RUN_LAMMPS=0, skipping native LAMMPS execution"
  echo "[validation] to run later: RUN_LAMMPS=1 ./validation.sh"
fi

if [[ "${EXTRACT_COMPARE}" == "1" ]]; then
  echo "[validation] extracting ASE/native-LAMMPS comparison"
  extractor_args=(
    --root "${VALIDATION_OUT}"
    --ase-barriers "${ASE_BARRIERS}"
    --out "${COMPARE_OUT}"
  )
  if [[ -f "${RATES}" ]]; then
    extractor_args+=(--rates "${RATES}")
  fi
  "${PYTHON_BIN}" ./helper/extractor.py "${extractor_args[@]}"
else
  echo "[validation] EXTRACT_COMPARE=0, skipping comparison extraction"
fi

echo "[validation] done"
echo "[validation] generated root: ${VALIDATION_OUT}"
echo "[validation] comparison: ${COMPARE_OUT}"
