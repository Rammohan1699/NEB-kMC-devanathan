#!/usr/bin/env bash
set -euo pipefail

# Submit two 10-seed Hyperion arrays:
#   1) all 50 H initially in bulk_grain_0
#   2) 50 H sampled from the union of bulk_grain_0 and bulk_grain_1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${KMC_PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PACKAGE_DIR"
mkdir -p slurm_logs

ARRAY_RANGE="${ARRAY_RANGE:-0-9}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-10}"
ONE_SEED_BASE="${ONE_SEED_BASE:-5001}"
BOTH_SEED_BASE="${BOTH_SEED_BASE:-6001}"
SUBMIT_WHICH="${SUBMIT_WHICH:-all}"
BATCH_SCRIPT="scripts/slurm_external_50h_fenwick_2m_array.sbatch"

submit_one() {
  sbatch \
    --array="${ARRAY_RANGE}%${ARRAY_CONCURRENCY}" \
    --job-name=ext1g_50h_fenwick \
    --output=slurm_logs/ext1g_50h_fenwick-%A_%a.out \
    --error=slurm_logs/ext1g_50h_fenwick-%A_%a.err \
    --export=ALL,KMC_PACKAGE_DIR="$PACKAGE_DIR",H_LAYOUT=one_grain,KMC_SEED_BASE="$ONE_SEED_BASE" \
    "$BATCH_SCRIPT"
}

submit_both() {
  sbatch \
    --array="${ARRAY_RANGE}%${ARRAY_CONCURRENCY}" \
    --job-name=ext2g_50h_fenwick \
    --output=slurm_logs/ext2g_50h_fenwick-%A_%a.out \
    --error=slurm_logs/ext2g_50h_fenwick-%A_%a.err \
    --export=ALL,KMC_PACKAGE_DIR="$PACKAGE_DIR",H_LAYOUT=both_grains,KMC_SEED_BASE="$BOTH_SEED_BASE" \
    "$BATCH_SCRIPT"
}

case "$SUBMIT_WHICH" in
  all)
    submit_one
    submit_both
    ;;
  one|one_grain|one-grain)
    submit_one
    ;;
  both|both_grains|both-grains)
    submit_both
    ;;
  *)
    echo "SUBMIT_WHICH must be all, one, or both; got '$SUBMIT_WHICH'" >&2
    exit 2
    ;;
esac
