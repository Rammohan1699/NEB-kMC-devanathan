#!/usr/bin/env bash
set -euo pipefail

# Serial A/B benchmark from the 7M-step Devanathan restart:
#   1. previous full event rebuild/selection path
#   2. incremental impacted-area event table with Fenwick sampling
#
# Run from anywhere. Outputs stay under practice-version/devanathan-kmc-base/runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRACTICE_ROOT"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PRACTICE_ROOT" "$1" ;;
  esac
}

copy_practice_code_snapshot() {
  local destination="$1"
  mkdir -p "$destination"

  for dir in kmc scripts tools helper docs; do
    if [[ -d "$PRACTICE_ROOT/$dir" ]]; then
      cp -R "$PRACTICE_ROOT/$dir" "$destination/"
    fi
  done

  for file in README.md README_LAMMPS_ONLY.md DEVANATHAN_PRACTICE.md \
              KMC_REFRACTORED_CODE_USAGE.md WORKSPACE_LAYOUT.md \
              requirements.txt validation.sh .gitignore; do
    if [[ -f "$PRACTICE_ROOT/$file" ]]; then
      cp "$PRACTICE_ROOT/$file" "$destination/"
    fi
  done
}

build_combined_preload_cache() {
  local output_file="$1"
  local source_cache_dir="$2"
  local schema="$3"
  local size="$4"

  mkdir -p "$(dirname "$output_file")"
  OUTPUT_FILE="$output_file" SOURCE_CACHE_DIR="$source_cache_dir" CACHE_SCHEMA_VALUE="$schema" MPI_SIZE_VALUE="$size" \
    python3 - <<'PY'
import os
import pickle
from pathlib import Path

output_file = Path(os.environ["OUTPUT_FILE"])
source_cache_dir = Path(os.environ["SOURCE_CACHE_DIR"])
schema = os.environ["CACHE_SCHEMA_VALUE"]
mpi_size = int(os.environ["MPI_SIZE_VALUE"])

merged: dict = {}
master = source_cache_dir / f"barrier_cache_{schema}.pkl"
if master.exists():
    with master.open("rb") as fh:
        data = pickle.load(fh)
    merged.update(dict(data))

for rank in range(mpi_size):
    rank_cache = source_cache_dir / f"barrier_cache_rank{rank}_{schema}.pkl"
    if rank_cache.exists():
        with rank_cache.open("rb") as fh:
            data = pickle.load(fh)
        merged.update(dict(data))

    delta = Path(str(rank_cache) + ".delta.pkl")
    if delta.exists():
        with delta.open("rb") as fh:
            while True:
                try:
                    key, value = pickle.load(fh)
                except EOFError:
                    break
                merged[key] = value

output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("wb") as fh:
    pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote combined preload cache {output_file} with {len(merged)} entries")
PY
}

latest_checkpoint_step() {
  local checkpoint_dir="$1"
  find "$checkpoint_dir" -maxdepth 1 -type f \
    -name 'kmc_restart_checkpoint_step*.pkl' -print \
    | sed -E 's/.*_step([0-9]+)\.pkl/\1/' \
    | sort -n \
    | tail -1
}

run_case() {
  local label="$1"
  local incremental_flag="$2"
  local code_root="$3"
  local run_root="$4"

  if [[ -e "$run_root" ]]; then
    echo "Refusing to overwrite existing run directory: $run_root" >&2
    exit 1
  fi
  mkdir -p "$run_root"

  echo
  echo "=== Running $label ==="
  echo "  code_root=$code_root"
  echo "  run_root=$run_root"
  echo "  KMC_INCREMENTAL_EVENTS=$incremental_flag"

  (
    cd "$code_root"
    SOURCE_RUN="$ABS_SOURCE_RUN" \
    RUN_ROOT="$run_root" \
    STEPS="$TARGET_STEPS" \
    MPI_RANKS="$MPI_RANKS" \
    LAMMPS_NEB_REPLICAS="$LAMMPS_NEB_REPLICAS" \
    NIMG="$NIMG" \
    CACHE_SCHEMA="$CACHE_SCHEMA" \
    BARRIER_CACHE_FILE="$ABS_CACHE_FILE" \
    DEBUG_MODE=1 \
    DEBUG_LOGGING=1 \
    DUMP_EVERY_STEPS="$DUMP_EVERY_STEPS" \
    RESTART_CHECKPOINT_INTERVAL="$RESTART_CHECKPOINT_INTERVAL" \
    KMC_INCREMENTAL_EVENTS="$incremental_flag" \
    KMC_INCREMENTAL_IMPACT_RADIUS_A="$KMC_INCREMENTAL_IMPACT_RADIUS_A" \
    bash scripts/run_devanathan_restart.sh 2>&1 | tee "$run_root/run_stdout.log"
  )
}

SOURCE_RUN="${SOURCE_RUN:-runs/devanathan_generated_90x20x20_sink0p8_7000000_restart_from_6500000_serial}"
ABS_SOURCE_RUN="$(abs_path "$SOURCE_RUN")"
if [[ ! -d "$ABS_SOURCE_RUN/checkpoints" ]]; then
  echo "Source checkpoint directory not found: $ABS_SOURCE_RUN/checkpoints" >&2
  exit 1
fi

LATEST_SOURCE_STEP="$(latest_checkpoint_step "$ABS_SOURCE_RUN/checkpoints")"
SOURCE_STEP="${SOURCE_STEP:-$LATEST_SOURCE_STEP}"
if [[ -z "$SOURCE_STEP" ]]; then
  echo "No step-specific source checkpoint found in $ABS_SOURCE_RUN/checkpoints" >&2
  exit 1
fi
if [[ "$SOURCE_STEP" != "$LATEST_SOURCE_STEP" ]]; then
  echo "Requested SOURCE_STEP=$SOURCE_STEP, but run_devanathan_restart.sh will use latest checkpoint $LATEST_SOURCE_STEP." >&2
  echo "Use a SOURCE_RUN whose latest checkpoint is the requested start point." >&2
  exit 1
fi

SOURCE_CHECKPOINT="$ABS_SOURCE_RUN/checkpoints/kmc_restart_checkpoint_step${SOURCE_STEP}.pkl"
if [[ ! -s "$SOURCE_CHECKPOINT" ]]; then
  echo "Source checkpoint is missing or empty: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

TARGET_STEPS="${TARGET_STEPS:-$((SOURCE_STEP + 100000))}"
if (( TARGET_STEPS <= SOURCE_STEP )); then
  echo "TARGET_STEPS must be greater than SOURCE_STEP ($SOURCE_STEP): $TARGET_STEPS" >&2
  exit 1
fi

CACHE_SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx}"
CACHE_FILE="${CACHE_FILE:-$ABS_SOURCE_RUN/cache/barrier_cache_${CACHE_SCHEMA}.pkl}"
ABS_SOURCE_CACHE_FILE="$(abs_path "$CACHE_FILE")"
if [[ ! -s "$ABS_SOURCE_CACHE_FILE" ]]; then
  echo "Source barrier cache is missing or empty: $ABS_SOURCE_CACHE_FILE" >&2
  exit 1
fi

MPI_RANKS="${MPI_RANKS:-6}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
NIMG="${NIMG:-3}"
DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-1000}"
RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-1000}"
KMC_INCREMENTAL_IMPACT_RADIUS_A="${KMC_INCREMENTAL_IMPACT_RADIUS_A:-15.0}"

BENCH_ROOT="${BENCH_ROOT:-runs/devanathan_incremental_ab_100k_from_${SOURCE_STEP}}"
ABS_BENCH_ROOT="$(abs_path "$BENCH_ROOT")"
PREVIOUS_CODE_ROOT="$ABS_BENCH_ROOT/worktrees/previous_logic"
INCREMENTAL_CODE_ROOT="$ABS_BENCH_ROOT/worktrees/incremental_events"
PREVIOUS_RUN_ROOT="$ABS_BENCH_ROOT/previous_logic_100k_debug"
INCREMENTAL_RUN_ROOT="$ABS_BENCH_ROOT/incremental_events_100k_debug"

echo "Devanathan incremental A/B benchmark"
echo "  practice_root=$PRACTICE_ROOT"
echo "  source_run=$ABS_SOURCE_RUN"
echo "  source_step=$SOURCE_STEP"
echo "  target_steps=$TARGET_STEPS"
echo "  source_cache_file=$ABS_SOURCE_CACHE_FILE"
echo "  bench_root=$ABS_BENCH_ROOT"
echo "  mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS"
echo "  impact_radius=$KMC_INCREMENTAL_IMPACT_RADIUS_A A"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "DRY_RUN=1: validation passed; no benchmark directories were created."
  exit 0
fi

if [[ -e "$ABS_BENCH_ROOT" ]]; then
  echo "Refusing to overwrite existing benchmark directory: $ABS_BENCH_ROOT" >&2
  echo "Move it aside or set BENCH_ROOT to a new path." >&2
  exit 1
fi

mkdir -p "$ABS_BENCH_ROOT/worktrees"
ABS_CACHE_FILE="$ABS_BENCH_ROOT/preload_cache/barrier_cache_${CACHE_SCHEMA}.pkl"
build_combined_preload_cache "$ABS_CACHE_FILE" "$ABS_SOURCE_RUN/cache" "$CACHE_SCHEMA" "$MPI_RANKS"
copy_practice_code_snapshot "$PREVIOUS_CODE_ROOT"
copy_practice_code_snapshot "$INCREMENTAL_CODE_ROOT"

cat > "$ABS_BENCH_ROOT/benchmark_manifest.txt" <<EOF
source_run=$ABS_SOURCE_RUN
source_step=$SOURCE_STEP
target_steps=$TARGET_STEPS
source_cache_file=$ABS_SOURCE_CACHE_FILE
combined_preload_cache_file=$ABS_CACHE_FILE
mpi_ranks=$MPI_RANKS
lammps_neb_replicas=$LAMMPS_NEB_REPLICAS
nimg=$NIMG
dump_every_steps=$DUMP_EVERY_STEPS
restart_checkpoint_interval=$RESTART_CHECKPOINT_INTERVAL
incremental_impact_radius_a=$KMC_INCREMENTAL_IMPACT_RADIUS_A
previous_run_root=$PREVIOUS_RUN_ROOT
incremental_run_root=$INCREMENTAL_RUN_ROOT
EOF

run_case "previous_logic_100k_debug" "0" "$PREVIOUS_CODE_ROOT" "$PREVIOUS_RUN_ROOT"
run_case "incremental_events_100k_debug" "1" "$INCREMENTAL_CODE_ROOT" "$INCREMENTAL_RUN_ROOT"

echo
echo "A/B benchmark completed"
echo "  previous=$PREVIOUS_RUN_ROOT"
echo "  incremental=$INCREMENTAL_RUN_ROOT"
echo "  manifest=$ABS_BENCH_ROOT/benchmark_manifest.txt"
