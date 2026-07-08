#!/usr/bin/env bash
set -euo pipefail

# Memory-instrumented Devanathan restart launcher.
# This wraps run_devanathan_kmc.sh and records system/process memory state while it runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRACTICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRACTICE_ROOT"

STEPS="${STEPS:-10000}"
MPI_RANKS="${MPI_RANKS:-3}"
LAMMPS_NEB_REPLICAS="${LAMMPS_NEB_REPLICAS:-3}"
NIMG="${NIMG:-3}"

RUN_ROOT="${RUN_ROOT:-runs/devanathan_generated_90x20x20_memdebug_${STEPS}}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-5}"
MONITOR_DIR="${MONITOR_DIR:-$RUN_ROOT/memory_monitor}"
mkdir -p "$MONITOR_DIR"

SCHEMA="${CACHE_SCHEMA:-devanathan:v2_fe_fraction:lammps_only:generated:90x20x20:mode:shell_IR8.0_OR10.0:source10A_fe_frac0.01:sink0.8Lx}"
RECOVERED_CACHE="${RECOVERED_CACHE:-runs/devanathan_generated_90x20x20_10000/cache/barrier_cache_${SCHEMA}.pkl}"
RUN_CACHE_DIR="$RUN_ROOT/cache"
RUN_CACHE_FILE="$RUN_CACHE_DIR/barrier_cache_${SCHEMA}.pkl"
mkdir -p "$RUN_CACHE_DIR"
if [[ -f "$RECOVERED_CACHE" && ! -f "$RUN_CACHE_FILE" ]]; then
  cp "$RECOVERED_CACHE" "$RUN_CACHE_FILE"
fi

export RUN_ROOT
export STEPS
export MPI_RANKS
export LAMMPS_NEB_REPLICAS
export NIMG
export DEBUG_MODE="${DEBUG_MODE:-1}"
export LAMMPS_NEB_DEBUG_MODE="${LAMMPS_NEB_DEBUG_MODE:-0}"
export DUMP_EVERY_STEPS="${DUMP_EVERY_STEPS:-1}"
export RESTART_MODE="${RESTART_MODE:-0}"
export RESTART_DIR="${RESTART_DIR:-}"
export RESTART_STEP="${RESTART_STEP:-0}"
if [[ -z "${KMC_BARRIER_CACHE_FILE:-}" && -f "$RUN_CACHE_FILE" ]]; then
  export KMC_BARRIER_CACHE_FILE="$RUN_CACHE_FILE"
fi
export RESTART_CHECKPOINT_INTERVAL="${RESTART_CHECKPOINT_INTERVAL:-100}"

cat > "$MONITOR_DIR/launch_env.txt" <<EOF
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
practice_root=$PRACTICE_ROOT
run_root=$RUN_ROOT
steps=$STEPS
mpi_ranks=$MPI_RANKS
lammps_neb_replicas=$LAMMPS_NEB_REPLICAS
nimg=$NIMG
debug_mode=$DEBUG_MODE
lammps_neb_debug_mode=$LAMMPS_NEB_DEBUG_MODE
restart_mode=$RESTART_MODE
restart_dir=$RESTART_DIR
restart_step=$RESTART_STEP
kmc_barrier_cache_file=${KMC_BARRIER_CACHE_FILE:-}
monitor_interval_s=$MONITOR_INTERVAL_S
EOF

memory_monitor() {
  local interval="$1"
  local out_dir="$2"
  local root="$3"
  local process_csv="$out_dir/process_rss.csv"
  local vm_log="$out_dir/vm_stat.log"
  local pressure_log="$out_dir/memory_pressure.log"
  local top_log="$out_dir/top_memory_processes.log"
  local sample_ts

  printf "timestamp,pid,ppid,rss_kb,vsz_kb,cpu_pct,mem_pct,command,args\n" > "$process_csv"
  : > "$vm_log"
  : > "$pressure_log"
  : > "$top_log"

  while true; do
    sample_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    ps -axo pid=,ppid=,rss=,vsz=,%cpu=,%mem=,comm=,args= |
      awk -v ts="$sample_ts" -v root="$root" '
        {
          pid=$1; ppid=$2; rss=$3; vsz=$4; cpu=$5; mem=$6; comm=$7
          args=$0
          gsub(/^[[:space:]]+/, "", args)
          gsub(/"/, "\"\"", args)
          printf "%s,%s,%s,%s,%s,%s,%s,%s,\"%s\"\n", ts,pid,ppid,rss,vsz,cpu,mem,comm,args
        }
      ' >> "$process_csv"

    {
      echo "===== $sample_ts ====="
      vm_stat
    } >> "$vm_log" 2>&1
    {
      echo "===== $sample_ts ====="
      memory_pressure
    } >> "$pressure_log" 2>&1 || true
    {
      echo "===== $sample_ts ====="
      ps -axo pid,ppid,rss,vsz,%cpu,%mem,comm,args -r | head -25
    } >> "$top_log" 2>&1

    sleep "$interval"
  done
}

memory_monitor "$MONITOR_INTERVAL_S" "$MONITOR_DIR" "$PRACTICE_ROOT" &
MONITOR_PID="$!"

cleanup() {
  if kill -0 "$MONITOR_PID" >/dev/null 2>&1; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Running memory-monitored Devanathan KMC"
echo "  run_root=$RUN_ROOT"
echo "  monitor_dir=$MONITOR_DIR"
echo "  cache=${KMC_BARRIER_CACHE_FILE:-<schema-local or cold>}"
echo "  restart=${RESTART_DIR} step=${RESTART_STEP}"
echo "  mpi_ranks=$MPI_RANKS replicas=$LAMMPS_NEB_REPLICAS debug=$DEBUG_MODE"

bash scripts/run_devanathan_kmc.sh
