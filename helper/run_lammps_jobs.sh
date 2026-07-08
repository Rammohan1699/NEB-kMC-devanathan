#!/usr/bin/env bash
set -euo pipefail

# -------------------- CONFIG (override via env) --------------------
ROOT="${ROOT:-validation_dump}"                         # validation root (rank*_step*_h*/ live here)
MPIRUN="${MPIRUN:-mpirun}"                              # or full path to mpirun
LMP="${LMP:-/Users/rtirunelveli/mylammps/build2/lmp}"    # your lmp binary
NP_ENV="${NP:-}"                                        # -np override; defaults to partition size
PARTITION="${PARTITION:-3x1}"                           # -partition
MAX_PARALLEL="${MAX_PARALLEL:-3}"                       # concurrent jobs
OVERSUBSCRIBE="${OVERSUBSCRIBE:-0}"                     # 1 → add -oversubscribe
DRY="${DRY:-0}"                                         # 1 → print only, don’t run
SKIP_DONE="${SKIP_DONE:-1}"                             # 1 → skip if log.run exists
FRESH_LOG="${FRESH_LOG:-0}"                              # 1 → remove old log.run before run
# ------------------------------------------------------------------

[[ -d "$ROOT" ]] || { echo "ERROR: ROOT '$ROOT' not found"; exit 1; }

partition_slots() {
  local spec="$1"
  local clean="${spec//X/x}"
  local total=1 part
  local IFS='x'
  local -a parts=()
  read -r -a parts <<< "$clean"
  for part in "${parts[@]}"; do
    if [[ -z "$part" || "$part" =~ [^0-9] || "$part" == "0" ]]; then
      echo "ERROR: Invalid PARTITION spec '$spec' (expected positive integers like 3x1)" >&2
      exit 1
    fi
    total=$(( total * part ))
  done
  echo "$total"
}

PARTITION_SLOTS="$(partition_slots "$PARTITION")"
if [[ -n "$NP_ENV" ]]; then
  NP="$NP_ENV"
else
  NP="$PARTITION_SLOTS"
fi

if [[ "$NP" -ne "$PARTITION_SLOTS" ]]; then
  echo "ERROR: NP=$NP but PARTITION=$PARTITION requires $PARTITION_SLOTS ranks. Adjust NP or PARTITION."
  exit 1
fi

# ---- move to ROOT so all paths are local/relative (your preferred behavior)
echo "[INFO] Changing directory to ROOT: $ROOT"
cd "$ROOT"

# ---------- Discover jobs (portable for macOS Bash 3.2) ----------
JOBS_FILE="$(mktemp -t lmp_jobs)"
# Find all in.lmp, list their dirs (relative), unique-sort
find . -type f -name 'in.lmp' -print0 \
| while IFS= read -r -d '' f; do
    d="$(dirname "$f")"
    # strip leading ./ for prettier logs
    d="${d#./}"
    echo "$d"
  done \
| sort -u > "$JOBS_FILE"

JOBS_COUNT=$(wc -l < "$JOBS_FILE" | tr -d ' ')
echo "[INFO] Found ${JOBS_COUNT} jobs under '$(pwd)'"

# ---------- Helpers ----------
scrub_env() {
  # Scrub inherited MPI env; OpenMPI/PMI vars can break nested mpirun
  for kv in $(env); do
    k="${kv%%=*}"
    case "$k" in
      OMPI_*|PMI_*|MPI_LOCALRANKID|MPI_LOCALNRANKS|MV2_*) unset "$k" ;;
    esac
  done
}

run_one() {
  local dir="$1"
  [[ -d "$dir" ]] || { echo "[WARN] Missing job dir: $dir"; return 1; }

  local abs_dir
  abs_dir="$(cd "$dir" && pwd)"

  local log="$abs_dir/log.run"
  local cmdfile="$abs_dir/cmd.sh"

  if [[ "$SKIP_DONE" == "1" && -s "$log" ]]; then
    echo "[SKIP] $dir (log.run exists)"
    return 0
  fi
  [[ "$FRESH_LOG" == "1" ]] && rm -f "$log"

  # per-job TMP to avoid OpenMPI session-dir collisions
  local tmp="$abs_dir/.mpi_tmp.$$"
  mkdir -p "$tmp"

  # Build command exactly like your manual run
  local cmd=( "$MPIRUN" -np "$NP" )
  if [[ "$OVERSUBSCRIBE" == "1" ]]; then
    cmd+=( -oversubscribe )
  fi
  cmd+=( "$LMP" -partition "$PARTITION" -in in.lmp )

  # Write cmd.sh for reproducibility
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export TMPDIR=$(printf %q "$tmp")"
    echo 'for v in $(env | cut -d= -f1); do case "$v" in OMPI_*|PMI_*|MPI_LOCALRANKID|MPI_LOCALNRANKS|MV2_*) unset "$v" ;; esac; done'
    printf "%q " "${cmd[@]}"
    echo
  } > "$cmdfile"
  chmod +x "$cmdfile"

  # create/truncate log, then run
  : > "$log"
  echo "\$ ${cmd[*]}" >> "$log"

  (
    cd "$abs_dir"
    scrub_env
    export TMPDIR="$tmp"
    # Use bash -lc to mimic interactive env (PATH/DYLD)
    /bin/bash -lc "${cmd[*]}" >>"$log" 2>&1
  ) && echo "[OK]   $dir" || { echo "[FAIL] $dir (see $log)"; return 1; }
}

# ---------- Parallel launcher compatible with old bash ----------
fail_count=0
running=0
pids=()
launched=0

start_job() {
  local d="$1"
  run_one "$d" </dev/null &
  pids+=( $! )
  running=$((running+1))
  launched=$((launched+1))
}

wait_some() {
  # Reap one job if too many running
  while (( running >= MAX_PARALLEL )); do
    # bash 3.2 doesn't have wait -n; emulate by waiting for the first pid
    local anypid="${pids[0]}"
    if wait "$anypid"; then
      :
    else
      fail_count=$((fail_count+1))
    fi
    pids=( "${pids[@]:1}" )
    running=$((running-1))
  done
}

# Start jobs with concurrency limit
while IFS= read -r d; do
  [[ -z "$d" ]] && continue
  wait_some
  start_job "$d"
done < "$JOBS_FILE"

# Wait for remaining
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    :
  else
    fail_count=$((fail_count+1))
  fi
done

rm -f "$JOBS_FILE"
echo "[DONE] Completed ${launched}/${JOBS_COUNT} job(s). Failures: $fail_count"
if (( launched != JOBS_COUNT )); then
  echo "[WARN] Launched $launched job(s) although $JOBS_COUNT were discovered. Re-run with FRESH_LOG=1 SKIP_DONE=0 and check for earlier errors."
fi
exit $fail_count
