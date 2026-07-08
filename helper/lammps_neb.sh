#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./lammps_neb.sh /path/to/validation_dump [lammps_cmd]
#
# This scans validation folders like:
#   validation_dump/rankX_stepY_hZ/
# and creates per-neighbor jobs under:
#   validation_dump/rankX_stepY_hZ/lammps/nxx/
# using initial.data and neighbors.csv|neighbors.txt found in that folder.
#
# Env overrides:
#   LMP=lmp                    # or lmp_serial, etc.
#   POTFILE=PotentialB3410-modified.fs
#   RUN=0                      # set RUN=1 to actually run LAMMPS
#   COPY_INIT=1                # 1=copy init.data, 0=symlink
#   MAX_NEI=0                  # 0 = all neighbors; >0 = cap at that many
#   KS=5.0                     # NEB spring constant
#   NSTEPS1=10000              # single-stage NEB iterations (KMC-style)
#   NSTEPS2=0                  # climbing iterations; 0 keeps this single-stage
#   F1=1.0e-6                  # single-stage force tolerance (KMC-style)
#   DUMP=50                    # dump frequency
#   SKIP_OCCUPIED=1            # 1=skip targets already occupied by fixed neighbor H
#   NEB_REPLICAS=3             # number of NEB replicas/images used by -partition

ROOT="${1:-}"
if [[ -z "${ROOT}" || ! -d "${ROOT}" ]]; then
  echo "ERROR: Provide the validation_dump directory."
  echo "Example: ./lammps_neb.sh validation_dump"
  exit 1
fi

LMP="${2:-${LMP:-lmp}}"
POTFILE="${POTFILE:-PotentialB3410-modified.fs}"
RUN="${RUN:-0}"
COPY_INIT="${COPY_INIT:-1}"
MAX_NEI="${MAX_NEI:-0}"
KS="${KS:-5.0}"
NSTEPS1="${NSTEPS1:-10000}"
NSTEPS2="${NSTEPS2:-0}"
F1="${F1:-1.0e-6}"
DUMP="${DUMP:-50}"
SKIP_OCCUPIED="${SKIP_OCCUPIED:-1}"
NEB_REPLICAS="${NEB_REPLICAS:-3}"

echo "[Info] Validation ROOT=${ROOT}"
echo "[Info] LAMMPS cmd: ${LMP}"
echo "[Info] Potential file: ${POTFILE} (will be copied)"
echo "[Info] RUN=${RUN} (1=run, 0=generate only)"
echo "[Info] COPY_INIT=${COPY_INIT} (1=copy, 0=symlink)"
echo "[Info] MAX_NEI=${MAX_NEI} (0 = all)"
echo "[Info] Mode: SINGLE-STAGE NEB (KMC-style)"
echo "[Info] NSTEPS1=${NSTEPS1}, NSTEPS2=${NSTEPS2}"
echo "[Info] Lattice: Fe and mover H relax; spectator H is fixed (KMC-style)"
echo "[Info] SKIP_OCCUPIED=${SKIP_OCCUPIED} (1=skip occupied final H sites)"
echo "[Info] NEB_REPLICAS=${NEB_REPLICAS}"

shopt -s nullglob
folders=( "${ROOT}"/rank*_step*_h* )
if [[ ${#folders[@]} -eq 0 ]]; then
  echo "ERROR: No validation folders found in ${ROOT}"
  exit 1
fi

write_in_lmp() {
  local out="$1"
  local dumpfreq="$2"
  local kspring="$3"
  local f1="$4" n1="$5" n2="$6" mover_id="$7" replicas="$8"
  cat >"${out}" <<EOF_IN
units           metal
atom_style      atomic
boundary        p p p
dimension       3
atom_modify     map hash

read_data       init.data

pair_style      eam/fs
pair_coeff      * * __POTFILE__ Fe H

timestep        0.001
neighbor        2.0 bin
neigh_modify    delay 5

# --- Fix spectator H; Fe and the hopping H are free to relax ---
group           fe type 1
group           h  type 2
group           mover_h id ${mover_id}
group           fixed_h subtract h mover_h
fix             freeze_h fixed_h setforce 0 0 0

variable        u uloop ${replicas}
variable        dumpfreq equal ${dumpfreq}

thermo_style    custom step pe etotal fnorm
thermo          50

dump            d1 all atom \${dumpfreq} dump.all.\$u
dump_modify     d1 sort id

variable        kspring equal ${kspring}
fix             neb all neb \${kspring}

min_style       fire
min_modify      dmax 0.1

# --- Single-stage NEB (matches KMC stream): one pass only ---
# Match ASE-style force-only tolerance: etol=0.0, ftol=${f1}
neb 0.0 ${f1} ${n1} ${n2} 100  final neb_final.txt

undump d1
write_dump all custom neb.final.\$u id type xu yu zu
EOF_IN
}

patch_potfile() {
  local file="$1"
  local pot="$2"
  sed -i.bak "s#__POTFILE__#${pot}#g" "${file}"
  rm -f "${file}.bak"
}

find_mover_h_id() {
  local data="$1"
  awk '
    BEGIN{atoms=0}
    /^Atoms/ {atoms=1; next}
    atoms==1 && NF==0 {next}
    atoms==1 && $1 ~ /^[0-9]+$/ {
      id=$1; type=$2;
      if(type!=2){ next }
      if(index($0, "# neighbor") == 0){ print id; exit }
      if(first_h == ""){ first_h = id }
    }
    END{
      if(first_h != ""){ print first_h }
    }' "$data"
}

count_h_atoms() {
  local data="$1"
  awk '
    BEGIN{atoms=0}
    /^Atoms/ {atoms=1; next}
    atoms==1 && NF==0 {next}
    atoms==1 && $1 ~ /^[0-9]+$/ && $2 == 2 {count++}
    END{print count+0}' "$data"
}

count_tagged_neighbor_h() {
  local data="$1"
  awk '
    BEGIN{atoms=0}
    /^Atoms/ {atoms=1; next}
    atoms==1 && NF==0 {next}
    atoms==1 && $1 ~ /^[0-9]+$/ && $2 == 2 && index($0, "# neighbor") > 0 {count++}
    END{print count+0}' "$data"
}

is_target_occupied_by_fixed_h() {
  local data="$1"
  local target_x="$2"
  local target_y="$3"
  local target_z="$4"
  awk -v tx="${target_x}" -v ty="${target_y}" -v tz="${target_z}" '
    BEGIN{atoms=0}
    /^Atoms/ {atoms=1; next}
    atoms==1 && NF==0 {next}
    atoms==1 && $1 ~ /^[0-9]+$/ {
      if($2 != 2){ next }
      if(index($0, "# neighbor") == 0){ next }
      if(($3 - tx < 1e-6) && (tx - $3 < 1e-6) &&
         ($4 - ty < 1e-6) && (ty - $4 < 1e-6) &&
         ($5 - tz < 1e-6) && (tz - $5 < 1e-6)){
        print $1
        exit
      }
    }' "$data"
}

count_atoms() {
  local data="$1"
  awk '
    BEGIN{atoms=0}
    /^Atoms/ {atoms=1; next}
    atoms==1 && NF==0 {next}
    atoms==1 && $1 ~ /^[0-9]+$/ {count++}
    END{print count}' "$data"
}

emit_neighbor_xyz() {
  local conf_dir="$1"
  local csv="${conf_dir}/neighbors.csv"
  local txt="${conf_dir}/neighbors.txt"
  if [[ -f "${csv}" ]]; then
    tail -n +2 "${csv}" | awk -F, '{printf "%s %s %s %s\n",$1,$2,$3,$4}'
    return 0
  fi
  if [[ -f "${txt}" ]]; then
    awk 'NR==1{next}{printf "%d %s %s %s\n", NR-1, $1, $2, $3}' "${txt}"
    return 0
  fi
  return 1
}

for conf_dir in "${folders[@]}"; do
  init_data_path="${conf_dir}/initial.data"
  if [[ ! -s "${init_data_path}" ]]; then
    echo "[WARN] ${conf_dir}: missing or empty initial.data (VALIDATION_FORMAT?)."
    continue
  fi

  mover_lmp_id="$(find_mover_h_id "${init_data_path}" || true)"
  if [[ -z "${mover_lmp_id}" ]]; then
    echo "[WARN] ${conf_dir}: could not detect H (type==2) in initial.data; skipping."
    continue
  fi

  if ! coords="$(emit_neighbor_xyz "${conf_dir}")"; then
    echo "[WARN] ${conf_dir}: neighbors.csv|neighbors.txt not found; skipping."
    continue
  fi

  runs_root="${conf_dir}/lammps"
  mkdir -p "${runs_root}"
  echo "[Info] ${conf_dir} -> ${runs_root} (H id=${mover_lmp_id})"

  atom_count="$(count_atoms "${init_data_path}" || echo "?")"
  h_count="$(count_h_atoms "${init_data_path}" || echo "0")"
  tagged_neighbor_h_count="$(count_tagged_neighbor_h "${init_data_path}" || echo "0")"
  echo "[Info] ${conf_dir}: init.data contains ${atom_count} atoms"
  echo "[Info] ${conf_dir}: init.data contains ${h_count} H atoms (${tagged_neighbor_h_count} tagged neighbors)"

  idx=0
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    idx=$((idx+1))
    if [[ "${MAX_NEI}" -gt 0 && "${idx}" -gt "${MAX_NEI}" ]]; then
      break
    fi
    set -- ${line}
    nidx="$1"; x="$2"; y="$3"; z="$4"

    occupied_neighbor_id=""
    if [[ "${SKIP_OCCUPIED}" == "1" ]]; then
      occupied_neighbor_id="$(is_target_occupied_by_fixed_h "${init_data_path}" "${x}" "${y}" "${z}" || true)"
    fi
    if [[ -n "${occupied_neighbor_id}" ]]; then
      echo "[WARN] ${conf_dir}/${nidx}: target (${x}, ${y}, ${z}) is already occupied by fixed neighbor H atom id=${occupied_neighbor_id}; skipping."
      continue
    fi

    run_dir="${runs_root}/${nidx}"
    mkdir -p "${run_dir}"

    if [[ "${COPY_INIT}" == "1" ]]; then
      cp -f "${init_data_path}" "${run_dir}/init.data"
    else
      ln -sf "$(realpath "${init_data_path}")" "${run_dir}/init.data"
    fi

    {
      echo "#Final minimized atom --- One compulsory comment line"
      echo "1"
      printf "%s %s %s %s\n" "${mover_lmp_id}" "${x}" "${y}" "${z}"
    } >"${run_dir}/neb_final.txt"

    write_in_lmp "${run_dir}/in.lmp" "${DUMP}" "${KS}" "${F1}" "${NSTEPS1}" "${NSTEPS2}" "${mover_lmp_id}" "${NEB_REPLICAS}"
    patch_potfile "${run_dir}/in.lmp" "${POTFILE}"
    cp -f "${POTFILE}" "${run_dir}/"

    if [[ "${RUN}" == "1" ]]; then
      echo "[Run] ${LMP} -in in.lmp (${conf_dir}/${nidx})"
      ( cd "${run_dir}" && "${LMP}" -in in.lmp > log.neb 2>&1 )
    fi
  done <<<"${coords}"
done

if [[ "${RUN}" == "1" ]]; then
  echo "[Done] Executions submitted serially."
else
  echo "[Done] Generation complete. Set RUN=1 to execute."
fi
