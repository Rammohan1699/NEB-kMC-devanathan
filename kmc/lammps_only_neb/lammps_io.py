from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import LammpsOnlyNEBConfig
from .geometry import FE_TYPE, H_TYPE
from .models import LammpsSystem

FE_MASS = 55.845
H_MASS = 1.008


def write_lammps_data(path: Path, system: LammpsSystem, positions: np.ndarray | None = None) -> None:
    pos_arr = system.positions if positions is None else np.asarray(positions, dtype=float)
    if pos_arr.shape != system.positions.shape:
        raise ValueError(f"positions shape mismatch: got {pos_arr.shape}, expected {system.positions.shape}")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{system.mode} Fe-H native NEB local system\n\n")
        fh.write(f"{len(pos_arr)} atoms\n")
        fh.write("2 atom types\n\n")
        fh.write(f"0.0 {system.box[0]:.16g} xlo xhi\n")
        fh.write(f"0.0 {system.box[1]:.16g} ylo yhi\n")
        fh.write(f"0.0 {system.box[2]:.16g} zlo zhi\n\n")
        fh.write("Masses\n\n")
        fh.write(f"{FE_TYPE} {FE_MASS:.8f}\n")
        fh.write(f"{H_TYPE} {H_MASS:.8f}\n\n")
        fh.write("Atoms # atomic\n\n")
        for atom_id, (atype, pos) in enumerate(zip(system.types, pos_arr), start=1):
            fh.write(f"{atom_id} {int(atype)} {pos[0]:.16g} {pos[1]:.16g} {pos[2]:.16g}\n")


def write_final_coords(path: Path, system: LammpsSystem) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# number of atom-coordinate rows follows\n")
        fh.write(f"{len(system.final_positions)}\n")
        for atom_id in sorted(system.final_positions):
            pos = system.final_positions[atom_id]
            fh.write(f"{atom_id} {pos[0]:.16g} {pos[1]:.16g} {pos[2]:.16g}\n")


def write_final_coords_from_positions(path: Path, positions: np.ndarray) -> None:
    pos_arr = np.asarray(positions, dtype=float)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# number of atom-coordinate rows follows\n")
        fh.write(f"{len(pos_arr)}\n")
        for atom_id, pos in enumerate(pos_arr, start=1):
            fh.write(f"{atom_id} {pos[0]:.16g} {pos[1]:.16g} {pos[2]:.16g}\n")


def write_input_script(path: Path, system: LammpsSystem, cfg: LammpsOnlyNEBConfig) -> None:
    if system.frozen_ids:
        frozen_ids = " ".join(str(i) for i in system.frozen_ids)
        freeze = f"group frozen id {frozen_ids}\nfix freeze_shell frozen setforce 0.0 0.0 0.0"
    else:
        freeze = ""
    dump_commands = ""
    final_dump_commands = ""
    text = f"""units metal
atom_style atomic
boundary {system.boundary}
dimension 3
atom_modify map hash

read_data init.data

pair_style eam/fs
pair_coeff * * {cfg.potential_file.resolve()} Fe H

timestep {cfg.timestep:.16g}
neighbor 2.0 bin
neigh_modify delay 5
variable u uloop {cfg.replicas_per_neb}

group fe type {FE_TYPE}
group h type {H_TYPE}
group mover_h id {system.h_id}
{freeze}
fix neb_atoms all neb {cfg.spring_constant:.16g}
min_style {cfg.min_style}
min_modify dmax 0.1

thermo_style custom step pe etotal fnorm
thermo {cfg.thermo_every}
{dump_commands}

neb 0.0 {cfg.force_tolerance:.16g} {cfg.max_steps} {cfg.climbing_steps} {cfg.nevery} final coords.final

{final_dump_commands}
"""
    path.write_text(text, encoding="utf-8")


def write_endpoint_minimize_script(path: Path, system: LammpsSystem, cfg: LammpsOnlyNEBConfig) -> None:
    if system.frozen_ids:
        frozen_ids = " ".join(str(i) for i in system.frozen_ids)
        freeze = f"group frozen id {frozen_ids}\nfix freeze_shell frozen setforce 0.0 0.0 0.0"
    else:
        freeze = ""
    text = f"""units metal
atom_style atomic
boundary {system.boundary}
dimension 3
atom_modify map hash

read_data endpoint.data

pair_style eam/fs
pair_coeff * * {cfg.potential_file.resolve()} Fe H

timestep {cfg.timestep:.16g}
neighbor 2.0 bin
neigh_modify delay 5

group fe type {FE_TYPE}
group h type {H_TYPE}
group mover_h id {system.h_id}
{freeze}
min_style {cfg.endpoint_min_style}
min_modify dmax {cfg.endpoint_dmax:.16g}

thermo_style custom step pe etotal fnorm
thermo {cfg.thermo_every}

minimize {cfg.endpoint_energy_tolerance:.16g} {cfg.endpoint_force_tolerance:.16g} {cfg.endpoint_max_steps} {cfg.endpoint_max_evals}
"""
    path.write_text(text, encoding="utf-8")


def parse_last_barriers(log_path: Path) -> tuple[float | None, float | None]:
    ebf_idx = None
    ebr_idx = None
    ebf = None
    ebr = None
    have_header = False
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            fields = line.split()
            if not fields:
                continue
            upper = [field.upper() for field in fields]
            if "EBF" in upper or "EBR" in upper:
                ebf_idx = upper.index("EBF") if "EBF" in upper else None
                ebr_idx = upper.index("EBR") if "EBR" in upper else None
                have_header = True
                continue
            if not have_header:
                continue
            try:
                float(fields[0])
            except ValueError:
                continue
            if ebf_idx is not None and len(fields) > ebf_idx:
                ebf = float(fields[ebf_idx])
            if ebr_idx is not None and len(fields) > ebr_idx:
                ebr = float(fields[ebr_idx])
    return ebf, ebr


def choose_barrier(ebf: float | None, ebr: float | None) -> float:
    values = [value for value in (ebf, ebr) if value is not None and math.isfinite(value)]
    return max(values) if values else math.inf
