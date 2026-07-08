"""
Structure-building utilities for KMC/NEB hydrogen diffusion.

This module is intentionally separated from MPI scheduling and the KMC loop.
It focuses only on converting global interstitial-site hops into ASE Atoms
objects usable by an NEB engine.

The functions here are conservative first-pass extractions from the working
monolithic implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
try:
    from ase import Atoms
    from ase.build import make_supercell
except ImportError:  # LAMMPS-only native generated runs can avoid ASE structures.
    Atoms = None  # type: ignore
    make_supercell = None  # type: ignore

try:
    from .lattice import pbc_diff
except ImportError:  # allow script-style testing
    from lattice import pbc_diff


TYPE_MAP = {"Fe": 1, "H": 2}
MASS_MAP = {"Fe": 55.845, "H": 1.008}


def _require_ase() -> None:
    if Atoms is None:
        raise RuntimeError("ASE is required for this structure-building path; use NEB_ENGINE=lammps_only for native generated runs or install ASE.")


@dataclass(frozen=True)
class SupercellSpec:
    """Basic orthogonal BCC Fe supercell description."""
    lattice_a: float = 2.8601
    nx: int = 30
    ny: int = 30
    nz: int = 30
    box_override: Optional[Tuple[float, float, float]] = None

    @property
    def box(self) -> np.ndarray:
        if self.box_override is not None:
            return np.asarray(self.box_override, dtype=float)
        return np.array(
            [
                self.lattice_a * self.nx,
                self.lattice_a * self.ny,
                self.lattice_a * self.nz,
            ],
            dtype=float,
        )


def wrap_positions_to_box(positions: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Return positions wrapped into the [0, L) interval expected by PBC KD-trees."""
    box_arr = np.asarray(box, dtype=float)
    pos = np.asarray(positions, dtype=float)
    if pos.size == 0:
        return pos.reshape((-1, 3))
    return np.mod(pos, box_arr)


def load_kmc_site_map(path: str | Path) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load a KMC site map produced by the map-generator pipeline.

    Supported formats are the stage-3 NPZ file and the CSV export with x/y/z
    columns. The NPZ path is preferred because it carries the periodic box.
    """
    p = Path(path).expanduser()
    if p.suffix.lower() == ".npz":
        data = np.load(p, allow_pickle=True)
        positions = np.asarray(data["positions"], dtype=float)
        site_types = np.asarray(data["site_types"], dtype=int) if "site_types" in data.files else None
        box = np.asarray(data["box_lengths"], dtype=float) if "box_lengths" in data.files else None
        origin = np.asarray(data["box_origin"], dtype=float) if "box_origin" in data.files else np.zeros(3)
        positions = positions - origin
        if box is not None:
            positions = wrap_positions_to_box(positions, box)
        return positions, site_types, box

    if p.suffix.lower() == ".csv":
        import csv

        positions: list[list[float]] = []
        site_types: list[int] = []
        with p.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                positions.append([float(row["x"]), float(row["y"]), float(row["z"])])
                if row.get("site_type") not in (None, ""):
                    site_types.append(int(float(row["site_type"])))
        type_arr = np.asarray(site_types, dtype=int) if len(site_types) == len(positions) else None
        return np.asarray(positions, dtype=float), type_arr, None

    raise ValueError(f"Unsupported KMC site-map format: {p}")


def load_lammps_atomic_structure(
    path: str | Path,
    *,
    fe_type: int = 1,
) -> tuple[Atoms, np.ndarray, np.ndarray]:
    """
    Load an orthogonal LAMMPS atomic data file used as the host structure.

    The current map-generator output writes `Atoms # atomic` rows as:
    atom_id atom_type x y z. Positions are normalized to a zero-origin box.
    """
    p = Path(path).expanduser()
    origin = np.zeros(3, dtype=float)
    hi = np.zeros(3, dtype=float)
    rows: list[tuple[int, np.ndarray]] = []
    in_atoms = False
    saw_atoms_header = False

    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                if saw_atoms_header:
                    in_atoms = True
                    saw_atoms_header = False
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
                origin[0] = float(parts[0])
                hi[0] = float(parts[1])
                continue
            if len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
                origin[1] = float(parts[0])
                hi[1] = float(parts[1])
                continue
            if len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
                origin[2] = float(parts[0])
                hi[2] = float(parts[1])
                continue
            if parts[0].lower() == "atoms":
                saw_atoms_header = True
                in_atoms = False
                continue
            if in_atoms:
                if len(parts) < 5:
                    continue
                try:
                    atype = int(parts[1])
                    pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
                except ValueError:
                    continue
                rows.append((atype, pos - origin))

    if not rows:
        raise ValueError(f"No Atoms # atomic rows found in {p}")
    box = hi - origin
    if np.any(box <= 0.0):
        raise ValueError(f"Invalid or missing box bounds in {p}: {box}")

    types = np.asarray([atype for atype, _pos in rows], dtype=int)
    positions = wrap_positions_to_box(np.asarray([pos for _atype, pos in rows], dtype=float), box)
    symbols = ["Fe" if int(atype) == int(fe_type) else "H" for atype in types]
    atoms = Atoms(symbols=symbols, positions=positions, cell=box, pbc=True)
    return assign_mass_and_type(atoms), types, box


def build_full_structure_from_host(
    host_atoms: Atoms,
    sites: np.ndarray,
    h_site_indices: Sequence[int],
) -> Atoms:
    """Build full host + occupied-H structure from an externally loaded host."""
    valid_indices = [int(i) for i in h_site_indices if int(i) < len(sites)]
    h_positions = np.asarray(sites[valid_indices], dtype=float)
    h_atoms = Atoms("H" * len(h_positions), positions=h_positions, cell=host_atoms.cell, pbc=True)
    return assign_mass_and_type(host_atoms.copy() + h_atoms)


@dataclass(frozen=True)
class LocalEnvironmentSpec:
    """
    Controls how local NEB structures are built.

    mode:
      - "wrapped": 3a template plus H occupancy from the wrapped 3x3x3 window.
      - "radial": 3a template plus only supplied radial neighbor H atoms.
      - "shell": spherical local environment with frozen outer shell.
    """
    mode: str = "wrapped"
    shell_inner_radius_a: float = 5.0
    shell_outer_radius_a: float = 6.0
    shrinkwrap_pad_a: float = 2.0


def assign_mass_and_type(atoms: Atoms) -> Atoms:
    """Attach LAMMPS-friendly mass/type/id arrays to an ASE Atoms object."""
    symbols = atoms.get_chemical_symbols()
    types = np.array([TYPE_MAP[s] for s in symbols], dtype=int)
    masses = np.array([MASS_MAP[s] for s in symbols], dtype=float)
    ids = np.arange(1, len(symbols) + 1, dtype=int)

    atoms.set_atomic_numbers([26 if s == "Fe" else 1 for s in symbols])
    atoms.set_array("type", types)
    atoms.set_array("mass", masses)
    atoms.set_array("id", ids)
    return atoms


def build_orthogonal_bcc_fe_supercell(spec: SupercellSpec) -> Atoms:
    """Build an orthogonal BCC Fe supercell with ASE."""
    a = spec.lattice_a
    basis = Atoms(
        "Fe2",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[a, a, a],
        pbc=True,
    )
    supercell = make_supercell(basis, np.diag([spec.nx, spec.ny, spec.nz]))
    supercell.set_cell(spec.box)
    supercell.set_pbc(True)
    return assign_mass_and_type(supercell)


def build_full_structure(
    sites: np.ndarray,
    h_site_indices: Sequence[int],
    spec: SupercellSpec,
) -> Atoms:
    """Build full Fe + H structure from occupied interstitial site indices."""
    valid_indices = [int(i) for i in h_site_indices if int(i) < len(sites)]
    fe_atoms = build_orthogonal_bcc_fe_supercell(spec)
    h_positions = np.asarray(sites[valid_indices], dtype=float)
    h_atoms = Atoms("H" * len(h_positions), positions=h_positions, cell=fe_atoms.cell, pbc=True)
    return assign_mass_and_type(fe_atoms + h_atoms)


def update_full_structure_h_atom(
    full_structure: Atoms,
    sites: np.ndarray,
    h_slot: int,
    new_site_idx: int,
) -> None:
    """
    Update a moved H atom in-place.

    Assumes H atoms are appended after Fe atoms in the same slot order as
    h_site_indices, matching the current working implementation.
    """
    h_count = int(sum(1 for s in full_structure.get_chemical_symbols() if s == "H"))
    h_offset = len(full_structure) - h_count
    atom_idx = h_offset + int(h_slot)
    if atom_idx < 0 or atom_idx >= len(full_structure):
        raise IndexError(
            f"H slot {h_slot} maps to invalid atom index {atom_idx}; "
            f"len={len(full_structure)}, h_offset={h_offset}"
        )
    full_structure[atom_idx].position = np.asarray(sites[int(new_site_idx)], dtype=float)


def build_fe_3a_template(lattice_a: float) -> Atoms:
    """Build a periodic 3a x 3a x 3a BCC Fe template containing 54 Fe atoms."""
    a = lattice_a
    cell = np.array([3 * a, 3 * a, 3 * a], dtype=float)
    positions = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base = np.array([i, j, k], dtype=float) * a
                positions.append(base)
                positions.append(base + 0.5 * a * np.array([1.0, 1.0, 1.0], dtype=float))
    atoms = Atoms("Fe" * len(positions), positions=np.asarray(positions), cell=cell, pbc=True)
    atoms.wrap()
    return assign_mass_and_type(atoms)

def build_3a_corner_freeze_mask(atoms: Atoms, lattice_a: float) -> np.ndarray:
    """
    Freeze one corner Fe atom in the canonical 3a template.

    This mirrors the existing anchor-style local template behavior.
    """
    del lattice_a
    positions = np.asarray(atoms.get_positions(), dtype=float)
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    anchor = np.array([0.0, 0.0, 0.0], dtype=float)
    is_anchor = np.all(np.isclose(positions, anchor[None, :], atol=1e-8), axis=1)
    return np.logical_and(atomic_numbers == 26, is_anchor)


def set_nonperiodic_local_box(
    initial: Atoms,
    final: Atoms,
    padding: float = 2.0,
) -> None:
    """
    Shift two local endpoint structures into a non-periodic box that contains both.
    """
    combined = np.vstack((initial.get_positions(), final.get_positions()))
    mins = combined.min(axis=0) - float(padding)
    maxs = combined.max(axis=0) + float(padding)
    cell = np.maximum(maxs - mins, 1.0)

    for atoms in (initial, final):
        atoms.positions[:] = atoms.get_positions() - mins
        atoms.set_cell(cell)
        atoms.set_pbc(False)




def set_periodic_local_box(initial: Atoms, final: Atoms, lattice_a: float) -> None:
    """Keep local endpoints in the canonical periodic 3a bulk template box."""
    cell = np.array([3.0 * lattice_a, 3.0 * lattice_a, 3.0 * lattice_a], dtype=float)
    for atoms in (initial, final):
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        atoms.wrap()


def _wrap_cell_delta(c: int, c0: int, n: int) -> int:
    d = int(c) - int(c0)
    d -= int(n * np.round(d / n))
    return int(d)


def _neighbor_entry_to_pos(entry: Any) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Normalize neighbor entry formats:
      - ndarray/list position
      - (position, index)
      - {"pos": position, "index": index}
    """
    if entry is None:
        return None, None
    idx = None
    pos = entry
    if isinstance(entry, dict):
        pos = entry.get("pos")
        idx = entry.get("index")
    elif isinstance(entry, (list, tuple)):
        if len(entry) == 0:
            return None, None
        pos = entry[0]
        if len(entry) > 1:
            idx = entry[1]
    arr = np.asarray(pos, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None, None
    return arr, None if idx is None else int(idx)


def build_local_neb_structures_fast(
    rel_initial: np.ndarray,
    rel_final: np.ndarray,
    spec: SupercellSpec,
    fe_3a_template: Atoms,
    *,
    mode: str = "wrapped",
    neighbor_positions: Optional[Iterable[Any]] = None,
    h_cell_index: Optional[Dict[tuple[int, int, int], List[tuple[np.ndarray, Optional[int]]]]] = None,
    shrinkwrap_pad_a: float = 2.0,
) -> Tuple[Atoms, Atoms, int]:
    """
    Build local NEB endpoints using a reusable 3a Fe template.

    This function supports both:
      - mode="wrapped": add occupied H atoms from h_cell_index in the 3x3x3 window.
      - mode="radial": add only supplied neighbor_positions.
    """
    if mode not in {"wrapped", "radial"}:
        raise ValueError(f"Unsupported fast local mode: {mode}")

    a = spec.lattice_a
    rel_initial = np.asarray(rel_initial, dtype=float)
    rel_final = np.asarray(rel_final, dtype=float)

    i, j, k = np.floor(rel_initial / a).astype(int)
    r0 = rel_initial - np.array([i, j, k], dtype=float) * a
    offset = np.array([1.0, 1.0, 1.0], dtype=float) * a

    d_global = pbc_diff(rel_initial, rel_final, spec.box)

    initial = fe_3a_template.copy()
    final = fe_3a_template.copy()

    pos0 = offset + r0
    pos1 = pos0 + d_global

    initial += Atoms("H", positions=[pos0], cell=initial.cell, pbc=True)
    final += Atoms("H", positions=[pos1], cell=final.cell, pbc=True)
    mover_idx = len(initial) - 1

    neighbor_entries: List[tuple[np.ndarray, Optional[int]]] = []
    seen = set()

    def register_neighbor(pos_abs: np.ndarray, neighbor_idx: Optional[int]) -> None:
        pos_abs = np.asarray(pos_abs, dtype=float)
        if pos_abs.shape != (3,) or not np.isfinite(pos_abs).all():
            return
        if np.allclose(pbc_diff(rel_initial, pos_abs, spec.box), 0.0, atol=1e-6):
            return
        key = tuple(np.round(pos_abs, 8))
        if key in seen:
            return
        seen.add(key)
        neighbor_entries.append((pos_abs.copy(), neighbor_idx))

    if mode == "wrapped":
        occupied_h_by_cell = h_cell_index or {}
        for sx in (-1, 0, 1):
            cx = (int(i) + sx) % spec.nx
            for sy in (-1, 0, 1):
                cy = (int(j) + sy) % spec.ny
                for sz in (-1, 0, 1):
                    cz = (int(k) + sz) % spec.nz
                    for pos_abs, neighbor_idx in occupied_h_by_cell.get((cx, cy, cz), ()):
                        register_neighbor(pos_abs, neighbor_idx)
    else:
        for entry in neighbor_positions or []:
            pos_abs, neighbor_idx = _neighbor_entry_to_pos(entry)
            if pos_abs is not None:
                register_neighbor(pos_abs, neighbor_idx)

    neighbor_meta = []
    for pos_abs, neighbor_idx in neighbor_entries:
        cx, cy, cz = np.floor(pos_abs / a + 1e-12).astype(int)
        dx = _wrap_cell_delta(cx, i, spec.nx)
        dy = _wrap_cell_delta(cy, j, spec.ny)
        dz = _wrap_cell_delta(cz, k, spec.nz)

        if (abs(dx) > 1) or (abs(dy) > 1) or (abs(dz) > 1):
            continue

        r = pos_abs - np.array([cx * a, cy * a, cz * a], dtype=float)
        r_local = np.array([dx + 1, dy + 1, dz + 1], dtype=float) * a + r

        initial += Atoms("H", positions=[r_local], cell=initial.cell, pbc=True)
        init_idx = len(initial) - 1
        final += Atoms("H", positions=[r_local], cell=final.cell, pbc=True)
        final_idx = len(final) - 1
        neighbor_meta.append((init_idx, final_idx, neighbor_idx))

    freeze_mask = build_3a_corner_freeze_mask(initial, a)
    if 0 <= mover_idx < len(freeze_mask):
        freeze_mask[mover_idx] = False

    if mode == "radial":
        set_periodic_local_box(initial, final, a)
    else:
        set_nonperiodic_local_box(initial, final, padding=shrinkwrap_pad_a)

    assign_mass_and_type(initial)
    assign_mass_and_type(final)

    initial.set_array("freeze_mask", np.asarray(freeze_mask, dtype=bool))
    final.set_array("freeze_mask", np.asarray(freeze_mask, dtype=bool))

    neighbor_idx_init = np.full(len(initial), -1, dtype=int)
    neighbor_idx_final = np.full(len(final), -1, dtype=int)
    for init_idx, final_idx, neighbor_idx in neighbor_meta:
        if neighbor_idx is None:
            continue
        neighbor_idx_init[init_idx] = int(neighbor_idx)
        neighbor_idx_final[final_idx] = int(neighbor_idx)

    initial.set_array("neighbor_index", neighbor_idx_init)
    final.set_array("neighbor_index", neighbor_idx_final)

    n_fe = sum(1 for s in initial.get_chemical_symbols() if s == "Fe")
    if n_fe != 54:
        raise RuntimeError(f"Fast local template Fe count is {n_fe}; expected 54.")

    return initial, final, mover_idx


def build_local_neb_structures_shell(
    rel_initial: np.ndarray,
    rel_final: np.ndarray,
    spec: SupercellSpec,
    *,
    h_cell_index: Optional[Dict[tuple[int, int, int], List[tuple[np.ndarray, Optional[int]]]]] = None,
    inner_radius_a: float = 5.0,
    outer_radius_a: float = 6.0,
    shrinkwrap_pad_a: float = 2.0,
) -> Tuple[Atoms, Atoms, int]:
    """
    Build a spherical shell local environment.

    Atoms inside inner_radius_a are movable. Atoms between inner_radius_a and
    outer_radius_a are included and frozen. The moving H remains unconstrained.
    """
    a = spec.lattice_a
    rel_initial = np.asarray(rel_initial, dtype=float)
    rel_final = np.asarray(rel_final, dtype=float)
    inner = max(0.0, float(inner_radius_a))
    outer = max(inner, float(outer_radius_a))

    i, j, k = np.floor(rel_initial / a + 1e-12).astype(int)
    cell_pad = max(1, int(np.ceil(outer / a)) + 1)
    d_global = pbc_diff(rel_initial, rel_final, spec.box)

    local_pad = max(a, float(np.linalg.norm(d_global))) + float(shrinkwrap_pad_a)
    local_half = outer + local_pad
    local_box = np.array([2.0 * local_half, 2.0 * local_half, 2.0 * local_half], dtype=float)
    local_center = np.array([local_half, local_half, local_half], dtype=float)

    loc_pos: list[np.ndarray] = []
    loc_sym: list[str] = []
    freeze_mask: list[bool] = []
    dist_from_center: list[float] = []
    mover_idx: Optional[int] = None
    mover_dist = float("inf")
    seen_fe: set[tuple[int, int, int, int]] = set()
    seen_h: set[Any] = set()

    def append_atom(
        symbol: str,
        pos_local: np.ndarray,
        dist: float,
        *,
        is_mover: bool = False,
        dedupe_key: Any = None,
    ) -> None:
        nonlocal mover_idx, mover_dist
        if dedupe_key is not None:
            if symbol == "Fe":
                if dedupe_key in seen_fe:
                    return
                seen_fe.add(dedupe_key)
            else:
                if dedupe_key in seen_h:
                    return
                seen_h.add(dedupe_key)

        loc_pos.append(np.asarray(pos_local, dtype=float))
        loc_sym.append(symbol)
        dist_from_center.append(float(dist))
        freeze_mask.append((dist > inner) and (dist <= outer))
        idx = len(loc_pos) - 1
        if is_mover and dist < mover_dist:
            mover_dist = float(dist)
            mover_idx = idx

    for sx in range(-cell_pad, cell_pad + 1):
        cx = (int(i) + sx) % spec.nx
        x0 = cx * a
        for sy in range(-cell_pad, cell_pad + 1):
            cy = (int(j) + sy) % spec.ny
            y0 = cy * a
            for sz in range(-cell_pad, cell_pad + 1):
                cz = (int(k) + sz) % spec.nz
                z0 = cz * a
                base = np.array([x0, y0, z0], dtype=float)
                for basis_idx, basis in enumerate(
                    (
                        np.array([0.0, 0.0, 0.0], dtype=float),
                        0.5 * a * np.array([1.0, 1.0, 1.0], dtype=float),
                    )
                ):
                    pos_abs = base + basis
                    diff = pbc_diff(rel_initial, pos_abs, spec.box)
                    dist = float(np.linalg.norm(diff))
                    if dist > outer + 1e-8:
                        continue
                    append_atom(
                        "Fe",
                        local_center + diff,
                        dist,
                        dedupe_key=(cx, cy, cz, basis_idx),
                    )

    occupied_h_by_cell = h_cell_index or {}
    for sx in range(-cell_pad, cell_pad + 1):
        cx = (int(i) + sx) % spec.nx
        for sy in range(-cell_pad, cell_pad + 1):
            cy = (int(j) + sy) % spec.ny
            for sz in range(-cell_pad, cell_pad + 1):
                cz = (int(k) + sz) % spec.nz
                for pos_abs, neighbor_idx in occupied_h_by_cell.get((cx, cy, cz), ()):
                    pos_abs = np.asarray(pos_abs, dtype=float)
                    if pos_abs.shape != (3,) or not np.isfinite(pos_abs).all():
                        continue
                    diff = pbc_diff(rel_initial, pos_abs, spec.box)
                    dist = float(np.linalg.norm(diff))
                    if dist > outer + 1e-8:
                        continue
                    if neighbor_idx is not None:
                        dedupe_key = ("idx", int(neighbor_idx))
                    else:
                        dedupe_key = ("pos", tuple(np.round(pos_abs, 8)))
                    append_atom(
                        "H",
                        local_center + diff,
                        dist,
                        is_mover=(dist <= 0.25),
                        dedupe_key=dedupe_key,
                    )

    if not loc_pos:
        raise RuntimeError("Shell local environment kept 0 atoms")
    if mover_idx is None or mover_dist > 0.25:
        raise RuntimeError(f"Shell local environment moving-H match failed (distance={mover_dist:.3f} A)")

    initial = Atoms(symbols=loc_sym, positions=np.asarray(loc_pos, dtype=float), cell=local_box, pbc=False)
    final = initial.copy()
    final[mover_idx].position = initial[mover_idx].position + d_global

    set_nonperiodic_local_box(initial, final, padding=shrinkwrap_pad_a)
    assign_mass_and_type(initial)
    assign_mass_and_type(final)

    freeze_mask_arr = np.asarray(freeze_mask, dtype=bool)
    freeze_mask_arr[mover_idx] = False
    initial.set_array("freeze_mask", freeze_mask_arr.copy())
    final.set_array("freeze_mask", freeze_mask_arr.copy())
    initial.set_array("shell_distance", np.asarray(dist_from_center, dtype=float))
    final.set_array("shell_distance", np.asarray(dist_from_center, dtype=float))

    return initial, final, int(mover_idx)


def build_local_neb_structures_shell_from_host(
    rel_initial: np.ndarray,
    rel_final: np.ndarray,
    host_fe_positions: np.ndarray,
    box: Sequence[float],
    *,
    host_fe_tree: Any = None,
    h_positions: Optional[np.ndarray] = None,
    h_indices: Optional[np.ndarray] = None,
    inner_radius_a: float = 5.0,
    outer_radius_a: float = 6.0,
    shrinkwrap_pad_a: float = 2.0,
) -> Tuple[Atoms, Atoms, int]:
    """
    Build a shell local environment from an explicit host structure.

    This is the external-map counterpart of build_local_neb_structures_shell().
    It uses the bicrystal/GB Fe coordinates directly instead of regenerating an
    ideal BCC stencil around the hopping H.
    """
    rel_initial = np.asarray(rel_initial, dtype=float)
    rel_final = np.asarray(rel_final, dtype=float)
    host_fe_positions = np.asarray(host_fe_positions, dtype=float)
    box_arr = np.asarray(box, dtype=float)
    inner = max(0.0, float(inner_radius_a))
    outer = max(inner, float(outer_radius_a))
    d_global = pbc_diff(rel_initial, rel_final, box_arr)

    local_pad = max(1.0, float(np.linalg.norm(d_global))) + float(shrinkwrap_pad_a)
    local_half = outer + local_pad
    local_box = np.array([2.0 * local_half, 2.0 * local_half, 2.0 * local_half], dtype=float)
    local_center = np.array([local_half, local_half, local_half], dtype=float)

    loc_pos: list[np.ndarray] = []
    loc_sym: list[str] = []
    freeze_mask: list[bool] = []
    dist_from_center: list[float] = []
    neighbor_index: list[int] = []
    mover_idx: Optional[int] = None
    mover_dist = float("inf")

    def append_atom(
        symbol: str,
        pos_local: np.ndarray,
        dist: float,
        *,
        neighbor_idx: int = -1,
        is_mover: bool = False,
    ) -> None:
        nonlocal mover_idx, mover_dist
        loc_pos.append(np.asarray(pos_local, dtype=float))
        loc_sym.append(symbol)
        dist_from_center.append(float(dist))
        freeze_mask.append((dist > inner) and (dist <= outer))
        neighbor_index.append(int(neighbor_idx))
        idx = len(loc_pos) - 1
        if is_mover and dist < mover_dist:
            mover_dist = float(dist)
            mover_idx = idx

    if host_fe_tree is None:
        host_indices: Iterable[int] = range(len(host_fe_positions))
    else:
        center_wrapped = np.mod(rel_initial, box_arr)
        host_indices = sorted(
            int(index)
            for index in host_fe_tree.query_ball_point(
                center_wrapped,
                outer + 1.0e-8,
            )
        )

    for host_index in host_indices:
        pos_abs = host_fe_positions[int(host_index)]
        diff = pbc_diff(rel_initial, pos_abs, box_arr)
        dist = float(np.linalg.norm(diff))
        if dist <= outer + 1e-8:
            append_atom("Fe", local_center + diff, dist)

    if h_positions is not None and len(h_positions):
        h_positions = np.asarray(h_positions, dtype=float)
        idx_array = None if h_indices is None else np.asarray(h_indices, dtype=int)
        for occ_idx, pos_abs in enumerate(h_positions):
            diff = pbc_diff(rel_initial, pos_abs, box_arr)
            dist = float(np.linalg.norm(diff))
            if dist > outer + 1e-8:
                continue
            site_idx = int(idx_array[occ_idx]) if idx_array is not None and occ_idx < len(idx_array) else -1
            append_atom(
                "H",
                local_center + diff,
                dist,
                neighbor_idx=site_idx,
                is_mover=(dist <= 0.25),
            )

    if not loc_pos:
        raise RuntimeError("External shell local environment kept 0 atoms")
    if mover_idx is None or mover_dist > 0.25:
        raise RuntimeError(f"External shell moving-H match failed (distance={mover_dist:.3f} A)")

    initial = Atoms(symbols=loc_sym, positions=np.asarray(loc_pos, dtype=float), cell=local_box, pbc=False)
    final = initial.copy()
    final[mover_idx].position = initial[mover_idx].position + d_global

    set_nonperiodic_local_box(initial, final, padding=shrinkwrap_pad_a)
    assign_mass_and_type(initial)
    assign_mass_and_type(final)

    freeze_mask_arr = np.asarray(freeze_mask, dtype=bool)
    freeze_mask_arr[mover_idx] = False
    initial.set_array("freeze_mask", freeze_mask_arr.copy())
    final.set_array("freeze_mask", freeze_mask_arr.copy())
    initial.set_array("shell_distance", np.asarray(dist_from_center, dtype=float))
    final.set_array("shell_distance", np.asarray(dist_from_center, dtype=float))
    initial.set_array("neighbor_index", np.asarray(neighbor_index, dtype=int))
    final.set_array("neighbor_index", np.asarray(neighbor_index, dtype=int))

    return initial, final, int(mover_idx)


def build_h_cell_index(
    h_positions: np.ndarray,
    h_indices: Optional[np.ndarray],
    spec: SupercellSpec,
) -> Dict[tuple[int, int, int], List[tuple[np.ndarray, Optional[int]]]]:
    """Build cell-indexed occupied H lookup for wrapped 3a local environments."""
    cell_index: Dict[tuple[int, int, int], List[tuple[np.ndarray, Optional[int]]]] = {}
    if h_positions is None or len(h_positions) == 0:
        return cell_index

    h_positions = np.asarray(h_positions, dtype=float)
    idx_array = None if h_indices is None else np.asarray(h_indices, dtype=int)

    for occ_idx, pos_abs in enumerate(h_positions):
        if pos_abs.shape != (3,) or not np.isfinite(pos_abs).all():
            continue
        cx, cy, cz = np.floor(pos_abs / spec.lattice_a + 1e-12).astype(int)
        key = (int(cx % spec.nx), int(cy % spec.ny), int(cz % spec.nz))
        neighbor_idx = None
        if idx_array is not None and occ_idx < len(idx_array):
            neighbor_idx = int(idx_array[occ_idx])
        cell_index.setdefault(key, []).append((np.array(pos_abs, dtype=float), neighbor_idx))

    return cell_index
