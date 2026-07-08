#!/usr/bin/env python3
"""
Stage 2 MPI local-sphere relaxation/filtering for GB/transition H site candidates.
Patched to initialize one independent LAMMPSlib instance per Python rank using MPI.COMM_SELF.

Purpose
-------
This is the next stage after:
  1) bulk tetra template mapping
  2) GB/transition Voronoi candidate generation

It takes a fixed Fe host structure and a list of candidate H positions, then either:
  A) geometry-screens the sites only, or
  B) inserts one H at a time and relaxes it with ASE + LAMMPSlib.

The output is a deduplicated list of physically accepted relaxed H sites.

Typical use, geometry-only quick check:
  python3 relax_filter_interstitial_sites.py sigma5_210-20-20-5.lmp \
      --sites sigma5_gb_voronoi_candidates.csv \
      --mode geometry \
      --out-prefix sigma5_gb_sites_geom

Typical use, H-only relaxation with LAMMPSlib:
  python3 relax_filter_interstitial_sites.py sigma5_210-20-20-5.lmp \
      --sites sigma5_gb_voronoi_candidates.csv \
      --mode lammps \
      --relax h-only \
      --lammps-cmd "pair_style eam/fs" \
      --lammps-cmd "pair_coeff * * FeH.eam.fs Fe H" \
      --out-prefix sigma5_gb_sites_relaxed

Notes
-----
- Fe host atoms are fixed by default. This is intentional for site validation.
- If you later want local Fe relaxation, use --relax local and set --local-fe-radius.
- This script does NOT compute migration barriers; it validates site minima.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    raise SystemExit("ERROR: scipy is required for cKDTree. Install scipy first.") from exc

try:
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.io import read
    from ase.optimize import FIRE, BFGS
except Exception:
    Atoms = None
    FixAtoms = None
    read = None
    FIRE = None
    BFGS = None


REGION_NAME = {
    0: "UNKNOWN",
    1: "BULK_TEMPLATE",
    2: "TRANSITION",
    3: "GB_VORONOI",
}


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    lengths: np.ndarray


@dataclass
class Candidate:
    cid: int
    pos: np.ndarray
    ctype: int = 0
    region: str = "UNKNOWN"


@dataclass
class Result:
    cid: int
    accepted: bool
    reason: str
    initial_pos: np.ndarray
    final_pos: np.ndarray
    initial_min_fe_dist: float
    final_min_fe_dist: float
    displacement: float
    energy: float = float("nan")
    force_max: float = float("nan")
    source_type: int = 0
    source_region: str = "UNKNOWN"


def parse_lammps_atoms_data(path: str) -> Tuple[np.ndarray, np.ndarray, Box]:
    """Minimal parser for orthogonal LAMMPS data files with Atoms section.

    Returns atom ids, types, positions, and box. Supports common atom styles:
    atomic: id type x y z
    charge/full-like: id mol type q x y z or id type q x y z
    The coordinate columns are inferred from line length.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    xlo = xhi = ylo = yhi = zlo = zhi = None
    for line in lines:
        words = line.split()
        if len(words) >= 4 and words[-2:] == ["xlo", "xhi"]:
            xlo, xhi = float(words[0]), float(words[1])
        elif len(words) >= 4 and words[-2:] == ["ylo", "yhi"]:
            ylo, yhi = float(words[0]), float(words[1])
        elif len(words) >= 4 and words[-2:] == ["zlo", "zhi"]:
            zlo, zhi = float(words[0]), float(words[1])
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError("Could not parse orthogonal box bounds from LAMMPS data file")
    box = Box(
        lo=np.array([xlo, ylo, zlo], dtype=float),
        hi=np.array([xhi, yhi, zhi], dtype=float),
        lengths=np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float),
    )

    atoms_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_start = i + 1
            break
    if atoms_start is None:
        raise ValueError("No Atoms section found")

    ids, types, pos = [], [], []
    for line in lines[atoms_start:]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Stop at next section header. Section names usually contain letters.
        if any(ch.isalpha() for ch in s.split()[0]):
            break
        parts = s.split("#", 1)[0].split()
        if len(parts) < 5:
            continue
        vals = [float(x) for x in parts]
        aid = int(vals[0])
        atype = int(vals[1])
        # Coordinates are almost always the last 3 numeric fields for the styles we use here.
        x, y, z = vals[-3:]
        ids.append(aid)
        types.append(atype)
        pos.append([x, y, z])

    if not pos:
        raise ValueError("No atoms parsed from Atoms section")
    order = np.argsort(ids)
    return np.asarray(ids, int)[order], np.asarray(types, int)[order], np.asarray(pos, float)[order], box


def wrap_positions(pos: np.ndarray, box: Box) -> np.ndarray:
    return box.lo + np.mod(pos - box.lo, box.lengths)


def minimum_image_delta(a: np.ndarray, b: np.ndarray, box: Box) -> np.ndarray:
    d = a - b
    d -= box.lengths * np.round(d / box.lengths)
    return d


def pbc_distance(a: np.ndarray, b: np.ndarray, box: Box) -> float:
    return float(np.linalg.norm(minimum_image_delta(a, b, box)))


def scaled_positions(pos: np.ndarray, box: Box) -> np.ndarray:
    return np.mod((pos - box.lo) / box.lengths, 1.0)


def build_periodic_tree(pos: np.ndarray, box: Box) -> cKDTree:
    return cKDTree(scaled_positions(pos, box), boxsize=1.0)


def query_distances(tree: cKDTree, pos: np.ndarray, box: Box, k: int = 1):
    frac = scaled_positions(np.atleast_2d(pos), box)
    d_frac, idx = tree.query(frac, k=k)
    return np.asarray(d_frac) * np.mean(box.lengths), idx  # not used for exact distance except fallback


def exact_nearest_distance(pos: np.ndarray, fe_pos: np.ndarray, tree: cKDTree, box: Box, k: int = 16) -> Tuple[float, int]:
    frac = scaled_positions(np.atleast_2d(pos), box)
    _, idxs = tree.query(frac, k=min(k, len(fe_pos)))
    idxs = np.atleast_1d(idxs[0] if np.ndim(idxs) == 2 else idxs)
    best_d = float("inf")
    best_i = -1
    for ii in idxs:
        d = pbc_distance(pos, fe_pos[int(ii)], box)
        if d < best_d:
            best_d = d
            best_i = int(ii)
    return best_d, best_i


def exact_neighbors_within(pos: np.ndarray, fe_pos: np.ndarray, tree: cKDTree, box: Box, radius: float) -> np.ndarray:
    # Query in fractional metric conservatively. Exact filter follows.
    frac_radius = radius / float(np.min(box.lengths))
    idxs = tree.query_ball_point(scaled_positions(np.atleast_2d(pos), box)[0], r=frac_radius)
    keep = []
    for ii in idxs:
        if pbc_distance(pos, fe_pos[int(ii)], box) <= radius:
            keep.append(int(ii))
    return np.asarray(keep, dtype=int)


def load_candidates(path: str) -> List[Candidate]:
    ext = os.path.splitext(path)[1].lower()
    candidates: List[Candidate] = []

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        # Try common keys from earlier scripts.
        for key in ["sites", "positions", "candidate_positions", "pos"]:
            if key in data:
                arr = np.asarray(data[key], float)
                break
        else:
            raise KeyError(f"No positions key found in {path}. Available keys: {list(data.keys())}")
        types = np.asarray(data["types"], int) if "types" in data else np.zeros(len(arr), int)
        for i, p in enumerate(arr):
            candidates.append(Candidate(i, np.asarray(p, float), int(types[i]) if i < len(types) else 0))
        return candidates

    if ext == ".csv":
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            def find_col(options):
                for opt in options:
                    if opt in cols:
                        return opt
                return None
            xcol = find_col(["x", "X", "pos_x"])
            ycol = find_col(["y", "Y", "pos_y"])
            zcol = find_col(["z", "Z", "pos_z"])
            if xcol is None or ycol is None or zcol is None:
                raise ValueError(f"CSV must contain x,y,z columns. Found: {cols}")
            type_col = find_col(["type", "site_type", "candidate_type"])
            region_col = find_col(["region", "site_region"])
            id_col = find_col(["id", "site_id", "candidate_id"])
            for n, row in enumerate(reader):
                cid = int(row[id_col]) if id_col and row[id_col] not in ("", None) else n
                ctype = int(float(row[type_col])) if type_col and row[type_col] not in ("", None) else 0
                region = row[region_col] if region_col and row[region_col] else REGION_NAME.get(ctype, "UNKNOWN")
                candidates.append(Candidate(cid, np.array([float(row[xcol]), float(row[ycol]), float(row[zcol])]), ctype, region))
        return candidates

    # Basic/extended XYZ. Coordinates are columns 2-4 after symbol.
    if ext == ".xyz":
        with open(path, "r") as f:
            lines = f.readlines()
        try:
            n = int(lines[0].strip())
            atom_lines = lines[2:2+n]
        except Exception:
            atom_lines = lines
        for i, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                continue
            # If the earlier candidate XYZ used symbols C/T or H, keep type approximate.
            sym = parts[0]
            ctype = 1 if sym.upper().startswith("G") else 2 if sym.upper().startswith("T") else 0
            candidates.append(Candidate(i, np.array([x, y, z], float), ctype, REGION_NAME.get(ctype, "UNKNOWN")))
        return candidates

    raise ValueError(f"Unsupported site file extension: {ext}")


def merge_results(results: List[Result], box: Box, merge_radius: float) -> List[Result]:
    accepted = [r for r in results if r.accepted]
    if not accepted:
        return []
    pts = np.array([wrap_positions(r.final_pos, box) for r in accepted])
    tree = build_periodic_tree(pts, box)
    visited = np.zeros(len(pts), dtype=bool)
    merged: List[Result] = []
    for i in range(len(pts)):
        if visited[i]:
            continue
        idxs = exact_neighbors_within(pts[i], pts, tree, box, merge_radius)
        if len(idxs) == 0:
            idxs = np.array([i], int)
        visited[idxs] = True
        # Multiple candidates can relax into the same metastable basin. Keep one
        # representative per merge-radius cluster; energy is the best tie-breaker
        # when LAMMPS energies are available.
        group = [accepted[j] for j in idxs]
        finite = [g for g in group if np.isfinite(g.energy)]
        if finite:
            rep = min(finite, key=lambda r: r.energy)
        else:
            rep = min(group, key=lambda r: r.displacement)
        # Average positions around the representative using minimum-image
        # displacements, so clusters crossing a periodic boundary remain compact.
        accum = rep.final_pos.copy()
        shifts = []
        for g in group:
            shifts.append(rep.final_pos + minimum_image_delta(g.final_pos, rep.final_pos, box))
        mean_pos = wrap_positions(np.mean(np.vstack(shifts), axis=0), box)
        merged.append(Result(
            cid=rep.cid,
            accepted=True,
            reason=f"merged_{len(group)}",
            initial_pos=rep.initial_pos,
            final_pos=mean_pos,
            initial_min_fe_dist=rep.initial_min_fe_dist,
            final_min_fe_dist=rep.final_min_fe_dist,
            displacement=rep.displacement,
            energy=rep.energy,
            force_max=rep.force_max,
            source_type=rep.source_type,
            source_region=rep.source_region,
        ))
    return merged


def make_ase_atoms(fe_pos: np.ndarray, h_pos: np.ndarray, box: Box):
    if Atoms is None:
        raise RuntimeError("ASE is required for --mode lammps")
    symbols = ["Fe"] * len(fe_pos) + ["H"]
    positions = np.vstack([fe_pos, h_pos.reshape(1, 3)])
    cell = np.diag(box.lengths)
    atoms = Atoms(symbols=symbols, positions=positions - box.lo, cell=cell, pbc=True)
    return atoms


def make_local_cluster_atoms(fe_pos: np.ndarray, h_pos: np.ndarray, fe_tree: cKDTree, box: Box, cluster_radius: float, vacuum: float):
    """Build a small non-periodic Fe+H cluster around h_pos.

    Fe positions are unwrapped relative to H by minimum image, so a cluster crossing
    a periodic boundary remains contiguous. The last atom is H.
    """
    if Atoms is None:
        raise RuntimeError("ASE is required for --mode lammps")
    h_wrapped = wrap_positions(h_pos, box)
    local_idxs = exact_neighbors_within(h_wrapped, fe_pos, fe_tree, box, cluster_radius)
    if len(local_idxs) == 0:
        raise RuntimeError("No Fe atoms found inside local cluster radius")
    unwrapped_fe = []
    for ii in local_idxs:
        d = minimum_image_delta(fe_pos[int(ii)], h_wrapped, box)
        unwrapped_fe.append(h_wrapped + d)
    unwrapped_fe = np.asarray(unwrapped_fe, float)
    coords = np.vstack([unwrapped_fe, h_wrapped.reshape(1, 3)])
    lo = coords.min(axis=0) - vacuum
    hi = coords.max(axis=0) + vacuum
    cell_lengths = np.maximum(hi - lo, np.array([20.0, 20.0, 20.0]))
    atoms = Atoms(symbols=["Fe"] * len(local_idxs) + ["H"], positions=coords - lo, cell=np.diag(cell_lengths), pbc=False)
    return atoms, local_idxs, lo


def make_lammps_calculator(commands: List[str], mpi_comm_self=None, rank: int = 0):
    """Create an ASE LAMMPSlib calculator safely inside an MPI Python job.

    Important: when this script itself is launched with mpirun, every Python
    rank must create an independent LAMMPS library instance on MPI.COMM_SELF.
    If LAMMPSlib uses MPI.COMM_WORLD, all ranks enter the same LAMMPS instance
    while trying to process different candidates, which can deadlock.
    """
    try:
        from ase.calculators.lammpslib import LAMMPSlib
    except Exception as exc:
        raise RuntimeError("ASE LAMMPSlib calculator unavailable. Install ASE with LAMMPS support.") from exc
    if not commands:
        raise ValueError("--mode lammps requires at least one --lammps-cmd, e.g. pair_style and pair_coeff")

    kwargs = dict(lmpcmds=commands, atom_types={"Fe": 1, "H": 2}, keep_alive=True, log_file="none")

    if mpi_comm_self is not None:
        try:
            return LAMMPSlib(**kwargs, comm=mpi_comm_self)
        except TypeError:
            if rank == 0:
                print("WARNING: This ASE LAMMPSlib does not accept comm=MPI.COMM_SELF; "
                      "multi-rank runs may hang. Consider using a newer ASE or "
                      "a subprocess-based LAMMPS validator.", flush=True)

    return LAMMPSlib(**kwargs)


def relax_one_site(
    fe_pos: np.ndarray,
    cand: Candidate,
    box: Box,
    fe_tree: cKDTree,
    args,
    calc,
) -> Result:
    h0 = wrap_positions(cand.pos, box)
    d0, _ = exact_nearest_distance(h0, fe_pos, fe_tree, box)
    # Cheap geometry rejection before creating an ASE/LAMMPS system. These
    # limits are deliberately the same before and after relaxation.
    if d0 < args.min_fe_dist:
        return Result(cand.cid, False, "initial_too_close_to_fe", h0, h0, d0, d0, 0.0, source_type=cand.ctype, source_region=cand.region)
    if d0 > args.max_fe_dist:
        return Result(cand.cid, False, "initial_too_far_from_fe", h0, h0, d0, d0, 0.0, source_type=cand.ctype, source_region=cand.region)

    if args.mode == "geometry":
        return Result(cand.cid, True, "geometry_pass", h0, h0, d0, d0, 0.0, source_type=cand.ctype, source_region=cand.region)

    cluster_mode = args.relax == "local-sphere-cluster"
    local_origin = None
    if cluster_mode:
        # Fast production path: cut a non-periodic local Fe cluster around the
        # candidate H. Fe atoms inside local-fe-radius are mobile; outer cluster
        # Fe atoms provide boundary stiffness and remain fixed.
        atoms, local_idxs, local_origin = make_local_cluster_atoms(
            fe_pos, h0, fe_tree, box, args.cluster_radius, args.cluster_vacuum
        )
        atoms.calc = calc
        mobile_fe_local = []
        for j, orig_i in enumerate(local_idxs):
            if pbc_distance(fe_pos[int(orig_i)], h0, box) <= args.local_fe_radius:
                mobile_fe_local.append(j)
        movable = set(mobile_fe_local + [len(atoms) - 1])
        fixed = [j for j in range(len(atoms)) if j not in movable]
        atoms.set_constraint(FixAtoms(indices=fixed))
    else:
        atoms = make_ase_atoms(fe_pos, h0, box)
        atoms.calc = calc
        if args.relax == "h-only":
            # Conservative validation mode: all host Fe atoms are fixed and only
            # the inserted H can move.
            atoms.set_constraint(FixAtoms(indices=list(range(len(fe_pos)))))
        elif args.relax == "local":
            # Full periodic cell mode with a mobile Fe sphere around H. This is
            # slower than local-sphere-cluster but useful for debugging.
            local = exact_neighbors_within(h0, fe_pos, fe_tree, box, args.local_fe_radius)
            movable = set(int(i) for i in local)
            fixed = [i for i in range(len(fe_pos)) if i not in movable]
            atoms.set_constraint(FixAtoms(indices=fixed))
        else:
            raise ValueError("--relax must be h-only, local, or local-sphere-cluster")

    opt_cls = FIRE if args.optimizer.lower() == "fire" else BFGS
    opt = opt_cls(atoms, logfile=None)
    converged = False
    try:
        converged = opt.run(fmax=args.fmax, steps=args.steps)
        e = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
    except Exception as exc:
        return Result(cand.cid, False, f"relax_failed:{type(exc).__name__}", h0, h0, d0, d0, 0.0, source_type=cand.ctype, source_region=cand.region)

    if cluster_mode:
        h1_wrapped = wrap_positions(atoms.positions[-1] + local_origin, box)
    else:
        h1_wrapped = wrap_positions(atoms.positions[-1] + box.lo, box)

    # Anchor final H to the same periodic image as the initial representative
    # candidate. This prevents a site near x=0 from being written as x≈Lx.
    if getattr(args, "anchor_final_to_initial", False):
        h1 = h0 + minimum_image_delta(h1_wrapped, h0, box)
    else:
        h1 = h1_wrapped

    d1, _ = exact_nearest_distance(h1, fe_pos, fe_tree, box)
    disp = pbc_distance(h1, h0, box)

    if not converged and args.reject_unconverged:
        return Result(cand.cid, False, "not_converged", h0, h1, d0, d1, disp, e, fmax, cand.ctype, cand.region)
    if d1 < args.min_fe_dist:
        return Result(cand.cid, False, "final_too_close_to_fe", h0, h1, d0, d1, disp, e, fmax, cand.ctype, cand.region)
    if d1 > args.max_fe_dist:
        return Result(cand.cid, False, "final_too_far_from_fe", h0, h1, d0, d1, disp, e, fmax, cand.ctype, cand.region)
    if disp > args.max_relax_displacement:
        return Result(cand.cid, False, "moved_too_far", h0, h1, d0, d1, disp, e, fmax, cand.ctype, cand.region)

    return Result(cand.cid, True, "relaxed_minimum", h0, h1, d0, d1, disp, e, fmax, cand.ctype, cand.region)


def write_csv(path: str, results: List[Result]) -> None:
    fields = [
        "id", "accepted", "reason", "x0", "y0", "z0", "x", "y", "z",
        "initial_min_fe_dist", "final_min_fe_dist", "displacement", "energy", "force_max",
        "source_type", "source_region",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "id": r.cid,
                "accepted": int(r.accepted),
                "reason": r.reason,
                "x0": r.initial_pos[0], "y0": r.initial_pos[1], "z0": r.initial_pos[2],
                "x": r.final_pos[0], "y": r.final_pos[1], "z": r.final_pos[2],
                "initial_min_fe_dist": r.initial_min_fe_dist,
                "final_min_fe_dist": r.final_min_fe_dist,
                "displacement": r.displacement,
                "energy": r.energy,
                "force_max": r.force_max,
                "source_type": r.source_type,
                "source_region": r.source_region,
            })


def write_extended_xyz(path: str, results: List[Result], box: Box, symbol: str = "H") -> None:
    accepted = [r for r in results if r.accepted]
    lattice = f'{box.lengths[0]} 0 0 0 {box.lengths[1]} 0 0 0 {box.lengths[2]}'
    with open(path, "w") as f:
        f.write(f"{len(accepted)}\n")
        f.write(f'Lattice="{lattice}" Origin="{box.lo[0]} {box.lo[1]} {box.lo[2]}" Properties=species:S:1:pos:R:3:id:I:1:source_type:I:1:energy:R:1 pbc="T T T"\n')
        for r in accepted:
            p = wrap_positions(r.final_pos, box)
            f.write(f"{symbol} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {r.cid:d} {r.source_type:d} {r.energy:.12g}\n")


def write_lammps_sites(path: str, results: List[Result], box: Box) -> None:
    accepted = [r for r in results if r.accepted]
    with open(path, "w") as f:
        f.write("Accepted relaxed H interstitial sites\n\n")
        f.write(f"{len(accepted)} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"{box.lo[0]:.12f} {box.hi[0]:.12f} xlo xhi\n")
        f.write(f"{box.lo[1]:.12f} {box.hi[1]:.12f} ylo yhi\n")
        f.write(f"{box.lo[2]:.12f} {box.hi[2]:.12f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.008 # H_GB_CORE\n")
        f.write("2 1.008 # H_TRANSITION\n\n")
        f.write("Atoms # atomic\n\n")
        for i, r in enumerate(accepted, start=1):
            p = wrap_positions(r.final_pos, box)
            atype = 1 if r.source_type == 1 else 2
            f.write(f"{i:d} {atype:d} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def write_lammps_sites_anchored(path: str, results: List[Result], box: Box, pad: float = 2.0) -> None:
    accepted = [r for r in results if r.accepted]
    if accepted:
        pts = np.array([r.final_pos for r in accepted], float)
        lo = np.minimum(box.lo, pts.min(axis=0) - pad)
        hi = np.maximum(box.hi, pts.max(axis=0) + pad)
    else:
        lo, hi = box.lo, box.hi
    with open(path, "w") as f:
        f.write("Accepted relaxed H sites anchored to representative candidate image\n\n")
        f.write(f"{len(accepted)} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"{lo[0]:.12f} {hi[0]:.12f} xlo xhi\n")
        f.write(f"{lo[1]:.12f} {hi[1]:.12f} ylo yhi\n")
        f.write(f"{lo[2]:.12f} {hi[2]:.12f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.008 # H_GB_CORE\n")
        f.write("2 1.008 # H_TRANSITION\n\n")
        f.write("Atoms # atomic\n\n")
        for i, r in enumerate(accepted, start=1):
            p = r.final_pos
            atype = 1 if r.source_type == 1 else 2
            f.write(f"{i:d} {atype:d} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def write_extended_xyz_anchored(path: str, results: List[Result], box: Box, symbol: str = "H") -> None:
    accepted = [r for r in results if r.accepted]
    if accepted:
        pts = np.array([r.final_pos for r in accepted], float)
        lo = np.minimum(box.lo, pts.min(axis=0) - 2.0)
        hi = np.maximum(box.hi, pts.max(axis=0) + 2.0)
        L = hi - lo
    else:
        lo = box.lo
        L = box.lengths
    lattice = f'{L[0]} 0 0 0 {L[1]} 0 0 0 {L[2]}'
    with open(path, "w") as f:
        f.write(f"{len(accepted)}\n")
        f.write(f'Lattice="{lattice}" Origin="{lo[0]} {lo[1]} {lo[2]}" Properties=species:S:1:pos:R:3:id:I:1:source_type:I:1:energy:R:1 pbc="T T T"\n')
        for r in accepted:
            p = r.final_pos
            f.write(f"{symbol} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {r.cid:d} {r.source_type:d} {r.energy:.12g}\n")


def write_summary(path: str, args, candidates: List[Candidate], results: List[Result], merged: List[Result]) -> None:
    from collections import Counter
    rejected = [r for r in results if not r.accepted]
    c_rej = Counter(r.reason for r in rejected)
    c_acc = Counter(r.reason for r in results if r.accepted)
    with open(path, "w") as f:
        f.write("Interstitial site relaxation/filter summary\n")
        f.write("==========================================\n\n")
        f.write(f"Fe structure: {args.fe_structure}\n")
        f.write(f"Candidate sites: {args.sites}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Relax mode: {args.relax}\n")
        f.write(f"Input candidates: {len(candidates)}\n")
        f.write(f"Accepted before merge: {sum(r.accepted for r in results)}\n")
        f.write(f"Accepted after merge: {len(merged)}\n")
        f.write(f"Rejected: {len(rejected)}\n\n")
        f.write("Acceptance reasons:\n")
        for k, v in sorted(c_acc.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nRejection reasons:\n")
        for k, v in sorted(c_rej.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nParameters:\n")
        for key, val in sorted(vars(args).items()):
            if key != "lammps_cmd":
                f.write(f"  {key}: {val}\n")
        if args.lammps_cmd:
            f.write("  lammps_cmd:\n")
            for cmd in args.lammps_cmd:
                f.write(f"    {cmd}\n")


def main() -> None:
    """MPI-parallel local-sphere validator for GB/transition H candidates."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except Exception:
        comm = None
        rank = 0
        size = 1

    ap = argparse.ArgumentParser(description="Stage 2 MPI local-sphere relaxation/filtering of GB H interstitial candidates.")
    ap.add_argument("fe_structure", help="LAMMPS data file for Fe host structure")
    ap.add_argument("--sites", required=True, help="Candidate sites CSV/XYZ/NPZ from Stage 1 Voronoi discovery")
    ap.add_argument("--mode", choices=["lammps"], default="lammps")
    ap.add_argument("--relax", choices=["local-sphere-cluster", "h-only"], default="local-sphere-cluster")
    ap.add_argument("--local-fe-radius", type=float, default=10.0, help="Mobile Fe radius around H. Default: 10 Å")
    ap.add_argument("--cluster-radius", type=float, default=12.0, help="Included Fe cluster radius. Default: 12 Å")
    ap.add_argument("--cluster-vacuum", type=float, default=6.0)
    ap.add_argument("--min-fe-dist", type=float, default=1.05)
    ap.add_argument("--max-fe-dist", type=float, default=2.35)
    ap.add_argument("--max-relax-displacement", type=float, default=1.50)
    ap.add_argument("--merge-radius", type=float, default=0.25)
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--optimizer", choices=["fire", "bfgs"], default="fire")
    ap.add_argument("--reject-unconverged", action="store_true")
    ap.add_argument("--lammps-cmd", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rank-progress", type=int, default=25, help="Print per-rank progress every N local candidates. Set 0 to disable. Default: 25")
    ap.add_argument("--out-prefix", default="stage2_gb_relaxed_mpi")
    ap.add_argument("--no-anchor-final-to-initial", dest="anchor_final_to_initial", action="store_false")
    ap.set_defaults(anchor_final_to_initial=True)
    args = ap.parse_args()

    if not args.lammps_cmd:
        if rank == 0:
            raise SystemExit("ERROR: --mode lammps requires --lammps-cmd entries for pair_style/pair_coeff")
        return

    ids, types, pos, box = parse_lammps_atoms_data(args.fe_structure)
    fe_mask = types == 1
    fe_pos = wrap_positions(pos[fe_mask], box)
    fe_tree = build_periodic_tree(fe_pos, box)

    candidates = load_candidates(args.sites)
    if args.stride > 1:
        candidates = candidates[::args.stride]
    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]

    # Static block distribution by candidate index. This keeps rank ownership
    # reproducible and makes rank#### partial CSV files easy to interpret.
    local_candidates = candidates[rank::size]

    if rank == 0:
        print("Stage 2 MPI local-sphere GB site validation")
        print(f"Read Fe host: {len(fe_pos)} Fe atoms")
        print(f"Read candidates: {len(candidates)}")
        print(f"MPI ranks: {size}")
        print(f"Relax: {args.relax}")
        print(f"local-fe-radius={args.local_fe_radius:.3f} Å; cluster-radius={args.cluster_radius:.3f} Å")
        print(f"Each rank will process about {math.ceil(len(candidates)/max(size,1))} candidates")
        sys.stdout.flush()
    if comm is not None:
        comm.Barrier()

    print(
        f"[rank {rank:03d}/{size}] starting: assigned {len(local_candidates)} candidates "
        f"(first_id={local_candidates[0].cid if local_candidates else 'none'}, "
        f"last_id={local_candidates[-1].cid if local_candidates else 'none'})",
        flush=True,
    )

    if comm is not None:
        comm.Barrier()

    if Atoms is None:
        raise SystemExit("ERROR: ASE is required for Stage 2 LAMMPS relaxation")

    # One calculator per Python rank. The COMM_SELF handoff is the key guard
    # against MPI collectives inside LAMMPSlib crossing candidate work between
    # ranks.
    print(f"[rank {rank:03d}/{size}] creating LAMMPSlib calculator on MPI.COMM_SELF", flush=True)
    mpi_comm_self = None
    try:
        from mpi4py import MPI as _MPI
        mpi_comm_self = _MPI.COMM_SELF
    except Exception:
        mpi_comm_self = None
    calc = make_lammps_calculator(args.lammps_cmd, mpi_comm_self=mpi_comm_self, rank=rank)
    print(f"[rank {rank:03d}/{size}] LAMMPSlib calculator ready", flush=True)

    local_results: List[Result] = []
    rank_t0 = time.time()
    accepted_count = 0
    for local_i, cand in enumerate(local_candidates, start=1):
        cand_t0 = time.time()
        r = relax_one_site(fe_pos, cand, box, fe_tree, args, calc)
        cand_elapsed = time.time() - cand_t0
        local_results.append(r)
        if r.accepted:
            accepted_count += 1
        if args.rank_progress > 0 and (local_i % args.rank_progress == 0 or local_i == len(local_candidates)):
            elapsed = time.time() - rank_t0
            avg = elapsed / max(local_i, 1)
            remaining = max(len(local_candidates) - local_i, 0)
            eta = avg * remaining
            print(
                f"[rank {rank:03d}/{size}] "
                f"processed {local_i}/{len(local_candidates)}; "
                f"accepted={accepted_count}; "
                f"elapsed={elapsed/60.0:.2f} min; "
                f"avg={avg:.2f} s/site; "
                f"eta={eta/60.0:.2f} min; "
                f"last_id={cand.cid}; last={r.reason}; last_time={cand_elapsed:.2f} s",
                flush=True,
            )

    total_elapsed = time.time() - rank_t0
    print(
        f"[rank {rank:03d}/{size}] finished: "
        f"processed={len(local_candidates)}; accepted={accepted_count}; "
        f"elapsed={total_elapsed/60.0:.2f} min; "
        f"avg={total_elapsed/max(len(local_candidates),1):.2f} s/site",
        flush=True,
    )
    partial_csv = f"{args.out_prefix}_rank{rank:04d}_all_results.csv"
    write_csv(partial_csv, local_results)

    # Gather Python Result objects only after all ranks have written their own
    # partial CSV. The rank files are useful recovery points if a later merge or
    # filesystem write fails.
    if comm is not None:
        gathered = comm.gather(local_results, root=0)
    else:
        gathered = [local_results]
    if rank != 0:
        return

    all_results: List[Result] = []
    for block in gathered:
        all_results.extend(block)
    all_results.sort(key=lambda r: r.cid)
    # Stage 2 outputs relaxed GB/transition minima. Merge here, before Stage 3,
    # so the final unified map receives one site per relaxed basin.
    merged = merge_results(all_results, box, args.merge_radius)

    print("\nStage 2 MPI summary", flush=True)
    print(f"Input candidates:        {len(candidates)}")
    print(f"Accepted before merge:   {sum(r.accepted for r in all_results)}")
    print(f"Accepted after merge:    {len(merged)}")
    print(f"Rejected:                {sum(not r.accepted for r in all_results)}")

    write_csv(args.out_prefix + "_all_results.csv", all_results)
    write_csv(args.out_prefix + "_accepted_sites.csv", merged)
    write_extended_xyz(args.out_prefix + "_accepted_sites.xyz", merged, box)
    write_lammps_sites(args.out_prefix + "_accepted_sites.lmp", merged, box)
    write_csv(args.out_prefix + "_accepted_sites_anchored.csv", merged)
    write_extended_xyz_anchored(args.out_prefix + "_accepted_sites_anchored.xyz", merged, box)
    write_lammps_sites_anchored(args.out_prefix + "_accepted_sites_anchored.lmp", merged, box)
    np.savez(
        args.out_prefix + "_accepted_sites.npz",
        positions=np.array([r.final_pos for r in merged], float),
        energies=np.array([r.energy for r in merged], float),
        source_type=np.array([r.source_type for r in merged], int),
        ids=np.array([r.cid for r in merged], int),
        box_lo=box.lo,
        box_hi=box.hi,
        box_lengths=box.lengths,
    )
    write_summary(args.out_prefix + "_summary.txt", args, candidates, all_results, merged)

    print("Wrote:")
    print(f"  {args.out_prefix}_all_results.csv")
    print(f"  {args.out_prefix}_accepted_sites.csv")
    print(f"  {args.out_prefix}_accepted_sites.xyz")
    print(f"  {args.out_prefix}_accepted_sites.lmp")
    print(f"  {args.out_prefix}_accepted_sites_anchored.csv")
    print(f"  {args.out_prefix}_accepted_sites_anchored.xyz")
    print(f"  {args.out_prefix}_accepted_sites_anchored.lmp")
    print(f"  {args.out_prefix}_accepted_sites.npz")
    print(f"  {args.out_prefix}_summary.txt")
    print(f"  {args.out_prefix}_rank####_all_results.csv")


if __name__ == "__main__":
    main()
