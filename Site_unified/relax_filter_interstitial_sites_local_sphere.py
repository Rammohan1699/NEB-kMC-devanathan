#!/usr/bin/env python3
"""
Relax/filter candidate H interstitial sites for BCC Fe bicrystal/polycrystal site discovery.

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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from lammps_data_utils import read_lammps_atomic_data, select_atom_indices

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
    """Parse common LAMMPS atom styles into the local Box representation."""
    data = read_lammps_atomic_data(path)
    lo = data.box[:, 0]
    hi = data.box[:, 1]
    box = Box(
        lo=lo.copy(),
        hi=hi.copy(),
        lengths=hi - lo,
    )
    return data.ids, data.types, data.pos, box


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
        # Pick lowest energy when available, otherwise smallest displacement.
        group = [accepted[j] for j in idxs]
        finite = [g for g in group if np.isfinite(g.energy)]
        if finite:
            rep = min(finite, key=lambda r: r.energy)
        else:
            rep = min(group, key=lambda r: r.displacement)
        # Average positions around representative using minimum image.
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


def make_lammps_calculator(commands: List[str]):
    try:
        from ase.calculators.lammpslib import LAMMPSlib
    except Exception as exc:
        raise RuntimeError("ASE LAMMPSlib calculator unavailable. Install ASE with LAMMPS support.") from exc
    if not commands:
        raise ValueError("--mode lammps requires at least one --lammps-cmd, e.g. pair_style and pair_coeff")
    return LAMMPSlib(lmpcmds=commands, atom_types={"Fe": 1, "H": 2}, keep_alive=True, log_file="none")


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
    if d0 < args.min_fe_dist:
        return Result(cand.cid, False, "initial_too_close_to_fe", h0, h0, d0, d0, 0.0, source_type=cand.ctype, source_region=cand.region)
    if d0 > args.max_fe_dist:
        return Result(cand.cid, False, "initial_too_far_from_fe", h0, h0, d0, d0, 0.0, source_type=cand.ctype, source_region=cand.region)

    if args.mode == "geometry":
        return Result(cand.cid, True, "geometry_pass", h0, h0, d0, d0, 0.0, source_type=cand.ctype, source_region=cand.region)

    cluster_mode = args.relax == "local-sphere-cluster"
    local_origin = None
    if cluster_mode:
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
            atoms.set_constraint(FixAtoms(indices=list(range(len(fe_pos)))))
        elif args.relax == "local":
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
        h1 = wrap_positions(atoms.positions[-1] + local_origin, box)
    else:
        h1 = wrap_positions(atoms.positions[-1] + box.lo, box)
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
    ap = argparse.ArgumentParser(description="Relax/filter candidate H interstitial sites in Fe.")
    ap.add_argument("fe_structure", help="LAMMPS data file for Fe host structure")
    ap.add_argument("--sites", required=True, help="Candidate sites CSV/XYZ/NPZ")
    ap.add_argument("--fe-type", type=int, default=1, help="Fe atom type. Use 0 if all atom types are Fe grain labels.")
    ap.add_argument("--fe-types", default=None, help="Comma-separated Fe atom types. Overrides --fe-type.")
    ap.add_argument("--mode", choices=["geometry", "lammps"], default="geometry", help="geometry screen only, or LAMMPSlib relaxation")
    ap.add_argument("--relax", choices=["h-only", "local", "local-sphere-cluster"], default="h-only", help="For lammps mode: h-only, full-cell local Fe, or fast local sphere cluster")
    ap.add_argument("--cluster-radius", type=float, default=8.0, help="For --relax local-sphere-cluster: Fe atoms within this radius are included in the local cluster")
    ap.add_argument("--cluster-vacuum", type=float, default=6.0, help="Vacuum padding around each non-periodic local cluster")
    ap.add_argument("--local-fe-radius", type=float, default=3.5, help="Fe atoms within this radius are movable for --relax local")
    ap.add_argument("--min-fe-dist", type=float, default=1.05, help="Reject sites with nearest Fe closer than this after relaxation")
    ap.add_argument("--max-fe-dist", type=float, default=2.35, help="Reject sites with nearest Fe farther than this after relaxation")
    ap.add_argument("--max-relax-displacement", type=float, default=1.50, help="Reject sites if H moves farther than this from initial candidate")
    ap.add_argument("--merge-radius", type=float, default=0.25, help="Merge accepted minima within this PBC distance")
    ap.add_argument("--fmax", type=float, default=0.03, help="ASE optimizer force tolerance in eV/Ang")
    ap.add_argument("--steps", type=int, default=200, help="ASE optimizer max steps per site")
    ap.add_argument("--optimizer", choices=["fire", "bfgs"], default="fire")
    ap.add_argument("--reject-unconverged", action="store_true", help="Reject sites if ASE optimizer does not report convergence")
    ap.add_argument("--lammps-cmd", action="append", default=[], help="LAMMPSlib command; repeat for pair_style/pair_coeff/etc")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N candidates; 0 means all")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth candidate")
    ap.add_argument("--out-prefix", default="relaxed_sites", help="Output prefix")
    args = ap.parse_args()

    ids, types, pos, box = parse_lammps_atoms_data(args.fe_structure)
    fe_idx = select_atom_indices(types, args.fe_type, args.fe_types)
    if len(fe_idx) == 0:
        raise SystemExit(f"ERROR: no Fe atoms selected from atom types {sorted(set(types.tolist()))}; check --fe-type/--fe-types")
    fe_pos = wrap_positions(pos[fe_idx], box)
    fe_tree = build_periodic_tree(fe_pos, box)

    candidates = load_candidates(args.sites)
    if args.stride > 1:
        candidates = candidates[::args.stride]
    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]

    print(f"Read Fe host: {len(fe_pos)} Fe atoms")
    print(f"Read candidates: {len(candidates)}")
    print(f"Mode: {args.mode}")
    print(f"Relax: {args.relax}")
    if args.relax == "local-sphere-cluster":
        print(f"Local sphere: mobile Fe radius={args.local_fe_radius:.3f} Å; cluster radius={args.cluster_radius:.3f} Å")

    calc = None
    if args.mode == "lammps":
        if Atoms is None:
            raise SystemExit("ERROR: ASE is required for --mode lammps")
        calc = make_lammps_calculator(args.lammps_cmd)

    results: List[Result] = []
    for i, cand in enumerate(candidates, start=1):
        r = relax_one_site(fe_pos, cand, box, fe_tree, args, calc)
        results.append(r)
        if i % 100 == 0 or i == len(candidates):
            nacc = sum(x.accepted for x in results)
            print(f"  processed {i}/{len(candidates)}; accepted={nacc}; last={r.reason}", flush=True)

    merged = merge_results(results, box, args.merge_radius)
    print(f"Accepted before merge: {sum(r.accepted for r in results)}")
    print(f"Accepted after merge:  {len(merged)}")

    write_csv(args.out_prefix + "_all_results.csv", results)
    write_csv(args.out_prefix + "_accepted_sites.csv", merged)
    write_extended_xyz(args.out_prefix + "_accepted_sites.xyz", merged, box)
    write_lammps_sites(args.out_prefix + "_accepted_sites.lmp", merged, box)
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
    write_summary(args.out_prefix + "_summary.txt", args, candidates, results, merged)

    print("Wrote:")
    print(f"  {args.out_prefix}_all_results.csv")
    print(f"  {args.out_prefix}_accepted_sites.csv")
    print(f"  {args.out_prefix}_accepted_sites.xyz")
    print(f"  {args.out_prefix}_accepted_sites.lmp")
    print(f"  {args.out_prefix}_accepted_sites.npz")
    print(f"  {args.out_prefix}_summary.txt")


if __name__ == "__main__":
    main()
