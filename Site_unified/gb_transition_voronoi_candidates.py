#!/usr/bin/env python3
"""
gb_transition_voronoi_candidates.py

Stage-3 candidate-site generator for Fe grain-boundary / transition regions.

Purpose
-------
Use the atom-region mask from the generic grain classifier to generate candidate
interstitial positions only in TRANSITION and GB_VORONOI regions. This script is
intended to complement the fast bulk tetrahedral mapper:

  BULK_TEMPLATE atoms  -> bulk_tetra_template_mapper_fast_cell.py
  TRANSITION/GB atoms  -> this script

The output is NOT yet the final KMC site graph. These are candidate H starting
positions that should be relaxed/minimized, then merged into final stable GB
sites.

Method
------
For each selected seed Fe atom in the TRANSITION/GB region:
  1. Build a small local Fe environment using periodic minimum-image vectors.
  2. Run a local 3D Voronoi construction on that small environment only.
  3. Keep Voronoi vertices that lie near the seed and satisfy Fe-H distance filters.
  4. Require the candidate to be supported by nearby TRANSITION/GB atoms, so it
     does not populate deep bulk interiors.
  5. Merge duplicate candidates globally using PBC-aware binning.

This avoids global Voronoi on the whole simulation box.

Dependencies
------------
numpy, scipy

Example
-------
python3 gb_transition_voronoi_candidates.py sigma5_210-20-20-5.lmp \
  --masks sigma5_regions_grain_masks.npz \
  --local-radius 5.0 \
  --collect-radius 3.2 \
  --min-fe-dist 1.05 \
  --max-fe-dist 2.35 \
  --region-support-radius 3.0 \
  --merge-radius 0.25 \
  --out-prefix sigma5_gb_voronoi_candidates

Outputs
-------
<out-prefix>.csv
<out-prefix>.xyz         extended XYZ with cell
<out-prefix>.lmp         OVITO-readable LAMMPS data file, atom type = candidate class
<out-prefix>.npz
<out-prefix>_summary.txt

Candidate classes in output:
  type 1 / GB_CORE      candidate nearest to GB_VORONOI atoms
  type 2 / TRANSITION   candidate nearest to TRANSITION atoms
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial import Voronoi, QhullError
except Exception as exc:  # pragma: no cover
    Voronoi = None
    QhullError = Exception
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None

from lammps_data_utils import read_lammps_atomic_data, select_atom_indices


def wrap_positions(pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    lo = box[:, 0]
    L = box[:, 1] - box[:, 0]
    return lo + np.mod(pos - lo, L)


def minimum_image(delta: np.ndarray, L: np.ndarray) -> np.ndarray:
    return delta - L * np.round(delta / L)


class PeriodicCellList:
    def __init__(self, positions: np.ndarray, box: np.ndarray, cutoff: float):
        self.pos = wrap_positions(np.asarray(positions, dtype=float), box)
        self.box = box
        self.lo = box[:, 0]
        self.L = box[:, 1] - box[:, 0]
        self.cutoff = float(cutoff)
        self.ncell = np.maximum(np.floor(self.L / self.cutoff).astype(int), 1)
        self.cell_size = self.L / self.ncell
        self.cells: Dict[Tuple[int, int, int], List[int]] = {}
        frac = (self.pos - self.lo) / self.cell_size
        self.atom_cells = np.floor(frac).astype(int) % self.ncell
        for i, c in enumerate(self.atom_cells):
            self.cells.setdefault(tuple(int(x) for x in c), []).append(i)
        # Include enough neighbor cells for a query cutoff up to ~2x build cutoff.
        self.max_shell = 2
        rng = range(-self.max_shell, self.max_shell + 1)
        self.offsets = [(i, j, k) for i in rng for j in rng for k in rng]

    def query_point(self, p: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cutoff2 = cutoff * cutoff
        p = wrap_positions(np.asarray(p, dtype=float).reshape(1, 3), self.box)[0]
        ci = np.floor((p - self.lo) / self.cell_size).astype(int) % self.ncell
        ids: List[int] = []
        vecs: List[np.ndarray] = []
        dists: List[float] = []
        seen = set()
        for off in self.offsets:
            ck = tuple(((ci + np.array(off, dtype=int)) % self.ncell).tolist())
            for j in self.cells.get(ck, []):
                if j in seen:
                    continue
                seen.add(j)
                dv = minimum_image(self.pos[j] - p, self.L)
                d2 = float(np.dot(dv, dv))
                if d2 <= cutoff2:
                    ids.append(j)
                    vecs.append(dv)
                    dists.append(math.sqrt(d2))
        if not ids:
            return np.empty(0, dtype=int), np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
        order = np.argsort(dists)
        return np.array(ids, dtype=int)[order], np.array(vecs, dtype=float)[order], np.array(dists, dtype=float)[order]


def load_region_masks(mask_path: str, n_atoms: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.load(mask_path)
    fe_global = m["fe_indices_global"].astype(int) if "fe_indices_global" in m else np.arange(n_atoms, dtype=int)
    if len(fe_global) and fe_global.min() == 1:
        fe_global = fe_global - 1
    if "region_code_all" in m:
        region = m["region_code_all"].astype(int)
    elif "atom_region_code" in m:
        region = m["atom_region_code"].astype(int)
    else:
        # Build from expected Fe-only masks.
        region = np.zeros(n_atoms, dtype=int)
        if "fe_bulk_template" in m:
            region[fe_global[m["fe_bulk_template"].astype(bool)]] = 1
        if "fe_transition" in m:
            region[fe_global[m["fe_transition"].astype(bool)]] = 2
        if "fe_gb_voronoi" in m:
            region[fe_global[m["fe_gb_voronoi"].astype(bool)]] = 3

    if "grain_id_all" in m:
        grain = m["grain_id_all"].astype(int)
    elif "fe_grain_id" in m and "fe_indices_global" in m:
        grain = -np.ones(n_atoms, dtype=int)
        grain[fe_global] = m["fe_grain_id"].astype(int)
    else:
        grain = -np.ones(n_atoms, dtype=int)

    if len(region) != n_atoms:
        raise ValueError(f"Region mask length {len(region)} does not match atom count {n_atoms}.")
    return region, grain, fe_global


def candidate_grain_context(
    point: np.ndarray,
    fe_cell: PeriodicCellList,
    fe_region: np.ndarray,
    fe_grain: np.ndarray,
    support_radius: float,
) -> Tuple[str, int, str]:
    ids, _, _ = fe_cell.query_point(point, support_radius)
    if len(ids) == 0:
        return "", 0, "UNRESOLVED"
    support = ids[(fe_region[ids] == 2) | (fe_region[ids] == 3)]
    grains = sorted({int(g) for g in fe_grain[support] if int(g) >= 0})
    grains_text = ";".join(str(g) for g in grains)
    if len(grains) == 0:
        cls = "UNRESOLVED"
    elif len(grains) == 1:
        cls = "SINGLE_GRAIN_TRANSITION"
    elif len(grains) == 2:
        cls = "BICRYSTAL_INTERFACE"
    else:
        cls = "POLYCRYSTAL_JUNCTION"
    return grains_text, len(grains), cls


def merge_pbc_points(points: np.ndarray, classes: np.ndarray, box: np.ndarray, merge_radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points.reshape(0, 3), classes.astype(int), np.empty(0, dtype=int)
    pts = wrap_positions(points, box)
    lo = box[:, 0]
    L = box[:, 1] - box[:, 0]
    cell = float(merge_radius)
    ncell = np.maximum(np.floor(L / cell).astype(int), 1)
    cell_size = L / ncell
    bins: Dict[Tuple[int, int, int], List[int]] = {}
    merged: List[np.ndarray] = []
    mclass: List[int] = []
    mult: List[int] = []
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]

    for p, c in zip(pts, classes):
        ci = tuple((np.floor((p - lo) / cell_size).astype(int) % ncell).tolist())
        found = None
        for off in offsets:
            ck = tuple(((np.array(ci) + np.array(off)) % ncell).tolist())
            for mi in bins.get(ck, []):
                dv = minimum_image(p - merged[mi], L)
                if float(np.dot(dv, dv)) <= merge_radius * merge_radius:
                    found = mi
                    break
            if found is not None:
                break
        if found is None:
            idx = len(merged)
            merged.append(p.copy())
            mclass.append(int(c))
            mult.append(1)
            bins.setdefault(ci, []).append(idx)
        else:
            # Periodic running average: move old point toward new point by min-image vector.
            old = merged[found]
            dv = minimum_image(p - old, L)
            mult[found] += 1
            merged[found] = wrap_positions((old + dv / mult[found]).reshape(1, 3), box)[0]
            # GB_CORE wins over TRANSITION if either source was GB_CORE.
            mclass[found] = min(mclass[found], int(c))
    return np.array(merged, dtype=float), np.array(mclass, dtype=int), np.array(mult, dtype=int)


def write_extended_xyz(path: str, pos: np.ndarray, classes: np.ndarray, box: np.ndarray):
    L = box[:, 1] - box[:, 0]
    lattice = f'{L[0]:.10f} 0 0 0 {L[1]:.10f} 0 0 0 {L[2]:.10f}'
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        f.write(f'Lattice="{lattice}" Origin="{box[0,0]:.10f} {box[1,0]:.10f} {box[2,0]:.10f}" Properties=species:S:1:pos:R:3:site_type:I:1 pbc="T T T"\n')
        for p, c in zip(pos, classes):
            sym = "HGB" if c == 1 else "HTR"
            f.write(f"{sym} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {int(c)}\n")


def write_lammps_sites(path: str, pos: np.ndarray, classes: np.ndarray, box: np.ndarray):
    with open(path, "w") as f:
        f.write("LAMMPS data file: GB/transition Voronoi candidate sites\n\n")
        f.write(f"{len(pos)} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"{box[0,0]:.10f} {box[0,1]:.10f} xlo xhi\n")
        f.write(f"{box[1,0]:.10f} {box[1,1]:.10f} ylo yhi\n")
        f.write(f"{box[2,0]:.10f} {box[2,1]:.10f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.00784 # GB_CORE_candidate\n")
        f.write("2 1.00784 # TRANSITION_candidate\n\n")
        f.write("Atoms # atomic\n\n")
        for i, (p, c) in enumerate(zip(pos, classes), start=1):
            f.write(f"{i} {int(c)} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def main():
    ap = argparse.ArgumentParser(description="Generate Voronoi candidate H sites in transition/GB regions only.")
    ap.add_argument("lammps_data")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--fe-type", type=int, default=1, help="Fallback Fe atom type if masks do not contain fe_indices_global.")
    ap.add_argument("--fe-types", default=None, help="Comma-separated fallback Fe atom types. Overrides --fe-type.")
    ap.add_argument("--local-radius", type=float, default=5.0, help="Fe environment radius for each local Voronoi patch.")
    ap.add_argument("--collect-radius", type=float, default=3.2, help="Keep vertices within this distance of the seed atom.")
    ap.add_argument("--min-fe-dist", type=float, default=1.05)
    ap.add_argument("--max-fe-dist", type=float, default=2.35)
    ap.add_argument("--region-support-radius", type=float, default=3.0, help="Candidate must have transition/GB Fe support within this radius.")
    ap.add_argument("--min-region-support", type=int, default=1)
    ap.add_argument("--merge-radius", type=float, default=0.25)
    ap.add_argument("--seed-stride", type=int, default=1, help="Use every Nth seed atom for quicker tests.")
    ap.add_argument("--max-seeds", type=int, default=0, help="Optional cap on number of seed atoms; 0 means all.")
    ap.add_argument("--out-prefix", default="gb_transition_voronoi_candidates")
    args = ap.parse_args()

    if Voronoi is None:
        raise RuntimeError(f"scipy.spatial.Voronoi is required. Import error: {SCIPY_IMPORT_ERROR}")

    data = read_lammps_atomic_data(args.lammps_data)
    pos = wrap_positions(data.pos, data.box)
    region, grain, fe_global = load_region_masks(args.masks, len(pos))
    with np.load(args.masks) as mask_npz:
        mask_keys = set(mask_npz.files)
    if len(fe_global) == len(pos) and "fe_indices_global" not in mask_keys:
        fe_global = select_atom_indices(data.types, args.fe_type, args.fe_types)
    if len(fe_global) == 0:
        raise ValueError(f"No Fe atoms selected from atom types {sorted(set(data.types.tolist()))}; check --fe-type/--fe-types")
    fe_pos = pos[fe_global]
    fe_region = region[fe_global]
    fe_grain = grain[fe_global]

    trans_mask = fe_region == 2
    gb_mask = fe_region == 3
    seed_mask = trans_mask | gb_mask
    seed_indices = np.where(seed_mask)[0]
    if args.seed_stride > 1:
        seed_indices = seed_indices[:: args.seed_stride]
    if args.max_seeds and args.max_seeds > 0:
        seed_indices = seed_indices[: args.max_seeds]

    print(f"Read {len(pos)} atoms; Fe atoms used: {len(fe_pos)}")
    print(f"TRANSITION atoms: {int(np.sum(trans_mask))}")
    print(f"GB_VORONOI atoms: {int(np.sum(gb_mask))}")
    print(f"Voronoi seed atoms used: {len(seed_indices)}")
    print(f"local_radius={args.local_radius:.3f} collect_radius={args.collect_radius:.3f}")

    search_cut = max(args.local_radius, args.region_support_radius, args.max_fe_dist) + 0.5
    cl = PeriodicCellList(fe_pos, data.box, cutoff=max(2.0, min(search_cut, 4.0)))

    raw_pts: List[np.ndarray] = []
    raw_cls: List[int] = []
    failed = 0
    small = 0
    vertices_total = 0

    L = data.box[:, 1] - data.box[:, 0]
    for count, seed in enumerate(seed_indices, start=1):
        center = pos[seed]
        ids, vecs, dists = cl.query_point(center, args.local_radius)
        if len(ids) < 8:
            small += 1
            continue
        local_pts = center + vecs  # unwrapped local cloud around center
        # Remove accidental duplicates that can upset Qhull.
        local_pts = np.unique(np.round(local_pts, decimals=8), axis=0)
        if len(local_pts) < 8:
            small += 1
            continue
        try:
            vor = Voronoi(local_pts)
        except QhullError:
            failed += 1
            continue
        for v in vor.vertices:
            if not np.all(np.isfinite(v)):
                continue
            if np.linalg.norm(v - center) > args.collect_radius:
                continue
            vw = wrap_positions(v.reshape(1, 3), data.box)[0]
            nids, _, nd = cl.query_point(vw, args.max_fe_dist)
            if len(nids) == 0:
                continue
            dmin = float(nd[0])
            if dmin < args.min_fe_dist or dmin > args.max_fe_dist:
                continue
            # Region support: this candidate must be near at least one transition/GB atom.
            rids, _, rd = cl.query_point(vw, args.region_support_radius)
            if len(rids) == 0:
                continue
            support_regions = fe_region[rids]
            n_reg = int(np.sum((support_regions == 2) | (support_regions == 3)))
            if n_reg < args.min_region_support:
                continue
            cls = 1 if np.any(support_regions == 3) else 2
            raw_pts.append(vw)
            raw_cls.append(cls)
        vertices_total += len(vor.vertices)
        if count % 500 == 0:
            print(f"  processed {count}/{len(seed_indices)} seeds; raw candidates={len(raw_pts)}")

    raw_pts_arr = np.array(raw_pts, dtype=float).reshape((-1, 3)) if raw_pts else np.empty((0, 3), dtype=float)
    raw_cls_arr = np.array(raw_cls, dtype=int) if raw_cls else np.empty(0, dtype=int)
    print(f"Raw Voronoi candidates: {len(raw_pts_arr)}")
    merged, mclass, mult = merge_pbc_points(raw_pts_arr, raw_cls_arr, data.box, args.merge_radius)
    print(f"Merged candidates: {len(merged)}")
    contexts = [
        candidate_grain_context(p, cl, fe_region, fe_grain, args.region_support_radius)
        for p in merged
    ]
    support_grains = np.array([x[0] for x in contexts], dtype=object)
    n_support_grains = np.array([x[1] for x in contexts], dtype=int)
    interface_class = np.array([x[2] for x in contexts], dtype=object)

    out = args.out_prefix
    csv_path = f"{out}.csv"
    with open(csv_path, "w") as f:
        f.write("site_id,x,y,z,candidate_class,multiplicity,support_grains,n_support_grains,interface_class\n")
        for i, (p, c, m, grains, n_grains, iface) in enumerate(
            zip(merged, mclass, mult, support_grains, n_support_grains, interface_class), start=1
        ):
            cname = "GB_CORE" if c == 1 else "TRANSITION"
            f.write(f"{i},{p[0]:.10f},{p[1]:.10f},{p[2]:.10f},{cname},{int(m)},{grains},{int(n_grains)},{iface}\n")
    xyz_path = f"{out}.xyz"
    lmp_path = f"{out}.lmp"
    npz_path = f"{out}.npz"
    write_extended_xyz(xyz_path, merged, mclass, data.box)
    write_lammps_sites(lmp_path, merged, mclass, data.box)
    np.savez_compressed(
        npz_path,
        positions=merged,
        candidate_class=mclass,
        multiplicity=mult,
        support_grains=support_grains,
        n_support_grains=n_support_grains,
        interface_class=interface_class,
        box=data.box,
    )

    iface_counts = {str(name): int(np.count_nonzero(interface_class == name)) for name in sorted(set(interface_class.tolist()))}
    iface_lines = "\n".join(f"  {name}: {count}" for name, count in iface_counts.items()) or "  none"

    summary = f"""Input: {args.lammps_data}
Masks: {args.masks}
Atoms: {len(pos)}
Fe atoms used: {len(fe_pos)}
TRANSITION atoms: {int(np.sum(trans_mask))}
GB_VORONOI atoms: {int(np.sum(gb_mask))}
Seed atoms used: {len(seed_indices)}

Parameters:
  local_radius = {args.local_radius:.6f} Å
  collect_radius = {args.collect_radius:.6f} Å
  min_fe_dist = {args.min_fe_dist:.6f} Å
  max_fe_dist = {args.max_fe_dist:.6f} Å
  region_support_radius = {args.region_support_radius:.6f} Å
  min_region_support = {args.min_region_support}
  merge_radius = {args.merge_radius:.6f} Å

Voronoi diagnostics:
  total Voronoi vertices examined = {vertices_total}
  local patches too small = {small}
  local Qhull failures = {failed}
  raw candidates = {len(raw_pts_arr)}
  merged candidates = {len(merged)}
  GB_CORE candidates = {int(np.sum(mclass == 1))}
  TRANSITION candidates = {int(np.sum(mclass == 2))}

Grain-interface context:
{iface_lines}

Outputs:
  {csv_path}
  {xyz_path}
  {lmp_path}
  {npz_path}
"""
    Path(f"{out}_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
