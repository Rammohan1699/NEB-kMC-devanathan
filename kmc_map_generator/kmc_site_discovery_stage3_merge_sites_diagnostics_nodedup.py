#!/usr/bin/env python3
"""
Stage 3: merge bulk tetrahedral sites and relaxed GB sites with diagnostics.

This version intentionally does NOT do a final all-site deduplication.
It performs only the physically meaningful interface cleanup:

    1. Read host LAMMPS box.
    2. Read bulk theoretical tetra sites.
    3. Read relaxed GB/metastable sites.
    4. Report bulk<->GB overlap diagnostics before removal.
    5. Remove bulk sites that overlap/duplicate GB sites within a cutoff.
    6. Concatenate remaining bulk sites + GB sites.
    7. Write OVITO-readable LAMMPS/XYZ/CSV/NPZ outputs.

Site types in output:
    type 1 = BULK_TETRA
    type 2 = GB_RELAXED
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Box:
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    @property
    def lengths(self) -> np.ndarray:
        return np.array([self.xhi - self.xlo, self.yhi - self.ylo, self.zhi - self.zlo], dtype=float)

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.xlo, self.ylo, self.zlo], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge bulk tetra sites and relaxed GB sites with overlap diagnostics, no final dedup."
    )
    p.add_argument("host_lmp", help="LAMMPS data file for host Fe structure; used only for box/cell.")
    p.add_argument("--bulk-sites", required=True, help="CSV file containing bulk tetrahedral site positions.")
    p.add_argument("--gb-sites", required=True, help="CSV file containing relaxed GB site positions.")
    p.add_argument("--bulk-gb-exclusion", type=float, default=0.35,
                   help="Remove bulk sites within this distance of any GB site. Default: 0.35 Å")
    p.add_argument("--diagnostic-thresholds", default="0.15,0.25,0.35,0.50,0.75,1.00",
                   help="Comma-separated thresholds for overlap diagnostics in Å.")
    p.add_argument("--out-prefix", default="stage3_unified",
                   help="Output prefix. Default: stage3_unified")
    return p.parse_args()


def read_lammps_box(path: str) -> Box:
    xlo = xhi = ylo = yhi = zlo = zhi = None
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
                xlo, xhi = float(parts[0]), float(parts[1])
            elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
                ylo, yhi = float(parts[0]), float(parts[1])
            elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
                zlo, zhi = float(parts[0]), float(parts[1])
            if None not in (xlo, xhi, ylo, yhi, zlo, zhi):
                return Box(xlo, xhi, ylo, yhi, zlo, zhi)
    raise ValueError(f"Could not read x/y/z box bounds from {path}")


def _find_xyz_columns(fieldnames: Iterable[str]) -> Tuple[str, str, str]:
    names = list(fieldnames)
    lower = {n.lower(): n for n in names}
    candidates = [
        ("x", "y", "z"),
        ("x_ang", "y_ang", "z_ang"),
        ("rx", "ry", "rz"),
        ("pos_x", "pos_y", "pos_z"),
    ]
    for a, b, c in candidates:
        if a in lower and b in lower and c in lower:
            return lower[a], lower[b], lower[c]
    # Fallback: any fields ending in x/y/z-like order.
    raise ValueError(f"Could not identify coordinate columns in CSV fields: {names}")


def read_sites_csv(path: str) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        xcol, ycol, zcol = _find_xyz_columns(reader.fieldnames)
        coords = []
        for row in reader:
            try:
                coords.append([float(row[xcol]), float(row[ycol]), float(row[zcol])])
            except Exception as exc:
                raise ValueError(f"Failed reading coordinates from {path}: row={row}") from exc
            rows.append(row)
    return np.asarray(coords, dtype=float), rows


def wrap_positions(pos: np.ndarray, box: Box) -> np.ndarray:
    L = box.lengths
    o = box.origin
    return ((pos - o) % L) + o


def build_periodic_tree(pos: np.ndarray, box: Box) -> cKDTree:
    # cKDTree with boxsize requires coordinates in [0, L).
    scaled = (pos - box.origin) % box.lengths
    return cKDTree(scaled, boxsize=box.lengths)


def periodic_query_nearest(query: np.ndarray, tree: cKDTree, box: Box) -> Tuple[np.ndarray, np.ndarray]:
    q = (query - box.origin) % box.lengths
    return tree.query(q, k=1)


def periodic_query_pairs_count(a: np.ndarray, b_tree: cKDTree, box: Box, radius: float) -> int:
    aq = (a - box.origin) % box.lengths
    lists = b_tree.query_ball_point(aq, r=radius)
    return int(sum(len(x) for x in lists))


def percentile_safe(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def write_overlap_diagnostics(prefix: str, thresholds: List[float], bulk_pos: np.ndarray, gb_pos: np.ndarray, box: Box) -> List[Dict[str, float]]:
    bulk_tree = build_periodic_tree(bulk_pos, box)
    gb_tree = build_periodic_tree(gb_pos, box)

    # Measure both directions. bulk->GB controls which bulk sites will be
    # removed; GB->bulk is diagnostic and helps tune the exclusion cutoff.
    d_bulk_to_gb, _ = periodic_query_nearest(bulk_pos, gb_tree, box)
    d_gb_to_bulk, _ = periodic_query_nearest(gb_pos, bulk_tree, box)

    print("\nOverlap diagnostics before removal:", flush=True)
    print("threshold_A  bulk_sites_near_GB  gb_sites_near_bulk  bulk_GB_pairs", flush=True)
    rows: List[Dict[str, float]] = []
    for t in thresholds:
        n_bulk = int(np.count_nonzero(d_bulk_to_gb <= t))
        n_gb = int(np.count_nonzero(d_gb_to_bulk <= t))
        n_pairs = periodic_query_pairs_count(bulk_pos, gb_tree, box, t)
        print(f"{t:10.3f}  {n_bulk:18d}  {n_gb:18d}  {n_pairs:13d}", flush=True)
        rows.append({
            "threshold_A": t,
            "bulk_sites_near_GB": n_bulk,
            "gb_sites_near_bulk": n_gb,
            "bulk_GB_pairs": n_pairs,
        })

    print("\nNearest GB distance for bulk sites:", flush=True)
    print(f"  min  = {np.min(d_bulk_to_gb):.4f} Å", flush=True)
    print(f"  p01  = {percentile_safe(d_bulk_to_gb, 1):.4f} Å", flush=True)
    print(f"  p05  = {percentile_safe(d_bulk_to_gb, 5):.4f} Å", flush=True)
    print(f"  p50  = {percentile_safe(d_bulk_to_gb, 50):.4f} Å", flush=True)
    print(f"  p95  = {percentile_safe(d_bulk_to_gb, 95):.4f} Å", flush=True)

    print("\nNearest bulk distance for GB sites:", flush=True)
    print(f"  min  = {np.min(d_gb_to_bulk):.4f} Å", flush=True)
    print(f"  p01  = {percentile_safe(d_gb_to_bulk, 1):.4f} Å", flush=True)
    print(f"  p05  = {percentile_safe(d_gb_to_bulk, 5):.4f} Å", flush=True)
    print(f"  p50  = {percentile_safe(d_gb_to_bulk, 50):.4f} Å", flush=True)
    print(f"  p95  = {percentile_safe(d_gb_to_bulk, 95):.4f} Å", flush=True)

    with open(f"{prefix}_overlap_diagnostics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold_A", "bulk_sites_near_GB", "gb_sites_near_bulk", "bulk_GB_pairs"])
        writer.writeheader()
        writer.writerows(rows)

    # Also write nearest-distance distributions for later plotting if desired.
    with open(f"{prefix}_nearest_distances.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site_family", "site_index", "nearest_other_family_distance_A"])
        for i, d in enumerate(d_bulk_to_gb, start=1):
            writer.writerow(["bulk_to_gb", i, f"{d:.10f}"])
        for i, d in enumerate(d_gb_to_bulk, start=1):
            writer.writerow(["gb_to_bulk", i, f"{d:.10f}"])

    return rows


def write_sites_csv(path: str, pos: np.ndarray, types: np.ndarray, labels: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site_id", "x", "y", "z", "site_type", "site_label"])
        for i, (r, t, lab) in enumerate(zip(pos, types, labels), start=1):
            writer.writerow([i, f"{r[0]:.10f}", f"{r[1]:.10f}", f"{r[2]:.10f}", int(t), lab])


def write_lammps_sites(path: str, pos: np.ndarray, types: np.ndarray, box: Box) -> None:
    n = len(pos)
    unique_types = sorted(set(int(x) for x in types))
    with open(path, "w") as f:
        f.write("LAMMPS data file: unified KMC sites only\n\n")
        f.write(f"{n} atoms\n")
        f.write(f"{len(unique_types)} atom types\n\n")
        f.write(f"{box.xlo:.10f} {box.xhi:.10f} xlo xhi\n")
        f.write(f"{box.ylo:.10f} {box.yhi:.10f} ylo yhi\n")
        f.write(f"{box.zlo:.10f} {box.zhi:.10f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t in unique_types:
            # Dummy masses for OVITO visualization.
            f.write(f"{t} 1.0\n")
        f.write("\nAtoms # atomic\n\n")
        for i, (r, t) in enumerate(zip(pos, types), start=1):
            f.write(f"{i} {int(t)} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n")


def write_extended_xyz(path: str, pos: np.ndarray, types: np.ndarray, box: Box) -> None:
    L = box.lengths
    # Orthorhombic only, matching the user's current LAMMPS boxes.
    lattice = f'{L[0]:.10f} 0 0 0 {L[1]:.10f} 0 0 0 {L[2]:.10f}'
    species = {1: "Bt", 2: "Gb"}
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        f.write(f'Lattice="{lattice}" Origin="{box.xlo:.10f} {box.ylo:.10f} {box.zlo:.10f}" Properties=species:S:1:pos:R:3:site_type:I:1 pbc="T T T"\n')
        for r, t in zip(pos, types):
            f.write(f"{species.get(int(t), 'X')} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f} {int(t)}\n")


def main() -> None:
    args = parse_args()
    thresholds = [float(x) for x in args.diagnostic_thresholds.split(",") if x.strip()]

    print("Stage 3 merge with diagnostics, no final dedup", flush=True)
    # Stage 3 only needs the host for the periodic box. The Fe atoms themselves
    # are not used because the site coordinates were already generated/relaxed
    # in Stages 1 and 2.
    box = read_lammps_box(args.host_lmp)
    bulk_pos, _ = read_sites_csv(args.bulk_sites)
    gb_pos, _ = read_sites_csv(args.gb_sites)
    # Normalize all sites into the host box before any PBC queries. This keeps
    # anchored Stage-2 outputs and ordinary wrapped outputs equivalent.
    bulk_pos = wrap_positions(bulk_pos, box)
    gb_pos = wrap_positions(gb_pos, box)

    print(f"Host:       {args.host_lmp}", flush=True)
    print(f"Bulk sites: {args.bulk_sites}  count={len(bulk_pos)}", flush=True)
    print(f"GB sites:   {args.gb_sites}  count={len(gb_pos)}", flush=True)
    print(f"Box lengths: {box.lengths}", flush=True)

    write_overlap_diagnostics(args.out_prefix, thresholds, bulk_pos, gb_pos, box)

    gb_tree = build_periodic_tree(gb_pos, box)
    d_bulk_to_gb, _ = periodic_query_nearest(bulk_pos, gb_tree, box)
    # The only destructive cleanup here is bulk-vs-GB overlap removal. Relaxed
    # GB sites win over analytical bulk tetra sites in the interface region.
    remove_bulk = d_bulk_to_gb <= args.bulk_gb_exclusion
    n_remove = int(np.count_nonzero(remove_bulk))
    n_keep = int(len(bulk_pos) - n_remove)

    print(f"\nRemoving bulk sites within {args.bulk_gb_exclusion:.3f} Å of GB sites: {n_remove}", flush=True)
    print(f"Keeping bulk sites: {n_keep}", flush=True)
    print(f"Keeping GB sites:   {len(gb_pos)}", flush=True)

    clean_bulk = bulk_pos[~remove_bulk]

    unified_pos = np.vstack([clean_bulk, gb_pos]) if len(clean_bulk) else gb_pos.copy()
    unified_types = np.concatenate([
        np.ones(len(clean_bulk), dtype=int),
        np.full(len(gb_pos), 2, dtype=int),
    ])
    labels = ["BULK_TETRA"] * len(clean_bulk) + ["GB_RELAXED"] * len(gb_pos)

    unified_pos = wrap_positions(unified_pos, box)

    print("\nFinal unified site counts:", flush=True)
    print(f"  BULK_TETRA = {len(clean_bulk)}", flush=True)
    print(f"  GB_RELAXED = {len(gb_pos)}", flush=True)
    print(f"  TOTAL      = {len(unified_pos)}", flush=True)
    # Keep this intentionally off. A blind all-site dedup can remove physically
    # meaningful GB minima that happen to lie near bulk-template positions; the
    # explicit bulk_gb_exclusion above is the controlled interface cleanup.
    print("  final all-site deduplication: skipped intentionally", flush=True)

    write_sites_csv(f"{args.out_prefix}_unified_sites.csv", unified_pos, unified_types, labels)
    write_lammps_sites(f"{args.out_prefix}_unified_sites.lmp", unified_pos, unified_types, box)
    write_extended_xyz(f"{args.out_prefix}_unified_sites.xyz", unified_pos, unified_types, box)
    np.savez_compressed(
        f"{args.out_prefix}_unified_sites.npz",
        positions=unified_pos,
        site_types=unified_types,
        box_origin=box.origin,
        box_lengths=box.lengths,
        bulk_gb_exclusion=args.bulk_gb_exclusion,
        n_bulk_original=len(bulk_pos),
        n_gb_original=len(gb_pos),
        n_bulk_removed=n_remove,
        n_bulk_kept=len(clean_bulk),
    )

    with open(f"{args.out_prefix}_summary.txt", "w") as f:
        f.write("Stage 3 merge with diagnostics, no final dedup\n")
        f.write(f"Host: {args.host_lmp}\n")
        f.write(f"Bulk sites file: {args.bulk_sites}\n")
        f.write(f"GB sites file: {args.gb_sites}\n")
        f.write(f"Original bulk sites: {len(bulk_pos)}\n")
        f.write(f"Original GB sites: {len(gb_pos)}\n")
        f.write(f"bulk_gb_exclusion_A: {args.bulk_gb_exclusion:.6f}\n")
        f.write(f"Bulk sites removed by GB overlap: {n_remove}\n")
        f.write(f"Bulk sites kept: {len(clean_bulk)}\n")
        f.write(f"GB sites kept: {len(gb_pos)}\n")
        f.write(f"Unified total sites: {len(unified_pos)}\n")
        f.write("Final all-site deduplication: skipped intentionally\n")
        f.write("Output type 1 = BULK_TETRA\n")
        f.write("Output type 2 = GB_RELAXED\n")

    print("\nWrote:", flush=True)
    print(f"  {args.out_prefix}_overlap_diagnostics.csv", flush=True)
    print(f"  {args.out_prefix}_nearest_distances.csv", flush=True)
    print(f"  {args.out_prefix}_unified_sites.csv", flush=True)
    print(f"  {args.out_prefix}_unified_sites.xyz", flush=True)
    print(f"  {args.out_prefix}_unified_sites.lmp", flush=True)
    print(f"  {args.out_prefix}_unified_sites.npz", flush=True)
    print(f"  {args.out_prefix}_summary.txt", flush=True)


if __name__ == "__main__":
    main()
