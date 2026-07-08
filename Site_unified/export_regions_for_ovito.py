#!/usr/bin/env python3
"""
Export region/grain classification masks to an OVITO-readable LAMMPS data file.

This script reads:
  1. Original LAMMPS data file containing Fe atoms
  2. *_masks.npz produced by the GB/bulk classifier

and writes a LAMMPS atomic-style data file where the atom type encodes region:

  type 1 = BULK_TEMPLATE
  type 2 = TRANSITION
  type 3 = GB_VORONOI
  type 4 = OTHER / unclassified

It also writes a CSV containing atom id, position, region label, region id, and grain id.

In OVITO:
  - Load the output .lmp file
  - Color by Particle Type
  - type 1 bulk, type 2 transition, type 3 GB/Voronoi

Usage:
  python3 export_regions_for_ovito.py sigma5_210-20-20-5.lmp \
      --masks sigma5_regions_grain_masks.npz \
      --out-prefix sigma5_regions_ovito
"""

import argparse
import csv
from pathlib import Path
import numpy as np

from lammps_data_utils import read_lammps_atomic_data

REGION_NAMES = {
    1: "BULK_TEMPLATE",
    2: "TRANSITION",
    3: "GB_VORONOI",
    4: "OTHER",
}


def parse_lammps_data(path):
    data = read_lammps_atomic_data(path)
    return data.ids, data.types, data.pos, data.box


def get_mask_array(npz, possible_names, n, default=False):
    for name in possible_names:
        if name in npz:
            arr = npz[name]
            if arr.dtype == bool and arr.shape[0] == n:
                return arr.copy()
            # Some scripts store global indices instead of bool masks.
            if arr.dtype != bool:
                mask = np.zeros(n, dtype=bool)
                idx = arr.astype(int)
                idx = idx[(idx >= 0) & (idx < n)]
                mask[idx] = True
                return mask
    return np.full(n, default, dtype=bool)


def load_masks(mask_path, n_atoms):
    npz = np.load(mask_path, allow_pickle=True)
    keys = set(npz.files)

    # Region masks. Support several historical names from our scripts.
    bulk = get_mask_array(npz, ["fe_bulk_template", "bulk_template_mask", "bulk_mask", "is_bulk_template"], n_atoms)
    trans = get_mask_array(npz, ["fe_transition", "transition_mask", "is_transition"], n_atoms)
    gb = get_mask_array(npz, ["fe_gb_voronoi", "gb_voronoi_mask", "gb_mask", "is_gb_voronoi"], n_atoms)

    # If masks are only stored for Fe atoms, map them through fe_indices_global.
    if "fe_indices_global" in keys:
        fe_idx = npz["fe_indices_global"].astype(int)
        for name in ["fe_bulk_template", "fe_transition", "fe_gb_voronoi"]:
            if name in keys and npz[name].dtype == bool and npz[name].shape[0] == len(fe_idx):
                full = np.zeros(n_atoms, dtype=bool)
                full[fe_idx] = npz[name]
                if name == "fe_bulk_template": bulk = full
                elif name == "fe_transition": trans = full
                elif name == "fe_gb_voronoi": gb = full

    grain = np.full(n_atoms, -1, dtype=int)
    for name in ["grain_id", "grain_ids", "fe_grain_id", "fe_grain_ids"]:
        if name in keys:
            arr = npz[name].astype(int)
            if arr.shape[0] == n_atoms:
                grain = arr.copy()
            elif "fe_indices_global" in keys and arr.shape[0] == len(npz["fe_indices_global"]):
                grain[npz["fe_indices_global"].astype(int)] = arr
            break

    region_id = np.full(n_atoms, 4, dtype=int)
    # Priority: GB > TRANSITION > BULK, because overlaps should be conservatively non-bulk.
    region_id[bulk] = 1
    region_id[trans] = 2
    region_id[gb] = 3

    return region_id, grain, sorted(keys)


def write_lammps_atomic(out_path, atom_ids, xyz, box, region_id):
    with open(out_path, "w") as f:
        f.write("LAMMPS data file exported for OVITO region debugging\n\n")
        f.write(f"{len(atom_ids)} atoms\n")
        f.write("4 atom types\n\n")
        f.write(f"{box[0,0]:.12f} {box[0,1]:.12f} xlo xhi\n")
        f.write(f"{box[1,0]:.12f} {box[1,1]:.12f} ylo yhi\n")
        f.write(f"{box[2,0]:.12f} {box[2,1]:.12f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 55.845 # BULK_TEMPLATE\n")
        f.write("2 55.845 # TRANSITION\n")
        f.write("3 55.845 # GB_VORONOI\n")
        f.write("4 55.845 # OTHER\n\n")
        f.write("Atoms # atomic\n\n")
        for aid, rid, pos in zip(atom_ids, region_id, xyz):
            f.write(f"{aid:d} {rid:d} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}\n")


def write_csv(out_path, atom_ids, xyz, region_id, grain):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["atom_id", "x", "y", "z", "region_id", "region", "grain_id"])
        for aid, pos, rid, gid in zip(atom_ids, xyz, region_id, grain):
            w.writerow([aid, pos[0], pos[1], pos[2], int(rid), REGION_NAMES[int(rid)], int(gid)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lammps_data")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out-prefix", default="regions_ovito")
    args = ap.parse_args()

    atom_ids, atom_types, xyz, box = parse_lammps_data(args.lammps_data)
    region_id, grain, keys = load_masks(args.masks, len(atom_ids))

    out_lmp = f"{args.out_prefix}.lmp"
    out_csv = f"{args.out_prefix}.csv"
    out_xyz = f"{args.out_prefix}.xyz"

    write_lammps_atomic(out_lmp, atom_ids, xyz, box, region_id)
    write_csv(out_csv, atom_ids, xyz, region_id, grain)

    # Also write XYZ with species labels readable by simple viewers.
    labels = {1: "B", 2: "T", 3: "G", 4: "O"}
    with open(out_xyz, "w") as f:
        f.write(f"{len(atom_ids)}\n")
        f.write("B=BULK_TEMPLATE T=TRANSITION G=GB_VORONOI O=OTHER\n")
        for rid, pos in zip(region_id, xyz):
            f.write(f"{labels[int(rid)]} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

    print("Mask keys found:")
    for k in keys:
        print(f"  {k}")
    print("\nRegion counts:")
    for rid in [1, 2, 3, 4]:
        print(f"  {REGION_NAMES[rid]:14s}: {np.count_nonzero(region_id == rid)}")
    print("\nGrain counts, excluding grain=-1:")
    for gid in sorted(set(grain.tolist())):
        if gid >= 0:
            print(f"  grain {gid}: {np.count_nonzero(grain == gid)}")
    print("\nWrote:")
    print(f"  {out_lmp}")
    print(f"  {out_csv}")
    print(f"  {out_xyz}")
    print("\nOVITO coloring:")
    print("  type 1 = BULK_TEMPLATE")
    print("  type 2 = TRANSITION")
    print("  type 3 = GB_VORONOI")
    print("  type 4 = OTHER")


if __name__ == "__main__":
    main()
