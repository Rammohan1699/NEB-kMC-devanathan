#!/usr/bin/env python3
"""Project Fe-atom region labels onto KMC interstitial sites.

The site-discovery classifier labels Fe host atoms as bulk, transition, or
grain-boundary regions, while the KMC driver places H atoms on interstitial site
ids. This tool bridges those coordinate systems by assigning each KMC site to
the nearest Fe atom under periodic boundary conditions and writing site-level
region metadata next to the KMC map inputs.

The generated CSV/NPZ outputs are consumed by the driver and post-processing
tools for region-aware initial H placement and region/grain transition
summaries.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REGION_NAME_BY_CODE = {
    1: "bulk",
    2: "transition",
    3: "grain_boundary",
}


def _load_site_map(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    positions = np.asarray(data["positions"], dtype=float)
    site_types = np.asarray(data["site_types"], dtype=int) if "site_types" in data.files else None
    box = np.asarray(data["box_lengths"], dtype=float)
    origin = np.asarray(data["box_origin"], dtype=float) if "box_origin" in data.files else np.zeros(3)
    return np.mod(positions - origin, box), site_types, box


def _load_lammps_atomic_positions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.zeros(3, dtype=float)
    hi = np.zeros(3, dtype=float)
    rows: list[tuple[int, np.ndarray]] = []
    in_atoms = False
    saw_atoms_header = False

    with path.open("r", encoding="utf-8") as fh:
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
        raise ValueError(f"No Atoms # atomic rows found in {path}")
    box = hi - origin
    if np.any(box <= 0.0):
        raise ValueError(f"Invalid or missing box bounds in {path}: {box}")
    atom_types = np.asarray([atype for atype, _pos in rows], dtype=int)
    positions = np.mod(np.asarray([pos for _atype, pos in rows], dtype=float), box)
    return positions, atom_types, box


def _load_fe_region_arrays(mask_path: Path, n_host_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    masks = np.load(mask_path, allow_pickle=True)
    region_code = np.zeros(n_host_atoms, dtype=np.int16)
    grain_id = np.full(n_host_atoms, -1, dtype=np.int32)

    if "region_code_all" in masks.files:
        region_code = np.asarray(masks["region_code_all"], dtype=np.int16)
    elif {"fe_indices_global", "fe_bulk_template", "fe_transition", "fe_gb_voronoi"}.issubset(masks.files):
        fe_indices = np.asarray(masks["fe_indices_global"], dtype=int)
        region_code[fe_indices[np.asarray(masks["fe_bulk_template"], dtype=bool)]] = 1
        region_code[fe_indices[np.asarray(masks["fe_transition"], dtype=bool)]] = 2
        region_code[fe_indices[np.asarray(masks["fe_gb_voronoi"], dtype=bool)]] = 3
    else:
        raise ValueError(f"{mask_path} does not contain recognized Fe region masks")

    if "grain_id_all" in masks.files:
        grain_id = np.asarray(masks["grain_id_all"], dtype=np.int32)
    elif {"fe_indices_global", "fe_grain_id"}.issubset(masks.files):
        fe_indices = np.asarray(masks["fe_indices_global"], dtype=int)
        grain_id[fe_indices] = np.asarray(masks["fe_grain_id"], dtype=np.int32)

    if len(region_code) != n_host_atoms:
        raise ValueError(
            f"Region mask length {len(region_code)} does not match host atom count {n_host_atoms}"
        )
    if len(grain_id) != n_host_atoms:
        raise ValueError(f"Grain-id length {len(grain_id)} does not match host atom count {n_host_atoms}")
    return region_code, grain_id


def _selection_label(region_code: int, grain_id: int) -> str:
    if region_code == 1 and grain_id >= 0:
        return f"bulk_grain_{grain_id}"
    return REGION_NAME_BY_CODE.get(int(region_code), "unknown")


def _write_csv(
    path: Path,
    *,
    site_positions: np.ndarray,
    site_types: np.ndarray | None,
    site_region_code: np.ndarray,
    site_grain_id: np.ndarray,
    site_selection_label: np.ndarray,
    nearest_fe_atom_index: np.ndarray,
    nearest_fe_dist: np.ndarray,
) -> None:
    fields = [
        "site_id",
        "x",
        "y",
        "z",
        "site_type",
        "region_code",
        "region_name",
        "grain_id",
        "selection_label",
        "nearest_fe_atom_index",
        "nearest_fe_dist",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, pos in enumerate(site_positions):
            code = int(site_region_code[i])
            writer.writerow(
                {
                    "site_id": i,
                    "x": f"{pos[0]:.10f}",
                    "y": f"{pos[1]:.10f}",
                    "z": f"{pos[2]:.10f}",
                    "site_type": "" if site_types is None else int(site_types[i]),
                    "region_code": code,
                    "region_name": REGION_NAME_BY_CODE.get(code, "unknown"),
                    "grain_id": int(site_grain_id[i]),
                    "selection_label": str(site_selection_label[i]),
                    "nearest_fe_atom_index": int(nearest_fe_atom_index[i]),
                    "nearest_fe_dist": f"{nearest_fe_dist[i]:.8f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", default="Site_discovery/sigma5_regions_grain_masks.npz")
    parser.add_argument("--site-map", default="kmc_map_inputs/sigma5_stage3_unified_sites.npz")
    parser.add_argument("--host", default="kmc_map_inputs/sigma5_210-20-20-5.lmp")
    parser.add_argument("--out-dir", default="kmc_map_inputs")
    parser.add_argument("--out-prefix", default="sigma5_site_regions")
    parser.add_argument("--copy-masks", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    mask_path = Path(args.masks)
    site_map_path = Path(args.site_map)
    host_path = Path(args.host)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    site_positions, site_types, site_box = _load_site_map(site_map_path)
    host_positions, _host_types, host_box = _load_lammps_atomic_positions(host_path)
    box = np.asarray(site_box if site_box is not None else host_box, dtype=float)
    if not np.allclose(box, host_box, atol=1.0e-6):
        raise ValueError(f"Site-map box {box.tolist()} does not match host box {host_box.tolist()}")

    fe_region_code, fe_grain_id = _load_fe_region_arrays(mask_path, len(host_positions))
    fe_positions = np.mod(host_positions, box)
    site_positions = np.mod(np.asarray(site_positions, dtype=float), box)

    tree = cKDTree(fe_positions, boxsize=box)
    nearest_fe_dist, nearest_fe_atom_index = tree.query(site_positions, k=1)
    site_region_code = fe_region_code[nearest_fe_atom_index]
    site_grain_id = fe_grain_id[nearest_fe_atom_index]
    site_selection_label = np.asarray(
        [_selection_label(int(code), int(gid)) for code, gid in zip(site_region_code, site_grain_id)],
        dtype=object,
    )

    out_npz = out_dir / f"{args.out_prefix}.npz"
    out_csv = out_dir / f"{args.out_prefix}.csv"
    np.savez(
        out_npz,
        site_region_code=site_region_code.astype(np.int16),
        site_region_name=np.asarray([REGION_NAME_BY_CODE.get(int(c), "unknown") for c in site_region_code], dtype=object),
        site_grain_id=site_grain_id.astype(np.int32),
        site_selection_label=site_selection_label,
        nearest_fe_atom_index=nearest_fe_atom_index.astype(np.int32),
        nearest_fe_dist=nearest_fe_dist.astype(float),
        region_code_legend=np.asarray(["1=bulk", "2=transition", "3=grain_boundary"], dtype=object),
        source_masks=str(mask_path),
        source_site_map=str(site_map_path),
        source_host=str(host_path),
        box_lengths=box,
    )
    _write_csv(
        out_csv,
        site_positions=site_positions,
        site_types=site_types,
        site_region_code=site_region_code,
        site_grain_id=site_grain_id,
        site_selection_label=site_selection_label,
        nearest_fe_atom_index=nearest_fe_atom_index,
        nearest_fe_dist=nearest_fe_dist,
    )

    if args.copy_masks:
        shutil.copy2(mask_path, out_dir / mask_path.name)

    labels, counts = np.unique(site_selection_label.astype(str), return_counts=True)
    print(f"Wrote {out_npz}")
    print(f"Wrote {out_csv}")
    if args.copy_masks:
        print(f"Copied {mask_path} -> {out_dir / mask_path.name}")
    print("Selection labels:")
    for label, count in zip(labels, counts):
        print(f"  {label}: {int(count)}")


if __name__ == "__main__":
    main()
