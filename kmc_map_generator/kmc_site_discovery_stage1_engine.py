#!/usr/bin/env python3
"""
kmc_site_discovery_stage1_engine.py

Unified stage-1 site discovery driver for Fe bicrystal/polycrystal KMC.

This script intentionally stops BEFORE LAMMPS relaxation. It builds the files
needed for the next MPI/local-sphere relaxation stage:

  1. Generic grain/GB classifier, no predefined GB plane.
  2. Fast bulk BCC tetrahedral site map with proper simulation cell metadata.
  3. Full unresolved-region Voronoi/transition candidate generation.
  4. Optional coordinate-band filtering for cases where the active GB width is
     known from convergence/relaxation diagnostics.

It is a clean orchestrator around the working modules produced during the
workflow. Keep the module scripts in the same directory, or pass --script-dir.

Required companion scripts:
  - gb_region_classifier_grain_graph_fast.py
  - bulk_tetra_template_mapper_fast_cell.py
  - gb_transition_voronoi_candidates.py

Example:
  python3 kmc_site_discovery_stage1_engine.py sigma5_210-20-20-5.lmp \
      --a0 2.856 \
      --out-prefix sigma5_stage1

Optional active-band filter, e.g. only keep GB candidates in x=[119.371,136.077]:
  python3 kmc_site_discovery_stage1_engine.py sigma5_210-20-20-5.lmp \
      --a0 2.856 \
      --focus-axis x --focus-range 119.371,136.077 \
      --out-prefix sigma5_stage1_focus

Outputs:
  <prefix>_regions_*                   classifier outputs
  <prefix>_bulk_tetra_*                bulk tetra outputs
  <prefix>_gb_candidates.*             full GB/transition candidates
  <prefix>_gb_candidates_focus.*       optional active-band candidates
  <prefix>_stage1_manifest.txt         reproducibility manifest
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

AXIS_TO_COL = {"x": 0, "y": 1, "z": 2}


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_range(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be min,max")
    a, b = float(parts[0]), float(parts[1])
    if b < a:
        a, b = b, a
    return a, b


def read_lammps_box(path: Path) -> np.ndarray:
    box = np.zeros((3, 2), dtype=float)
    for line in path.read_text().splitlines()[:200]:
        p = line.split()
        if len(p) >= 4 and p[2] == "xlo" and p[3] == "xhi":
            box[0] = [float(p[0]), float(p[1])]
        elif len(p) >= 4 and p[2] == "ylo" and p[3] == "yhi":
            box[1] = [float(p[0]), float(p[1])]
        elif len(p) >= 4 and p[2] == "zlo" and p[3] == "zhi":
            box[2] = [float(p[0]), float(p[1])]
    if np.any(box[:, 1] <= box[:, 0]):
        raise ValueError(f"Could not parse orthogonal box from {path}")
    return box


def wrap_positions(pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    lo = box[:, 0]
    L = box[:, 1] - box[:, 0]
    return lo + np.mod(pos - lo, L)


def load_candidate_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for col in ("x", "y", "z"):
        if col not in fieldnames:
            raise ValueError(f"{path} missing required coordinate column '{col}'")
    return rows, fieldnames


def write_candidate_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_extended_xyz(path: Path, rows: List[Dict[str, str]], box: np.ndarray, symbol_by_class: bool = True) -> None:
    L = box[:, 1] - box[:, 0]
    lattice = f'{L[0]:.10f} 0 0 0 {L[1]:.10f} 0 0 0 {L[2]:.10f}'
    with path.open("w") as f:
        f.write(f"{len(rows)}\n")
        f.write(f'Lattice="{lattice}" Origin="{box[0,0]:.10f} {box[1,0]:.10f} {box[2,0]:.10f}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for r in rows:
            cls = str(r.get("candidate_class", r.get("class", "GB_CORE"))).upper()
            sym = "He" if symbol_by_class and "GB" in cls else "H"
            f.write(f"{sym} {float(r['x']):.10f} {float(r['y']):.10f} {float(r['z']):.10f}\n")


def write_lammps_sites(path: Path, rows: List[Dict[str, str]], box: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("LAMMPS data file - filtered site candidates\n\n")
        f.write(f"{len(rows)} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"{box[0,0]:.10f} {box[0,1]:.10f} xlo xhi\n")
        f.write(f"{box[1,0]:.10f} {box[1,1]:.10f} ylo yhi\n")
        f.write(f"{box[2,0]:.10f} {box[2,1]:.10f} zlo zhi\n\n")
        f.write("Atoms # atomic\n\n")
        for i, r in enumerate(rows, start=1):
            cls = str(r.get("candidate_class", r.get("class", "GB_CORE"))).upper()
            typ = 1 if "GB" in cls else 2
            f.write(f"{i} {typ} {float(r['x']):.10f} {float(r['y']):.10f} {float(r['z']):.10f}\n")


def write_npz(path: Path, rows: List[Dict[str, str]], box: np.ndarray) -> None:
    coords = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    labels = np.array([str(r.get("candidate_class", r.get("class", "UNKNOWN"))) for r in rows])
    ids = np.array([int(float(r.get("candidate_id", r.get("id", i)))) for i, r in enumerate(rows)], dtype=int)
    np.savez(path, positions=coords, labels=labels, candidate_ids=ids, box=box)


def filter_candidates_by_band(
    csv_in: Path,
    data_file: Path,
    out_prefix: Path,
    axis: str,
    span: Tuple[float, float],
    padding: float,
) -> Path:
    rows, fieldnames = load_candidate_csv(csv_in)
    box = read_lammps_box(data_file)
    axis_col = axis
    lo, hi = span[0] - padding, span[1] + padding
    kept: List[Dict[str, str]] = []
    for r in rows:
        v = float(r[axis_col])
        if lo <= v <= hi:
            kept.append(r)
    csv_out = out_prefix.with_suffix(".csv")
    xyz_out = out_prefix.with_suffix(".xyz")
    lmp_out = out_prefix.with_suffix(".lmp")
    npz_out = out_prefix.with_suffix(".npz")
    write_candidate_csv(csv_out, kept, fieldnames)
    write_extended_xyz(xyz_out, kept, box)
    write_lammps_sites(lmp_out, kept, box)
    write_npz(npz_out, kept, box)
    print(f"Filtered candidates by {axis} in [{lo:.6f}, {hi:.6f}]: {len(kept)}/{len(rows)} kept")
    print(f"Wrote: {csv_out}, {xyz_out}, {lmp_out}, {npz_out}")
    return csv_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified stage-1 Fe GB/polycrystal site discovery engine before LAMMPS relaxation."
    )
    ap.add_argument("data_file", help="LAMMPS Fe host structure")
    ap.add_argument("--script-dir", default=None, help="Directory containing companion scripts. Default: same directory as this engine")
    ap.add_argument("--python", default=sys.executable, help="Python executable for companion scripts")
    ap.add_argument("--a0", type=float, default=2.856)
    ap.add_argument("--fe-type", type=int, default=1)
    ap.add_argument("--out-prefix", default="kmc_stage1")

    # Region classifier settings selected from the working path.
    ap.add_argument("--orientation-cutoff-deg", type=float, default=8.0)
    ap.add_argument("--gb-buffer", type=float, default=6.0)
    ap.add_argument("--transition-buffer", type=float, default=9.0)
    ap.add_argument("--max-frame-error", type=float, default=0.20)

    # GB/Voronoi candidate generation settings selected from the working path.
    ap.add_argument("--local-radius", type=float, default=5.0)
    ap.add_argument("--collect-radius", type=float, default=3.2)
    ap.add_argument("--min-fe-dist", type=float, default=1.05)
    ap.add_argument("--max-fe-dist", type=float, default=2.35)
    ap.add_argument("--region-support-radius", type=float, default=3.0)
    ap.add_argument("--merge-radius", type=float, default=0.25)
    ap.add_argument("--seed-stride", type=int, default=1)

    # Optional active band for your observation that only a narrow GB strip changes.
    ap.add_argument("--focus-axis", choices=["x", "y", "z"], default=None)
    ap.add_argument("--focus-range", type=parse_range, default=None, help="Optional min,max coordinate range for GB candidates")
    ap.add_argument("--focus-padding", type=float, default=0.0)

    # Recommended next-stage local-sphere settings recorded in manifest only.
    ap.add_argument("--recommended-local-fe-radius", type=float, default=10.0)
    ap.add_argument("--recommended-cluster-radius", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_file = Path(args.data_file).resolve()
    require_file(data_file, "input LAMMPS data file")

    script_dir = Path(args.script_dir).resolve() if args.script_dir else Path(__file__).resolve().parent
    classifier = script_dir / "gb_region_classifier_grain_graph_fast.py"
    bulk_mapper = script_dir / "bulk_tetra_template_mapper_fast_cell.py"
    gb_voronoi = script_dir / "gb_transition_voronoi_candidates.py"
    require_file(classifier, "classifier script")
    require_file(bulk_mapper, "bulk tetra mapper script")
    require_file(gb_voronoi, "GB Voronoi candidate script")

    prefix = Path(args.out_prefix)
    region_prefix = Path(f"{prefix}_regions")
    bulk_prefix = Path(f"{prefix}_bulk_tetra")
    gb_prefix = Path(f"{prefix}_gb_candidates")

    # Stage 1A: classify the Fe host first. The resulting mask file is the
    # shared contract for both downstream candidate generators:
    #   - bulk mapper consumes BULK_TEMPLATE atoms
    #   - Voronoi mapper consumes TRANSITION/GB atoms
    cmd_region = [
        args.python, str(classifier), str(data_file),
        "--fe-type", str(args.fe_type),
        "--a0", str(args.a0),
        "--orientation-cutoff-deg", str(args.orientation_cutoff_deg),
        "--gb-buffer", str(args.gb_buffer),
        "--transition-buffer", str(args.transition_buffer),
        "--max-frame-error", str(args.max_frame_error),
        "--out-prefix", str(region_prefix),
    ]
    run_cmd(cmd_region, args.dry_run)

    masks = Path(f"{region_prefix}_masks.npz")
    if not args.dry_run:
        require_file(masks, "region masks")

    # Stage 1B: build analytical tetrahedral sites only in bulk-like regions.
    # These sites are not relaxed here because they are already standard BCC
    # tetra sites and will later be merged with relaxed GB sites.
    cmd_bulk = [
        args.python, str(bulk_mapper), str(data_file),
        "--masks", str(masks),
        "--a0", str(args.a0),
        "--out-prefix", str(bulk_prefix),
    ]
    run_cmd(cmd_bulk, args.dry_run)

    # Stage 1C: generate unresolved-interface candidates with local Voronoi
    # patches. These are geometric guesses and should be validated/relaxed in
    # Stage 2 before becoming KMC sites.
    cmd_gb = [
        args.python, str(gb_voronoi), str(data_file),
        "--masks", str(masks),
        "--local-radius", str(args.local_radius),
        "--collect-radius", str(args.collect_radius),
        "--min-fe-dist", str(args.min_fe_dist),
        "--max-fe-dist", str(args.max_fe_dist),
        "--region-support-radius", str(args.region_support_radius),
        "--merge-radius", str(args.merge_radius),
        "--seed-stride", str(args.seed_stride),
        "--out-prefix", str(gb_prefix),
    ]
    run_cmd(cmd_gb, args.dry_run)

    focus_csv = None
    if args.focus_axis and args.focus_range and not args.dry_run:
        # Optional convenience path for systems where the active GB band is
        # already known from inspection or convergence tests. This does not
        # change the full GB candidate output; it writes a filtered sibling.
        focus_prefix = Path(f"{prefix}_gb_candidates_focus")
        focus_csv = filter_candidates_by_band(
            Path(f"{gb_prefix}.csv"), data_file, focus_prefix, args.focus_axis, args.focus_range, args.focus_padding
        )
    elif args.focus_axis or args.focus_range:
        print("Note: --focus-axis and --focus-range must both be provided to filter candidates.")

    manifest = Path(f"{prefix}_stage1_manifest.txt")
    if not args.dry_run:
        # Keep a plain-text manifest next to the generated files so the exact
        # classifier/Voronoi parameters can be reproduced after long relax runs.
        manifest.write_text(
            f"""KMC site discovery stage-1 manifest
Input structure: {data_file}
Output prefix: {prefix}

a0: {args.a0}
fe_type: {args.fe_type}

Region classifier:
  script: {classifier}
  orientation_cutoff_deg: {args.orientation_cutoff_deg}
  gb_buffer: {args.gb_buffer}
  transition_buffer: {args.transition_buffer}
  max_frame_error: {args.max_frame_error}
  masks: {masks}

Bulk tetra map:
  script: {bulk_mapper}
  outputs: {bulk_prefix}_sites.csv/.xyz/.lmp/.npz

GB/transition candidates:
  script: {gb_voronoi}
  local_radius: {args.local_radius}
  collect_radius: {args.collect_radius}
  min_fe_dist: {args.min_fe_dist}
  max_fe_dist: {args.max_fe_dist}
  region_support_radius: {args.region_support_radius}
  merge_radius: {args.merge_radius}
  seed_stride: {args.seed_stride}
  full outputs: {gb_prefix}.csv/.xyz/.lmp/.npz

Optional focus filter:
  focus_axis: {args.focus_axis}
  focus_range: {args.focus_range}
  focus_padding: {args.focus_padding}
  focus_csv: {focus_csv}

Recommended next-stage local-sphere LAMMPS relaxation:
  local_fe_radius: {args.recommended_local_fe_radius}
  cluster_radius: {args.recommended_cluster_radius}
  note: positions only need to be near metastable minima because NEB will re-minimize endpoints.
"""
        )
        print(f"\nWrote manifest: {manifest}")

    print("\nStage-1 complete.")
    print(f"Bulk sites: {bulk_prefix}_sites.lmp")
    print(f"GB candidates: {gb_prefix}.lmp")
    if focus_csv:
        print(f"Focused GB candidates: {Path(str(focus_csv)).with_suffix('.lmp')}")


if __name__ == "__main__":
    main()
