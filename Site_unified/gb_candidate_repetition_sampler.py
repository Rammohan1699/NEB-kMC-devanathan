#!/usr/bin/env python3
"""
gb_candidate_repetition_sampler.py

Select one representative periodic tile from GB/transition candidate sites.

Purpose
-------
For periodic bicrystal/polycrystal cells, many geometrically discovered GB sites are
periodic repeats along directions tangent to the boundary/network. This script detects
(or accepts user-specified) translational repetitions and writes a reduced candidate
set for expensive LAMMPS H-relaxation validation.

Typical use
-----------
python3 gb_candidate_repetition_sampler.py sigma5_210-20-20-5.lmp \
  --sites sigma5_gb_voronoi_candidates.xyz \
  --normal-axis x \
  --max-reps 40 \
  --out-prefix sigma5_gb_tile

Then validate only:
python3 relax_filter_interstitial_sites.py sigma5_210-20-20-5.lmp \
  --sites sigma5_gb_tile_representative_candidates.csv \
  --mode lammps ...

Notes
-----
- This does NOT prove energetic equivalence; it gives a representative periodic tile.
- For bicrystal GBs, use --normal-axis x/y/z if known. If omitted, it tries to infer.
- You can override automatic periods with --tile-reps or --tile-lengths.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from lammps_data_utils import read_lammps_box_bounds

AXES = {"x": 0, "y": 1, "z": 2}
AXIS_NAMES = ["x", "y", "z"]


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    L: np.ndarray


def read_lammps_box(path: str) -> Box:
    bounds = read_lammps_box_bounds(path)
    lo=bounds[:,0].copy(); hi=bounds[:,1].copy()
    return Box(lo=lo, hi=hi, L=hi-lo)


def parse_extxyz_lattice(comment: str) -> Optional[Box]:
    m=re.search(r'Lattice="([^"]+)"', comment)
    if not m:
        return None
    vals=[float(x) for x in m.group(1).split()]
    if len(vals)!=9:
        return None
    mat=np.array(vals,float).reshape(3,3)
    L=np.array([np.linalg.norm(mat[0]),np.linalg.norm(mat[1]),np.linalg.norm(mat[2])])
    origin=np.zeros(3)
    mo=re.search(r'Origin="([^"]+)"', comment)
    if mo:
        o=[float(x) for x in mo.group(1).split()]
        if len(o)==3: origin=np.array(o,float)
    return Box(lo=origin, hi=origin+L, L=L)


def read_sites(path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str,str]], Optional[Box]]:
    p=Path(path)
    rows=[]; box=None
    if p.suffix.lower()==".xyz":
        with open(path,"r") as f:
            n=int(f.readline().strip())
            comment=f.readline().strip()
            box=parse_extxyz_lattice(comment)
            coords=[]; stypes=[]
            for i,line in enumerate(f):
                if not line.strip(): continue
                parts=line.split()
                if len(parts)<4: continue
                coords.append([float(parts[1]),float(parts[2]),float(parts[3])])
                stype=parts[4] if len(parts)>=5 else "0"
                stypes.append(int(float(stype)))
                rows.append({"original_index":str(i),"species":parts[0],"site_type":str(stype)})
        return np.asarray(coords,float), np.asarray(stypes,int), rows, box
    else:
        with open(path,newline="") as f:
            r=csv.DictReader(f)
            if r.fieldnames is None:
                raise ValueError("CSV file needs header")
            fields={name.lower():name for name in r.fieldnames}
            def find(names):
                for nm in names:
                    if nm in fields: return fields[nm]
                return None
            fx=find(["x","posx","site_x"]); fy=find(["y","posy","site_y"]); fz=find(["z","posz","site_z"])
            if fx is None or fy is None or fz is None:
                raise ValueError(f"Could not find x,y,z columns in {path}; fields={r.fieldnames}")
            ft=find(["site_type","type","region_type","candidate_type"])
            coords=[]; stypes=[]
            for i,row in enumerate(r):
                coords.append([float(row[fx]),float(row[fy]),float(row[fz])])
                stypes.append(int(float(row[ft]))) if ft else stypes.append(0)
                rr=dict(row); rr.setdefault("original_index",str(i)); rows.append(rr)
        return np.asarray(coords,float), np.asarray(stypes,int), rows, None


def wrap_frac(coords: np.ndarray, box: Box) -> np.ndarray:
    return ((coords - box.lo) / box.L) % 1.0


def infer_normal_axis(frac: np.ndarray, bins: int = 160) -> int:
    """Infer normal as axis with strongest localization / most empty space."""
    scores=[]
    for ax in range(3):
        hist,_=np.histogram(frac[:,ax], bins=bins, range=(0,1))
        occupied=np.count_nonzero(hist)
        maxgap=largest_periodic_empty_gap(hist)
        # localized bands: fewer occupied bins and large empty gaps
        scores.append(maxgap + (bins-occupied)/bins)
    return int(np.argmax(scores))


def largest_periodic_empty_gap(hist: np.ndarray) -> float:
    occ=hist>0
    if np.all(occ): return 0.0
    doubled=np.r_[occ,occ]
    best=cur=0
    for v in doubled:
        if not v:
            cur+=1; best=max(best,cur)
        else:
            cur=0
    best=min(best,len(hist))
    return best/len(hist)


def fold_points_periodic(frac2: np.ndarray, reps: Tuple[int,int]) -> np.ndarray:
    # map fractional coordinates into representative tile [0,1/n)
    out=frac2.copy()
    for j,n in enumerate(reps):
        out[:,j] = (out[:,j]*n) % 1.0 / n
    return out


def tile_signature_score(frac: np.ndarray, axes: Tuple[int,int], reps: Tuple[int,int], sample_max: int = 4000) -> Tuple[float,int,float]:
    """Return RMS nearest-neighbor mismatch between folded all-sites and tile-0 sites.
    Lower is better. Also returns number of tile-0 points and tile population CV.
    """
    f2=frac[:,list(axes)]
    n1,n2=reps
    tile_ids=np.floor(f2*np.array([n1,n2])).astype(int)
    tile_ids=np.clip(tile_ids,0,np.array([n1-1,n2-1]))
    flat=tile_ids[:,0]*n2 + tile_ids[:,1]
    counts=np.bincount(flat, minlength=n1*n2)
    cv=float(np.std(counts)/(np.mean(counts)+1e-12))
    in0=(flat==0)
    n0=int(np.count_nonzero(in0))
    if n0 < 5:
        return 1e9,n0,cv
    folded=fold_points_periodic(f2, reps)
    ref=folded[in0]
    pts=folded
    if len(pts)>sample_max:
        rng=np.random.default_rng(12345)
        idx=rng.choice(len(pts), size=sample_max, replace=False)
        pts=pts[idx]
    tree=cKDTree(ref, boxsize=[1.0/n1,1.0/n2])
    d,_=tree.query(pts,k=1)
    return float(np.sqrt(np.mean(d*d))), n0, cv


def choose_reps(frac: np.ndarray, tangent_axes: Tuple[int,int], max_reps: int, tol_frac: float) -> Tuple[int,int,Dict[Tuple[int,int],Tuple[float,int,float]]]:
    scores={}
    # Test candidates that divide the pattern into many tiles. Prefer largest number of reps if score acceptable.
    for n1 in range(1,max_reps+1):
        for n2 in range(1,max_reps+1):
            if n1*n2 > max_reps*max_reps:
                continue
            score,n0,cv=tile_signature_score(frac,tangent_axes,(n1,n2))
            scores[(n1,n2)]=(score,n0,cv)
    # acceptable: small mismatch and reasonably balanced tile counts
    acceptable=[]
    for reps,(score,n0,cv) in scores.items():
        if score <= tol_frac and n0 >= 5 and cv < 0.35:
            acceptable.append((reps[0]*reps[1], reps, score, n0, cv))
    if acceptable:
        acceptable.sort(key=lambda x:(x[0], -x[2]))
        _,reps,_,_,_=acceptable[-1]
        return reps[0],reps[1],scores
    # Fallback to best score among nontrivial, but avoid pathological tiny n0.
    best=min(scores.items(), key=lambda kv:(kv[1][0]+0.05*kv[1][2], -kv[0][0]*kv[0][1]))[0]
    return best[0],best[1],scores


def select_tile(frac: np.ndarray, tangent_axes: Tuple[int,int], reps: Tuple[int,int], tile_index: Tuple[int,int], margin_frac: float) -> np.ndarray:
    f2=frac[:,list(tangent_axes)]
    n1,n2=reps; i,j=tile_index
    lo=np.array([i/n1, j/n2]) - margin_frac
    hi=np.array([(i+1)/n1, (j+1)/n2]) + margin_frac
    # periodic interval membership
    mask=np.ones(len(frac), dtype=bool)
    for k,(a,b) in enumerate(zip(lo,hi)):
        x=f2[:,k]
        if a < 0:
            m=(x >= (a%1.0)) | (x <= b)
        elif b >= 1:
            m=(x >= a) | (x <= (b%1.0))
        else:
            m=(x >= a) & (x <= b)
        mask &= m
    return mask


def write_csv(path: str, coords: np.ndarray, stypes: np.ndarray, orig_indices: np.ndarray, folded: np.ndarray, axes: Tuple[int,int], reps: Tuple[int,int]):
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["candidate_id","original_index","x","y","z","site_type","folded_u","folded_v","tile_reps_u","tile_reps_v","tangent_axis_u","tangent_axis_v"])
        for k,(p,t,oi,ff) in enumerate(zip(coords,stypes,orig_indices,folded)):
            w.writerow([k,int(oi),f"{p[0]:.10f}",f"{p[1]:.10f}",f"{p[2]:.10f}",int(t),f"{ff[0]:.10f}",f"{ff[1]:.10f}",reps[0],reps[1],AXIS_NAMES[axes[0]],AXIS_NAMES[axes[1]]])


def write_extxyz(path: str, coords: np.ndarray, stypes: np.ndarray, box: Box):
    with open(path,"w") as f:
        f.write(f"{len(coords)}\n")
        f.write('Lattice="{:.10f} 0 0 0 {:.10f} 0 0 0 {:.10f}" Origin="{:.10f} {:.10f} {:.10f}" Properties=species:S:1:pos:R:3:site_type:I:1 pbc="T T T"\n'.format(box.L[0],box.L[1],box.L[2],box.lo[0],box.lo[1],box.lo[2]))
        for p,t in zip(coords,stypes):
            f.write(f"HGB {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {int(t)}\n")


def write_lmp_sites(path: str, coords: np.ndarray, stypes: np.ndarray, box: Box):
    # LAMMPS data containing candidate sites as atoms; types preserve site type if possible.
    unique=sorted(set(int(x) for x in stypes)) or [1]
    tmap={old:i+1 for i,old in enumerate(unique)}
    with open(path,"w") as f:
        f.write("Representative GB candidate tile generated by gb_candidate_repetition_sampler.py\n\n")
        f.write(f"{len(coords)} atoms\n")
        f.write(f"{len(unique)} atom types\n\n")
        f.write(f"{box.lo[0]:.10f} {box.hi[0]:.10f} xlo xhi\n")
        f.write(f"{box.lo[1]:.10f} {box.hi[1]:.10f} ylo yhi\n")
        f.write(f"{box.lo[2]:.10f} {box.hi[2]:.10f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for old,new in tmap.items():
            f.write(f"{new} 1.00784 # original_site_type_{old}\n")
        f.write("\nAtoms # atomic\n\n")
        for i,(p,t) in enumerate(zip(coords,stypes),start=1):
            f.write(f"{i} {tmap[int(t)]} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="Original LAMMPS Fe data file, used only for box dimensions.")
    ap.add_argument("--sites", required=True, help="Candidate sites CSV or extended XYZ from Voronoi step.")
    ap.add_argument("--normal-axis", choices=["x","y","z","auto"], default="auto", help="GB/network normal. Tangent axes are sampled for repetitions.")
    ap.add_argument("--tile-reps", default=None, help="Manual repetitions along tangent axes, e.g. 20,5. Overrides auto.")
    ap.add_argument("--tile-lengths", default=None, help="Manual tile lengths in Å along tangent axes, e.g. 2.856,6.386. Overrides auto.")
    ap.add_argument("--max-reps", type=int, default=40, help="Maximum repetitions to test per tangent axis during auto detection.")
    ap.add_argument("--tol-frac", type=float, default=0.004, help="Folded-pattern RMS tolerance in fractional box units for auto repetition detection.")
    ap.add_argument("--tile-index", default="0,0", help="Which tile to keep along tangent axes, e.g. 0,0 or 3,2.")
    ap.add_argument("--margin", type=float, default=0.0, help="Optional margin in Å added around selected tile.")
    ap.add_argument("--out-prefix", default="gb_representative_tile")
    args=ap.parse_args()

    box_lmp=read_lammps_box(args.structure)
    coords, stypes, rows, box_xyz = read_sites(args.sites)
    box=box_xyz if box_xyz is not None else box_lmp
    # Use LAMMPS box origin if XYZ lacks/uses same origin; robust for candidates from same workflow.
    if box_xyz is None:
        box=box_lmp

    frac=wrap_frac(coords, box)
    if args.normal_axis == "auto":
        normal=infer_normal_axis(frac)
    else:
        normal=AXES[args.normal_axis]
    tangent=tuple(ax for ax in range(3) if ax != normal)

    if args.tile_lengths:
        vals=[float(x) for x in args.tile_lengths.split(",")]
        if len(vals)!=2: raise ValueError("--tile-lengths needs two comma-separated values")
        reps=(max(1,int(round(box.L[tangent[0]]/vals[0]))), max(1,int(round(box.L[tangent[1]]/vals[1]))))
        scores={}
    elif args.tile_reps:
        vals=[int(x) for x in args.tile_reps.split(",")]
        if len(vals)!=2: raise ValueError("--tile-reps needs two comma-separated integers")
        reps=(vals[0],vals[1]); scores={}
    else:
        reps=choose_reps(frac,tangent,args.max_reps,args.tol_frac)[:2]
        scores=choose_reps(frac,tangent,args.max_reps,args.tol_frac)[2]

    tile_idx=tuple(int(x) for x in args.tile_index.split(","))
    if len(tile_idx)!=2: raise ValueError("--tile-index needs i,j")
    margin_frac=args.margin / max(box.L[tangent[0]], box.L[tangent[1]]) if args.margin>0 else 0.0
    mask=select_tile(frac,tangent,reps,tile_idx,margin_frac)

    selected=np.where(mask)[0]
    f2=frac[:,list(tangent)]
    folded=fold_points_periodic(f2,reps)

    out=args.out_prefix
    write_csv(out+"_representative_candidates.csv", coords[selected], stypes[selected], selected, folded[selected], tangent, reps)
    write_extxyz(out+"_representative_candidates.xyz", coords[selected], stypes[selected], box)
    write_lmp_sites(out+"_representative_candidates.lmp", coords[selected], stypes[selected], box)

    # Mapping file: every original candidate -> folded location and tile id.
    n1,n2=reps
    tile_ids=np.floor(f2*np.array([n1,n2])).astype(int)
    tile_ids=np.clip(tile_ids,0,np.array([n1-1,n2-1]))
    with open(out+"_periodic_mapping.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["original_index","x","y","z","site_type","tile_i","tile_j","folded_u","folded_v"])
        for i,(p,t,tid,ff) in enumerate(zip(coords,stypes,tile_ids,folded)):
            w.writerow([i,f"{p[0]:.10f}",f"{p[1]:.10f}",f"{p[2]:.10f}",int(t),int(tid[0]),int(tid[1]),f"{ff[0]:.10f}",f"{ff[1]:.10f}"])

    with open(out+"_summary.txt","w") as f:
        f.write(f"Input structure: {args.structure}\n")
        f.write(f"Input sites: {args.sites}\n")
        f.write(f"Total candidates: {len(coords)}\n")
        f.write(f"Selected representative candidates: {len(selected)}\n")
        f.write(f"Box lo: {box.lo.tolist()}\n")
        f.write(f"Box lengths: {box.L.tolist()}\n")
        f.write(f"Normal axis: {AXIS_NAMES[normal]}\n")
        f.write(f"Tangent axes: {AXIS_NAMES[tangent[0]]}, {AXIS_NAMES[tangent[1]]}\n")
        f.write(f"Detected/used repetitions along tangent axes: {reps[0]}, {reps[1]}\n")
        f.write(f"Representative tile index: {tile_idx[0]}, {tile_idx[1]}\n")
        f.write(f"Tile lengths Å: {box.L[tangent[0]]/reps[0]:.10f}, {box.L[tangent[1]]/reps[1]:.10f}\n")
        if scores:
            score,n0,cv=scores.get(reps,(math.nan,0,math.nan))
            f.write(f"Auto repetition score RMS frac: {score:.8g}\n")
            f.write(f"Tile-0 candidate count during scoring: {n0}\n")
            f.write(f"Tile population CV: {cv:.8g}\n")
        f.write("\nOutputs:\n")
        f.write(f"  {out}_representative_candidates.csv\n")
        f.write(f"  {out}_representative_candidates.xyz\n")
        f.write(f"  {out}_representative_candidates.lmp\n")
        f.write(f"  {out}_periodic_mapping.csv\n")

    print(f"Read {len(coords)} candidates")
    print(f"Normal axis: {AXIS_NAMES[normal]}; tangent axes: {AXIS_NAMES[tangent[0]]}, {AXIS_NAMES[tangent[1]]}")
    print(f"Using repetitions: {reps[0]} x {reps[1]}")
    print(f"Selected tile {tile_idx}: {len(selected)} candidates")
    print(f"Wrote {out}_representative_candidates.csv/.xyz/.lmp and {out}_periodic_mapping.csv")


if __name__ == "__main__":
    main()
