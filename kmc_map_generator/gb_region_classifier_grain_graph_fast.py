#!/usr/bin/env python3
"""Fast topology-agnostic grain/GB classifier for BCC Fe using periodic KDTree.

No GB plane assumptions. It identifies bulk regions by local BCC frame continuity.
GB seeds are atoms that are non-BCC-like or touch a different grain orientation in
PBC neighbor space. Seeds are dilated into GB_VORONOI and TRANSITION masks.
"""
from __future__ import annotations
import argparse, math, itertools
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

@dataclass
class LammpsData:
    ids: np.ndarray; types: np.ndarray; pos: np.ndarray; box: np.ndarray

def read_lammps_atomic_data(path: str) -> LammpsData:
    lines = Path(path).read_text().splitlines(); box=np.zeros((3,2)); n_atoms=None
    for line in lines[:300]:
        p=line.split()
        if len(p)>=2 and p[-1]=='atoms': n_atoms=int(p[0])
        elif len(p)>=4 and p[2]=='xlo' and p[3]=='xhi': box[0]=[float(p[0]),float(p[1])]
        elif len(p)>=4 and p[2]=='ylo' and p[3]=='yhi': box[1]=[float(p[0]),float(p[1])]
        elif len(p)>=4 and p[2]=='zlo' and p[3]=='zhi': box[2]=[float(p[0]),float(p[1])]
    start=None
    for i,line in enumerate(lines):
        if line.strip().startswith('Atoms'):
            start=i+1; break
    if start is None: raise ValueError('Could not find Atoms section')
    rows=[]
    for line in lines[start:]:
        s=line.strip()
        if not s or s.startswith('#'): continue
        p=s.split()
        if not p[0].lstrip('+-').isdigit():
            if rows: break
            continue
        if len(p)<5: continue
        rows.append((int(p[0]),int(p[1]),float(p[2]),float(p[3]),float(p[4])))
        if n_atoms is not None and len(rows)>=n_atoms: break
    a=np.array(rows,dtype=float)
    return LammpsData(a[:,0].astype(int),a[:,1].astype(int),a[:,2:5],box)

def wrap_to_box(pos, box):
    lo=box[:,0]; L=box[:,1]-box[:,0]
    return lo + np.mod(pos-lo,L)

def shifted(pos, box):
    lo=box[:,0]; L=box[:,1]-box[:,0]
    return np.mod(pos-lo,L)

def min_image_vec(a,b,L):
    d=a-b
    return d - L*np.round(d/L)

IDEAL_DIAGONALS=np.array([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],float)/math.sqrt(3)

def kabsch(A,B):
    H=A.T@B; U,_,Vt=np.linalg.svd(H); R=Vt.T@U.T
    if np.linalg.det(R)<0: Vt[-1]*=-1; R=Vt.T@U.T
    return R

def fit_frame(vecs):
    if len(vecs)<8: return None, float('inf')
    V=vecs[:8]; n=np.linalg.norm(V,axis=1)
    if np.any(n<1e-10): return None, float('inf')
    U=V/n[:,None]; dots=U@IDEAL_DIAGONALS.T
    pairs=[]; used_o=set(); used_i=set()
    for _ in range(8):
        best=None; val=-2
        for i in range(8):
            if i in used_o: continue
            for j in range(8):
                if j in used_i: continue
                if dots[i,j]>val: val=dots[i,j]; best=(i,j)
        if best is None: break
        i,j=best; used_o.add(i); used_i.add(j); pairs.append((i,j))
    if len(pairs)!=8: return None,float('inf')
    A=np.array([IDEAL_DIAGONALS[j] for _,j in pairs]); B=np.array([U[i] for i,_ in pairs])
    R=kabsch(A,B); pred=A@R.T
    err=float(np.sqrt(np.mean(np.sum((pred-B)**2,axis=1))))
    return R,err

def cube_rots():
    out=[]
    for perm in itertools.permutations(range(3)):
        P=np.zeros((3,3));
        for i,j in enumerate(perm): P[i,j]=1
        for signs in itertools.product([-1.,1.], repeat=3):
            S=np.diag(signs)@P
            if round(np.linalg.det(S))==1: out.append(S)
    return out
CUBIC=cube_rots()

def angle(M):
    c=np.clip((np.trace(M)-1)/2, -1, 1)
    return math.degrees(math.acos(float(c)))

def misori(R1,R2):
    D=R1.T@R2
    return min(angle(S@D) for S in CUBIC)

class DSU:
    def __init__(self,n): self.p=np.arange(n); self.sz=np.ones(n,dtype=int)
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        ra=self.find(a); rb=self.find(b)
        if ra==rb: return
        if self.sz[ra]<self.sz[rb]: ra,rb=rb,ra
        self.p[rb]=ra; self.sz[ra]+=self.sz[rb]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('data_file'); ap.add_argument('--fe-type',type=int,default=1); ap.add_argument('--a0',type=float,default=2.856)
    ap.add_argument('--first-cut',type=float,default=None); ap.add_argument('--max-frame-error',type=float,default=0.20)
    ap.add_argument('--orientation-cutoff-deg',type=float,default=8.0)
    ap.add_argument('--boundary-neighbor-cut',type=float,default=None)
    ap.add_argument('--gb-buffer',type=float,default=6.0); ap.add_argument('--transition-buffer',type=float,default=9.0)
    ap.add_argument('--out-prefix',default='regions_grain_fast')
    args=ap.parse_args()
    data=read_lammps_atomic_data(args.data_file); data.pos=wrap_to_box(data.pos,data.box)
    lo=data.box[:,0]; L=data.box[:,1]-data.box[:,0]
    fe_global=np.where(data.types==args.fe_type)[0]; fe_pos=data.pos[fe_global]; fe_shift=shifted(fe_pos,data.box)
    tree=cKDTree(fe_shift, boxsize=L)
    r1=math.sqrt(3)*args.a0/2; r2=args.a0
    first_cut=args.first_cut if args.first_cut else 0.5*(r1+r2)
    boundary_cut=args.boundary_neighbor_cut if args.boundary_neighbor_cut else 1.15*args.a0
    print(f'Read {len(data.pos)} atoms; Fe atoms: {len(fe_pos)}')
    print(f'first_cut={first_cut:.4f} Å boundary_cut={boundary_cut:.4f} Å')
    d,idx=tree.query(fe_shift,k=9,distance_upper_bound=first_cut)
    frames=np.zeros((len(fe_pos),3,3)); ferr=np.full(len(fe_pos),np.inf); good=np.zeros(len(fe_pos),bool)
    # Fit a local BCC orientation frame around each Fe atom from its eight
    # nearest body-diagonal neighbors. Atoms with poor local frames become
    # boundary seeds because they are not clearly bulk-like.
    for i in range(len(fe_pos)):
        neigh=idx[i][1:]; neigh=neigh[neigh<len(fe_pos)]
        vecs=np.array([min_image_vec(fe_shift[j],fe_shift[i],L) for j in neigh])
        # query returns sorted by distance, so vecs already first-neighbor sorted.
        R,e=fit_frame(vecs)
        if R is not None and e<=args.max_frame_error:
            frames[i]=R; ferr[i]=e; good[i]=True
        if (i+1)%5000==0: print(f'  local frames {i+1}/{len(fe_pos)} BCC-like={int(good.sum())}')
    pairs=list(tree.query_pairs(boundary_cut, output_type='set'))
    dsu=DSU(len(fe_pos))
    checked=0
    # Join neighboring BCC-like atoms into grains when their fitted local
    # orientations agree up to cubic crystal symmetry.
    for i,j in pairs:
        if good[i] and good[j]:
            if misori(frames[i],frames[j]) <= args.orientation_cutoff_deg:
                dsu.union(i,j)
        checked+=1
    grain_id=np.full(len(fe_pos),-1,dtype=int); root_to_gid={}
    for i in np.where(good)[0]:
        r=dsu.find(i)
        if r not in root_to_gid: root_to_gid[r]=len(root_to_gid)
        grain_id[i]=root_to_gid[r]
    boundary_seed=~good.copy()
    # A seed is either non-BCC-like or a BCC-like atom touching a different
    # grain. The seed set is later dilated into core GB and transition masks.
    for i,j in pairs:
        if not good[i] or not good[j]:
            if good[i]: boundary_seed[i]=True
            if good[j]: boundary_seed[j]=True
        elif grain_id[i]!=grain_id[j]:
            boundary_seed[i]=True; boundary_seed[j]=True
    gb=boundary_seed.copy(); trans=np.zeros(len(fe_pos),bool)
    seed_idx=np.where(boundary_seed)[0]
    if len(seed_idx):
        near_gb=tree.query_ball_point(fe_shift[seed_idx], args.gb_buffer)
        for lst in near_gb: gb[np.asarray(lst,dtype=int)]=True
        near_tr=tree.query_ball_point(fe_shift[seed_idx], args.transition_buffer)
        for lst in near_tr: trans[np.asarray(lst,dtype=int)]=True
    trans = trans & ~gb
    bulk = good & ~gb & ~trans
    prefix=args.out_prefix
    # This mask file is the API consumed by the downstream bulk tetra mapper and
    # GB/transition Voronoi candidate generator.
    np.savez(f'{prefix}_masks.npz', fe_indices_global=fe_global, fe_bulk_template=bulk, fe_transition=trans,
             fe_gb_voronoi=gb, fe_grain_id=grain_id, fe_bcc_like=good, fe_boundary_seed=boundary_seed,
             fe_frame_error=ferr, box=data.box, a0=args.a0)
    with open(f'{prefix}_atoms.csv','w') as f:
        f.write('local_fe_index,global_atom_index,atom_id,x,y,z,grain_id,bcc_like,boundary_seed,region,frame_error\n')
        for i,gi in enumerate(fe_global):
            region='BULK_TEMPLATE' if bulk[i] else ('TRANSITION' if trans[i] else 'GB_VORONOI')
            p=fe_pos[i]
            f.write(f'{i},{gi},{data.ids[gi]},{p[0]:.10f},{p[1]:.10f},{p[2]:.10f},{grain_id[i]},{int(good[i])},{int(boundary_seed[i])},{region},{ferr[i]:.8g}\n')
    with open(f'{prefix}_region_atoms.xyz','w') as f:
        f.write(f'{len(fe_pos)}\nFe=bulk H=transition He=GB/discovery\n')
        for i,p in enumerate(fe_pos):
            sym='Fe' if bulk[i] else ('H' if trans[i] else 'He')
            f.write(f'{sym} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n')
    counts=np.bincount(grain_id[grain_id>=0]) if np.any(grain_id>=0) else np.array([])
    grain_lines='\n'.join([f'  grain {i}: {c}' for i,c in enumerate(counts)]) or '  none'
    summary=f'''Input: {args.data_file}
Fe atoms: {len(fe_pos)}
Box lengths: {L}

Parameters:
  a0 = {args.a0:.6f} Å
  first_cut = {first_cut:.6f} Å
  max_frame_error = {args.max_frame_error:.6f}
  orientation_cutoff_deg = {args.orientation_cutoff_deg:.6f}
  boundary_neighbor_cut = {boundary_cut:.6f} Å
  gb_buffer = {args.gb_buffer:.6f} Å
  transition_buffer = {args.transition_buffer:.6f} Å

Detected grains: {len(counts)}
{grain_lines}

Counts:
  BCC-like Fe atoms: {int(good.sum())}
  boundary/non-BCC seed atoms: {int(boundary_seed.sum())}
  BULK_TEMPLATE atoms: {int(bulk.sum())}
  TRANSITION atoms: {int(trans.sum())}
  GB_VORONOI atoms: {int(gb.sum())}

Wrote:
  {prefix}_atoms.csv
  {prefix}_region_atoms.xyz
  {prefix}_masks.npz
'''
    Path(f'{prefix}_summary.txt').write_text(summary)
    print(summary)
if __name__=='__main__': main()
