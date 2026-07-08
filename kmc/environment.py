"""Local environment fingerprinting utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def pbc_diff(pos_i, pos_j, box):
    delta = np.asarray(pos_j, dtype=float) - np.asarray(pos_i, dtype=float)
    box = np.asarray(box, dtype=float)
    return delta - box * np.round(delta / box)

def build_hydrogen_kdtree(h_positions, box=None):
    if len(h_positions)==0:
        return None
    if box is None:
        return cKDTree(np.asarray(h_positions))
    return cKDTree(np.asarray(h_positions), boxsize=np.asarray(box, dtype=float))

def query_local_neighbors(tree, center, radius):
    if tree is None:
        return []
    return tree.query_ball_point(center, radius)

def make_env_key(
    start_site,
    end_site,
    h_positions,
    *,
    h_tree=None,
    box=None,
    radius_A=5.0,
    pos_bin_A=0.10,
    hop_bin_A=0.02,
    mode="env_plus_dir",
    radius=None,
) -> Any:
    """
    Build the same cache key shape as the reference monolith.

    The key is translation-invariant and atom-order invariant: it stores the
    sorted, quantized PBC displacement vectors from the moving H to nearby H
    atoms, plus the quantized PBC hop vector when mode is ``env_plus_dir``.
    """
    if radius is not None:
        radius_A = radius
    if box is None:
        raise ValueError("make_env_key requires box lengths for monolith-compatible PBC keys")

    start = np.asarray(start_site, dtype=float)
    end = np.asarray(end_site, dtype=float)
    positions = np.asarray(h_positions, dtype=float)
    tree = h_tree if h_tree is not None else build_hydrogen_kdtree(positions, box=box)

    nbr_idx = [] if tree is None else tree.query_ball_point(start, r=float(radius_A))
    disp = []
    for i in nbr_idx:
        d = pbc_diff(start, positions[int(i)], box)
        if np.allclose(d, 0.0, atol=0.2):
            continue
        disp.append(tuple(int(x) for x in np.round(d / float(pos_bin_A)).astype(int)))

    disp_sorted = tuple(sorted(disp))

    if mode == "env_only":
        return ("ENVONLY", float(pos_bin_A), float(radius_A), disp_sorted)

    hop = pbc_diff(start, end, box)
    hop_q = tuple(int(x) for x in np.round(hop / float(hop_bin_A)).astype(int))
    return ("ENV+DIR", float(pos_bin_A), float(radius_A), float(hop_bin_A), disp_sorted, hop_q)

def compare_envs(sig1,sig2):
    return sig1==sig2
