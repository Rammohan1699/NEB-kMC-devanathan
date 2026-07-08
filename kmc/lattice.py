"""
Pure lattice + interstitial site geometry utilities.

This version mirrors the site basis and KNN neighbour topology used by the
working monolithic cache_new.py reference:
  - 12 tetrahedral sites per BCC conventional cell
  - 6 octahedral sites per BCC conventional cell
  - KNN neighbour generation with legacy default K=8
"""
from __future__ import annotations
import numpy as np
from itertools import product


def minimum_image(vec, box_lengths):
    v = np.array(vec, dtype=float).copy()
    for i, L in enumerate(box_lengths):
        v[i] -= np.round(v[i] / L) * L
    return v


def pbc_diff(a, b, box_lengths):
    return minimum_image(np.array(b, dtype=float) - np.array(a, dtype=float), box_lengths)


def pbc_distance(a, b, box_lengths):
    return float(np.linalg.norm(pbc_diff(a, b, box_lengths)))


def build_bcc_fe_supercell(nx, ny, nz, a0):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=float)
    atoms = []
    for i, j, k in product(range(nx), range(ny), range(nz)):
        cell = np.array([i, j, k], dtype=float)
        for b in basis:
            atoms.append((cell + b) * a0)
    return np.array(atoms, dtype=float)


def generate_tetrahedral_sites(nx, ny, nz, a0):
    """Generate the legacy 12-site BCC tetrahedral basis from cache_new.py."""
    tetra_basis = np.array([
        [0.25, 0.5, 0.0], [0.75, 0.5, 0.0],
        [0.5, 0.25, 0.0], [0.5, 0.75, 0.0],
        [0.25, 0.0, 0.5], [0.75, 0.0, 0.5],
        [0.5, 0.0, 0.25], [0.5, 0.0, 0.75],
        [0.0, 0.25, 0.5], [0.0, 0.75, 0.5],
        [0.0, 0.5, 0.25], [0.0, 0.5, 0.75],
    ], dtype=float)
    grid = np.indices((nx, ny, nz)).reshape(3, -1).T * a0
    return ((tetra_basis * a0)[:, np.newaxis, :] + grid).reshape(-1, 3)


def generate_octahedral_sites(nx, ny, nz, a0):
    """Generate the legacy 6-site BCC octahedral basis from cache_new.py."""
    oct_basis = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ], dtype=float)
    grid = np.indices((nx, ny, nz)).reshape(3, -1).T * a0
    return ((oct_basis * a0)[:, np.newaxis, :] + grid).reshape(-1, 3)


def generate_all_interstitial_sites(nx, ny, nz, a0):
    t = generate_tetrahedral_sites(nx, ny, nz, a0)
    o = generate_octahedral_sites(nx, ny, nz, a0)
    return {"tetra": t, "octa": o, "all": np.vstack([t, o])}


def build_neighbor_list(site_positions, cutoff, box_lengths):
    n = len(site_positions)
    nbrs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if pbc_distance(site_positions[i], site_positions[j], box_lengths) <= cutoff:
                nbrs[i].append(j)
                nbrs[j].append(i)
    return nbrs


def validate_site_uniqueness(sites, tol=1e-6):
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if np.linalg.norm(sites[i] - sites[j]) < tol:
                raise ValueError(f"Duplicate sites found: {i},{j}")
    return True


def get_k_nearest_neighbors(sites, box_lengths, k=6, same_type=False, per_cell=12):
    """
    Return list-of-lists of k nearest neighbour site indices under PBC.

    This mirrors cache_new.py's KNN event graph: query k+1 with scipy cKDTree,
    drop self, and optionally prefer same interstitial type using the legacy
    12-tetra + 6-octa repeating basis convention.
    """
    from scipy.spatial import cKDTree

    sites = np.asarray(sites, dtype=float)
    box = np.asarray(box_lengths, dtype=float)
    if len(sites) == 0:
        return []

    k = max(1, int(k))
    query_k = min(k + 1, len(sites))
    tree = cKDTree(sites, boxsize=box)
    _dists, idxs = tree.query(sites, k=query_k, workers=-1)

    if query_k == 1:
        return [[] for _ in range(len(sites))]

    idxs = np.asarray(idxs)
    if idxs.ndim == 1:
        idxs = idxs[:, None]

    nearest = idxs[:, 1:]

    if not same_type:
        return [list(map(int, row[:k])) for row in nearest]

    def is_tet(i):
        return (int(i) % int(per_cell)) < 12

    out = []
    for i, row in enumerate(nearest):
        row = list(map(int, row))
        same = [j for j in row if is_tet(i) == is_tet(j)]
        other = [j for j in row if is_tet(i) != is_tet(j)]
        out.append((same + other)[:k])
    return out
