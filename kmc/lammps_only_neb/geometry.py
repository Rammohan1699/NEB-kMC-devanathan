from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .config import LammpsOnlyNEBConfig
from .models import EnvironmentNEBBatch, HopSpec, LammpsSystem


FE_TYPE = 1
H_TYPE = 2


def minimum_image_delta(start: np.ndarray, end: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    box_arr = np.asarray(box, dtype=float)
    return delta - box_arr * np.round(delta / box_arr)


def tetrahedral_sites(nx: int, ny: int, nz: int, a0: float) -> np.ndarray:
    basis = np.array(
        [
            [0.25, 0.5, 0.0],
            [0.75, 0.5, 0.0],
            [0.5, 0.25, 0.0],
            [0.5, 0.75, 0.0],
            [0.25, 0.0, 0.5],
            [0.75, 0.0, 0.5],
            [0.5, 0.0, 0.25],
            [0.5, 0.0, 0.75],
            [0.0, 0.25, 0.5],
            [0.0, 0.75, 0.5],
            [0.0, 0.5, 0.25],
            [0.0, 0.5, 0.75],
        ],
        dtype=float,
    )
    grid = np.indices((nx, ny, nz)).reshape(3, -1).T * a0
    return ((basis * a0)[:, np.newaxis, :] + grid).reshape(-1, 3)


def bcc_fe_positions(nx: int, ny: int, nz: int, a0: float) -> np.ndarray:
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float)
    cells = np.indices((nx, ny, nz)).reshape(3, -1).T.astype(float)
    return ((basis * a0)[:, np.newaxis, :] + cells * a0).reshape(-1, 3)


def knn_sites(sites: np.ndarray, box: np.ndarray, k: int = 6) -> list[list[int]]:
    tree = cKDTree(np.asarray(sites, dtype=float), boxsize=np.asarray(box, dtype=float))
    _dists, idxs = tree.query(sites, k=int(k) + 1, workers=-1)
    return [list(map(int, row[1:])) for row in np.asarray(idxs)]


def build_batch_from_sites(
    *,
    env_key: object,
    h_site: int,
    sites: np.ndarray,
    neighbor_indices: Sequence[int],
    fe_positions: np.ndarray,
    box: np.ndarray,
    lattice_a: float,
) -> EnvironmentNEBBatch:
    h_pos = np.asarray(sites[int(h_site)], dtype=float)
    hops = []
    for slot, n_site in enumerate(neighbor_indices):
        final = h_pos + minimum_image_delta(h_pos, np.asarray(sites[int(n_site)], dtype=float), box)
        hops.append(
            HopSpec(
                h_site=int(h_site),
                n_site=int(n_site),
                slot=int(slot),
                initial=h_pos.copy(),
                final=final,
                hop_key=(env_key, int(h_site), int(n_site)),
            )
        )
    return EnvironmentNEBBatch(
        env_key=env_key,
        h_site=int(h_site),
        h_position=h_pos.copy(),
        hops=tuple(hops),
        sites=np.asarray(sites, dtype=float),
        fe_positions=np.asarray(fe_positions, dtype=float),
        box=np.asarray(box, dtype=float),
        lattice_a=float(lattice_a),
    )


def build_shell_system(batch: EnvironmentNEBBatch, hop: HopSpec, cfg: LammpsOnlyNEBConfig) -> LammpsSystem:
    inner = max(0.0, float(cfg.shell_inner_radius_a))
    outer = max(inner, float(cfg.shell_outer_radius_a))
    hop_delta = minimum_image_delta(hop.initial, hop.final, batch.box)
    local_pad = max(float(batch.lattice_a), float(np.linalg.norm(hop_delta))) + float(cfg.shell_pad_a)
    local_half = outer + local_pad
    local_center = np.array([local_half, local_half, local_half], dtype=float)
    local_box = np.array([2.0 * local_half, 2.0 * local_half, 2.0 * local_half], dtype=float)

    rel = batch.fe_positions - hop.initial.reshape(1, 3)
    rel -= batch.box.reshape(1, 3) * np.round(rel / batch.box.reshape(1, 3))
    dist = np.linalg.norm(rel, axis=1)
    keep = np.where(dist <= outer + 1.0e-8)[0]
    if len(keep) == 0:
        raise RuntimeError("Shell extraction kept zero Fe atoms")

    positions: list[np.ndarray] = []
    types: list[int] = []
    frozen_ids: list[int] = []
    atom_id = 1
    for idx in keep:
        d = float(dist[idx])
        positions.append(local_center + rel[idx])
        types.append(FE_TYPE)
        if d > inner:
            frozen_ids.append(atom_id)
        atom_id += 1

    h_id = atom_id
    positions.append(local_center.copy())
    types.append(H_TYPE)

    final_positions = {
        local_id: np.asarray(pos, dtype=float)
        for local_id, pos in enumerate(positions, start=1)
    }
    final_positions[h_id] = local_center + hop_delta

    return LammpsSystem(
        positions=np.asarray(positions, dtype=float),
        types=np.asarray(types, dtype=int),
        box=local_box,
        h_id=h_id,
        final_positions=final_positions,
        frozen_ids=tuple(frozen_ids),
        boundary="f f f",
        mode="shell",
        n_fe=int(len(keep)),
    )
