from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from kmc.structures import build_local_neb_structures_shell_from_host


def test_periodic_host_tree_matches_full_host_scan() -> None:
    box = np.array([10.0, 10.0, 10.0])
    host = np.array(
        [
            [0.2, 0.2, 0.2],
            [9.8, 0.2, 0.2],
            [5.0, 5.0, 5.0],
            [1.0, 1.0, 1.0],
        ]
    )
    center = np.array([0.1, 0.2, 0.2])
    h_positions = center.reshape((1, 3))
    tree = cKDTree(host, boxsize=box)

    scanned, _, scanned_mover = build_local_neb_structures_shell_from_host(
        center,
        center,
        host,
        box,
        h_positions=h_positions,
        h_indices=np.array([7]),
        inner_radius_a=1.0,
        outer_radius_a=1.5,
    )
    indexed, _, indexed_mover = build_local_neb_structures_shell_from_host(
        center,
        center,
        host,
        box,
        host_fe_tree=tree,
        h_positions=h_positions,
        h_indices=np.array([7]),
        inner_radius_a=1.0,
        outer_radius_a=1.5,
    )

    assert scanned_mover == indexed_mover
    assert scanned.get_chemical_symbols() == indexed.get_chemical_symbols()
    np.testing.assert_allclose(scanned.get_positions(), indexed.get_positions())
    np.testing.assert_array_equal(
        scanned.arrays["neighbor_index"],
        indexed.arrays["neighbor_index"],
    )
