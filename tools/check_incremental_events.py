#!/usr/bin/env python3
"""Smoke checks for incremental event-table primitives."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kmc.event_manager import CandidateBuildResult  # noqa: E402
from kmc.incremental_events import (  # noqa: E402
    FenwickTree,
    IncrementalEventTable,
    affected_h_sites_for_particle_changes,
    affected_h_sites_for_move,
)


def check_fenwick() -> None:
    tree = FenwickTree([1.0, 2.0, 3.0])
    assert abs(tree.total() - 6.0) < 1.0e-12
    assert tree.find_prefix_index(0.0) == 0
    assert tree.find_prefix_index(1.01) == 1
    assert tree.find_prefix_index(3.01) == 2
    tree.update(1, 4.0)
    assert abs(tree.total() - 8.0) < 1.0e-12
    assert tree.find_prefix_index(4.5) == 1


def check_event_table() -> None:
    initial = CandidateBuildResult(
        valid_moves=[(5, 6), (1, 2), (5, 7)],
        rates=[5.0, 1.0, 7.0],
        barriers={(5, 6): 0.2, (1, 2): 0.1, (5, 7): 0.3},
    )
    table = IncrementalEventTable.from_candidate_result(initial)
    assert table.moves == [(1, 2), (5, 6), (5, 7)]
    selected = table.select(random.Random(123))
    assert selected is not None
    assert selected.total_rate_hz == 13.0

    replacement = CandidateBuildResult(
        valid_moves=[(5, 8), (9, 10)],
        rates=[8.0, 10.0],
        barriers={(5, 8): 0.4, (9, 10): 0.5},
    )
    stats = table.update_source_h_sites([5], replacement)
    assert stats.removed_events == 2
    assert stats.added_events == 2
    assert table.moves == [(1, 2), (5, 8), (9, 10)]
    assert abs(table._fenwick.total() - 19.0) < 1.0e-12


def check_affected_sites() -> None:
    sites = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    affected = affected_h_sites_for_move(
        sites=sites,
        h_indices=[1, 2, 3],
        box=[10.0, 10.0, 10.0],
        old_site=0,
        new_site=1,
        impact_radius_a=2.1,
    )
    assert affected == [1, 2, 3]

    changed = affected_h_sites_for_particle_changes(
        sites=sites,
        h_indices=[1, 2, 3],
        box=[10.0, 10.0, 10.0],
        changed_sites=[0],
        impact_radius_a=2.1,
    )
    assert changed == [1, 3]


def main() -> None:
    check_fenwick()
    check_event_table()
    check_affected_sites()
    print("incremental event checks passed")


if __name__ == "__main__":
    main()
