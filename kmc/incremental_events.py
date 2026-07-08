"""
Incremental KMC event storage and weighted selection.

The normal driver can rebuild every H->neighbor event each step.  This module
keeps a rank-0 event table keyed by move and updates only source H sites whose
local environment can have changed after the last accepted hop.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    from .event_manager import CandidateBuildResult, Move, SelectedEvent
    from .lattice import pbc_diff
except ImportError:  # pragma: no cover - direct script execution fallback
    from event_manager import CandidateBuildResult, Move, SelectedEvent
    from lattice import pbc_diff


class FenwickTree:
    """Fenwick tree for mutable non-negative event weights."""

    def __init__(self, weights: Sequence[float] = ()):
        self.weights = [0.0 for _ in weights]
        self.tree = [0.0 for _ in range(len(weights) + 1)]
        for idx, weight in enumerate(weights):
            self.update(idx, float(weight))

    def __len__(self) -> int:
        return len(self.weights)

    def total(self) -> float:
        return self.prefix_sum(len(self.weights) - 1) if self.weights else 0.0

    def prefix_sum(self, idx: int) -> float:
        if idx < 0:
            return 0.0
        idx = min(int(idx), len(self.weights) - 1) + 1
        total = 0.0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx
        return float(total)

    def update(self, idx: int, new_weight: float) -> None:
        idx = int(idx)
        if idx < 0 or idx >= len(self.weights):
            raise IndexError(idx)
        new_weight = float(new_weight)
        if not np.isfinite(new_weight) or new_weight < 0.0:
            new_weight = 0.0
        delta = new_weight - self.weights[idx]
        self.weights[idx] = new_weight
        tree_idx = idx + 1
        while tree_idx < len(self.tree):
            self.tree[tree_idx] += delta
            tree_idx += tree_idx & -tree_idx

    def find_prefix_index(self, threshold: float) -> int:
        """Return smallest index whose cumulative weight is greater than threshold."""
        if not self.weights:
            raise ValueError("cannot sample an empty Fenwick tree")
        total = self.total()
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("cannot sample a zero-total Fenwick tree")
        threshold = min(max(float(threshold), 0.0), np.nextafter(total, 0.0))

        idx = 0
        bit = 1 << (len(self.weights).bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt < len(self.tree) and self.tree[nxt] <= threshold:
                idx = nxt
                threshold -= self.tree[nxt]
            bit >>= 1
        return min(idx, len(self.weights) - 1)


@dataclass(frozen=True)
class IncrementalUpdateStats:
    removed_events: int = 0
    added_events: int = 0
    rebuilt_h_sites: int = 0


class IncrementalEventTable:
    """
    Mutable global event table.

    Moves are kept sorted by (h_site, n_site) after each update so a fixed RNG
    seed sees the same deterministic event order as the full rebuild path.
    """

    def __init__(self) -> None:
        self.moves: list[Move] = []
        self.rates: list[float] = []
        self.barriers: dict[Move, float] = {}
        self._source_to_moves: dict[int, set[Move]] = {}
        self._fenwick = FenwickTree()

    @classmethod
    def from_candidate_result(cls, result: CandidateBuildResult) -> "IncrementalEventTable":
        table = cls()
        table.replace_all(result.valid_moves, result.rates, result.barriers)
        return table

    def replace_all(
        self,
        moves: Sequence[Move],
        rates: Sequence[float],
        barriers: Optional[Mapping[Move, float]] = None,
    ) -> None:
        if len(moves) != len(rates):
            raise ValueError("moves and rates must have the same length")
        rows = []
        for move, rate in zip(moves, rates):
            clean_move = (int(move[0]), int(move[1]))
            clean_rate = float(rate)
            if np.isfinite(clean_rate) and clean_rate > 0.0:
                rows.append((clean_move, clean_rate))
        rows.sort(key=lambda item: (item[0][0], item[0][1]))

        self.moves = [move for move, _rate in rows]
        self.rates = [rate for _move, rate in rows]
        source_barriers = barriers or {}
        self.barriers = {
            move: float(source_barriers[move])
            for move in self.moves
            if move in source_barriers and np.isfinite(float(source_barriers[move]))
        }
        self._rebuild_indices()

    def update_source_h_sites(
        self,
        remove_h_sites: Iterable[int],
        replacement: CandidateBuildResult,
    ) -> IncrementalUpdateStats:
        """Remove all events from source H sites and add replacement events."""
        remove_set = {int(h) for h in remove_h_sites}
        old_count = len(self.moves)
        kept_moves: list[Move] = []
        kept_rates: list[float] = []
        kept_barriers: dict[Move, float] = {}
        for move, rate in zip(self.moves, self.rates):
            if int(move[0]) in remove_set:
                continue
            kept_moves.append(move)
            kept_rates.append(rate)
            if move in self.barriers:
                kept_barriers[move] = self.barriers[move]

        add_count = 0
        for move, rate in zip(replacement.valid_moves, replacement.rates):
            rate = float(rate)
            if not np.isfinite(rate) or rate <= 0.0:
                continue
            clean_move = (int(move[0]), int(move[1]))
            kept_moves.append(clean_move)
            kept_rates.append(rate)
            add_count += 1
            if clean_move in replacement.barriers:
                kept_barriers[clean_move] = float(replacement.barriers[clean_move])

        self.replace_all(kept_moves, kept_rates, kept_barriers)
        return IncrementalUpdateStats(
            removed_events=max(0, old_count - len(kept_moves) + add_count),
            added_events=add_count,
            rebuilt_h_sites=len(remove_set),
        )

    def select(self, rng: Optional[random.Random] = None) -> Optional[SelectedEvent]:
        total_rate = self._fenwick.total()
        if total_rate <= 0.0 or not np.isfinite(total_rate):
            return None

        if rng is None:
            dt = float(np.random.exponential(1.0 / total_rate))
            threshold = float(random.random()) * total_rate
            r_time = float("nan")
            r_index = float("nan")
        else:
            r_index = float(rng.random())
            threshold = r_index * total_rate
            r_time = max(float(rng.random()), np.finfo(float).tiny)
            dt = -math.log(r_time) / total_rate

        idx = self._fenwick.find_prefix_index(threshold)
        move = self.moves[idx]
        return SelectedEvent(
            h_site=int(move[0]),
            n_site=int(move[1]),
            rate_hz=float(self.rates[idx]),
            barrier_ev=self.barriers.get(move),
            dt_s=float(dt),
            total_rate_hz=float(total_rate),
            random_index_value=r_index,
            random_time_value=r_time,
        )

    def _rebuild_indices(self) -> None:
        self._source_to_moves = {}
        for move in self.moves:
            self._source_to_moves.setdefault(int(move[0]), set()).add(move)
        self._fenwick = FenwickTree(self.rates)


def max_neighbor_hop_distance(
    sites: np.ndarray,
    neighbors: Sequence[Sequence[int]],
    box: Sequence[float],
) -> float:
    """Return the maximum neighbor-hop distance in the site graph."""
    max_dist = 0.0
    for h_site, nbrs in enumerate(neighbors):
        h_pos = sites[int(h_site)]
        for n_site in nbrs:
            dist = float(np.linalg.norm(pbc_diff(h_pos, sites[int(n_site)], box)))
            max_dist = max(max_dist, dist)
    return max_dist


def affected_h_sites_for_move(
    *,
    sites: np.ndarray,
    h_indices: Sequence[int],
    box: Sequence[float],
    old_site: int,
    new_site: int,
    impact_radius_a: float,
) -> list[int]:
    """
    Occupied H source sites whose candidate environment may have changed.

    The region is conservative: any currently occupied H within impact_radius_a
    of either the old or new mover position is rebuilt.
    """
    if impact_radius_a <= 0.0:
        return sorted({int(old_site), int(new_site)} & {int(h) for h in h_indices})

    old_pos = np.asarray(sites[int(old_site)], dtype=float)
    new_pos = np.asarray(sites[int(new_site)], dtype=float)
    affected: set[int] = set()
    for h_site in map(int, h_indices):
        h_pos = np.asarray(sites[h_site], dtype=float)
        d_old = float(np.linalg.norm(pbc_diff(old_pos, h_pos, box)))
        d_new = float(np.linalg.norm(pbc_diff(new_pos, h_pos, box)))
        if min(d_old, d_new) <= float(impact_radius_a):
            affected.add(h_site)
    return sorted(affected)


def affected_h_sites_for_particle_changes(
    *,
    sites: np.ndarray,
    h_indices: Sequence[int],
    box: Sequence[float],
    changed_sites: Iterable[int],
    impact_radius_a: float,
) -> list[int]:
    """Occupied H sites whose event environments can change after insert/delete."""
    changed = sorted({int(site) for site in changed_sites})
    if not changed:
        return []
    occupied = sorted({int(site) for site in h_indices})
    if impact_radius_a <= 0.0:
        return sorted(set(changed) & set(occupied))

    changed_positions = [np.asarray(sites[site], dtype=float) for site in changed]
    affected: set[int] = set()
    for h_site in occupied:
        h_pos = np.asarray(sites[h_site], dtype=float)
        if any(
            float(np.linalg.norm(pbc_diff(changed_pos, h_pos, box)))
            <= float(impact_radius_a)
            for changed_pos in changed_positions
        ):
            affected.add(h_site)
    return sorted(affected)
