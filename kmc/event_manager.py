"""
KMC event construction and rejection-free event selection.

This module is the bridge between:
  - current occupied H site state
  - neighbor topology
  - barrier cache lookup
  - NEB miss-job creation
  - rate calculation
  - rejection-free KMC move selection

It intentionally does NOT run NEB and does NOT perform MPI collectives.
That work belongs to scheduler.py and the NEB engine/driver layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import random

import numpy as np


EventKey = Any
Move = Tuple[int, int]             # h_site, n_site
MissJob = Tuple[EventKey, Tuple[int, int, int]]  # key, (src_rank, h_site, n_site)


@dataclass(frozen=True)
class KineticParameters:
    """Physical parameters for KMC rate calculation."""
    temperature_k: float = 300.0
    attempt_frequency_hz: float = 1.0e13
    boltzmann_ev_per_k: float = 8.617333262145e-5

    def rate_from_barrier(self, barrier_ev: float) -> float:
        """Arrhenius rate nu * exp(-Ea/kBT)."""
        if barrier_ev is None or not np.isfinite(barrier_ev):
            return 0.0
        if barrier_ev < 0.0:
            # Keep compatibility with current practice: negative/near-zero barriers
            # become very fast but finite rates.
            barrier_ev = 0.0
        denom = self.boltzmann_ev_per_k * self.temperature_k
        if denom <= 0.0:
            raise ValueError("temperature_k must be positive")
        return float(self.attempt_frequency_hz * math.exp(-float(barrier_ev) / denom))


@dataclass(frozen=True)
class EventManagerConfig:
    """Controls candidate construction and event selection."""
    env_radius_a: float = 5.0
    use_cache_for_jobs: bool = True
    cache_only: bool = False
    low_barrier_ev: float = 1.0e-3
    skip_failed_keys: bool = True
    rank: int = 0


@dataclass
class RateRow:
    """Diagnostic row for a candidate rate."""
    step: int
    rank: int
    h_site: int
    n_site: int
    barrier_ev: float
    rate_hz: float
    source: str
    env_kind: str
    status: str

    def as_tuple(self) -> tuple:
        return (
            self.step,
            self.rank,
            self.h_site,
            self.n_site,
            self.barrier_ev,
            self.rate_hz,
            self.source,
            self.env_kind,
            self.status,
        )


@dataclass
class CandidateBuildResult:
    """
    Output of building events from cache.

    valid_moves/rates are ready for KMC selection.
    miss_jobs should be sent to NEB scheduling.
    """
    valid_moves: List[Move] = field(default_factory=list)
    rates: List[float] = field(default_factory=list)
    barriers: Dict[Move, float] = field(default_factory=dict)
    miss_jobs: List[MissJob] = field(default_factory=list)
    move_log: Dict[int, List[Tuple[int, str]]] = field(default_factory=dict)
    rate_rows: List[RateRow] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass(frozen=True)
class SelectedEvent:
    """Chosen rejection-free KMC event."""
    h_site: int
    n_site: int
    rate_hz: float
    barrier_ev: Optional[float]
    dt_s: float
    total_rate_hz: float
    random_index_value: float
    random_time_value: float


def valid_neighbor_sites(
    h_site: int,
    neighbors: Sequence[Sequence[int]],
    occupied_sites: set[int],
) -> List[int]:
    """Return unoccupied neighbor sites available to a given H."""
    if int(h_site) >= len(neighbors):
        return []
    return [
        int(n)
        for n in neighbors[int(h_site)]
        if int(n) != int(h_site) and int(n) not in occupied_sites
    ]


def build_candidate_events(
    *,
    step: int,
    rank: int,
    local_h_sites: Sequence[int],
    neighbors: Sequence[Sequence[int]],
    occupied_sites: Iterable[int],
    cache: Mapping[Any, Any],
    make_key: Callable[[int, int], Any],
    kinetics: KineticParameters,
    cfg: Optional[EventManagerConfig] = None,
    failed_keys: Optional[set[Any]] = None,
    reported_failed_keys: Optional[set[Any]] = None,
) -> CandidateBuildResult:
    """
    Build cache-hit events and NEB miss jobs for a rank-local subset of H sites.

    Parameters
    ----------
    make_key:
        Callable receiving (h_site, n_site) and returning the cache/environment key.
        This preserves compatibility with the existing environment-key system.
    cache:
        Mapping-like barrier cache.
    failed_keys:
        Keys known to have produced non-finite barriers.
    reported_failed_keys:
        Optional set used to suppress duplicate failed-rate diagnostics.

    Notes
    -----
    This function deliberately does not build local NEB structures. It only
    creates miss jobs in the same shape expected by scheduler.py:
        (key, (src_rank, h_site, n_site))
    """
    if cfg is None:
        cfg = EventManagerConfig(rank=rank)

    occupied = set(map(int, occupied_sites))
    failed_keys = failed_keys if failed_keys is not None else set()
    reported_failed_keys = reported_failed_keys if reported_failed_keys is not None else set()

    result = CandidateBuildResult()

    for h in map(int, local_h_sites):
        move_entries: List[Tuple[int, str]] = []

        if h not in occupied:
            # Local assignment can be stale after moves; skip robustly.
            result.move_log[h] = move_entries
            continue

        for n in valid_neighbor_sites(h, neighbors, occupied):
            try:
                key = make_key(h, n)

                if cfg.skip_failed_keys and key in failed_keys:
                    move_entries.append((n, "F"))
                    if key not in reported_failed_keys:
                        result.rate_rows.append(
                            RateRow(
                                step=step,
                                rank=rank,
                                h_site=h,
                                n_site=n,
                                barrier_ev=float("inf"),
                                rate_hz=float("nan"),
                                source="failed_env",
                                env_kind="failed_env",
                                status="fail",
                            )
                        )
                        reported_failed_keys.add(key)
                    continue

                if cfg.use_cache_for_jobs and key in cache:
                    barrier = float(cache[key])
                    result.cache_hits += 1

                    if not np.isfinite(barrier):
                        move_entries.append((n, "C"))
                        failed_keys.add(key)
                        if key not in reported_failed_keys:
                            result.rate_rows.append(
                                RateRow(
                                    step=step,
                                    rank=rank,
                                    h_site=h,
                                    n_site=n,
                                    barrier_ev=float("nan"),
                                    rate_hz=float("nan"),
                                    source="cache",
                                    env_kind="cache",
                                    status="bad_cached",
                                )
                            )
                            reported_failed_keys.add(key)
                        continue

                    rate = kinetics.rate_from_barrier(barrier)
                    if rate <= 0.0 or not np.isfinite(rate):
                        move_entries.append((n, "0"))
                        result.rate_rows.append(
                            RateRow(
                                step=step,
                                rank=rank,
                                h_site=h,
                                n_site=n,
                                barrier_ev=barrier,
                                rate_hz=rate,
                                source="cache",
                                env_kind="cache",
                                status="zero_rate",
                            )
                        )
                        continue

                    move = (h, n)
                    result.valid_moves.append(move)
                    result.rates.append(rate)
                    result.barriers[move] = barrier
                    move_entries.append((n, "C"))
                    result.rate_rows.append(
                        RateRow(
                            step=step,
                            rank=rank,
                            h_site=h,
                            n_site=n,
                            barrier_ev=barrier,
                            rate_hz=rate,
                            source="cache",
                            env_kind="cache",
                            status="ok",
                        )
                    )
                else:
                    result.cache_misses += 1
                    if cfg.cache_only:
                        result.rate_rows.append(
                            RateRow(
                                step=step,
                                rank=rank,
                                h_site=h,
                                n_site=n,
                                barrier_ev=float("nan"),
                                rate_hz=float("nan"),
                                source="cache",
                                env_kind="cache",
                                status="miss",
                            )
                        )
                        move_entries.append((n, "cache_miss"))
                    else:
                        result.miss_jobs.append((key, (rank, h, n)))
                        move_entries.append((n, "M"))

            except Exception:
                move_entries.append((n, "error"))
                result.rate_rows.append(
                    RateRow(
                        step=step,
                        rank=rank,
                        h_site=h,
                        n_site=n,
                        barrier_ev=float("nan"),
                        rate_hz=float("nan"),
                        source="event_build",
                        env_kind="event_build",
                        status="error",
                    )
                )

        result.move_log[h] = move_entries

    return result


def fold_neb_results_into_events(
    *,
    step: int,
    rank: Optional[int],
    local_miss_jobs: Sequence[MissJob],
    fold_results: Mapping[Any, float],
    kinetics: KineticParameters,
    failed_keys: Optional[set[Any]] = None,
    reported_failed_keys: Optional[set[Any]] = None,
) -> CandidateBuildResult:
    """
    Convert NEB results for previously missed jobs into KMC-ready events.

    This should be called after scheduler/NEB returns `fold_results`.
    """
    failed_keys = failed_keys if failed_keys is not None else set()
    reported_failed_keys = reported_failed_keys if reported_failed_keys is not None else set()

    result = CandidateBuildResult()

    fold_all_ranks = rank is None or int(rank) < 0

    for key, (src_rank, h, n) in local_miss_jobs:
        if not fold_all_ranks and int(src_rank) != int(rank):
            # In normal local fold mode this should not occur, but global mode
            # may pass all keys. Keep only this rank's original candidates.
            continue

        if key not in fold_results:
            continue

        barrier = float(fold_results[key])
        if not np.isfinite(barrier):
            failed_keys.add(key)
            if key not in reported_failed_keys:
                result.rate_rows.append(
                    RateRow(
                        step=step,
                        rank=int(src_rank),
                        h_site=int(h),
                        n_site=int(n),
                        barrier_ev=float("inf"),
                        rate_hz=float("nan"),
                        source="neb",
                        env_kind="unique",
                        status="fail",
                    )
                )
                reported_failed_keys.add(key)
            continue

        rate = kinetics.rate_from_barrier(barrier)
        if rate <= 0.0 or not np.isfinite(rate):
            result.rate_rows.append(
                RateRow(
                    step=step,
                    rank=int(src_rank),
                    h_site=int(h),
                    n_site=int(n),
                    barrier_ev=barrier,
                    rate_hz=rate,
                    source="neb",
                    env_kind="unique",
                    status="zero_rate",
                )
            )
            continue

        move = (int(h), int(n))
        result.valid_moves.append(move)
        result.rates.append(rate)
        result.barriers[move] = barrier
        result.rate_rows.append(
            RateRow(
                step=step,
                rank=int(src_rank),
                h_site=int(h),
                n_site=int(n),
                barrier_ev=barrier,
                rate_hz=rate,
                source="neb",
                env_kind="unique",
                status="ok",
            )
        )

    return result


def merge_candidate_results(*parts: CandidateBuildResult) -> CandidateBuildResult:
    """Merge cache-hit and NEB-result candidate sets."""
    merged = CandidateBuildResult()
    for part in parts:
        merged.valid_moves.extend(part.valid_moves)
        merged.rates.extend(part.rates)
        merged.barriers.update(part.barriers)
        merged.miss_jobs.extend(part.miss_jobs)
        merged.cache_hits += part.cache_hits
        merged.cache_misses += part.cache_misses
        merged.rate_rows.extend(part.rate_rows)
        for h, entries in part.move_log.items():
            merged.move_log.setdefault(h, []).extend(entries)
    return merged


def select_rejection_free_event(
    moves: Sequence[Move],
    rates: Sequence[float],
    *,
    barriers: Optional[Mapping[Move, float]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[SelectedEvent]:
    """
    Select one KMC event using the standard rejection-free algorithm.

    Returns None if no positive finite rates are available.
    """
    if len(moves) != len(rates):
        raise ValueError("moves and rates must have the same length")

    clean_moves: List[Move] = []
    clean_rates: List[float] = []
    for move, rate in zip(moves, rates):
        rate = float(rate)
        if np.isfinite(rate) and rate > 0.0:
            clean_moves.append((int(move[0]), int(move[1])))
            clean_rates.append(rate)

    if not clean_moves:
        return None

    total_rate = float(sum(clean_rates))
    if total_rate <= 0.0 or not np.isfinite(total_rate):
        return None

    if rng is None:
        # Match cache_new.py exactly: NumPy draws the residence time, then
        # Python's module-level random.choices selects the weighted move.
        dt = float(np.random.exponential(1.0 / total_rate))
        h, n = random.choices(clean_moves, weights=clean_rates, k=1)[0]
        move = (int(h), int(n))
        barrier = None
        if barriers is not None and move in barriers:
            barrier = float(barriers[move])
        return SelectedEvent(
            h_site=move[0],
            n_site=move[1],
            rate_hz=float(clean_rates[clean_moves.index(move)]),
            barrier_ev=barrier,
            dt_s=dt,
            total_rate_hz=total_rate,
            random_index_value=float("nan"),
            random_time_value=float("nan"),
        )

    # Alternate deterministic sampler kept for direct tests that inject a
    # private RNG; the driver uses the monolith-compatible path above.
    r_index = float(rng.random())
    threshold = r_index * total_rate

    cumulative = 0.0
    chosen_idx = len(clean_moves) - 1
    for idx, rate in enumerate(clean_rates):
        cumulative += rate
        if cumulative >= threshold:
            chosen_idx = idx
            break

    r_time = max(float(rng.random()), np.finfo(float).tiny)
    dt = -math.log(r_time) / total_rate

    h, n = clean_moves[chosen_idx]
    move = (h, n)
    barrier = None
    if barriers is not None and move in barriers:
        barrier = float(barriers[move])

    return SelectedEvent(
        h_site=h,
        n_site=n,
        rate_hz=float(clean_rates[chosen_idx]),
        barrier_ev=barrier,
        dt_s=float(dt),
        total_rate_hz=total_rate,
        random_index_value=r_index,
        random_time_value=r_time,
    )


def apply_selected_event_to_sites(
    h_indices: np.ndarray,
    selected: SelectedEvent,
) -> tuple[np.ndarray, int]:
    """
    Return updated occupied site indices after applying selected event.

    Returns:
      new_h_indices, moved_slot

    moved_slot is the index into h_indices corresponding to selected.h_site.
    """
    arr = np.asarray(h_indices, dtype=int).copy()
    matches = np.where(arr == int(selected.h_site))[0]
    if len(matches) == 0:
        raise ValueError(f"Selected H site {selected.h_site} not found in occupied indices")
    slot = int(matches[0])
    arr[slot] = int(selected.n_site)
    return arr, slot


def affected_h_after_move(
    *,
    h_indices: Sequence[int],
    sites: np.ndarray,
    moved_from: int,
    moved_to: int,
    affect_radius_a: float,
    pbc_distance: Callable[[np.ndarray, np.ndarray], float],
) -> set[int]:
    """
    Determine which occupied H sites should recompute rates next step.

    This mirrors the current idea of recomputing H near the moved-from/moved-to
    region while allowing caller to provide the exact PBC distance function.
    """
    affected: set[int] = set()
    sites = np.asarray(sites, dtype=float)
    p_from = sites[int(moved_from)]
    p_to = sites[int(moved_to)]

    for h in map(int, h_indices):
        p = sites[h]
        if (
            pbc_distance(p, p_from) <= affect_radius_a
            or pbc_distance(p, p_to) <= affect_radius_a
        ):
            affected.add(h)

    return affected
