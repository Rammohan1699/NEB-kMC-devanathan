from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DevanathanBoundaryConfig:
    source_x_min_a: float = 0.0
    source_x_max_a: float = 10.0
    source_target_fraction: float = 0.05
    left_sink_x_max_a: float = 0.0
    sink_x_min_a: float = 0.0
    cross_section_area_a2: float = 0.0
    trim_source_excess: bool = True
    seed: int | None = 42

    def normalized(self, box: np.ndarray) -> "DevanathanBoundaryConfig":
        box = np.asarray(box, dtype=float)
        sink_x = self.sink_x_min_a if self.sink_x_min_a > 0.0 else float(box[0])
        area = self.cross_section_area_a2 if self.cross_section_area_a2 > 0.0 else float(box[1] * box[2])
        cfg = DevanathanBoundaryConfig(
            source_x_min_a=float(self.source_x_min_a),
            source_x_max_a=float(self.source_x_max_a),
            source_target_fraction=float(self.source_target_fraction),
            left_sink_x_max_a=float(self.left_sink_x_max_a),
            sink_x_min_a=float(sink_x),
            cross_section_area_a2=float(area),
            trim_source_excess=bool(self.trim_source_excess),
            seed=self.seed,
        )
        if cfg.source_x_max_a <= cfg.source_x_min_a:
            raise ValueError("DEVANATHAN_SOURCE_X_MAX_A must be greater than DEVANATHAN_SOURCE_X_MIN_A")
        if cfg.sink_x_min_a <= cfg.source_x_max_a:
            raise ValueError("DEVANATHAN_SINK_X_MIN_A must be greater than DEVANATHAN_SOURCE_X_MAX_A")
        if cfg.left_sink_x_max_a > cfg.source_x_min_a:
            raise ValueError(
                "DEVANATHAN_LEFT_SINK_X_MAX_A must be less than or equal to "
                "DEVANATHAN_SOURCE_X_MIN_A"
            )
        if not 0.0 <= cfg.source_target_fraction <= 1.0:
            raise ValueError("DEVANATHAN_SOURCE_FRACTION must be in [0, 1]")
        if cfg.cross_section_area_a2 <= 0.0:
            raise ValueError("Devanathan cross-section area must be positive")
        return cfg


@dataclass(frozen=True)
class DevanathanUpdate:
    h_indices: np.ndarray
    h_unwrapped_positions: np.ndarray
    inserted_sites: tuple[int, ...]
    deleted_sink_sites: tuple[int, ...]
    trimmed_source_sites: tuple[int, ...]
    source_sites: int
    source_fe_atoms: int
    source_target_h: int
    source_occupied: int
    sink_occupied: int
    total_occupied: int
    cumulative_inserted: int
    cumulative_removed: int
    flux_per_a2_s: float | None
    deleted_left_sink_sites: tuple[int, ...] = ()
    cumulative_left_removed: int = 0
    left_sink_occupied: int = 0

    @property
    def particle_count_changed(self) -> bool:
        return bool(
            self.inserted_sites
            or self.deleted_sink_sites
            or self.deleted_left_sink_sites
            or self.trimmed_source_sites
        )


class DevanathanBoundary:
    """Constant-concentration source and absorbing-sink helper for KMC."""

    def __init__(
        self,
        config: DevanathanBoundaryConfig,
        box: np.ndarray,
        *,
        left_sink_site_indices: Sequence[int] | np.ndarray | None = None,
        left_sink_label: str = "left_sink",
    ) -> None:
        self.config = config.normalized(box)
        self.left_sink_site_indices = (
            None
            if left_sink_site_indices is None
            else tuple(sorted({int(site) for site in left_sink_site_indices}))
        )
        self.left_sink_site_set = (
            None if self.left_sink_site_indices is None else set(self.left_sink_site_indices)
        )
        self.left_sink_label = str(left_sink_label or "left_sink")
        self.rng = random.Random(self.config.seed)
        self.cumulative_inserted = 0
        self.cumulative_removed = 0
        self.cumulative_left_removed = 0

    def source_sites(self, sites: np.ndarray) -> tuple[int, ...]:
        positions = np.asarray(sites, dtype=float)
        x = positions[:, 0]
        mask = (x >= self.config.source_x_min_a) & (x < self.config.source_x_max_a)
        return tuple(int(i) for i in np.flatnonzero(mask))

    def source_fe_count(self, fe_positions: np.ndarray) -> int:
        positions = np.asarray(fe_positions, dtype=float)
        if positions.size == 0:
            return 0
        x = positions[:, 0]
        mask = (x >= self.config.source_x_min_a) & (x < self.config.source_x_max_a)
        return int(np.count_nonzero(mask))

    def target_source_count(self, sites: np.ndarray, fe_positions: np.ndarray) -> int:
        del sites
        return int(round(self.config.source_target_fraction * self.source_fe_count(fe_positions)))

    def initialize_source(self, sites: np.ndarray, fe_positions: np.ndarray) -> np.ndarray:
        source = self.source_sites(sites)
        target = self.target_source_count(sites, fe_positions)
        if target > len(source):
            raise ValueError("Devanathan source target exceeds available source sites")
        return np.asarray(self.rng.sample(source, target), dtype=int)

    def left_sink_sites(self, sites: np.ndarray) -> tuple[int, ...]:
        if self.config.left_sink_x_max_a <= 0.0:
            return ()
        if self.left_sink_site_indices is not None:
            n_sites = len(sites)
            invalid = [
                site
                for site in self.left_sink_site_indices
                if site < 0 or site >= n_sites
            ]
            if invalid:
                raise ValueError(
                    f"Devanathan {self.left_sink_label} contains invalid site indices: "
                    f"{invalid[:5]}"
                )
            return self.left_sink_site_indices
        x = np.asarray(sites, dtype=float)[:, 0]
        return tuple(int(i) for i in np.flatnonzero(x < self.config.left_sink_x_max_a))

    def _is_left_sink_site(self, site: int, site_x: float) -> bool:
        if self.config.left_sink_x_max_a <= 0.0:
            return False
        if self.left_sink_site_set is not None:
            return int(site) in self.left_sink_site_set
        return float(site_x) < self.config.left_sink_x_max_a

    def apply(
        self,
        *,
        sites: np.ndarray,
        fe_positions: np.ndarray,
        h_indices: np.ndarray,
        h_unwrapped_positions: np.ndarray,
        elapsed_time_s: float,
    ) -> DevanathanUpdate:
        current_indices = [int(x) for x in np.asarray(h_indices, dtype=int)]
        current_unwrapped = [np.asarray(p, dtype=float) for p in np.asarray(h_unwrapped_positions, dtype=float)]
        if len(current_indices) != len(current_unwrapped):
            raise ValueError("h_indices and h_unwrapped_positions length mismatch")

        deleted_left_sink: list[int] = []
        deleted_sink: list[int] = []
        kept_indices: list[int] = []
        kept_unwrapped: list[np.ndarray] = []
        for site, pos in zip(current_indices, current_unwrapped):
            site_x = float(sites[int(site), 0])
            if self._is_left_sink_site(int(site), site_x):
                deleted_left_sink.append(int(site))
            elif site_x >= self.config.sink_x_min_a:
                deleted_sink.append(int(site))
            else:
                kept_indices.append(int(site))
                kept_unwrapped.append(pos)
        current_indices = kept_indices
        current_unwrapped = kept_unwrapped

        source_set = set(self.source_sites(sites))
        source_fe_atoms = self.source_fe_count(fe_positions)
        target = self.target_source_count(sites, fe_positions)
        source_slots = [idx for idx, site in enumerate(current_indices) if site in source_set]
        trimmed: list[int] = []
        if self.config.trim_source_excess and len(source_slots) > target:
            trim_slots = sorted(self.rng.sample(source_slots, len(source_slots) - target), reverse=True)
            for slot in trim_slots:
                trimmed.append(int(current_indices[slot]))
                del current_indices[slot]
                del current_unwrapped[slot]

        occupied = set(current_indices)
        source_occupied = sum(1 for site in current_indices if site in source_set)
        deficit = max(0, target - source_occupied)
        source_vacant = [site for site in source_set if site not in occupied]
        if deficit > len(source_vacant):
            raise ValueError("Devanathan source layer does not have enough vacant sites to refill")

        inserted = list(self.rng.sample(source_vacant, deficit))
        for site in inserted:
            current_indices.append(int(site))
            current_unwrapped.append(np.asarray(sites[int(site)], dtype=float).copy())

        self.cumulative_inserted += len(inserted)
        self.cumulative_removed += len(deleted_sink)
        self.cumulative_left_removed += len(deleted_left_sink)

        out_indices = np.asarray(current_indices, dtype=int)
        out_unwrapped = (
            np.asarray(current_unwrapped, dtype=float).reshape((len(current_unwrapped), 3))
            if current_unwrapped
            else np.empty((0, 3), dtype=float)
        )
        source_occupied = sum(1 for site in out_indices if int(site) in source_set)
        sink_occupied = sum(1 for site in out_indices if float(sites[int(site), 0]) >= self.config.sink_x_min_a)
        left_sink_occupied = sum(
            1
            for site in out_indices
            if self._is_left_sink_site(int(site), float(sites[int(site), 0]))
        )
        flux = None
        if elapsed_time_s > 0.0:
            flux = self.cumulative_removed / self.config.cross_section_area_a2 / elapsed_time_s

        return DevanathanUpdate(
            h_indices=out_indices,
            h_unwrapped_positions=out_unwrapped,
            inserted_sites=tuple(inserted),
            deleted_sink_sites=tuple(deleted_sink),
            trimmed_source_sites=tuple(trimmed),
            source_sites=len(source_set),
            source_fe_atoms=source_fe_atoms,
            source_target_h=target,
            source_occupied=source_occupied,
            sink_occupied=sink_occupied,
            total_occupied=len(out_indices),
            cumulative_inserted=self.cumulative_inserted,
            cumulative_removed=self.cumulative_removed,
            flux_per_a2_s=flux,
            deleted_left_sink_sites=tuple(deleted_left_sink),
            cumulative_left_removed=self.cumulative_left_removed,
            left_sink_occupied=left_sink_occupied,
        )


class DevanathanPulseBoundary(DevanathanBoundary):
    """Finite source pulse with absorbing sink deletion and no source refill."""

    def __init__(
        self,
        config: DevanathanBoundaryConfig,
        box: np.ndarray,
        *,
        initial_count: int,
        control_site_indices: Sequence[int] | np.ndarray | None = None,
        control_fe_count: int | None = None,
        region_label: str = "pulse",
        left_sink_site_indices: Sequence[int] | np.ndarray | None = None,
        left_sink_label: str = "left_sink",
    ) -> None:
        super().__init__(
            config,
            box,
            left_sink_site_indices=left_sink_site_indices,
            left_sink_label=left_sink_label,
        )
        self.initial_count = max(0, int(initial_count))
        self.control_site_indices = (
            None
            if control_site_indices is None
            else tuple(sorted({int(site) for site in control_site_indices}))
        )
        self.control_fe_count = None if control_fe_count is None else int(control_fe_count)
        self.region_label = str(region_label or "pulse")

    def source_sites(self, sites: np.ndarray) -> tuple[int, ...]:
        if self.control_site_indices is None:
            return super().source_sites(sites)
        n_sites = len(sites)
        invalid = [
            site
            for site in self.control_site_indices
            if site < 0 or site >= n_sites
        ]
        if invalid:
            raise ValueError(
                f"Devanathan {self.region_label} contains invalid site indices: "
                f"{invalid[:5]}"
            )
        return self.control_site_indices

    def source_fe_count(self, fe_positions: np.ndarray) -> int:
        if self.control_fe_count is not None:
            return max(0, int(self.control_fe_count))
        return super().source_fe_count(fe_positions)

    def target_source_count(self, sites: np.ndarray, fe_positions: np.ndarray) -> int:
        del sites, fe_positions
        return int(self.initial_count)

    def initialize_source(self, sites: np.ndarray, fe_positions: np.ndarray) -> np.ndarray:
        del fe_positions
        source = self.source_sites(sites)
        if self.initial_count > len(source):
            raise ValueError(
                f"Devanathan {self.region_label} initial count {self.initial_count} "
                f"exceeds available sites {len(source)}"
            )
        return np.asarray(self.rng.sample(source, self.initial_count), dtype=int)

    def apply(
        self,
        *,
        sites: np.ndarray,
        fe_positions: np.ndarray,
        h_indices: np.ndarray,
        h_unwrapped_positions: np.ndarray,
        elapsed_time_s: float,
    ) -> DevanathanUpdate:
        current_indices = [int(x) for x in np.asarray(h_indices, dtype=int)]
        current_unwrapped = [
            np.asarray(p, dtype=float)
            for p in np.asarray(h_unwrapped_positions, dtype=float)
        ]
        if len(current_indices) != len(current_unwrapped):
            raise ValueError("h_indices and h_unwrapped_positions length mismatch")

        deleted_left_sink: list[int] = []
        deleted_sink: list[int] = []
        kept_indices: list[int] = []
        kept_unwrapped: list[np.ndarray] = []
        for site, pos in zip(current_indices, current_unwrapped):
            site_x = float(sites[int(site), 0])
            if self._is_left_sink_site(int(site), site_x):
                deleted_left_sink.append(int(site))
            elif site_x >= self.config.sink_x_min_a:
                deleted_sink.append(int(site))
            else:
                kept_indices.append(int(site))
                kept_unwrapped.append(pos)

        self.cumulative_removed += len(deleted_sink)
        self.cumulative_left_removed += len(deleted_left_sink)

        out_indices = np.asarray(kept_indices, dtype=int)
        out_unwrapped = (
            np.asarray(kept_unwrapped, dtype=float).reshape((len(kept_unwrapped), 3))
            if kept_unwrapped
            else np.empty((0, 3), dtype=float)
        )
        source_set = set(self.source_sites(sites))
        source_fe_atoms = self.source_fe_count(fe_positions)
        source_occupied = sum(1 for site in out_indices if int(site) in source_set)
        sink_occupied = sum(
            1 for site in out_indices if float(sites[int(site), 0]) >= self.config.sink_x_min_a
        )
        left_sink_occupied = sum(
            1
            for site in out_indices
            if self._is_left_sink_site(int(site), float(sites[int(site), 0]))
        )
        flux = None
        if elapsed_time_s > 0.0:
            flux = self.cumulative_removed / self.config.cross_section_area_a2 / elapsed_time_s

        return DevanathanUpdate(
            h_indices=out_indices,
            h_unwrapped_positions=out_unwrapped,
            inserted_sites=(),
            deleted_sink_sites=tuple(deleted_sink),
            trimmed_source_sites=(),
            source_sites=len(source_set),
            source_fe_atoms=source_fe_atoms,
            source_target_h=self.initial_count,
            source_occupied=source_occupied,
            sink_occupied=sink_occupied,
            total_occupied=len(out_indices),
            cumulative_inserted=self.cumulative_inserted,
            cumulative_removed=self.cumulative_removed,
            flux_per_a2_s=flux,
            deleted_left_sink_sites=tuple(deleted_left_sink),
            cumulative_left_removed=self.cumulative_left_removed,
            left_sink_occupied=left_sink_occupied,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "rng_state": self.rng.getstate(),
            "cumulative_inserted": int(self.cumulative_inserted),
            "cumulative_removed": int(self.cumulative_removed),
            "cumulative_left_removed": int(self.cumulative_left_removed),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)  # type: ignore[arg-type]
        self.cumulative_inserted = int(state.get("cumulative_inserted", 0))
        self.cumulative_removed = int(state.get("cumulative_removed", 0))
        self.cumulative_left_removed = int(state.get("cumulative_left_removed", 0))


def filter_fixed_x_neighbors(
    neighbors: list[list[int]] | np.ndarray,
    sites: np.ndarray,
    box: np.ndarray,
) -> list[list[int]]:
    """Drop neighbor links that wrap across the x periodic seam."""
    positions = np.asarray(sites, dtype=float)
    box_x = float(np.asarray(box, dtype=float)[0])
    filtered: list[list[int]] = []
    for src, row in enumerate(neighbors):
        kept: list[int] = []
        for dst in row:
            dx = abs(float(positions[int(dst), 0] - positions[int(src), 0]))
            if dx <= 0.5 * box_x:
                kept.append(int(dst))
        filtered.append(kept)
    return filtered


def filter_x_direction_neighbors(
    neighbors: list[list[int]] | np.ndarray,
    sites: np.ndarray,
    direction: str,
    *,
    eps_a: float = 1.0e-9,
) -> list[list[int]]:
    """Keep only neighbor links whose raw x displacement matches direction."""
    token = str(direction or "all").strip().lower().replace("-", "_")
    if token in {"", "all", "both", "bidirectional", "none"}:
        return [[int(dst) for dst in row] for row in neighbors]
    if token in {"right", "positive", "positive_x", "+x", "forward"}:
        sign = 1
    elif token in {"left", "negative", "negative_x", "-x", "backward"}:
        sign = -1
    else:
        raise ValueError(
            "KMC_HOP_X_DIRECTION must be one of all, right/positive, or left/negative"
        )

    positions = np.asarray(sites, dtype=float)
    filtered: list[list[int]] = []
    for src, row in enumerate(neighbors):
        src_x = float(positions[int(src), 0])
        kept: list[int] = []
        for dst in row:
            dx = float(positions[int(dst), 0] - src_x)
            if sign > 0 and dx > eps_a:
                kept.append(int(dst))
            elif sign < 0 and dx < -eps_a:
                kept.append(int(dst))
        filtered.append(kept)
    return filtered
