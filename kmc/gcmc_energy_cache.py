"""Persistent cache for local GCMC source insertion energies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .lattice import pbc_diff
    from .services.cache import BarrierCache
except ImportError:  # pragma: no cover - direct script execution fallback
    from lattice import pbc_diff
    from services.cache import BarrierCache


def canonical_insertion_environment_key(
    *,
    sites: np.ndarray,
    lower_occupancy_indices: Sequence[int],
    trial_site: int,
    box: Sequence[float],
    lattice_a: float,
    shell_inner_a: float,
    shell_outer_a: float,
    position_bin_a: float,
    potential_tag: Any,
    host_geometry_tag: Any = None,
) -> tuple[Any, ...]:
    """Build a translation-invariant key for E(with H)-E(without H).

    ``lower_occupancy_indices`` must describe the state without the trial H.
    The same key therefore serves an insertion and its reverse deletion.
    """
    positions = np.asarray(sites, dtype=float)
    center = np.asarray(positions[int(trial_site)], dtype=float)
    box_arr = np.asarray(box, dtype=float)
    position_bin_a = float(position_bin_a)
    if position_bin_a <= 0.0:
        raise ValueError("GCMC energy-cache position bin must be positive")

    displacements: list[tuple[int, int, int]] = []
    for occupied_site in map(int, lower_occupancy_indices):
        if occupied_site == int(trial_site):
            continue
        displacement = pbc_diff(center, positions[occupied_site], box_arr)
        if float(np.linalg.norm(displacement)) > float(shell_outer_a) + 1.0e-8:
            continue
        displacements.append(
            tuple(
                int(value)
                for value in np.round(displacement / position_bin_a).astype(int)
            )
        )

    site_basis = tuple(
        int(value)
        for value in np.round(
            np.mod(center, float(lattice_a)) / position_bin_a
        ).astype(int)
    )
    box_key = tuple(round(float(value), 12) for value in box_arr)
    # Translation reuse is unsafe when a shell reaches across half of a box
    # dimension because wrapped cells can deduplicate differently by center.
    undersized_box_site = (
        int(trial_site)
        if float(np.min(box_arr)) <= 2.0 * float(shell_outer_a) + 1.0e-8
        else None
    )
    common = (
        round(float(lattice_a), 12),
        box_key,
        round(float(shell_inner_a), 12),
        round(float(shell_outer_a), 12),
        round(position_bin_a, 12),
        potential_tag,
        undersized_box_site,
        site_basis,
        tuple(sorted(displacements)),
    )
    if host_geometry_tag is None:
        return ("GCMC_SOURCE_INSERTION_DE_V1", *common)
    return (
        "GCMC_SOURCE_INSERTION_DE_V2_HOST",
        *common[:6],
        host_geometry_tag,
        *common[6:],
    )


@dataclass
class GCMCEnergyCacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0


class PersistentGCMCEnergyCache:
    """Pickle-backed cache storing canonical insertion energy differences."""

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        autosave_interval: int = 100,
    ) -> None:
        self.path = Path(path).expanduser()
        self.cache = BarrierCache(str(self.path), enabled=enabled)
        self.enabled = bool(enabled)
        self.autosave_interval = max(0, int(autosave_interval))
        self.stats = GCMCEnergyCacheStats()

    def get(self, key: Any) -> float | None:
        if not self.enabled or key not in self.cache:
            self.stats.misses += 1
            return None
        value = float(self.cache[key])
        if not np.isfinite(value):
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return value

    def set(self, key: Any, insertion_delta_energy_ev: float) -> None:
        value = float(insertion_delta_energy_ev)
        if not self.enabled or not np.isfinite(value):
            return
        self.cache[key] = value
        self.stats.stores += 1
        if (
            self.autosave_interval > 0
            and self.stats.stores % self.autosave_interval == 0
        ):
            self.save()

    def save(self, *, full: bool = False) -> None:
        self.cache.save(full=full)

    def __len__(self) -> int:
        return len(self.cache)
