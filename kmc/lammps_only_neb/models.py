from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HopSpec:
    """One candidate hop from an occupied interstitial site to a neighbor."""

    h_site: int
    n_site: int
    slot: int
    initial: np.ndarray
    final: np.ndarray
    hop_key: Any = None


@dataclass(frozen=True)
class EnvironmentNEBBatch:
    """Grouped NEB work for all neighbor hops from one local environment."""

    env_key: Any
    h_site: int
    h_position: np.ndarray
    hops: tuple[HopSpec, ...]
    sites: np.ndarray
    fe_positions: np.ndarray
    box: np.ndarray
    lattice_a: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LammpsSystem:
    """Local LAMMPS data for one hop."""

    positions: np.ndarray
    types: np.ndarray
    box: np.ndarray
    h_id: int
    final_positions: dict[int, np.ndarray]
    frozen_ids: tuple[int, ...]
    boundary: str
    mode: str
    n_fe: int
    n_h: int = 1


@dataclass(frozen=True)
class HopResult:
    """Result for one native-LAMMPS NEB hop."""

    env_key: Any
    hop_key: Any
    h_site: int
    n_site: int
    slot: int
    barrier_eV: float
    ebf: float | None
    ebr: float | None
    wall_s: float
    atoms: int
    frozen: int
    run_dir: Path
    io_write_s: float = 0.0
    endpoint_opt_s: float = 0.0
    lammps_run_s: float = 0.0
    io_parse_s: float = 0.0
    io_cleanup_s: float = 0.0
