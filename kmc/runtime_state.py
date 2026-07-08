"""Small mutable runtime containers for KMC/NEB simulations.

The production driver mostly stores state on ``KMCSimulation`` itself, but this
container remains useful for services and compatibility paths that need to pass
simulation state explicitly. It groups changing objects such as site arrays,
occupied H indices, unwrapped H positions, NEB engine handles, and diagnostic
sequence counters without falling back to module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Any, Optional


@dataclass
class RuntimeState:
    """Mutable state that changes during a simulation run."""

    sites: Any = None
    full_structure: Any = None
    fe_3a_template: Any = None

    h_positions: Any = None
    h_indices: Any = None
    h_unwrapped_positions: Any = None
    h_initial_unwrapped_positions: Any = None
    h_cell_index: Optional[dict[tuple[int, int, int], list[tuple[Any, Optional[int]]]]] = None

    neb_engine: Any = None
    lammps_engine_logged: bool = False

    mpi_seq_counter: Any = field(default_factory=count)
    ase_log_seq: Any = field(default_factory=count)
    struct_traj_seq: Any = field(default_factory=count)
    struct_snap_seq: Any = field(default_factory=count)

    def reset_neb_engine(self) -> None:
        self.neb_engine = None

    def next_mpi_seq(self) -> int:
        return int(next(self.mpi_seq_counter))

    def next_ase_log_seq(self) -> int:
        return int(next(self.ase_log_seq))

    def next_struct_traj_seq(self) -> int:
        return int(next(self.struct_traj_seq))

    def next_struct_snap_seq(self) -> int:
        return int(next(self.struct_snap_seq))


__all__ = ["RuntimeState"]
