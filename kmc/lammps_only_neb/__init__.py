"""LAMMPS-only NEB development package.

This package holds the grouped native-LAMMPS NEB path separately from the
current ASE-backed production KMC implementation.
"""

from .config import LammpsOnlyNEBConfig
from .models import EnvironmentNEBBatch, HopSpec, HopResult
from .scheduler import group_id_for_rank, split_group_comm, slots_for_group

__all__ = [
    "EnvironmentNEBBatch",
    "HopResult",
    "HopSpec",
    "LammpsOnlyNEBConfig",
    "group_id_for_rank",
    "slots_for_group",
    "split_group_comm",
]
