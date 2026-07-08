"""
Command-line entry point for the MPI KMC/NEB driver.

Run with, for example:
    mpirun -np 4 python -m kmc.main

Configuration is loaded from environment variables and optional config files by
``kmc.config``. The resulting ``KMCSimulation`` owns initialization, MPI job
scheduling, NEB barrier evaluation, event selection, checkpointing, and output
generation.
"""
from __future__ import annotations

try:
    from .kmc_driver import build_simulation_from_env
except ImportError:
    from kmc_driver import build_simulation_from_env


def main() -> None:
    sim = build_simulation_from_env()
    sim.run()


if __name__ == "__main__":
    main()
