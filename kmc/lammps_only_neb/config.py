from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LammpsOnlyNEBConfig:
    """Configuration for grouped native-LAMMPS NEB execution."""

    replicas_per_neb: int = 3
    spring_constant: float = 5.0
    force_tolerance: float = 1.0e-5
    max_steps: int = 1000
    climbing_steps: int = 0
    nevery: int = 50
    thermo_every: int = 50
    dump_every: int = 50
    timestep: float = 0.01
    min_style: str = "quickmin"
    debug_mode: bool = False

    endpoint_optimize: bool = False
    endpoint_min_style: str = "quickmin"
    endpoint_energy_tolerance: float = 0.0
    endpoint_force_tolerance: float = 0.05
    endpoint_max_steps: int = 200
    endpoint_max_evals: int = 2000
    endpoint_dmax: float = 0.1

    shell_inner_radius_a: float = 8.0
    shell_outer_radius_a: float = 10.0
    shell_pad_a: float = 2.0

    potential_file: Path = Path("kmc/PotentialB3410-modified.fs")
    work_dir: Path = Path("development/lammps_only_neb_runs")
    lammps_python_path: Path | None = None
    lammps_lib_dir: Path | None = None

    def normalized(self) -> "LammpsOnlyNEBConfig":
        if self.replicas_per_neb < 2:
            raise ValueError("replicas_per_neb must be at least 2")
        if self.force_tolerance <= 0.0:
            raise ValueError("force_tolerance must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.min_style not in {"quickmin", "fire"}:
            raise ValueError("min_style must be 'quickmin' or 'fire'")
        if self.endpoint_min_style not in {"quickmin", "fire", "cg", "sd", "hftn"}:
            raise ValueError("endpoint_min_style must be one of quickmin, fire, cg, sd, hftn")
        if self.endpoint_force_tolerance < 0.0:
            raise ValueError("endpoint_force_tolerance must be non-negative")
        if self.endpoint_energy_tolerance < 0.0:
            raise ValueError("endpoint_energy_tolerance must be non-negative")
        if self.endpoint_max_steps <= 0:
            raise ValueError("endpoint_max_steps must be positive")
        if self.endpoint_max_evals <= 0:
            raise ValueError("endpoint_max_evals must be positive")
        if self.endpoint_dmax <= 0.0:
            raise ValueError("endpoint_dmax must be positive")
        if self.shell_outer_radius_a < self.shell_inner_radius_a:
            raise ValueError("shell_outer_radius_a must be >= shell_inner_radius_a")
        return self
