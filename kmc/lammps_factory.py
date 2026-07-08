"""
LAMMPS calculator factory.

This module builds ASE LAMMPSlib calculators from resolved configuration.
It intentionally does not import MPI at module import time except inside
the factory function, so config and tests can import this file safely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, List
import os


@dataclass(frozen=True)
class LammpsCalculatorConfig:
    potential: str = "EAM"
    potential_eam_file: str = "PotentialB3410-modified.fs"
    potential_nnp_dir: str = ""
    lammps_engine: str = "eam_fs"

    lammps_pair_style: str = ""
    lammps_pair_coeff: str = ""
    lammps_files: str = ""

    nnp_dir: str = ""
    nnp_elements: str = "Fe H"
    nnp_cutoff: float = 6.60
    nnp_showewsum: int = 1000
    nnp_maxew: int = 1000000
    nnp_cflength: float = 1.8897261328
    nnp_cfenergy: float = 0.0367493254


def _parse_lammps_files(raw: str | Sequence[str] | None, *, base_dir: str = ".") -> List[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        values = [str(x).strip() for x in raw if str(x).strip()]
    else:
        values = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
    return [v if os.path.isabs(v) else os.path.join(base_dir, v) for v in values]


def build_lammps_commands(cfg: LammpsCalculatorConfig, *, base_dir: str = ".") -> tuple[list[str], list[str], str]:
    """
    Return (lmpcmds, files, potential_label).
    """
    potential = (cfg.potential or "EAM").strip().upper()

    if cfg.lammps_pair_style and cfg.lammps_pair_coeff:
        lmpcmds = [cfg.lammps_pair_style, cfg.lammps_pair_coeff]
        files = _parse_lammps_files(cfg.lammps_files, base_dir=base_dir)
        return lmpcmds, files, "CUSTOM"

    if potential == "NNP":
        nnp_dir = cfg.nnp_dir or cfg.potential_nnp_dir
        lmpcmds = [
            (
                f'pair_style hdnnp {cfg.nnp_cutoff:.2f} dir "{nnp_dir}" '
                f"showew no showewsum {int(cfg.nnp_showewsum)} resetew yes "
                f"maxew {int(cfg.nnp_maxew)} cflength {cfg.nnp_cflength} "
                f"cfenergy {cfg.nnp_cfenergy}"
            ),
            f"pair_coeff * * {cfg.nnp_elements}",
        ]
        return lmpcmds, [], "NNP"

    eam_file = cfg.potential_eam_file
    if eam_file and not os.path.isabs(eam_file):
        eam_file = os.path.join(base_dir, eam_file)

    lmpcmds = ["pair_style eam/fs", f"pair_coeff * * {eam_file} Fe H"]
    files = _parse_lammps_files(cfg.lammps_files, base_dir=base_dir)
    if eam_file and eam_file not in files:
        files.append(eam_file)
    return lmpcmds, files, "EAM"


class LammpsCalculatorFactory:
    """
    Factory object that creates independent LAMMPSlib calculators.

    A logger can be supplied to print the chosen potential once.
    """

    def __init__(
        self,
        cfg: LammpsCalculatorConfig,
        *,
        base_dir: str = ".",
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.base_dir = base_dir
        self.logger = logger
        self._logged = False

    def create(self):
        from mpi4py import MPI  # type: ignore
        from ase.calculators.lammpslib import LAMMPSlib  # type: ignore

        lmpcmds, files, potential_label = build_lammps_commands(self.cfg, base_dir=self.base_dir)

        if self.logger is not None and not self._logged:
            self.logger(f"POTENTIAL='{potential_label}' using commands: {lmpcmds}")
            self._logged = True

        return LAMMPSlib(
            lmpcmds=lmpcmds,
            atom_types={"Fe": 1, "H": 2},
            files=files,
            keep_alive=True,
            comm=MPI.COMM_SELF,
        )

    def create_pool(self, n: int):
        return [self.create() for _ in range(int(n))]
