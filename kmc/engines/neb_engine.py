# engines/neb_engine.py
from typing import Any, Protocol, runtime_checkable

import os
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.mep import NEB

_DEFAULT_FIX_FE = False  # do NOT fix Fe by default
_DEFAULT_FIX_OTHER_H = True
_FALSES = {"0", "false", "False", "FALSE", "no", "No", "NO", "off", "OFF"}


def _env_flag(name: str, default_bool: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default_bool
    return val not in _FALSES


def build_fix_mask(at: Atoms, mover_idx=None):
    fix_fe = _env_flag("FIX_FE", _DEFAULT_FIX_FE)
    fix_other_h = _env_flag("FIX_OTHER_H", _DEFAULT_FIX_OTHER_H)
    Z = at.get_atomic_numbers()
    is_H = (Z == 1)
    is_Fe = (Z == 26)
    mask = np.zeros(len(at), dtype=bool)
    if fix_other_h:
        h_idx = np.where(is_H)[0]
        mask[h_idx] = True
        if mover_idx is not None and 0 <= mover_idx < len(at) and is_H[mover_idx]:
            mask[mover_idx] = False
    # per request: Fe are NOT fixed unless FIX_FE=1
    if fix_fe:
        mask |= is_Fe
    return mask


def build_neb_images(initial: Atoms, final: Atoms, n_images: int, mover_idx=None):
    images = [initial.copy()] + [initial.copy() for _ in range(n_images - 2)] + [final.copy()]
    neb = NEB(images)
    mask = build_fix_mask(images[0], mover_idx=mover_idx)
    if mask.any():
        for im in images:
            im.set_constraint(FixAtoms(mask=mask))
        neb.interpolate(mic=True, apply_constraint=True)
    else:
        neb.interpolate(mic=True)
    return images, neb


@runtime_checkable
class INEBEngine(Protocol):
    def barrier(self, initial: Atoms, final: Atoms, *, rank: int, **kwargs: Any) -> float:
        """Compute a NEB barrier (eV). Extra kwargs are engine-defined."""
