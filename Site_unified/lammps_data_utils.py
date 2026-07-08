#!/usr/bin/env python3
"""Shared LAMMPS data parsing helpers for the Site_unified pipeline.

The site-discovery algorithms in this directory use orthogonal periodic
minimum-image distances. The parser accepts common LAMMPS Atoms styles, but it
rejects tilted triclinic boxes with a clear error instead of silently producing
incorrect distances.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LammpsBox:
    bounds: np.ndarray
    tilt: np.ndarray

    @property
    def lo(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def hi(self) -> np.ndarray:
        return self.bounds[:, 1]

    @property
    def lengths(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def is_orthogonal(self) -> bool:
        return bool(np.allclose(self.tilt, 0.0))

    def require_orthogonal(self, source: str) -> None:
        if not self.is_orthogonal:
            xy, xz, yz = self.tilt
            raise ValueError(
                f"{source} uses a tilted triclinic box (xy={xy}, xz={xz}, yz={yz}). "
                "Site_unified currently requires an orthogonalized LAMMPS data file "
                "because its KD-tree and cell-list searches use orthogonal periodic "
                "minimum-image distances."
            )


@dataclass
class LammpsData:
    ids: np.ndarray
    types: np.ndarray
    pos: np.ndarray
    box: np.ndarray
    atom_style: str = "unknown"
    tilt: np.ndarray | None = None


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_atoms_style(header: str) -> str:
    if "#" not in header:
        return "unknown"
    comment = header.split("#", 1)[1].strip().lower()
    return comment.split()[0] if comment else "unknown"


def read_lammps_box(path: str | Path, require_orthogonal: bool = True) -> LammpsBox:
    xlo = xhi = ylo = yhi = zlo = zhi = None
    tilt = np.zeros(3, dtype=float)
    source = str(path)

    for line in Path(path).read_text().splitlines():
        parts = _strip_comment(line).split()
        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            xlo, xhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            ylo, yhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            zlo, zhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 6 and parts[-3:] == ["xy", "xz", "yz"]:
            tilt = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"Could not read x/y/z box bounds from {source}")
    bounds = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]], dtype=float)
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"Invalid box bounds in {source}: {bounds}")

    box = LammpsBox(bounds=bounds, tilt=tilt)
    if require_orthogonal:
        box.require_orthogonal(source)
    return box


def read_lammps_box_bounds(path: str | Path, require_orthogonal: bool = True) -> np.ndarray:
    return read_lammps_box(path, require_orthogonal=require_orthogonal).bounds.copy()


def select_atom_indices(types: np.ndarray, atom_type: int = 1, atom_types: str | None = None) -> np.ndarray:
    """Select atoms by one type, many comma-separated types, or all types.

    Passing `atom_type <= 0` selects every atom. This is useful for Fe
    bicrystal/polycrystal structures where separate atom types are used only to
    tag grains, not chemical species.
    """
    types = np.asarray(types, dtype=int)
    if atom_types:
        selected = np.array([int(x.strip()) for x in atom_types.split(",") if x.strip()], dtype=int)
        if selected.size == 0:
            raise ValueError("atom_types was provided but no atom types were parsed")
        return np.where(np.isin(types, selected))[0]
    if atom_type <= 0:
        return np.arange(len(types), dtype=int)
    return np.where(types == atom_type)[0]


def _known_style_columns(style: str) -> Optional[Tuple[int, int]]:
    """Return (type_col, coord_start_col) for known LAMMPS atom styles."""
    aliases = {
        "atomic": (1, 2),
        "charge": (1, 3),
        "molecular": (2, 3),
        "bond": (2, 3),
        "angle": (2, 3),
        "full": (2, 4),
    }
    return aliases.get(style.lower())


def _fallback_columns(n_cols: int) -> list[Tuple[int, int, str]]:
    """Likely atom-style columns for unlabeled Atoms sections.

    Optional image flags add three trailing integer columns, so atomic/full rows
    often have 8/10 columns respectively.
    """
    if n_cols in (5, 8):
        return [(1, 2, "atomic"), (1, 3, "charge"), (2, 3, "molecular"), (2, 4, "full")]
    if n_cols in (7, 10):
        return [(2, 4, "full"), (1, 2, "atomic"), (1, 3, "charge"), (2, 3, "molecular")]
    if n_cols in (6, 9):
        return [(1, 3, "charge"), (2, 3, "molecular"), (1, 2, "atomic"), (2, 4, "full")]
    return [(1, 2, "atomic"), (1, 3, "charge"), (2, 3, "molecular"), (2, 4, "full")]


def _coord_score(xyz: np.ndarray, bounds: np.ndarray) -> float:
    lengths = bounds[:, 1] - bounds[:, 0]
    pad = np.maximum(5.0, 0.25 * lengths)
    lo = bounds[:, 0] - pad
    hi = bounds[:, 1] + pad
    outside = np.maximum(lo - xyz, 0.0) + np.maximum(xyz - hi, 0.0)
    return float(np.linalg.norm(outside / np.maximum(lengths, 1.0)))


def _parse_atom_row(parts: Iterable[str], style: str, bounds: np.ndarray) -> Tuple[int, int, np.ndarray]:
    tokens = list(parts)
    if len(tokens) < 5:
        raise ValueError("Atom row has fewer than 5 columns")
    vals = [float(x) for x in tokens]
    aid = int(vals[0])

    known = _known_style_columns(style)
    candidates = []
    if known is not None:
        candidates.append((known[0], known[1], style))
    candidates.extend(_fallback_columns(len(vals)))

    best = None
    best_score = float("inf")
    best_rank = 10**6
    seen = set()
    for rank, (type_col, coord_col, _label) in enumerate(candidates):
        key = (type_col, coord_col)
        if key in seen:
            continue
        seen.add(key)
        if len(vals) <= coord_col + 2 or len(vals) <= type_col:
            continue
        try:
            atype_float = vals[type_col]
            atype = int(atype_float)
        except Exception:
            continue
        if abs(atype_float - atype) > 1e-8:
            continue
        xyz = np.array(vals[coord_col : coord_col + 3], dtype=float)
        score = _coord_score(xyz, bounds)
        if score < best_score or (score == best_score and rank < best_rank):
            best = (atype, xyz)
            best_score = score
            best_rank = rank

    if best is None:
        raise ValueError(f"Could not infer atom type/coordinates from row: {' '.join(tokens)}")
    return aid, best[0], best[1]


def read_lammps_atomic_data(path: str | Path, require_orthogonal: bool = True) -> LammpsData:
    source = str(path)
    lines = Path(path).read_text().splitlines()
    box_info = read_lammps_box(path, require_orthogonal=require_orthogonal)

    n_atoms = None
    for line in lines[:500]:
        parts = _strip_comment(line).split()
        if len(parts) >= 2 and parts[-1] == "atoms":
            n_atoms = int(parts[0])
            break

    atoms_header = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_header = i
            break
    if atoms_header is None:
        raise ValueError(f"Could not find Atoms section in {source}")

    atom_style = _parse_atoms_style(lines[atoms_header])
    ids = []
    types = []
    pos = []

    for line in lines[atoms_header + 1 :]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        first = s.split(None, 1)[0]
        if not first.lstrip("+-").isdigit():
            if ids:
                break
            continue
        parts = _strip_comment(s).split()
        if len(parts) < 5:
            continue
        aid, atype, xyz = _parse_atom_row(parts, atom_style, box_info.bounds)
        ids.append(aid)
        types.append(atype)
        pos.append(xyz)
        if n_atoms is not None and len(ids) >= n_atoms:
            break

    if not ids:
        raise ValueError(f"No atoms parsed from Atoms section in {source}")
    if n_atoms is not None and len(ids) != n_atoms:
        raise ValueError(f"Header says {n_atoms} atoms, but parsed {len(ids)} atoms from {source}")

    return LammpsData(
        ids=np.asarray(ids, dtype=int),
        types=np.asarray(types, dtype=int),
        pos=np.asarray(pos, dtype=float),
        box=box_info.bounds.copy(),
        atom_style=atom_style,
        tilt=box_info.tilt.copy(),
    )


def wrap_positions(pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    lo = box[:, 0]
    lengths = box[:, 1] - box[:, 0]
    return lo + np.mod(np.asarray(pos, dtype=float) - lo, lengths)
