from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE_UNIFIED = ROOT / "Site_unified"
sys.path.insert(0, str(SITE_UNIFIED))

from lammps_data_utils import read_lammps_atomic_data, select_atom_indices  # noqa: E402


def test_parser_reads_full_style_with_image_flags(tmp_path: Path) -> None:
    data_file = tmp_path / "poly_full.lmp"
    data_file.write_text(
        """LAMMPS data file

2 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # full

2 1 1 -0.100 1.0 2.0 3.0 0 0 0
5 1 2  0.200 4.0 5.0 6.0 1 0 -1
"""
    )

    parsed = read_lammps_atomic_data(data_file)

    assert parsed.atom_style == "full"
    np.testing.assert_array_equal(parsed.ids, np.array([2, 5]))
    np.testing.assert_array_equal(parsed.types, np.array([1, 2]))
    np.testing.assert_allclose(parsed.pos, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_parser_infers_atomic_rows_with_image_flags(tmp_path: Path) -> None:
    data_file = tmp_path / "atomic_images.lmp"
    data_file.write_text(
        """LAMMPS data file

1 atoms
1 atom types

-5.0 5.0 xlo xhi
-5.0 5.0 ylo yhi
-5.0 5.0 zlo zhi

Atoms

10 1 1.5 2.5 3.5 0 0 -1
"""
    )

    parsed = read_lammps_atomic_data(data_file)

    np.testing.assert_array_equal(parsed.ids, np.array([10]))
    np.testing.assert_array_equal(parsed.types, np.array([1]))
    np.testing.assert_allclose(parsed.pos, np.array([[1.5, 2.5, 3.5]]))


def test_parser_infers_unlabeled_full_style_from_row_width(tmp_path: Path) -> None:
    data_file = tmp_path / "unlabeled_full.lmp"
    data_file.write_text(
        """LAMMPS data file

1 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms

4 99 1 -0.250 7.0 8.0 9.0 0 0 0
"""
    )

    parsed = read_lammps_atomic_data(data_file)

    np.testing.assert_array_equal(parsed.ids, np.array([4]))
    np.testing.assert_array_equal(parsed.types, np.array([1]))
    np.testing.assert_allclose(parsed.pos, np.array([[7.0, 8.0, 9.0]]))


def test_parser_rejects_triclinic_box_for_orthogonal_algorithms(tmp_path: Path) -> None:
    data_file = tmp_path / "tilted.lmp"
    data_file.write_text(
        """LAMMPS data file

1 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi
0.5 0.0 0.0 xy xz yz

Atoms # atomic

1 1 1.0 2.0 3.0
"""
    )

    with pytest.raises(ValueError, match="tilted triclinic"):
        read_lammps_atomic_data(data_file)


def test_select_atom_indices_supports_multiple_grain_label_types() -> None:
    types = np.array([1, 2, 3, 4, 2])

    np.testing.assert_array_equal(select_atom_indices(types, atom_type=0), np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(select_atom_indices(types, atom_type=1), np.array([0]))
    np.testing.assert_array_equal(select_atom_indices(types, atom_types="2,4"), np.array([1, 3, 4]))
