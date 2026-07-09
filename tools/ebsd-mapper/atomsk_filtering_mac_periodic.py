#!/usr/bin/env python3
"""
Mac/Linux EBSD grain-list -> Atomsk polycrystal helper, with optional
orientation-aware periodic box sizing.

Compared with the first macOS version:
  - Keeps the fixed centroid normalization: (x - x_min) / (x_max - x_min).
  - Keeps all generated files inside an output directory.
  - Adds --periodic-sizing:
      Treats --target-width and --box-z as initial guesses, then snaps the
      actual Atomsk box dimensions to nearby values that are more compatible
      with crystallographic repeat lengths estimated from the selected grains'
      Euler orientations.
  - Writes periodicity_report.csv when --periodic-sizing is enabled.

Important scientific note:
  For arbitrary EBSD Euler orientations, an exactly periodic rectangular box
  that is simultaneously commensurate with every grain usually does not exist.
  The implemented algorithm is therefore a practical approximation: it finds
  the closest low-index crystal repeat direction to each global box axis for
  every selected grain, then chooses a box length near the user-supplied initial
  guess that minimizes the integer-multiple mismatch against those repeat
  lengths. The residual mismatch and angular mismatch are written to the report.

Examples:
  Dry run without Atomsk:
    python3 atomsk_filtering_mac_periodic.py grains.csv --target-width 1000 --skip-atomsk

  Periodic-size dry run:
    python3 atomsk_filtering_mac_periodic.py grains.csv --target-width 1000 \
      --periodic-sizing --periodic-axes xy --skip-atomsk

  Run Atomsk after generating files:
    python3 atomsk_filtering_mac_periodic.py grains.csv --target-width 1000 \
      --periodic-sizing
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = [
    "ID",
    "Centroid X",
    "Centroid Y",
    "Area",
    "Mean Euler orientation 1",
    "Mean Euler orientation 2",
    "Mean Euler orientation 3",
]

NUMERIC_COLUMNS = REQUIRED_COLUMNS.copy()

AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def prompt_float(message: str, default: Optional[float] = None) -> float:
    """Prompt until a valid float is entered."""
    while True:
        if default is None:
            raw = input(f"{message}: ").strip()
        else:
            raw = input(f"{message} [{default}]: ").strip()
            if raw == "":
                return float(default)

        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def prompt_path(message: str) -> Path:
    """Prompt for a path and clean macOS/Windows drag-and-drop quoting."""
    raw = input(f"{message}: ").strip().replace('"', "").replace("'", "")
    return Path(raw).expanduser().resolve()


def read_ebsd_csv(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_csv}")

    data = pd.read_csv(input_csv)

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    # Convert required fields to numeric. This also removes the EBSD unit row
    # in files where the first data row contains µm, µm², degree symbols, etc.
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=NUMERIC_COLUMNS).copy()

    if data.empty:
        raise ValueError("No valid numeric grain rows were found after reading the CSV.")

    return data


def filter_grains(
    data: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    min_area: float,
) -> pd.DataFrame:
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min.")
    if min_area < 0:
        raise ValueError("min_area must be >= 0.")

    filtered = data[
        (data["Centroid X"] >= x_min)
        & (data["Centroid X"] <= x_max)
        & (data["Centroid Y"] >= y_min)
        & (data["Centroid Y"] <= y_max)
        & (data["Area"] >= min_area)
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No grains remain after applying the X/Y crop and minimum-area filter."
        )

    return filtered[REQUIRED_COLUMNS].copy()


def normalization_geometry(
    grains: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    normalization_base: str,
) -> dict:
    """Return origin, selected width/height, and aspect ratio for normalization."""
    if normalization_base == "crop":
        origin_x = x_min
        origin_y = y_min
        image_width = x_max - x_min
        image_height = y_max - y_min
    elif normalization_base == "filtered":
        origin_x = float(grains["Centroid X"].min())
        origin_y = float(grains["Centroid Y"].min())
        image_width = float(grains["Centroid X"].max() - grains["Centroid X"].min())
        image_height = float(grains["Centroid Y"].max() - grains["Centroid Y"].min())
    else:
        raise ValueError("normalization_base must be either 'crop' or 'filtered'.")

    if image_width <= 0:
        raise ValueError("The selected X range has zero width.")
    if image_height <= 0:
        raise ValueError("The selected Y range has zero height.")

    return {
        "normalization_base": normalization_base,
        "origin_x_um": float(origin_x),
        "origin_y_um": float(origin_y),
        "image_width_um": float(image_width),
        "image_height_um": float(image_height),
        "aspect_ratio_width_over_height": float(image_width / image_height),
    }


def normalize_and_scale(
    grains: pd.DataFrame,
    target_width: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    normalization_base: str = "crop",
    target_height: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Normalize centroids and scale them to the Atomsk target size.

    normalization_base="crop":
        Use the user-selected EBSD crop window as the normalization box.
        x_norm = (x - selected_x_min) / (selected_x_max - selected_x_min)

    normalization_base="filtered":
        Use the min/max of the remaining filtered grain centroids.
        This is closer to the original script, but corrected by subtracting min.

    If target_height is None, height is obtained from target_width while
    preserving the selected EBSD aspect ratio. If target_height is supplied,
    X and Y are scaled independently. This is useful after periodic sizing,
    where X and Y may snap to different commensurate lengths.
    """
    if target_width <= 0:
        raise ValueError("target_width must be greater than zero.")

    geom = normalization_geometry(grains, x_min, x_max, y_min, y_max, normalization_base)
    aspect_ratio = geom["aspect_ratio_width_over_height"]

    if target_height is None:
        target_height = target_width / aspect_ratio
    if target_height <= 0:
        raise ValueError("target_height must be greater than zero.")

    normal_data = grains.copy()

    # Fixed normalization: subtract the origin before dividing by the size.
    normal_data["Centroid X"] = (
        normal_data["Centroid X"] - geom["origin_x_um"]
    ) / geom["image_width_um"]
    normal_data["Centroid Y"] = (
        normal_data["Centroid Y"] - geom["origin_y_um"]
    ) / geom["image_height_um"]

    scaled_data = normal_data.copy()
    scaled_data["Centroid X"] = scaled_data["Centroid X"] * target_width
    scaled_data["Centroid Y"] = scaled_data["Centroid Y"] * target_height

    meta = {
        **geom,
        "target_width_angstrom": float(target_width),
        "target_height_angstrom": float(target_height),
        "target_aspect_ratio_width_over_height": float(target_width / target_height),
    }

    return normal_data, scaled_data, meta


def gcd3(a: int, b: int, c: int) -> int:
    return math.gcd(math.gcd(abs(a), abs(b)), abs(c))


def canonical_direction(u: int, v: int, w: int) -> tuple[int, int, int]:
    """Reduce [uvw] and choose a unique sign convention."""
    g = gcd3(u, v, w)
    if g == 0:
        raise ValueError("Zero crystallographic direction is invalid.")

    u, v, w = u // g, v // g, w // g

    # Make [uvw] and [-u -v -w] identical. The first non-zero component is positive.
    for value in (u, v, w):
        if value < 0:
            return (-u, -v, -w)
        if value > 0:
            return (u, v, w)
    return (u, v, w)


def repeat_length_cubic(
    direction: tuple[int, int, int],
    lattice_parameter: float,
    crystal_structure: str,
) -> float:
    """
    Conservative crystallographic repeat length along a cubic [uvw] direction.

    For bcc, the body-centering translation makes the repeat length half of
    a*[uvw] when the reduced u, v, w are all odd, e.g. [111]. Otherwise the
    conventional cubic repeat a*[uvw] is used. This is appropriate for BCC Fe.
    """
    u, v, w = canonical_direction(*direction)
    base_length = lattice_parameter * math.sqrt(u * u + v * v + w * w)
    structure = crystal_structure.lower().strip()

    if structure == "bcc" and all(abs(n) % 2 == 1 for n in (u, v, w)):
        return 0.5 * base_length

    return base_length


def enumerate_crystal_directions(
    max_index: int,
    lattice_parameter: float,
    crystal_structure: str,
) -> list[dict]:
    if max_index < 1:
        raise ValueError("periodic_max_index must be >= 1.")

    seen: set[tuple[int, int, int]] = set()
    directions: list[dict] = []

    for u in range(-max_index, max_index + 1):
        for v in range(-max_index, max_index + 1):
            for w in range(-max_index, max_index + 1):
                if u == 0 and v == 0 and w == 0:
                    continue
                primitive = canonical_direction(u, v, w)
                if primitive in seen:
                    continue
                seen.add(primitive)
                norm = math.sqrt(sum(n * n for n in primitive))
                directions.append(
                    {
                        "u": primitive[0],
                        "v": primitive[1],
                        "w": primitive[2],
                        "unit": tuple(n / norm for n in primitive),
                        "repeat_length_angstrom": repeat_length_cubic(
                            primitive, lattice_parameter, crystal_structure
                        ),
                    }
                )

    return directions


def bunge_matrix(phi1_deg: float, Phi_deg: float, phi2_deg: float) -> list[list[float]]:
    """
    Bunge ZXZ Euler orientation matrix.

    The matrix convention here is the common EBSD/Bunge form often used as a
    sample-to-crystal orientation matrix. The command-line option
    --euler-convention can switch whether this matrix or its transpose is used
    when projecting global box axes into the crystal frame.
    """
    phi1 = math.radians(phi1_deg)
    Phi = math.radians(Phi_deg)
    phi2 = math.radians(phi2_deg)

    c1 = math.cos(phi1)
    s1 = math.sin(phi1)
    c = math.cos(Phi)
    s = math.sin(Phi)
    c2 = math.cos(phi2)
    s2 = math.sin(phi2)

    return [
        [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
        [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
        [s1 * s, -c1 * s, c],
    ]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def mat_vec(matrix: list[list[float]], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[i][j] * vector[j] for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def unit_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return tuple(x / norm for x in vector)  # type: ignore[return-value]


def closest_direction(
    axis_in_crystal_frame: tuple[float, float, float],
    directions: list[dict],
) -> dict:
    axis = unit_vector(axis_in_crystal_frame)

    best = None
    best_abs_dot = -1.0
    for direction in directions:
        candidate = direction["unit"]
        dot = sum(axis[i] * candidate[i] for i in range(3))
        abs_dot = abs(dot)
        if abs_dot > best_abs_dot:
            best_abs_dot = abs_dot
            best = direction

    if best is None:
        raise RuntimeError("No crystallographic direction candidates were generated.")

    clamped = max(-1.0, min(1.0, best_abs_dot))
    angle_deg = math.degrees(math.acos(clamped))

    return {
        **best,
        "axis_crystal_x": axis[0],
        "axis_crystal_y": axis[1],
        "axis_crystal_z": axis[2],
        "angular_mismatch_deg": angle_deg,
    }


def validate_periodic_axes(axes: str) -> str:
    axes_clean = "".join(ch for ch in axes.lower().strip() if not ch.isspace())
    if not axes_clean:
        raise ValueError("periodic axes cannot be empty when --periodic-sizing is used.")
    invalid = [ch for ch in axes_clean if ch not in AXIS_VECTORS]
    if invalid:
        raise ValueError("--periodic-axes may only contain x, y, and/or z.")
    # Preserve x/y/z order and remove duplicates.
    return "".join(ch for ch in "xyz" if ch in axes_clean)


def compute_periodicity_rows(
    grains: pd.DataFrame,
    axes: str,
    lattice_parameter: float,
    crystal_structure: str,
    max_index: int,
    euler_convention: str,
) -> list[dict]:
    directions = enumerate_crystal_directions(max_index, lattice_parameter, crystal_structure)
    rows: list[dict] = []

    for _, grain in grains.iterrows():
        phi1 = float(grain["Mean Euler orientation 1"])
        Phi = float(grain["Mean Euler orientation 2"])
        phi2 = float(grain["Mean Euler orientation 3"])
        matrix = bunge_matrix(phi1, Phi, phi2)

        if euler_convention == "bunge-sample-to-crystal":
            sample_to_crystal = matrix
        elif euler_convention == "bunge-crystal-to-sample":
            sample_to_crystal = transpose(matrix)
        else:
            raise ValueError(
                "euler_convention must be 'bunge-sample-to-crystal' or 'bunge-crystal-to-sample'."
            )

        for axis_name in axes:
            axis_sample = AXIS_VECTORS[axis_name]
            axis_crystal = mat_vec(sample_to_crystal, axis_sample)
            best = closest_direction(axis_crystal, directions)

            rows.append(
                {
                    "grain_id": int(grain["ID"]),
                    "axis": axis_name,
                    "euler_1_deg": phi1,
                    "euler_2_deg": Phi,
                    "euler_3_deg": phi2,
                    "axis_crystal_x": best["axis_crystal_x"],
                    "axis_crystal_y": best["axis_crystal_y"],
                    "axis_crystal_z": best["axis_crystal_z"],
                    "closest_u": best["u"],
                    "closest_v": best["v"],
                    "closest_w": best["w"],
                    "angular_mismatch_deg": best["angular_mismatch_deg"],
                    "repeat_length_angstrom": best["repeat_length_angstrom"],
                }
            )

    return rows


def nearest_integer_residual(length: float, period: float) -> tuple[int, float, float]:
    n = max(1, int(round(length / period)))
    residual = abs(length - n * period)
    fractional_residual = residual / period
    return n, residual, fractional_residual


def choose_periodic_length(
    initial_guess: float,
    periods: list[float],
    search_fraction: float,
    size_penalty: float,
) -> dict:
    """
    Choose a length near initial_guess that best matches integer multiples of periods.

    Exact least-common-multiple is not meaningful for floating-point irrational
    lengths from arbitrary orientations, so we search nearby candidate multiples
    and minimize RMS residual, max residual, and distance from the initial guess.
    """
    if initial_guess <= 0:
        raise ValueError("Initial length guess must be positive.")
    if not periods:
        raise ValueError("No periods were supplied for periodic sizing.")
    if search_fraction < 0:
        raise ValueError("periodic_search_fraction must be >= 0.")
    if size_penalty < 0:
        raise ValueError("periodic_size_penalty must be >= 0.")

    lower = max(min(periods), initial_guess * (1.0 - search_fraction))
    upper = initial_guess * (1.0 + search_fraction)
    if upper <= lower:
        upper = lower + max(periods)

    candidates: dict[float, float] = {}
    for period in periods:
        n_min = max(1, int(math.floor(lower / period)) - 1)
        n_max = max(1, int(math.ceil(upper / period)) + 1)
        for n in range(n_min, n_max + 1):
            length = n * period
            if lower <= length <= upper:
                # Round only for de-duplication; keep the original float value.
                candidates[round(length, 6)] = length

    # Include the initial guess as a fallback, even if it is not a multiple of any period.
    candidates[round(initial_guess, 6)] = initial_guess

    best: Optional[dict] = None
    for length in candidates.values():
        residuals = [nearest_integer_residual(length, p)[1] for p in periods]
        rms_residual = math.sqrt(sum(r * r for r in residuals) / len(residuals))
        max_residual = max(residuals)
        mean_residual = sum(residuals) / len(residuals)
        size_delta = abs(length - initial_guess)

        # Periodicity dominates; size_penalty only breaks ties / avoids huge changes.
        score = rms_residual + 0.25 * max_residual + size_penalty * size_delta

        item = {
            "initial_guess_angstrom": float(initial_guess),
            "selected_length_angstrom": float(length),
            "size_change_angstrom": float(length - initial_guess),
            "size_change_percent": float(100.0 * (length - initial_guess) / initial_guess),
            "rms_residual_angstrom": float(rms_residual),
            "mean_residual_angstrom": float(mean_residual),
            "max_residual_angstrom": float(max_residual),
            "score": float(score),
            "search_lower_angstrom": float(lower),
            "search_upper_angstrom": float(upper),
            "candidate_count": int(len(candidates)),
        }

        if best is None or item["score"] < best["score"]:
            best = item

    if best is None:
        raise RuntimeError("Failed to choose a periodic length.")

    return best


def periodic_sizing(
    grains: pd.DataFrame,
    initial_box: dict,
    axes: str,
    lattice_parameter: float,
    crystal_structure: str,
    max_index: int,
    search_fraction: float,
    size_penalty: float,
    euler_convention: str,
) -> tuple[dict, list[dict], dict]:
    axes = validate_periodic_axes(axes)
    rows = compute_periodicity_rows(
        grains=grains,
        axes=axes,
        lattice_parameter=lattice_parameter,
        crystal_structure=crystal_structure,
        max_index=max_index,
        euler_convention=euler_convention,
    )

    final_box = dict(initial_box)
    axis_choices = {}

    for axis_name in axes:
        periods = [row["repeat_length_angstrom"] for row in rows if row["axis"] == axis_name]
        choice = choose_periodic_length(
            initial_guess=initial_box[axis_name],
            periods=periods,
            search_fraction=search_fraction,
            size_penalty=size_penalty,
        )
        final_box[axis_name] = choice["selected_length_angstrom"]
        axis_choices[axis_name] = choice

    for row in rows:
        length = final_box[row["axis"]]
        period = row["repeat_length_angstrom"]
        multiple, residual, fractional = nearest_integer_residual(length, period)
        row["final_box_length_angstrom"] = length
        row["final_nearest_multiple"] = multiple
        row["final_length_residual_angstrom"] = residual
        row["final_fractional_residual"] = fractional

    angle_values = [row["angular_mismatch_deg"] for row in rows]
    residual_values = [row["final_length_residual_angstrom"] for row in rows]

    summary = {
        "enabled": True,
        "axes": axes,
        "crystal_structure": crystal_structure,
        "lattice_parameter_angstrom": lattice_parameter,
        "periodic_max_index": max_index,
        "periodic_search_fraction": search_fraction,
        "periodic_size_penalty": size_penalty,
        "euler_convention": euler_convention,
        "initial_box_guess_angstrom": initial_box,
        "final_box_angstrom": final_box,
        "axis_choices": axis_choices,
        "max_angular_mismatch_deg": max(angle_values) if angle_values else None,
        "mean_angular_mismatch_deg": sum(angle_values) / len(angle_values) if angle_values else None,
        "max_final_length_residual_angstrom": max(residual_values) if residual_values else None,
        "mean_final_length_residual_angstrom": (
            sum(residual_values) / len(residual_values) if residual_values else None
        ),
    }

    return final_box, rows, summary


def write_polycrystal_txt(
    scaled_data: pd.DataFrame,
    output_file: Path,
    target_width: float,
    target_height: float,
    box_z: float,
    padding_fraction: float,
) -> dict:
    if box_z <= 0:
        raise ValueError("box_z must be greater than zero.")
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be >= 0.")

    box_x = target_width * (1.0 + padding_fraction)
    box_y = target_height * (1.0 + padding_fraction)
    z = box_z / 2.0

    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"box {box_x:.8f} {box_y:.8f} {box_z:.8f}\n")

        # Important fix: do not skip the first valid grain.
        for _, row in scaled_data.iterrows():
            x = row["Centroid X"]
            y = row["Centroid Y"]
            e1 = row["Mean Euler orientation 1"]
            e2 = row["Mean Euler orientation 2"]
            e3 = row["Mean Euler orientation 3"]
            f.write(f"node {x:.4f} {y:.4f} {z:.4f} {e1:.4f} {e2:.4f} {e3:.4f}\n")

    return {
        "box_x_angstrom": float(box_x),
        "box_y_angstrom": float(box_y),
        "box_z_angstrom": float(box_z),
        "node_z_angstrom": float(z),
        "padding_fraction": float(padding_fraction),
    }


def run_command(command: list[str], cwd: Path) -> None:
    print("\nRunning:", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def run_atomsk(
    output_dir: Path,
    atomsk_command: str,
    crystal_structure: str,
    lattice_parameter: float,
    element: str,
    output_cfg: str,
) -> None:
    atomsk_path = shutil.which(atomsk_command)
    if atomsk_path is None:
        raise FileNotFoundError(
            f"Could not find '{atomsk_command}' on PATH. Generated polycrystal.txt, "
            "but did not run Atomsk. On macOS, check that Atomsk is installed and "
            "that the command works from Terminal with: atomsk --version"
        )

    run_command(
        [atomsk_path, "--create", crystal_structure, str(lattice_parameter), element, "xsf"],
        cwd=output_dir,
    )

    unit_cell_file = f"{element}.xsf"
    run_command(
        [atomsk_path, "--polycrystal", unit_cell_file, "polycrystal.txt", output_cfg],
        cwd=output_dir,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an EBSD grain-list CSV into Atomsk polycrystal input on macOS/Linux."
    )
    parser.add_argument("input_csv", nargs="?", help="Path to EBSD grain-list CSV.")
    parser.add_argument("--x-min", type=float, help="Minimum Centroid X to include, in EBSD units, usually µm.")
    parser.add_argument("--x-max", type=float, help="Maximum Centroid X to include, in EBSD units, usually µm.")
    parser.add_argument("--y-min", type=float, help="Minimum Centroid Y to include, in EBSD units, usually µm.")
    parser.add_argument("--y-max", type=float, help="Maximum Centroid Y to include, in EBSD units, usually µm.")
    parser.add_argument("--min-area", type=float, help="Minimum grain area to include, usually µm^2.")
    parser.add_argument("--target-width", type=float, help="Initial target Atomsk width in Angstroms.")
    parser.add_argument("--output-dir", default="atomsk_output", help="Directory for generated files.")
    parser.add_argument("--normalization-base", choices=["crop", "filtered"], default="crop", help="Use selected crop window or filtered grain min/max for normalization. Default: crop.")
    parser.add_argument("--box-z", type=float, default=10.0, help="Initial Atomsk box thickness in Angstroms. Default: 10.")
    parser.add_argument("--padding-fraction", type=float, default=0.10, help="Extra box padding fraction in X and Y. Default: 0.10.")
    parser.add_argument("--lattice-parameter", type=float, default=2.856, help="Lattice parameter for Atomsk unit cell. Default: 2.856 Å.")
    parser.add_argument("--element", default="Fe", help="Element symbol for Atomsk. Default: Fe.")
    parser.add_argument("--crystal-structure", default="bcc", help="Crystal structure for Atomsk --create. Default: bcc.")
    parser.add_argument("--atomsk-command", default="atomsk", help="Atomsk command name/path. Default: atomsk.")
    parser.add_argument("--output-cfg", default="final.cfg", help="Final Atomsk output structure file. Default: final.cfg.")
    parser.add_argument("--skip-atomsk", action="store_true", help="Only write CSV/XLSX/polycrystal.txt files; do not run Atomsk.")

    parser.add_argument("--periodic-sizing", action="store_true", help="Treat target size as an initial guess and snap the Atomsk box to orientation-derived repeat lengths.")
    parser.add_argument("--periodic-axes", default="xy", help="Axes to snap when --periodic-sizing is used. Use x, y, z, xy, xyz, etc. Default: xy, because EBSD maps are 2D and box_z is usually a separate modelling choice.")
    parser.add_argument("--periodic-max-index", type=int, default=4, help="Maximum crystallographic direction index searched for closest [uvw]. Default: 4.")
    parser.add_argument("--periodic-search-fraction", type=float, default=0.25, help="Allowed fractional change around the initial box length during periodic snapping. Default: 0.25.")
    parser.add_argument("--periodic-size-penalty", type=float, default=0.02, help="Penalty for moving away from the initial size during periodic snapping. Larger values keep the size closer to the guess. Default: 0.02.")
    parser.add_argument("--euler-convention", choices=["bunge-sample-to-crystal", "bunge-crystal-to-sample"], default="bunge-sample-to-crystal", help="Orientation matrix convention used for periodicity analysis. Default: bunge-sample-to-crystal.")
    parser.add_argument("--angle-warning-deg", type=float, default=5.0, help="Warn if closest crystallographic axis mismatch is above this angle. Default: 5 degrees.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        input_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv else prompt_path("Please input the CSV file path")
        data = read_ebsd_csv(input_csv)

        global_x_min = float(data["Centroid X"].min())
        global_x_max = float(data["Centroid X"].max())
        global_y_min = float(data["Centroid Y"].min())
        global_y_max = float(data["Centroid Y"].max())
        global_area_min = float(data["Area"].min())
        global_area_max = float(data["Area"].max())

        print(f"Loaded {len(data)} valid grain rows from: {input_csv}")
        print(f"Centroid X range: {global_x_min} to {global_x_max}")
        print(f"Centroid Y range: {global_y_min} to {global_y_max}")
        print(f"Area range: {global_area_min} to {global_area_max}")

        x_min = args.x_min if args.x_min is not None else prompt_float("Enter X min", global_x_min)
        x_max = args.x_max if args.x_max is not None else prompt_float("Enter X max", global_x_max)
        y_min = args.y_min if args.y_min is not None else prompt_float("Enter Y min", global_y_min)
        y_max = args.y_max if args.y_max is not None else prompt_float("Enter Y max", global_y_max)
        min_area = args.min_area if args.min_area is not None else prompt_float("Enter minimum grain Area", global_area_min)
        target_width_guess = args.target_width if args.target_width is not None else prompt_float("Enter initial target Atomsk width in Angstroms")

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        filtered = filter_grains(data, x_min, x_max, y_min, y_max, min_area)

        # Compute the original aspect ratio and initial dimensions.
        geom = normalization_geometry(filtered, x_min, x_max, y_min, y_max, args.normalization_base)
        target_height_guess = target_width_guess / geom["aspect_ratio_width_over_height"]

        initial_atomsk_box = {
            "x": float(target_width_guess * (1.0 + args.padding_fraction)),
            "y": float(target_height_guess * (1.0 + args.padding_fraction)),
            "z": float(args.box_z),
        }

        periodic_summary = {"enabled": False}
        periodic_rows: list[dict] = []
        final_atomsk_box = dict(initial_atomsk_box)

        if args.periodic_sizing:
            final_atomsk_box, periodic_rows, periodic_summary = periodic_sizing(
                grains=filtered,
                initial_box=initial_atomsk_box,
                axes=args.periodic_axes,
                lattice_parameter=args.lattice_parameter,
                crystal_structure=args.crystal_structure,
                max_index=args.periodic_max_index,
                search_fraction=args.periodic_search_fraction,
                size_penalty=args.periodic_size_penalty,
                euler_convention=args.euler_convention,
            )

        # Convert final Atomsk box back into the inner scaled region used for nodes.
        final_target_width = final_atomsk_box["x"] / (1.0 + args.padding_fraction)
        final_target_height = final_atomsk_box["y"] / (1.0 + args.padding_fraction)
        final_box_z = final_atomsk_box["z"]

        normal_data, scaled_data, norm_meta = normalize_and_scale(
            filtered,
            target_width=final_target_width,
            target_height=final_target_height,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            normalization_base=args.normalization_base,
        )

        normal_csv = output_dir / "Normal_data.csv"
        scaled_xlsx = output_dir / "User_Data.xlsx"
        poly_txt = output_dir / "polycrystal.txt"
        summary_json = output_dir / "run_summary.json"
        periodic_report_csv = output_dir / "periodicity_report.csv"

        normal_data.to_csv(normal_csv, index=False, sep=";", decimal=".")
        scaled_data.to_excel(scaled_xlsx, index=False)

        if args.periodic_sizing:
            pd.DataFrame(periodic_rows).to_csv(periodic_report_csv, index=False)

        box_meta = write_polycrystal_txt(
            scaled_data=scaled_data,
            output_file=poly_txt,
            target_width=norm_meta["target_width_angstrom"],
            target_height=norm_meta["target_height_angstrom"],
            box_z=final_box_z,
            padding_fraction=args.padding_fraction,
        )

        summary = {
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "valid_input_grains": int(len(data)),
            "filtered_grains_written_as_nodes": int(len(filtered)),
            "filters": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "min_area": min_area,
            },
            "normalization": norm_meta,
            "initial_size_guess": {
                "target_width_before_padding_angstrom": float(target_width_guess),
                "target_height_before_padding_angstrom": float(target_height_guess),
                "box_z_angstrom": float(args.box_z),
                "atomsk_box_with_padding_angstrom": initial_atomsk_box,
            },
            "periodic_sizing": periodic_summary,
            "atomsk_box": box_meta,
            "atomsk": {
                "element": args.element,
                "crystal_structure": args.crystal_structure,
                "lattice_parameter_angstrom": args.lattice_parameter,
                "command": args.atomsk_command,
                "skipped": bool(args.skip_atomsk),
                "output_cfg": args.output_cfg,
            },
            "outputs": {
                "normal_csv": str(normal_csv),
                "scaled_xlsx": str(scaled_xlsx),
                "polycrystal_txt": str(poly_txt),
                "summary_json": str(summary_json),
                "periodicity_report_csv": str(periodic_report_csv) if args.periodic_sizing else None,
            },
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nGenerated files:")
        print(f"  - {normal_csv}")
        print(f"  - {scaled_xlsx}")
        print(f"  - {poly_txt}")
        print(f"  - {summary_json}")
        if args.periodic_sizing:
            print(f"  - {periodic_report_csv}")

        print(f"\nFiltered grains written as Atomsk nodes: {len(filtered)}")
        print(
            "Initial Atomsk box guess with padding: "
            f"{initial_atomsk_box['x']:.4f} Å x {initial_atomsk_box['y']:.4f} Å x {initial_atomsk_box['z']:.4f} Å"
        )
        print(
            "Final Atomsk box: "
            f"{box_meta['box_x_angstrom']:.4f} Å x {box_meta['box_y_angstrom']:.4f} Å x {box_meta['box_z_angstrom']:.4f} Å"
        )

        if args.periodic_sizing:
            print("\nPeriodic sizing summary:")
            for axis_name, choice in periodic_summary["axis_choices"].items():
                print(
                    f"  {axis_name.upper()}: {choice['initial_guess_angstrom']:.4f} Å -> "
                    f"{choice['selected_length_angstrom']:.4f} Å "
                    f"({choice['size_change_percent']:+.3f}%), "
                    f"max residual {choice['max_residual_angstrom']:.4f} Å"
                )
            max_angle = periodic_summary.get("max_angular_mismatch_deg")
            max_res = periodic_summary.get("max_final_length_residual_angstrom")
            if max_angle is not None:
                print(f"  Max closest-direction angular mismatch: {max_angle:.4f}°")
            if max_res is not None:
                print(f"  Max final length residual: {max_res:.4f} Å")
            if max_angle is not None and max_angle > args.angle_warning_deg:
                print(
                    "\nWARNING: Some EBSD orientations are not closely aligned with a low-index "
                    "repeat direction along the rectangular box axes. Exact rectangular periodicity "
                    "is probably not achievable for the full multi-grain system; inspect "
                    "periodicity_report.csv before trusting periodic boundaries."
                )

        if args.skip_atomsk:
            print("\nSkipped Atomsk execution because --skip-atomsk was used.")
        else:
            run_atomsk(
                output_dir=output_dir,
                atomsk_command=args.atomsk_command,
                crystal_structure=args.crystal_structure,
                lattice_parameter=args.lattice_parameter,
                element=args.element,
                output_cfg=args.output_cfg,
            )
            print(f"\nAtomsk finished. Final structure should be in: {output_dir / args.output_cfg}")

        print("\nProcess finished.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
