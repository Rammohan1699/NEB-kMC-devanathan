#!/usr/bin/env python3
"""
Mac/Linux version of the EBSD grain-list -> Atomsk polycrystal helper.

Main changes from the original Windows script:
  - Uses the Unix Atomsk command: `atomsk` instead of `atomsk.exe`.
  - Does NOT delete files in the working directory.
  - Writes all generated files into a chosen output directory.
  - Fixes centroid normalization: (x - x_min) / (x_max - x_min).
  - Does not skip the first valid grain when writing Atomsk nodes.
  - Adds command-line options while keeping an interactive mode.

Example:
  python3 atomsk_filtering_mac.py grains.csv --target-width 1000 --min-area 5

Dry run without calling Atomsk:
  python3 atomsk_filtering_mac.py grains.csv --target-width 1000 --min-area 5 --skip-atomsk
"""

from __future__ import annotations

import argparse
import json
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

NUMERIC_COLUMNS = [
    "ID",
    "Centroid X",
    "Centroid Y",
    "Area",
    "Mean Euler orientation 1",
    "Mean Euler orientation 2",
    "Mean Euler orientation 3",
]


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


def normalize_and_scale(
    grains: pd.DataFrame,
    target_width: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    normalization_base: str = "crop",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Normalize centroids and scale them to the Atomsk target size.

    normalization_base="crop":
        Use the user-selected EBSD crop window as the normalization box.
        x_norm = (x - selected_x_min) / (selected_x_max - selected_x_min)

    normalization_base="filtered":
        Use the min/max of the remaining filtered grain centroids.
        This is closer to the original script, but corrected by subtracting min.
    """
    if target_width <= 0:
        raise ValueError("target_width must be greater than zero.")

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

    aspect_ratio = image_width / image_height
    target_height = target_width / aspect_ratio

    normal_data = grains.copy()

    # Fixed normalization: subtract the origin before dividing by the size.
    normal_data["Centroid X"] = (normal_data["Centroid X"] - origin_x) / image_width
    normal_data["Centroid Y"] = (normal_data["Centroid Y"] - origin_y) / image_height

    scaled_data = normal_data.copy()
    scaled_data["Centroid X"] = scaled_data["Centroid X"] * target_width
    scaled_data["Centroid Y"] = scaled_data["Centroid Y"] * target_height

    meta = {
        "normalization_base": normalization_base,
        "origin_x_um": origin_x,
        "origin_y_um": origin_y,
        "image_width_um": image_width,
        "image_height_um": image_height,
        "aspect_ratio_width_over_height": aspect_ratio,
        "target_width_angstrom": target_width,
        "target_height_angstrom": target_height,
    }

    return normal_data, scaled_data, meta


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
        "box_x_angstrom": box_x,
        "box_y_angstrom": box_y,
        "box_z_angstrom": box_z,
        "node_z_angstrom": z,
        "padding_fraction": padding_fraction,
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
    parser.add_argument("--target-width", type=float, help="Target Atomsk width in Angstroms.")
    parser.add_argument("--output-dir", default="atomsk_output", help="Directory for generated files.")
    parser.add_argument("--normalization-base", choices=["crop", "filtered"], default="crop", help="Use selected crop window or filtered grain min/max for normalization. Default: crop.")
    parser.add_argument("--box-z", type=float, default=10.0, help="Atomsk box thickness in Angstroms. Default: 10.")
    parser.add_argument("--padding-fraction", type=float, default=0.10, help="Extra box padding fraction in X and Y. Default: 0.10.")
    parser.add_argument("--lattice-parameter", type=float, default=2.856, help="Lattice parameter for Atomsk unit cell. Default: 2.856 Å.")
    parser.add_argument("--element", default="Fe", help="Element symbol for Atomsk. Default: Fe.")
    parser.add_argument("--crystal-structure", default="bcc", help="Crystal structure for Atomsk --create. Default: bcc.")
    parser.add_argument("--atomsk-command", default="atomsk", help="Atomsk command name/path. Default: atomsk.")
    parser.add_argument("--output-cfg", default="final.cfg", help="Final Atomsk output structure file. Default: final.cfg.")
    parser.add_argument("--skip-atomsk", action="store_true", help="Only write CSV/XLSX/polycrystal.txt files; do not run Atomsk.")
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
        target_width = args.target_width if args.target_width is not None else prompt_float("Enter target Atomsk width in Angstroms")

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        filtered = filter_grains(data, x_min, x_max, y_min, y_max, min_area)
        normal_data, scaled_data, norm_meta = normalize_and_scale(
            filtered,
            target_width=target_width,
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

        normal_data.to_csv(normal_csv, index=False, sep=";", decimal=".")
        scaled_data.to_excel(scaled_xlsx, index=False)

        box_meta = write_polycrystal_txt(
            scaled_data=scaled_data,
            output_file=poly_txt,
            target_width=norm_meta["target_width_angstrom"],
            target_height=norm_meta["target_height_angstrom"],
            box_z=args.box_z,
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
            },
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nGenerated files:")
        print(f"  - {normal_csv}")
        print(f"  - {scaled_xlsx}")
        print(f"  - {poly_txt}")
        print(f"  - {summary_json}")
        print(f"\nFiltered grains written as Atomsk nodes: {len(filtered)}")
        print(f"Target box before padding: {norm_meta['target_width_angstrom']:.4f} Å x {norm_meta['target_height_angstrom']:.4f} Å")
        print(f"Atomsk box with padding: {box_meta['box_x_angstrom']:.4f} Å x {box_meta['box_y_angstrom']:.4f} Å x {box_meta['box_z_angstrom']:.4f} Å")

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
