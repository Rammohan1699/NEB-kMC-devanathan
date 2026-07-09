# EBSD Mapper

This tool converts an EBSD grain-list CSV into Atomsk polycrystal input files.
It is useful when preparing polycrystalline or bicrystal host structures before
site discovery and KMC map generation.

## Contents

- `atomsk_filtering_mac.py`: basic EBSD CSV to Atomsk `polycrystal.txt` helper.
- `atomsk_filtering_mac_periodic.py`: extended helper with orientation-aware
  periodic box-size snapping.
- `example_data/grains.csv`: example EBSD grain data sheet.
- `example_data/IPFX+GB+leyenda.bmp`: reference EBSD map/legend image associated
  with the example data.

Large generated Atomsk products, including `final.cfg`, are intentionally not
tracked. Recreate them from the example data when needed.

## Dependencies

Install the repository requirements first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

To generate final structures rather than dry-run input files, install Atomsk and
make sure the `atomsk` command is available on `PATH`.

## Required CSV Columns

The scripts expect an EBSD grain-list CSV with these columns:

- `ID`
- `Centroid X`
- `Centroid Y`
- `Area`
- `Mean Euler orientation 1`
- `Mean Euler orientation 2`
- `Mean Euler orientation 3`

Rows containing units or non-numeric values in these fields are dropped during
parsing.

## Basic Usage

Dry-run the basic conversion without calling Atomsk:

```bash
.venv/bin/python tools/ebsd-mapper/atomsk_filtering_mac.py \
  tools/ebsd-mapper/example_data/grains.csv \
  --x-min 0.79 \
  --x-max 123.86 \
  --y-min 0.10 \
  --y-max 84.89 \
  --target-width 1000 \
  --min-area 5 \
  --output-dir tools/ebsd-mapper/atomsk_output \
  --skip-atomsk
```

Run the periodic-sizing version without Atomsk:

```bash
.venv/bin/python tools/ebsd-mapper/atomsk_filtering_mac_periodic.py \
  tools/ebsd-mapper/example_data/grains.csv \
  --x-min 0.79 \
  --x-max 123.86 \
  --y-min 0.10 \
  --y-max 84.89 \
  --min-area 5 \
  --target-width 1000 \
  --periodic-sizing \
  --periodic-axes xy \
  --output-dir tools/ebsd-mapper/atomsk_output \
  --skip-atomsk
```

Run Atomsk after generating the filtered EBSD files:

```bash
.venv/bin/python tools/ebsd-mapper/atomsk_filtering_mac_periodic.py \
  tools/ebsd-mapper/example_data/grains.csv \
  --x-min 0.79 \
  --x-max 123.86 \
  --y-min 0.10 \
  --y-max 84.89 \
  --min-area 5 \
  --target-width 1000 \
  --periodic-sizing \
  --output-dir tools/ebsd-mapper/atomsk_output
```

Generated files are written under `atomsk_output/` by default. The examples
above direct those files to `tools/ebsd-mapper/atomsk_output/`, which is ignored
by Git. Use `--output-dir <path>` to keep output somewhere else.

## Important Options

- `--x-min`, `--x-max`, `--y-min`, `--y-max`: crop the EBSD map by centroid.
- `--min-area`: remove grains below a selected area threshold.
- `--target-width`: target Atomsk X dimension, in Angstrom.
- `--box-z`: target or initial Z thickness, in Angstrom.
- `--normalization-base crop`: normalize against the selected crop window.
- `--normalization-base filtered`: normalize against the filtered grain extent.
- `--padding-fraction`: add extra X/Y padding around generated nodes.
- `--lattice-parameter`: cubic lattice parameter used for Atomsk creation.
- `--element`: Atomsk element symbol, default `Fe`.
- `--crystal-structure`: Atomsk crystal structure, default `bcc`.
- `--periodic-sizing`: snap requested box lengths to nearby crystallographic
  repeat lengths estimated from grain Euler orientations.
- `--periodic-axes`: axes considered by periodic sizing, default `xy`.
- `--skip-atomsk`: write CSV/XLSX/`polycrystal.txt` outputs without launching
  Atomsk.

## Outputs

Typical outputs include:

- `Normal_data.csv`: normalized EBSD centroids.
- `User_Data.xlsx`: scaled grain table.
- `polycrystal.txt`: Atomsk polycrystal node file.
- `run_summary.json`: selected parameters and output paths.
- `periodicity_report.csv`: mismatch report when `--periodic-sizing` is used.
- `final.cfg`: final Atomsk structure when Atomsk execution is enabled.

For arbitrary EBSD orientations, exact rectangular periodicity is generally not
guaranteed. Treat `--periodic-sizing` as a practical approximation and inspect
`periodicity_report.csv` before using periodic boundaries in production.
