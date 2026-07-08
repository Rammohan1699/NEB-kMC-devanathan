# Installation

## Python Environment

Use Python 3.10 or newer. The current local validation has been run with Python 3.13.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

Core Python packages:

- `numpy`
- `scipy`
- `ase`
- `mpi4py`
- `pytest`

## MPI

Multi-rank KMC and grouped native LAMMPS-only NEB require an MPI launcher:

```bash
mpirun -np 6 .venv/bin/python -m kmc.main
```

For LAMMPS-only NEB, `MPI_RANKS` must be divisible by `LAMMPS_NEB_REPLICAS`. For example, `MPI_RANKS=18` and `LAMMPS_NEB_REPLICAS=3` gives six concurrent NEB worker groups.

## LAMMPS Python Module

ASE-NEB and native LAMMPS-only NEB both require a usable LAMMPS backend for production runs. Set these when auto-detection is not enough:

```bash
export LAMMPS_LIB_DIR=/path/to/lammps/build
export LAMMPS_PYTHON_PATH=/path/to/python/package/parent
export DYLD_LIBRARY_PATH="$LAMMPS_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$LAMMPS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

On macOS, the scripts also support copying a source `lammps/` Python package into the run directory with `LAMMPS_SOURCE_PYTHON_PATH` and symlinking `liblammps.dylib`.

## Validate The Checkout

Fast source and unit checks:

```bash
.venv/bin/python -m pytest tests
python3 -m py_compile kmc/*.py kmc/lammps_only_neb/*.py tools/*.py
bash -n scripts/run_devanathan_kmc.sh
bash -n scripts/run_external_pulse50_barrier_diagnostic_pair.sh
```

Dry-run launchers before long campaigns:

```bash
DRY_RUN=1 bash scripts/run_external_pulse50_barrier_diagnostic_pair.sh
DRY_RUN=1 bash scripts/run_devanathan_gcmc_fenwick_serial_7m.sh
DRY_RUN=1 bash scripts/run_devanathan_bicrystal_gcmc_fenwick_serial_7m.sh
```

## Hyperion Notes

Cluster scripts live in `scripts/*.sbatch`. They expect the package root to be exported as `KMC_PACKAGE_DIR` or inferred from the submitted script location. Keep run products outside source history and pull back only consolidated CSV/Markdown summaries or compact cache files when needed.
