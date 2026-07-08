# Devanathan KMC Base

This is the latest base version of the hydrogen KMC/NEB code. It promotes the current practice implementation into the root package and keeps generated runs, caches, checkpoints, logs, release zips, and analysis products out of the tracked source tree.

## What Is Included

- `kmc/`: production KMC driver, event manager, cache management, structure builders, restart handling, diagnostics, and both ASE/LAMMPS and native LAMMPS-only NEB paths.
- `kmc/lammps_only_neb/`: grouped MPI native LAMMPS NEB engine with compact scratch cleanup and optional native endpoint optimization.
- `kmc/devanathan.py`: fixed-x Devanathan source/sink, pulse-source, left-absorber, and hop-direction filters.
- `kmc/devanathan_gcmc.py` and `kmc/gcmc_energy_cache.py`: fixed chemical-potential GCMC initialization/maintenance and persistent local insertion-energy caches.
- `kmc/incremental_events.py`: impacted-region event rebuilds and Fenwick-weighted event selection support.
- `Site_unified/`: source-only lattice-site discovery pipeline for parsing orthogonal LAMMPS bicrystal/polycrystal host structures and generating grain-aware bulk/GB KMC site maps.
- `examples/previously_analyzed/`: analyzed example inputs, including the Sigma5 KMC map files exposed through the `kmc_map_inputs` compatibility symlink.
- `scripts/`: local, segmented, cluster, pulse diagnostic, GCMC bulk, bicrystal, and external-map launchers.
- `tools/`: postprocessing, cache merge/compact, GCMC sweep, restart preparation, flux/permeability, and barrier-comparison utilities.
- `tests/`: focused tests for Devanathan boundaries, GCMC behavior, energy-cache keys, external shell handling, and sweep selection.

## Install

See [INSTALL.md](INSTALL.md) for Python, MPI, ASE, and LAMMPS setup. A minimal local development environment is:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

Native LAMMPS-only runs also need a working LAMMPS Python module and library path.

## Quick Checks

```bash
.venv/bin/python -m pytest tests
python3 -m py_compile kmc/*.py kmc/lammps_only_neb/*.py tools/*.py
DRY_RUN=1 bash scripts/run_external_pulse50_barrier_diagnostic_pair.sh
```

## Common Runs

Generated-cell Devanathan source/sink with native LAMMPS-only NEB:

```bash
STEPS=10000 MPI_RANKS=6 bash scripts/run_devanathan_kmc.sh
```

External Sigma5 50-H pulse diagnostic comparing ASE-NEB, LAMMPS-only, and LAMMPS-only endpoint optimization:

```bash
STEPS=10000 MPI_RANKS=6 bash scripts/run_external_pulse50_barrier_diagnostic_pair.sh
```

Segmented GCMC/Fenwick production:

```bash
DRY_RUN=1 bash scripts/run_devanathan_gcmc_fenwick_serial_7m.sh
```

## Documentation

- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md): run modes and environment variables.
- [docs/RUN_MODES_AND_RESULTS.md](docs/RUN_MODES_AND_RESULTS.md): engine modes, Devanathan modes, cache modes, Fenwick/incremental path, and the latest pulse50 result summary.
- [README_LAMMPS_ONLY.md](README_LAMMPS_ONLY.md): native LAMMPS-only NEB details.

## Repository Hygiene

The tracked base should contain source, launchers, tests, compact docs, and reusable tools. Keep heavy products under ignored locations such as `runs/`, `cluster_results/`, `hyperion_results/`, `practice-version/`, `cache/`, `logs/`, `diagnostics/`, `checkpoints/`, `neb/`, and generated `Site_unified/` map outputs.
