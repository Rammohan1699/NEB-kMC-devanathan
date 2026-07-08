# LAMMPS-only NEB
This repository includes a native LAMMPS NEB path alongside the original ASE/LAMMPS NEB path.

The original ASE/LAMMPS NEB path is still present. Select the new engine with:

```bash
NEB_ENGINE=lammps_only
```

The native path currently targets generated/shell systems and uses grouped MPI scheduling:

- `LAMMPS_NEB_REPLICAS=3` means each NEB calculation uses three MPI ranks.
- A six-rank run therefore creates two NEB worker groups.
- Rank 0 plans cache-miss environment batches and broadcasts that batch plan.
- Each batch contains the associated nearest-neighbor hops for one H environment.

Useful local smoke-test variables:

```bash
NEB_ENGINE=lammps_only
NX=10 NY=10 NZ=10
NUM_H=5
STEPS=1
LOCAL_ENV_MODE=shell
SHELL_INNER_RADIUS_A=8
SHELL_OUTER_RADIUS_A=10
LAMMPS_NEB_REPLICAS=3
LAMMPS_NEB_STEPS=1000
LAMMPS_PYTHON_PATH=../tests/native_lammps_neb_test
LAMMPS_LIB_DIR=/Users/rtirunelveli/mylammps/build2
```

Normal mode uses compact scratch: temporary LAMMPS files are written only while a hop is running, barriers are parsed, and scratch is deleted immediately. Use `LAMMPS_NEB_DEBUG_MODE=1` to retain LAMMPS input/log files for a failing hop.

Timing CSV files include native NEB I/O sub-buckets:

- `neb_io_write`
- `neb_lammps_run`
- `neb_io_parse`
- `neb_io_cleanup`

These are diagnostic subcomponents of `barrier_neb` and are not counted again in `components_sum`.
