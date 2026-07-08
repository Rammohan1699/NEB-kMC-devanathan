# KMC Job Script Index

This file is the current map of scripts and when to use them.

## Workspace Layout

```text
kmc/                         active development KMC engine
kmc_cluster_fresh/           cluster-ready package
kmc_local_share/             clean local package for sharing
kmc_map_inputs/              local map inputs used by development runs
kmc_map_generator/           site-map generation tools
tools/                       local analysis/helper tools
local_runs/                  local completed runs moved out of the root
cluster_results/             copied cluster outputs
archived_root_runs/          old root-level run artifacts/checkpoints/logs
local_packages/              zip/share archives
docs/                        copied reference docs
```

## Local Run Scripts

These are in `kmc_local_share/scripts/`.

### `launch_kmc_job.sh`

General local launcher. Use this for a normal local KMC run.

```bash
cd /Users/rtirunelveli/kMC_code/latest-version/kmc_local_share

JOB_NAME=one_grain_seed101 \
KMC_SITE_SOURCE=external \
KMC_INITIAL_H_REGIONS=bulk_grain_0 \
KMC_SEED=101 \
NUM_H=50 \
STEPS=10000 \
NP=6 \
./scripts/launch_kmc_job.sh
```

Useful overrides:

```bash
BARRIER_CACHE_FILE=/path/to/master_cache.pkl
WRITE_RATES_ALLRANKS=0
DUMP_EVERY_STEPS=100
PYTHON_BIN=/path/to/python
MPIRUN_BIN=/path/to/mpirun
```

### `postprocess_run.sh`

Postprocess one completed local run.

```bash
cd /Users/rtirunelveli/kMC_code/latest-version/kmc_local_share
RUN_DIR=runs/one_grain_seed101 ./scripts/postprocess_run.sh
```

### `run_benchmark_10step_local.sh`

Short local performance benchmark matching the cluster benchmark.

```bash
cd /Users/rtirunelveli/kMC_code/latest-version/kmc_local_share
./scripts/run_benchmark_10step_local.sh
```

### `summarize_benchmark_run.sh`

Summarize a benchmark/run directory.

```bash
./scripts/summarize_benchmark_run.sh runs/one_grain_seed101
```

### `extract_cluster_run_health.sh`

Lightweight health extractor. Can be run locally on copied cluster results or on
the cluster.

```bash
./scripts/extract_cluster_run_health.sh /path/to/runs > run_health.txt
```

## Cluster Run Scripts

These are in `kmc_cluster_fresh/scripts/`. Submit from the cluster package root:

```bash
cd /scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh
mkdir -p slurm_logs
```

### `slurm_10_one_grain_array.sbatch`

Ten jobs using the external Sigma5 map, with all H initialized in
`bulk_grain_0`. This is the “one-grain” run set.

```bash
sbatch --export=ALL,KMC_PACKAGE_DIR=/scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh \
  scripts/slurm_10_one_grain_array.sbatch
```

### `slurm_10_bicrystal_array.sbatch`

Ten Sigma5 bicrystal jobs. The first five start in `bulk_grain_0`; the next five
start in `bulk_grain_1`.

```bash
sbatch --export=ALL,KMC_PACKAGE_DIR=/scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh \
  scripts/slurm_10_bicrystal_array.sbatch
```

### `submit_both_10_job_arrays.sh`

Submits both arrays above.

```bash
./scripts/submit_both_10_job_arrays.sh
```

Optional:

```bash
STEPS=50000 NUM_H=50 ./scripts/submit_both_10_job_arrays.sh
```

### `slurm_long_one_crystal_master_cache.sbatch`

Long one-grain run using a master cache.

Current defaults for this script:

```bash
STEPS=500000
KMC_INITIAL_H_REGIONS=bulk_grain_0
WRITE_RATES_ALLRANKS=0
DUMP_EVERY_STEPS=100
JOB_ASSIGNMENT_MODE=cache_dedupe
```

Submit:

```bash
sbatch --export=ALL,KMC_PACKAGE_DIR=/scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh \
  scripts/slurm_long_one_crystal_master_cache.sbatch
```

Cache-only diagnostic:

```bash
sbatch --export=ALL,KMC_PACKAGE_DIR=/scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh,STEPS=100,JOB_ASSIGNMENT_MODE=cache_only,WRITE_RATES_ALLRANKS=0,DUMP_EVERY_STEPS=100 \
  scripts/slurm_long_one_crystal_master_cache.sbatch
```

### `slurm_benchmark_10step_one_crystal.sbatch`

Short cluster benchmark matching the local benchmark.

```bash
sbatch --export=ALL,KMC_PACKAGE_DIR=/scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh \
  scripts/slurm_benchmark_10step_one_crystal.sbatch
```

### `launch_kmc_job.sh`

Single-run launcher used by the Slurm scripts. You can run it directly inside an
interactive cluster allocation.

```bash
JOB_NAME=manual_test \
KMC_SITE_SOURCE=external \
KMC_INITIAL_H_REGIONS=bulk_grain_0 \
NUM_H=2 \
STEPS=1 \
NP=1 \
./scripts/launch_kmc_job.sh
```

### `postprocess_run.sh`

Postprocess a cluster run.

```bash
RUN_DIR=runs/one_grain/one_grain_bulk0_seed101 ./scripts/postprocess_run.sh
```

### `prepare_one_crystal_master_cache.sh`

Copies a merged cache from a completed one-crystal/one-grain run into:

```text
cache_bank/one_crystal_master_cache.pkl
```

Usage:

```bash
ONE_CRYSTAL_RUN_DIR=/path/to/completed/run ./scripts/prepare_one_crystal_master_cache.sh
```

### `extract_cluster_run_health.sh`

Diagnose incomplete/stalled cluster jobs without copying full run directories.

```bash
./scripts/extract_cluster_run_health.sh runs > cluster_run_health.txt
gzip -f cluster_run_health.txt
```

## Legacy / Compatibility Scripts

These still exist but are less preferred:

- `slurm_10_single_crystal_array.sbatch`: compatibility name for the one-grain array. Prefer `slurm_10_one_grain_array.sbatch`.
- `launch_10_single_crystal_jobs.sh`: background launcher, not scheduler-native. Prefer the Slurm array.
- `launch_10_bicrystal_jobs.sh`: background launcher, not scheduler-native. Prefer the Slurm array.
- `slurm_template.sbatch`: generic template only.

## Copy Updated Scripts To Cluster

```bash
rsync -avh --progress \
  /Users/rtirunelveli/kMC_code/latest-version/kmc_cluster_fresh/scripts/ \
  rtirunelveli@hyperion-01.sw.ehu.es:/scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh/scripts/
```

If code changed too:

```bash
rsync -avh --progress \
  /Users/rtirunelveli/kMC_code/latest-version/kmc_cluster_fresh/kmc/ \
  rtirunelveli@hyperion-01.sw.ehu.es:/scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh/kmc/
```

## Clean Cluster Root Clutter

On the cluster:

```bash
cd /scratch/rtirunelveli/LAMMPS/kmc_cluster_fresh
mkdir -p slurm_logs misc_logs
mv kmc_*.out kmc_*.err slurm_logs/ 2>/dev/null
mv slurm-*.out slurm-*.err slurm_logs/ 2>/dev/null
mv log.lammps cluster_run_health.txt misc_logs/ 2>/dev/null
```

Remove heavy old per-run files only after confirming they are no longer needed:

```bash
find runs -type f -name 'rates_allranks.csv' -delete
find runs -type d -name 'ase_logs' -prune -exec rm -rf {} +
```
