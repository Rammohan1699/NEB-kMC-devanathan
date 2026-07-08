# Refactored KMC + NEB Code Usage Guide

This guide explains how to run the refactored MPI KMC code, how the files fit together, and where to change inputs for bulk-generated sites or an externally generated `kmc_map` from a previously constructed structure.

The code is controlled mainly through **environment variables**, not by editing the main loop. In normal use, you should change inputs by exporting variables before launching `main.py` / `kmc.main`.

---

## 1. What this code does

The refactored driver performs a rejection-free kinetic Monte Carlo simulation for H diffusion in Fe, while calculating missing migration barriers with LAMMPS/ASE NEB. At a high level, each KMC step:

1. Initializes or loads interstitial sites.
2. Randomly places the requested number of H atoms on those sites.
3. Builds a nearest-neighbor hopping graph between sites.
4. Checks the barrier cache for each possible H hop.
5. Sends cache misses to MPI ranks for NEB calculation.
6. Converts barriers to rates using an Arrhenius expression.
7. Selects one rejection-free KMC event.
8. Updates the H position, trajectory, MSD, timing, and cache files.

The new integration allows two site sources:

- `generated`: ideal BCC tetrahedral sites generated from `NX`, `NY`, `NZ`, and `LATTICE_A`.
- `external`: sites loaded from a `kmc_map` generated from a constructed structure, plus the matching LAMMPS host structure.

---

## 2. Main files and their roles

| File | Purpose |
|---|---|
| `main.py` | Entry point. Builds the simulation from environment variables and calls `sim.run()`. |
| `config.py` | Central place for all user-facing input variables. Reads environment variables and creates a `SimulationConfig`. |
| `kmc_driver.py` | Top-level simulation driver. Initializes sites/H atoms, builds events, calls scheduler, runs NEB jobs, selects events, updates state, and writes outputs. |
| `lattice.py` | Generates ideal BCC Fe and tetrahedral/octahedral site positions; builds K-nearest-neighbor site graph. |
| `structures.py` | Loads external `kmc_map` and LAMMPS host structures; builds ASE structures for full systems and local NEB environments. |
| `environment.py` | Builds local H-environment cache keys from nearby H positions and hop direction. |
| `event_manager.py` | Builds candidate KMC events, cache-miss jobs, rates, and selected rejection-free events. |
| `scheduler.py` | MPI scheduling of NEB jobs and collection of NEB results. |
| `lammps_factory.py` | Builds ASE `LAMMPSlib` calculators using EAM, NNP, or custom LAMMPS pair commands. |
| `lammps_neb.py` | Runs actual NEB barrier calculations using ASE NEB/BFGS and LAMMPS calculators. |
| `cache.py`, `cache_manager.py` | Persistent barrier cache. Saves rank-local cache files and merges them. |
| `msd_tracker.py` | Tracks MSD vs KMC time and writes `msd_vs_time.csv` / `msd_plot.png`. |
| `logging_utils.py` | Rank logs and per-step timing CSV/log formatting. |
| `mpi_context.py` | MPI wrapper for rank, size, collectives, and phase guards. |

---

## 3. Basic run command

From the parent directory that contains the `kmc` package/module:

```bash
mpirun -np 4 python -m kmc.main
```

If you are running from the same folder as the Python files and not as a package:

```bash
mpirun -np 4 python main.py
```

For a single-rank smoke test:

```bash
python main.py
```

Before a full run, check the resolved configuration:

```bash
python config.py
```

This prints a compact summary and the full resolved configuration.

---

## 4. Recommended workflow

### Step 1 — Put required files in the run directory

For an EAM run, make sure the potential file exists, for example:

```text
PotentialB3410-modified.fs
```

For external-map mode, also prepare:

```text
kmc_site_map.npz          # preferred
host_structure.data       # LAMMPS atomic data file for the same structure/box
```

The `.npz` site map is preferred because it includes the periodic box lengths. CSV site maps are supported, but they do not carry the box, so the host structure box becomes more important.

### Step 2 — Choose site source

Use one of:

```bash
export KMC_SITE_SOURCE=generated
```

or

```bash
export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE=/path/to/kmc_site_map.npz
export KMC_HOST_STRUCTURE_FILE=/path/to/host_structure.data
```

If either `KMC_SITE_MAP_FILE` or `KMC_HOST_STRUCTURE_FILE` is set, the config automatically switches to external-map mode even if `KMC_SITE_SOURCE` is not explicitly set.

### Step 3 — Set KMC size and H count

For example:

```bash
export LATTICE_A=2.8601
export NX=30
export NY=30
export NZ=30
export NUM_H=540
export STEPS=100
export KMC_SEED=42
```

In external-map mode, the physical simulation box comes from the map/host structure, but `NX`, `NY`, `NZ`, and `LATTICE_A` are still used as fallback/config metadata in parts of the code. Keep them consistent with the constructed structure when possible.

### Step 4 — Choose local NEB environment mode

For ideal bulk-generated sites:

```bash
export LOCAL_ENV_MODE=radial
```

Other options:

```bash
export LOCAL_ENV_MODE=wrapped
export LOCAL_ENV_MODE=shell
```

For external-map mode, the driver uses explicit Fe positions from the loaded host structure and effectively uses shell-style local environments around each hop. Recommended external-map settings:

```bash
export LOCAL_ENV_MODE=shell
export SHELL_INNER_RADIUS_A=6.0
export SHELL_OUTER_RADIUS_A=8.0
```

### Step 5 — Choose LAMMPS potential

For the default Fe-H EAM/fs potential:

```bash
export POTENTIAL=EAM
export POTENTIAL_EAM_FILE=/path/to/PotentialB3410-modified.fs
```

For NNP/HDNNP:

```bash
export POTENTIAL=NNP
export NNP_DIR=/path/to/NNP/POTENTIAL
export NNP_ELEMENTS="Fe H"
export NNP_CUTOFF=6.60
```

For custom LAMMPS pair commands:

```bash
export LAMMPS_PAIR_STYLE='pair_style eam/fs'
export LAMMPS_PAIR_COEFF='pair_coeff * * /path/to/PotentialB3410-modified.fs Fe H'
export LAMMPS_FILES=/path/to/PotentialB3410-modified.fs
```

If `LAMMPS_PAIR_STYLE` and `LAMMPS_PAIR_COEFF` are set, they override the standard `POTENTIAL=EAM/NNP` setup.

### Step 6 — Run

```bash
mpirun -np 4 python -m kmc.main
```

---

## 5. Complete example: generated ideal BCC tetrahedral sites

Use this when you want the code to generate ideal BCC tetrahedral sites internally.

```bash
export KMC_SITE_SOURCE=generated

export LATTICE_A=2.8601
export NX=30
export NY=30
export NZ=30
export NUM_H=540
export STEPS=100
export KMC_SEED=42

export KNN_K=6
export LOCAL_ENV_MODE=radial
export ENV_RADIUS_A=5.0
export POS_BIN_A=0.10
export HOP_BIN_A=0.02
export ENV_KEY_MODE=env_plus_dir

export POTENTIAL=EAM
export POTENTIAL_EAM_FILE=/path/to/PotentialB3410-modified.fs
export LAMMPS_FILES=/path/to/PotentialB3410-modified.fs

export NIMG=3
export OPTIMIZE_ENDPOINTS=1
export NEB_OPTIMIZER_FMAX=0.05
export NEB_OPTIMIZER_STEPS=300
export ENDPOINT_OPTIMIZER_STEPS=200

export DEBUG_MODE=1

mpirun -np 4 python -m kmc.main
```

---

## 6. Complete example: external `kmc_map` from a constructed structure

Use this when you already built a structure, generated a KMC site map from it, and want KMC/NEB to use that exact geometry.

```bash
export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE=/path/to/stage3_kmc_site_map.npz
export KMC_HOST_STRUCTURE_FILE=/path/to/host_structure.data
export KMC_HOST_FE_TYPE=1

export NUM_H=100
export STEPS=50
export KMC_SEED=42

export KNN_K=6
export SAME_TYPE=0

export LOCAL_ENV_MODE=shell
export SHELL_INNER_RADIUS_A=6.0
export SHELL_OUTER_RADIUS_A=8.0

export ENV_KEY_MODE=env_plus_dir
export ENV_RADIUS_A=5.0
export POS_BIN_A=0.10
export HOP_BIN_A=0.02

export POTENTIAL=EAM
export POTENTIAL_EAM_FILE=/path/to/PotentialB3410-modified.fs
export LAMMPS_FILES=/path/to/PotentialB3410-modified.fs

export NIMG=3
export OPTIMIZE_ENDPOINTS=1
export NEB_MIN_BATCH=64

export DEBUG_MODE=1

mpirun -np 4 python -m kmc.main
```

Important external-map requirements:

1. The `kmc_map` and host structure must describe the same periodic cell.
2. The `.npz` map should contain `positions`; it may also contain `site_types`, `box_lengths`, and `box_origin`.
3. The host LAMMPS data file must have orthogonal box bounds and an `Atoms # atomic` section with rows like:

```text
atom_id atom_type x y z
```

4. `KMC_HOST_FE_TYPE` tells the loader which LAMMPS atom type is Fe. The default is `1`.

---

## 7. Where to change each input

Most inputs should be changed in the shell before running. Editing source files is usually unnecessary.

### Main simulation size

| Input | Default | Meaning |
|---|---:|---|
| `LATTICE_A` | `2.8601` | BCC Fe lattice parameter in Angstrom. |
| `NX`, `NY`, `NZ` | `30`, `30`, `30` | Supercell dimensions for generated-site mode. |
| `NUM_H` | `540` | Number of H atoms randomly placed on available KMC sites. |
| `STEPS` | `100` | Number of KMC steps to attempt. |
| `KMC_SEED` | `None` in config, fallback `42` in driver | Random seed for reproducible H placement and KMC choices. |

Change these with:

```bash
export NUM_H=50
export STEPS=10
```

### Site source and external structure map

| Input | Default | Meaning |
|---|---|---|
| `KMC_SITE_SOURCE` | `generated` | `generated` or `external`. Aliases `map`, `file`, and `external_map` also mean external. |
| `KMC_SITE_MAP_FILE` | empty | Path to `.npz` or `.csv` KMC site map. |
| `SITE_MAP_FILE` | empty | Backward-compatible alias for `KMC_SITE_MAP_FILE`. |
| `KMC_HOST_STRUCTURE_FILE` | empty | Path to LAMMPS host structure data file. |
| `HOST_STRUCTURE_FILE` | empty | Backward-compatible alias for `KMC_HOST_STRUCTURE_FILE`. |
| `KMC_HOST_FE_TYPE` | `1` | Atom type in the LAMMPS file that should be interpreted as Fe. |
| `KMC_INITIAL_H_REGION_FILE` | empty | Optional site-region `.npz` from `tools/prepare_kmc_region_inputs.py`. |
| `KMC_INITIAL_H_REGIONS` | empty | Optional comma-separated region labels for initial random H placement. |

For Sigma5 region-aware initial H placement:

```bash
export KMC_INITIAL_H_REGION_FILE=kmc_map_inputs/sigma5_site_regions.npz
export KMC_INITIAL_H_REGIONS=bulk_grain_0
```

Supported labels include `bulk_grain_0`, `bulk_grain_1`, `transition`, and
`grain_boundary`. `bulk` selects all bulk grains. `grain` and `gb` are aliases
for `grain_boundary`. `bulk_0` and `bulk0` are shorthand for `bulk_grain_0`.

### Neighbor graph

| Input | Default | Meaning |
|---|---:|---|
| `KNN_K` | `6` | Number of nearest-neighbor site hops considered from each site. |
| `SAME_TYPE` | `0`/false | If true, prefers same interstitial type in neighbor selection. |

For a tetrahedral network, `KNN_K=6` is the current driver default.

### Local environment and cache key

| Input | Default | Meaning |
|---|---:|---|
| `LOCAL_ENV_MODE` | `radial` | Local NEB structure type: `radial`, `wrapped`, or `shell`. External maps force shell-style handling. |
| `ENV_RADIUS_A` | `5.0` | Radius for nearby H atoms included in the environment key. |
| `ENV_KEY_MODE` | `env_plus_dir` | `env_plus_dir` includes local H environment + hop direction; `env_only` ignores hop direction. |
| `POS_BIN_A` | `0.10` | Quantization bin for local H displacement vectors in the cache key. |
| `HOP_BIN_A` | `0.02` | Quantization bin for hop direction vector in the cache key. |
| `ENV_AFFECT_RADIUS_A` | `6.0` | Radius used by environment/state logic where applicable. |
| `SHELL_INNER_RADIUS_A` | `5.0` in loader, commonly `6.0` in driver schema | Inner shell radius; atoms inside this are movable. |
| `SHELL_OUTER_RADIUS_A` | `6.0` in loader, commonly `8.0` in driver schema | Outer shell radius; atoms between inner and outer shell are included and frozen. |

Recommended starting point:

```bash
export LOCAL_ENV_MODE=radial      # generated bulk
# or
export LOCAL_ENV_MODE=shell       # external map / GB / non-ideal host

export ENV_RADIUS_A=5.0
export POS_BIN_A=0.10
export HOP_BIN_A=0.02
export ENV_KEY_MODE=env_plus_dir
```

### Local environment build-only mode

This mode does not run KMC. It builds and dumps local NEB environments so you can inspect them before a full NEB/KMC run.

| Input | Default | Meaning |
|---|---:|---|
| `BUILD_LOCAL_ENV_ONLY` | `0` | If true, stop after writing local environment examples. |
| `BUILD_LOCAL_ENV_COUNT` | `10` | Number of local environments to dump. |
| `BUILD_LOCAL_ENV_OUT` | `local_env_builds` | Output folder. |
| `BUILD_LOCAL_ENV_SELECTOR` | `random` | `random`, `boundary`, or `cluster`. |
| `BOUNDARY_MARGIN_A` | `5.0` | Boundary selector distance from box edge. |
| `CLUSTER_RADIUS_A` | `5.0` | Cluster selector radius around seed site. |

Example:

```bash
export BUILD_LOCAL_ENV_ONLY=1
export BUILD_LOCAL_ENV_COUNT=20
export BUILD_LOCAL_ENV_OUT=local_env_builds_test
export BUILD_LOCAL_ENV_SELECTOR=random
mpirun -np 1 python -m kmc.main
```

Inspect the generated `initial.xyz`, `final.xyz`, `atom_roles.csv`, and `meta.json` files before running expensive NEB calculations.

### NEB settings

| Input | Default | Meaning |
|---|---:|---|
| `NIMG` | `3` | Number of NEB images/calculators. Minimum is forced to 3. |
| `NEB_MIN_BATCH` | `64` | Minimum job batch size before chunking. |
| `OPTIMIZE_ENDPOINTS` | `1` | Whether initial/final endpoints are optimized before NEB. |
| `NEB_OPTIMIZER_FMAX` | `0.05` | Force convergence criterion for NEB optimizer. |
| `NEB_OPTIMIZER_STEPS` | `300` | Maximum NEB optimizer steps. |
| `ENDPOINT_OPTIMIZER_STEPS` | `200` | Maximum endpoint optimization steps. |
| `FIX_OTHER_H` | `0` in `lammps_neb.py` | If true, freezes non-moving H atoms in NEB. |
| `FIX_FE` | `0` | If true, freezes Fe atoms in NEB. |
| `FIX_SUBLATTICE_H` | unset | Optional override for fixing H sublattice. |
| `FIX_SUBLATTICE_FE` | unset | Optional override for fixing Fe sublattice. |

In shell mode, atoms in the outer shell are frozen via the `freeze_mask` array regardless of `FIX_FE`/`FIX_OTHER_H`.

### LAMMPS potential settings

| Input | Default | Meaning |
|---|---|---|
| `POTENTIAL` | `EAM` | `EAM` or `NNP`. |
| `POTENTIAL_EAM_FILE` | `PotentialB3410-modified.fs` relative to code directory | EAM/fs potential file. |
| `LAMMPS_FILES` | EAM file | Comma-separated files copied/attached to LAMMPSlib. |
| `LAMMPS_PAIR_STYLE` | empty | Custom LAMMPS `pair_style` line. |
| `LAMMPS_PAIR_COEFF` | empty | Custom LAMMPS `pair_coeff` line. |
| `POTENTIAL_NNP_DIR` | user-specific default path in `config.py` | NNP potential directory. |
| `NNP_DIR` | `POTENTIAL_NNP_DIR` | Runtime NNP directory. |
| `NNP_ELEMENTS` | `Fe H` | Element order for NNP `pair_coeff`. |
| `NNP_CUTOFF` | `6.60` | HDNNP cutoff. |
| `NNP_SHOWEWSUM` | `1000` | HDNNP setting. |
| `NNP_MAXEW` | `1000000` | HDNNP setting. |
| `NNP_CFLENGTH` | `1.8897261328` | HDNNP length conversion. |
| `NNP_CFENERGY` | `0.0367493254` | HDNNP energy conversion. |

### Cache and MPI job scheduling

| Input | Default | Meaning |
|---|---|---|
| `CACHE_SCHEMA` | auto-generated | Manual override for cache file naming. Use this if you want to reuse or isolate barrier caches. |
| `BARRIER_CACHE_FILE` | empty | Optional explicit cache file to preload, independent of `CACHE_SCHEMA`. Cache hits use this data; cache misses still run NEB unless `JOB_ASSIGNMENT_MODE=all_jobs`. Aliases: `KMC_BARRIER_CACHE_FILE`, `MASTER_BARRIER_CACHE_FILE`. |
| `BARRIER_MERGE_MODE` | `global` | `global`, `local`, or `mixed`. Controls how NEB results are shared across ranks. |
| `BARRIER_MERGE_INTERVAL` | `100` | Interval for mixed cache sharing. |
| `JOB_ASSIGNMENT_MODE` | `cache_dedupe` | `all_jobs`, `cache_no_dedupe`, `cache_dedupe`, or `cache_only`. Use `cache_dedupe` with `BARRIER_CACHE_FILE` when you want master-cache hits plus NEB for misses. |

Cache files are written as:

```text
barrier_cache_rank<RANK>_<SCHEMA>.pkl
barrier_cache_rank<RANK>_<SCHEMA>.pkl.delta.pkl
barrier_cache_<SCHEMA>.pkl
```

Use a new `CACHE_SCHEMA` when you change anything that changes the physical meaning of a barrier, such as potential, local environment mode, shell radius, map file, or binning settings.

Example:

```bash
export CACHE_SCHEMA=my_gb_sigma5_shell6_8_eam
```

### Diagnostics and validation

| Input | Default | Meaning |
|---|---:|---|
| `DEBUG_MODE` | `1` | Enables rank logs, timing CSVs, NEB diagnostics, ASE logs. |
| `ASE_LOG_DIR` | `ase_logs` | Directory for ASE optimizer logs and optional structure snapshots. |
| `DUMP_EVERY_STEPS` | `1` | Diagnostic dump interval. |
| `MSD_LOG_INTERVAL` | `1000` | MSD output interval. |
| `MSD_LOG_FILE` | `msd_vs_time.csv` | MSD CSV path. |
| `MSD_PLOT_FILE` | `msd_plot.png` | MSD plot path. |
| `NEB_DIAG` | `1` | Enables NEB diagnostic summaries when debug is on. |
| `NEB_JOB_TRACE` | `0` | More verbose NEB job tracing. |
| `NEB_DUMP` | `0` | Extra NEB dump mode. |
| `NEB_FAILURE_CAPTURE` | `0` | Capture failed NEB cases. |
| `NEB_FAILURE_DIR` | `neb_failure_capture` | Failure-capture output folder. |
| `VALIDATION_MODE` | `0` | Write validation structures for NEB jobs. |
| `VALIDATION_OUT` | `validation_dump` | Validation output directory. |
| `VALIDATION_ASE_DUMP` | `0` | Store ASE NEB image dumps when validation is enabled. |
| `VALIDATION_ASE_DUMP_PRE` | `0` | Store pre-optimization ASE NEB image dumps. |
| `VALIDATION_ASE_DIR` | `ase` | Subdirectory under each validation case. |

For a less noisy production run:

```bash
export DEBUG_MODE=0
```

For debugging a structure/NEB problem:

```bash
export DEBUG_MODE=1
export VALIDATION_MODE=1
export VALIDATION_ASE_DUMP=1
export NEB_DUMP=1
```

Note: validation ASE band dumps are disabled automatically for multi-rank runs in `config.py`.

---

## 8. Output files to expect

Common outputs:

| Output | Meaning |
|---|---|
| `log_rank0.txt`, `log_rank1.txt`, ... | Per-rank runtime logs when debug is enabled. |
| `timing_rank0.csv`, ... | Timing buckets per KMC step. |
| `kmc_diagnostics_rank0.log` | Selected KMC moves, time increments, total simulated time. |
| `kmc_timestep_vs_step.csv` | KMC timestep and total simulated time per step. |
| `H_trajectory_onlyH.lammpstrj` | H-only trajectory. |
| `rates_allranks.csv` | Candidate barriers/rates and their source/status. |
| `neb_diag_rank<RANK>.csv` | NEB timing and atom-count diagnostics for each rank. |
| `neb_diag_all.csv` | Combined NEB diagnostics written on rank 0. |
| `msd_vs_time.csv` | MSD as a function of KMC time. |
| `msd_plot.png` | MSD plot. |
| `barrier_cache_rank*.pkl`, `.delta.pkl` | Rank-local barrier cache and append-only delta cache. |
| `barrier_cache_<SCHEMA>.pkl` | Merged barrier cache after the run. |
| `kmc_restart_checkpoint.pkl` | Latest lightweight restart checkpoint with H state, KMC time, RNG state, and cache schema. |
| `kmc_restart_checkpoint_step<N>.pkl` | Step-specific checkpoint written every `RESTART_CHECKPOINT_INTERVAL` accepted steps. |
| `validation_dump/` | NEB validation structures when `VALIDATION_MODE=1`. |
| `local_env_builds/` | Local environment examples when `BUILD_LOCAL_ENV_ONLY=1`. |

### Restart inputs

Restart mode reads a previous run directory and starts from an existing H-only
trajectory frame. The restart step is the `ITEM: TIMESTEP` value in
`H_trajectory_onlyH.lammpstrj`; the next KMC loop begins at that same step.
For example, restarting from trajectory frame `100000` and setting
`STEPS=120000` continues with KMC steps `100000..119999`.

Required previous-run files/directories:

| Input | Required | Meaning |
|---|---:|---|
| `kmc_restart_checkpoint_step<N>.pkl` or `kmc_restart_checkpoint.pkl` | preferred | Restores occupied H sites, unwrapped positions, simulated time, cache schema, and Python/NumPy RNG state. This gives an exact continuation from a checkpointed step. |
| `H_trajectory_onlyH.lammpstrj` | yes | H occupancy and unwrapped H positions at `RESTART_STEP`. `H_only.lammpstrj` is accepted as a fallback filename. |
| `kmc_timestep_vs_step.csv` | preferred | Restores the total simulated KMC time. Frame `N` uses the CSV `total` from row `step=N-1`. |
| `barrier_cache_<SCHEMA>.pkl` | yes by default | Preferred restart barrier cache. Must match the current `CACHE_SCHEMA`. |
| `barrier_cache_rank<RANK>_<SCHEMA>.pkl` and/or `.delta.pkl` | fallback | Used when the merged cache is absent. Delta files are replayed automatically. |
| Original site-map input files | yes | Use the same `KMC_SITE_MAP_FILE` and `KMC_HOST_STRUCTURE_FILE` for external-map runs. |
| Original physical/cache settings | yes | Keep potential, environment mode, shell radii, binning, map, and `CACHE_SCHEMA` consistent with the previous run. |
| `kmc_diagnostics_rank0.log` | time fallback | Used to recover `t_total` if `kmc_timestep_vs_step.csv` is missing/incomplete; also useful to verify selected moves and run history. |

Example:

```bash
export RESTART_MODE=1
export RESTART_DIR=/path/to/previous_run
export RESTART_STEP=100000
export STEPS=120000

# Keep these consistent with the previous run:
export NUM_H=50
export KMC_SITE_MAP_FILE=/path/to/sigma5_stage3_unified_sites.npz
export KMC_HOST_STRUCTURE_FILE=/path/to/sigma5_210-20-20-5.lmp
export CACHE_SCHEMA='v4_envkey:mode:shell_IR5.0_OR6.0:env_plus_dir:R5.0_PB0.1_HB0.02:map:sigma5_stage3_unified_sites'
```

Optional restart variables:

| Variable | Default | Meaning |
|---|---|---|
| `RESTART_TRAJECTORY_FILE` | `H_trajectory_onlyH.lammpstrj` | Trajectory file inside `RESTART_DIR`, or an absolute path. |
| `RESTART_TIMESTEP_FILE` | `kmc_timestep_vs_step.csv` | KMC time CSV inside `RESTART_DIR`, or an absolute path. |
| `RESTART_CHECKPOINT_FILE` | auto | Exact checkpoint file to load. If unset, restart first tries `kmc_restart_checkpoint_step<RESTART_STEP>.pkl`, then `kmc_restart_checkpoint.pkl`. |
| `RESTART_STRICT` | `1` | Fail if trajectory/time/cache inputs are missing or inconsistent. Set `0` only for manual recovery. |

Checkpoint writing variables:

| Variable | Default | Meaning |
|---|---|---|
| `RESTART_CHECKPOINT_INTERVAL` | `1000` | Write a checkpoint every N accepted KMC steps. Set `0` to disable periodic checkpoints. |
| `RESTART_CHECKPOINT_DIR` | `.` | Directory where checkpoint files are written. |
| `RESTART_CHECKPOINT_PREFIX` | `kmc_restart_checkpoint` | Prefix for latest and step-specific checkpoint files. |

---

## 9. How external-map mode works internally

When `KMC_SITE_SOURCE=external`, the driver:

1. Loads site positions from `KMC_SITE_MAP_FILE` using `load_kmc_site_map()`.
2. Loads Fe host atoms from `KMC_HOST_STRUCTURE_FILE` using `load_lammps_atomic_structure()`.
3. Checks that the map box and host box match when both are available.
4. Stores the external Fe coordinates instead of regenerating an ideal BCC Fe stencil.
5. Builds the full structure as `host Fe + occupied H`.
6. For each NEB hop, builds a shell local environment directly from the explicit host Fe positions.

This is the correct mode for grain boundaries, bicrystals, strained structures, surfaces, or any non-ideal Fe geometry where regenerated bulk BCC Fe positions would be wrong.

---

## 10. Common changes and exactly what to edit

### Change from 540 H to 100 H

Do not edit the code. Run:

```bash
export NUM_H=100
```

### Change from 100 KMC steps to 5000 steps

```bash
export STEPS=5000
```

### Use a different EAM potential

```bash
export POTENTIAL=EAM
export POTENTIAL_EAM_FILE=/absolute/path/to/my_potential.fs
export LAMMPS_FILES=/absolute/path/to/my_potential.fs
```

### Use a different external map

```bash
export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE=/absolute/path/to/new_map.npz
export KMC_HOST_STRUCTURE_FILE=/absolute/path/to/new_host.data
export CACHE_SCHEMA=new_map_shell6_8_eam
```

Changing the map should also change the cache schema unless you intentionally want to mix barrier data.

### Make NEB faster but less relaxed

```bash
export OPTIMIZE_ENDPOINTS=0
export NEB_OPTIMIZER_STEPS=100
```

Use this only for testing. For production barriers, keep endpoint optimization on unless you have verified it is unnecessary.

### Inspect local environments without running KMC

```bash
export BUILD_LOCAL_ENV_ONLY=1
export BUILD_LOCAL_ENV_COUNT=10
export LOCAL_ENV_MODE=shell
mpirun -np 1 python -m kmc.main
```

Then inspect:

```text
local_env_builds/env_*/initial.xyz
local_env_builds/env_*/final.xyz
local_env_builds/env_*/atom_roles.csv
local_env_builds/env_*/meta.json
```

---

## 11. Troubleshooting

### Error: external map requires site map and host structure

You are in external mode but one path is missing. Set both:

```bash
export KMC_SITE_MAP_FILE=/path/to/map.npz
export KMC_HOST_STRUCTURE_FILE=/path/to/host.data
```

### Error: site-map box does not match host box

The map and host structure were probably generated from different cells or one file has a shifted/incorrect box. Regenerate the map from the same host structure or verify `box_lengths` in the `.npz` file.

### Error: no `Atoms # atomic` rows found

The host structure loader expects a LAMMPS atomic data file with an `Atoms # atomic` section. Confirm the file is not a dump file and that atom rows look like:

```text
atom_id atom_type x y z
```

### NEB jobs return `inf` barriers

Possible causes:

- Endpoint optimization failed.
- Initial/final local structures have different atom counts.
- The moving H could not be identified in shell mode.
- The potential file path is wrong.
- LAMMPSlib failed silently after a bad structure.

Recommended debug settings:

```bash
export DEBUG_MODE=1
export VALIDATION_MODE=1
export NEB_DUMP=1
export STRUCTURE_LOG_ALL=1
```

Then inspect rank logs, validation dumps, and `ase_logs/`.

### Cache seems to reuse wrong barriers

Change the cache schema:

```bash
export CACHE_SCHEMA=my_new_physics_case_v1
```

Use a different schema whenever you change potential, site map, shell radii, local environment key settings, or any setting that changes the physical barrier.

### Multi-rank run hangs

MPI hangs usually mean one rank hit an exception before another collective. Check all `log_rank*.txt` files for the first rank that reports an error. For debugging, reduce to one rank:

```bash
mpirun -np 1 python -m kmc.main
```

Then increase ranks after the single-rank run is stable.

---

## 12. Minimal command templates

### Minimal generated-site run

```bash
export KMC_SITE_SOURCE=generated
export NUM_H=10
export STEPS=5
export POTENTIAL=EAM
export POTENTIAL_EAM_FILE=/path/to/PotentialB3410-modified.fs
mpirun -np 1 python -m kmc.main
```

### Minimal external-map run

```bash
export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE=/path/to/kmc_map.npz
export KMC_HOST_STRUCTURE_FILE=/path/to/host_structure.data
export NUM_H=10
export STEPS=5
export LOCAL_ENV_MODE=shell
export POTENTIAL=EAM
export POTENTIAL_EAM_FILE=/path/to/PotentialB3410-modified.fs
mpirun -np 1 python -m kmc.main
```

---

## 13. Practical notes

- Keep input changes in a small run script, for example `run_kmc.sh`, rather than editing the Python files for every case.
- Use absolute paths for potential files, maps, and host structures when running from different directories.
- Use build-only mode before expensive external-map production runs.
- Start with `mpirun -np 1` until initialization and one NEB job work correctly.
- Increase ranks only after the single-rank setup is stable.
- Change `CACHE_SCHEMA` whenever the physical setup changes.
- For grain boundary or other non-ideal structures, prefer external-map mode with shell local environments.
