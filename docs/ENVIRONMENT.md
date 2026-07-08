# Environment Reference

The launchers configure the driver through environment variables. This file lists the variables that matter for the consolidated base.

## Basic Run Shape

- `STEPS`: final KMC step count for the current run or segment.
- `MPI_RANKS` / `NP`: number of MPI ranks used by launcher scripts.
- `KMC_SEED`: KMC random seed. `DEVANATHAN_SEED` defaults to this when unset.
- `NX`, `NY`, `NZ`, `LATTICE_A`: generated BCC cell dimensions and lattice constant.
- `NUM_H`: initial H count for ordinary generated/external initializations.
- `KNN_K`: nearest-neighbor count for building candidate site connectivity.

## Site Maps And Geometry

- `KMC_SITE_SOURCE`: `generated` or `external`.
- `KMC_SITE_MAP_FILE`: `.npz` site map for external-map runs.
- `KMC_HOST_STRUCTURE_FILE`: LAMMPS data file for the explicit Fe host.
- `KMC_HOST_FE_TYPE`: Fe atom type in the host data file, normally `1`.
- `KMC_INITIAL_H_REGION_FILE`: region mask file, usually `sigma5_site_regions.npz`.
- `KMC_INITIAL_H_REGIONS`: comma-separated region names for ordinary initial H selection.
- `KMC_HOP_X_DIRECTION`: `all`, `right`/`positive`, or `left`/`negative`; pulse diagnostics normally use `right`.

## Local Environments And Barrier Cache Keys

- `LOCAL_ENV_MODE`: `shell`, `radial`, or `wrapped`. Production LAMMPS-only runs normally use `shell`.
- `ENV_KEY_MODE`: `env_plus_dir` includes hop direction in the cache key; `env_only` does not.
- `ENV_RADIUS_A`: radial key radius.
- `POS_BIN_A`: position quantization for environment keys.
- `HOP_BIN_A`: hop-vector quantization for environment keys.
- `SHELL_INNER_RADIUS_A`: movable inner shell radius.
- `SHELL_OUTER_RADIUS_A`: frozen outer shell radius.
- `CACHE_SCHEMA`: semantic cache namespace. Change this when engine, shell, map, source mode, endpoint optimization, or key definition changes.
- `BARRIER_CACHE_FILE`: optional existing migration-barrier cache to preload.
- `KMC_CACHE_DIR`: output directory for merged barrier caches and rank deltas.
- `JOB_ASSIGNMENT_MODE`: usually `cache_dedupe`.
- `BARRIER_MERGE_MODE`: `global` for native LAMMPS-only grouped runs; `local` can be used for ASE-NEB comparisons.
- `BARRIER_MERGE_INTERVAL`: merge cadence in steps.

## NEB Engines

- `NEB_ENGINE=ase_lammps`: legacy ASE NEB using LAMMPS calculators.
- `NEB_ENGINE=lammps_only`: native grouped LAMMPS NEB.
- `NEB_ENGINE=lammps_only_endpoint_opt`: native grouped LAMMPS NEB with native LAMMPS endpoint minimization before NEB.
- `NIMG`: number of NEB images.
- `OPTIMIZE_ENDPOINTS`: ASE-side endpoint optimization toggle.
- `NEB_OPTIMIZER_FMAX`, `NEB_OPTIMIZER_STEPS`, `ENDPOINT_OPTIMIZER_STEPS`: ASE optimizer controls.

Native LAMMPS-only controls:

- `LAMMPS_NEB_REPLICAS`: MPI ranks per NEB calculation. `MPI_RANKS % LAMMPS_NEB_REPLICAS` must be zero.
- `LAMMPS_NEB_STEPS`: native NEB max steps.
- `LAMMPS_NEB_FTOL`: native NEB force tolerance.
- `LAMMPS_NEB_MIN_STYLE`: e.g. `quickmin`.
- `LAMMPS_NEB_DEBUG_MODE`: `0` removes scratch after parsing; `1` retains LAMMPS inputs/logs.
- `LAMMPS_NEB_SCRATCH_DIR`: native NEB scratch root.
- `LAMMPS_NEB_MIN_OPEN_FILES`: launcher target for raising `ulimit -n`.
- `LAMMPS_ONLY_ENDPOINT_OPTIMIZE`: native endpoint optimization toggle.
- `LAMMPS_ENDPOINT_OPT_MIN_STYLE`, `LAMMPS_ENDPOINT_OPT_ETOL`, `LAMMPS_ENDPOINT_OPT_FTOL`, `LAMMPS_ENDPOINT_OPT_STEPS`, `LAMMPS_ENDPOINT_OPT_MAXEVAL`, `LAMMPS_ENDPOINT_OPT_DMAX`: native endpoint minimization controls.

## Devanathan Modes

Enable the boundary framework with:

```bash
export DEVANATHAN_ENABLED=1
```

Common geometry:

- `DEVANATHAN_SOURCE_X_MIN_A`: left edge of the controlled source/charging region.
- `DEVANATHAN_SOURCE_X_MAX_A`: right edge of the source/charging region.
- `DEVANATHAN_SOURCE_LAYER_A`: width used when `SOURCE_X_MAX_A` is not set.
- `DEVANATHAN_SOURCE_FRACTION`: target H/Fe or source-site fraction, depending on mode.
- `DEVANATHAN_SINK_X_MIN_A`: right absorbing boundary.
- `DEVANATHAN_LEFT_SINK_X_MAX_A`: optional left absorber.
- `DEVANATHAN_LEFT_SINK_REGION` / `DEVANATHAN_LEFT_SINK_REGIONS`: region-mask version of the left absorber.
- `DEVANATHAN_CROSS_SECTION_A2`: flux normalization area; defaults to `Ly * Lz`.
- `DEVANATHAN_TRIM_SOURCE_EXCESS`: trim source occupancy when above target.

`DEVANATHAN_SOURCE_MODE=constant`:

- Initializes and maintains a constant source concentration.
- Uses `DEVANATHAN_SOURCE_FRACTION` and the source Fe count.

`DEVANATHAN_SOURCE_MODE=pulse`:

- Initializes a finite pulse and does not refill the source.
- `DEVANATHAN_PULSE_N_H`, `DEVANATHAN_CHARGING_ZONE_N_H`, or `DEVANATHAN_SOURCE_TARGET_N_H`: pulse size.
- `DEVANATHAN_PULSE_REGION`, `DEVANATHAN_CHARGING_ZONE_REGION`, or `DEVANATHAN_SOURCE_REGION`: region-mask source selector.
- Often paired with `KMC_STOP_WHEN_NO_H=1` and `KMC_HOP_X_DIRECTION=right`.

`DEVANATHAN_SOURCE_MODE=gcmc`:

- Performs fixed chemical-potential GCMC initialization of the source and production maintenance.
- Good for charging-zone source behavior.

`DEVANATHAN_SOURCE_MODE=gcmc_bulk`:

- Controls a bulk x-window or external-map region rather than only the narrow source layer.
- `DEVANATHAN_BULK_X_MIN_A`, `DEVANATHAN_BULK_X_MAX_A`: controlled x-window.
- `DEVANATHAN_BULK_REGION` / `DEVANATHAN_BULK_REGIONS`: external-map region selector.
- `DEVANATHAN_BULK_TARGET_H_PER_FE` / `DEVANATHAN_BULK_TARGET_FRACTION`: target concentration.
- `DEVANATHAN_BULK_TARGET_N_H`: direct target H-count override.
- `DEVANATHAN_BULK_FE_COUNT`: manual Fe count override.
- `DEVANATHAN_BULK_SINK_GAP_A`: default gap between bulk control max and sink.

## GCMC Controls

- `DEVANATHAN_GCMC_MU_EV`: H chemical potential in eV.
- `DEVANATHAN_GCMC_TEMPERATURE_K`: GCMC temperature.
- `DEVANATHAN_GCMC_KB_EV_PER_K`: Boltzmann constant.
- `DEVANATHAN_GCMC_INITIALIZATION_START`: `empty` or `target`.
- `DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE`: `mean` or `target`.
- `DEVANATHAN_GCMC_PRODUCTION_MODE`: `insert_only`, `insert_delete`, or `balanced`.
- `DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT`: proposals after each KMC step.
- `DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS`, `DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS`: initialization bounds.
- `DEVANATHAN_GCMC_CONVERGENCE_WINDOW`, `DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL`, `DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H`, `DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE`, `DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS`: convergence criteria.
- `DEVANATHAN_GCMC_ACCEPT_NEAR_TARGET_AT_MAX`: allow near-target initialization at max attempts.
- `DEVANATHAN_GCMC_PROGRESS_INTERVAL`: progress diagnostic cadence.
- `DEVANATHAN_GCMC_ENERGY_CACHE_ENABLED`: toggle persistent source-energy cache.
- `DEVANATHAN_GCMC_ENERGY_CACHE_FILE`: persistent GCMC insertion-energy cache path.
- `DEVANATHAN_GCMC_ENERGY_CACHE_POS_BIN_A`: GCMC environment position bin.
- `DEVANATHAN_GCMC_ENERGY_CACHE_SAVE_INTERVAL`: cache flush cadence.

## Incremental/Fenwick Event Path

- `KMC_INCREMENTAL_EVENTS=1`: use impacted-region event rebuilding and Fenwick weighted selection.
- `KMC_INCREMENTAL_IMPACT_RADIUS_A`: radius used to rebuild impacted H event sources after moves, insertions, and deletions.

This mode is intended for long GCMC/Fenwick campaigns where only a small region of the event table changes after each accepted KMC move.

## Restart And Output

- `RESTART_MODE=1`: resume from checkpoint.
- `RESTART_DIR`: previous run directory.
- `RESTART_STEP`: step represented by the checkpoint.
- `RESTART_CHECKPOINT_FILE`: checkpoint path relative to `RESTART_DIR` or absolute.
- `RESTART_STRICT`: fail if checkpoint inputs are inconsistent.
- `RESTART_CHECKPOINT_INTERVAL`: checkpoint cadence.
- `KMC_APPEND_OUTPUTS=1`: append diagnostics/trajectories in a resumed directory.
- `KMC_LOG_DIR`, `KMC_DIAGNOSTICS_DIR`, `KMC_TRAJECTORY_DIR`, `KMC_NEB_DIR`: output roots.
- `DUMP_EVERY_STEPS`: H trajectory cadence.
- `WRITE_RATES_ALLRANKS`: write per-rank rates table.
- `DEBUG_MODE` / `DEBUG_LOGGING`: rank logs and verbose diagnostics.
- `KMC_HEARTBEAT` / `KMC_PHASE_HEARTBEAT`: write rank heartbeat files.
- `KMC_STACK_DUMP_SIGNAL`: install SIGUSR1 stack dump handler.

## Validation Dumps

- `VALIDATION_MODE`: enable validation outputs.
- `VALIDATION_ASE_DUMP`: dump selected ASE structures.
- `VALIDATION_ASE_DUMP_PRE`: dump pre-optimization ASE structures.
- `VALIDATION_OUT`: validation output directory.
