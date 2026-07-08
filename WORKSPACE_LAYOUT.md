# Workspace Layout

The root directory is now organized as follows.

```text
kmc/                         active development KMC engine
kmc_cluster_fresh/           cluster-ready package and Slurm scripts
kmc_local_share/             clean local/shareable package
kmc_map_inputs/              local Sigma5 map inputs
kmc_map_generator/           map-generation scripts
tools/                       local postprocess/helper scripts
local_runs/                  local completed experiment runs
cluster_results/             copied cluster results
archived_root_runs/          old root-level run outputs moved out of the root
local_packages/              zip packages for sharing
docs/                        reference docs copied from the root
```

The primary script guide is:

```text
JOB_SCRIPT_INDEX.md
```

The root should no longer be used as a run directory. Use `RUN_ROOT`, package
launchers, or Slurm scripts so outputs are written under `runs/`,
`benchmark_runs/`, `local_runs/`, or `cluster_results/`.
