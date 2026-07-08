# KMC Map Generator Routine

This directory contains the working site-map generation pipeline used for the
CSL sigma-5 Fe bicrystal. The goal is to produce a unified list of hydrogen
interstitial sites for KMC: bulk BCC tetrahedral sites away from the interface,
plus relaxed grain-boundary/transition sites near the interface.

The current pipeline is organized as three stages:

1. `kmc_site_discovery_stage1_engine.py`
2. `kmc_site_discovery_stage2_mpi_progress_commself.py`
3. `kmc_site_discovery_stage3_merge_sites_diagnostics_nodedup.py`

Older component scripts are kept because stage 1 calls them directly and they
are useful for debugging individual pieces.

## Inputs

The required starting point is an orthogonal LAMMPS atomic data file containing
the Fe host structure. For the validated example this is:

```bash
sigma5_210-20-20-5.lmp
```

The workflow assumes:

- Fe atom type is `1`.
- The structure is periodic and orthogonal.
- The BCC lattice parameter is supplied with `--a0`.
- For LAMMPS relaxation, an Fe-H EAM/fs potential is available.

## Stage 1: Region Classification And Candidate Generation

Driver:

```bash
python kmc_site_discovery_stage1_engine.py sigma5_210-20-20-5.lmp \
  --a0 2.856 \
  --out-prefix sigma5_stage1
```

Stage 1 does not relax hydrogen. It creates the geometric candidate map needed
for relaxation.

Internally it runs:

- `gb_region_classifier_grain_graph_fast.py`
- `bulk_tetra_template_mapper_fast_cell.py`
- `gb_transition_voronoi_candidates.py`

### 1A. Grain/GB Region Classifier

Script:

```bash
gb_region_classifier_grain_graph_fast.py
```

Purpose:

- Fits local BCC orientation frames around Fe atoms.
- Groups atoms into grain regions by orientation continuity.
- Labels atoms as bulk-template, transition, or GB/Voronoi region.

Main outputs:

```text
sigma5_stage1_regions_masks.npz
sigma5_stage1_regions_atoms.csv
sigma5_stage1_regions_region_atoms.xyz
sigma5_stage1_regions_summary.txt
```

The masks file is consumed by the bulk tetra mapper and the GB Voronoi candidate
generator.

### 1B. Bulk Tetrahedral Site Mapping

Script:

```bash
bulk_tetra_template_mapper_fast_cell.py
```

Purpose:

- Uses only `BULK_TEMPLATE` Fe atoms.
- Fits a local BCC frame.
- Places analytical BCC tetrahedral sites in bulk-like regions.
- Excludes candidates near GB/core regions.
- Merges duplicate tetra sites generated from neighboring Fe atoms.

Main outputs:

```text
sigma5_stage1_bulk_tetra_sites.csv
sigma5_stage1_bulk_tetra_sites.xyz
sigma5_stage1_bulk_tetra_sites.lmp
sigma5_stage1_bulk_tetra_sites.npz
sigma5_stage1_bulk_tetra_summary.txt
```

### 1C. GB/Transition Voronoi Candidate Generation

Script:

```bash
gb_transition_voronoi_candidates.py
```

Purpose:

- Generates candidate H positions only in transition and GB/core regions.
- Builds local Fe neighborhoods around selected seed atoms.
- Uses local Voronoi vertices as candidate interstitial sites.
- Applies Fe-H distance filters and region-support checks.
- Merges duplicate candidates.

Main outputs:

```text
sigma5_stage1_gb_candidates.csv
sigma5_stage1_gb_candidates.xyz
sigma5_stage1_gb_candidates.lmp
sigma5_stage1_gb_candidates.npz
sigma5_stage1_gb_candidates_summary.txt
```

For the sigma-5 example, the recorded stage-1 manifest is:

```text
a0 = 2.856
orientation_cutoff_deg = 8.0
gb_buffer = 6.0
transition_buffer = 9.0
max_frame_error = 0.20
local_radius = 5.0
collect_radius = 3.2
min_fe_dist = 1.05
max_fe_dist = 2.35
region_support_radius = 3.0
merge_radius = 0.25
```

## Stage 2: Relax GB/Transition Candidate Sites

Driver:

```bash
mpirun -np 8 python kmc_site_discovery_stage2_mpi_progress_commself.py \
  sigma5_210-20-20-5.lmp \
  --sites sigma5_stage1_gb_candidates.csv \
  --mode lammps \
  --relax local-sphere-cluster \
  --local-fe-radius 10.0 \
  --cluster-radius 12.0 \
  --cluster-vacuum 6.0 \
  --min-fe-dist 1.05 \
  --max-fe-dist 2.35 \
  --max-relax-displacement 1.50 \
  --merge-radius 0.25 \
  --fmax 0.03 \
  --steps 200 \
  --optimizer fire \
  --rank-progress 100 \
  --out-prefix mpi_test \
  --lammps-cmd "pair_style eam/fs" \
  --lammps-cmd "pair_coeff * * PotentialB3410-modified.fs Fe H" \
  --lammps-cmd "neighbor 2.0 bin" \
  --lammps-cmd "neigh_modify delay 0 every 1 check yes"
```

Purpose:

- Inserts one H candidate at a time into the Fe host.
- Relaxes the local H/Fe cluster using ASE + LAMMPSlib.
- Filters by Fe-H distance, displacement, and convergence rules.
- Merges accepted minima into relaxed GB/transition sites.

Important implementation detail:

- Each Python MPI rank creates its own LAMMPSlib instance with `MPI.COMM_SELF`.
  This avoids sharing a LAMMPS communicator across ranks.

Main outputs:

```text
mpi_test_all_results.csv
mpi_test_accepted_sites.csv
mpi_test_accepted_sites.xyz
mpi_test_accepted_sites.lmp
mpi_test_accepted_sites.npz
mpi_test_accepted_sites_anchored.csv
mpi_test_accepted_sites_anchored.xyz
mpi_test_accepted_sites_anchored.lmp
mpi_test_rank####_all_results.csv
mpi_test_summary.txt
```

For the sigma-5 example:

```text
Input GB candidates: 51400
Accepted before merge: 51400
Accepted after merge: 43977
Rejected: 0
```

## Stage 3: Merge Bulk And Relaxed GB Sites

Driver:

```bash
python kmc_site_discovery_stage3_merge_sites_diagnostics_nodedup.py \
  sigma5_210-20-20-5.lmp \
  --bulk-sites sigma5_stage1_bulk_tetra_sites.csv \
  --gb-sites mpi_test_accepted_sites.csv \
  --bulk-gb-exclusion 0.35 \
  --out-prefix sigma5_stage3
```

Purpose:

- Reads the bulk tetrahedral sites from stage 1.
- Reads the relaxed GB/transition sites from stage 2.
- Removes bulk sites that overlap relaxed GB sites.
- Concatenates the remaining bulk sites and GB sites.
- Writes diagnostics and final unified site outputs.

This stage intentionally does not do a final all-site deduplication. It only
removes bulk sites that conflict with GB-relaxed sites. This preserves the
physical distinction between bulk-template sites and GB-relaxed minima.

Main outputs:

```text
sigma5_stage3_overlap_diagnostics.csv
sigma5_stage3_nearest_distances.csv
sigma5_stage3_unified_sites.csv
sigma5_stage3_unified_sites.xyz
sigma5_stage3_unified_sites.lmp
sigma5_stage3_unified_sites.npz
sigma5_stage3_summary.txt
```

For the sigma-5 example:

```text
Original bulk sites: 194800
Original GB sites: 43977
Bulk sites removed by GB overlap: 8400
Bulk sites kept: 186400
GB sites kept: 43977
Unified total sites: 230377
```

Output site types:

```text
type 1 = BULK_TETRA
type 2 = GB_RELAXED
```

## Final Site Map For KMC

The primary final files are:

```text
sigma5_stage3_unified_sites.csv
sigma5_stage3_unified_sites.lmp
sigma5_stage3_unified_sites.npz
```

The CSV has columns:

```text
site_id,x,y,z,site_type,site_label
```

The NPZ is the most convenient input for code because it keeps positions,
types, labels, and box metadata in structured arrays.

## Script Inventory

Current staged drivers:

```text
kmc_site_discovery_stage1_engine.py
kmc_site_discovery_stage2_mpi_progress_commself.py
kmc_site_discovery_stage3_merge_sites_diagnostics_nodedup.py
```

Component/helper scripts:

```text
gb_region_classifier_grain_graph_fast.py
bulk_tetra_template_mapper_fast_cell.py
gb_transition_voronoi_candidates.py
relax_filter_interstitial_sites_local_sphere_tiled.py
```

The `relax_filter_interstitial_sites_local_sphere_tiled.py` script appears to
be an earlier or alternate non-MPI/local-sphere relaxation path. The
`kmc_site_discovery_stage2_mpi_progress_commself.py` driver is the clearer
current Stage-2 entry point for the sigma-5 workflow.

## Notes For Code Cleanup

Recommended cleanup order:

1. Add a short module docstring to every script saying whether it is a staged
   driver or a helper.
2. Keep the three staged drivers as the public entry points.
3. Move repeated LAMMPS-data parsing, PBC wrapping, CSV/XYZ/LMP writing, and
   site-file loading into shared utilities.
4. Rename older helper scripts only after the staged workflow is covered by a
   smoke test or saved command manifest.
5. Keep current output filenames stable until the KMC driver consumes the
   unified site map directly.

