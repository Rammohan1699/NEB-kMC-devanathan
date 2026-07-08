# Site Unified

`Site_unified` is the source-only lattice-site discovery pipeline. It keeps the scripts that parse LAMMPS host structures, classify bulk and grain-boundary regions, generate candidate interstitial sites, relax/filter candidates, and merge the accepted candidates into a unified KMC lattice map.

## Complex Structures

The pipeline is intended for BCC Fe bicrystals and polycrystals whose simulation cell can be represented with orthogonal periodic box bounds. The shared `lammps_data_utils.py` parser accepts common LAMMPS `Atoms` styles used by structure builders:

- `Atoms # atomic`: `id type x y z [ix iy iz]`
- `Atoms # charge`: `id type q x y z [ix iy iz]`
- `Atoms # molecular`, `bond`, or `angle`: `id mol type x y z [ix iy iz]`
- `Atoms # full`: `id mol type q x y z [ix iy iz]`
- unlabeled `Atoms` sections, using coordinate-column inference for the styles above

Tilted triclinic boxes are rejected with an explicit error. Orthogonalize/convert those structures before using this pipeline because the current KD-tree and cell-list algorithms use orthogonal periodic minimum-image distances.

For polycrystals, `gb_region_classifier_grain_graph_fast.py` detects any number of grains from local BCC orientation continuity. It writes both Fe-local masks and all-atom `region_code_all`/`grain_id_all` arrays. Downstream map outputs preserve grain context:

- bulk tetra sites include `grain_id` from the nearest Fe grain.
- GB/transition candidates include `support_grains`, `n_support_grains`, and `interface_class`.
- final unified maps preserve those fields, with `interface_class` values such as `BICRYSTAL_INTERFACE` and `POLYCRYSTAL_JUNCTION`.

If a structure builder uses atom types to label grains, pass `--fe-type 0` to treat every atom type as Fe, or pass `--fe-types 1,2,3` to select specific Fe grain-label types. These options are available on the stage-1 driver, classifier, GB candidate generator, and relaxation/filter scripts.

## Pipeline

1. `kmc_site_discovery_stage1_engine.py`: run region classification plus bulk tetrahedral and GB candidate generation for an input host structure.
2. `kmc_site_discovery_stage2_mpi_progress_commself.py`: relax and filter candidate sites in parallel using independent LAMMPS communicators.
3. `kmc_site_discovery_stage3_merge_sites_diagnostics_nodedup.py`: merge accepted bulk/GB candidates and emit diagnostics plus the final unified map.

The lower-level helpers remain available for focused debugging:

- `bulk_tetra_template_mapper_fast_cell.py`
- `gb_region_classifier_grain_graph_fast.py`
- `gb_candidate_repetition_sampler.py`
- `gb_transition_voronoi_candidates.py`
- `relax_filter_interstitial_sites.py`
- `relax_filter_interstitial_sites_local_sphere.py`
- `relax_filter_interstitial_sites_local_sphere_tiled.py`
- `export_regions_for_ovito.py`

Generated `.csv`, `.lmp`, `.npz`, `.xyz`, and summary `.txt` files are intentionally ignored. Put run products in a dedicated run directory rather than committing them here.
