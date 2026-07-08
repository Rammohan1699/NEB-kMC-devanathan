# Previously Analyzed Sigma5 KMC Map Inputs

This example directory contains a previously analyzed Sigma5 bicrystal KMC map
and the matching host/region files. The repository root keeps `kmc_map_inputs`
as a compatibility symlink to this directory, so existing scripts that reference
`kmc_map_inputs/...` continue to work.

- `sigma5_stage3_unified_sites.npz`: unified KMC site map from stage 3.
- `sigma5_stage3_unified_sites.csv`: human-readable export of the same map.
- `sigma5_210-20-20-5.lmp`: Fe host structure used to generate the map.
- `sigma5_regions_grain_masks.npz`: Fe-atom region masks from the site-discovery workflow.
- `sigma5_site_regions.npz`: KMC-site region labels parsed from the Fe masks.
- `sigma5_site_regions.csv`: human-readable export of the parsed KMC-site region labels.

`sigma5_site_regions.*` assigns every KMC site to the nearest Fe atom's region
under PBC. The `selection_label` values are intended for later region-aware H
initialization:

- `bulk_grain_0`
- `bulk_grain_1`
- `transition`
- `grain_boundary`

Use these files with external-map mode:

```bash
export KMC_SITE_SOURCE=external
export KMC_SITE_MAP_FILE=kmc_map_inputs/sigma5_stage3_unified_sites.npz
export KMC_HOST_STRUCTURE_FILE=kmc_map_inputs/sigma5_210-20-20-5.lmp
export LOCAL_ENV_MODE=shell
```

To randomly place initial H only in selected region voids:

```bash
export KMC_INITIAL_H_REGION_FILE=kmc_map_inputs/sigma5_site_regions.npz
export KMC_INITIAL_H_REGIONS=bulk_grain_0
```

`KMC_INITIAL_H_REGIONS` accepts one or more comma-separated labels. Exact
labels include `bulk_grain_0`, `bulk_grain_1`, `transition`, and
`grain_boundary`. Aliases include `bulk` for all bulk grains and `grain` or
`gb` for `grain_boundary`. `bulk_0` and `bulk0` are accepted as shorthand for
`bulk_grain_0`.

Regenerate the region inputs after replacing the site map, host structure, or
site-discovery masks:

```bash
python3 tools/prepare_kmc_region_inputs.py
```
