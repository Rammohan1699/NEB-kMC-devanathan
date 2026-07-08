# Publication Checklist

This directory is a public-release candidate copied from `base-working-version`.
It has not been pushed yet.

## Created From

- Source directory: `base-working-version`
- Export command: `rsync -a --delete` with caches/editor files excluded
- Compatibility symlink preserved: `kmc_map_inputs -> examples/previously_analyzed/sigma5_kmc_map_inputs`

## Pre-Push Verification

- Choose and add a `LICENSE` file before making the repository public.
- Review large example files before pushing. Files larger than 10 MB are:
  - `tools/sigma5_stage3_unified_sites.lmp`
  - `tools/sigma5_stage3_unified_sites.csv`
  - `tools/sigma5_stage3_unified_sites.xyz`
  - `examples/previously_analyzed/sigma5_kmc_map_inputs/sigma5_stage3_unified_sites.csv`
  - `examples/previously_analyzed/sigma5_kmc_map_inputs/sigma5_site_regions.csv`
- Review machine-specific paths before pushing. The audit found local paths under:
  - `JOB_SCRIPT_INDEX.md`
  - several scripts in `scripts/`
  - `validation.sh`
  - `helper/run_lammps_jobs.sh`
  - `kmc/config.py`
  - `kmc/cache_new.py`
  - `kmc/lammps_only_neb/scripts/`
  - `README.md`
  - `README_LAMMPS_ONLY.md`
- Decide whether cluster-specific Hyperion/SLURM examples should remain, be generalized, or be removed.

## Suggested Final Push Commands

Run these only after the checklist above is resolved:

```bash
cd public-devanathan-kmc-base
git init
git add -A
git commit -m "Initial public release"
git branch -M main
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```
