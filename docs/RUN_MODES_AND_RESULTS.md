# Run Modes And Latest Results

## NEB Engine Modes

The consolidated base supports three production engine choices:

- `NEB_ENGINE=ase_lammps`: legacy ASE NEB with LAMMPS energy/force calls.
- `NEB_ENGINE=lammps_only`: native LAMMPS NEB with grouped MPI workers and compact scratch cleanup.
- `NEB_ENGINE=lammps_only_endpoint_opt`: native LAMMPS NEB plus native LAMMPS endpoint minimization before NEB.

Use `NEB_ENGINE_MODE` with `scripts/run_external_pulse50_barrier_diagnostic.sh`:

- `ase_neb`
- `lammps_only`
- `lammps_only_endpoint_opt`

## Devanathan Source Modes

- `constant`: deterministic source refill to a target concentration, with right sink deletion.
- `pulse`: finite initial source pulse, no refill, optional stop when all H are removed.
- `gcmc`: source region initialized and maintained by fixed chemical-potential GCMC.
- `gcmc_bulk`: bulk-region or external-map region controlled by GCMC target concentration or target H count.

GCMC production maintenance can run as:

- `insert_only`: only refill below setpoint.
- `insert_delete`: insert below setpoint and delete above setpoint.
- `balanced`: allow both directions as configured by the GCMC boundary.

## Cache And Performance Modes

- Migration-barrier cache keys are namespaced by `CACHE_SCHEMA`. Treat schema strings as part of the scientific setup.
- Use separate schemas for ASE vs LAMMPS-only, shell radii, endpoint optimization, source mode, map, and hop-direction filters.
- `BARRIER_MERGE_MODE=global` is the normal native LAMMPS-only grouped path.
- `BARRIER_MERGE_MODE=local` is useful for isolated ASE comparisons.
- `KMC_INCREMENTAL_EVENTS=1` enables impacted-region event rebuilds and Fenwick weighted selection. This is the long-run production path for GCMC/Fenwick campaigns.
- GCMC insertion energies use a separate persistent cache through `DEVANATHAN_GCMC_ENERGY_CACHE_FILE`.

## Latest Pulse50 Engine Comparison

Source run:

```text
practice-version/devanathan-kmc-base/runs/external_pulse50_barrier_pair_20260706_143812
```

The run initializes exactly 50 H in the Sigma5 charging region, restricts hops to positive x, deletes H at the right sink, and compares candidate and selected barriers across engines. The central GB window inferred from labels is `116.548335` to `138.900071 A`, centered at `127.724203 A`.

### LAMMPS-only 8/10 A Shell vs ASE 5/6 A Shell

- LAMMPS-only finite candidate rows: `1,293,960` over `13,310` steps; mean `0.122 eV`, median `0.0475 eV`, p90 `0.3297 eV`.
- ASE-NEB finite candidate rows: `1,107,865` over `11,303` steps; mean `0.1659 eV`, median `0.05193 eV`, p90 `0.5164 eV`.
- Exact matched hops: `12,741`; central-GB exact matched hops: `5,366`.
- Overall exact ASE-LAMMPS mean delta: `0.04389 eV`; mean absolute delta: `0.05289 eV`.
- Central-GB exact mean delta: `0.06565 eV`; mean absolute delta: `0.08015 eV`.
- Same selected hop at same step: `7/11,303`.

This comparison includes a shell-size difference, so it should not be interpreted as a pure engine-only delta.

### LAMMPS-only 5/6 A Shell vs ASE 5/6 A Shell

- LAMMPS-only finite candidate rows: `1,078,315` over `11,481` steps; mean `0.1196 eV`, median `0.04496 eV`, p90 `0.3223 eV`.
- Exact matched hops: `10,793`; central-GB exact matched hops: `2,978`.
- Overall exact ASE-LAMMPS mean delta: `-0.01298 eV`; mean absolute delta: `0.03477 eV`.
- Central-GB exact mean delta: `-0.02091 eV`; mean absolute delta: `0.02209 eV`.
- Pair-median central-GB mean delta: `-0.03003 eV`; mean absolute delta: `0.06884 eV`.
- Same selected hop at same step: `4/11,303`.

This is the cleaner same-shell comparison. Trajectories still diverge quickly, so site-pair medians are more useful than same-step selected events.

### LAMMPS-only Endpoint-Optimized 5/6 A Shell vs ASE 5/6 A Shell

- LAMMPS-only endpoint-optimized finite candidate rows: `1,063,815` over `9,988` steps; mean `0.0912 eV`, median `0.04496 eV`, p90 `0.1629 eV`.
- Exact matched hops: `8,541`; central-GB exact matched hops: `1,366`.
- Overall exact ASE-LAMMPS mean delta: `-0.009227 eV`; mean absolute delta: `0.04408 eV`.
- Central-GB exact mean delta: `-0.09032 eV`; mean absolute delta: `0.102 eV`.
- Pair-median central-GB mean delta: `0.00329 eV`; mean absolute delta: `0.08234 eV`.
- Same selected hop at same step: `6/9,988`.

Endpoint optimization narrows the LAMMPS-only candidate distribution but can alter central-GB same-step comparisons strongly. Use the endpoint-optimized engine deliberately and keep its cache schema separate.

## Practical Interpretation

- The native LAMMPS-only path is now the default production engine because it avoids ASE overhead and supports grouped MPI scheduling.
- ASE-NEB remains valuable as a reference/validation engine.
- Same-step selected trajectories are not a stable comparison metric after early divergence.
- Same-hop exact matches and site-pair medians are better for engine calibration.
- Shell size and endpoint optimization are scientific settings, not implementation details; encode them in `CACHE_SCHEMA`.
