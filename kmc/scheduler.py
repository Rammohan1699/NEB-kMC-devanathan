"""
MPI job scheduling utilities for NEB barrier calculations.

This module contains pure job-list operations plus thin MPI collective wrappers.
It is designed so the KMC loop can delegate miss-job deduplication, assignment,
and result merging without embedding collective details everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from collections import defaultdict
import pickle
import numpy as np


JobKey = Any
JobPayload = Tuple[int, int, int]  # src_rank, h_site, n_site
Job = Tuple[JobKey, JobPayload]
Result = Tuple[JobKey, int, float]  # key, src_rank, barrier_eV


@dataclass(frozen=True)
class SchedulerConfig:
    """Controls job assignment and result sharing."""
    assignment_mode: str = "cache_dedupe"  # all_jobs | cache_no_dedupe | cache_dedupe | cache_only
    merge_mode: str = "global"            # global | local | mixed
    min_batch: int = 64

    def __post_init__(self):
        object.__setattr__(self, "assignment_mode", str(self.assignment_mode).lower())
        object.__setattr__(self, "merge_mode", str(self.merge_mode).lower())

    @property
    def use_cache_for_jobs(self) -> bool:
        return self.assignment_mode != "all_jobs"

    @property
    def dedupe_jobs_enabled(self) -> bool:
        return self.assignment_mode == "cache_dedupe"


def dedupe_jobs(job_list: Sequence[Job]) -> List[Job]:
    """Keep first occurrence of each key."""
    seen = set()
    out: List[Job] = []
    for key, payload in job_list:
        if key not in seen:
            seen.add(key)
            out.append((key, payload))
    return out


def assign_jobs_round_robin(all_jobs: Sequence[Job], n_ranks: int) -> Dict[int, List[Job]]:
    """Return rank -> jobs using round-robin assignment."""
    assignments: Dict[int, List[Job]] = {r: [] for r in range(int(n_ranks))}
    for i, job in enumerate(all_jobs):
        assignments[i % int(n_ranks)].append(job)
    return assignments


def chunk_jobs(jobs: Sequence[Job], min_batch: int) -> List[List[Job]]:
    """
    Chunk jobs for local execution.

    Legacy behavior:
      - if total jobs <= NEB_MIN_BATCH, use a single chunk.
      - otherwise split into chunks of NEB_MIN_BATCH.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    min_batch = max(1, int(min_batch))
    if len(jobs) <= min_batch:
        return [jobs]
    return [jobs[i:i + min_batch] for i in range(0, len(jobs), min_batch)]


def flatten_gathered_jobs(gathered: Sequence[Sequence[Job]]) -> List[Job]:
    """Flatten MPI-gathered job lists."""
    return [job for part in gathered for job in part]


def build_key_request_ranks(jobs: Sequence[Job]) -> Dict[JobKey, set[int]]:
    """Map each key to all source ranks that requested it."""
    key_request_ranks: Dict[JobKey, set[int]] = {}
    for key, (src_rank, _h, _n) in jobs:
        key_request_ranks.setdefault(key, set()).add(int(src_rank))
    return key_request_ranks


def prepare_scatter_payload(
    gathered_jobs: Sequence[Sequence[Job]],
    *,
    n_ranks: int,
    dedupe: bool = True,
) -> tuple[List[List[Job]], Dict[JobKey, set[int]], List[Job]]:
    """
    Prepare per-rank scatter payload from gathered miss jobs.

    Returns:
      scatter_payload, key_request_ranks, global_jobs
    """
    flat_jobs = flatten_gathered_jobs(gathered_jobs)
    key_request_ranks = build_key_request_ranks(flat_jobs)
    global_jobs = dedupe_jobs(flat_jobs) if dedupe else list(flat_jobs)
    assignments = assign_jobs_round_robin(global_jobs, n_ranks)
    scatter_payload = [assignments[r] for r in range(int(n_ranks))]
    return scatter_payload, key_request_ranks, global_jobs


def merge_global_results(all_results: Sequence[Sequence[Result]]) -> Dict[JobKey, float]:
    """Merge allgathered results: first result for a key wins."""
    merged: Dict[JobKey, float] = {}
    for part in all_results:
        for key, _src_rank, value in part:
            if key not in merged:
                merged[key] = float(value)
    return merged


def split_results_by_source_rank(
    gathered_results: Sequence[Sequence[Result]],
    *,
    n_ranks: int,
    key_request_ranks: Optional[Mapping[JobKey, set[int]]] = None,
) -> tuple[List[Dict[JobKey, float]], set[JobKey]]:
    """
    Convert gathered results into per-source-rank dictionaries.

    Used by local/mixed merge modes where each requesting rank only receives
    the barriers it needs.
    """
    results_by_rank: Dict[int, Dict[JobKey, float]] = {r: {} for r in range(int(n_ranks))}
    failed_keys: set[JobKey] = set()

    for part in gathered_results:
        for key, src_rank, value in part:
            value = float(value)
            if not np.isfinite(value):
                failed_keys.add(key)

            if key_request_ranks is not None:
                target_ranks = key_request_ranks.get(key, {int(src_rank)})
            else:
                target_ranks = {int(src_rank)}

            for target_rank in target_ranks:
                bucket = results_by_rank.setdefault(int(target_rank), {})
                if key not in bucket:
                    bucket[key] = value

    scatter_payload = [results_by_rank[r] for r in range(int(n_ranks))]
    return scatter_payload, failed_keys


def finite_result_mapping(results: Mapping[JobKey, float]) -> Dict[JobKey, float]:
    """Return only finite barrier results."""
    return {k: float(v) for k, v in results.items() if np.isfinite(v)}


class MPIJobScheduler:
    """
    Small wrapper for MPI collectives used in NEB job scheduling.

    `ctx` is expected to provide:
      - comm
      - rank
      - size
      - phase_guard(step, name)
    This matches the generated mpi_context.py shape.
    """

    def __init__(
        self,
        ctx,
        cfg: SchedulerConfig,
        *,
        logger: Optional[Callable[[str], None]] = None,
        debug: bool = False,
    ) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.logger = logger
        self.debug = bool(debug)

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger(msg)

    def gather_and_scatter_miss_jobs(
        self,
        local_miss_jobs: Sequence[Job],
        *,
        step: int,
    ) -> tuple[List[Job], Optional[Dict[JobKey, set[int]]]]:
        """
        Gather miss jobs to root, optionally dedupe, assign round-robin, scatter.

        Returns:
          my_jobs, key_request_ranks

        On non-root ranks, key_request_ranks is None unless caller broadcasts it.
        For global merge mode it is usually unnecessary.
        """
        comm = self.ctx.comm
        rank = self.ctx.rank
        size = self.ctx.size

        if self.debug:
            self._log(f"[Rank {rank}] Miss-job gather: count={len(local_miss_jobs)}")

        self.ctx.phase_guard(step, "MISS_GATHER")
        gathered = comm.gather(list(local_miss_jobs), root=0)

        key_request_ranks = None
        if rank == 0:
            scatter_payload, key_request_ranks, global_jobs = prepare_scatter_payload(
                gathered,
                n_ranks=size,
                dedupe=self.cfg.dedupe_jobs_enabled,
            )
            if self.debug:
                sizes = [len(x) for x in scatter_payload]
                self._log(
                    f"[Rank 0] Miss-job scatter sizes={sizes}, "
                    f"global_jobs={len(global_jobs)}"
                )
        else:
            scatter_payload = None

        # Broadcast the request map as well. Root is the only rank that needs it
        # for the current local scatter, but making it available everywhere keeps
        # mixed/local modes deterministic and easier to diagnose.
        self.ctx.phase_guard(step, "MISS_REQUEST_MAP_BCAST")
        key_request_ranks = comm.bcast(key_request_ranks, root=0)

        self.ctx.phase_guard(step, "MISS_SCATTER")
        my_jobs = comm.scatter(scatter_payload, root=0)

        if self.debug:
            self._log(f"[Rank {rank}] Miss-job scatter: received {len(my_jobs)} job(s)")

        return my_jobs, key_request_ranks

    def collect_results_global(
        self,
        results_local: Sequence[Result],
        *,
        step: int,
    ) -> Dict[JobKey, float]:
        """Allgather results and make every rank receive the merged mapping."""
        comm = self.ctx.comm
        rank = self.ctx.rank

        if self.debug:
            try:
                nbytes = len(pickle.dumps(results_local, protocol=4))
            except Exception:
                nbytes = -1
            self._log(
                f"[Rank {rank}] Post-NEB allgather: "
                f"{len(results_local)} result(s), pickle_bytes={nbytes}"
            )

        self.ctx.phase_guard(step, "POSTNEB_ALLGATHER_RESULTS")
        gathered = comm.allgather(list(results_local))
        merged = merge_global_results(gathered)

        if self.debug:
            counts = [len(x) for x in gathered]
            self._log(
                f"[Rank {rank}] Post-NEB allgather complete: "
                f"recv_counts={counts}, merged={len(merged)}"
            )

        return merged

    def collect_results_local(
        self,
        results_local: Sequence[Result],
        *,
        step: int,
        key_request_ranks: Optional[Mapping[JobKey, set[int]]] = None,
    ) -> tuple[Dict[JobKey, float], set[JobKey]]:
        """
        Gather results to root and scatter only requested keys back to source ranks.
        """
        comm = self.ctx.comm
        rank = self.ctx.rank
        size = self.ctx.size

        self.ctx.phase_guard(step, "POSTNEB_GATHER_RESULTS_LOCAL")
        gathered = comm.gather(list(results_local), root=0)

        if rank == 0:
            scatter_payload, failed_keys = split_results_by_source_rank(
                gathered,
                n_ranks=size,
                key_request_ranks=key_request_ranks,
            )
            if self.debug:
                sizes = [len(x) for x in scatter_payload]
                self._log(f"[Rank 0] Post-NEB scatter local sizes={sizes}")
        else:
            scatter_payload = None
            failed_keys = None

        self.ctx.phase_guard(step, "POSTNEB_SCATTER_RESULTS_LOCAL")
        fold_results = comm.scatter(scatter_payload, root=0)
        failed_keys = comm.bcast(failed_keys, root=0)

        return fold_results, failed_keys
