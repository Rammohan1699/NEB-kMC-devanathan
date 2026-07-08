"""Rank-aware logging and timing helpers for KMC/NEB runs.

The simulation writes plain text rank logs plus structured timing CSV files.
These helpers keep that I/O consistent across the driver and service modules:
each rank can write ``log_rank{rank}.txt``, rank 0 can mirror messages to
stdout, and timed phases can be appended with stable bucket names for later
post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import os
import time
from typing import Any, Iterable, Mapping, Optional


TIMING_BUCKET_FIELDS = [
    "build_struct",
    "build_H_tree",
    "move_loop",
    "batch_extract_env",
    "cache_lookup",
    "barrier_neb",
    "neb_io_write",
    "neb_lammps_run",
    "neb_io_parse",
    "neb_io_cleanup",
    "selection_comm",
    "selection_cpu",
    "logging",
    "extract_env",
    "validation",
    "job_comm",
    "job_schedule",
    "neb_struct",
    "result_comm",
    "result_merge",
    "move_reduce",
    "cache_update",
    "cache_save",
    "rate_comm",
    "rate_io",
    "state_comm",
    "state_update",
    "dump_diag",
    "dump_traj",
    "timing_append",
]

NON_ADDITIVE_TIMING_BUCKET_FIELDS = {
    "neb_io_write",
    "neb_lammps_run",
    "neb_io_parse",
    "neb_io_cleanup",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class RankLogger:
    """Rank-aware plain-text logger.

    Parameters mirror the legacy ``log`` behavior: write to a rank-specific
    file when debug logging is enabled or on rank 0, and print rank-0 messages
    to stdout.
    """

    rank: int
    debug: bool = True
    log_dir: str = "."
    filename_template: str = "log_rank{rank}.txt"
    mirror_root_to_stdout: bool = True
    reset_on_start: bool = False
    _path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ensure_dir(self.log_dir)
        self._path = os.path.join(self.log_dir, self.filename_template.format(rank=self.rank))
        if self.reset_on_start:
            with open(self._path, "w", encoding="utf-8") as fh:
                fh.write(f"[Rank {self.rank}] Starting new KMC-NEB simulation log\n")

    @property
    def path(self) -> str:
        return self._path

    def __call__(self, msg: str, step: Optional[int] = None, banner: bool = False) -> None:
        self.log(msg, step=step, banner=banner)

    def log(self, msg: str, step: Optional[int] = None, banner: bool = False) -> None:
        if self.debug or self.rank == 0:
            with open(self._path, "a", encoding="utf-8") as fh:
                if banner and step is not None:
                    fh.write(f"\n========== STEP {step} ==========\n")
                elif banner:
                    fh.write("\n" + "=" * 40 + "\n")
                fh.write(str(msg) + "\n")
        if self.rank == 0 and self.mirror_root_to_stdout:
            print(msg)


@dataclass
class StepTimer:
    """Accumulate named timing buckets for one KMC step."""

    fields: Iterable[str] = field(default_factory=lambda: TIMING_BUCKET_FIELDS)

    def __post_init__(self) -> None:
        self.values = {name: 0.0 for name in self.fields}

    def add(self, name: str, seconds: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(seconds)

    def set(self, name: str, seconds: float) -> None:
        self.values[name] = float(seconds)

    def snapshot(self) -> dict[str, float]:
        return dict(self.values)

    def total_components(self) -> float:
        return float(sum(self.values.values()))


class TimingCSVWriter:
    """Append per-step timing rows with the legacy column order."""

    def __init__(
        self,
        path: str,
        fields: Iterable[str] = TIMING_BUCKET_FIELDS,
        enabled: bool = True,
        append_existing: bool = False,
    ):
        self.path = path
        self.fields = list(fields)
        self.enabled = enabled
        if self.enabled and not (
            append_existing
            and os.path.exists(self.path)
            and os.path.getsize(self.path) > 0
        ):
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["step", "wall_total", *self.fields, "components_sum", "unaccounted"])

    def append(self, step: int, wall_total: float, values: Mapping[str, float]) -> None:
        if not self.enabled:
            return
        row_values = [float(values.get(name, 0.0)) for name in self.fields]
        components_sum = float(
            sum(
                float(values.get(name, 0.0))
                for name in self.fields
                if name not in NON_ADDITIVE_TIMING_BUCKET_FIELDS
            )
        )
        unaccounted = float(wall_total) - components_sum
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([int(step), float(wall_total), *row_values, components_sum, unaccounted])


def log_step_timing(
    logger: RankLogger,
    step: int,
    times_dict: dict[str, float],
    total_wall: Optional[float] = None,
) -> None:
    """Write the human-readable timing table used by the legacy code."""

    order = TIMING_BUCKET_FIELDS
    colw = 14
    for key in order:
        times_dict.setdefault(key, 0.0)

    total = float(total_wall) if total_wall is not None else sum(times_dict[key] for key in order)
    header = "Stage:       " + " ".join(f"{key:<{colw}}" for key in order)
    times = "Time (s):    " + " ".join(f"{times_dict[key]:<{colw}.2f}" for key in order)
    perc = "Percent:     " + " ".join(
        f"{(times_dict[key] / total * 100 if total > 0 else 0):<{colw}.1f}%" for key in order
    )

    logger.log(f"========== STEP {step} TIMING ==========" , banner=True)
    logger.log(f"Total: {total:.2f} s")
    logger.log(header)
    logger.log(times)
    logger.log(perc)

    if total_wall is not None:
        other = float(total_wall) - sum(
            times_dict[key]
            for key in order
            if key not in NON_ADDITIVE_TIMING_BUCKET_FIELDS
        )
        if abs(other) > 1e-3:
            logger.log(f"Unaccounted/other: {other:.2f} s")


def log_friendly(value: Any) -> Any:
    """Recursively convert numpy-like scalars/arrays for cleaner logging."""

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: log_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(log_friendly(val) for val in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(log_friendly(val) for val in value)
    return value


__all__ = [
    "TIMING_BUCKET_FIELDS",
    "RankLogger",
    "StepTimer",
    "TimingCSVWriter",
    "ensure_dir",
    "log_friendly",
    "log_step_timing",
]
