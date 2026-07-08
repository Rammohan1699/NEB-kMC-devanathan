"""MPI runtime helpers for the KMC/NEB simulation.

This module intentionally keeps MPI-specific global state in one place.
It provides a small wrapper around ``mpi4py.MPI.COMM_WORLD`` plus helpers for
collective ordering diagnostics.  The goal is to replace scattered uses of
``comm``, ``rank``, ``size``, ``MPI.COMM_SELF``, and ``phase_guard`` in the
simulation stack with a single explicit context object.

The functions here are conservative: they do not change collective behavior;
they only centralize it and add optional breadcrumbs for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import os
import time
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar

try:  # pragma: no cover - depends on MPI runtime availability
    from mpi4py import MPI  # type: ignore
except Exception as exc:  # pragma: no cover
    MPI = None  # type: ignore[assignment]
    _MPI_IMPORT_ERROR = exc
else:
    _MPI_IMPORT_ERROR = None

T = TypeVar("T")
Logger = Callable[[str], None]


def _noop_logger(_msg: str) -> None:
    return None


def _heartbeat_enabled() -> bool:
    raw = os.environ.get("KMC_HEARTBEAT", os.environ.get("KMC_PHASE_HEARTBEAT", "0"))
    return str(raw).lower() not in {"0", "false", "no", "off", ""}


def _write_phase_heartbeat(rank: int, step: int, name: str, point: str) -> None:
    if not _heartbeat_enabled():
        return
    try:
        log_dir = os.environ.get("KMC_LOG_DIR", os.environ.get("LOG_DIR", "logs")) or "logs"
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"heartbeat_rank{int(rank)}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"time_epoch={time.time():.6f}\n")
            fh.write(f"rank={int(rank)}\n")
            fh.write(f"step={int(step)}\n")
            fh.write(f"phase={str(name)}\n")
            fh.write(f"point={str(point)}\n")
    except Exception:
        pass


@dataclass
class MPIContext:
    """Small explicit wrapper for MPI communicator state.

    Parameters
    ----------
    comm:
        MPI communicator. Defaults to ``MPI.COMM_WORLD``.
    logger:
        Optional callable accepting a single message string.
    debug:
        If true, ``phase_guard`` emits log messages on rank 0.
    set_ase_rank_env:
        If true, sets ``ASE_RANK`` to the MPI rank, matching the legacy code.
    """

    comm: Any = None
    logger: Logger = _noop_logger
    debug: bool = False
    set_ase_rank_env: bool = True
    _phase_counter: Any = field(default_factory=count, init=False, repr=False)

    def __post_init__(self) -> None:
        if MPI is None:
            raise RuntimeError(
                "mpi4py could not be imported. Install mpi4py or run in an MPI-enabled environment."
            ) from _MPI_IMPORT_ERROR
        if self.comm is None:
            self.comm = MPI.COMM_WORLD
        self.rank = int(self.comm.Get_rank())
        self.size = int(self.comm.Get_size())
        if self.set_ase_rank_env:
            os.environ["ASE_RANK"] = str(self.rank)

    @classmethod
    def create(
        cls,
        *,
        comm: Any = None,
        logger: Logger = _noop_logger,
        debug: bool = False,
        set_ase_rank_env: bool = True,
    ) -> "MPIContext":
        """Create the default MPI context.

        This method matches the API expected by kmc_driver.py while preserving
        the existing create_mpi_context() helper.
        """
        return cls(
            comm=comm,
            logger=logger,
            debug=debug,
            set_ase_rank_env=set_ase_rank_env,
        )

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    @property
    def comm_self(self) -> Any:
        """Return ``MPI.COMM_SELF`` for per-rank calculators."""

        if MPI is None:  # defensive; __post_init__ normally prevents this
            raise RuntimeError("mpi4py is unavailable")
        return MPI.COMM_SELF

    def barrier(self) -> None:
        self.comm.Barrier()

    def bcast(self, value: Optional[T], root: int = 0) -> T:
        return self.comm.bcast(value, root=root)

    def gather(self, value: T, root: int = 0) -> Optional[list[T]]:
        return self.comm.gather(value, root=root)

    def scatter(self, values: Optional[Sequence[T]], root: int = 0) -> T:
        return self.comm.scatter(values, root=root)

    def allgather(self, value: T) -> list[T]:
        return self.comm.allgather(value)

    def reduce_sum(self, value: Any, root: int = 0) -> Any:
        return self.comm.reduce(value, op=MPI.SUM, root=root)

    def allreduce_sum(self, value: Any) -> Any:
        return self.comm.allreduce(value, op=MPI.SUM)

    def abort(self, code: int = 1) -> None:
        self.comm.Abort(code)

    def root_print(self, message: str) -> None:
        if self.is_root:
            print(message)

    def log(self, message: str, *, root_only: bool = False) -> None:
        if root_only and not self.is_root:
            return
        self.logger(message)

    def phase_guard(self, step: int, name: str) -> int:
        """Guard collective ordering across ranks.

        This mirrors the legacy ``phase_guard`` behavior: every rank enters a
        barrier, and rank 0 advances a monotonic sequence.  It is intentionally
        simple because it is used around MPI collectives to detect crossed or
        missing collective calls during debugging.

        Returns
        -------
        int
            The local phase sequence value.  Only rank 0 increments it; other
            ranks receive the broadcast value for consistent breadcrumbs.
        """

        if self.is_root:
            seq = next(self._phase_counter) + 1
        else:
            seq = None
        _write_phase_heartbeat(self.rank, step, name, "before_barrier")
        self.barrier()
        _write_phase_heartbeat(self.rank, step, name, "after_barrier_before_bcast")
        seq = self.bcast(seq, root=0)
        _write_phase_heartbeat(self.rank, step, name, "after_bcast")
        if self.is_root and self.debug:
            self.logger(f"[Rank 0] phase_guard seq={seq}, step={int(step)}, name={str(name)}")
        return int(seq)

    def split_evenly(self, items: Sequence[T]) -> list[T]:
        """Return this rank's strided share of a sequence.

        Equivalent to the legacy pattern ``items[rank::size]``.
        """

        return list(items[self.rank :: self.size])

    def assignment_counts(self, assignments: Sequence[Sequence[Any]]) -> list[int]:
        """Return assignment sizes, useful for rank-0 diagnostics."""

        return [len(chunk) for chunk in assignments]


# ---------------------------------------------------------------------------
# Backwards-compatible functional helpers
# ---------------------------------------------------------------------------

def create_mpi_context(
    *,
    logger: Logger = _noop_logger,
    debug: bool = False,
    set_ase_rank_env: bool = True,
) -> MPIContext:
    """Create the default world MPI context."""

    return MPIContext(logger=logger, debug=debug, set_ase_rank_env=set_ase_rank_env)


def phase_guard(
    comm: Any,
    rank: int,
    step: int,
    name: str,
    *,
    logger: Logger = _noop_logger,
    debug: bool = False,
    counter: Optional[Iterable[int]] = None,
) -> int:
    """Standalone compatibility version of the legacy phase guard.

    Prefer ``MPIContext.phase_guard`` in new code. This helper exists so the
    legacy driver can migrate one call site at a time.
    """

    if counter is None:
        # Local fallback. For persistent sequencing, use MPIContext instead.
        seq_value = 1 if rank == 0 else None
    else:
        seq_value = next(counter) if rank == 0 else None
    _write_phase_heartbeat(rank, step, name, "before_barrier")
    comm.Barrier()
    _write_phase_heartbeat(rank, step, name, "after_barrier_before_bcast")
    seq_value = comm.bcast(seq_value, root=0)
    _write_phase_heartbeat(rank, step, name, "after_bcast")
    if rank == 0 and debug:
        logger(f"[Rank 0] phase_guard seq={seq_value}, step={int(step)}, name={str(name)}")
    return int(seq_value)


__all__ = [
    "MPI",
    "MPIContext",
    "create_mpi_context",
    "phase_guard",
]
