"""Mean-squared displacement tracking for KMC runs."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np


class MSDTracker:
    """Record MSD vs KMC time and optionally emit a plot."""

    def __init__(
        self,
        *,
        initial_indices: Sequence[int],
        box: Sequence[float],
        initial_positions: Optional[np.ndarray] = None,
        use_unwrapped_positions: bool = False,
        log_interval: int = 100,
        log_file: str = "msd_vs_time.csv",
        plot_file: str = "msd_plot.png",
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            return

        self.log_interval = max(1, int(log_interval))
        self.log_file = str(log_file)
        self.plot_file = str(plot_file)
        self.box = np.asarray(box, dtype=float)
        self.initial_indices = np.asarray(initial_indices, dtype=int)
        self.initial_positions = (
            np.asarray(initial_positions, dtype=float).copy()
            if initial_positions is not None
            else None
        )
        self.use_unwrapped_positions = bool(use_unwrapped_positions)
        self.records: list[tuple[int, float, float]] = []
        self._last_step: Optional[int] = None
        self._prepare_log()

    def _prepare_log(self) -> None:
        dirname = os.path.dirname(self.log_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as fh:
            fh.write("step,time_s,msd_A2\n")

    def _compute_msd(
        self,
        *,
        current_positions: Optional[np.ndarray] = None,
        current_indices: Optional[Iterable[int]] = None,
        sites: Optional[np.ndarray] = None,
    ) -> float:
        if current_positions is not None:
            curr = np.asarray(current_positions, dtype=float)
        else:
            if current_indices is None or sites is None:
                raise ValueError("current_positions or (current_indices and sites) must be provided")
            curr = np.asarray(sites, dtype=float)[np.asarray(current_indices, dtype=int)]

        if self.initial_positions is not None:
            init = self.initial_positions
        else:
            if sites is None:
                raise ValueError("sites is required when initial_positions is not set")
            init = np.asarray(sites, dtype=float)[self.initial_indices]

        if curr.shape != init.shape:
            raise ValueError("Current and initial position arrays must have the same shape")

        delta = curr - init
        if not self.use_unwrapped_positions:
            delta -= self.box * np.round(delta / self.box)
        sq = np.einsum("ij,ij->i", delta, delta)
        return float(np.mean(sq)) if sq.size else 0.0

    def record(
        self,
        step: int,
        sim_time: float,
        *,
        current_positions: Optional[np.ndarray] = None,
        current_indices: Optional[Iterable[int]] = None,
        sites: Optional[np.ndarray] = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        step = int(step)
        if not force and (step % self.log_interval != 0):
            return
        if self._last_step == step:
            return

        msd = self._compute_msd(
            current_positions=current_positions,
            current_indices=current_indices,
            sites=sites,
        )
        self.records.append((step, float(sim_time), msd))
        self._last_step = step

        with open(self.log_file, "a", encoding="utf-8") as fh:
            fh.write(f"{step},{float(sim_time):.6e},{msd:.6f}\n")

    def plot(self) -> bool:
        if not self.enabled or not self.records:
            return False
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("[MSDTracker] matplotlib unavailable; skipping MSD plot.")
            return False

        _steps, times, msds = zip(*self.records)
        fig, ax = plt.subplots()
        ax.plot(times, msds, marker="o", linestyle="-")
        ax.set_xlabel("KMC time (s)")
        ax.set_ylabel("MSD (A^2)")
        ax.set_title("Mean Squared Displacement vs KMC time")
        ax.grid(True)
        fig.tight_layout()

        dirname = os.path.dirname(self.plot_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(self.plot_file)
        plt.close(fig)
        print(f"[MSDTracker] Saved MSD plot to {self.plot_file}")
        return True
