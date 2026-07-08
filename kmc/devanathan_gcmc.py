from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import random
import time
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from .devanathan import DevanathanBoundaryConfig, DevanathanUpdate


MoveKind = Literal["insert", "delete"]
EnergyDifference = Callable[[np.ndarray, int, MoveKind], float]
InitializationProgress = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class GCMCReservoirConfig:
    chemical_potential_ev: float
    temperature_k: float = 300.0
    boltzmann_ev_per_k: float = 8.617333262145e-5
    attempts_per_adjustment: int = 1
    initialization_start: str = "empty"
    maintenance_setpoint_mode: str = "mean"
    production_maintenance_mode: str = "insert_only"
    initialization_min_attempts: int = 15000
    initialization_max_attempts: int = 50000
    initialization_progress_interval: int = 100
    convergence_window_attempts: int = 7500
    convergence_check_interval: int = 500
    convergence_drift_tolerance_h: float = 2.0
    target_fraction_tolerance: float = 0.005
    convergence_required_checks: int = 2
    accept_near_target_at_max: bool = False

    def normalized(self) -> "GCMCReservoirConfig":
        production_mode = self.production_maintenance_mode.strip().lower().replace("-", "_")
        if self.temperature_k <= 0.0:
            raise ValueError("DEVANATHAN_GCMC_TEMPERATURE_K must be positive")
        if self.boltzmann_ev_per_k <= 0.0:
            raise ValueError("DEVANATHAN_GCMC_KB_EV_PER_K must be positive")
        if self.attempts_per_adjustment < 1:
            raise ValueError("DEVANATHAN_GCMC_ATTEMPTS_PER_ADJUSTMENT must be at least 1")
        if self.initialization_start not in {"empty", "target"}:
            raise ValueError(
                "DEVANATHAN_GCMC_INITIALIZATION_START must be 'empty' or 'target'"
            )
        if self.maintenance_setpoint_mode not in {"mean", "target"}:
            raise ValueError(
                "DEVANATHAN_GCMC_MAINTENANCE_SETPOINT_MODE must be 'mean' or 'target'"
            )
        if production_mode not in {
            "insert_only",
            "insert_delete",
            "balanced",
        }:
            raise ValueError(
                "DEVANATHAN_GCMC_PRODUCTION_MODE must be 'insert_only', "
                "'insert_delete', or 'balanced'"
            )
        if self.initialization_min_attempts < 1:
            raise ValueError("DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS must be at least 1")
        if self.initialization_max_attempts < 1:
            raise ValueError("DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS must be at least 1")
        if self.initialization_max_attempts < self.initialization_min_attempts:
            raise ValueError(
                "DEVANATHAN_GCMC_INITIALIZATION_MAX_ATTEMPTS must be greater than "
                "or equal to DEVANATHAN_GCMC_INITIALIZATION_MIN_ATTEMPTS"
            )
        if self.initialization_progress_interval < 1:
            raise ValueError("DEVANATHAN_GCMC_PROGRESS_INTERVAL must be at least 1")
        if self.convergence_window_attempts < 2:
            raise ValueError("DEVANATHAN_GCMC_CONVERGENCE_WINDOW must be at least 2")
        if self.convergence_check_interval < 1:
            raise ValueError("DEVANATHAN_GCMC_CONVERGENCE_CHECK_INTERVAL must be at least 1")
        if self.convergence_drift_tolerance_h < 0.0:
            raise ValueError("DEVANATHAN_GCMC_CONVERGENCE_DRIFT_H must be non-negative")
        if self.target_fraction_tolerance < 0.0:
            raise ValueError("DEVANATHAN_GCMC_TARGET_FRACTION_TOLERANCE must be non-negative")
        if self.convergence_required_checks < 1:
            raise ValueError("DEVANATHAN_GCMC_CONVERGENCE_REQUIRED_CHECKS must be at least 1")
        if production_mode != self.production_maintenance_mode:
            return replace(self, production_maintenance_mode=production_mode)
        return self

    @property
    def kbt_ev(self) -> float:
        return self.boltzmann_ev_per_k * self.temperature_k


@dataclass(frozen=True)
class GCMCProposal:
    kind: MoveKind
    site: int
    delta_energy_ev: float
    acceptance_probability: float
    random_value: float
    accepted: bool


@dataclass(frozen=True)
class GCMCInitializationSummary:
    attempts: int
    accepted_moves: int
    final_h: int
    mean_h: float
    mean_fraction_h_per_fe: float
    half_drift_h: float
    target_h: int
    target_fraction_h_per_fe: float
    maintenance_setpoint_h: int
    converged: bool


@dataclass(frozen=True)
class GCMCDevanathanUpdate(DevanathanUpdate):
    gcmc_proposals: tuple[GCMCProposal, ...] = ()
    cumulative_gcmc_attempts: int = 0
    cumulative_gcmc_accepts: int = 0
    maintenance_setpoint_h: int = 0


class GCMCDevanathanBoundary:
    """Devanathan controlled region coupled to a chemical-potential reservoir.

    Initialization performs unconstrained site-flip GCMC until the controlled
    region occupancy converges near the requested H/Fe concentration. During
    KMC, the default production mode only applies accepted insertion proposals
    when the region is below the converged setpoint. The optional insert/delete
    mode also applies accepted deletion proposals when the region is above
    setpoint.
    """

    def __init__(
        self,
        config: DevanathanBoundaryConfig,
        reservoir: GCMCReservoirConfig,
        box: np.ndarray,
        energy_difference: EnergyDifference,
        initialization_progress: InitializationProgress | None = None,
        region_label: str = "source",
        control_site_indices: Sequence[int] | np.ndarray | None = None,
        control_fe_count: int | None = None,
        left_sink_site_indices: Sequence[int] | np.ndarray | None = None,
        left_sink_label: str = "left_sink",
    ) -> None:
        self.config = config.normalized(box)
        self.reservoir = reservoir.normalized()
        self.energy_difference = energy_difference
        self.initialization_progress = initialization_progress
        self.region_label = str(region_label or "source")
        self.control_site_indices = (
            None
            if control_site_indices is None
            else tuple(sorted({int(site) for site in control_site_indices}))
        )
        self.control_fe_count = (
            None if control_fe_count is None else int(control_fe_count)
        )
        self.left_sink_site_indices = (
            None
            if left_sink_site_indices is None
            else tuple(sorted({int(site) for site in left_sink_site_indices}))
        )
        self.left_sink_site_set = (
            None if self.left_sink_site_indices is None else set(self.left_sink_site_indices)
        )
        self.left_sink_label = str(left_sink_label or "left_sink")
        self.rng = random.Random(self.config.seed)
        self.cumulative_inserted = 0
        self.cumulative_removed = 0
        self.cumulative_left_removed = 0
        self.cumulative_gcmc_attempts = 0
        self.cumulative_gcmc_accepts = 0
        self.maintenance_setpoint_h: int | None = None
        self.initialization_summary: GCMCInitializationSummary | None = None
        self.initialization_history: tuple[int, ...] = ()

    def source_sites(self, sites: np.ndarray) -> tuple[int, ...]:
        if self.control_site_indices is not None:
            n_sites = len(sites)
            invalid = [
                site
                for site in self.control_site_indices
                if site < 0 or site >= n_sites
            ]
            if invalid:
                raise ValueError(
                    f"GCMC {self.region_label} control contains invalid site indices: "
                    f"{invalid[:5]}"
                )
            return self.control_site_indices
        x = np.asarray(sites, dtype=float)[:, 0]
        mask = (x >= self.config.source_x_min_a) & (x < self.config.source_x_max_a)
        return tuple(int(i) for i in np.flatnonzero(mask))

    def source_fe_count(self, fe_positions: np.ndarray) -> int:
        if self.control_fe_count is not None:
            return max(0, int(self.control_fe_count))
        x = np.asarray(fe_positions, dtype=float)[:, 0]
        mask = (x >= self.config.source_x_min_a) & (x < self.config.source_x_max_a)
        return int(np.count_nonzero(mask))

    def left_sink_sites(self, sites: np.ndarray) -> tuple[int, ...]:
        if self.config.left_sink_x_max_a <= 0.0:
            return ()
        if self.left_sink_site_indices is not None:
            n_sites = len(sites)
            invalid = [
                site
                for site in self.left_sink_site_indices
                if site < 0 or site >= n_sites
            ]
            if invalid:
                raise ValueError(
                    f"GCMC {self.left_sink_label} contains invalid site indices: "
                    f"{invalid[:5]}"
                )
            return self.left_sink_site_indices
        x = np.asarray(sites, dtype=float)[:, 0]
        return tuple(int(i) for i in np.flatnonzero(x < self.config.left_sink_x_max_a))

    def _is_left_sink_site(self, site: int, site_x: float) -> bool:
        if self.config.left_sink_x_max_a <= 0.0:
            return False
        if self.left_sink_site_set is not None:
            return int(site) in self.left_sink_site_set
        return float(site_x) < self.config.left_sink_x_max_a

    def target_source_count(self, sites: np.ndarray, fe_positions: np.ndarray) -> int:
        del sites
        return int(round(self.config.source_target_fraction * self.source_fe_count(fe_positions)))

    def _acceptance_probability(self, kind: MoveKind, delta_energy_ev: float) -> float:
        mu = self.reservoir.chemical_potential_ev
        if kind == "insert":
            exponent = -(delta_energy_ev - mu) / self.reservoir.kbt_ev
        else:
            exponent = -(delta_energy_ev + mu) / self.reservoir.kbt_ev
        return 1.0 if exponent >= 0.0 else math.exp(exponent)

    def _attempt(
        self,
        h_indices: list[int],
        source_set: set[int],
        kind: MoveKind,
        *,
        site: int | None = None,
    ) -> GCMCProposal | None:
        occupied = set(h_indices)
        if site is None:
            if kind == "insert":
                candidates = sorted(source_set - occupied)
            else:
                candidates = sorted(source_set & occupied)
            if not candidates:
                return None
            site = int(self.rng.choice(candidates))
        else:
            site = int(site)
            if site not in source_set:
                raise ValueError(
                    f"GCMC proposal site {site} is outside the {self.region_label} region"
                )
            if kind == "insert" and site in occupied:
                raise ValueError(f"Cannot insert at occupied {self.region_label} site {site}")
            if kind == "delete" and site not in occupied:
                raise ValueError(f"Cannot delete vacant {self.region_label} site {site}")
        current = np.asarray(h_indices, dtype=int)
        delta_energy = float(self.energy_difference(current, site, kind))
        probability = self._acceptance_probability(kind, delta_energy)
        draw = self.rng.random()
        accepted = draw < probability
        self.cumulative_gcmc_attempts += 1
        if accepted:
            self.cumulative_gcmc_accepts += 1
            if kind == "insert":
                h_indices.append(site)
            else:
                h_indices.remove(site)
        return GCMCProposal(
            kind=kind,
            site=site,
            delta_energy_ev=delta_energy,
            acceptance_probability=probability,
            random_value=draw,
            accepted=accepted,
        )

    @staticmethod
    def _window_statistics(history: list[int], window: int) -> tuple[float, float]:
        sample = np.asarray(history[-window:], dtype=float)
        midpoint = max(1, len(sample) // 2)
        first = sample[:midpoint]
        second = sample[midpoint:]
        if second.size == 0:
            second = first
        return float(sample.mean()), float(second.mean() - first.mean())

    def initialize_source(self, sites: np.ndarray, fe_positions: np.ndarray) -> np.ndarray:
        source_set = set(self.source_sites(sites))
        if not source_set:
            raise RuntimeError(
                f"GCMC {self.region_label} initialization has no {self.region_label}-region sites"
            )
        source_sites = tuple(sorted(source_set))
        source_fe_atoms = self.source_fe_count(fe_positions)
        if source_fe_atoms <= 0:
            raise RuntimeError(
                f"GCMC {self.region_label} initialization has no {self.region_label}-region Fe atoms"
            )
        target = self.target_source_count(sites, fe_positions)
        if self.reservoir.initialization_start == "target":
            if target > len(source_sites):
                raise RuntimeError(
                    f"GCMC target occupancy {target} exceeds {len(source_sites)} source sites"
                )
            current = [
                int(site)
                for site in self.rng.sample(source_sites, target)
            ]
        else:
            current = []
        occupancy_history: list[int] = []
        accepted_moves = 0
        consecutive_converged_checks = 0
        converged = False
        mean_h = 0.0
        half_drift_h = float("inf")
        initialization_started = time.monotonic()

        for attempts in range(1, self.reservoir.initialization_max_attempts + 1):
            site = int(self.rng.choice(source_sites))
            kind: MoveKind = "delete" if site in set(current) else "insert"
            proposal = self._attempt(current, source_set, kind, site=site)
            if proposal is not None and proposal.accepted:
                accepted_moves += 1
            occupancy_history.append(len(current))

            if (
                self.initialization_progress is not None
                and (
                    attempts == 1
                    or attempts % self.reservoir.initialization_progress_interval == 0
                )
            ):
                self.initialization_progress(
                    {
                        "attempts": attempts,
                        "accepted_moves": accepted_moves,
                        "current_h": len(current),
                        "current_fraction_h_per_fe": len(current) / source_fe_atoms,
                        "target_h": target,
                        "target_fraction_h_per_fe": self.config.source_target_fraction,
                        "elapsed_s": time.monotonic() - initialization_started,
                    }
                )

            enough_samples = (
                attempts >= self.reservoir.initialization_min_attempts
                and len(occupancy_history) >= self.reservoir.convergence_window_attempts
            )
            should_check = (
                enough_samples
                and attempts % self.reservoir.convergence_check_interval == 0
            )
            if not should_check:
                continue

            mean_h, half_drift_h = self._window_statistics(
                occupancy_history,
                self.reservoir.convergence_window_attempts,
            )
            mean_fraction = mean_h / source_fe_atoms
            near_target = (
                abs(mean_fraction - self.config.source_target_fraction)
                <= self.reservoir.target_fraction_tolerance
            )
            stable = (
                abs(half_drift_h)
                <= self.reservoir.convergence_drift_tolerance_h
            )
            if near_target and stable:
                consecutive_converged_checks += 1
            else:
                consecutive_converged_checks = 0
            if consecutive_converged_checks >= self.reservoir.convergence_required_checks:
                converged = True
                break

        accepted_near_target_at_max = False
        if not converged:
            if occupancy_history:
                mean_h, half_drift_h = self._window_statistics(
                    occupancy_history,
                    min(len(occupancy_history), self.reservoir.convergence_window_attempts),
                )
            mean_fraction = mean_h / source_fe_atoms
            accepted_near_target_at_max = (
                self.reservoir.accept_near_target_at_max
                and abs(mean_fraction - self.config.source_target_fraction)
                <= self.reservoir.target_fraction_tolerance
            )
            if not accepted_near_target_at_max:
                raise RuntimeError(
                    f"GCMC {self.region_label} initialization did not converge near "
                    "the requested concentration: "
                    f"attempts={len(occupancy_history)} final_H={len(current)} "
                    f"window_mean_H={mean_h:.6g} window_half_drift_H={half_drift_h:.6g} "
                    f"mean_H_per_Fe={mean_fraction:.6g} "
                    f"target_H_per_Fe={self.config.source_target_fraction:.6g} "
                    f"mu_eV={self.reservoir.chemical_potential_ev:g}"
                )

        if self.reservoir.maintenance_setpoint_mode == "target":
            self.maintenance_setpoint_h = max(0, int(target))
        else:
            self.maintenance_setpoint_h = max(0, int(round(mean_h)))
        self.initialization_history = tuple(occupancy_history)
        self.initialization_summary = GCMCInitializationSummary(
            attempts=len(occupancy_history),
            accepted_moves=accepted_moves,
            final_h=len(current),
            mean_h=mean_h,
            mean_fraction_h_per_fe=mean_h / source_fe_atoms,
            half_drift_h=half_drift_h,
            target_h=target,
            target_fraction_h_per_fe=self.config.source_target_fraction,
            maintenance_setpoint_h=self.maintenance_setpoint_h,
            converged=not accepted_near_target_at_max,
        )
        self.cumulative_inserted += len(current)
        return np.asarray(current, dtype=int)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "rng_state": self.rng.getstate(),
            "cumulative_inserted": int(self.cumulative_inserted),
            "cumulative_removed": int(self.cumulative_removed),
            "cumulative_left_removed": int(self.cumulative_left_removed),
            "cumulative_gcmc_attempts": int(self.cumulative_gcmc_attempts),
            "cumulative_gcmc_accepts": int(self.cumulative_gcmc_accepts),
            "maintenance_setpoint_h": self.maintenance_setpoint_h,
            "initialization_summary": (
                None
                if self.initialization_summary is None
                else asdict(self.initialization_summary)
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)
        self.cumulative_inserted = int(state.get("cumulative_inserted", 0))
        self.cumulative_removed = int(state.get("cumulative_removed", 0))
        self.cumulative_left_removed = int(state.get("cumulative_left_removed", 0))
        self.cumulative_gcmc_attempts = int(state.get("cumulative_gcmc_attempts", 0))
        self.cumulative_gcmc_accepts = int(state.get("cumulative_gcmc_accepts", 0))
        raw_setpoint = state.get("maintenance_setpoint_h")
        self.maintenance_setpoint_h = (
            None if raw_setpoint is None else int(raw_setpoint)
        )
        raw_summary = state.get("initialization_summary")
        self.initialization_summary = (
            None
            if not isinstance(raw_summary, Mapping)
            else GCMCInitializationSummary(**dict(raw_summary))
        )

    def apply(
        self,
        *,
        sites: np.ndarray,
        fe_positions: np.ndarray,
        h_indices: np.ndarray,
        h_unwrapped_positions: np.ndarray,
        elapsed_time_s: float,
    ) -> GCMCDevanathanUpdate:
        current_indices = [int(x) for x in np.asarray(h_indices, dtype=int)]
        current_unwrapped = [
            np.asarray(p, dtype=float) for p in np.asarray(h_unwrapped_positions, dtype=float)
        ]
        if len(current_indices) != len(current_unwrapped):
            raise ValueError("h_indices and h_unwrapped_positions length mismatch")

        deleted_left_sink: list[int] = []
        deleted_sink: list[int] = []
        kept_indices: list[int] = []
        kept_unwrapped: list[np.ndarray] = []
        for site, pos in zip(current_indices, current_unwrapped):
            site_x = float(sites[site, 0])
            if self._is_left_sink_site(site, site_x):
                deleted_left_sink.append(site)
            elif site_x >= self.config.sink_x_min_a:
                deleted_sink.append(site)
            else:
                kept_indices.append(site)
                kept_unwrapped.append(pos)
        current_indices = kept_indices
        current_unwrapped = kept_unwrapped

        source_set = set(self.source_sites(sites))
        source_fe_atoms = self.source_fe_count(fe_positions)
        target = self.target_source_count(sites, fe_positions)
        setpoint = (
            target
            if self.maintenance_setpoint_h is None
            else int(self.maintenance_setpoint_h)
        )
        proposals: list[GCMCProposal] = []
        source_deleted: list[int] = []
        delete_enabled = self.reservoir.production_maintenance_mode in {
            "insert_delete",
            "balanced",
        }

        for _ in range(self.reservoir.attempts_per_adjustment):
            source_occupied = sum(site in source_set for site in current_indices)
            if source_occupied < setpoint:
                kind: MoveKind = "insert"
            elif source_occupied > setpoint and delete_enabled:
                kind = "delete"
            else:
                break
            indices_before = list(current_indices)
            proposal = self._attempt(current_indices, source_set, kind)
            if proposal is None:
                break
            proposals.append(proposal)
            if proposal.accepted and proposal.kind == "insert":
                current_unwrapped.append(np.asarray(sites[proposal.site], dtype=float).copy())
            elif proposal.accepted and proposal.kind == "delete":
                source_deleted.append(int(proposal.site))
                delete_slot = None
                for index, site in enumerate(indices_before):
                    if int(site) == int(proposal.site):
                        delete_slot = index
                        break
                if delete_slot is None:
                    raise RuntimeError(
                        f"Accepted GCMC deletion site {proposal.site} was not present"
                    )
                del current_unwrapped[delete_slot]

        inserted = tuple(
            proposal.site for proposal in proposals if proposal.kind == "insert" and proposal.accepted
        )
        self.cumulative_inserted += len(inserted)
        self.cumulative_removed += len(deleted_sink)
        self.cumulative_left_removed += len(deleted_left_sink)

        out_indices = np.asarray(current_indices, dtype=int)
        out_unwrapped = (
            np.asarray(current_unwrapped, dtype=float).reshape((len(current_unwrapped), 3))
            if current_unwrapped
            else np.empty((0, 3), dtype=float)
        )
        source_occupied = sum(int(site) in source_set for site in out_indices)
        sink_occupied = sum(
            float(sites[int(site), 0]) >= self.config.sink_x_min_a for site in out_indices
        )
        left_sink_occupied = sum(
            self._is_left_sink_site(int(site), float(sites[int(site), 0]))
            for site in out_indices
        )
        flux = None
        if elapsed_time_s > 0.0:
            flux = self.cumulative_removed / self.config.cross_section_area_a2 / elapsed_time_s

        return GCMCDevanathanUpdate(
            h_indices=out_indices,
            h_unwrapped_positions=out_unwrapped,
            inserted_sites=inserted,
            deleted_sink_sites=tuple(deleted_sink),
            trimmed_source_sites=tuple(source_deleted),
            source_sites=len(source_set),
            source_fe_atoms=source_fe_atoms,
            source_target_h=target,
            source_occupied=source_occupied,
            sink_occupied=sink_occupied,
            total_occupied=len(out_indices),
            cumulative_inserted=self.cumulative_inserted,
            cumulative_removed=self.cumulative_removed,
            flux_per_a2_s=flux,
            deleted_left_sink_sites=tuple(deleted_left_sink),
            cumulative_left_removed=self.cumulative_left_removed,
            left_sink_occupied=left_sink_occupied,
            gcmc_proposals=tuple(proposals),
            cumulative_gcmc_attempts=self.cumulative_gcmc_attempts,
            cumulative_gcmc_accepts=self.cumulative_gcmc_accepts,
            maintenance_setpoint_h=setpoint,
        )
