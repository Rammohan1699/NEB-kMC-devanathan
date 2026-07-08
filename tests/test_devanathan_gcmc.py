from __future__ import annotations

import math

import numpy as np

from kmc.devanathan import DevanathanBoundaryConfig
from kmc.devanathan_gcmc import GCMCDevanathanBoundary, GCMCReservoirConfig


SITES = np.array(
    [
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [5.5, 0.0, 0.0],
        [9.5, 0.0, 0.0],
    ]
)
FE = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])


def make_boundary(
    delta_e: float = 0.0,
    production_mode: str = "insert_only",
) -> GCMCDevanathanBoundary:
    return GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=3.0,
            source_target_fraction=0.5,
            sink_x_min_a=9.0,
            cross_section_area_a2=10.0,
            seed=4,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=-0.2,
            temperature_k=300.0,
            attempts_per_adjustment=1,
            production_maintenance_mode=production_mode,
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda _indices, _site, _kind: delta_e,
    )


def test_acceptance_equations_match_sweep_tool() -> None:
    boundary = make_boundary()
    kbt = boundary.reservoir.kbt_ev
    assert math.isclose(
        boundary._acceptance_probability("insert", -0.1),
        math.exp(-(-0.1 + 0.2) / kbt),
    )
    assert boundary._acceptance_probability("delete", 0.1) == 1.0


def test_below_target_proposes_gcmc_insertion() -> None:
    boundary = make_boundary(delta_e=-1.0)
    update = boundary.apply(
        sites=SITES,
        fe_positions=FE,
        h_indices=np.array([], dtype=int),
        h_unwrapped_positions=np.empty((0, 3)),
        elapsed_time_s=1.0,
    )
    assert len(update.gcmc_proposals) == 1
    assert update.gcmc_proposals[0].kind == "insert"
    assert update.gcmc_proposals[0].accepted
    assert update.source_occupied == update.source_target_h == 1


def test_above_target_retains_returning_h() -> None:
    boundary = make_boundary(delta_e=-1.0)
    update = boundary.apply(
        sites=SITES,
        fe_positions=FE,
        h_indices=np.array([0, 1], dtype=int),
        h_unwrapped_positions=SITES[[0, 1]].copy(),
        elapsed_time_s=1.0,
    )
    assert update.gcmc_proposals == ()
    assert update.trimmed_source_sites == ()
    assert update.source_occupied == 2
    assert update.source_target_h == 1


def test_insert_delete_mode_deletes_source_h_above_setpoint() -> None:
    observed_moves: list[tuple[tuple[int, ...], int, str]] = []

    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=3.0,
            source_target_fraction=0.5,
            sink_x_min_a=9.0,
            cross_section_area_a2=10.0,
            seed=4,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=-0.2,
            temperature_k=300.0,
            attempts_per_adjustment=1,
            production_maintenance_mode="insert_delete",
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda indices, site, kind: (
            observed_moves.append((tuple(map(int, indices)), int(site), kind))
            or 0.0
        ),
    )

    update = boundary.apply(
        sites=SITES,
        fe_positions=FE,
        h_indices=np.array([0, 1], dtype=int),
        h_unwrapped_positions=SITES[[0, 1]].copy(),
        elapsed_time_s=1.0,
    )

    assert len(update.gcmc_proposals) == 1
    assert update.gcmc_proposals[0].kind == "delete"
    assert update.gcmc_proposals[0].accepted
    assert update.trimmed_source_sites == (update.gcmc_proposals[0].site,)
    assert update.source_occupied == update.source_target_h == 1
    assert len(update.h_indices) == len(update.h_unwrapped_positions) == 1
    assert observed_moves == [((0, 1), update.gcmc_proposals[0].site, "delete")]


def test_initialization_uses_full_gcmc_and_sets_converged_mean() -> None:
    progress_rows: list[dict[str, object]] = []
    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=2.0,
            source_target_fraction=0.5,
            sink_x_min_a=9.0,
            cross_section_area_a2=10.0,
            seed=9,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=0.0,
            temperature_k=300.0,
            attempts_per_adjustment=1,
            initialization_min_attempts=40,
            initialization_max_attempts=1000,
            initialization_progress_interval=10,
            convergence_window_attempts=40,
            convergence_check_interval=10,
            convergence_drift_tolerance_h=0.5,
            target_fraction_tolerance=0.25,
            convergence_required_checks=1,
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda _indices, _site, _kind: 0.0,
        lambda row: progress_rows.append(dict(row)),
    )

    occupied = boundary.initialize_source(SITES, FE)
    summary = boundary.initialization_summary

    assert summary is not None and summary.converged
    assert summary.attempts >= 40
    assert 0 <= len(occupied) <= 2
    assert boundary.maintenance_setpoint_h == round(summary.mean_h)
    assert progress_rows
    assert progress_rows[0]["attempts"] == 1
    assert any(int(row["attempts"]) % 10 == 0 for row in progress_rows)


def test_initialization_can_start_at_target_occupancy() -> None:
    observed_initial_counts: list[int] = []

    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=2.0,
            source_target_fraction=0.5,
            sink_x_min_a=9.0,
            cross_section_area_a2=10.0,
            seed=9,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=0.0,
            initialization_start="target",
            initialization_min_attempts=2,
            initialization_max_attempts=2,
            convergence_window_attempts=2,
            convergence_check_interval=1,
            convergence_required_checks=3,
            target_fraction_tolerance=1.0,
            accept_near_target_at_max=True,
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda indices, _site, _kind: (
            observed_initial_counts.append(len(indices)) or 0.0
        ),
    )

    boundary.initialize_source(SITES, FE)

    assert observed_initial_counts[0] == 1


def test_initialization_accepts_near_target_at_max_when_enabled() -> None:
    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=2.0,
            source_target_fraction=0.5,
            sink_x_min_a=9.0,
            cross_section_area_a2=10.0,
            seed=9,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=0.0,
            initialization_start="target",
            initialization_min_attempts=20,
            initialization_max_attempts=20,
            convergence_window_attempts=20,
            convergence_check_interval=1,
            convergence_required_checks=21,
            target_fraction_tolerance=1.0,
            accept_near_target_at_max=True,
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda _indices, _site, _kind: 0.0,
    )

    occupied = boundary.initialize_source(SITES, FE)

    assert len(occupied) <= 2
    assert boundary.initialization_summary is not None
    assert not boundary.initialization_summary.converged


def test_target_maintenance_mode_uses_requested_concentration_count() -> None:
    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=2.0,
            source_target_fraction=0.5,
            sink_x_min_a=9.0,
            cross_section_area_a2=10.0,
            seed=9,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=0.0,
            initialization_start="target",
            maintenance_setpoint_mode="target",
            initialization_min_attempts=20,
            initialization_max_attempts=20,
            convergence_window_attempts=20,
            convergence_check_interval=1,
            convergence_required_checks=21,
            target_fraction_tolerance=1.0,
            accept_near_target_at_max=True,
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda _indices, _site, _kind: 0.0,
    )

    boundary.initialize_source(SITES, FE)

    assert boundary.maintenance_setpoint_h == 1
    assert boundary.initialization_summary is not None
    assert boundary.initialization_summary.maintenance_setpoint_h == 1


def test_gcmc_state_round_trip_preserves_setpoint_and_counters() -> None:
    boundary = make_boundary(delta_e=-1.0)
    boundary.maintenance_setpoint_h = 3
    boundary.cumulative_inserted = 4
    boundary.cumulative_removed = 2
    boundary.cumulative_left_removed = 5
    state = boundary.state_dict()

    restored = make_boundary(delta_e=-1.0)
    restored.load_state_dict(state)

    assert restored.maintenance_setpoint_h == 3
    assert restored.cumulative_inserted == 4
    assert restored.cumulative_removed == 2
    assert restored.cumulative_left_removed == 5


def test_gcmc_left_absorber_does_not_increment_right_flux_counter() -> None:
    sites = np.array(
        [
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ]
    )
    fe_positions = np.array([[4.0, 0.0, 0.0]])
    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=3.0,
            source_x_max_a=5.0,
            source_target_fraction=0.0,
            left_sink_x_max_a=2.0,
            sink_x_min_a=8.0,
            cross_section_area_a2=10.0,
            seed=4,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=-0.2,
            attempts_per_adjustment=1,
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda _indices, _site, _kind: 0.0,
    )

    update = boundary.apply(
        sites=sites,
        fe_positions=fe_positions,
        h_indices=np.array([0, 2], dtype=int),
        h_unwrapped_positions=sites[[0, 2]].copy(),
        elapsed_time_s=2.0,
    )

    assert update.deleted_left_sink_sites == (0,)
    assert update.deleted_sink_sites == (2,)
    assert update.cumulative_left_removed == 1
    assert update.cumulative_removed == 1
    assert math.isclose(update.flux_per_a2_s or 0.0, 0.05)


def test_bulk_region_insert_delete_maintains_target_h_per_fe() -> None:
    sites = np.array([[x + 0.5, 0.0, 0.0] for x in range(10)], dtype=float)
    fe_positions = np.column_stack(
        (
            np.linspace(0.01, 7.99, 1000),
            np.zeros(1000),
            np.zeros(1000),
        )
    )
    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=8.0,
            source_target_fraction=0.001,
            sink_x_min_a=9.5,
            cross_section_area_a2=10.0,
            seed=4,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=-0.2,
            attempts_per_adjustment=1,
            maintenance_setpoint_mode="target",
            production_maintenance_mode="insert_delete",
        ),
        np.array([10.0, 1.0, 1.0]),
        lambda _indices, _site, _kind: -1.0,
        region_label="bulk",
    )

    assert boundary.region_label == "bulk"
    assert boundary.source_fe_count(fe_positions) == 1000
    assert boundary.target_source_count(sites, fe_positions) == 1

    refill = boundary.apply(
        sites=sites,
        fe_positions=fe_positions,
        h_indices=np.array([], dtype=int),
        h_unwrapped_positions=np.empty((0, 3)),
        elapsed_time_s=1.0,
    )
    assert len(refill.gcmc_proposals) == 1
    assert refill.gcmc_proposals[0].kind == "insert"
    assert refill.gcmc_proposals[0].accepted
    assert refill.source_occupied == refill.source_target_h == 1

    rebalance = boundary.apply(
        sites=sites,
        fe_positions=fe_positions,
        h_indices=np.array([0, 1], dtype=int),
        h_unwrapped_positions=sites[[0, 1]].copy(),
        elapsed_time_s=1.0,
    )
    assert len(rebalance.gcmc_proposals) == 1
    assert rebalance.gcmc_proposals[0].kind == "delete"
    assert rebalance.gcmc_proposals[0].accepted
    assert rebalance.source_occupied == rebalance.source_target_h == 1


def test_control_site_indices_override_x_window_for_external_bulk() -> None:
    sites = np.array([[x + 0.5, 0.0, 0.0] for x in range(6)], dtype=float)
    fe_positions = np.array([[x + 0.5, 0.0, 0.0] for x in range(20)], dtype=float)
    boundary = GCMCDevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=0.0,
            source_x_max_a=5.0,
            source_target_fraction=0.5,
            sink_x_min_a=5.5,
            cross_section_area_a2=10.0,
            seed=4,
        ),
        GCMCReservoirConfig(
            chemical_potential_ev=-0.2,
            attempts_per_adjustment=2,
            maintenance_setpoint_mode="target",
            production_maintenance_mode="insert_delete",
        ),
        np.array([6.0, 1.0, 1.0]),
        lambda _indices, _site, _kind: -1.0,
        region_label="bulk:bulk_grain_0",
        control_site_indices=(1, 3),
        control_fe_count=4,
    )

    assert boundary.source_sites(sites) == (1, 3)
    assert boundary.source_fe_count(fe_positions) == 4
    assert boundary.target_source_count(sites, fe_positions) == 2

    update = boundary.apply(
        sites=sites,
        fe_positions=fe_positions,
        h_indices=np.array([], dtype=int),
        h_unwrapped_positions=np.empty((0, 3)),
        elapsed_time_s=1.0,
    )

    assert set(update.inserted_sites) == {1, 3}
    assert update.source_occupied == update.source_target_h == 2
