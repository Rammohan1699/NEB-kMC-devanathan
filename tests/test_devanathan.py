from __future__ import annotations

import math

import numpy as np

from kmc.devanathan import (
    DevanathanBoundary,
    DevanathanBoundaryConfig,
    DevanathanPulseBoundary,
)


def test_left_absorber_is_counted_separately_from_right_permeation_flux() -> None:
    sites = np.array(
        [
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ]
    )
    fe_positions = np.array([[4.0, 0.0, 0.0]])
    boundary = DevanathanBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=3.0,
            source_x_max_a=5.0,
            source_target_fraction=0.0,
            left_sink_x_max_a=2.0,
            sink_x_min_a=8.0,
            cross_section_area_a2=10.0,
            seed=7,
        ),
        np.array([10.0, 1.0, 1.0]),
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
    assert update.left_sink_occupied == 0
    assert update.sink_occupied == 0
    assert math.isclose(update.flux_per_a2_s or 0.0, 0.05)
    assert update.particle_count_changed


def test_left_absorber_cannot_overlap_source() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "DEVANATHAN_LEFT_SINK_X_MAX_A",
    ):
        DevanathanBoundary(
            DevanathanBoundaryConfig(
                source_x_min_a=3.0,
                source_x_max_a=5.0,
                left_sink_x_max_a=3.1,
                sink_x_min_a=8.0,
            ),
            np.array([10.0, 1.0, 1.0]),
        )


def test_pulse_boundary_deletes_at_sink_without_refilling() -> None:
    sites = np.array(
        [
            [4.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ]
    )
    fe_positions = np.array([[4.1, 0.0, 0.0], [4.9, 0.0, 0.0]])
    boundary = DevanathanPulseBoundary(
        DevanathanBoundaryConfig(
            source_x_min_a=3.0,
            source_x_max_a=5.0,
            source_target_fraction=0.0,
            sink_x_min_a=8.0,
            cross_section_area_a2=10.0,
            seed=11,
        ),
        np.array([10.0, 1.0, 1.0]),
        initial_count=2,
        control_site_indices=[0, 1],
        control_fe_count=2,
    )

    initial = boundary.initialize_source(sites, fe_positions)
    assert sorted(initial.tolist()) == [0, 1]

    update = boundary.apply(
        sites=sites,
        fe_positions=fe_positions,
        h_indices=np.array([0, 3], dtype=int),
        h_unwrapped_positions=sites[[0, 3]].copy(),
        elapsed_time_s=4.0,
    )

    assert update.inserted_sites == ()
    assert update.trimmed_source_sites == ()
    assert update.deleted_sink_sites == (3,)
    assert update.h_indices.tolist() == [0]
    assert update.source_target_h == 2
    assert update.source_occupied == 1
    assert update.total_occupied == 1
    assert update.cumulative_removed == 1
    assert math.isclose(update.flux_per_a2_s or 0.0, 0.025)
