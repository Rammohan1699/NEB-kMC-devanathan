from __future__ import annotations

from argparse import Namespace

import numpy as np

from tools.gcmc_external_charging_sweep import (
    aggregate_rows,
    select_source_region,
    target_h_count,
)


def test_region_selection_uses_labels_x_window_and_unique_fe(tmp_path) -> None:
    region_file = tmp_path / "sigma5_site_regions.npz"
    np.savez(
        region_file,
        site_selection_label=np.asarray(
            [
                "bulk_grain_0",
                "bulk_grain_0",
                "bulk_grain_1",
                "bulk_grain_0",
                "bulk_grain_0",
                "grain_boundary",
            ],
            dtype=object,
        ),
        nearest_fe_atom_index=np.asarray([10, 10, 20, 11, 12, 30], dtype=int),
    )
    sites = np.asarray([[x, 0.0, 0.0] for x in range(6)], dtype=float)
    args = Namespace(
        region_file=str(region_file),
        region_labels="bulk_grain_0",
        bulk_x_min=0.0,
        bulk_x_max=4.5,
        sink_x_min=None,
        bulk_sink_gap=1.0,
        source_x_min=0.0,
        source_width=2.0,
        target_n_h=2,
        target_fraction=0.5,
    )

    selection = select_source_region(
        args,
        sites=sites,
        host_fe=np.empty((0, 3)),
        box=np.asarray([10.0, 1.0, 1.0]),
    )

    assert selection.mode == "region"
    assert selection.region_labels == ("bulk_grain_0",)
    assert selection.site_indices.tolist() == [0, 1, 3, 4]
    assert selection.fe_count == 3
    assert target_h_count(args, selection.fe_count, len(selection.site_indices)) == 2


def test_aggregate_recommendation_distance_uses_absolute_target_h() -> None:
    rows = [
        {
            "mu_eV": "-1.76",
            "mean_N_H": "42.0",
            "mean_c_H_per_Fe": "0.00262",
            "half_drift_N_H": "1.0",
            "accepted_transitions": "12",
            "target_N_H": "50",
            "target_c_H_per_Fe": "0.00312",
            "source_Fe": "16037",
        },
        {
            "mu_eV": "-1.74",
            "mean_N_H": "53.0",
            "mean_c_H_per_Fe": "0.00331",
            "half_drift_N_H": "1.0",
            "accepted_transitions": "14",
            "target_N_H": "50",
            "target_c_H_per_Fe": "0.00312",
            "source_Fe": "16037",
        },
    ]

    aggregate = aggregate_rows(rows)

    assert aggregate[0]["distance_from_target_N_H"] == 8.0
    assert aggregate[1]["distance_from_target_N_H"] == 3.0
