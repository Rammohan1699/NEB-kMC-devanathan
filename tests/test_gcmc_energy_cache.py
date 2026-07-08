from __future__ import annotations

import numpy as np

from kmc.gcmc_energy_cache import (
    PersistentGCMCEnergyCache,
    canonical_insertion_environment_key,
)


def make_key(sites: np.ndarray, occupied: list[int], trial_site: int):
    return canonical_insertion_environment_key(
        sites=sites,
        lower_occupancy_indices=occupied,
        trial_site=trial_site,
        box=[20.0, 20.0, 20.0],
        lattice_a=2.0,
        shell_inner_a=3.0,
        shell_outer_a=4.0,
        position_bin_a=0.1,
        potential_tag=("EAM", "test"),
    )


def test_environment_key_is_translation_and_order_invariant() -> None:
    sites = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [3.5, 1.0, 0.0],
        ]
    )
    first = make_key(sites, [1, 4], 0)
    translated = make_key(sites, [5, 3], 2)
    assert first == translated


def test_pickle_cache_persists_energy_difference(tmp_path) -> None:
    path = tmp_path / "gcmc_energy.pkl"
    key = ("environment", 1)

    cache = PersistentGCMCEnergyCache(path)
    assert cache.get(key) is None
    cache.set(key, -1.6)
    cache.save(full=True)

    restored = PersistentGCMCEnergyCache(path)
    assert restored.get(key) == -1.6
    assert len(restored) == 1


def test_undersized_box_disables_translation_reuse() -> None:
    sites = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ]
    )
    first = canonical_insertion_environment_key(
        sites=sites,
        lower_occupancy_indices=[1],
        trial_site=0,
        box=[8.0, 8.0, 8.0],
        lattice_a=2.0,
        shell_inner_a=3.0,
        shell_outer_a=4.0,
        position_bin_a=0.1,
        potential_tag=("EAM", "test"),
    )
    translated = canonical_insertion_environment_key(
        sites=sites,
        lower_occupancy_indices=[3],
        trial_site=2,
        box=[8.0, 8.0, 8.0],
        lattice_a=2.0,
        shell_inner_a=3.0,
        shell_outer_a=4.0,
        position_bin_a=0.1,
        potential_tag=("EAM", "test"),
    )
    assert first != translated


def test_external_host_geometry_namespaces_cache_entries() -> None:
    sites = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ]
    )
    common = dict(
        sites=sites,
        lower_occupancy_indices=[1],
        trial_site=0,
        box=[20.0, 20.0, 20.0],
        lattice_a=2.0,
        shell_inner_a=3.0,
        shell_outer_a=4.0,
        position_bin_a=0.1,
        potential_tag=("EAM", "test"),
    )

    first = canonical_insertion_environment_key(
        **common,
        host_geometry_tag=("external_host", "sigma5-a"),
    )
    second = canonical_insertion_environment_key(
        **common,
        host_geometry_tag=("external_host", "sigma5-b"),
    )

    assert first != second
    assert first[0] == "GCMC_SOURCE_INSERTION_DE_V2_HOST"


def test_generated_bulk_key_keeps_v1_cache_namespace() -> None:
    sites = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ]
    )
    key = make_key(sites, [1], 0)

    assert key[0] == "GCMC_SOURCE_INSERTION_DE_V1"
    assert len(key) == 10
