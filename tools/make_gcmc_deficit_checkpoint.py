#!/usr/bin/env python3
"""Create a validation checkpoint with a one-H source-zone imbalance."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from kmc.lattice import generate_all_interstitial_sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("deficit", "excess"), default="deficit")
    parser.add_argument("--source-x-min-a", type=float, default=0.0)
    parser.add_argument("--source-x-max-a", type=float, default=10.0)
    parser.add_argument("--nx", type=int, default=90)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--lattice-a", type=float, default=2.8601)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary restart checkpoint")

    sites = generate_all_interstitial_sites(
        args.nx, args.ny, args.nz, args.lattice_a
    )["tetra"]
    h_indices = np.asarray(checkpoint["h_indices"], dtype=int)
    source_slots = [
        slot
        for slot, site in enumerate(h_indices)
        if args.source_x_min_a <= float(sites[int(site), 0]) < args.source_x_max_a
    ]
    if args.mode == "deficit":
        if not source_slots:
            raise RuntimeError("Checkpoint has no H in the requested source zone")
        changed_slot = source_slots[0]
        changed_site = int(h_indices[changed_slot])
        checkpoint["h_indices"] = np.delete(h_indices, changed_slot)
        checkpoint["h_unwrapped_positions"] = np.delete(
            np.asarray(checkpoint["h_unwrapped_positions"], dtype=float),
            changed_slot,
            axis=0,
        )
    else:
        occupied = set(int(site) for site in h_indices)
        vacant_source = [
            site
            for site in range(len(sites))
            if site not in occupied
            and args.source_x_min_a <= float(sites[site, 0]) < args.source_x_max_a
        ]
        if not vacant_source:
            raise RuntimeError("Checkpoint has no vacant site in the requested source zone")
        changed_site = int(vacant_source[0])
        checkpoint["h_indices"] = np.append(h_indices, changed_site)
        checkpoint["h_unwrapped_positions"] = np.vstack(
            (
                np.asarray(checkpoint["h_unwrapped_positions"], dtype=float),
                np.asarray(sites[changed_site], dtype=float),
            )
        )
    checkpoint["num_h"] = int(len(checkpoint["h_indices"]))
    checkpoint["validation_source_imbalance_mode"] = args.mode
    checkpoint["validation_changed_source_site"] = changed_site

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output}")
    print(
        f"Created source {args.mode} at site {changed_site}; "
        f"H count is now {checkpoint['num_h']}"
    )


if __name__ == "__main__":
    main()
