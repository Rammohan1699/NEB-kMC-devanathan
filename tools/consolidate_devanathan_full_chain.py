#!/usr/bin/env python3
"""Consolidate the Devanathan Sigma5 mu=-1.7215 run from step 0 onward."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from consolidate_segmented_external_results import (
    Segment,
    compile_trajectory,
    concat_step_log,
    filter_concat_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("hyperion_results/runs"),
        help="Directory containing copied Hyperion run directories.",
    )
    parser.add_argument(
        "--continuation",
        type=Path,
        default=Path(
            "hyperion_results/runs/consolidated/"
            "devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p7215_7m_serial_6003223"
            "__through_step1012564_partial_segment20"
        ),
        help="Consolidated continuation from step 365000 onward.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "hyperion_results/runs/consolidated/"
            "devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p7215_full_from_step0"
            "__through_step1012564_partial_segment20"
        ),
        help="Destination consolidated run directory.",
    )
    return parser.parse_args()


def build_segments(runs_root: Path, continuation: Path) -> list[Segment]:
    run_5978093 = runs_root / "devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p7215_7m_serial_5978093"
    run_5985504 = runs_root / "devanathan_sigma5_bicrystal_gcmc_fenwick_mu_m1p7215_7m_serial_5985504"

    return [
        Segment(1, 0, 100000, run_5978093 / "segment_01_step0000000_to_0500000", "bootstrap_external"),
        Segment(2, 100000, 200000, run_5985504 / "segment_02_step0100000_to_0200000", "completed"),
        Segment(3, 200000, 300000, run_5985504 / "segment_03_step0200000_to_0300000", "completed"),
        Segment(4, 300000, 365000, run_5985504 / "segment_04_step0300000_to_0400000", "interrupted_at_365000"),
        Segment(5, 365000, 1012565, continuation, "consolidated_partial_continuation"),
    ]


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_chain_manifest(segments: list[Segment], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["segment", "start_step", "end_step_exclusive", "local_root", "status"])
        for segment in segments:
            writer.writerow([segment.index, segment.start, segment.end, segment.root, segment.status])


def main() -> None:
    args = parse_args()
    segments = build_segments(args.runs_root, args.continuation)
    missing = [str(segment.root) for segment in segments if not segment.root.exists()]
    if missing:
        raise FileNotFoundError("Missing chain segment roots:\n" + "\n".join(missing))

    out_dir = args.out_dir
    diagnostics = out_dir / "diagnostics"
    logs = out_dir / "logs"

    selected_rows = filter_concat_csv(segments, "kmc_selected_events.csv", diagnostics / "kmc_selected_events.csv")
    timestep_rows = filter_concat_csv(segments, "kmc_timestep_vs_step.csv", diagnostics / "kmc_timestep_vs_step.csv")
    devanathan_rows = filter_concat_csv(segments, "kmc_devanathan.csv", diagnostics / "kmc_devanathan.csv")
    diag_rows = concat_step_log(segments, Path("diagnostics/kmc_diagnostics_rank0.log"), out_dir / "kmc_diagnostics_rank0.log")
    rank0_log_rows = concat_step_log(segments, Path("logs/log_rank0.txt"), logs / "log_rank0.txt")
    traj_frames, traj_first, traj_last = compile_trajectory(segments, out_dir / "trajectories/H_trajectory_onlyH.lammpstrj")

    write_chain_manifest(segments, out_dir / "full_chain_manifest.tsv")
    copy_if_exists(segments[0].root / "run_config_snapshot.env", out_dir / "run_config_snapshot.env")
    copy_if_exists(segments[0].root / "diagnostics" / "gcmc_initialization.csv", diagnostics / "gcmc_initialization.csv")
    copy_if_exists(
        segments[0].root / "diagnostics" / "gcmc_initialization_progress.csv",
        diagnostics / "gcmc_initialization_progress.csv",
    )

    readme = [
        "# Full Devanathan Sigma5 GCMC chain",
        "",
        "- chain: 5978093 segment 1, 5985504 segments 2-4, 6003223 partial continuation",
        "- first_step: 0",
        "- final_included_step: 1012564",
        f"- kmc_selected_events_rows: {selected_rows}",
        f"- kmc_devanathan_rows: {devanathan_rows}",
        f"- kmc_timestep_rows: {timestep_rows}",
        f"- kmc_diagnostics_step_rows: {diag_rows}",
        f"- rank0_log_step_rows: {rank0_log_rows}",
        f"- trajectory_frames: {traj_frames}",
        f"- trajectory_first_step: {traj_first}",
        f"- trajectory_last_step: {traj_last}",
        "",
        "NEB directories were excluded during rsync and are not part of this consolidation.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"out_dir={out_dir}")
    print(f"selected_rows={selected_rows}")
    print(f"devanathan_rows={devanathan_rows}")
    print(f"timestep_rows={timestep_rows}")
    print(f"trajectory_frames={traj_frames}")
    print(f"trajectory_first_step={traj_first}")
    print(f"trajectory_last_step={traj_last}")


if __name__ == "__main__":
    main()
