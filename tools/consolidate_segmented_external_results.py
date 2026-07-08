#!/usr/bin/env python3
"""Consolidate copied segmented external KMC runs without NEB scratch trees."""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Segment:
    index: int
    start: int
    end: int
    root: Path
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("hyperion_results/devanathan_external_completed"),
        help="Root containing copied external_* arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("hyperion_results/devanathan_external_completed/consolidated"),
        help="Destination for consolidated per-seed outputs.",
    )
    parser.add_argument(
        "--status",
        default="completed",
        help="Manifest status to include.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Specific segmented run directory to consolidate. May be passed more than once.",
    )
    return parser.parse_args()


def discover_runs(root: Path) -> list[Path]:
    candidates = {p.parent for p in root.glob("external_*/*/serial_manifest.tsv")}
    candidates.update(p.parent for p in root.glob("*/serial_manifest.tsv"))
    return sorted(candidates)


def manifest_paths(seed_dir: Path) -> list[Path]:
    paths = [seed_dir / "serial_manifest.tsv"]
    paths.extend(sorted(seed_dir.glob("extension_*_manifest_*.tsv")))
    return [path for path in paths if path.exists()]


def read_segments(seed_dir: Path, status: str) -> list[Segment]:
    segments: list[Segment] = []
    seen: set[tuple[int, int, int]] = set()
    for manifest in manifest_paths(seed_dir):
        with manifest.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if row.get("status") != status:
                    continue
                run_path = row.get("run_dir") or row.get("run_root")
                if not run_path:
                    continue
                segment_key = (int(row["segment"]), int(row["start_step"]), int(row["end_step"]))
                if segment_key in seen:
                    continue
                seen.add(segment_key)
                remote_run_dir = Path(run_path)
                local_root = seed_dir / remote_run_dir.name
                segments.append(
                    Segment(
                        index=segment_key[0],
                        start=segment_key[1],
                        end=segment_key[2],
                        root=local_root,
                        status=row["status"],
                    )
                )
    return sorted(segments, key=lambda s: (s.start, s.index))


def output_name(seed_dir: Path) -> str:
    if not seed_dir.parent.name.startswith("external_"):
        return seed_dir.name
    return f"{seed_dir.parent.name}__{seed_dir.name}"


def csv_path(segment_root: Path, name: str) -> Path:
    nested = segment_root / "diagnostics" / name
    return nested if nested.exists() else segment_root / name


def filter_concat_csv(segments: list[Segment], filename: str, out_path: Path) -> int:
    rows = 0
    header: str | None = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as out:
        for segment in segments:
            path = csv_path(segment.root, filename)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                local_header = fh.readline()
                if not local_header:
                    continue
                if header is None:
                    header = local_header
                    out.write(header)
                for line in fh:
                    first = line.split(",", 1)[0]
                    try:
                        step = int(first)
                    except ValueError:
                        continue
                    if segment.start <= step < segment.end:
                        out.write(line)
                        rows += 1
    return rows


STEP_LINE_RE = re.compile(r"^\s*(\d+)\s+\|")


def concat_step_log(segments: list[Segment], source_rel: Path, out_path: Path) -> int:
    rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for segment in segments:
            path = segment.root / source_rel
            if not path.exists():
                continue
            out.write(f"\n===== {segment.root.name} {segment.start}-{segment.end} =====\n")
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    match = STEP_LINE_RE.match(line)
                    if match:
                        step = int(match.group(1))
                        if not (segment.start <= step < segment.end):
                            continue
                        rows += 1
                    out.write(line)
    return rows


def trajectory_frames(path: Path) -> Iterator[tuple[int, list[str]]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            first = fh.readline()
            if not first:
                return
            frame = [first]
            if first.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"{path}: expected ITEM: TIMESTEP, got {first!r}")
            step_line = fh.readline()
            if not step_line:
                return
            frame.append(step_line)
            step = int(step_line.strip())
            marker = fh.readline()
            frame.append(marker)
            if marker.strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"{path}: expected atom count marker at timestep {step}")
            natoms_line = fh.readline()
            frame.append(natoms_line)
            natoms = int(natoms_line.strip())
            for _ in range(5 + natoms):
                item = fh.readline()
                if not item:
                    raise ValueError(f"{path}: truncated frame at timestep {step}")
                frame.append(item)
            yield step, frame


def compile_trajectory(segments: list[Segment], out_path: Path) -> tuple[int, int | None, int | None]:
    written = 0
    first_step: int | None = None
    last_step: int | None = None
    seen: set[int] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for segment in segments:
            path = segment.root / "trajectories" / "H_trajectory_onlyH.lammpstrj"
            if not path.exists():
                path = segment.root / "H_trajectory_onlyH.lammpstrj"
            if not path.exists():
                continue
            for step, frame in trajectory_frames(path):
                if step in seen:
                    continue
                if segment.start <= step <= segment.end:
                    out.writelines(frame)
                    seen.add(step)
                    written += 1
                    if first_step is None:
                        first_step = step
                    last_step = step
    return written, first_step, last_step


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def final_cache(segment: Segment) -> Path | None:
    cache_dir = segment.root / "cache"
    if not cache_dir.exists():
        return None
    merged = sorted(p for p in cache_dir.glob("*.pkl") if not p.name.endswith(".delta.pkl"))
    return merged[-1] if merged else None


def final_checkpoint(segment: Segment) -> Path | None:
    checkpoint = segment.root / "checkpoints" / f"kmc_restart_checkpoint_step{segment.end}.pkl"
    if checkpoint.exists():
        return checkpoint
    checkpoint = segment.root / "checkpoints" / "kmc_restart_checkpoint.pkl"
    return checkpoint if checkpoint.exists() else None


def consolidate_seed(seed_dir: Path, out_root: Path, status: str) -> dict[str, object]:
    segments = read_segments(seed_dir, status)
    if not segments:
        return {"seed": seed_dir.name, "segments": 0, "skipped": "no matching segments"}

    missing = [str(s.root) for s in segments if not s.root.is_dir()]
    if missing:
        raise FileNotFoundError(f"{seed_dir}: missing segment directories: {missing[:3]}")

    out_dir = out_root / output_name(seed_dir)
    diagnostics = out_dir / "diagnostics"
    logs = out_dir / "logs"

    selected_rows = filter_concat_csv(segments, "kmc_selected_events.csv", diagnostics / "kmc_selected_events.csv")
    timestep_rows = filter_concat_csv(segments, "kmc_timestep_vs_step.csv", diagnostics / "kmc_timestep_vs_step.csv")
    devanathan_rows = filter_concat_csv(segments, "kmc_devanathan.csv", diagnostics / "kmc_devanathan.csv")
    diag_rows = concat_step_log(segments, Path("diagnostics/kmc_diagnostics_rank0.log"), out_dir / "kmc_diagnostics_rank0.log")
    rank0_log_rows = concat_step_log(segments, Path("logs/log_rank0.txt"), logs / "log_rank0.txt")
    traj_frames, traj_first, traj_last = compile_trajectory(segments, out_dir / "trajectories/H_trajectory_onlyH.lammpstrj")

    for manifest in manifest_paths(seed_dir):
        copy_if_exists(manifest, out_dir / manifest.name)
    copy_if_exists(seed_dir / "array_task_status.env", out_dir / "array_task_status.env")
    copy_if_exists(segments[0].root / "run_config_snapshot.env", out_dir / "run_config_snapshot.env")
    copy_if_exists(segments[0].root / "diagnostics" / "gcmc_initialization.csv", diagnostics / "gcmc_initialization.csv")
    copy_if_exists(
        segments[0].root / "diagnostics" / "gcmc_initialization_progress.csv",
        diagnostics / "gcmc_initialization_progress.csv",
    )

    last_segment = segments[-1]
    checkpoint = final_checkpoint(last_segment)
    cache = final_cache(last_segment)
    if checkpoint is not None:
        relative_symlink(checkpoint, out_dir / "checkpoints" / checkpoint.name)
        relative_symlink(checkpoint, out_dir / "checkpoints" / "latest_checkpoint.pkl")
    if cache is not None:
        relative_symlink(cache, out_dir / "cache" / cache.name)
        relative_symlink(cache, out_dir / "cache" / "latest_barrier_cache.pkl")

    readme = [
        "# Consolidated segmented KMC run",
        "",
        f"- source_seed_dir: {seed_dir}",
        f"- included_segments: {len(segments)}",
        f"- first_step: {segments[0].start}",
        f"- final_step: {last_segment.end}",
        f"- kmc_selected_events_rows: {selected_rows}",
        f"- kmc_devanathan_rows: {devanathan_rows}",
        f"- kmc_timestep_rows: {timestep_rows}",
        f"- kmc_diagnostics_step_rows: {diag_rows}",
        f"- rank0_log_step_rows: {rank0_log_rows}",
        f"- trajectory_frames: {traj_frames}",
        f"- trajectory_first_step: {traj_first}",
        f"- trajectory_last_step: {traj_last}",
        f"- latest_checkpoint: {checkpoint if checkpoint is not None else 'missing'}",
        f"- latest_barrier_cache: {cache if cache is not None else 'missing'}",
        f"- gcmc_initialization_source: {segments[0].root / 'diagnostics'}",
        "",
        "NEB scratch directories are intentionally not copied or consolidated.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    return {
        "seed": seed_dir.name,
        "group": seed_dir.parent.name,
        "out_dir": str(out_dir),
        "segments": len(segments),
        "first_step": segments[0].start,
        "final_step": last_segment.end,
        "selected_rows": selected_rows,
        "devanathan_rows": devanathan_rows,
        "timestep_rows": timestep_rows,
        "trajectory_frames": traj_frames,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs = [p.expanduser().resolve() for p in args.run_dir] if args.run_dir else discover_runs(root)
    summaries = []
    for seed_dir in run_dirs:
        if not (seed_dir / "serial_manifest.tsv").exists():
            raise FileNotFoundError(f"{seed_dir}: missing serial_manifest.tsv")
        if out_root in seed_dir.resolve().parents:
            continue
        summaries.append(consolidate_seed(seed_dir, out_root, args.status))

    summary_path = out_root / "consolidation_summary.tsv"
    fields = [
        "group",
        "seed",
        "segments",
        "first_step",
        "final_step",
        "selected_rows",
        "devanathan_rows",
        "timestep_rows",
        "trajectory_frames",
        "out_dir",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"consolidated_runs={len(summaries)}")
    print(f"summary={summary_path}")
    for row in summaries:
        print(
            f"{row.get('group')}/{row.get('seed')}: "
            f"segments={row.get('segments')} final_step={row.get('final_step')} "
            f"selected_rows={row.get('selected_rows')} "
            f"devanathan_rows={row.get('devanathan_rows')} "
            f"trajectory_frames={row.get('trajectory_frames')}"
        )


if __name__ == "__main__":
    main()
