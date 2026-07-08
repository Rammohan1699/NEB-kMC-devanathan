#!/usr/bin/env python3
"""Compile restart-chain KMC outputs into one postprocessable run directory."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Segment:
    root: Path
    start: int
    end: int


STEP_LINE_RE = re.compile(r"^\s*(\d+)\s+\|")


def parse_segment(value: str) -> Segment:
    try:
        root_s, range_s = value.split(":", 1)
        start_s, end_s = range_s.split("-", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("segment must be RUN_DIR:START-END") from exc
    return Segment(Path(root_s).expanduser(), int(start_s), int(end_s))


def trajectory_frames(path: Path) -> Iterator[tuple[int, list[str]]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            frame = [line]
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"{path}: expected ITEM: TIMESTEP, got {line!r}")
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


def compile_trajectory(segments: list[Segment], out_path: Path, sample_every: int) -> tuple[int, int, int]:
    written = 0
    first_step = -1
    last_step = -1
    seen: set[int] = set()
    with out_path.open("w", encoding="utf-8") as out:
        if segments and segments[0].start <= 0 and 0 % sample_every == 0:
            zero_frame = synthesize_step_zero_frame(segments[0].root)
            if zero_frame is not None:
                out.writelines(zero_frame)
                seen.add(0)
                written += 1
                first_step = 0
                last_step = 0
        for segment in segments:
            path = segment.root / "H_trajectory_onlyH.lammpstrj"
            for step, frame in trajectory_frames(path):
                if step < segment.start or step > segment.end:
                    continue
                if step % sample_every != 0:
                    continue
                if step in seen:
                    continue
                out.writelines(frame)
                seen.add(step)
                written += 1
                if first_step < 0:
                    first_step = step
                last_step = step
    return written, first_step, last_step


def synthesize_step_zero_frame(root: Path) -> list[str] | None:
    """Reconstruct timestep 0 from frame 1 and the selected step-0 move row."""
    traj_path = root / "H_trajectory_onlyH.lammpstrj"
    selected_path = segment_csv_path(root, "kmc_selected_events.csv")
    if not traj_path.exists() or not selected_path.exists():
        return None
    try:
        first_step, frame = next(trajectory_frames(traj_path))
    except StopIteration:
        return None
    if first_step == 0:
        return frame
    if first_step != 1:
        return None

    step0_row: dict[str, str] | None = None
    with selected_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row.get("step", "-1")) == 0:
                step0_row = row
                break
    if step0_row is None:
        return None

    atom_id = int(step0_row["atom_id"])
    h_site = int(step0_row["h_site"])
    from_x = float(step0_row["from_x_A"])
    from_y = float(step0_row["from_y_A"])
    from_z = float(step0_row["from_z_A"])

    frame = list(frame)
    frame[1] = "0\n"
    atom_start = 9
    for idx in range(atom_start, len(frame)):
        parts = frame[idx].split()
        if parts and int(parts[0]) == atom_id:
            atom_type = parts[1] if len(parts) > 1 else "2"
            frame[idx] = f"{atom_id} {atom_type} {from_x:.10f} {from_y:.10f} {from_z:.10f} {h_site}\n"
            return frame
    return None


def compile_step_log(segments: list[Segment], out_path: Path, filename: str) -> int:
    written = 0
    wrote_header = False
    with out_path.open("w", encoding="utf-8") as out:
        for segment in segments:
            path = segment.root / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    match = STEP_LINE_RE.match(line)
                    if match:
                        step = int(match.group(1))
                        if segment.start <= step <= segment.end:
                            out.write(line)
                            written += 1
                    elif not wrote_header:
                        out.write(line)
            wrote_header = True
    return written


def segment_csv_path(root: Path, name: str) -> Path:
    nested = root / "diagnostics" / name
    return nested if nested.exists() else root / name


def compile_csv(segments: list[Segment], out_path: Path, filename: str, step_column: str = "step") -> int:
    written = 0
    fieldnames: list[str] | None = None
    with out_path.open("w", encoding="utf-8", newline="") as out_fh:
        writer: csv.DictWriter | None = None
        for segment in segments:
            path = segment_csv_path(segment.root, filename)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace", newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                if reader.fieldnames is None:
                    continue
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames)
                    writer = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                assert writer is not None
                for row in reader:
                    try:
                        step = int(row[step_column])
                    except Exception:
                        continue
                    if segment.start <= step <= segment.end:
                        writer.writerow(row)
                        written += 1
    return written


def link_or_copy_inputs(source: Path, out_dir: Path) -> None:
    for name in ("kmc_map_inputs", "PotentialB3410-modified.fs"):
        target = out_dir / name
        if target.exists() or target.is_symlink():
            continue
        src = source / name
        try:
            target.symlink_to(src.resolve())
        except OSError:
            if src.is_dir():
                shutil.copytree(src, target)
            else:
                shutil.copy2(src, target)


def write_manifest(
    out_dir: Path,
    segments: list[Segment],
    *,
    sample_every: int,
    traj_summary: tuple[int, int, int],
    diag_rows: int,
    selected_rows: int,
    timestep_rows: int,
) -> None:
    lines = [
        "# Compiled KMC Restart Chain",
        "",
        f"- trajectory_sample_every_steps: {sample_every}",
        f"- trajectory_frames_written: {traj_summary[0]}",
        f"- trajectory_first_step: {traj_summary[1]}",
        f"- trajectory_last_step: {traj_summary[2]}",
        f"- kmc_diagnostics_rows: {diag_rows}",
        f"- kmc_selected_events_rows: {selected_rows}",
        f"- kmc_timestep_rows: {timestep_rows}",
        "",
        "Segments:",
    ]
    for segment in segments:
        lines.append(f"- {segment.root}: {segment.start}-{segment.end}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--segment", action="append", type=parse_segment, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    segments = [Segment(s.root.resolve(), s.start, s.end) for s in args.segment]
    link_or_copy_inputs(segments[0].root, out_dir)

    traj_summary = compile_trajectory(segments, out_dir / "H_trajectory_onlyH.lammpstrj", args.sample_every)
    diag_rows = compile_step_log(segments, out_dir / "kmc_diagnostics_rank0.log", "kmc_diagnostics_rank0.log")
    selected_rows = compile_csv(segments, diagnostics_dir / "kmc_selected_events.csv", "kmc_selected_events.csv")
    timestep_rows = compile_csv(segments, diagnostics_dir / "kmc_timestep_vs_step.csv", "kmc_timestep_vs_step.csv")

    write_manifest(
        out_dir,
        segments,
        sample_every=args.sample_every,
        traj_summary=traj_summary,
        diag_rows=diag_rows,
        selected_rows=selected_rows,
        timestep_rows=timestep_rows,
    )
    print(f"compiled_dir={out_dir}")
    print(f"trajectory_frames={traj_summary[0]} first={traj_summary[1]} last={traj_summary[2]}")
    print(f"kmc_diagnostics_rows={diag_rows}")
    print(f"kmc_selected_events_rows={selected_rows}")
    print(f"kmc_timestep_rows={timestep_rows}")


if __name__ == "__main__":
    main()
