#!/usr/bin/env python3
"""Trim run outputs to a restart checkpoint while preserving cache files."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "step" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return int(payload["step"])


def atomic_replace(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".restart.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def trim_step_csv(path: Path, restart_step: int) -> tuple[int, int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or not rows[0] or rows[0][0].strip().lower() != "step":
        return 0, 0
    kept = [rows[0]]
    removed = 0
    for row in rows[1:]:
        if not row:
            continue
        try:
            step = int(row[0])
        except ValueError:
            kept.append(row)
            continue
        if step < restart_step:
            kept.append(row)
        else:
            removed += 1
    temporary = path.with_suffix(path.suffix + ".restart.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(kept)
    os.replace(temporary, path)
    return len(kept) - 1, removed


def trim_kmc_log(path: Path, restart_step: int) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept: list[str] = []
    removed = 0
    for line in lines:
        prefix = line.split("|", 1)[0].strip()
        try:
            step = int(prefix)
        except ValueError:
            kept.append(line)
            continue
        if step < restart_step:
            kept.append(line)
        else:
            removed += 1
    atomic_replace(path, "".join(kept))
    return len(kept), removed


def trim_lammpstrj(path: Path, restart_step: int) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    frames: list[tuple[int, list[str]]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "ITEM: TIMESTEP" or index + 1 >= len(lines):
            raise ValueError(f"Malformed trajectory near line {index + 1}: {path}")
        start = index
        step = int(lines[index + 1].strip())
        index += 2
        while index < len(lines) and lines[index].strip() != "ITEM: TIMESTEP":
            index += 1
        frames.append((step, lines[start:index]))
    kept = [frame for step, frame in frames if step <= restart_step]
    removed = len(frames) - len(kept)
    atomic_replace(path, "".join(line for frame in kept for line in frame))
    return len(kept), removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    restart_step = checkpoint_step(checkpoint)
    print(f"Preparing {run_dir} for in-place restart at step {restart_step}")

    candidates = [
        run_dir / "diagnostics" / "kmc_selected_events.csv",
        run_dir / "diagnostics" / "kmc_timestep_vs_step.csv",
        run_dir / "diagnostics" / "kmc_devanathan.csv",
        run_dir / "diagnostics" / "rates_allranks.csv",
    ]
    candidates.extend((run_dir / "logs").glob("timing_rank*.csv"))
    candidates.extend((run_dir / "neb").glob("neb_diag*.csv"))
    for path in candidates:
        if path.exists():
            kept, removed = trim_step_csv(path, restart_step)
            print(f"  {path.relative_to(run_dir)}: kept={kept} removed={removed}")

    kmc_log = run_dir / "diagnostics" / "kmc_diagnostics_rank0.log"
    if kmc_log.exists():
        kept, removed = trim_kmc_log(kmc_log, restart_step)
        print(f"  {kmc_log.relative_to(run_dir)}: kept_lines={kept} removed={removed}")

    trajectory = run_dir / "trajectories" / "H_trajectory_onlyH.lammpstrj"
    if trajectory.exists():
        kept, removed = trim_lammpstrj(trajectory, restart_step)
        print(f"  {trajectory.relative_to(run_dir)}: kept_frames={kept} removed={removed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
