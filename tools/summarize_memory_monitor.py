#!/usr/bin/env python3
"""Summarize memory_monitor/process_rss.csv from a Devanathan debug run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def mib(kb: float) -> float:
    return kb / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="Run directory containing memory_monitor/process_rss.csv")
    parser.add_argument("--top", type=int, default=12, help="Number of peak processes to print")
    args = parser.parse_args()

    path = Path(args.run_root) / "memory_monitor" / "process_rss.csv"
    if not path.exists():
        raise SystemExit(f"missing monitor CSV: {path}")

    total_by_ts: dict[str, int] = defaultdict(int)
    peak_by_pid: dict[str, dict[str, str]] = {}
    sample_count = 0

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sample_count += 1
            ts = row["timestamp"]
            rss = int(float(row["rss_kb"] or 0))
            total_by_ts[ts] += rss
            pid = row["pid"]
            prior = peak_by_pid.get(pid)
            if prior is None or rss > int(float(prior["rss_kb"] or 0)):
                peak_by_pid[pid] = row

    if not total_by_ts:
        raise SystemExit(f"no samples in {path}")

    peak_ts, peak_total_kb = max(total_by_ts.items(), key=lambda item: item[1])
    print(f"samples: {sample_count}")
    print(f"peak_tracked_rss: {mib(peak_total_kb):.1f} MiB at {peak_ts}")
    print()
    print("top_process_peaks:")
    rows = sorted(
        peak_by_pid.values(),
        key=lambda row: int(float(row["rss_kb"] or 0)),
        reverse=True,
    )
    for row in rows[: max(1, args.top)]:
        rss_kb = int(float(row["rss_kb"] or 0))
        print(
            f"  pid={row['pid']} rss={mib(rss_kb):.1f} MiB "
            f"cmd={row['command']} args={row['args'][:180]}"
        )

    print()
    print(f"raw_csv: {path}")
    print(f"vm_stat: {path.parent / 'vm_stat.log'}")
    print(f"memory_pressure: {path.parent / 'memory_pressure.log'}")
    print(f"top_processes: {path.parent / 'top_memory_processes.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
