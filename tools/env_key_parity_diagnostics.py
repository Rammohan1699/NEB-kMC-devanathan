#!/usr/bin/env python3
"""Compare current environment keys against legacy step-0 NEB key logs.

This diagnostic reconstructs environment keys from ``rates_allranks.csv`` rows
using the current ``kmc.environment`` implementation, then compares them with
keys logged by the older monolithic driver for the same H-site moves. It helps
verify that cache-key compatibility has been preserved after changes to local
environment construction, binning, or neighbor handling.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kmc.environment import build_hydrogen_kdtree, make_env_key
from kmc.lattice import generate_all_interstitial_sites


START_RE = re.compile(r"NEB: starting job .*?\(key=(.*), h=(\d+), n=(\d+)\)\.")
STEP_RE = re.compile(r"=+ STEP (\d+)")


def normalize_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(normalize_key(v) for v in value)
    if isinstance(value, list):
        return tuple(normalize_key(v) for v in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def load_step_rows(path: Path, step: int) -> list[dict[str, str]]:
    if not path.exists() and path == Path("rates_allranks.csv"):
        moved = Path("diagnostics") / "rates_allranks.csv"
        if moved.exists():
            path = moved
    rows: list[dict[str, str]] = []
    started = False
    with path.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["step"]) == step:
                rows.append(row)
                started = True
            elif started:
                break
    return rows


def parse_monolith_logged_jobs(log_dir: Path, step: int) -> dict[tuple[int, int], Any]:
    jobs: dict[tuple[int, int], Any] = {}
    duplicates: list[tuple[int, int]] = []
    if not any(log_dir.glob("log_rank[0-9]*.txt")) and (log_dir / "logs").exists():
        log_dir = log_dir / "logs"
    for log_path in sorted(log_dir.glob("log_rank[0-9]*.txt")):
        if not re.fullmatch(r"log_rank\d+\.txt", log_path.name):
            continue
        current_step = 0
        with log_path.open("r", errors="replace") as fh:
            for line in fh:
                step_match = STEP_RE.search(line)
                if step_match:
                    current_step = int(step_match.group(1))
                    if current_step > step:
                        break
                    continue
                if current_step != step:
                    continue
                match = START_RE.search(line)
                if not match:
                    continue
                key_text, h_text, n_text = match.groups()
                h_site = int(h_text)
                n_site = int(n_text)
                key = normalize_key(ast.literal_eval(key_text))
                move = (h_site, n_site)
                if move in jobs and jobs[move] != key:
                    duplicates.append(move)
                jobs[move] = key
    if duplicates:
        dup_preview = ", ".join(f"{h}->{n}" for h, n in duplicates[:10])
        raise RuntimeError(f"monolith logged duplicate moves with different keys: {dup_preview}")
    return jobs


def compute_refactor_keys(
    rows: list[dict[str, str]],
    *,
    nx: int,
    ny: int,
    nz: int,
    lattice_a: float,
    radius_a: float,
    pos_bin_a: float,
    hop_bin_a: float,
    mode: str,
) -> dict[tuple[int, int], Any]:
    sites = generate_all_interstitial_sites(nx, ny, nz, lattice_a)["tetra"]
    h_sites = sorted({int(row["h_site"]) for row in rows})
    h_positions = np.asarray(sites[h_sites], dtype=float)
    box = np.array([nx, ny, nz], dtype=float) * lattice_a
    h_tree = build_hydrogen_kdtree(h_positions, box=box)

    out: dict[tuple[int, int], Any] = {}
    for row in rows:
        h_site = int(row["h_site"])
        n_site = int(row["n_site"])
        key = make_env_key(
            sites[h_site],
            sites[n_site],
            h_positions,
            h_tree=h_tree,
            box=box,
            radius_A=radius_a,
            pos_bin_A=pos_bin_a,
            hop_bin_A=hop_bin_a,
            mode=mode,
        )
        out[(h_site, n_site)] = normalize_key(key)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=Path, default=Path("rates_allranks.csv"))
    parser.add_argument("--monolith-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=30)
    parser.add_argument("--nz", type=int, default=30)
    parser.add_argument("--lattice-a", type=float, default=2.8601)
    parser.add_argument("--radius-a", type=float, default=5.0)
    parser.add_argument("--pos-bin-a", type=float, default=0.10)
    parser.add_argument("--hop-bin-a", type=float, default=0.02)
    parser.add_argument("--mode", default="env_plus_dir")
    parser.add_argument("--out-md", type=Path, default=Path("env_key_parity_diagnostics.md"))
    parser.add_argument("--out-csv", type=Path, default=Path("env_key_parity_mismatches.csv"))
    args = parser.parse_args()

    rows = load_step_rows(args.rates, args.step)
    refactor_keys = compute_refactor_keys(
        rows,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lattice_a=args.lattice_a,
        radius_a=args.radius_a,
        pos_bin_a=args.pos_bin_a,
        hop_bin_a=args.hop_bin_a,
        mode=args.mode,
    )
    monolith_keys = parse_monolith_logged_jobs(args.monolith_dir, args.step)

    all_moves = set(refactor_keys) | set(monolith_keys)
    common = set(refactor_keys) & set(monolith_keys)
    mismatches = [move for move in sorted(common) if refactor_keys[move] != monolith_keys[move]]

    refactor_key_counts = Counter(refactor_keys.values())
    monolith_key_counts = Counter(monolith_keys.values())
    mismatch_rows: list[dict[str, Any]] = []
    mismatch_reason_counts: Counter[str] = Counter()

    for h_site, n_site in mismatches:
        left = refactor_keys[(h_site, n_site)]
        right = monolith_keys[(h_site, n_site)]
        reason = "different_hop" if left[-1] != right[-1] else "different_environment"
        mismatch_reason_counts[reason] += 1
        mismatch_rows.append(
            {
                "h_site": h_site,
                "n_site": n_site,
                "reason": reason,
                "refactor_key": repr(left),
                "monolith_key": repr(right),
                "refactor_cluster_size": refactor_key_counts[left],
                "monolith_cluster_size": monolith_key_counts[right],
            }
        )

    refactor_groups: defaultdict[Any, set[tuple[int, int]]] = defaultdict(set)
    monolith_groups: defaultdict[Any, set[tuple[int, int]]] = defaultdict(set)
    for move, key in refactor_keys.items():
        refactor_groups[key].add(move)
    for move, key in monolith_keys.items():
        monolith_groups[key].add(move)

    same_key_same_moves = 0
    same_key_different_moves = 0
    for key in set(refactor_groups) & set(monolith_groups):
        if refactor_groups[key] == monolith_groups[key]:
            same_key_same_moves += 1
        else:
            same_key_different_moves += 1

    write_csv(
        args.out_csv,
        mismatch_rows,
        [
            "h_site",
            "n_site",
            "reason",
            "refactor_key",
            "monolith_key",
            "refactor_cluster_size",
            "monolith_cluster_size",
        ],
    )

    md = [
        "# Env-Key Parity Diagnostics",
        "",
        f"Step: `{args.step}`",
        "",
        "## Summary",
        "",
        f"- Refactor moves: `{len(refactor_keys)}`",
        f"- Monolith logged NEB moves: `{len(monolith_keys)}`",
        f"- Common moves: `{len(common)}`",
        f"- Only refactor moves: `{len(set(refactor_keys) - set(monolith_keys))}`",
        f"- Only monolith logged moves: `{len(set(monolith_keys) - set(refactor_keys))}`",
        f"- Exact key matches on common moves: `{len(common) - len(mismatches)}`",
        f"- Key mismatches on common moves: `{len(mismatches)}`",
        f"- Refactor unique keys across all rate rows: `{len(refactor_key_counts)}`",
        f"- Monolith unique logged keys: `{len(monolith_key_counts)}`",
        f"- Shared key values: `{len(set(refactor_key_counts) & set(monolith_key_counts))}`",
        f"- Shared keys with identical move clusters: `{same_key_same_moves}`",
        f"- Shared keys with different move clusters: `{same_key_different_moves}`",
        f"- Mismatch reasons: `{dict(mismatch_reason_counts)}`",
        "",
        "## Largest Refactor Clusters",
        "",
        "| cluster size | key |",
        "| ---: | --- |",
    ]
    for key, count in refactor_key_counts.most_common(10):
        md.append(f"| {count} | `{repr(key)}` |")

    md.extend(["", "## Largest Monolith Logged Clusters", "", "| cluster size | key |", "| ---: | --- |"])
    for key, count in monolith_key_counts.most_common(10):
        md.append(f"| {count} | `{repr(key)}` |")

    md.extend(["", "## First Mismatches", "", "| h_site | n_site | reason |", "| ---: | ---: | --- |"])
    for row in mismatch_rows[:20]:
        md.append(f"| {row['h_site']} | {row['n_site']} | `{row['reason']}` |")

    args.out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
