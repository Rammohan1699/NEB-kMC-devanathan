#!/usr/bin/env python3
"""Analyze barrier reuse and deduplication in ``rates_allranks.csv``.

The KMC driver can collapse many candidate hops onto shared environment keys so
only unique cache misses require NEB calculations. This diagnostic reads one or
more all-rank rates files for a chosen step, groups finite barriers by rounded
value, and reports how much reuse is present along with status/source counts.
It is intended for checking cache-dedup behavior after scheduler or environment
key changes.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class RateRow:
    step: int
    rank: int
    h_site: int
    n_site: int
    barrier_ev: float
    rate_hz: float
    source: str
    env_kind: str
    status: str


def load_rates(path: Path, step: int) -> list[RateRow]:
    rows: list[RateRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            try:
                row_step = int(raw.get("step", ""))
            except ValueError:
                continue
            if row_step != step:
                if rows:
                    break
                continue
            try:
                rows.append(
                    RateRow(
                        step=row_step,
                        rank=int(raw.get("rank", 0)),
                        h_site=int(raw.get("h_site", 0)),
                        n_site=int(raw.get("n_site", 0)),
                        barrier_ev=float(raw.get("barrier_eV", "nan")),
                        rate_hz=float(raw.get("rate_Hz", "nan")),
                        source=str(raw.get("source", "")),
                        env_kind=str(raw.get("env_kind", "")),
                        status=str(raw.get("status", "")),
                    )
                )
            except ValueError:
                continue
    return rows


def finite_rows(rows: Iterable[RateRow]) -> list[RateRow]:
    return [row for row in rows if math.isfinite(row.barrier_ev)]


def grouped_by_barrier(rows: Iterable[RateRow], decimals: int) -> dict[float, list[RateRow]]:
    groups: dict[float, list[RateRow]] = defaultdict(list)
    for row in finite_rows(rows):
        groups[round(row.barrier_ev, decimals)].append(row)
    return dict(groups)


def summarize(label: str, rows: list[RateRow], decimals: int) -> dict[str, object]:
    finite = finite_rows(rows)
    groups = grouped_by_barrier(rows, decimals)
    sizes = [len(group) for group in groups.values()]
    barriers = [row.barrier_ev for row in finite]
    status_counts = Counter(row.status for row in rows)
    candidate_counts = Counter(Counter(row.h_site for row in rows).values())

    return {
        "label": label,
        "rows": len(rows),
        "finite": len(finite),
        "nonfinite": len(rows) - len(finite),
        "status_counts": dict(status_counts),
        "unique_barriers": len(groups),
        "reuse_factor": (len(finite) / len(groups)) if groups else 0.0,
        "cluster_min": min(sizes) if sizes else 0,
        "cluster_median": statistics.median(sizes) if sizes else 0.0,
        "cluster_mean": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "cluster_max": max(sizes) if sizes else 0,
        "clusters_ge_2": sum(1 for size in sizes if size >= 2),
        "clusters_ge_6": sum(1 for size in sizes if size >= 6),
        "clusters_ge_12": sum(1 for size in sizes if size >= 12),
        "barrier_min": min(barriers) if barriers else float("nan"),
        "barrier_mean": (sum(barriers) / len(barriers)) if barriers else float("nan"),
        "barrier_max": max(barriers) if barriers else float("nan"),
        "candidate_count_histogram": dict(sorted(candidate_counts.items())),
        "top_clusters": sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[:10],
    }


def matched_comparison(left: list[RateRow], right: list[RateRow]) -> dict[str, object]:
    left_map = {(row.h_site, row.n_site): row for row in left}
    right_map = {(row.h_site, row.n_site): row for row in right}
    common = sorted(set(left_map) & set(right_map))
    only_left = sorted(set(left_map) - set(right_map))
    only_right = sorted(set(right_map) - set(left_map))

    diffs = []
    for key in common:
        lval = left_map[key].barrier_ev
        rval = right_map[key].barrier_ev
        if math.isfinite(lval) and math.isfinite(rval):
            diffs.append((abs(lval - rval), key, lval, rval))
    diffs.sort(reverse=True)
    absvals = [item[0] for item in diffs]
    return {
        "common": len(common),
        "only_left": len(only_left),
        "only_right": len(only_right),
        "mae": (sum(absvals) / len(absvals)) if absvals else None,
        "median_abs": statistics.median(absvals) if absvals else None,
        "max_abs": max(absvals) if absvals else None,
        "largest_diffs": diffs[:10],
        "only_left_examples": only_left[:10],
        "only_right_examples": only_right[:10],
    }


def fmt_float(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.12g}"
    return str(value)


def write_markdown(
    path: Path,
    left_summary: dict[str, object],
    right_summary: Optional[dict[str, object]],
    comparison: Optional[dict[str, object]],
    decimals: int,
) -> None:
    lines: list[str] = []
    lines.append("# Barrier Dedupe Diagnostics")
    lines.append("")
    lines.append(f"Barrier grouping tolerance: rounded to `{decimals}` decimal places.")
    lines.append("")

    summaries = [left_summary] + ([right_summary] if right_summary is not None else [])
    lines.append("## Summary")
    lines.append("")
    lines.append("| run | rows | finite | nonfinite | unique rounded barriers | reuse factor | barrier min | barrier mean | barrier max | max cluster |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        assert summary is not None
        lines.append(
            "| {label} | {rows} | {finite} | {nonfinite} | {unique_barriers} | {reuse_factor} | "
            "{barrier_min} | {barrier_mean} | {barrier_max} | {cluster_max} |".format(
                label=summary["label"],
                rows=summary["rows"],
                finite=summary["finite"],
                nonfinite=summary["nonfinite"],
                unique_barriers=summary["unique_barriers"],
                reuse_factor=fmt_float(summary["reuse_factor"]),
                barrier_min=fmt_float(summary["barrier_min"]),
                barrier_mean=fmt_float(summary["barrier_mean"]),
                barrier_max=fmt_float(summary["barrier_max"]),
                cluster_max=summary["cluster_max"],
            )
        )
    lines.append("")

    for summary in summaries:
        assert summary is not None
        lines.append(f"## {summary['label']} Clusters")
        lines.append("")
        lines.append(f"Status counts: `{summary['status_counts']}`")
        lines.append("")
        lines.append(f"Candidate count histogram per H: `{summary['candidate_count_histogram']}`")
        lines.append("")
        lines.append("| rounded barrier | cluster size | example moves |")
        lines.append("| ---: | ---: | --- |")
        for barrier, rows in summary["top_clusters"]:  # type: ignore[index]
            examples = ", ".join(f"{row.h_site}->{row.n_site}" for row in rows[:6])
            lines.append(f"| {barrier:.12g} | {len(rows)} | `{examples}` |")
        lines.append("")

    if comparison is not None:
        lines.append("## Matched-Hop Comparison")
        lines.append("")
        lines.append(f"Common `(h_site,n_site)` moves: `{comparison['common']}`")
        lines.append(f"Only left: `{comparison['only_left']}`")
        lines.append(f"Only reference: `{comparison['only_right']}`")
        lines.append(f"MAE on common moves: `{fmt_float(comparison['mae'])}`")
        lines.append(f"Median abs diff: `{fmt_float(comparison['median_abs'])}`")
        lines.append(f"Max abs diff: `{fmt_float(comparison['max_abs'])}`")
        lines.append("")
        if comparison["only_left_examples"]:
            lines.append(f"Left-only examples: `{comparison['only_left_examples']}`")
        if comparison["only_right_examples"]:
            lines.append(f"Reference-only examples: `{comparison['only_right_examples']}`")
        lines.append("")
        if comparison["common"] == 0:
            lines.append(
                "No exact matched hops were found. This usually means the two runs use different "
                "target site IDs for physically equivalent periodic hops, or the initial neighbor "
                "topology differs before environment-key parity is evaluated."
            )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cluster_csv(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rounded_barrier_eV", "cluster_size", "examples"])
        for barrier, rows in summary["top_clusters"]:  # type: ignore[index]
            writer.writerow([f"{barrier:.12g}", len(rows), " ".join(f"{row.h_site}->{row.n_site}" for row in rows[:20])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", default="rates_allranks.csv", help="Rates CSV to analyze.")
    parser.add_argument("--reference-rates", default=None, help="Optional reference rates CSV.")
    parser.add_argument("--step", type=int, default=0, help="KMC step to analyze.")
    parser.add_argument("--round-decimals", type=int, default=12, help="Barrier rounding decimals for grouping.")
    parser.add_argument("--out", default="dedupe_diagnostics.md", help="Markdown report path.")
    parser.add_argument("--cluster-csv", default="dedupe_clusters.csv", help="Top-cluster CSV path.")
    args = parser.parse_args()

    rates_path = Path(args.rates)
    if not rates_path.exists() and rates_path == Path("rates_allranks.csv"):
        moved = Path("diagnostics") / "rates_allranks.csv"
        if moved.exists():
            rates_path = moved
    left_rows = load_rates(rates_path, args.step)
    left_summary = summarize(rates_path.name, left_rows, args.round_decimals)

    right_summary = None
    comparison = None
    if args.reference_rates:
        right_path = Path(args.reference_rates)
        right_rows = load_rates(right_path, args.step)
        right_summary = summarize(right_path.name, right_rows, args.round_decimals)
        comparison = matched_comparison(left_rows, right_rows)

    write_markdown(Path(args.out), left_summary, right_summary, comparison, args.round_decimals)
    write_cluster_csv(Path(args.cluster_csv), left_summary)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.cluster_csv}")


if __name__ == "__main__":
    main()
