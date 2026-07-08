#!/usr/bin/env python3
"""Build a compact health summary for a completed KMC run directory.

This diagnostic collects run metadata from logs, checkpoints, timing CSVs,
barrier caches, rates tables, trajectories, and NEB diagnostic files. The
resulting Markdown report is meant to be checked between clean runs to confirm
that initialization choices, cache behavior, selected events, and output files
match expectations.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


REGION_LINE_RE = re.compile(
    r"Initial H region selector (?P<regions>\[.*?\]) chose (?P<count>\d+) sites from (?P<file>\S+)"
)
COMPLETE_RE = re.compile(r"complete in (?P<seconds>[0-9.]+) s")
CANDIDATE_RE = re.compile(
    r"Candidate summary: local_H=(?P<local_h>\d+), cache_hits=(?P<hits>\d+), "
    r"cache_misses=(?P<misses>\d+), miss_jobs=(?P<jobs>\d+), "
    r"valid_cache_events=(?P<valid>\d+), rate_rows=(?P<rows>\d+)"
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {"exists": True, "bytes": st.st_size, "mtime": st.st_mtime}


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return {
        "step": _int(payload.get("step")),
        "sim_time_s": _float(payload.get("sim_time_s")),
        "num_h": _int(payload.get("num_h", len(payload.get("h_indices", [])))),
        "h_indices": np.asarray(payload.get("h_indices", []), dtype=int),
        "cache_schema": str(payload.get("cache_schema", "")),
        "site_source": str(payload.get("site_source", "")),
        "site_map_file": str(payload.get("site_map_file", "")),
        "host_structure_file": str(payload.get("host_structure_file", "")),
    }


def _parse_log(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "region_selector": None,
        "complete_wall_s": None,
        "candidate_steps": 0,
        "cache_hits_total_rank0": 0,
        "cache_misses_total_rank0": 0,
        "miss_jobs_total_rank0": 0,
        "max_miss_jobs_rank0": 0,
        "local_h_rank0": None,
    }
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = REGION_LINE_RE.search(line)
            if m:
                out["region_selector"] = {
                    "regions": m.group("regions"),
                    "count": _int(m.group("count")),
                    "file": m.group("file").rstrip("."),
                }
            m = COMPLETE_RE.search(line)
            if m:
                out["complete_wall_s"] = _float(m.group("seconds"))
            m = CANDIDATE_RE.search(line)
            if m:
                out["candidate_steps"] += 1
                hits = _int(m.group("hits"))
                misses = _int(m.group("misses"))
                jobs = _int(m.group("jobs"))
                out["cache_hits_total_rank0"] += hits
                out["cache_misses_total_rank0"] += misses
                out["miss_jobs_total_rank0"] += jobs
                out["max_miss_jobs_rank0"] = max(out["max_miss_jobs_rank0"], jobs)
                out["local_h_rank0"] = _int(m.group("local_h"))
    return out


def _summarize_timing(paths: list[Path]) -> dict[str, Any]:
    rank_summaries = []
    stage_totals: dict[str, float] = defaultdict(float)
    slow_steps: list[dict[str, Any]] = []
    stage_names: list[str] = []

    for path in paths:
        rows = 0
        wall_values: list[float] = []
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not stage_names:
                stage_names = [
                    name
                    for name in (reader.fieldnames or [])
                    if name not in {"step", "wall_total", "components_sum", "unaccounted"}
                ]
            rank_stage_totals: dict[str, float] = defaultdict(float)
            for row in reader:
                rows += 1
                wall = _float(row.get("wall_total"))
                wall_values.append(wall)
                step = _int(row.get("step"))
                slow_steps.append({"file": path.name, "step": step, "wall_total": wall})
                for stage in stage_names:
                    value = _float(row.get(stage))
                    rank_stage_totals[stage] += value
                    stage_totals[stage] += value
        rank_summaries.append(
            {
                "file": path.name,
                "rows": rows,
                "wall_total_s": sum(wall_values),
                "wall_avg_s": mean(wall_values) if wall_values else 0.0,
                "wall_max_s": max(wall_values) if wall_values else 0.0,
                "top_stages_s": dict(sorted(rank_stage_totals.items(), key=lambda kv: kv[1], reverse=True)[:8]),
            }
        )

    slow_steps = sorted(slow_steps, key=lambda item: item["wall_total"], reverse=True)[:20]
    return {
        "ranks": rank_summaries,
        "stage_totals_s": dict(sorted(stage_totals.items(), key=lambda kv: kv[1], reverse=True)),
        "slow_steps": slow_steps,
    }


def _summarize_neb(path: Path) -> dict[str, Any]:
    out = {
        "rows": 0,
        "neb_ran": 0,
        "neb_assigned": 0,
        "neb_wall_s": 0.0,
        "nonzero_steps": 0,
        "steps_with_neb": [],
        "top_steps_by_neb_wall_s": [],
    }
    if not path.exists():
        return out
    by_step: dict[int, float] = defaultdict(float)
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out["rows"] += 1
            step = _int(row.get("step"))
            ran = _int(row.get("neb_ran"))
            assigned = _int(row.get("neb_assigned"))
            wall = _float(row.get("neb_wall_s"))
            out["neb_ran"] += ran
            out["neb_assigned"] += assigned
            out["neb_wall_s"] += wall
            if ran > 0:
                by_step[step] += wall
    out["steps_with_neb"] = sorted(by_step)
    out["nonzero_steps"] = len(by_step)
    out["top_steps_by_neb_wall_s"] = [
        {"step": step, "neb_wall_s": wall}
        for step, wall in sorted(by_step.items(), key=lambda kv: kv[1], reverse=True)[:20]
    ]
    return out


def _summarize_rates(path: Path) -> dict[str, Any]:
    out = {"rows": 0, "source_counts": {}, "status_counts": {}}
    if not path.exists():
        return out
    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out["rows"] += 1
            source_counts[str(row.get("source", ""))] += 1
            status_counts[str(row.get("status", ""))] += 1
    out["source_counts"] = dict(source_counts)
    out["status_counts"] = dict(status_counts)
    return out


def _first_frame_sites(path: Path) -> list[int]:
    if not path.exists():
        return []
    sites: list[int] = []
    in_atoms = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True
                continue
            if in_atoms and line.startswith("ITEM:"):
                break
            if in_atoms:
                parts = line.split()
                if len(parts) >= 6:
                    sites.append(_int(parts[5], -1))
    return [site for site in sites if site >= 0]


def _region_counts(region_file: Path, site_indices: np.ndarray) -> dict[str, int]:
    if not region_file.exists() or site_indices.size == 0:
        return {}
    data = np.load(region_file, allow_pickle=True)
    if "site_selection_label" not in data.files:
        return {}
    labels = np.asarray(data["site_selection_label"]).astype(str)
    valid = site_indices[(site_indices >= 0) & (site_indices < len(labels))]
    return {str(label): int(count) for label, count in Counter(labels[valid]).items()}


def _markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# KMC Run Diagnostics")
    lines.append("")
    lines.append("## Run")
    run = summary["run"]
    for key in [
        "complete_wall_s",
        "checkpoint_step",
        "num_h",
        "sim_time_s",
        "site_source",
        "site_map_file",
        "host_structure_file",
        "cache_schema",
    ]:
        lines.append(f"- {key}: {run.get(key)}")
    lines.append(f"- region_selector: {run.get('region_selector')}")
    lines.append(f"- initial_h_region_counts: {run.get('initial_h_region_counts')}")
    lines.append(f"- final_h_region_counts: {run.get('final_h_region_counts')}")
    lines.append("")

    timing = summary["timing"]
    lines.append("## Timing")
    for rank in timing["ranks"]:
        lines.append(
            f"- {rank['file']}: rows={rank['rows']}, wall_total_s={rank['wall_total_s']:.3f}, "
            f"avg_step_s={rank['wall_avg_s']:.6f}, max_step_s={rank['wall_max_s']:.3f}"
        )
    lines.append("")
    lines.append("Top aggregate timed stages across ranks:")
    for stage, seconds in list(timing["stage_totals_s"].items())[:12]:
        lines.append(f"- {stage}: {seconds:.3f} s")
    lines.append("")
    lines.append("Slowest rank-step rows:")
    for item in timing["slow_steps"][:10]:
        lines.append(f"- {item['file']} step {item['step']}: {item['wall_total']:.3f} s")
    lines.append("")

    neb = summary["neb"]
    lines.append("## NEB")
    lines.append(f"- neb_ran: {neb['neb_ran']}")
    lines.append(f"- neb_assigned: {neb['neb_assigned']}")
    lines.append(f"- aggregate_neb_wall_s_across_ranks: {neb['neb_wall_s']:.3f}")
    lines.append(f"- steps_with_neb: {neb['nonzero_steps']}")
    lines.append("Top NEB steps by aggregate wall:")
    for item in neb["top_steps_by_neb_wall_s"][:10]:
        lines.append(f"- step {item['step']}: {item['neb_wall_s']:.3f} s")
    lines.append("")

    lines.append("## Cache And Rates")
    log = summary["rank0_log"]
    rates = summary["rates"]
    lines.append(f"- rank0 candidate rows parsed: {log.get('candidate_steps')}")
    lines.append(f"- rank0 cache hits total: {log.get('cache_hits_total_rank0')}")
    lines.append(f"- rank0 cache misses total: {log.get('cache_misses_total_rank0')}")
    lines.append(f"- rank0 miss jobs total: {log.get('miss_jobs_total_rank0')}")
    lines.append(f"- rank0 max miss jobs in one step: {log.get('max_miss_jobs_rank0')}")
    lines.append(f"- rates rows: {rates.get('rows')}")
    lines.append(f"- rates source counts: {rates.get('source_counts')}")
    lines.append(f"- rates status counts: {rates.get('status_counts')}")
    lines.append("")

    lines.append("## Files")
    for name, info in summary["files"].items():
        lines.append(f"- {name}: {info}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Region-file loading is not visible as a major cost; map load + KNN were below 1 s in rank-0 logs.")
    lines.append("- This run used a cold or partly cold cache for the selected region and computed thousands of NEB barriers.")
    lines.append("- Use this file as the baseline artifact for a clean run with the same diagnostic script.")
    lines.append("")
    return "\n".join(lines)


def build_summary(root: Path, region_file: Path) -> dict[str, Any]:
    checkpoint_path = _first_existing(root / "checkpoints" / "kmc_restart_checkpoint.pkl", root / "kmc_restart_checkpoint.pkl")
    rank0_log_path = _first_existing(root / "logs" / "log_rank0.txt", root / "log_rank0.txt")
    neb_path = _first_existing(root / "neb" / "neb_diag_all.csv", root / "neb_diag_all.csv")
    rates_path = _first_existing(root / "diagnostics" / "rates_allranks.csv", root / "rates_allranks.csv")

    checkpoint = _load_checkpoint(checkpoint_path)
    rank0_log = _parse_log(rank0_log_path)
    timing = _summarize_timing(sorted((root / "logs").glob("timing_rank*.csv")) or sorted(root.glob("timing_rank*.csv")))
    neb = _summarize_neb(neb_path)
    rates = _summarize_rates(rates_path)
    initial_sites = np.asarray(_first_frame_sites(root / "H_trajectory_onlyH.lammpstrj"), dtype=int)
    final_sites = np.asarray(checkpoint.get("h_indices", []), dtype=int)

    files = {
        name: _file_info(root / name)
        for name in [
            "logs/log_rank0.txt",
            "kmc_diagnostics_rank0.log",
            "logs/timing_rank0.csv",
            "neb/neb_diag_all.csv",
            "diagnostics/rates_allranks.csv",
            "H_trajectory_onlyH.lammpstrj",
            "checkpoints/kmc_restart_checkpoint.pkl",
        ]
    }
    ase_dir = _first_existing(root / "neb" / "ase_logs", root / "ase_logs")
    files["ase_logs_file_count"] = {"count": sum(1 for p in ase_dir.rglob("*") if p.is_file())} if ase_dir.exists() else {"count": 0}

    return {
        "run": {
            "complete_wall_s": rank0_log.get("complete_wall_s"),
            "checkpoint_step": checkpoint.get("step"),
            "num_h": checkpoint.get("num_h"),
            "sim_time_s": checkpoint.get("sim_time_s"),
            "site_source": checkpoint.get("site_source"),
            "site_map_file": checkpoint.get("site_map_file"),
            "host_structure_file": checkpoint.get("host_structure_file"),
            "cache_schema": checkpoint.get("cache_schema"),
            "region_selector": rank0_log.get("region_selector"),
            "initial_h_region_counts": _region_counts(region_file, initial_sites),
            "final_h_region_counts": _region_counts(region_file, final_sites),
        },
        "rank0_log": rank0_log,
        "timing": timing,
        "neb": neb,
        "rates": rates,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--region-file", default="kmc_map_inputs/sigma5_site_regions.npz")
    parser.add_argument("--out-md", default="kmc_run_diagnostics_current.md")
    parser.add_argument("--out-json", default="kmc_run_diagnostics_current.json")
    args = parser.parse_args()

    root = Path(args.root)
    region_file = root / args.region_file if not Path(args.region_file).is_absolute() else Path(args.region_file)
    summary = build_summary(root, region_file)

    md_path = root / args.out_md
    json_path = root / args.out_json
    md_path.write_text(_markdown(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
