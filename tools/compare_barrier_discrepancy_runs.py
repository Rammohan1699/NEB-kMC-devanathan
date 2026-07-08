#!/usr/bin/env python3
"""Compare paired lammps-only and ASE-NEB pulse diagnostic runs."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-run", type=Path, required=True)
    parser.add_argument("--ase-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--large-delta-e-v", type=float, default=0.05)
    return parser.parse_args()


def first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def finite_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def int_or_none(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def rate_rows_by_hop(rows: list[dict[str, str]]) -> dict[tuple[int, int, int], dict[str, str]]:
    out: dict[tuple[int, int, int], dict[str, str]] = {}
    for row in rows:
        step = int_or_none(row.get("step"))
        h_site = int_or_none(row.get("h_site"))
        n_site = int_or_none(row.get("n_site"))
        barrier = finite_float(row.get("barrier_eV"))
        if step is None or h_site is None or n_site is None or barrier is None:
            continue
        key = (step, h_site, n_site)
        existing = out.get(key)
        if existing is None or int_or_none(existing.get("selected")) != 1:
            out[key] = row
    return out


def selected_by_step(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        step = int_or_none(row.get("step"))
        h_site = int_or_none(row.get("h_site"))
        n_site = int_or_none(row.get("n_site"))
        if step is None or h_site is None or n_site is None:
            continue
        out[step] = row
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {
            "count": 0,
            "mean": "",
            "median": "",
            "max": "",
            "min": "",
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
        "min": min(values),
    }


def last_devanathan_row(run: Path) -> dict[str, str] | None:
    rows = read_csv(
        first_existing(
            run,
            (
                "diagnostics/kmc_devanathan.csv",
                "kmc_devanathan.csv",
            ),
        )
    )
    return rows[-1] if rows else None


def compare_rates(
    lammps_rows: list[dict[str, str]],
    ase_rows: list[dict[str, str]],
    *,
    large_delta_ev: float,
) -> tuple[list[dict[str, object]], list[float]]:
    lammps_by_key = rate_rows_by_hop(lammps_rows)
    ase_by_key = rate_rows_by_hop(ase_rows)
    common_keys = sorted(set(lammps_by_key) & set(ase_by_key))
    rows: list[dict[str, object]] = []
    deltas: list[float] = []
    for step, h_site, n_site in common_keys:
        lmp = lammps_by_key[(step, h_site, n_site)]
        ase = ase_by_key[(step, h_site, n_site)]
        lmp_barrier = finite_float(lmp.get("barrier_eV"))
        ase_barrier = finite_float(ase.get("barrier_eV"))
        if lmp_barrier is None or ase_barrier is None:
            continue
        delta = ase_barrier - lmp_barrier
        abs_delta = abs(delta)
        deltas.append(abs_delta)
        rows.append(
            {
                "step": step,
                "h_site": h_site,
                "n_site": n_site,
                "from_x_A": lmp.get("from_x_A", ""),
                "to_x_A": lmp.get("to_x_A", ""),
                "dx_A": lmp.get("dx_A", ""),
                "lammps_barrier_eV": f"{lmp_barrier:.12g}",
                "ase_barrier_eV": f"{ase_barrier:.12g}",
                "delta_ase_minus_lammps_eV": f"{delta:.12g}",
                "abs_delta_eV": f"{abs_delta:.12g}",
                "large_delta": int(abs_delta >= large_delta_ev),
                "lammps_source": lmp.get("source", ""),
                "ase_source": ase.get("source", ""),
                "lammps_status": lmp.get("status", ""),
                "ase_status": ase.get("status", ""),
                "lammps_selected": lmp.get("selected", ""),
                "ase_selected": ase.get("selected", ""),
            }
        )
    rows.sort(key=lambda row: (-float(row["abs_delta_eV"]), int(row["step"]), int(row["h_site"]), int(row["n_site"])))
    return rows, deltas


def compare_selected(
    lammps_rows: list[dict[str, str]],
    ase_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    lammps_by_step = selected_by_step(lammps_rows)
    ase_by_step = selected_by_step(ase_rows)
    steps = sorted(set(lammps_by_step) | set(ase_by_step))
    rows: list[dict[str, object]] = []
    for step in steps:
        lmp = lammps_by_step.get(step, {})
        ase = ase_by_step.get(step, {})
        lmp_h = int_or_none(lmp.get("h_site"))
        lmp_n = int_or_none(lmp.get("n_site"))
        ase_h = int_or_none(ase.get("h_site"))
        ase_n = int_or_none(ase.get("n_site"))
        same = lmp_h is not None and lmp_h == ase_h and lmp_n == ase_n
        rows.append(
            {
                "step": step,
                "same_selected_hop": int(bool(same)),
                "lammps_h_site": "" if lmp_h is None else lmp_h,
                "lammps_n_site": "" if lmp_n is None else lmp_n,
                "ase_h_site": "" if ase_h is None else ase_h,
                "ase_n_site": "" if ase_n is None else ase_n,
                "lammps_barrier_eV": lmp.get("barrier_eV", ""),
                "ase_barrier_eV": ase.get("barrier_eV", ""),
                "lammps_from_x_A": lmp.get("from_x_A", ""),
                "lammps_to_x_A": lmp.get("to_x_A", ""),
                "ase_from_x_A": ase.get("from_x_A", ""),
                "ase_to_x_A": ase.get("to_x_A", ""),
                "lammps_total_time_s": lmp.get("total_time_s", ""),
                "ase_total_time_s": ase.get("total_time_s", ""),
            }
        )
    return rows


def write_summary(
    path: Path,
    *,
    lammps_run: Path,
    ase_run: Path,
    rate_rows: list[dict[str, object]],
    deltas: list[float],
    selected_rows: list[dict[str, object]],
    large_delta_ev: float,
) -> None:
    stats = summarize(deltas)
    def fmt_stat(name: str) -> str:
        value = stats[name]
        return "n/a" if value == "" else f"{float(value):.8g}"

    same_selected = sum(int(row["same_selected_hop"]) for row in selected_rows)
    compared_selected = sum(
        1
        for row in selected_rows
        if row["lammps_h_site"] != "" and row["ase_h_site"] != ""
    )
    lmp_dev = last_devanathan_row(lammps_run)
    ase_dev = last_devanathan_row(ase_run)
    large_count = sum(int(row["large_delta"]) for row in rate_rows)
    top_rows = rate_rows[:10]

    lines = [
        "# Barrier Discrepancy Pulse Comparison",
        "",
        f"- LAMMPS-only run: `{lammps_run}`",
        f"- ASE-NEB run: `{ase_run}`",
        f"- Common finite candidate hops: {stats['count']}",
        f"- Mean |delta barrier|: {fmt_stat('mean')} eV",
        f"- Median |delta barrier|: {fmt_stat('median')} eV",
        f"- Max |delta barrier|: {fmt_stat('max')} eV",
        f"- Hops with |delta| >= {large_delta_ev:g} eV: {large_count}",
        f"- Same selected hop by step: {same_selected}/{compared_selected}",
    ]
    if lmp_dev is not None:
        lines.append(
            "- LAMMPS-only final pulse state: "
            f"total_occupied={lmp_dev.get('total_occupied', '')}, "
            f"right_removed={lmp_dev.get('cumulative_removed', '')}, "
            f"left_removed={lmp_dev.get('cumulative_left_removed', '')}, "
            f"last_step={lmp_dev.get('step', '')}"
        )
    if ase_dev is not None:
        lines.append(
            "- ASE-NEB final pulse state: "
            f"total_occupied={ase_dev.get('total_occupied', '')}, "
            f"right_removed={ase_dev.get('cumulative_removed', '')}, "
            f"left_removed={ase_dev.get('cumulative_left_removed', '')}, "
            f"last_step={ase_dev.get('step', '')}"
        )
    lines.extend(["", "## Largest Candidate Barrier Differences", ""])
    if not top_rows:
        lines.append("No common finite candidate barriers were found.")
    else:
        lines.append("| step | hop | x from->to A | lammps eV | ase eV | delta eV |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
        for row in top_rows:
            lines.append(
                f"| {row['step']} | {row['h_site']}->{row['n_site']} | "
                f"{row['from_x_A']}->{row['to_x_A']} | "
                f"{row['lammps_barrier_eV']} | {row['ase_barrier_eV']} | "
                f"{row['delta_ase_minus_lammps_eV']} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lammps_rates = read_csv(
        first_existing(args.lammps_run, ("diagnostics/rates_allranks.csv", "rates_allranks.csv"))
    )
    ase_rates = read_csv(
        first_existing(args.ase_run, ("diagnostics/rates_allranks.csv", "rates_allranks.csv"))
    )
    lammps_selected = read_csv(
        first_existing(args.lammps_run, ("diagnostics/kmc_selected_events.csv", "kmc_selected_events.csv"))
    )
    ase_selected = read_csv(
        first_existing(args.ase_run, ("diagnostics/kmc_selected_events.csv", "kmc_selected_events.csv"))
    )

    rate_compare, deltas = compare_rates(
        lammps_rates,
        ase_rates,
        large_delta_ev=args.large_delta_e_v,
    )
    selected_compare = compare_selected(lammps_selected, ase_selected)

    write_csv(args.out_dir / "barrier_comparison_by_candidate_hop.csv", rate_compare)
    write_csv(args.out_dir / "selected_event_comparison_by_step.csv", selected_compare)
    write_summary(
        args.out_dir / "BARRIER_DISCREPANCY_PULSE_SUMMARY.md",
        lammps_run=args.lammps_run,
        ase_run=args.ase_run,
        rate_rows=rate_compare,
        deltas=deltas,
        selected_rows=selected_compare,
        large_delta_ev=args.large_delta_e_v,
    )
    print(f"out_dir={args.out_dir}")
    print(f"common_finite_candidate_hops={len(rate_compare)}")
    print(f"selected_steps_compared={len(selected_compare)}")


if __name__ == "__main__":
    main()
