#!/usr/bin/env python3
"""Analyze absorbing-sink H deletions from compiled Devanathan diagnostics."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--step-bin", type=int, default=500_000)
    parser.add_argument("--time-bin-ns", type=float, default=0.25)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--cross-section-a2", type=float, default=(20 * 2.8601) ** 2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []
    samples: list[tuple[int, float, int, int]] = []
    step_counts: dict[int, int] = {}
    time_counts: dict[int, int] = {}
    first_time = math.nan
    last_time = math.nan
    last_step = -1
    total_removed = 0

    with args.input.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            step = int(row["step"])
            time_s = float(row["time_s"])
            removed = int(row["removed_this_step"])
            total_h = int(row["total_occupied"])
            sink_occupied = int(row["sink_occupied"])
            if math.isnan(first_time):
                first_time = time_s
            last_time = time_s
            last_step = step
            if step % args.sample_every == 0 or removed:
                samples.append((step, time_s, total_h, total_removed + removed))
            if removed <= 0:
                continue
            for _ in range(removed):
                total_removed += 1
                previous_time = float(events[-1]["time_s"]) if events else 0.0
                events.append(
                    {
                        "event_index": total_removed,
                        "step": step,
                        "time_s": f"{time_s:.12e}",
                        "time_ns": f"{time_s * 1e9:.9f}",
                        "waiting_time_s": f"{time_s - previous_time:.12e}",
                        "waiting_time_ns": f"{(time_s - previous_time) * 1e9:.9f}",
                        "deleted_sink_sites": row.get("deleted_sink_sites", ""),
                        "total_occupied_after_update": total_h,
                        "sink_occupied_after_update": sink_occupied,
                    }
                )
            step_counts[step // args.step_bin] = step_counts.get(step // args.step_bin, 0) + removed
            time_idx = int(math.floor((time_s * 1e9) / args.time_bin_ns))
            time_counts[time_idx] = time_counts.get(time_idx, 0) + removed

    event_fields = [
        "event_index", "step", "time_s", "time_ns", "waiting_time_s", "waiting_time_ns",
        "deleted_sink_sites", "total_occupied_after_update", "sink_occupied_after_update",
    ]
    write_csv(args.out_dir / "sink_deletion_events.csv", event_fields, events)

    step_rows: list[dict[str, object]] = []
    n_step_bins = last_step // args.step_bin + 1
    for idx in range(n_step_bins):
        start = idx * args.step_bin
        end = min((idx + 1) * args.step_bin - 1, last_step)
        count = step_counts.get(idx, 0)
        step_rows.append({"start_step": start, "end_step": end, "deleted_H": count})
    write_csv(args.out_dir / "sink_deletions_by_step_bin.csv", ["start_step", "end_step", "deleted_H"], step_rows)

    time_rows: list[dict[str, object]] = []
    final_ns = last_time * 1e9
    n_time_bins = int(math.floor(final_ns / args.time_bin_ns)) + 1
    for idx in range(n_time_bins):
        start_ns = idx * args.time_bin_ns
        end_ns = min((idx + 1) * args.time_bin_ns, final_ns)
        duration_s = max(0.0, (end_ns - start_ns) * 1e-9)
        count = time_counts.get(idx, 0)
        flux = count / args.cross_section_a2 / duration_s if duration_s > 0 else 0.0
        time_rows.append(
            {
                "start_time_ns": f"{start_ns:.6f}",
                "end_time_ns": f"{end_ns:.6f}",
                "deleted_H": count,
                "deletion_rate_H_per_s": f"{count / duration_s:.8e}" if duration_s else "0",
                "flux_H_per_A2_s": f"{flux:.8e}",
            }
        )
    write_csv(
        args.out_dir / "sink_deletions_by_time_bin.csv",
        ["start_time_ns", "end_time_ns", "deleted_H", "deletion_rate_H_per_s", "flux_H_per_A2_s"],
        time_rows,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event_times = [float(row["time_ns"]) for row in events]
    event_cumulative = list(range(1, len(events) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step([0.0] + event_times, [0] + event_cumulative, where="post")
    ax.set(xlabel="KMC time (ns)", ylabel="Cumulative H deleted at sink", title="Absorbing-sink H arrivals")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "cumulative_sink_deletions_vs_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    centers = [(float(r["start_time_ns"]) + float(r["end_time_ns"])) / 2 for r in time_rows]
    counts = [int(r["deleted_H"]) for r in time_rows]
    ax.bar(centers, counts, width=args.time_bin_ns * 0.9)
    ax.set(xlabel="KMC time (ns)", ylabel=f"H deleted per {args.time_bin_ns:g} ns", title="Sink deletions by time interval")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "sink_deletions_by_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([t * 1e9 for _, t, _, _ in samples], [h for _, _, h, _ in samples], linewidth=1)
    for t in event_times:
        ax.axvline(t, color="tab:red", alpha=0.15, linewidth=0.8)
    ax.set(xlabel="KMC time (ns)", ylabel="H atoms in lattice", title="Hydrogen population and sink-arrival times")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "hydrogen_population_with_sink_events.png", dpi=180)
    plt.close(fig)

    elapsed_s = last_time - first_time
    global_flux = total_removed / args.cross_section_a2 / last_time if last_time > 0 else 0.0
    waiting_ns = [float(row["waiting_time_ns"]) for row in events[1:]]
    mean_wait = sum(waiting_ns) / len(waiting_ns) if waiting_ns else math.nan
    lines = [
        f"# Sink Deletion Study Through {last_step + 1:,} KMC Steps",
        "",
        f"- final_step: {last_step}",
        f"- final_KMC_time_ns: {last_time * 1e9:.9f}",
        f"- total_H_deleted_at_sink: {total_removed}",
        f"- first_sink_arrival_ns: {event_times[0]:.9f}" if event_times else "- first_sink_arrival_ns: none",
        f"- last_sink_arrival_ns: {event_times[-1]:.9f}" if event_times else "- last_sink_arrival_ns: none",
        f"- mean_interarrival_time_ns_excluding_initial_delay: {mean_wait:.9f}" if waiting_ns else "- mean_interarrival_time_ns_excluding_initial_delay: n/a",
        f"- average_sink_flux_H_per_A2_s: {global_flux:.8e}",
        f"- cross_section_A2: {args.cross_section_a2:.8f}",
        "",
        "The sink is absorbing. `sink_occupied` is therefore normally zero because H atoms are deleted in the same boundary update in which they are detected. The event count is obtained by summing `removed_this_step`; segment-local `cumulative_removed` values are not summed directly.",
        "",
        "## Event Times",
        "",
        "| Event | Step | Time (ns) | Wait since previous (ns) | Site |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in events:
        lines.append(
            f"| {row['event_index']} | {row['step']} | {row['time_ns']} | {row['waiting_time_ns']} | {row['deleted_sink_sites']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if events:
        lines.append(
            "Sink arrivals are discrete first-passage events rather than a continuous occupancy signal. The cumulative curve should be interpreted as transmitted-H count versus physical KMC time; plateaus are waiting periods with no H reaching the absorbing boundary."
        )
    else:
        lines.append("No H reached the absorbing sink during the compiled interval.")
    (args.out_dir / "sink_deletion_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"events={total_removed} final_step={last_step} final_time_ns={last_time * 1e9:.9f}")
    print(f"out_dir={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
