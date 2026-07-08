#!/usr/bin/env python3
"""
Join ASE and native LAMMPS NEB validation results into one comparison table.

This helper scans validation job folders produced by ``validation.sh``. For
each unique H-site hop it finds the LAMMPS NEB log, extracts the final forward
and reverse barriers, locates the matching ASE diagnostic outputs, and writes a
compact CSV suitable for plotting or spreadsheet review. It also reports basic
agreement statistics so validation runs can quickly show whether the ASE and
LAMMPS paths are consistent.

Output columns:
  rank,step,h_site,n_site,n_H,transition_kind,
  lammps_dump_path,lammps_dump_dir,lammps_dump_frame_atom_counts,
  ase_dump_path,ase_dump_dir,ase_dump_frame_atom_counts,
  ase_initial_log,ase_mid_log,ase_final_log,
  ase_barrier_predicted_eV,ase_barrier_eV,lmp_barrier_eV,abs_diff_eV,rel_diff_pct
"""

import os
import re
import csv
import argparse
import math
import io
from statistics import mean, median
from collections import OrderedDict

# ----------------- Config / parsing helpers -----------------

PATH_RX = re.compile(r"rank(?P<rank>\d+)_step(?P<step>\d+)_h(?P<h>\d+)")
LOG_FILENAMES = ["log.run", "log.neb"]
ASE_DUMP_DIR = "ase"
ASE_DUMP_PREFERRED = (
    "dump.trajectory",
    "dump.ase.post",
    "dump.ase.post.lammpstrj",
    "dump.ase.pre",
    "dump.ase.pre.lammpstrj",
    "dump.ase",
)
ASE_POSITIONS_RX = re.compile(
    rb'"positions\.":\s*\{\s*"ndarray":\s*\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
    re.MULTILINE,
)
LMP_CONSOLIDATED_NAME = "dump.lammps.lammpstrj"

def find_validation_jobs(root):
    """
    Yield (rank:int, step:int, h:int, n_site:int, log_path:str) for each job.
    Expects job dirs: validation_dump/rankX_stepY_hZ/lammps/<n_site>/
    """
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(os.path.dirname(dirpath)) != "lammps":
            continue
        n_site_str = os.path.basename(dirpath)
        if not n_site_str.isdigit():
            continue

        # up two levels: .../rankX_stepY_hZ
        parent = os.path.dirname(os.path.dirname(dirpath))
        base = os.path.basename(parent)
        m = PATH_RX.search(base) or PATH_RX.search(parent)
        if not m:
            continue

        rank = int(m.group("rank"))
        step = int(m.group("step"))
        h    = int(m.group("h"))
        n_site = int(n_site_str)

        # prefer log.run, then log.neb
        log_path = None
        for name in LOG_FILENAMES:
            p = os.path.join(dirpath, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                log_path = p
                break
        if log_path:
            yield (rank, step, h, n_site, log_path)

def parse_last_barriers_from_log(log_path):
    """
    Return the *last* (EBF, EBR) tuple seen in a LAMMPS NEB log.
    Each value may be None if the corresponding column never appeared.
    """
    ebf_idx = None
    ebr_idx = None
    last_ebf = None
    last_ebr = None
    header_fields = None

    def is_num(tok):
        try:
            float(tok)
            return True
        except Exception:
            return False

    with open(log_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            is_header = (
                " EBF " in f" {line} "
                or " EBR " in f" {line} "
                or (line.startswith("Step") and ("EBF" in line or "EBR" in line))
            )
            if is_header:
                fields = line.split()
                upper = [s.upper() for s in fields]
                if "EBF" in upper:
                    ebf_idx = upper.index("EBF")
                else:
                    ebf_idx = None
                if "EBR" in upper:
                    ebr_idx = upper.index("EBR")
                else:
                    ebr_idx = None
                if ebf_idx is not None or ebr_idx is not None:
                    header_fields = fields
                continue

            if header_fields is None:
                continue

            parts = line.split()
            if not parts or not is_num(parts[0]):
                continue

            if ebf_idx is not None and len(parts) > ebf_idx and is_num(parts[ebf_idx]):
                try:
                    last_ebf = float(parts[ebf_idx])
                except Exception:
                    pass
            if ebr_idx is not None and len(parts) > ebr_idx and is_num(parts[ebr_idx]):
                try:
                    last_ebr = float(parts[ebr_idx])
                except Exception:
                    pass
    return last_ebf, last_ebr


def choose_lammps_barrier(ebf, ebr):
    """
    Prefer a non-zero EBR when present (LAMMPS reports the barrier in that column
    when the forward image collapses). Fall back to EBF or whichever column is
    available/non-zero.
    """
    candidates = []
    if ebr is not None and math.isfinite(ebr):
        candidates.append(("EBR", ebr))
    if ebf is not None and math.isfinite(ebf):
        candidates.append(("EBF", ebf))
    if not candidates:
        return None

    # Prefer the largest positive barrier; ties go to whichever appeared first (EBR before EBF)
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][1]

def pick_lammps_dump_path(job_dir):
    dumps = _find_dump_files(job_dir)
    if not dumps:
        return ""
    return os.path.abspath(dumps[0])

def pick_ase_dump_path(job_dir, ase_dir=ASE_DUMP_DIR):
    """
    Return the preferred ASE dump file path (e.g., trajectory or dump.ase.post) if present.
    """
    job_root = os.path.dirname(os.path.dirname(job_dir))
    ase_root = os.path.join(job_root, ase_dir)
    if not os.path.isdir(ase_root):
        return ""

    n_site_dir = os.path.join(ase_root, os.path.basename(job_dir))
    if not os.path.isdir(n_site_dir):
        return ""

    try:
        names = os.listdir(n_site_dir)
    except Exception:
        return ""

    candidates = []
    for name in names:
        if not name.lower().startswith("dump"):
            continue
        path = os.path.join(n_site_dir, name)
        if os.path.isfile(path):
            candidates.append(path)

    if not candidates:
        return ""

    priority_index = {name: idx for idx, name in enumerate(ASE_DUMP_PREFERRED)}

    candidates.sort(
        key=lambda path: (
            priority_index.get(os.path.basename(path).lower(), len(ASE_DUMP_PREFERRED)),
            path,
        )
    )
    return os.path.abspath(candidates[0])

def load_rates_csv(path):
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        return rows, rdr.fieldnames

def normalize_log_path(path):
    """
    Return an absolute path for the given ASE log entry, skipping empty/"missing" values.
    """
    if not path:
        return ""
    candidate = path.strip()
    if not candidate or candidate.lower() == "missing":
        return ""
    return os.path.abspath(candidate)


def load_ase_barriers(path):
    """
    Load the ASE barrier table (`ase-barriers.csv`).
    Mapping keys use the src_rank in that file (treated as the validation rank).
    """
    if not os.path.isfile(path):
        return {}
    result = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                src_rank = int((row.get("src_rank") or row.get("rank") or "").strip() or 0)
                step = int(row.get("step", "").strip())
                h_site = int((row.get("h") or row.get("h_site") or "").strip())
                n_site = int((row.get("n") or row.get("n_site") or "").strip())
            except Exception:
                continue

            key = (src_rank, step, h_site, n_site)
            predicted_value = None
            pred_str = (row.get("predicted_barrier") or row.get("barrier_eV") or "").strip()
            if pred_str and pred_str.lower() != "missing":
                try:
                    predicted_value = float(pred_str)
                except Exception:
                    predicted_value = None

            result[key] = {
                "src_rank": src_rank,
                "predicted": predicted_value,
                "barrier": predicted_value,
                "initial_log": normalize_log_path(row.get("initial_log", "")),
                "mid_log": normalize_log_path(row.get("mid_log", "")),
                "final_log": normalize_log_path(row.get("final_log", "")),
            }
    return result

def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# ----------------- Main build logic -----------------

def main():
    ap = argparse.ArgumentParser(description="Join unique ASE barriers with LAMMPS NEB barriers (EBF/EBR) and summarize agreement.")
    ap.add_argument("--root", default="validation_dump", help="Validation dump root directory.")
    ap.add_argument("--rates", default="validation_rates_allranks.csv", help="Input validation rates CSV.")
    ap.add_argument("--out", default="validation_compare_unique.csv", help="Output compact CSV path.")
    ap.add_argument("--tol", type=float, default=0.01, help="Tolerance (eV) for match metric (default: 0.01 eV).")
    ap.add_argument("--axis-tol", type=float, default=1e-3, help="Axis-parallel tolerance (Å) for hop alignment classification (default: 1e-3).")
    default_ase_dir = os.environ.get("VALIDATION_ASE_DIR", ASE_DUMP_DIR)
    ap.add_argument("--ase-dir", default=default_ase_dir, help="ASE dump subdirectory name (default: VALIDATION_ASE_DIR or 'ase').")
    ap.add_argument("--ase-barriers", default="ase-barriers.csv", help="ASE barrier CSV with predicted barrier/log info.")
    args = ap.parse_args()
    ase_barrier_map = load_ase_barriers(args.ase_barriers)
    print(f"[INFO] Loaded {len(ase_barrier_map)} ASE barrier rows from '{args.ase_barriers}'.")

    # 1) Parse LAMMPS logs → map[(rank,step,h,n_site)] = ebf
    print(f"[INFO] Scanning LAMMPS jobs in '{args.root}' ...")
    lmp_map = {}
    hcount_map = {}
    tkind_map = {}
    dump_path_map = {}
    ase_dump_path_map = {}
    lmp_frame_counts_map = {}
    ase_frame_counts_map = {}
    total_jobs = 0
    barrier_found = 0
    other_type_map = {}
    for (rank, step, h, n_site, log_path) in find_validation_jobs(args.root):
        total_jobs += 1
        job_key = (rank, step, h, n_site)
        ebf, ebr = parse_last_barriers_from_log(log_path)
        best_barrier = choose_lammps_barrier(ebf, ebr)
        if best_barrier is not None:
            lmp_map[job_key] = best_barrier
            barrier_found += 1
        # also try to enrich with n_H and transition_kind
        job_dir = os.path.dirname(log_path)  # .../lammps/<n_site>/
        consolidated_path = pick_lammps_consolidated_dump(job_dir)
        if consolidated_path:
            dump_path_map[job_key] = consolidated_path
            lmp_frame_counts_map[job_key] = _count_atoms_in_lammpstrj(consolidated_path)
        else:
            dump_path = pick_lammps_dump_path(job_dir)
            if dump_path:
                dump_path_map[job_key] = dump_path
        ase_dump_path = pick_ase_dump_path(job_dir, args.ase_dir)
        if ase_dump_path:
            ase_dump_path_map[job_key] = ase_dump_path
            ase_frame_counts_map[job_key] = _count_atoms_in_ase_dump(ase_dump_path)
        nH = infer_h_count_from_init(job_dir)
        if nH is not None:
            hcount_map[job_key] = nH
        tkind = classify_transition_kind(job_dir, h, n_site, axis_tol=args.axis_tol)
        if tkind:
            tkind_map[job_key] = tkind
        other_type = infer_other_h_site_type(job_dir)
        if other_type:
            other_type_map[job_key] = other_type
    print(f"[INFO] Found {barrier_found}/{total_jobs} jobs with LAMMPS barrier values.")

    # 2) Load ASE barriers. Prefer legacy rates CSV when available; otherwise
    # use the refactored neb_validation_summary_rank*.csv supplied via
    # --ase-barriers.
    uniques = []
    if os.path.isfile(args.rates):
        print(f"[INFO] Loading rates from '{args.rates}' ...")
        rows, cols = load_rates_csv(args.rates)
        needed = {"rank", "step", "h_site", "n_site", "barrier_eV", "env_kind"}
        if not needed.issubset(set(c.strip() for c in cols)):
            raise SystemExit(f"[ERROR] '{args.rates}' missing required columns {needed}. Found: {cols}")

        for r in rows:
            if (r.get("env_kind", "").strip().lower() != "unique"):
                continue
            try:
                ase_b = float(r["barrier_eV"])
            except Exception:
                continue
            k = (int(r["rank"]), int(r["step"]), int(r["h_site"]), int(r["n_site"]))
            uniques.append((k, ase_b))
    else:
        print(f"[INFO] Rates file '{args.rates}' not found; using --ase-barriers rows as ASE barriers.")
        for k, entry in ase_barrier_map.items():
            ase_b = entry.get("barrier")
            if ase_b is None:
                continue
            uniques.append((k, float(ase_b)))

    # Deduplicate preserving last occurrence
    uniq_last = OrderedDict()
    for k, ase_b in uniques:
        uniq_last[k] = ase_b

    print(f"[INFO] Unique ASE rows: {len(uniq_last)}")

    # 3) Build compact table and compute diffs
    out_rows = []
    diffs = []
    matched = 0
    for (rank, step, h, n_site), ase_b in uniq_last.items():
        key = (rank, step, h, n_site)
        lmp_b = lmp_map.get(key)
        abs_diff = ""
        rel_pct = ""

        if lmp_b is not None and math.isfinite(lmp_b):
            matched += 1
            d = abs(lmp_b - ase_b)
            abs_diff = d
            diffs.append(d)
            rel_pct = (d / max(1e-12, abs(ase_b))) * 100.0

        n_H_val = ("" if key not in hcount_map else hcount_map[key])
        tkind = tkind_map.get(key, "")
        other_type = other_type_map.get(key, "")
        if tkind and other_type and (n_H_val not in ("", None)):
            try:
                if int(n_H_val) >= 2:
                    tkind = f"{tkind} ({other_type})"
            except Exception:
                pass

        ase_entry = ase_barrier_map.get(key)
        ase_pred = ase_entry["predicted"] if ase_entry else None
        ase_initial_log = ase_entry["initial_log"] if ase_entry else ""
        ase_mid_log = ase_entry["mid_log"] if ase_entry else ""
        ase_final_log = ase_entry["final_log"] if ase_entry else ""
        ase_src_rank = ase_entry["src_rank"] if ase_entry else ""

        out_rows.append({
            "rank": rank,
            "step": step,
            "h_site": h,
            "n_site": n_site,
            "ase_src_rank": ("" if ase_src_rank == "" else str(ase_src_rank)),
            "n_H": n_H_val,
            "transition_kind": tkind,
            "lammps_dump_path": dump_path_map.get(key, ""),
            "lammps_dump_dir": _get_parent_dir(dump_path_map.get(key, "")),
            "lammps_dump_frame_atom_counts": _format_frame_counts(lmp_frame_counts_map.get(key, [])),
            "ase_dump_path": ase_dump_path_map.get(key, ""),
            "ase_dump_dir": _get_parent_dir(ase_dump_path_map.get(key, "")),
            "ase_dump_frame_atom_counts": _format_frame_counts(ase_frame_counts_map.get(key, [])),
            "ase_initial_log": ase_initial_log,
            "ase_mid_log": ase_mid_log,
            "ase_final_log": ase_final_log,
            "ase_barrier_predicted_eV": ("" if ase_pred is None else f"{ase_pred:.9f}"),
            "ase_barrier_eV": f"{ase_b:.9f}",
            "lmp_barrier_eV": ("" if lmp_b is None else f"{lmp_b:.9f}"),
            "abs_diff_eV": ("" if abs_diff == "" else f"{abs_diff:.9f}"),
            "rel_diff_pct": ("" if rel_pct == "" else f"{rel_pct:.6f}"),
            "_lmp_counts": lmp_frame_counts_map.get(key, []),
            "_ase_counts": ase_frame_counts_map.get(key, []),
        })

    # 4) Write compact CSV
    max_lmp_frames = max((len(row.get("_lmp_counts", [])) for row in out_rows), default=0)
    max_ase_frames = max((len(row.get("_ase_counts", [])) for row in out_rows), default=0)
    lammps_atoms_fields = [f"lammps_atoms_{idx + 1}" for idx in range(max_lmp_frames)]
    ase_atoms_fields = [f"ase_atoms_{idx + 1}" for idx in range(max_ase_frames)]

    for row in out_rows:
        lmp_counts = row.pop("_lmp_counts", [])
        for idx, field in enumerate(lammps_atoms_fields):
            row[field] = lmp_counts[idx] if idx < len(lmp_counts) else ""
        ase_counts = row.pop("_ase_counts", [])
        for idx, field in enumerate(ase_atoms_fields):
            row[field] = ase_counts[idx] if idx < len(ase_counts) else ""

    base_fields = [
        "rank","step","h_site","n_site",
        "ase_src_rank",
        "n_H","transition_kind",
        "lammps_dump_path","lammps_dump_dir","lammps_dump_frame_atom_counts",
        "ase_dump_path","ase_dump_dir","ase_dump_frame_atom_counts",
        "ase_initial_log","ase_mid_log","ase_final_log",
        "ase_barrier_predicted_eV",
        "ase_barrier_eV","lmp_barrier_eV","abs_diff_eV","rel_diff_pct",
    ]
    fields = base_fields.copy()
    if lammps_atoms_fields:
        idx = fields.index("ase_dump_path")
        fields[idx:idx] = lammps_atoms_fields
    if ase_atoms_fields:
        idx = fields.index("ase_initial_log")
        fields[idx:idx] = ase_atoms_fields
    out_fields = fields
    write_csv(args.out, out_fields, out_rows)
    print(f"[INFO] Wrote compact file: {args.out} (rows: {len(out_rows)})")

    # 5) Print small analysis
    total = len(uniq_last)
    within = sum(1 for d in diffs if d <= args.tol)
    match_factor = (within / max(1, matched)) * 100.0  # % of matched rows within tol

    def safe_stat(fn, data):
        return fn(data) if data else float("nan")

    mean_abs = safe_stat(mean, diffs)
    med_abs  = safe_stat(median, diffs)
    rmse     = math.sqrt(safe_stat(mean, [d*d for d in diffs])) if diffs else float("nan")

    ase_predicted_matches = sum(1 for row in out_rows if row["ase_barrier_predicted_eV"])
    print("\n===== Comparison Summary (unique envs) =====")
    print(f"  Total unique ASE rows:  {total}")
    print(f"  Matched with LAMMPS:    {matched}")
    print(f"  Tolerance (eV):         {args.tol:.4f}")
    print(f"  Within tolerance:       {within} ({(within/max(1,matched))*100:.2f}%)")
    print(f"  ASE barrier records:    {ase_predicted_matches}/{total}")
    print(f"  Mean |Δ| (eV):          {mean_abs:.6f}")
    print(f"  Median |Δ| (eV):        {med_abs:.6f}")
    print(f"  RMSE (eV):              {rmse:.6f}")
    print(f"  Matching factor:        {match_factor:.2f}%  (fraction of matched rows with |Δ| ≤ tol)")
    print("===========================================")

########################
# New helper functions #
########################

def infer_h_count_from_init(job_dir):
    """
    Returns integer n_H inferred from total atoms in init.data / initial.data.
    Convention: total_atoms = 54 + n_H  =>  n_H = total_atoms - 54.
    """
    candidates = ["init.data", "initial.data"]
    for name in candidates:
        path = os.path.join(job_dir, name)
        if os.path.isfile(path):
            total = parse_total_atoms_from_lammps_data(path)
            if total is not None and total >= 54:
                return total - 54
    return None


def parse_total_atoms_from_lammps_data(path):
    """
    Parse a LAMMPS data file header for a line like: '55 atoms'.
    Returns int or None.
    """
    try:
        with open(path, "r", errors="ignore") as f:
            for raw in f:
                line = raw.strip().lower()
                if line.endswith(" atoms"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[-1] == "atoms":
                        try:
                            return int(parts[-2])
                        except Exception:
                            pass
    except Exception:
        pass
    return None


def _site_type_from_index(site_index, per_cell=18, tet_count=12):
    """
    Infer site type from interstitial index using the cache_new ordering:
    12 tet then 6 oct per cell, repeating every 18 indices.
    """
    try:
        idx = int(site_index)
    except Exception:
        return ""
    return "T" if (idx % per_cell) < tet_count else "O"

def _is_number_token(tok):
    try:
        float(tok)
        return True
    except Exception:
        return False

def _parse_lammps_data_atoms(path):
    """
    Parse LAMMPS data file for box bounds and atom records.
    Returns (bounds, atoms) where bounds is dict with xlo/xhi/ylo/yhi/zlo/zhi,
    and atoms is list of dicts with id, type, x, y, z.
    """
    bounds = {}
    atoms = []
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None, None

    for raw in lines:
        parts = raw.strip().split()
        if len(parts) >= 4 and parts[-2] in ("xlo", "ylo", "zlo") and parts[-1] in ("xhi", "yhi", "zhi"):
            if _is_number_token(parts[0]) and _is_number_token(parts[1]):
                key_lo = parts[-2]
                key_hi = parts[-1]
                bounds[key_lo] = float(parts[0])
                bounds[key_hi] = float(parts[1])

    atoms_idx = None
    for idx, raw in enumerate(lines):
        parts = raw.strip().split()
        if parts and parts[0].lower() == "atoms":
            atoms_idx = idx
            break

    if atoms_idx is None:
        return bounds, atoms

    start = None
    for j in range(atoms_idx + 1, len(lines)):
        if lines[j].strip() == "":
            start = j + 1
            break
    if start is None:
        return bounds, atoms

    for raw in lines[start:]:
        stripped = raw.strip()
        if not stripped:
            break
        parts = stripped.split()
        if not parts:
            break
        if not _is_number_token(parts[0]):
            break
        if len(parts) < 5:
            continue
        try:
            atom_id = int(float(parts[0]))
            atom_type = int(float(parts[1]))
            x = float(parts[-3])
            y = float(parts[-2])
            z = float(parts[-1])
            atoms.append({"id": atom_id, "type": atom_type, "x": x, "y": y, "z": z})
        except Exception:
            continue

    return bounds, atoms

def _fractional_pos(pos, origin, a):
    return [((pos[i] - origin[i]) / a) % 1.0 for i in range(3)]

def _frac_dist(f, g):
    return math.sqrt(sum(min(abs(f[i] - g[i]), 1.0 - abs(f[i] - g[i])) ** 2 for i in range(3)))

def _site_type_from_pos(pos, a, origin, tol_frac=1e-3, force_nearest=False):
    tet = [
        (0.25, 0.5, 0.0),
        (0.75, 0.5, 0.0),
        (0.5, 0.25, 0.0),
        (0.5, 0.75, 0.0),
        (0.25, 0.0, 0.5),
        (0.75, 0.0, 0.5),
        (0.5, 0.0, 0.25),
        (0.5, 0.0, 0.75),
        (0.0, 0.25, 0.5),
        (0.0, 0.75, 0.5),
        (0.0, 0.5, 0.25),
        (0.0, 0.5, 0.75),
    ]
    octa = [
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]

    frac = _fractional_pos(pos, origin, a)
    tet_dist = min(_frac_dist(frac, site) for site in tet)
    oct_dist = min(_frac_dist(frac, site) for site in octa)
    if force_nearest:
        return "T" if tet_dist <= oct_dist else "O"
    if tet_dist <= tol_frac and tet_dist <= oct_dist:
        return "T"
    if oct_dist <= tol_frac:
        return "O"
    return ""

def _dump_bounds_to_box(bounds):
    xlo = bounds.get("xlo", None)
    xhi = bounds.get("xhi", None)
    ylo = bounds.get("ylo", None)
    yhi = bounds.get("yhi", None)
    zlo = bounds.get("zlo", None)
    zhi = bounds.get("zhi", None)
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        return None, None, None
    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        return None, None, None
    a = (Lx + Ly + Lz) / 9.0  # L ~= 3a
    origin = (xlo, ylo, zlo)
    box = (Lx, Ly, Lz)
    return origin, a, box

def _parse_dump_first_frame(path):
    bounds = {}
    rows = []
    num_atoms = None
    headers = None

    try:
        with open(path, "r", errors="ignore") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("ITEM: NUMBER OF ATOMS"):
                    try:
                        num_atoms = int(f.readline().strip())
                    except Exception:
                        num_atoms = None
                elif line.startswith("ITEM: BOX BOUNDS"):
                    try:
                        xlo, xhi = map(float, f.readline().split()[:2])
                        ylo, yhi = map(float, f.readline().split()[:2])
                        zlo, zhi = map(float, f.readline().split()[:2])
                        bounds.update({
                            "xlo": xlo, "xhi": xhi,
                            "ylo": ylo, "yhi": yhi,
                            "zlo": zlo, "zhi": zhi,
                        })
                    except Exception:
                        continue
                elif line.startswith("ITEM: ATOMS"):
                    headers = line.split()[2:]
                    if not headers:
                        break
                    count = num_atoms if num_atoms is not None else 0
                    for _ in range(count):
                        row = f.readline()
                        if not row:
                            break
                        parts = row.strip().split()
                        if parts:
                            rows.append(parts)
                    break
    except Exception:
        return None, None

    if headers is None:
        return None, None

    idx = {name: i for i, name in enumerate(headers)}
    id_idx = idx.get("id", None)
    type_idx = idx.get("type", None)
    if id_idx is None or type_idx is None:
        return bounds, []

    coord_keys = None
    scaled = False
    for keys in (("xs", "ys", "zs"), ("xsu", "ysu", "zsu")):
        if all(k in idx for k in keys):
            coord_keys = keys
            scaled = True
            break
    if coord_keys is None:
        for keys in (("x", "y", "z"), ("xu", "yu", "zu")):
            if all(k in idx for k in keys):
                coord_keys = keys
                scaled = False
                break
    if coord_keys is None:
        return bounds, []

    origin, _, box = _dump_bounds_to_box(bounds)
    if scaled and (origin is None or box is None):
        return bounds, []

    atoms = []
    for parts in rows:
        if len(parts) <= max(id_idx, type_idx, idx[coord_keys[0]], idx[coord_keys[1]], idx[coord_keys[2]]):
            continue
        try:
            atom_id = int(float(parts[id_idx]))
            atom_type = int(float(parts[type_idx]))
            cx = float(parts[idx[coord_keys[0]]])
            cy = float(parts[idx[coord_keys[1]]])
            cz = float(parts[idx[coord_keys[2]]])
            if scaled:
                cx = origin[0] + cx * box[0]
                cy = origin[1] + cy * box[1]
                cz = origin[2] + cz * box[2]
            atoms_entry = {
                "id": atom_id,
                "type": atom_type,
                "x": cx,
                "y": cy,
                "z": cz,
            }
            atoms.append(atoms_entry)
        except Exception:
            continue

    return bounds, atoms

def _dump_h_positions(path):
    bounds, atoms = _parse_dump_first_frame(path)
    if not atoms:
        return bounds, {}
    h_atoms = {a["id"]: (a["x"], a["y"], a["z"]) for a in atoms if a.get("type") == 2}
    return bounds, h_atoms

def _find_dump_files(job_dir):
    dumps = []
    try:
        names = os.listdir(job_dir)
    except Exception:
        return []
    for name in names:
        m = re.match(r"^dump\.all\.(\d+)$", name)
        if m:
            dumps.append((int(m.group(1)), os.path.join(job_dir, name)))
    if not dumps:
        for name in names:
            if not name.startswith("dump"):
                continue
            m = re.search(r"(\d+)$", name)
            idx = int(m.group(1)) if m else None
            dumps.append((idx, os.path.join(job_dir, name)))
    dumps.sort(key=lambda item: (item[0] is None, item[0], item[1]))
    return [p for _, p in dumps]


def _get_parent_dir(path):
    """
    Return absolute directory for the provided path, or empty string if missing.
    """
    if not path:
        return ""
    return os.path.dirname(path)


def pick_lammps_consolidated_dump(job_dir, name=LMP_CONSOLIDATED_NAME):
    """
    Return the consolidated LAMMPS lammpstrj path if it exists inside the job dir.
    """
    consolidated_path = os.path.join(job_dir, name)
    return os.path.abspath(consolidated_path) if os.path.isfile(consolidated_path) else ""


def _count_atoms_in_lammpstrj(path):
    """
    Parse a LAMMPS lammpstrj file and count total atoms per frame.
    """
    counts = []
    try:
        with open(path, "r", errors="ignore") as f:
            num_atoms = None
            while True:
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("ITEM: NUMBER OF ATOMS"):
                    next_line = f.readline()
                    try:
                        num_atoms = int(next_line.strip())
                    except Exception:
                        num_atoms = None
                elif stripped.startswith("ITEM: ATOMS"):
                    headers = stripped.split()[2:]
                    if num_atoms is not None:
                        # skip atoms block and record constant count
                        for _ in range(num_atoms):
                            if not f.readline():
                                break
                        counts.append(num_atoms)
                    num_atoms = None
    except Exception:
        return []
    return counts


def _format_frame_counts(frame_counts):
    """
    Format a frame-count list as 'index:count' pairs joined by ';' for CSV.
    """
    if not frame_counts:
        return ""
    return ";".join(f"{idx}:{count}" for idx, count in enumerate(frame_counts))


def _count_atoms_in_xyz(path):
    counts = []
    try:
        with open(path, "r", errors="ignore") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    natoms = int(line)
                except Exception:
                    continue
                counts.append(natoms)
                # Skip the comment line
                f.readline()
                for _ in range(natoms):
                    if not f.readline():
                        break
    except Exception:
        return []
    return counts


def _count_atoms_from_ase_binary(path):
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return []
    counts = []
    for match in ASE_POSITIONS_RX.finditer(data):
        try:
            counts.append(int(match.group(1)))
        except Exception:
            continue
    return counts


def _count_atoms_in_ase_dump(path):
    if not path:
        return []
    candidate = path
    if not candidate.lower().endswith(".xyz"):
        candidate = f"{path}.xyz"
    if os.path.isfile(candidate):
        return _count_atoms_in_xyz(candidate)
    return _count_atoms_from_ase_binary(path)

def _dump_site_info(job_dir, tol_frac=0.08):
    dumps = _find_dump_files(job_dir)
    if not dumps:
        return "", "", None, None, None
    start_path = dumps[0]
    end_path = dumps[-1] if len(dumps) > 1 else dumps[0]

    bounds, start_h = _dump_h_positions(start_path)
    if not start_h:
        return "", "", None, None, None
    moving_id = min(start_h.keys())
    start_pos = start_h.get(moving_id)

    end_pos = None
    end_bounds = bounds
    if end_path:
        end_bounds, end_h = _dump_h_positions(end_path)
        if moving_id in end_h:
            end_pos = end_h[moving_id]
        elif end_h:
            end_pos = end_h[min(end_h.keys())]

    origin, a, box = _dump_bounds_to_box(end_bounds or bounds or {})
    if origin is None or a is None:
        return "", "", start_pos, end_pos, box

    start_type = _site_type_from_pos(start_pos, a, origin, tol_frac=tol_frac, force_nearest=True) if start_pos else ""
    end_type = _site_type_from_pos(end_pos, a, origin, tol_frac=tol_frac, force_nearest=True) if end_pos else ""
    return start_type, end_type, start_pos, end_pos, box

def _pbc_delta(start, end, box):
    if start is None or end is None or box is None:
        return None
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    dx -= box[0] * round(dx / box[0])
    dy -= box[1] * round(dy / box[1])
    dz -= box[2] * round(dz / box[2])
    return (dx, dy, dz)

def infer_other_h_site_type(job_dir):
    """
    Infer the nearest neighbor H site type (T/O) from initial.data, if present.
    """
    candidates = ["init.data", "initial.data"]
    for name in candidates:
        path = os.path.join(job_dir, name)
        if not os.path.isfile(path):
            continue
        bounds, atoms = _parse_lammps_data_atoms(path)
        if not atoms:
            continue
        h_atoms = [a for a in atoms if a.get("type") == 2]
        if len(h_atoms) < 2:
            return ""
        try:
            xlo = bounds.get("xlo", 0.0)
            xhi = bounds.get("xhi", None)
            ylo = bounds.get("ylo", 0.0)
            yhi = bounds.get("yhi", None)
            zlo = bounds.get("zlo", 0.0)
            zhi = bounds.get("zhi", None)
            if xhi is None or yhi is None or zhi is None:
                return ""
            Lx = xhi - xlo
            Ly = yhi - ylo
            Lz = zhi - zlo
            if Lx <= 0 or Ly <= 0 or Lz <= 0:
                return ""
            a = (Lx + Ly + Lz) / 9.0  # L ~= 3a
            origin = (xlo, ylo, zlo)
        except Exception:
            return ""

        moving = min(h_atoms, key=lambda a: a.get("id", 0))
        mx, my, mz = moving["x"], moving["y"], moving["z"]
        box = (Lx, Ly, Lz)

        def pbc_dist(ax, ay, az):
            dx = ax - mx
            dy = ay - my
            dz = az - mz
            dx -= box[0] * round(dx / box[0])
            dy -= box[1] * round(dy / box[1])
            dz -= box[2] * round(dz / box[2])
            return math.sqrt(dx * dx + dy * dy + dz * dz)

        others = [a for a in h_atoms if a is not moving]
        if not others:
            return ""
        nearest = min(others, key=lambda a: pbc_dist(a["x"], a["y"], a["z"]))
        return _site_type_from_pos((nearest["x"], nearest["y"], nearest["z"]), a, origin, force_nearest=True)
    return ""


def classify_transition_kind(job_dir, h_site, n_site, axis_tol=1e-3):
    """
    Classify transitions as {T|O}-{O|S}-{T|O} using:
      - start/end site type from dump coordinates when available (fallback to index)
      - mid-type from hop axis-alignment (O if axis-aligned, S otherwise)
    """
    # Prefer dump-based positions for site typing and hop vectors when available.
    dump_start, dump_end, dump_start_pos, dump_end_pos, dump_box = _dump_site_info(job_dir)
    hop_vec = _pbc_delta(dump_start_pos, dump_end_pos, dump_box) if dump_start_pos and dump_end_pos else None

    row = None
    if hop_vec is None:
        parent = os.path.dirname(job_dir)
        grandparent = os.path.dirname(parent) if parent else ""
        search_dirs = []
        for candidate in (job_dir, parent, grandparent):
            if candidate and candidate not in search_dirs:
                search_dirs.append(candidate)
        nb_path = None
        for directory in search_dirs:
            if not directory:
                continue
            for fname in ("neighbors.csv", "neighbors.txt"):
                candidate = os.path.join(directory, fname)
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    nb_path = candidate
                    break
            if nb_path:
                break
        if not nb_path:
            return ""

        row = get_neighbor_row(nb_path, n_site)
        if row is None:
            return ""

        hop_vec = extract_hop_vector(row)
        if hop_vec is None:
            origin = get_h_pos_from_init(job_dir)
            target = extract_target_xyz(row)
            if origin is None or target is None:
                return ""
            hop_vec = (target[0] - origin[0], target[1] - origin[1], target[2] - origin[2])

    axis_count = sum(1 for comp in hop_vec if abs(comp) > axis_tol)
    if axis_count == 1:
        mid = "O"
    elif axis_count >= 2:
        mid = "S"
    else:
        return ""

    start = dump_start or _site_type_from_index(h_site)
    end = dump_end or _site_type_from_index(n_site)
    if not start or not end:
        return ""
    return f"{start}-{mid}-{end}"


def get_neighbor_row(path, n_site):
    """
    Return the neighbor record (dict with lowercase headers) for the requested site.
    """
    key_cols = ("n_site", "neighbor", "index", "n")
    try:
        with open(path, "r", newline="", errors="ignore") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
            except Exception:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            rows = []
            for row in reader:
                cleaned = [col.strip() for col in row]
                if any(cleaned):
                    rows.append(cleaned)
    except Exception:
        return None

    if not rows:
        return None

    header = [h.strip().lower() for h in rows[0]]
    header_has_alpha = any(any(ch.isalpha() for ch in h) for h in header if h)
    if header_has_alpha:
        idx_map = {name: i for i, name in enumerate(header) if name}
        key_idx = next((idx_map[c] for c in key_cols if c in idx_map), None)
        if key_idx is not None:
            for row in rows[1:]:
                if len(row) <= key_idx:
                    continue
                try:
                    if int(str(row[key_idx]).strip()) == int(n_site):
                        return {header[i]: row[i] for i in range(len(header)) if header[i]}
                except Exception:
                    continue

    # fallback: treat first column as n_site and return positional keys
    for row in rows:
        if not row:
            continue
        try:
            if int(str(row[0]).strip()) == int(n_site):
                return {str(i): row[i] for i in range(len(row))}
        except Exception:
            continue
    return None


def extract_hop_vector(row):
    """
    Extract hop displacement (dx,dy,dz) from a neighbor row if present.
    """
    row_lower = {str(k).lower(): v for k, v in row.items()}
    dx_keys = ("dx", "hop_dx", "ddx", "vx", "sx")
    dy_keys = ("dy", "hop_dy", "ddy", "vy", "sy")
    dz_keys = ("dz", "hop_dz", "ddz", "vz", "sz")

    def pick(keys):
        for key in keys:
            if key in row_lower:
                try:
                    return float(str(row_lower[key]))
                except Exception:
                    continue
        return None

    dx = pick(dx_keys)
    dy = pick(dy_keys)
    dz = pick(dz_keys)
    if dx is None or dy is None or dz is None:
        return None
    return (dx, dy, dz)


def extract_target_xyz(row):
    """
    Pull absolute target coordinates from a neighbor row, if present.
    """
    row_lower = {str(k).lower(): v for k, v in row.items()}
    x_keys = ("nx", "x1", "xf", "x_t", "x_target", "x")
    y_keys = ("ny", "y1", "yf", "y_t", "y_target", "y")
    z_keys = ("nz", "z1", "zf", "z_t", "z_target", "z")

    def pick(keys):
        for key in keys:
            if key in row_lower:
                try:
                    return float(str(row_lower[key]))
                except Exception:
                    continue
        return None

    x = pick(x_keys)
    y = pick(y_keys)
    z = pick(z_keys)
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def get_h_pos_from_init(job_dir):
    """
    Best-effort extraction of the mobile H position from init.data / initial.data.
    Falls back to returning the coordinates of the final atom listed in the Atoms section.
    """
    for name in ("init.data", "initial.data"):
        path = os.path.join(job_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue

        atoms_idx = None
        for idx, line in enumerate(lines):
            if line.strip().lower().startswith("atoms"):
                atoms_idx = idx
                break
        if atoms_idx is None:
            continue

        start = None
        for j in range(atoms_idx + 1, len(lines)):
            if lines[j].strip() == "":
                start = j + 1
                break
        if start is None:
            continue

        atom_rows = []
        for line in lines[start:]:
            stripped = line.strip()
            if not stripped:
                break
            atom_rows.append(stripped)
        if not atom_rows:
            continue

        try:
            parts = atom_rows[-1].split()
            x, y, z = map(float, parts[-3:])
            return (x, y, z)
        except Exception:
            continue
    return None

if __name__ == "__main__":
    main()
