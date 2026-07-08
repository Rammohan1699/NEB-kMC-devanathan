#!/usr/bin/env python3
"""Merge every barrier-cache file found under one or more directory trees."""
from __future__ import annotations

import argparse
import fnmatch
import os
import pickle
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kmc.services.cache import BarrierCache  # noqa: E402


def _candidate_base_paths(root: Path, pattern: str) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if not fnmatch.fnmatch(name, pattern):
            continue
        if name.endswith(".delta.pkl"):
            base = Path(str(path)[: -len(".delta.pkl")])
        else:
            base = path
        if base in seen:
            continue
        seen.add(base)
        yield base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        required=True,
        help="Directory tree containing downloaded cache files. May repeat.",
    )
    parser.add_argument(
        "--pattern",
        default="barrier_cache*.pkl*",
        help="Filename glob to merge after stripping optional .delta.pkl suffix.",
    )
    parser.add_argument(
        "--require-text",
        action="append",
        default=[],
        help="Only merge cache paths containing this substring. May repeat.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = args.base.expanduser()
    output = args.output.expanduser()
    if not base.is_file():
        raise FileNotFoundError(base)
    if output.exists() and not args.dry_run:
        raise FileExistsError(output)

    with base.open("rb") as handle:
        merged = dict(pickle.load(handle))
    base_count = len(merged)
    loaded_files = 0
    loaded_entries = 0

    for root in args.root:
        root = root.expanduser()
        if not root.is_dir():
            raise NotADirectoryError(root)
        for path in _candidate_base_paths(root, args.pattern):
            path_text = str(path)
            if args.require_text and not all(text in path_text for text in args.require_text):
                continue
            try:
                cache = dict(BarrierCache(str(path)))
            except Exception as exc:
                print(f"warning: failed to load {path}: {exc}", file=sys.stderr)
                continue
            loaded_files += 1
            loaded_entries += len(cache)
            before = len(merged)
            merged.update(cache)
            print(
                f"merged {len(cache)} entries from {path} "
                f"(new_unique={len(merged) - before})"
            )

    print(
        f"base_entries={base_count} files={loaded_files} "
        f"loaded_entries={loaded_entries} output_entries={len(merged)}"
    )
    if args.dry_run:
        return 0
    if loaded_files == 0:
        raise SystemExit("No cache files matched the requested roots/patterns")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
