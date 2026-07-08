#!/usr/bin/env python3
"""Merge an accumulated Devanathan cache with cache entries from one run."""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kmc.services.cache import BarrierCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--ranks", type=int, default=3)
    args = parser.parse_args()

    with args.base.open("rb") as fh:
        merged = dict(pickle.load(fh))
    base_count = len(merged)

    cache_dir = args.run / "cache"
    added_sources = 0
    for rank in range(args.ranks):
        path = cache_dir / f"barrier_cache_rank{rank}_{args.schema}.pkl"
        delta = Path(f"{path}.delta.pkl")
        if not path.exists() and not delta.exists():
            continue
        merged.update(dict(BarrierCache(str(path))))
        added_sources += 1

    if added_sources == 0:
        raise FileNotFoundError(f"No rank cache files found under {cache_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as fh:
        pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(args.out)
    print(f"cache={args.out} base_entries={base_count} merged_entries={len(merged)}")


if __name__ == "__main__":
    main()
