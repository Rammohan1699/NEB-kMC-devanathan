#!/usr/bin/env python3
"""Compact a barrier-cache snapshot/delta pair into one full pickle."""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kmc.services.cache import BarrierCache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.expanduser()
    source_delta = Path(f"{source}.delta.pkl")
    if not source.exists() and not source_delta.exists():
        raise FileNotFoundError(
            f"Neither cache snapshot nor delta exists: {source}"
        )
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError(output)

    cache = BarrierCache(str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(dict(cache), handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    print(f"source={source} entries={len(cache)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
