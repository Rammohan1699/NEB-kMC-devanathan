"""
Cache management helpers for KMC/NEB barrier calculations.

This module wraps BarrierCache with the rank-aware filenames and merge/save
policies used by the current MPI KMC implementation.

It intentionally keeps persistence policy separate from the KMC loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Callable
import os
import pickle


@dataclass(frozen=True)
class CachePaths:
    """Rank-aware cache file paths."""
    rank_cache: Path
    merged_cache: Path
    delta_cache: Path
    schema: str
    rank: int


@dataclass(frozen=True)
class CacheManagerConfig:
    """Barrier cache runtime controls."""
    schema: str
    directory: str = "."
    merge_mode: str = "global"      # global | local | mixed
    merge_interval: int = 100
    enabled: bool = True
    preload_file: str = ""


def build_cache_paths(cfg: CacheManagerConfig, rank: int) -> CachePaths:
    """Create rank-local and merged cache paths from a schema string."""
    root = Path(cfg.directory)
    rank_cache = root / f"barrier_cache_rank{rank}_{cfg.schema}.pkl"
    merged_cache = root / f"barrier_cache_{cfg.schema}.pkl"
    return CachePaths(
        rank_cache=rank_cache,
        merged_cache=merged_cache,
        delta_cache=Path(str(rank_cache) + ".delta.pkl"),
        schema=cfg.schema,
        rank=int(rank),
    )


def load_pickle_mapping(path: os.PathLike | str) -> dict[Any, Any]:
    """Load a pickle mapping safely; return empty dict if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, Mapping):
        return dict(data)
    return dict(data)


def preload_merged_cache(paths: CachePaths, logger: Optional[Callable[[str], None]] = None) -> dict[Any, Any]:
    """Load the merged cache file if available."""
    if not paths.merged_cache.exists():
        return {}
    try:
        data = load_pickle_mapping(paths.merged_cache)
        if logger:
            logger(
                f"Loaded merged barrier cache {paths.merged_cache} "
                f"with {len(data)} entries"
            )
        return data
    except Exception as exc:
        if logger:
            logger(f"Warning: failed to load merged barrier cache {paths.merged_cache}: {exc}")
        return {}


def preload_explicit_cache(path: os.PathLike | str, logger: Optional[Callable[[str], None]] = None) -> dict[Any, Any]:
    """Load a user-selected cache file independent of the active cache schema."""
    cache_path = Path(path).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"Explicit barrier cache file not found: {cache_path}")
    data = load_pickle_mapping(cache_path)
    if logger:
        logger(f"Loaded explicit barrier cache {cache_path} with {len(data)} entries")
    return data


class BarrierCacheManager:
    """
    Thin wrapper around services.cache.BarrierCache.

    Usage:
        mgr = BarrierCacheManager.from_config(cfg, rank)
        cache = mgr.cache
    """

    def __init__(
        self,
        cache,
        paths: CachePaths,
        cfg: CacheManagerConfig,
        *,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cache = cache
        self.paths = paths
        self.cfg = cfg
        self.logger = logger

    @classmethod
    def from_config(
        cls,
        cfg: CacheManagerConfig,
        rank: int,
        *,
        barrier_cache_cls=None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> "BarrierCacheManager":
        if barrier_cache_cls is None:
            try:
                from .cache import BarrierCache  # type: ignore
            except Exception:
                try:
                    from services.cache import BarrierCache  # type: ignore
                except Exception:
                    from cache import BarrierCache  # type: ignore
            barrier_cache_cls = BarrierCache

        paths = build_cache_paths(cfg, rank)
        paths.rank_cache.parent.mkdir(parents=True, exist_ok=True)

        if cfg.preload_file:
            preload = preload_explicit_cache(cfg.preload_file, logger=logger)
        else:
            preload = preload_merged_cache(paths, logger=logger)
        cache = barrier_cache_cls(
            str(paths.rank_cache),
            enabled=cfg.enabled,
            initial_store=preload,
        )
        return cls(cache, paths, cfg, logger=logger)

    def update_finite(self, mapping: Mapping[Any, Any]) -> int:
        """Update cache only with finite numeric values."""
        import numpy as np

        finite_items = {
            k: v for k, v in mapping.items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        }
        if finite_items:
            self.cache.update(finite_items.items())
        return len(finite_items)

    def save_delta(self) -> None:
        """Persist dirty entries using append-only delta mode."""
        self.cache.save(full=False)

    def save_full(self) -> None:
        """Persist a compact full snapshot."""
        self.cache.save(full=True)

    def should_mixed_share(self, step: int) -> bool:
        """True when mixed cache sharing should broadcast accumulated entries."""
        if self.cfg.merge_mode != "mixed":
            return False
        interval = max(1, int(self.cfg.merge_interval))
        return (int(step) + 1) % interval == 0


def merge_rank_caches(
    cfg: CacheManagerConfig,
    size: int,
    *,
    barrier_cache_cls=None,
    logger: Optional[Callable[[str], None]] = None,
) -> dict[Any, Any]:
    """
    Merge all rank caches into one schema-wide cache file.

    This mirrors the legacy rank-0 post-processing behavior, including support
    for BarrierCache delta replay through the BarrierCache constructor.
    """
    if barrier_cache_cls is None:
        try:
            from .cache import BarrierCache  # type: ignore
        except Exception:
            try:
                from services.cache import BarrierCache  # type: ignore
            except Exception:
                from cache import BarrierCache  # type: ignore
        barrier_cache_cls = BarrierCache

    root = Path(cfg.directory)
    merged_path = root / f"barrier_cache_{cfg.schema}.pkl"
    merged: dict[Any, Any] = {}
    if cfg.preload_file:
        try:
            preload_path = Path(cfg.preload_file).expanduser()
            merged.update(load_pickle_mapping(preload_path))
            if logger:
                logger(
                    f"Preserving {len(merged)} preload entries from {preload_path}"
                )
        except Exception as exc:
            if logger:
                logger(
                    f"Warning: failed to preserve explicit preload cache "
                    f"{cfg.preload_file}: {exc}"
                )
    if merged_path.exists():
        try:
            merged.update(load_pickle_mapping(merged_path))
            if logger:
                logger(f"Preserving {len(merged)} existing entries from {merged_path}")
        except Exception as exc:
            if logger:
                logger(f"Warning: failed to preload existing merged cache {merged_path}: {exc}")

    for rank in range(int(size)):
        paths = build_cache_paths(cfg, rank)
        if (not paths.rank_cache.exists()) and (not paths.delta_cache.exists()):
            if logger:
                logger(f"Note: {paths.rank_cache} and delta not found; skipping")
            continue
        try:
            rank_cache = barrier_cache_cls(str(paths.rank_cache))
            data = dict(rank_cache)
            if logger:
                logger(f"Merging {len(data)} entries from {paths.rank_cache}")
            merged.update(data)
        except Exception as exc:
            if logger:
                logger(f"Error loading {paths.rank_cache}: {exc}")

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with merged_path.open("wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        if logger:
            logger(f"Wrote merged barrier cache with {len(merged)} entries to {merged_path}")
    except Exception as exc:
        if logger:
            logger(f"ERROR writing merged cache {merged_path}: {exc}")

    return merged
