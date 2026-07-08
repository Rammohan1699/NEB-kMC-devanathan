#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BarrierCache — persistent, thread-safe cache for NEB barrier data.

The cache supports two usage patterns:
1. Legacy mode keyed by (h_site, n_site, env_signature) where env_signature may
   be any JSON-serialisable structure describing the local H environment.
2. Direct mapping mode using the environment keys built in cache_new.make_env_key,
   which combines H-distance fingerprints and hop direction quantisation.

NEW (2025-11): Append-only persistence and dirty tracking
--------------------------------------------------------
* We keep a full snapshot at `path` (pickle of the whole dict), and an
  append-only delta file at `path + ".delta.pkl"`.
* New/updated entries are tracked in `self._dirty_keys` and appended to the
  delta file by `save()`; if there are no dirty entries, `save()` is a no-op.
* On load, we read the snapshot (if any) then replay all deltas.
* `save(full=True)` (or `save_full()`) writes a compact snapshot and clears
  the delta file.
"""

import hashlib
import json
import os
import pickle
import threading
from collections.abc import Mapping, MutableMapping
from typing import Any


class _DeltaDeleteMarker:
    """Internal pickle marker to signal deletions in the delta stream."""

    __slots__ = ()


_DELETE_MARKER = _DeltaDeleteMarker()


class BarrierCache(MutableMapping):
    """Thread-safe dictionary-based cache that can persist to disk (pickle)."""

    def __init__(self, path="barrier_cache.pkl", *, enabled=True, initial_store=None):
        preload = {}
        cache_path = path

        # Allow passing an existing mapping directly as the first argument.
        if isinstance(path, Mapping) and not isinstance(path, (str, bytes, os.PathLike)):
            preload.update(dict(path))
            cache_path = "barrier_cache.pkl"

        if initial_store:
            if isinstance(initial_store, Mapping):
                preload.update(initial_store)
            else:
                preload.update(dict(initial_store))

        self.path = cache_path if isinstance(cache_path, (str, bytes, os.PathLike)) else None
        self.enabled = enabled
        self.lock = threading.RLock()
        self.store: dict[Any, Any] = {}
        path_str = os.fspath(self.path) if self.path is not None else None
        self._delta_path = f"{path_str}.delta.pkl" if path_str else None
        self._dirty_keys: set[Any] = set()

        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, Mapping):
                    self.store.update(data)
                else:
                    self.store.update(dict(data))
            except Exception as e:
                print(f"[BarrierCache] Warning: failed to load existing cache ({e}); starting empty.")

        if preload:
            self.store.update(preload)

        if self._delta_path and os.path.exists(self._delta_path):
            try:
                with open(self._delta_path, "rb") as df:
                    while True:
                        try:
                            key, value = pickle.load(df)
                        except EOFError:
                            break
                        if isinstance(value, _DeltaDeleteMarker):
                            self.store.pop(key, None)
                        else:
                            self.store[key] = value
            except Exception as e:
                print(f"[BarrierCache] Warning: failed to load delta file ({e}); continuing without it.")

    @property
    def dirty_count(self) -> int:
        """Number of entries pending persistence."""
        return len(self._dirty_keys)

    # ------------------------------------------------------------------
    # Environment signature helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hashable_env_sig(env_sig):
        """Turn any env signature into a short, deterministic hash."""
        if env_sig is None:
            return None
        try:
            payload = json.dumps(env_sig, sort_keys=True)
        except Exception:
            payload = repr(env_sig)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return digest[:12]

    def _key(self, h_site, n_site, env_sig=None):
        """Compose the legacy tuple key with optional environment hash."""
        base = (int(h_site), int(n_site))
        env_hash = self._hashable_env_sig(env_sig)
        if env_hash is None:
            return base
        return (base[0], base[1], f"E:{env_hash}")

    # ------------------------------------------------------------------
    # Public helpers (legacy API)
    # ------------------------------------------------------------------
    def get(self, h_site, n_site, env_sig=None):
        """Retrieve a cached value by hop indices and optional env signature."""
        if not self.enabled:
            return None
        return self.store.get(self._key(h_site, n_site, env_sig))

    def set(self, h_site, n_site, value, env_sig=None):
        """Insert or update a value using hop indices + optional env signature."""
        if not self.enabled:
            return
        self[self._key(h_site, n_site, env_sig)] = value

    def save(self, *, full: bool = False):
        """
        Persist the cache to disk.

        * When `full=False` (default), append dirty entries to the delta file.
        * When `full=True`, write a compact snapshot and truncate the delta.
        """
        if not self.enabled or not self.path:
            return

        with self.lock:
            path_str = os.fspath(self.path)
            os.makedirs(os.path.dirname(path_str) or ".", exist_ok=True)
            try:
                if full:
                    with open(self.path, "wb") as f:
                        pickle.dump(self.store, f, protocol=pickle.HIGHEST_PROTOCOL)
                    if self._delta_path:
                        with open(self._delta_path, "wb"):
                            pass
                    self._dirty_keys.clear()
                    return

                if not self._dirty_keys:
                    return

                if not self._delta_path:
                    with open(self.path, "wb") as f:
                        pickle.dump(self.store, f, protocol=pickle.HIGHEST_PROTOCOL)
                    self._dirty_keys.clear()
                    return

                os.makedirs(os.path.dirname(self._delta_path) or ".", exist_ok=True)
                with open(self._delta_path, "ab") as df:
                    for key in list(self._dirty_keys):
                        value = self.store[key] if key in self.store else _DELETE_MARKER
                        pickle.dump((key, value), df, protocol=pickle.HIGHEST_PROTOCOL)
                self._dirty_keys.clear()
            except Exception as e:
                print(f"[BarrierCache] Warning: failed to save cache ({e}).")

    def save_full(self):
        """Force a compact snapshot (equivalent to `save(full=True)`)."""
        self.save(full=True)

    def clear(self):
        """Clear in-memory entries (does not delete the pickle file)."""
        with self.lock:
            if self.store:
                self._dirty_keys.update(list(self.store.keys()))
            self.store.clear()

    # ------------------------------------------------------------------
    # MutableMapping interface
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return self.store[key]

    def get_many(self, keys):
        """Fetch multiple keys at once. Returns dict of found entries."""
        if not self.enabled:
            return {}
        with self.lock:
            return {k: self.store[k] for k in keys if k in self.store}

    def set_many(self, mapping):
        """Set many key->value pairs at once."""
        if not self.enabled:
            return
        with self.lock:
            pairs = dict(mapping)
            self.store.update(pairs)
            self._dirty_keys.update(pairs.keys())

    def __setitem__(self, key, value):
        if not self.enabled:
            return
        with self.lock:
            self.store[key] = value
            self._dirty_keys.add(key)

    def __delitem__(self, key):
        with self.lock:
            del self.store[key]
            self._dirty_keys.add(key)

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __contains__(self, key):
        return key in self.store

    def update(self, other=(), **kwargs):
        """Thread-safe update from another mapping or iterable of pairs."""
        if not self.enabled:
            return
        items = []
        if other:
            if isinstance(other, Mapping):
                items.extend(other.items())
            else:
                items.extend(other)
        if kwargs:
            items.extend(kwargs.items())

        with self.lock:
            for k, v in items:
                self.store[k] = v
                self._dirty_keys.add(k)


# ----------------------------------------------------------------------
# Standalone smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    cache = BarrierCache("barrier_cache_test.pkl")
    sig = {"neighbors": [(0.12, 0.34, 0.56)], "count": 1}

    cache.set(10, 20, 0.045, env_sig=sig)
    cache[("custom", 1)] = 3.14  # direct key usage
    # Appends only the two new entries above
    cache.save()
    # Optionally, write a compact snapshot (and clear the delta file)
    # cache.save(full=True)

    print("Entries:", len(cache))
    print("Legacy get:", cache.get(10, 20, env_sig=sig))
    print("Direct get:", cache[("custom", 1)])
