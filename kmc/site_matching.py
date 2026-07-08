"""
Utilities for mapping atom positions -> nearest interstitial sites.
"""
from __future__ import annotations
import numpy as np

def nearest_site_index(position, site_positions):
    d=np.linalg.norm(site_positions-position, axis=1)
    return int(np.argmin(d)), float(np.min(d))

def match_position_to_site(position, site_positions, tol=0.5):
    idx,dist=nearest_site_index(position, site_positions)
    if dist>tol:
        raise ValueError(f"No site match within tolerance. Closest={dist:.4f} A")
    return idx

def match_positions_batch(positions, site_positions, tol=0.5):
    return [match_position_to_site(p, site_positions, tol) for p in positions]

def validate_matches(indices):
    if len(indices)!=len(set(indices)):
        raise ValueError("Duplicate occupancy detected")
    return True
