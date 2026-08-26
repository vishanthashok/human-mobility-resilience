"""Vectorized Haversine distances and radius of gyration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EARTH_RADIUS_KM


def haversine_km(
    lat1: np.ndarray | pd.Series | float,
    lon1: np.ndarray | pd.Series | float,
    lat2: np.ndarray | pd.Series | float,
    lon2: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Great-circle distance in kilometers (Wang & Taylor Earth radius)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def radius_of_gyration_km(latitudes: np.ndarray, longitudes: np.ndarray) -> float:
    """RMS Haversine distance of points from their coordinate-wise mean center."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    n = latitudes.size
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    lat_c = np.mean(latitudes)
    lon_c = np.mean(longitudes)
    d = haversine_km(latitudes, longitudes, lat_c, lon_c)
    return float(np.sqrt(np.mean(d**2)))
