"""Daily mobility metrics, home location, and baseline summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_DAY_END,
    BASELINE_DAY_START,
    EPS_COUNT,
    LOCATION_DECIMALS,
)
from src.geo import haversine_km


def add_spatial_cell(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lat_cell = out["latitude"].round(LOCATION_DECIMALS)
    lon_cell = out["longitude"].round(LOCATION_DECIMALS)
    out["cell"] = lat_cell.astype(str) + "," + lon_cell.astype(str)
    return out


def estimate_home(df: pd.DataFrame) -> pd.DataFrame:
    """Median baseline coordinate per user (robust stand-in for home / routine center)."""
    baseline = df["day_relative"].between(BASELINE_DAY_START, BASELINE_DAY_END)
    home = (
        df.loc[baseline]
        .groupby(["disaster", "user_anon"], as_index=False)
        .agg(home_lat=("latitude", "median"), home_lon=("longitude", "median"))
    )
    return home


def shannon_entropy(values: pd.Series) -> float:
    counts = values.value_counts()
    if counts.empty:
        return float("nan")
    p = counts.to_numpy(dtype=float)
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def compute_daily_metrics(df: pd.DataFrame, home: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tweet-level coordinates → user × disaster × day_relative metrics."""
    if home is None:
        home = estimate_home(df)
    work = add_spatial_cell(df)
    work = work.merge(home, on=["disaster", "user_anon"], how="left")
    work = work.sort_values(
        ["disaster", "user_anon", "day_relative", "time"], kind="mergesort"
    )

    keys = ["disaster", "user_anon", "day_relative"]
    center_lat = work.groupby(keys)["latitude"].transform("mean")
    center_lon = work.groupby(keys)["longitude"].transform("mean")
    work["d2_center"] = (
        haversine_km(work["latitude"], work["longitude"], center_lat, center_lon) ** 2
    )
    work["disp_home"] = haversine_km(
        work["latitude"], work["longitude"], work["home_lat"], work["home_lon"]
    )
    prev_lat = work.groupby(keys)["latitude"].shift(1)
    prev_lon = work.groupby(keys)["longitude"].shift(1)
    work["hop_km"] = haversine_km(
        prev_lat, prev_lon, work["latitude"], work["longitude"]
    )

    daily = work.groupby(keys, as_index=False).agg(
        n_obs=("latitude", "size"),
        radius=("d2_center", lambda s: float(np.sqrt(np.nanmean(s.to_numpy())))),
        mobility_distance=("hop_km", "sum"),
        displacement=("disp_home", "mean"),
        max_displacement=("disp_home", "max"),
        unique_locations=("cell", "nunique"),
        disaster_type=("disaster_type", "first"),
    )
    daily["radius"] = daily["radius"].fillna(0.0)
    daily["mobility_distance"] = daily["mobility_distance"].fillna(0.0)
    return daily


def compute_baseline_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Median daily metrics on baseline days, plus observation counts and entropy."""
    base_mask = daily["day_relative"].between(BASELINE_DAY_START, BASELINE_DAY_END)
    baseline_daily = daily.loc[base_mask]
    features = baseline_daily.groupby(["disaster", "user_anon"], as_index=False).agg(
        baseline_radius=("radius", "median"),
        baseline_mobility_distance=("mobility_distance", "median"),
        baseline_displacement=("displacement", "median"),
        baseline_unique_locations=("unique_locations", "median"),
        baseline_max_displacement=("max_displacement", "median"),
        baseline_n_days=("day_relative", "nunique"),
        baseline_n_obs=("n_obs", "sum"),
        disaster_type=("disaster_type", "first"),
    )

    tweet_base = df["day_relative"].between(BASELINE_DAY_START, BASELINE_DAY_END)
    cells = add_spatial_cell(df.loc[tweet_base, ["disaster", "user_anon", "latitude", "longitude"]])
    entropy = (
        cells.groupby(["disaster", "user_anon"])["cell"]
        .apply(shannon_entropy)
        .rename("baseline_entropy")
        .reset_index()
    )
    features = features.merge(entropy, on=["disaster", "user_anon"], how="left")
    features["baseline_unique_locations"] = features["baseline_unique_locations"].clip(
        lower=EPS_COUNT
    )
    return features
