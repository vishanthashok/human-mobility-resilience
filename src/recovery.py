"""Mobility Disruption Score and recovery-time labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EPS_COUNT,
    EPS_KM,
    MDS_WEIGHTS,
    RECOVERY_CONSECUTIVE_DAYS,
    RECOVERY_TOLERANCE,
)

DISTANCE_METRICS = ("radius", "displacement", "mobility_distance")
RECOVERY_METRICS = DISTANCE_METRICS


def _eps(metric: str) -> float:
    return EPS_COUNT if metric == "unique_locations" else EPS_KM


def relative_deviation(current: pd.Series, baseline: pd.Series, metric: str) -> pd.Series:
    denom = np.maximum(baseline.to_numpy(dtype=float), _eps(metric))
    return np.abs(current.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)) / denom


def within_baseline_band(current: pd.Series, baseline: pd.Series, metric: str) -> pd.Series:
    denom = np.maximum(baseline.to_numpy(dtype=float), _eps(metric))
    return np.abs(current.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)) <= (
        RECOVERY_TOLERANCE * denom
    )


def attach_disruption(daily: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "disaster",
        "user_anon",
        "baseline_radius",
        "baseline_mobility_distance",
        "baseline_displacement",
        "baseline_unique_locations",
    ]
    out = daily.merge(baseline[cols], on=["disaster", "user_anon"], how="left")
    out["dev_radius"] = relative_deviation(out["radius"], out["baseline_radius"], "radius")
    out["dev_displacement"] = relative_deviation(
        out["displacement"], out["baseline_displacement"], "displacement"
    )
    out["dev_mobility_distance"] = relative_deviation(
        out["mobility_distance"], out["baseline_mobility_distance"], "mobility_distance"
    )
    out["dev_unique_locations"] = relative_deviation(
        out["unique_locations"], out["baseline_unique_locations"], "unique_locations"
    )
    out["mds"] = (
        MDS_WEIGHTS["radius"] * out["dev_radius"]
        + MDS_WEIGHTS["displacement"] * out["dev_displacement"]
        + MDS_WEIGHTS["mobility_distance"] * out["dev_mobility_distance"]
    )
    out["recovery_score"] = 1.0 / (1.0 + out["mds"])
    in_band = np.ones(len(out), dtype=bool)
    for metric in RECOVERY_METRICS:
        in_band &= within_baseline_band(out[metric], out[f"baseline_{metric}"], metric)
    out["in_band"] = in_band
    return out


def label_recovery(disrupted: pd.DataFrame) -> pd.DataFrame:
    """First post-impact day whose next consecutive day is also within 20% of baseline."""
    if RECOVERY_CONSECUTIVE_DAYS != 2:
        raise NotImplementedError("Only a 2-day recovery streak is implemented.")

    work = disrupted.sort_values(
        ["disaster", "user_anon", "day_relative"], kind="mergesort"
    ).copy()
    g = work.groupby(["disaster", "user_anon"], sort=False)
    work["next_day"] = g["day_relative"].shift(-1)
    work["next_in_band"] = g["in_band"].shift(-1)
    next_ok = work["next_in_band"].eq(True)
    work["is_recovery_start"] = (
        (work["day_relative"] >= 0)
        & work["in_band"]
        & next_ok
        & (work["next_day"] == work["day_relative"] + 1)
    )
    recovered = (
        work.loc[work["is_recovery_start"]]
        .groupby(["disaster", "user_anon"], as_index=False)["day_relative"]
        .min()
        .rename(columns={"day_relative": "recovery_days"})
    )
    recovered["event_observed"] = 1

    horizon = (
        work.loc[work["day_relative"] >= 0]
        .groupby(["disaster", "user_anon"], as_index=False)
        .agg(
            last_day=("day_relative", "max"),
            disaster_type=("disaster_type", "first"),
            n_post_days=("day_relative", "nunique"),
        )
    )
    labels = horizon.merge(recovered, on=["disaster", "user_anon"], how="left")
    labels["event_observed"] = labels["event_observed"].fillna(0).astype(int)
    censored = labels["event_observed"] == 0
    labels.loc[censored, "recovery_days"] = labels.loc[censored, "last_day"]
    labels["recovery_days"] = labels["recovery_days"].astype(int)
    labels["censored"] = censored.astype(int)
    return labels


def leakage_safe_user_table(
    disrupted: pd.DataFrame,
    baseline: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """One row per user. Features use baseline + day 0 only."""
    day0 = disrupted.loc[disrupted["day_relative"] == 0].copy()
    day0 = day0.rename(
        columns={
            "mds": "day0_mds",
            "max_displacement": "day0_max_displacement",
            "n_obs": "day0_n_obs",
            "radius": "day0_radius",
            "displacement": "day0_displacement",
            "mobility_distance": "day0_mobility_distance",
            "unique_locations": "day0_unique_locations",
        }
    )
    keep = [
        "disaster",
        "user_anon",
        "day0_mds",
        "day0_max_displacement",
        "day0_n_obs",
        "day0_radius",
        "day0_displacement",
        "day0_mobility_distance",
        "day0_unique_locations",
    ]
    users = baseline.merge(day0[keep], on=["disaster", "user_anon"], how="inner")
    users = users.merge(
        labels[
            [
                "disaster",
                "user_anon",
                "recovery_days",
                "event_observed",
                "censored",
                "last_day",
            ]
        ],
        on=["disaster", "user_anon"],
        how="inner",
    )
    users["user_id"] = users["disaster"] + "_" + users["user_anon"].astype(str)
    return users.reset_index(drop=True)
