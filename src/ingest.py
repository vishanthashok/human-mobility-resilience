"""Load raw tweets, attach disaster clocks, filter sparse users."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_DAY_END,
    BASELINE_DAY_START,
    DURING_DAYS,
    EVENTS_PATH,
    MIN_OBS_AFTER,
    MIN_OBS_BASELINE,
    MIN_OBS_DURING,
    RAW_CSV_NAME,
    RAW_DIR,
)


def load_events(path: Path | None = None) -> pd.DataFrame:
    events = pd.read_csv(path or EVENTS_PATH)
    events["event_datetime_utc"] = pd.to_datetime(
        events["event_datetime_utc"], utc=True
    )
    return events


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or (RAW_DIR / RAW_CSV_NAME)
    if not path.exists():
        matches = list(RAW_DIR.glob("*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No CSV in {RAW_DIR}. Run: python scripts/download_data.py"
            )
        path = matches[0]
    df = pd.read_csv(path)
    return standardize_raw(df)


def standardize_raw(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "disaster.event": "disaster",
        "user.anon": "user_anon",
        "longitude.anon": "longitude",
        "time": "time",
        "latitude": "latitude",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"disaster", "user_anon", "latitude", "longitude", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {missing}")
    df = df.copy()
    df["user_anon"] = df["user_anon"].astype(str)
    df["disaster"] = df["disaster"].astype(str)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude"])
    df = df.sort_values(["disaster", "user_anon", "time"], kind="mergesort")
    return df.reset_index(drop=True)


def attach_disaster_clock(
    df: pd.DataFrame, events: pd.DataFrame | None = None
) -> pd.DataFrame:
    events = load_events() if events is None else events
    clock = events.rename(columns={"event_id": "disaster"})[
        ["disaster", "event_datetime_utc", "disaster_type", "name", "location"]
    ]
    out = df.merge(clock, on="disaster", how="left")
    if out["event_datetime_utc"].isna().any():
        unknown = sorted(out.loc[out["event_datetime_utc"].isna(), "disaster"].unique())
        raise ValueError(f"No event clock for disasters: {unknown}")
    delta = out["time"] - out["event_datetime_utc"]
    out["hours_from_disaster"] = delta.dt.total_seconds() / 3600.0
    out["day_relative"] = np.floor(out["hours_from_disaster"] / 24.0).astype(int)
    return out


def _obs_counts(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["disaster", "user_anon"]
    baseline = df["day_relative"].between(BASELINE_DAY_START, BASELINE_DAY_END)
    during = df["day_relative"].isin(DURING_DAYS)
    after = df["day_relative"] > DURING_DAYS[-1]
    counts = (
        df.assign(
            n_baseline=baseline.astype(int),
            n_during=during.astype(int),
            n_after=after.astype(int),
        )
        .groupby(keys, as_index=False)[["n_baseline", "n_during", "n_after"]]
        .sum()
    )
    return counts


def eligible_users(df: pd.DataFrame) -> pd.DataFrame:
    """Users with enough pre / during / post observations to estimate a baseline."""
    counts = _obs_counts(df)
    keep = (
        (counts["n_baseline"] >= MIN_OBS_BASELINE)
        & (counts["n_during"] >= MIN_OBS_DURING)
        & (counts["n_after"] >= MIN_OBS_AFTER)
    )
    return counts.loc[keep].reset_index(drop=True)


def filter_users(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = eligible_users(df)
    filtered = df.merge(
        kept[["disaster", "user_anon"]],
        on=["disaster", "user_anon"],
        how="inner",
    )
    return filtered.reset_index(drop=True), kept


def filter_events(df: pd.DataFrame, events: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    if not events:
        return df
    return df.loc[df["disaster"].isin(list(events))].copy()


def event_time_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check that the curated clock falls inside each event's tweet window."""
    rows = []
    for disaster, g in df.groupby("disaster"):
        t0 = g["event_datetime_utc"].iloc[0]
        rows.append(
            {
                "disaster": disaster,
                "n_obs": len(g),
                "n_users": g["user_anon"].nunique(),
                "t_min": g["time"].min(),
                "t_max": g["time"].max(),
                "event_datetime_utc": t0,
                "clock_in_window": bool(g["time"].min() <= t0 <= g["time"].max()),
                "day_relative_min": int(g["day_relative"].min()),
                "day_relative_max": int(g["day_relative"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("disaster").reset_index(drop=True)
