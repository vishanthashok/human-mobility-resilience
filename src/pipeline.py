"""End-to-end pipeline: raw tweets → daily metrics → recovery labels → models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, FOCUS_EVENT, PROCESSED_DIR, V15_EVENTS
from src.features import compute_baseline_features, compute_daily_metrics, estimate_home
from src.ingest import (
    attach_disaster_clock,
    event_time_coverage,
    filter_events,
    filter_users,
    load_events,
    load_raw,
)
from src.models import (
    cross_validate_regressors,
    feature_importance_table,
    fit_cox,
    kaplan_meier,
    research_correlations,
    try_random_survival_forest,
    evaluate_unseen_disaster,
)
from src.recovery import attach_disruption, label_recovery, leakage_safe_user_table


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def parse_event_list(spec: str) -> list[str] | None:
    spec = spec.strip()
    if spec.lower() == "all":
        return None
    return [part.strip() for part in spec.split(",") if part.strip()]


def run_feature_pipeline(
    events: list[str] | None,
    raw_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    tweets = load_raw(raw_path)
    tweets = attach_disaster_clock(tweets, load_events())
    tweets = filter_events(tweets, events)
    coverage = event_time_coverage(tweets)
    filtered, kept_users = filter_users(tweets)
    home = estimate_home(filtered)
    daily = compute_daily_metrics(filtered, home)
    baseline = compute_baseline_features(filtered, daily)
    disrupted = attach_disruption(daily, baseline)
    labels = label_recovery(disrupted)
    users = leakage_safe_user_table(disrupted, baseline, labels)
    return {
        "tweets": tweets,
        "filtered_tweets": filtered,
        "coverage": coverage,
        "kept_users": kept_users,
        "home": home,
        "daily": daily,
        "baseline": baseline,
        "disrupted": disrupted,
        "labels": labels,
        "users": users,
    }


def save_processed(tables: dict[str, pd.DataFrame], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tables["coverage"].to_csv(ARTIFACTS_DIR / "event_coverage.csv", index=False)
    tables["labels"].groupby("disaster").agg(
        n_users=("user_anon", "nunique"),
        n_recovered=("event_observed", "sum"),
        median_last_day=("last_day", "median"),
    ).reset_index().to_csv(ARTIFACTS_DIR / "recovery_counts.csv", index=False)
    for name in (
        "coverage",
        "kept_users",
        "home",
        "daily",
        "baseline",
        "disrupted",
        "labels",
        "users",
        "filtered_tweets",
    ):
        path = out_dir / f"{name}.parquet"
        tables[name].to_parquet(path, index=False)
    return out_dir


def run_models(users: pd.DataFrame) -> dict:
    if len(users) < 10:
        raise ValueError(f"Need at least 10 users to model; got {len(users)}")
    metrics, fitted = cross_validate_regressors(users)
    km = kaplan_meier(users)
    cox = None
    try:
        cox = fit_cox(users)
    except Exception as exc:
        print(f"Cox PH skipped: {exc}")
    rsf, _ = try_random_survival_forest(users)
    importance_frames = []
    for name, pipe in fitted.items():
        importance_frames.append(feature_importance_table(pipe, name))
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(columns=["model", "feature", "importance", "kind"])
    )
    questions = research_correlations(users)
    holdout = evaluate_unseen_disaster(users)

    km_points = []
    sf = km.survival_function_
    for day in (3, 7, 14, 21):
        idx = sf.index[sf.index <= day]
        if len(idx) == 0:
            continue
        recovered = float(1.0 - sf.loc[idx[-1]].iloc[0])
        km_points.append({"day": day, "fraction_recovered": recovered})

    summary = {
        "n_users": int(len(users)),
        "n_recovered": int(users["event_observed"].sum()),
        "n_censored": int((users["event_observed"] == 0).sum()),
        "median_recovery_days_uncensored": float(
            users.loc[users["event_observed"] == 1, "recovery_days"].median()
        )
        if users["event_observed"].any()
        else None,
        "disasters": sorted(users["disaster"].unique().tolist()),
        "regression": metrics.to_dict(orient="records"),
        "kaplan_meier": km_points,
        "cox_concordance": float(cox.concordance_index_) if cox is not None else None,
        "rsf_fitted": rsf is not None,
        "research_correlations": questions.to_dict(orient="records"),
        "unseen_disaster_holdout": holdout,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "model_metrics.json").write_text(
        json.dumps(_json_ready(summary), indent=2)
    )
    metrics.to_csv(ARTIFACTS_DIR / "regression_cv.csv", index=False)
    importance.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
    questions.to_csv(ARTIFACTS_DIR / "research_correlations.csv", index=False)
    if cox is not None:
        cox.summary.to_csv(ARTIFACTS_DIR / "cox_summary.csv")
    return {
        "metrics": metrics,
        "fitted": fitted,
        "km": km,
        "cox": cox,
        "rsf": rsf,
        "importance": importance,
        "questions": questions,
        "summary": summary,
    }


def run(
    event_spec: str = FOCUS_EVENT,
    raw_path: Path | None = None,
    skip_models: bool = False,
) -> dict:
    events = parse_event_list(event_spec)
    tables = run_feature_pipeline(events, raw_path=raw_path)
    save_processed(tables)
    model_bundle = {}
    if not skip_models:
        if tables["users"].empty:
            print("No eligible users after filters; skipping models.")
        else:
            try:
                model_bundle = run_models(tables["users"])
            except Exception as exc:
                print(f"Modeling failed ({exc}); features and labels were still written.")
    return {"tables": tables, "models": model_bundle}


if __name__ == "__main__":
    run(V15_EVENTS[0])
