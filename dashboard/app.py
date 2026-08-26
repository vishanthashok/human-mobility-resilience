"""Streamlit dashboard for mobility disruption and recovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, PROCESSED_DIR  # noqa: E402
from src.plots import (  # noqa: E402
    mean_mds_by_day,
    plot_feature_importance,
    plot_kaplan_meier,
    plot_mds_over_time,
    plot_trajectories,
)
from src.models import kaplan_meier  # noqa: E402


st.set_page_config(page_title="Mobility Resilience", layout="wide")


@st.cache_data
def load_processed():
    users_path = PROCESSED_DIR / "users.parquet"
    disrupted_path = PROCESSED_DIR / "disrupted.parquet"
    if not users_path.exists() or not disrupted_path.exists():
        return None
    users = pd.read_parquet(users_path)
    disrupted = pd.read_parquet(disrupted_path)
    coverage = (
        pd.read_parquet(PROCESSED_DIR / "coverage.parquet")
        if (PROCESSED_DIR / "coverage.parquet").exists()
        else pd.DataFrame()
    )
    metrics = {}
    metrics_path = ARTIFACTS_DIR / "model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    importance = (
        pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
        if (ARTIFACTS_DIR / "feature_importance.csv").exists()
        else pd.DataFrame()
    )
    tweets_path = PROCESSED_DIR / "filtered_tweets.parquet"
    tweets = pd.read_parquet(tweets_path) if tweets_path.exists() else None
    return {
        "users": users,
        "disrupted": disrupted,
        "coverage": coverage,
        "metrics": metrics,
        "importance": importance,
        "tweets": tweets,
    }


def main() -> None:
    st.title("Human Mobility Resilience During Natural Disasters")
    st.caption(
        "Disruption and time-to-recovery from geotagged tweets. "
        "Coordinates are anonymized — maps are relative, not geographic."
    )

    data = load_processed()
    if data is None:
        st.error("Processed tables not found. Run the pipeline first.")
        st.code("python scripts/download_data.py\npython scripts/run_pipeline.py --events 14_Napa")
        return

    users = data["users"]
    disrupted = data["disrupted"]
    disasters = sorted(users["disaster"].unique().tolist())
    selected = st.sidebar.multiselect("Disasters", disasters, default=disasters)
    if not selected:
        st.warning("Select at least one disaster.")
        return

    users_s = users.loc[users["disaster"].isin(selected)]
    disrupted_s = disrupted.loc[disrupted["disaster"].isin(selected)]

    n_users = len(users_s)
    n_rec = int(users_s["event_observed"].sum())
    n_cen = n_users - n_rec
    median_rec = users_s.loc[users_s["event_observed"] == 1, "recovery_days"].median()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{n_users:,}")
    c2.metric("Recovered", f"{n_rec:,}")
    c3.metric("Censored", f"{n_cen:,}")
    c4.metric("Median recovery days", "—" if pd.isna(median_rec) else f"{median_rec:.0f}")

    st.subheader("Average Mobility Disruption Score")
    mds = mean_mds_by_day(disrupted_s)
    st.plotly_chart(plot_mds_over_time(mds), use_container_width=True)

    st.subheader("Kaplan–Meier recovery curve")
    km = kaplan_meier(users_s)
    st.plotly_chart(plot_kaplan_meier(km), use_container_width=True)

    metrics = data["metrics"]
    if metrics.get("regression"):
        st.subheader("Recovery-time models (user-level CV)")
        st.dataframe(pd.DataFrame(metrics["regression"]), use_container_width=True)
        st.caption(
            "Features use baseline behavior plus the first 24 hours only. "
            "RMSE/MAE are reported on uncensored users; C-index uses all users."
        )
        if metrics.get("unseen_disaster_holdout") and not metrics["unseen_disaster_holdout"].get("skipped"):
            st.subheader("Unseen-disaster holdout")
            st.json(metrics["unseen_disaster_holdout"])

    importance = data["importance"]
    if not importance.empty:
        st.subheader("Feature importance")
        model_names = importance["model"].unique().tolist()
        model = st.selectbox("Model", model_names)
        st.plotly_chart(
            plot_feature_importance(importance.loc[importance["model"] == model]),
            use_container_width=True,
        )

    if metrics.get("research_correlations"):
        st.subheader("Research questions")
        st.dataframe(pd.DataFrame(metrics["research_correlations"]), use_container_width=True)

    tweets = data["tweets"]
    if tweets is not None and not tweets.empty:
        st.subheader("Example user trajectory (anonymized coordinates)")
        d0 = selected[0]
        sample_users = (
            users_s.loc[users_s["disaster"] == d0, "user_anon"].astype(str).head(25).tolist()
        )
        if sample_users:
            uid = st.selectbox("User", sample_users)
            tsub = tweets.loc[tweets["disaster"] == d0]
            st.plotly_chart(
                plot_trajectories(tsub, uid, d0),
                use_container_width=True,
            )

    if not data["coverage"].empty:
        st.subheader("Event clock coverage")
        st.dataframe(data["coverage"], use_container_width=True)


if __name__ == "__main__":
    main()
