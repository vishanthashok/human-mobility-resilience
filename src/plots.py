"""Plot helpers for disruption curves, survival, trajectories, and importance."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter


def mean_mds_by_day(disrupted: pd.DataFrame) -> pd.DataFrame:
    summary = (
        disrupted.groupby(["disaster", "day_relative"], as_index=False)
        .agg(
            mean_mds=("mds", "mean"),
            median_mds=("mds", "median"),
            n_users=("user_anon", "nunique"),
        )
        .sort_values(["disaster", "day_relative"])
    )
    return summary


def plot_mds_over_time(summary: pd.DataFrame, title: str | None = None) -> go.Figure:
    fig = px.line(
        summary,
        x="day_relative",
        y="mean_mds",
        color="disaster",
        markers=True,
        title=title or "Average Mobility Disruption Score over time",
        labels={
            "day_relative": "Days relative to disaster",
            "mean_mds": "Mean Mobility Disruption Score",
            "disaster": "Disaster",
        },
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_white", legend_title_text="Disaster")
    return fig


def plot_kaplan_meier(km: KaplanMeierFitter, title: str | None = None) -> go.Figure:
    sf = km.survival_function_.reset_index()
    time_col = sf.columns[0]
    surv_col = sf.columns[1]
    recovered = 1.0 - sf[surv_col]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sf[time_col],
            y=recovered,
            mode="lines",
            name="Fraction recovered",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title or "Kaplan–Meier recovery curve",
        xaxis_title="Days since disaster",
        yaxis_title="Fraction recovered",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def plot_trajectories(
    tweets: pd.DataFrame,
    user_anon: str,
    disaster: str,
    title: str | None = None,
) -> go.Figure:
    """Anonymized coordinate space — not a geographic basemap."""
    sub = tweets.loc[
        (tweets["disaster"] == disaster) & (tweets["user_anon"].astype(str) == str(user_anon))
    ].copy()
    sub["period"] = pd.cut(
        sub["day_relative"],
        bins=[-10_000, -1, 2, 10_000],
        labels=["before", "during (0–2)", "after"],
    )
    fig = px.scatter(
        sub,
        x="longitude",
        y="latitude",
        color="period",
        hover_data=["time", "day_relative"],
        title=title
        or f"Anonymized trajectory for user {user_anon} ({disaster})",
        labels={"longitude": "Longitude (anonymized)", "latitude": "Latitude"},
    )
    fig.update_layout(template="plotly_white", yaxis_scaleanchor="x")
    return fig


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = importance.head(top_n).iloc[::-1]
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature importance ({importance['model'].iloc[0]})",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(template="plotly_white")
    return fig
