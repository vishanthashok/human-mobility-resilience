"""Export static result figures for the GitHub README."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR  # noqa: E402
from src.models import kaplan_meier  # noqa: E402
from src.plots import mean_mds_by_day  # noqa: E402

OUT = ROOT / "docs" / "figures"
ART = ROOT / "artifacts"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_mds_overall(disrupted: pd.DataFrame) -> None:
    window = disrupted.loc[disrupted["day_relative"].between(-14, 21)]
    daily = (
        window.groupby("day_relative")["mds"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.fill_between(
        daily["day_relative"], daily["q25"], daily["q75"], alpha=0.18, color="#1f4e79", label="IQR"
    )
    ax.plot(
        daily["day_relative"], daily["median"], color="#1f4e79", lw=2.4, marker="o", ms=4, label="Median"
    )
    ax.axvline(0, color="#666666", ls="--", lw=1.2)
    ax.set_xlabel("Days relative to disaster")
    ax.set_ylabel("Mobility Disruption Score")
    ax.set_title("Median Mobility Disruption Score over time")
    ax.legend(frameon=False)
    _save(fig, "mds_over_time.png")


def plot_mds_by_type(disrupted: pd.DataFrame) -> None:
    window = disrupted.loc[disrupted["day_relative"].between(-10, 14)]
    daily = (
        window.groupby(["disaster_type", "day_relative"], as_index=False)["mds"]
        .median()
    )
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.lineplot(
        data=daily,
        x="day_relative",
        y="mds",
        hue="disaster_type",
        marker="o",
        ax=ax,
        linewidth=2,
    )
    ax.axvline(0, color="#666666", ls="--", lw=1.2)
    ax.set_xlabel("Days relative to disaster")
    ax.set_ylabel("Median Mobility Disruption Score")
    ax.set_title("Median disruption by disaster type")
    ax.legend(title="Type", frameon=False)
    _save(fig, "mds_by_disaster_type.png")


def plot_kaplan(users: pd.DataFrame) -> None:
    km = kaplan_meier(users)
    sf = km.survival_function_.reset_index()
    time_col, surv_col = sf.columns[0], sf.columns[1]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(sf[time_col], 1.0 - sf[surv_col], color="#0b6e4f", lw=2.6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Days since disaster")
    ax.set_ylabel("Fraction recovered")
    ax.set_title("Kaplan–Meier recovery curve")
    for day, label_y in ((3, 0.18), (7, 0.28), (14, 0.38), (21, 0.48)):
        idx = sf.index[sf[time_col] <= day]
        if len(idx) == 0:
            continue
        recovered = float(1.0 - sf.loc[idx[-1], surv_col])
        ax.scatter([day], [recovered], color="#0b6e4f", zorder=3)
        ax.annotate(f"day {day}: {recovered:.0%}", xy=(day, recovered), xytext=(day + 0.6, label_y), fontsize=10)
    _save(fig, "kaplan_meier.png")


def plot_importance() -> None:
    imp = pd.read_csv(ART / "feature_importance.csv")
    rf = imp.loc[imp["model"] == "random_forest"].copy()
    rf["feature"] = (
        rf["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    rf = rf.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.barh(rf["feature"], rf["importance"], color="#1f4e79")
    ax.set_xlabel("Impurity importance")
    ax.set_title("Random forest feature importance for days-to-recovery")
    _save(fig, "feature_importance.png")


def plot_recovery_by_disaster() -> None:
    counts = pd.read_csv(ART / "recovery_counts.csv")
    counts["n_censored"] = counts["n_users"] - counts["n_recovered"]
    counts = counts.sort_values("n_users")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.barh(counts["disaster"], counts["n_recovered"], color="#0b6e4f", label="Recovered")
    ax.barh(
        counts["disaster"],
        counts["n_censored"],
        left=counts["n_recovered"],
        color="#c5c5c5",
        label="Censored",
    )
    ax.set_xlabel("Users passing observation filters")
    ax.set_title("Recovery vs censoring by disaster")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "recovery_by_disaster.png")


def plot_model_comparison() -> None:
    metrics = pd.read_csv(ART / "regression_cv.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.barplot(data=metrics, x="model", y="c_index_mean", ax=axes[0], color="#1f4e79")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("C-index")
    axes[0].set_xlabel("")
    axes[0].set_title("Ranking recovery times")
    sns.barplot(data=metrics, x="model", y="rmse_uncensored_mean", ax=axes[1], color="#b85c38")
    axes[1].set_ylabel("RMSE (uncensored, days)")
    axes[1].set_xlabel("")
    axes[1].set_title("Error on observed recoveries")
    fig.suptitle("Leakage-safe user-level cross-validation", y=1.02)
    _save(fig, "model_comparison.png")


def plot_research_correlations() -> None:
    corr = pd.read_csv(ART / "research_correlations.csv")
    labels = {
        "baseline_radius": "Baseline radius of gyration",
        "day0_max_displacement": "Day-0 max displacement",
        "day0_mds": "Day-0 disruption score",
        "baseline_entropy": "Baseline movement entropy",
    }
    corr["label"] = corr["feature"].map(labels).fillna(corr["feature"])
    corr = corr.sort_values("spearman_vs_recovery_days")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#1f4e79" if v > 0 else "#b85c38" for v in corr["spearman_vs_recovery_days"]]
    ax.barh(corr["label"], corr["spearman_vs_recovery_days"], color=colors)
    ax.axvline(0, color="#444444", lw=1)
    ax.set_xlabel("Spearman ρ vs days-to-recovery (uncensored users)")
    ax.set_title("What predicts longer recovery?")
    _save(fig, "research_correlations.png")


def plot_example_trajectory(tweets: pd.DataFrame, users: pd.DataFrame) -> None:
    candidates = users.loc[users["disaster"] == "08_Rammasun_Manila"].copy()
    if candidates.empty:
        candidates = users
    uid = str(candidates.sort_values("day0_mds", ascending=False)["user_anon"].iloc[0])
    disaster = str(candidates.sort_values("day0_mds", ascending=False)["disaster"].iloc[0])
    sub = tweets.loc[
        (tweets["disaster"] == disaster) & (tweets["user_anon"].astype(str) == uid)
    ].copy()
    sub["period"] = np.where(
        sub["day_relative"] < 0,
        "before",
        np.where(sub["day_relative"] <= 2, "during (0–2)", "after"),
    )
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    palette = {"before": "#4c6b8a", "during (0–2)": "#c0392b", "after": "#1e8449"}
    for period, g in sub.groupby("period", observed=False):
        ax.scatter(
            g["longitude"],
            g["latitude"],
            s=28,
            alpha=0.75,
            label=period,
            color=palette.get(str(period), "#333333"),
        )
    ax.set_xlabel("Longitude (anonymized)")
    ax.set_ylabel("Latitude (shifted)")
    ax.set_title(f"Example trajectory — user {uid} ({disaster})")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, "example_trajectory.png")


def plot_results_board(
    disrupted: pd.DataFrame, users: pd.DataFrame
) -> None:
    """One composite image that reads like a results screenshot."""
    km = kaplan_meier(users)
    sf = km.survival_function_.reset_index()
    time_col, surv_col = sf.columns[0], sf.columns[1]
    window = disrupted.loc[disrupted["day_relative"].between(-14, 21)]
    mds = (
        window.groupby("day_relative")["mds"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    metrics = pd.read_csv(ART / "regression_cv.csv")
    imp = pd.read_csv(ART / "feature_importance.csv")
    rf = imp.loc[imp["model"] == "random_forest"].copy()
    rf["feature"] = rf["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    rf = rf.head(8).iloc[::-1]

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.fill_between(mds["day_relative"], mds["q25"], mds["q75"], alpha=0.18, color="#1f4e79")
    ax1.plot(mds["day_relative"], mds["median"], color="#1f4e79", lw=2.2)
    ax1.axvline(0, color="#666666", ls="--", lw=1)
    ax1.set_title("Median disruption over time")
    ax1.set_xlabel("Days relative to disaster")
    ax1.set_ylabel("MDS")

    ax2.plot(sf[time_col], 1.0 - sf[surv_col], color="#0b6e4f", lw=2.4)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_title("Kaplan–Meier recovery")
    ax2.set_xlabel("Days since disaster")
    ax2.set_ylabel("Fraction recovered")

    ax3.barh(rf["feature"], rf["importance"], color="#1f4e79")
    ax3.set_title("RF feature importance")
    ax3.set_xlabel("Importance")

    ax4.bar(metrics["model"], metrics["c_index_mean"], color="#1f4e79")
    ax4.set_ylim(0.5, 1.0)
    ax4.set_title("C-index (user GroupKFold)")
    ax4.set_ylabel("C-index")

    fig.suptitle(
        "Human mobility resilience — 4.69M tweets, 15 disasters, 7,402 users",
        fontsize=16,
        y=0.98,
    )
    _save(fig, "results_overview.png")


def main() -> None:
    _style()
    disrupted = pd.read_parquet(PROCESSED_DIR / "disrupted.parquet")
    users = pd.read_parquet(PROCESSED_DIR / "users.parquet")
    tweets = pd.read_parquet(PROCESSED_DIR / "filtered_tweets.parquet")
    plot_mds_overall(disrupted)
    plot_mds_by_type(disrupted)
    plot_kaplan(users)
    plot_importance()
    plot_recovery_by_disaster()
    plot_model_comparison()
    plot_research_correlations()
    plot_example_trajectory(tweets, users)
    plot_results_board(disrupted, users)
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
