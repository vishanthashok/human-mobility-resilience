"""Leakage-safe recovery-time models: regression plus survival analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import N_CV_FOLDS, RANDOM_STATE

try:
    from xgboost import XGBRegressor
except Exception:  # OpenMP / libomp is often missing on macOS
    XGBRegressor = None

NUMERIC_FEATURES = [
    "baseline_radius",
    "baseline_mobility_distance",
    "baseline_displacement",
    "baseline_unique_locations",
    "baseline_max_displacement",
    "baseline_n_days",
    "baseline_n_obs",
    "baseline_entropy",
    "day0_mds",
    "day0_max_displacement",
    "day0_n_obs",
]
CATEGORICAL_FEATURES = ["disaster_type"]
TARGET = "recovery_days"


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def regression_estimators() -> dict[str, Any]:
    estimators: dict[str, Any] = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        estimators["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return estimators


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression_fold(
    y_true: np.ndarray, y_pred: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    """RMSE/MAE on uncensored users; C-index uses all users (higher pred = slower recovery)."""
    y_pred = np.clip(y_pred, 0, None)
    mask = observed.astype(bool)
    try:
        c_index = float(concordance_index(y_true, y_pred, event_observed=observed))
    except Exception:
        c_index = float("nan")
    out = {
        "n": int(len(y_true)),
        "n_uncensored": int(mask.sum()),
        "c_index": c_index,
    }
    if mask.sum() >= 2:
        out["rmse_uncensored"] = rmse(y_true[mask], y_pred[mask])
        out["mae_uncensored"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    else:
        out["rmse_uncensored"] = float("nan")
        out["mae_uncensored"] = float("nan")
    return out


def cross_validate_regressors(
    users: pd.DataFrame, n_splits: int = N_CV_FOLDS
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    groups = users["user_id"].to_numpy()
    n_splits = min(n_splits, users["user_id"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    X = users[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = users[TARGET].to_numpy(dtype=float)
    observed = users["event_observed"].to_numpy(dtype=int)
    y_log = np.log1p(y)

    rows = []
    fitted: dict[str, Pipeline] = {}
    for name, estimator in regression_estimators().items():
        oof = np.zeros(len(users), dtype=float)
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            pipe = Pipeline(
                steps=[("prep", _preprocessor()), ("model", estimator)]
            )
            if name == "linear":
                pipe.fit(X.iloc[train_idx], y_log[train_idx])
                pred = np.expm1(pipe.predict(X.iloc[test_idx]))
            else:
                pipe.fit(X.iloc[train_idx], y[train_idx])
                pred = pipe.predict(X.iloc[test_idx])
            pred = np.clip(pred, 0, None)
            oof[test_idx] = pred
            metrics = evaluate_regression_fold(y[test_idx], pred, observed[test_idx])
            metrics.update({"model": name, "fold": fold})
            fold_metrics.append(metrics)
        fitted[name] = Pipeline(
            steps=[("prep", _preprocessor()), ("model", estimator)]
        )
        if name == "linear":
            fitted[name].fit(X, y_log)
        else:
            fitted[name].fit(X, y)
        summary = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "model": name,
                "rmse_uncensored_mean": float(summary["rmse_uncensored"].mean()),
                "mae_uncensored_mean": float(summary["mae_uncensored"].mean()),
                "c_index_mean": float(summary["c_index"].mean()),
                "n_users": int(len(users)),
                "n_uncensored": int(observed.sum()),
            }
        )
        users[f"pred_{name}"] = oof
    return pd.DataFrame(rows), fitted


def kaplan_meier(users: pd.DataFrame) -> KaplanMeierFitter:
    km = KaplanMeierFitter()
    km.fit(
        durations=users[TARGET],
        event_observed=users["event_observed"],
        label="recovery",
    )
    return km


def fit_cox(users: pd.DataFrame) -> CoxPHFitter:
    cols = NUMERIC_FEATURES + [TARGET, "event_observed"]
    cox_df = users[cols].copy()
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna()
    if cox_df["event_observed"].sum() < 5:
        raise ValueError("Not enough uncensored recoveries to fit Cox PH.")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col=TARGET, event_col="event_observed")
    return cph


def try_random_survival_forest(users: pd.DataFrame, n_estimators: int = 200):
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except ImportError:
        return None, None
    X = users[NUMERIC_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = Surv.from_arrays(
        event=users["event_observed"].astype(bool).to_numpy(),
        time=users[TARGET].to_numpy(dtype=float),
    )
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=8,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    try:
        rsf.fit(X, y)
    except Exception:
        return None, None
    return rsf, X.columns.tolist()


def feature_importance_table(pipe: Pipeline, model_name: str) -> pd.DataFrame:
    prep: ColumnTransformer = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    names = prep.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        kind = "impurity_importance"
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        kind = "coefficient"
    else:
        return pd.DataFrame(columns=["model", "feature", "importance", "kind"])
    return pd.DataFrame(
        {
            "model": model_name,
            "feature": names,
            "importance": values,
            "kind": kind,
        }
    ).sort_values("importance", key=np.abs, ascending=False)


def evaluate_unseen_disaster(
    users: pd.DataFrame, test_disaster: str | None = None
) -> dict | None:
    """Train on all but one disaster; evaluate on a disaster never seen in training."""
    counts = users.groupby("disaster").size().sort_values(ascending=False)
    if len(counts) < 3:
        return None
    if test_disaster is None:
        # Prefer a mid-sized held-out event so train still has data.
        test_disaster = counts.index[len(counts) // 2]
    train = users.loc[users["disaster"] != test_disaster]
    test = users.loc[users["disaster"] == test_disaster]
    if len(train) < 30 or len(test) < 15:
        return {
            "test_disaster": test_disaster,
            "skipped": True,
            "reason": "too few users in train or test",
        }
    X_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    pipe = Pipeline(
        steps=[
            ("prep", _preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(train[X_cols], train[TARGET])
    pred = np.clip(pipe.predict(test[X_cols]), 0, None)
    metrics = evaluate_regression_fold(
        test[TARGET].to_numpy(dtype=float),
        pred,
        test["event_observed"].to_numpy(dtype=int),
    )
    metrics.update(
        {
            "test_disaster": test_disaster,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "skipped": False,
        }
    )
    return metrics


def research_correlations(users: pd.DataFrame) -> pd.DataFrame:
    """Simple associations for the research questions (uncensored users)."""
    observed = users.loc[users["event_observed"] == 1]
    pairs = [
        ("baseline_radius", "Do highly mobile people recover faster?"),
        ("day0_max_displacement", "Does initial displacement predict longer recovery?"),
        ("day0_mds", "Does initial disruption magnitude predict longer recovery?"),
        ("baseline_entropy", "Does more diverse routine movement change recovery?"),
    ]
    rows = []
    for col, question in pairs:
        if observed[col].nunique() < 3:
            corr = float("nan")
        else:
            corr = float(observed[col].corr(observed[TARGET], method="spearman"))
        rows.append({"feature": col, "spearman_vs_recovery_days": corr, "question": question})
    return pd.DataFrame(rows)
