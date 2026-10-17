"""Survival models and preprocessing.

Three model families (ridge Cox, elastic-net Cox, random survival forest) are
combined with three feature sets (clinical, PAM50 expression, both).

All feature selection / shrinkage / tuning happens inside `fit_model`, i.e. on the
training data only, so it can be called per CV fold without leakage.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv

from . import config
from .harmonize import clinical_feature_names

FEATURE_SETS = ("clinical", "pam50", "clinical+pam50")
MODEL_NAMES = ("cox", "coxnet", "rsf")


# --- data helpers ------------------------------------------------------------
def make_y(df: pd.DataFrame):
    """Structured array with fields 'event' (bool) and 'time' (float)."""
    return Surv.from_arrays(
        event=df["event"].astype(bool).to_numpy(),
        time=df["time"].to_numpy(dtype=float),
        name_event="event",
        name_time="time",
    )


def feature_columns(feature_set: str, genes: list[str]) -> list[str]:
    cols: list[str] = []
    if "clinical" in feature_set:
        cols += clinical_feature_names() + ["subtype"]
    if "pam50" in feature_set:
        cols += list(genes)
    return cols


def make_preprocessor(feature_set: str, genes: list[str]) -> ColumnTransformer:
    """Clinical: median-impute + scale numerics, one-hot subtype. Genes: passthrough.

    Genes are already z-scored within each cohort (see harmonize.build_cohort), so
    no further scaling is learned from the training cohort.
    """
    parts = []
    if "clinical" in feature_set:
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        parts.append(("num", numeric, clinical_feature_names()))
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["subtype"]))
    if "pam50" in feature_set:
        parts.append(("expr", "passthrough", list(genes)))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


# --- estimators --------------------------------------------------------------
def _estimator(name: str, seed: int, cox_ridge: float = 1.0):
    if name == "cox":
        # ridge-penalised Cox; the penalty also stabilises the one-hot columns
        return CoxPHSurvivalAnalysis(alpha=cox_ridge)
    if name == "coxnet":
        return CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=True, max_iter=100_000)
    if name == "rsf":
        return RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def _tune_coxnet(X, y, feature_set, genes, seed, n_alphas=40, cv_splits=5, n_jobs=1):
    """Pick the elastic-net penalty by inner CV on the training data only."""
    prep = make_preprocessor(feature_set, genes)
    path = CoxnetSurvivalAnalysis(
        l1_ratio=0.5, alpha_min_ratio=0.01, n_alphas=n_alphas, max_iter=100_000
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path.fit(prep.fit_transform(X), y)
    alphas = path.alphas_

    pipe = Pipeline([
        ("prep", make_preprocessor(feature_set, genes)),
        ("model", _estimator("coxnet", seed)),
    ])
    folds = list(StratifiedKFold(cv_splits, shuffle=True, random_state=seed).split(X, y["event"]))
    gcv = GridSearchCV(
        pipe,
        param_grid={"model__alphas": [[a] for a in alphas]},
        cv=folds,
        error_score=0.5,  # very small alphas can fail numerically; treat as chance
        n_jobs=n_jobs,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gcv.fit(X, y)
    return gcv.best_estimator_


def fit_model(model_name: str, feature_set: str, genes: list[str], train: pd.DataFrame,
              seed: int = config.SEED, n_jobs: int = 1) -> Pipeline:
    """Fit preprocessing + model on `train` (a design table with time/event columns)."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {model_name}")
    X = train[feature_columns(feature_set, genes)]
    y = make_y(train)
    if model_name == "coxnet":
        return _tune_coxnet(X, y, feature_set, genes, seed, n_jobs=n_jobs)
    pipe = Pipeline([
        ("prep", make_preprocessor(feature_set, genes)),
        ("model", _estimator(model_name, seed)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    return pipe


# --- prediction helpers ------------------------------------------------------
def predict_risk(pipe: Pipeline, df: pd.DataFrame, feature_set: str, genes: list[str]) -> np.ndarray:
    """Relative risk score; higher = worse prognosis."""
    return pipe.predict(df[feature_columns(feature_set, genes)])


def _step_value(fn, t: float) -> float:
    lo, hi = fn.domain
    if t < lo:
        return 1.0
    return float(fn(min(t, hi)))  # beyond the last observed time, carry the last value


def survival_at(pipe: Pipeline, df: pd.DataFrame, feature_set: str, genes: list[str],
                times) -> np.ndarray:
    """Predicted survival probabilities, shape (n_patients, len(times))."""
    Xt = pipe[:-1].transform(df[feature_columns(feature_set, genes)])
    fns = pipe[-1].predict_survival_function(Xt)
    return np.asarray([[_step_value(fn, t) for t in times] for fn in fns])


def combos(include_rsf: bool = True) -> list[tuple[str, str]]:
    """Sensible (feature_set, model) pairs to compare."""
    out = [("clinical", "cox"), ("pam50", "coxnet"), ("clinical+pam50", "coxnet")]
    if include_rsf:
        out += [("clinical", "rsf"), ("pam50", "rsf"), ("clinical+pam50", "rsf")]
    return out
