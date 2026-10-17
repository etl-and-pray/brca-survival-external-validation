"""Discrimination, calibration and error metrics for survival models."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator

from . import config
from .models import fit_model, make_y, predict_risk, survival_at

# Uno's C is truncated at tau (months). 60 = 5 years: both cohorts still have patients at
# risk there, so the estimate is comparable and not dominated by a handful of late cases.
#
# IPCW metrics (Uno's C, AUC(t), Brier) reweight by the *censoring distribution*. That
# distribution must describe the cohort being evaluated, so the "reference" cohort passed
# below is the evaluated cohort itself, never the training cohort: TCGA and METABRIC have
# very different follow-up, and using TCGA censoring for METABRIC would bias the weights.
UNO_TAU = 60.0


# --- discrimination ----------------------------------------------------------
def harrell_c(y, risk) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def uno_c(y_ref, y_test, risk, tau: float = UNO_TAU) -> float:
    """Uno's IPCW C-index; less sensitive to censoring than Harrell's."""
    tau = min(tau, y_ref["time"].max() - 1e-6, y_test["time"].max() - 1e-6)
    try:
        return float(concordance_index_ipcw(y_ref, y_test, risk, tau=tau)[0])
    except (ValueError, ZeroDivisionError) as e:
        warnings.warn(f"Uno's C failed: {e}")
        return float("nan")


def _align(y_train, y_test, times):
    """sksurv time-dependent metrics need test times inside the train / test range.

    Returns (mask over test patients to keep, usable times).
    """
    mask = y_test["time"] <= y_train["time"].max()
    t_test = y_test["time"][mask]
    usable = np.asarray([t for t in times if t_test.min() <= t < t_test.max()], dtype=float)
    return mask, usable


def time_dependent_auc(y_train, y_test, risk, times) -> pd.Series:
    """Cumulative/dynamic AUC(t). Returns a Series indexed by time (months)."""
    mask, usable = _align(y_train, y_test, times)
    if usable.size == 0:
        return pd.Series(dtype=float)
    try:
        auc, _ = cumulative_dynamic_auc(y_train, y_test[mask], np.asarray(risk)[mask], usable)
    except ValueError as e:
        warnings.warn(f"AUC(t) failed: {e}")
        return pd.Series(np.nan, index=usable)
    return pd.Series(auc, index=usable, name="auc")


# --- error / calibration -----------------------------------------------------
def brier_table(y_train, y_test, surv_matrix, times, y_censor=None) -> pd.DataFrame:
    """Brier score at each time for the model and for a Kaplan-Meier (no covariates) baseline.

    `surv_matrix` has shape (n_test, len(times)) and must correspond to `times`
    exactly as passed in. Also returns the scaled score 1 - B_model / B_null.
    """
    times = np.asarray(times, dtype=float)
    ref = y_train if y_censor is None else y_censor  # censoring distribution for IPCW
    mask, usable = _align(ref, y_test, times)
    keep = np.isin(times, usable)
    if not keep.any():
        return pd.DataFrame()

    yt = y_test[mask]
    S = np.asarray(surv_matrix)[mask][:, keep]

    # baseline: the training-set Kaplan-Meier curve for everyone
    x, s = kaplan_meier_estimator(y_train["event"], y_train["time"])
    s0 = np.array([1.0 if t < x[0] else s[np.searchsorted(x, t, side="right") - 1] for t in usable])
    S0 = np.tile(s0, (len(yt), 1))

    _, b_model = brier_score(ref, yt, S, usable)
    _, b_null = brier_score(ref, yt, S0, usable)
    out = pd.DataFrame({"brier": b_model, "brier_km": b_null}, index=usable)
    out["scaled_brier"] = 1 - out["brier"] / out["brier_km"]
    if len(usable) >= 2:
        out.attrs["ibs"] = float(integrated_brier_score(ref, yt, S, usable))
    return out


def calibration_table(surv_at_t, y, t: float, n_groups: int = 5) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier observed event risk at time t, by predicted-risk group."""
    from lifelines import KaplanMeierFitter

    pred_risk = 1.0 - np.asarray(surv_at_t, dtype=float)
    groups = pd.qcut(pred_risk, n_groups, labels=False, duplicates="drop")
    rows = []
    for g in np.unique(groups):
        m = groups == g
        kmf = KaplanMeierFitter().fit(y["time"][m], y["event"][m])
        obs = 1.0 - float(kmf.survival_function_at_times(t).iloc[0])
        rows.append({"group": int(g) + 1, "n": int(m.sum()), "events": int(y["event"][m].sum()),
                     "predicted": float(pred_risk[m].mean()), "observed": obs})
    return pd.DataFrame(rows)


# --- uncertainty -------------------------------------------------------------
def bootstrap_c_index(y, risk, n_boot: int = 1000, seed: int = config.SEED, level: float = 0.95):
    """Percentile bootstrap CI for Harrell's C over *test patients*.

    This reflects sampling variability of the test set for a fixed fitted model;
    it does not include variability from re-training.
    """
    risk = np.asarray(risk)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(harrell_c(y[idx], risk[idx]))
        except Exception:  # no comparable pairs in this resample
            continue
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return harrell_c(y, risk), float(lo), float(hi)


# --- internal validation -----------------------------------------------------
def repeated_cv(model_name: str, feature_set: str, df: pd.DataFrame, genes: list[str],
                n_splits: int = 5, n_repeats: int = 3, seed: int = config.SEED,
                n_jobs: int = 1) -> pd.DataFrame:
    """Repeated stratified K-fold CV inside one cohort. One row per fold.

    Tuning (Coxnet) happens inside each training fold, so test folds never
    influence hyperparameters or gene shrinkage.
    """
    y_all = make_y(df)
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for i, (tr, te) in enumerate(splitter.split(df, y_all["event"])):
        train, test = df.iloc[tr], df.iloc[te]
        pipe = fit_model(model_name, feature_set, genes, train, seed=seed, n_jobs=n_jobs)
        risk = predict_risk(pipe, test, feature_set, genes)
        y_tr, y_te = make_y(train), make_y(test)
        rows.append({
            "model": model_name, "features": feature_set,
            "repeat": i // n_splits, "fold": i % n_splits,
            "n_test": len(test), "events_test": int(y_te["event"].sum()),
            "harrell_c": harrell_c(y_te, risk),
            "uno_c": uno_c(y_all, y_te, risk),
        })
    return pd.DataFrame(rows)


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of fold metrics per (features, model). SD is across overlapping folds,
    so it describes fold-to-fold spread, not a standard error."""
    return (cv.groupby(["features", "model"])[["harrell_c", "uno_c"]]
              .agg(["mean", "std"]).round(3))


# --- external validation -----------------------------------------------------
def evaluate_external(model_name: str, feature_set: str, train: pd.DataFrame, test: pd.DataFrame,
                      genes: list[str], times=None, seed: int = config.SEED, n_jobs: int = 1) -> dict:
    """Fit on `train`, evaluate on an independent cohort `test` with no refitting."""
    times = np.asarray(times if times is not None else config.EVAL_TIMES_MONTHS, dtype=float)
    pipe = fit_model(model_name, feature_set, genes, train, seed=seed, n_jobs=n_jobs)
    y_tr, y_te = make_y(train), make_y(test)

    risk = predict_risk(pipe, test, feature_set, genes)
    c, lo, hi = bootstrap_c_index(y_te, risk, seed=seed)
    surv = survival_at(pipe, test, feature_set, genes, times)

    # a denser grid gives a more stable integrated Brier score
    grid = np.arange(12, config.HORIZON_MONTHS - 11, 12, dtype=float)
    surv_grid = survival_at(pipe, test, feature_set, genes, grid)

    return {
        "pipe": pipe,
        "risk": risk,
        "times": times,
        "surv": surv,
        "summary": {
            "model": model_name, "features": feature_set,
            "harrell_c": c, "ci_low": lo, "ci_high": hi,
            "uno_c": uno_c(y_te, y_te, risk),
            "n_test": len(test), "events_test": int(y_te["event"].sum()),
        },
        "auc": time_dependent_auc(y_te, y_te, risk, times),
        "brier": brier_table(y_tr, y_te, surv_grid, grid, y_censor=y_te),
        "y_test": y_te,
    }
