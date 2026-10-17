"""Bring the two cohorts to a common schema: clinical covariates, endpoint, PAM50 expression."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .data import load_clinical, load_expression

SUBTYPE_LEVELS = ["LumA", "LumB", "Her2", "Basal", "Normal"]
STAGE_ROMAN = {"IV": 4, "III": 3, "II": 2, "I": 1}


# --- field parsers -----------------------------------------------------------
def parse_status(series: pd.Series) -> pd.Series:
    """cBioPortal status looks like '1:DECEASED' / '0:LIVING'. Return 1/0 (NaN if missing)."""
    code = series.astype("string").str.split(":", n=1).str[0]
    return pd.to_numeric(code, errors="coerce")


def parse_tcga_stage(series: pd.Series) -> pd.Series:
    """'STAGE IIA' -> 2, 'STAGE IV' -> 4, 'STAGE X' / missing -> NaN."""
    def one(v):
        if not isinstance(v, str):
            return np.nan
        m = re.match(r"\s*STAGE\s+(IV|III|II|I)", v.upper())
        return STAGE_ROMAN[m.group(1)] if m else np.nan
    return series.map(one)


def parse_subtype(series: pd.Series) -> pd.Series:
    """Map both cohorts' PAM50-style labels to one vocabulary.

    TCGA: 'BRCA_LumA'. METABRIC (CLAUDIN_SUBTYPE): 'LumA', 'claudin-low', 'NC'.
    Anything outside the five PAM50 classes becomes 'Other'; missing -> 'Unknown'.
    """
    s = series.astype("string").str.replace("BRCA_", "", regex=False)
    out = s.where(s.isin(SUBTYPE_LEVELS), "Other")
    return out.where(series.notna(), "Unknown").astype(str)


# --- one cohort --------------------------------------------------------------
def _clinical_frame(key: str, endpoint: str) -> pd.DataFrame:
    raw = load_clinical(key)
    time_col, status_col = config.ENDPOINT_COLUMNS[endpoint][key]

    age_col = {"tcga": "AGE", "metabric": "AGE_AT_DIAGNOSIS"}[key]
    subtype_col = {"tcga": "SUBTYPE", "metabric": "CLAUDIN_SUBTYPE"}[key]

    out = pd.DataFrame({
        "patient_id": raw["PATIENT_ID"],
        "time": pd.to_numeric(raw[time_col], errors="coerce"),
        "event": parse_status(raw[status_col]),
        "age": pd.to_numeric(raw[age_col], errors="coerce"),
        "subtype": parse_subtype(raw[subtype_col]),
    })
    if key == "tcga":
        out["stage"] = parse_tcga_stage(raw["AJCC_PATHOLOGIC_TUMOR_STAGE"])
    else:
        out["stage"] = pd.to_numeric(raw["TUMOR_STAGE"], errors="coerce")
    out["cohort"] = key
    out = out.set_index("patient_id")

    # keep patients with a usable endpoint
    out = out.dropna(subset=["time", "event"])
    out = out[out["time"] > 0]

    # keep the untruncated follow-up for descriptive statistics (Table 1)
    out["time_raw"] = out["time"]
    out["event_raw"] = out["event"].astype(int)

    # administrative censoring at a common horizon
    h = config.HORIZON_MONTHS
    late = out["time"] > h
    out.loc[late, "event"] = 0
    out.loc[late, "time"] = h
    out["event"] = out["event"].astype(int)
    return out


def _expression_frame(key: str, patients: pd.Index) -> pd.DataFrame:
    raw = load_clinical(key)
    sample_to_patient = dict(zip(raw["SAMPLE_ID"], raw["PATIENT_ID"]))
    expr = load_expression(key, config.PAM50, sample_to_patient)

    if key == "tcga":
        expr = np.log2(expr.clip(lower=0) + 1)  # RSEM counts -> log scale

    expr = expr.reindex(patients)
    # drop genes missing in too many patients, median-impute the rest
    keep = expr.columns[expr.isna().mean() < 0.05]
    expr = expr[keep]
    expr = expr.fillna(expr.median())
    return expr


def build_cohort(key: str, endpoint: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clinical, expression) for one cohort, aligned on patient_id.

    Expression is z-scored per gene *within the cohort*. This is a pragmatic way
    to remove platform scale differences (RNA-seq vs microarray). It uses the
    cohort's own expression distribution (never outcomes), but note that it means
    the external cohort is normalised with its own statistics.
    """
    endpoint = endpoint or config.ENDPOINT
    clin = _clinical_frame(key, endpoint)
    expr = _expression_frame(key, clin.index)

    have_expr = expr.notna().all(axis=1)
    clin, expr = clin.loc[have_expr], expr.loc[have_expr]

    expr = (expr - expr.mean()) / expr.std(ddof=0)
    return clin, expr


def build_datasets(endpoint: str | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """Build both cohorts on a shared gene set and save processed tables."""
    endpoint = endpoint or config.ENDPOINT
    data = {}
    for key in config.STUDIES:
        clin, expr = build_cohort(key, endpoint)
        data[key] = {"clin": clin, "expr": expr}

    shared = sorted(set.intersection(*(set(d["expr"].columns) for d in data.values())))
    missing = [g for g in config.PAM50 if g not in shared]
    if missing:
        print(f"PAM50 genes not available in both cohorts: {missing}")
    for d in data.values():
        d["expr"] = d["expr"][shared]

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for key, d in data.items():
        d["clin"].to_csv(config.DATA_PROCESSED / f"{key}_clinical_{endpoint}.csv")
        d["expr"].to_csv(config.DATA_PROCESSED / f"{key}_pam50.csv")
    return data


def load_processed(endpoint: str | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """Read the tables written by `build_datasets` (fast; no raw files needed)."""
    endpoint = endpoint or config.ENDPOINT
    data = {}
    for key in config.STUDIES:
        clin = pd.read_csv(config.DATA_PROCESSED / f"{key}_clinical_{endpoint}.csv", index_col="patient_id")
        expr = pd.read_csv(config.DATA_PROCESSED / f"{key}_pam50.csv", index_col=0)
        data[key] = {"clin": clin, "expr": expr}
    return data


def model_frame(clin: pd.DataFrame, expr: pd.DataFrame) -> pd.DataFrame:
    """One design table: clinical columns + gene columns (+ time/event for building y)."""
    cols = ["age", "subtype", "time", "event"]
    if config.USE_STAGE:
        cols.insert(1, "stage")
    return clin[cols].join(expr)


def clinical_feature_names() -> list[str]:
    return ["age", "stage"] if config.USE_STAGE else ["age"]
