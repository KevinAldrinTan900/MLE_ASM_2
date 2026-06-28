# utils/ml_utils.py
#
# Shared helpers for the ML side of the pipeline: loading the gold feature /
# label stores produced by the upstream datamart into pandas, decoding Spark ML
# vector columns, and joining features to labels with the correct 6-month
# label-maturity offset.

import os
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATAMART = os.path.join(PROJECT_ROOT, "datamart")
GOLD = os.path.join(DATAMART, "gold")
FEATURE_STORE = os.path.join(GOLD, "feature_store")
LABEL_STORE = os.path.join(GOLD, "label_store")
PREDICTION_STORE = os.path.join(GOLD, "model_predictions")
MONITOR_STORE = os.path.join(GOLD, "model_monitoring")
MODEL_BANK = os.path.join(PROJECT_ROOT, "model_bank")

LABEL_MOB_MONTHS = 6  # label for feature snapshot fd matures at fd + 6 months

ID_COLS = {"Customer_ID", "loan_id", "feature_snapshot_date", "label_snapshot_date", "snapshot_date", "label"}


def _decode_sparkml_vector(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Expand a Spark ML vector column (parquet struct of
    {type, size, indices, values}) into dense numeric columns."""
    def to_dense(v):
        if v is None:
            return None
        if v.get("type") == 1 or v.get("indices") is None:  # dense
            return np.asarray(v["values"], dtype=float)
        dense = np.zeros(int(v["size"]), dtype=float)
        idx = np.asarray(v["indices"], dtype=int)
        if idx.size:
            dense[idx] = np.asarray(v["values"], dtype=float)
        return dense

    dense_rows = series.map(to_dense)
    dim = int(max((len(r) for r in dense_rows if r is not None), default=0))
    mat = np.full((len(series), dim), np.nan)
    for i, r in enumerate(dense_rows):
        if r is not None:
            mat[i, : len(r)] = r
    return pd.DataFrame(mat, columns=[f"{prefix}_{i}" for i in range(dim)], index=series.index)


def load_feature_store() -> pd.DataFrame:
    """Load the gold feature store into pandas with vector columns expanded."""
    df = pads.dataset(FEATURE_STORE).to_table().to_pandas()

    vec_cols = [c for c in df.columns if isinstance(df[c].iloc[0], dict) and "values" in df[c].iloc[0]]
    for c in vec_cols:
        expanded = _decode_sparkml_vector(df[c], c.replace("_vec", ""))
        df = pd.concat([df.drop(columns=[c]), expanded], axis=1)

    df["feature_snapshot_date"] = pd.to_datetime(df["feature_snapshot_date"]).dt.date
    return df


def load_label_store() -> pd.DataFrame:
    df = pads.dataset(LABEL_STORE).to_table().to_pandas()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    # one label per customer per snapshot (defensive de-dup; takes worst case)
    df = (
        df.sort_values("label", ascending=False)
        .drop_duplicates(subset=["Customer_ID", "snapshot_date"])
        .rename(columns={"snapshot_date": "label_snapshot_date"})
    )
    return df[["Customer_ID", "label_snapshot_date", "label"]]


def join_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach the matured label for each feature snapshot (fd + 6 months)."""
    feats = features.copy()
    feats["label_snapshot_date"] = feats["feature_snapshot_date"].map(
        lambda d: d + relativedelta(months=LABEL_MOB_MONTHS)
    )
    return feats.merge(labels, on=["Customer_ID", "label_snapshot_date"], how="inner")


def feature_columns(df: pd.DataFrame) -> list:
    """All numeric model-input columns (excludes ids, dates, label)."""
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def psi(expected: np.ndarray, actual: np.ndarray, bin_edges: np.ndarray) -> float:
    """Population Stability Index between two score distributions."""
    e_counts, _ = np.histogram(expected, bins=bin_edges)
    a_counts, _ = np.histogram(actual, bins=bin_edges)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-4, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def parse_date(s) -> date:
    if isinstance(s, date) and not isinstance(s, str):
        return s
    return pd.to_datetime(s).date()
