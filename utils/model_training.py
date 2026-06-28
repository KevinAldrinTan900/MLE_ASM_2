# utils/model_training.py
#
# Trains candidate models on matured-label data as independent tasks, then
# selects the best by out-of-time (OOT) AUC and registers the winning artefact
# in the model bank.
#
# Split into one node per model (train_xgboost, train_logreg) plus a separate
# model_selection node:
#   train_<model>  -> fit + evaluate, save candidate artefact + metrics
#   select_model   -> compare candidate OOT AUCs, promote the champion
#
# Temporal design (no leakage):
#   - Model is trained/deployed at TRAIN-RUN date D (e.g. 2024-01-01).
#   - A label for feature snapshot fd only matures at fd + 6 months, so the
#     training pool is restricted to fd <= D - 6 months.
#   - Train window:  fd 2023-01-01 .. 2023-05-01 (random 80/20 train/test)
#   - OOT window:    fd 2023-06-01 .. 2023-07-01 (model selection)
#   - Preprocessing (imputer + scaler) is fit on the training split only.

import json
import os
import shutil

import joblib
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import ml_utils

RANDOM_STATE = 42

# rolling-window sizes (months); the windows slide forward each monthly retrain
TRAIN_MONTHS = 5
OOT_MONTHS = 2

# the two candidate models, one Airflow node each
CANDIDATE_MODELS = ["xgboost", "logreg"]


def _windows(deployment_date):
    """Rolling train/OOT feature windows for a given deployment date.

    A label for feature snapshot fd matures at fd + LABEL_MOB_MONTHS, so the
    most recent fully-matured feature month at deployment D is D - 6 months.
    The OOT window is the latest matured months; the train window precedes it.
    For D = 2024-01-01 this gives train 2023-01..05 / OOT 2023-06..07; each
    later monthly retrain slides both windows forward by a month.
    """
    oot_end = deployment_date - relativedelta(months=ml_utils.LABEL_MOB_MONTHS)
    oot_start = oot_end - relativedelta(months=OOT_MONTHS - 1)
    train_end = oot_start - relativedelta(months=1)
    train_start = train_end - relativedelta(months=TRAIN_MONTHS - 1)
    return (train_start, train_end), (oot_start, oot_end)


def _make_pipeline(name):
    """Fresh sklearn Pipeline for a candidate model."""
    if name == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
    if name == "xgboost":
        from xgboost import XGBClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Unknown candidate model: {name}")


def _version(deployment_date) -> str:
    return f"credit_model_{ml_utils.parse_date(deployment_date).isoformat()}"


def _candidate_dir(deployment_date, name) -> str:
    return os.path.join(ml_utils.MODEL_BANK, _version(deployment_date), "candidates", name)


def _prepare(deployment_date):
    """Load gold stores and build the train/test/OOT splits (deterministic)."""
    train_w, oot_w = _windows(deployment_date)
    features = ml_utils.load_feature_store()
    labels = ml_utils.load_label_store()
    df = ml_utils.join_labels(features, labels)

    train_df = df[df["feature_snapshot_date"].between(*train_w)]
    oot_df = df[df["feature_snapshot_date"].between(*oot_w)]
    feat_cols = ml_utils.feature_columns(df)

    X_full, y_full = train_df[feat_cols].astype(float), train_df["label"].astype(int)
    X_oot, y_oot = oot_df[feat_cols].astype(float), oot_df["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
    )
    return {
        "feat_cols": feat_cols, "n_train": len(train_df), "n_oot": len(oot_df),
        "X_full": X_full, "y_full": y_full, "X_oot": X_oot, "y_oot": y_oot,
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
    }


def train_candidate(name, deployment_date) -> dict:
    """Train one candidate model, evaluate, and persist its artefact + metrics."""
    deployment_date = ml_utils.parse_date(deployment_date)
    print(f"Training candidate '{name}' for deployment at {deployment_date}")
    d = _prepare(deployment_date)

    pipe = _make_pipeline(name)
    pipe.fit(d["X_tr"], d["y_tr"])
    metrics = {
        "model": name,
        "train_auc": float(roc_auc_score(d["y_tr"], pipe.predict_proba(d["X_tr"])[:, 1])),
        "test_auc": float(roc_auc_score(d["y_te"], pipe.predict_proba(d["X_te"])[:, 1])),
        "oot_auc": float(roc_auc_score(d["y_oot"], pipe.predict_proba(d["X_oot"])[:, 1])),
    }
    print(f"  {name}: {metrics}")

    # refit on the full train window (OOT stays untouched) for deployment
    final = _make_pipeline(name)
    final.fit(d["X_full"], d["y_full"])

    # PSI baseline: training-score distribution, decile bin edges
    baseline_scores = final.predict_proba(d["X_full"])[:, 1]
    edges = np.unique(np.quantile(baseline_scores, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf

    cdir = _candidate_dir(deployment_date, name)
    os.makedirs(cdir, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final,
            "feature_cols": d["feat_cols"],
            "baseline_scores": baseline_scores,
            "psi_bin_edges": edges,
        },
        os.path.join(cdir, "model.pkl"),
    )
    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({**metrics, "n_train": int(d["n_train"]), "n_oot": int(d["n_oot"]),
                   "n_features": len(d["feat_cols"])}, f, indent=2)
    print(f"Candidate artefact saved to {cdir}")
    return metrics


def select_model(deployment_date) -> str:
    """Compare candidate OOT AUCs and promote the champion to the model bank."""
    deployment_date = ml_utils.parse_date(deployment_date)
    version = _version(deployment_date)
    base = os.path.join(ml_utils.MODEL_BANK, version)

    candidates = {}
    for name in CANDIDATE_MODELS:
        with open(os.path.join(_candidate_dir(deployment_date, name), "metrics.json")) as f:
            candidates[name] = json.load(f)

    best_name = max(candidates, key=lambda k: candidates[k]["oot_auc"])
    print(f"Selected model: {best_name} (OOT AUC {candidates[best_name]['oot_auc']:.4f})")

    # promote the champion artefact to the canonical model-bank location
    shutil.copyfile(
        os.path.join(_candidate_dir(deployment_date, best_name), "model.pkl"),
        os.path.join(base, "model.pkl"),
    )

    train_w, oot_w = _windows(deployment_date)
    metadata = {
        "model_version": version,
        "deployment_date": deployment_date.isoformat(),
        "selected_model": best_name,
        "label_definition": "30dpd_6mob",
        "train_window": [d.isoformat() for d in train_w],
        "oot_window": [d.isoformat() for d in oot_w],
        "n_features": candidates[best_name].get("n_features"),
        "candidate_metrics": candidates,
        "selection_rule": "max OOT AUC",
    }
    with open(os.path.join(base, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Champion registered at {base}")
    return best_name


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    for m in CANDIDATE_MODELS:
        train_candidate(m, ds)
    select_model(ds)
