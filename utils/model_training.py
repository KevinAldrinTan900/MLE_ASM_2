# utils/model_training.py
#
# Trains candidate models on matured-label data, selects the best by
# out-of-time (OOT) AUC, and registers the winning artefact in the model bank.
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
from datetime import date

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import ml_utils

TRAIN_WINDOW = (date(2023, 1, 1), date(2023, 5, 1))
OOT_WINDOW = (date(2023, 6, 1), date(2023, 7, 1))
RANDOM_STATE = 42


def _make_candidates():
    def with_prep(model, scale=True):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    return {
        "logistic_regression": with_prep(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "random_forest": with_prep(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
            ),
            scale=False,
        ),
        "hist_gradient_boosting": with_prep(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE), scale=False
        ),
    }


def train_and_select(deployment_date) -> str:
    deployment_date = ml_utils.parse_date(deployment_date)
    print(f"Training model for deployment at {deployment_date}")

    features = ml_utils.load_feature_store()
    labels = ml_utils.load_label_store()
    df = ml_utils.join_labels(features, labels)

    train_df = df[df["feature_snapshot_date"].between(*TRAIN_WINDOW)]
    oot_df = df[df["feature_snapshot_date"].between(*OOT_WINDOW)]
    feat_cols = ml_utils.feature_columns(df)
    print(f"Train rows: {len(train_df)}, OOT rows: {len(oot_df)}, features: {len(feat_cols)}")

    X_train_full, y_train_full = train_df[feat_cols].astype(float), train_df["label"].astype(int)
    X_oot, y_oot = oot_df[feat_cols].astype(float), oot_df["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
    )

    results = {}
    for name, pipe in _make_candidates().items():
        pipe.fit(X_tr, y_tr)
        results[name] = {
            "train_auc": float(roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1])),
            "test_auc": float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])),
            "oot_auc": float(roc_auc_score(y_oot, pipe.predict_proba(X_oot)[:, 1])),
        }
        print(f"  {name}: {results[name]}")

    best_name = max(results, key=lambda k: results[k]["oot_auc"])
    print(f"Selected model: {best_name} (OOT AUC {results[best_name]['oot_auc']:.4f})")

    # refit the winner on the full train window (OOT stays untouched)
    best_pipe = _make_candidates()[best_name]
    best_pipe.fit(X_train_full, y_train_full)

    # PSI baseline: training-score distribution, decile bin edges
    baseline_scores = best_pipe.predict_proba(X_train_full)[:, 1]
    edges = np.unique(np.quantile(baseline_scores, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf

    version = f"credit_model_{deployment_date.isoformat()}"
    model_dir = os.path.join(ml_utils.MODEL_BANK, version)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(
        {
            "pipeline": best_pipe,
            "feature_cols": feat_cols,
            "baseline_scores": baseline_scores,
            "psi_bin_edges": edges,
        },
        os.path.join(model_dir, "model.pkl"),
    )

    metadata = {
        "model_version": version,
        "deployment_date": deployment_date.isoformat(),
        "selected_model": best_name,
        "label_definition": "30dpd_6mob",
        "train_window": [d.isoformat() for d in TRAIN_WINDOW],
        "oot_window": [d.isoformat() for d in OOT_WINDOW],
        "n_train": int(len(train_df)),
        "n_oot": int(len(oot_df)),
        "n_features": len(feat_cols),
        "candidate_metrics": results,
        "selection_rule": "max OOT AUC",
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model artefact saved to {model_dir}")
    return version


if __name__ == "__main__":
    import sys
    train_and_select(sys.argv[1] if len(sys.argv) > 1 else "2024-01-01")
