# utils/model_inference.py
#
# Loads the active model from the model bank and scores all loan applications
# for one feature snapshot. Predictions are stored as a gold table in the
# datamart, partitioned by snapshot_date.

import json
import os

import joblib
import pandas as pd

from utils import ml_utils


def get_active_model(as_of_date):
    """Pick the most recent model in the bank deployed on/before as_of_date."""
    as_of_date = ml_utils.parse_date(as_of_date)
    candidates = []
    for d in sorted(os.listdir(ml_utils.MODEL_BANK)):
        meta_path = os.path.join(ml_utils.MODEL_BANK, d, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        dep = ml_utils.parse_date(meta["deployment_date"])
        if dep <= as_of_date:
            candidates.append((dep, d, meta))
    if not candidates:
        raise RuntimeError(f"No deployed model in model bank as of {as_of_date}")
    dep, version, meta = max(candidates, key=lambda x: x[0])
    artefact = joblib.load(os.path.join(ml_utils.MODEL_BANK, version, "model.pkl"))
    return version, artefact, meta


def run_inference(snapshot_date) -> str:
    snapshot_date = ml_utils.parse_date(snapshot_date)
    version, artefact, _ = get_active_model(snapshot_date)
    print(f"Scoring snapshot {snapshot_date} with {version}")

    features = ml_utils.load_feature_store()
    snap = features[features["feature_snapshot_date"] == snapshot_date].copy()
    if snap.empty:
        raise RuntimeError(f"No feature rows for snapshot {snapshot_date}")

    # align to the training feature set (missing -> NaN, handled by imputer)
    X = snap.reindex(columns=artefact["feature_cols"]).astype(float)
    probs = artefact["pipeline"].predict_proba(X)[:, 1]

    out = pd.DataFrame(
        {
            "Customer_ID": snap["Customer_ID"].values,
            "snapshot_date": snapshot_date.isoformat(),
            "model_version": version,
            "default_probability": probs,
            "default_prediction": (probs >= 0.5).astype(int),
        }
    )

    part_dir = os.path.join(
        ml_utils.PREDICTION_STORE, f"snapshot_date={snapshot_date.isoformat()}"
    )
    os.makedirs(part_dir, exist_ok=True)
    out.drop(columns=["snapshot_date"]).to_parquet(
        os.path.join(part_dir, "predictions.parquet"), index=False
    )
    print(f"Wrote {len(out)} predictions to {part_dir}")
    return part_dir


if __name__ == "__main__":
    import sys
    run_inference(sys.argv[1])
