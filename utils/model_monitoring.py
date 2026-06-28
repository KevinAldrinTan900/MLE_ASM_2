# utils/model_monitoring.py
#
# At run date L, labels for the cohort scored at L - 6 months have just
# matured. This module joins that cohort's stored predictions to the gold
# label store, computes performance (AUC, Gini, precision/recall, calibration)
# and stability (PSI vs the training score distribution) metrics, appends them
# to the gold monitoring table, and regenerates the monitoring charts.

import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from dateutil.relativedelta import relativedelta
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import ml_utils

VIZ_DIR = os.path.join(ml_utils.GOLD, "monitoring_viz")

# governance thresholds (see SOP in the deck)
AUC_ALERT = 0.70
PSI_WATCH = 0.10
PSI_ALERT = 0.25


def run_monitoring(run_date) -> dict:
    run_date = ml_utils.parse_date(run_date)
    cohort_date = run_date - relativedelta(months=ml_utils.LABEL_MOB_MONTHS)
    print(f"Monitoring run {run_date}: evaluating cohort scored at {cohort_date}")

    pred_path = os.path.join(
        ml_utils.PREDICTION_STORE, f"snapshot_date={cohort_date.isoformat()}"
    )
    preds = pd.read_parquet(os.path.join(pred_path, "predictions.parquet"))

    labels = ml_utils.load_label_store()
    cohort_labels = labels[labels["label_snapshot_date"] == run_date]
    df = preds.merge(cohort_labels, on="Customer_ID", how="inner")
    if df.empty:
        raise RuntimeError(f"No matured labels found for cohort {cohort_date}")

    y, p = df["label"].astype(int), df["default_probability"]
    yhat = df["default_prediction"]

    version = df["model_version"].iloc[0]
    artefact = joblib.load(os.path.join(ml_utils.MODEL_BANK, version, "model.pkl"))
    psi_value = ml_utils.psi(
        artefact["baseline_scores"], p.values, artefact["psi_bin_edges"]
    )

    auc = float(roc_auc_score(y, p))
    row = {
        "cohort_snapshot_date": cohort_date.isoformat(),
        "monitored_at": run_date.isoformat(),
        "model_version": version,
        "n_scored": int(len(preds)),
        "n_labeled": int(len(df)),
        "auc": auc,
        "gini": 2 * auc - 1,
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "predicted_default_rate": float(p.mean()),
        "actual_default_rate": float(y.mean()),
        "psi": psi_value,
        "status": (
            "ALERT" if (auc < AUC_ALERT or psi_value > PSI_ALERT)
            else "WATCH" if psi_value > PSI_WATCH
            else "OK"
        ),
    }
    print(row)

    part_dir = os.path.join(
        ml_utils.MONITOR_STORE, f"cohort_snapshot_date={cohort_date.isoformat()}"
    )
    os.makedirs(part_dir, exist_ok=True)
    pd.DataFrame([row]).drop(columns=["cohort_snapshot_date"]).to_parquet(
        os.path.join(part_dir, "metrics.parquet"), index=False
    )
    print(f"Wrote monitoring metrics to {part_dir}")

    refresh_charts()
    return row


def load_monitoring_table() -> pd.DataFrame:
    df = pads.dataset(ml_utils.MONITOR_STORE, partitioning="hive").to_table().to_pandas()
    df["cohort_snapshot_date"] = df["cohort_snapshot_date"].astype(str)
    return df.sort_values("cohort_snapshot_date")


def refresh_charts():
    """Regenerate the performance & stability charts from the full gold
    monitoring table (called after every monitoring run, so the charts always
    reflect the latest state)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_monitoring_table()
    os.makedirs(VIZ_DIR, exist_ok=True)
    x = df["cohort_snapshot_date"]

    # 1. discrimination performance over time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, df["auc"], marker="o", label="AUC", color="#1f6feb")
    ax.plot(x, df["gini"], marker="s", label="Gini", color="#8957e5")
    ax.axhline(AUC_ALERT, color="#d1242f", ls="--", lw=1, label=f"AUC alert ({AUC_ALERT})")
    ax.set_title("Model performance by monthly application cohort")
    ax.set_xlabel("Cohort snapshot (application month)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "performance_over_time.png"), dpi=150)
    plt.close(fig)

    # 2. classification metrics over time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, c in [("accuracy", "#1f6feb"), ("precision", "#2da44e"),
                   ("recall", "#d1242f"), ("f1", "#8957e5")]:
        ax.plot(x, df[col], marker="o", label=col, color=c)
    ax.set_title("Classification metrics by cohort (threshold = 0.5)")
    ax.set_xlabel("Cohort snapshot (application month)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", ncols=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "classification_metrics.png"), dpi=150)
    plt.close(fig)

    # 3. calibration: predicted vs actual default rate
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, df["actual_default_rate"], width=0.55, label="Actual default rate",
           color="#cfd8e3")
    ax.plot(x, df["predicted_default_rate"], marker="o", color="#1f6feb",
            label="Mean predicted probability")
    ax.set_title("Calibration: predicted vs actual default rate by cohort")
    ax.set_xlabel("Cohort snapshot (application month)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "calibration_over_time.png"), dpi=150)
    plt.close(fig)

    # 4. stability: PSI of score distribution vs training baseline
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2da44e" if v <= PSI_WATCH else "#d4a72c" if v <= PSI_ALERT
              else "#d1242f" for v in df["psi"]]
    ax.bar(x, df["psi"], width=0.55, color=colors)
    ax.axhline(PSI_WATCH, color="#d4a72c", ls="--", lw=1, label=f"Watch ({PSI_WATCH})")
    ax.axhline(PSI_ALERT, color="#d1242f", ls="--", lw=1, label=f"Alert ({PSI_ALERT})")
    ax.set_title("Score stability: PSI vs training distribution")
    ax.set_xlabel("Cohort snapshot (application month)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "psi_over_time.png"), dpi=150)
    plt.close(fig)

    print(f"Charts refreshed in {VIZ_DIR}")


if __name__ == "__main__":
    import sys
    run_monitoring(sys.argv[1])
