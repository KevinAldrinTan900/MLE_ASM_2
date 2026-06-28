# dags/ml_pipeline_dag.py
#
# End-to-end ML pipeline for loan-default prediction, designed to be
# backfilled monthly from 2023-01-01 to 2024-12-01. Each run's logical date L
# activates a different phase via short-circuit gates:
#
#   2023-01 .. 2023-12  warm-up   : datamart only (labels still maturing)
#   2024-01             training  : labels for fd <= 2023-07 have matured ->
#                                   train candidates, select best by OOT AUC,
#                                   register in model bank, then score 2024-01
#   2024-01 .. 2024-06  inference : score that month's application cohort
#   2024-07 .. 2024-12  monitoring: labels for cohort L-6 just matured ->
#                                   performance + stability metrics and charts
#
# The upstream datamart (bronze/silver/gold feature & label stores) is the
# Assignment 1 pipeline by LAM NGUYEN THANH THAO, rebuilt by the first run.

from datetime import date, datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Monthly pipeline, backfilled from 2023-01-01 to 2024-12-01. The datamart is
# (re)ingested every month; the ML phases activate by logical date L via
# short-circuit gates:
#
#   2024-01 .. 2024-06  training  : retrain candidates each month, select best
#                                   by OOT AUC, register in the model bank
#   2024-01 .. 2024-06  inference : score that month's application cohort
#   2024-07 .. 2024-12  monitoring: labels for cohort L-6 just matured ->
#                                   performance + stability metrics and charts
TRAIN_START, TRAIN_END = date(2024, 1, 1), date(2024, 6, 1)
INFER_START, INFER_END = date(2024, 1, 1), date(2024, 6, 1)
MONITOR_START, MONITOR_END = date(2024, 7, 1), date(2024, 12, 1)


def _is_train_run(ds, **_):
    return TRAIN_START <= date.fromisoformat(ds) <= TRAIN_END


def _train_candidate(name, ds, **_):
    from utils import model_training
    model_training.train_candidate(name, ds)


def _select_model(ds, **_):
    from utils import model_training
    model_training.select_model(ds)


def _is_inference_run(ds, **_):
    return INFER_START <= date.fromisoformat(ds) <= INFER_END


def _inference(name, ds, **_):
    from utils import model_inference
    model_inference.run_inference_for(name, ds)


def _is_monitoring_run(ds, **_):
    return MONITOR_START <= date.fromisoformat(ds) <= MONITOR_END


def _monitoring(name, ds, **_):
    from utils import model_monitoring
    model_monitoring.run_monitoring_for(name, ds)


with DAG(
    dag_id="loan_default_ml_pipeline",
    description="Monthly datamart refresh + model training, inference and monitoring",
    schedule="@monthly",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,
    max_active_runs=1,
    default_args={"owner": "mle", "retries": 0},
    tags=["cs611", "assignment2"],
) as dag:

    # --- Datamart ETLs: dep_check -> bronze -> silver, one node per source ---
    # Tasks are instantiated layer-by-layer (all dep_checks, then all bronze,
    # then all silver) so the Airflow grid lists every bronze together, then
    # every silver, then gold — instead of interleaving by source.
    from utils import data_pipeline

    # (grid label, internal source key)
    SOURCES = [("financials", "financials"), ("attributes", "attributes"),
               ("clickstream", "clickstream"), ("lms", "loans")]

    dep_check, bronze, silver = {}, {}, {}

    for label, key in SOURCES:
        dep_check[key] = PythonOperator(
            task_id=f"dep_check_{label}",
            python_callable=data_pipeline.check_source,
            op_kwargs={"name": key},
        )

    for label, key in SOURCES:
        bronze[key] = PythonOperator(
            task_id=f"bronze_{label}",
            python_callable=data_pipeline.run_bronze,
            op_kwargs={"name": key, "ds": "{{ ds }}"},
        )

    for label, key in SOURCES:
        silver[key] = PythonOperator(
            task_id=f"silver_{label}",
            python_callable=data_pipeline.run_silver,
            op_kwargs={"name": key, "ds": "{{ ds }}"},
        )

    for _, key in SOURCES:
        dep_check[key] >> bronze[key] >> silver[key]

    gold_label_store = PythonOperator(
        task_id="gold_label_store",
        python_callable=data_pipeline.run_gold_label_store,
        op_kwargs={"ds": "{{ ds }}"},
    )

    gold_feature_store = PythonOperator(
        task_id="gold_feature_store",
        python_callable=data_pipeline.run_gold_feature_store,
    )

    datamart_ready = EmptyOperator(task_id="datamart_ready")

    # label store needs loans; feature store needs all four silver tables
    silver["loans"] >> gold_label_store
    list(silver.values()) >> gold_feature_store
    [gold_label_store, gold_feature_store] >> datamart_ready

    gate_training = ShortCircuitOperator(
        task_id="gate_training",
        python_callable=_is_train_run,
        ignore_downstream_trigger_rules=False,
    )

    train_xgboost = PythonOperator(
        task_id="train_xgboost",
        python_callable=_train_candidate,
        op_kwargs={"name": "xgboost", "ds": "{{ ds }}"},
    )

    train_logreg = PythonOperator(
        task_id="train_logreg",
        python_callable=_train_candidate,
        op_kwargs={"name": "logreg", "ds": "{{ ds }}"},
    )

    model_selection = PythonOperator(
        task_id="model_selection",
        python_callable=_select_model,
        op_kwargs={"ds": "{{ ds }}"},
    )

    gate_inference = ShortCircuitOperator(
        task_id="gate_inference",
        python_callable=_is_inference_run,
        ignore_downstream_trigger_rules=False,
        trigger_rule="none_failed",
    )

    xgboost_inference = PythonOperator(
        task_id="xgboost_inference",
        python_callable=_inference,
        op_kwargs={"name": "xgboost", "ds": "{{ ds }}"},
    )

    logreg_inference = PythonOperator(
        task_id="logreg_inference",
        python_callable=_inference,
        op_kwargs={"name": "logreg", "ds": "{{ ds }}"},
    )

    inference_completed = EmptyOperator(
        task_id="inference_completed",
        trigger_rule="none_failed",
    )

    gate_monitoring = ShortCircuitOperator(
        task_id="gate_monitoring",
        python_callable=_is_monitoring_run,
        ignore_downstream_trigger_rules=False,
        trigger_rule="none_failed",
    )

    xgboost_monitor = PythonOperator(
        task_id="xgboost_monitor",
        python_callable=_monitoring,
        op_kwargs={"name": "xgboost", "ds": "{{ ds }}"},
    )

    logreg_monitor = PythonOperator(
        task_id="logreg_monitor",
        python_callable=_monitoring,
        op_kwargs={"name": "logreg", "ds": "{{ ds }}"},
    )

    monitoring_completed = EmptyOperator(
        task_id="monitoring_completed",
        trigger_rule="none_failed",
    )

    pipeline_complete = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed",
    )

    datamart_ready >> gate_training >> [train_xgboost, train_logreg] >> model_selection
    model_selection >> gate_inference >> [xgboost_inference, logreg_inference] >> inference_completed
    inference_completed >> gate_monitoring >> [xgboost_monitor, logreg_monitor] >> monitoring_completed
    monitoring_completed >> pipeline_complete
