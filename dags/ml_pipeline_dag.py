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

TRAIN_RUN_DATE = date(2024, 1, 1)
INFER_START, INFER_END = date(2024, 1, 1), date(2024, 6, 1)
MONITOR_START, MONITOR_END = date(2024, 7, 1), date(2024, 12, 1)


def _build_datamart(ds, **_):
    from utils import data_pipeline
    # force a clean rebuild on the very first run; later runs reuse it
    data_pipeline.build_datamart(force=(ds == "2023-01-01"))


def _is_train_run(ds, **_):
    return date.fromisoformat(ds) == TRAIN_RUN_DATE


def _train(ds, **_):
    from utils import model_training
    model_training.train_and_select(ds)


def _is_inference_run(ds, **_):
    return INFER_START <= date.fromisoformat(ds) <= INFER_END


def _inference(ds, **_):
    from utils import model_inference
    model_inference.run_inference(ds)


def _is_monitoring_run(ds, **_):
    return MONITOR_START <= date.fromisoformat(ds) <= MONITOR_END


def _monitoring(ds, **_):
    from utils import model_monitoring
    model_monitoring.run_monitoring(ds)


with DAG(
    dag_id="loan_default_ml_pipeline",
    description="Datamart refresh + model training, inference and monitoring",
    schedule="@monthly",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,
    max_active_runs=1,
    default_args={"owner": "mle", "retries": 0},
    tags=["cs611", "assignment2"],
) as dag:

    build_datamart = PythonOperator(
        task_id="build_datamart",
        python_callable=_build_datamart,
    )

    gate_training = ShortCircuitOperator(
        task_id="gate_training",
        python_callable=_is_train_run,
        ignore_downstream_trigger_rules=False,
    )

    train_model = PythonOperator(
        task_id="train_and_select_model",
        python_callable=_train,
    )

    gate_inference = ShortCircuitOperator(
        task_id="gate_inference",
        python_callable=_is_inference_run,
        ignore_downstream_trigger_rules=False,
        trigger_rule="none_failed",
    )

    run_inference = PythonOperator(
        task_id="run_inference",
        python_callable=_inference,
    )

    gate_monitoring = ShortCircuitOperator(
        task_id="gate_monitoring",
        python_callable=_is_monitoring_run,
        ignore_downstream_trigger_rules=False,
        trigger_rule="none_failed",
    )

    run_monitoring = PythonOperator(
        task_id="run_model_monitoring",
        python_callable=_monitoring,
    )

    pipeline_complete = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed",
    )

    (
        build_datamart
        >> gate_training
        >> train_model
        >> gate_inference
        >> run_inference
        >> gate_monitoring
        >> run_monitoring
        >> pipeline_complete
    )
