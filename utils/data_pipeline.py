# utils/data_pipeline.py
#
# Wrapper around the upstream datamart pipeline (built by LAM NGUYEN THANH THAO
# for Assignment 1, consumed here as-is — it simulates a datamart owned by
# another tech family). Her modules use paths relative to the project root, so
# we chdir into PROJECT_ROOT before invoking them.

import os

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLD_FEATURE_STORE = os.path.join(PROJECT_ROOT, "datamart", "gold", "feature_store")
GOLD_LABEL_STORE = os.path.join(PROJECT_ROOT, "datamart", "gold", "label_store")


def create_spark_session():
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("LoanMLPipeline")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        # overwrite only the snapshot partition a task writes, keep the rest
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def datamart_is_built() -> bool:
    return (
        os.path.exists(os.path.join(GOLD_FEATURE_STORE, "_SUCCESS"))
        and os.path.exists(os.path.join(GOLD_LABEL_STORE, "_SUCCESS"))
    )


def _run_with_spark(fn):
    """chdir into PROJECT_ROOT, run fn(spark), always stop the session."""
    os.chdir(PROJECT_ROOT)
    spark = create_spark_session()
    try:
        fn(spark)
    finally:
        spark.stop()


# --- Per-source ETL tasks (one Airflow node each) ----------------------------
# The datamart is rebuilt every monthly run (full refresh over all snapshots).

def check_source(name: str):
    """Dependency check: fail the run if the raw CSV is missing from data/."""
    from utils.bronze_processing import BRONZE_SOURCES
    path = os.path.join(PROJECT_ROOT, BRONZE_SOURCES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required source CSV not found: {path}")
    print(f"✅ Source present: {BRONZE_SOURCES[name]}")


def run_bronze(name: str, ds: str):
    from utils.bronze_processing import ingest_bronze
    _run_with_spark(lambda spark: ingest_bronze(spark, name, ds))


def run_silver(name: str, ds: str):
    from utils import silver_processing
    cleaner = {"financials": silver_processing.clean_financials_table,
               "attributes": silver_processing.clean_attributes_table,
               "clickstream": silver_processing.clean_clickstream_table,
               "loans": silver_processing.clean_loans_table}[name]
    _run_with_spark(lambda spark: cleaner(spark, ds))


def run_gold_label_store(ds: str):
    from utils.gold_processing import build_label_store
    _run_with_spark(lambda spark: build_label_store(spark, ds, dpd_cutoff=30, mob_cutoff=6))


def run_gold_feature_store():
    # the feature store is recomputed from the accumulated silver each month so
    # its cross-snapshot joins and categorical encoders stay consistent
    from utils.gold_processing import build_feature_store
    _run_with_spark(build_feature_store)
