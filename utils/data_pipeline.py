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
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def datamart_is_built() -> bool:
    return (
        os.path.exists(os.path.join(GOLD_FEATURE_STORE, "_SUCCESS"))
        and os.path.exists(os.path.join(GOLD_LABEL_STORE, "_SUCCESS"))
    )


def build_datamart(force: bool = False):
    """Run the full upstream datamart pipeline: bronze -> silver -> gold.

    The upstream pipeline is a full refresh over all snapshots (not
    incremental), so we build it once and skip on subsequent DAG runs
    unless force=True or the gold stores are missing.
    """
    if datamart_is_built() and not force:
        print("Datamart gold stores already built - skipping rebuild.")
        return

    os.chdir(PROJECT_ROOT)

    from utils.bronze_processing import ingest_bronze_tables
    from utils.silver_processing import (
        clean_financials_table,
        clean_attributes_table,
        clean_clickstream_table,
        clean_loans_table,
    )
    from utils.gold_processing import build_label_store, build_feature_store

    spark = create_spark_session()
    try:
        for layer in ["datamart/bronze", "datamart/silver", "datamart/gold"]:
            os.makedirs(layer, exist_ok=True)

        ingest_bronze_tables(spark)
        print("Bronze complete")

        clean_financials_table(spark)
        clean_attributes_table(spark)
        clean_clickstream_table(spark)
        clean_loans_table(spark)
        print("Silver complete")

        build_label_store(spark, dpd_cutoff=30, mob_cutoff=6)
        build_feature_store(spark, dpd_cutoff=30, mob_cutoff=6)
        print("Gold complete")
    finally:
        spark.stop()


if __name__ == "__main__":
    import sys
    build_datamart(force="--force" in sys.argv)
