from pyspark.sql.functions import col

BRONZE_SOURCES = {
    "clickstream": "data/feature_clickstream.csv",
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
    "loans": "data/lms_loan_daily.csv"
}


def ingest_bronze(spark, name, ds):
    """Ingest one source's snapshot for month `ds` into bronze (incremental).

    Reads the raw CSV, keeps only rows for snapshot_date == ds, and writes just
    that partition. With dynamic partition overwrite the other months are left
    untouched, so bronze accumulates one partition per monthly run.
    """
    path = BRONZE_SOURCES[name]
    print(f"Ingesting {name} snapshot {ds} from {path}")
    df = spark.read.csv(path, header=True, inferSchema=True).filter(col("snapshot_date") == ds)

    df.write.partitionBy("snapshot_date").mode("overwrite").parquet(f"datamart/bronze/{name}")
    print(f"✅ Saved bronze/{name} snapshot {ds}")


