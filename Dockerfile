FROM apache/airflow:2.9.3-python3.11

# Java is required by PySpark (used for the datamart bronze/silver/gold jobs)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
