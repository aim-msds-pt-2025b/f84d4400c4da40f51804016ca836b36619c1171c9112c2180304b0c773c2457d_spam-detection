# Callables import our project modules mounted at /opt/airflow/src
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.run_pipeline import train_task

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    description="ML pipeline DAG",
) as dag:
    pipeline_task = PythonOperator(task_id="pipeline", python_callable=train_task)
