# Callables import our project modules mounted at /opt/airflow/src
from datetime import datetime, timedelta

from airflow.decorators import dag, task
# from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(seconds=10),
}


@dag(
    dag_id="ml_pipeline",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    description="ML pipeline DAG",
)
def ml_pipeline_dag():
    @task()
    def load_data_task():
        from src.data_preprocessing import load_spam_data

        data_path = load_spam_data()
        return data_path

    @task()
    def preprocess_data_task(data_path):
        from src.data_preprocessing import preprocess_data

        data_path = preprocess_data(data_path)
        return data_path

    @task()
    def encode_data_task(data_path):
        from src.feature_engineering import encode_dataset

        return encode_dataset(data_path)

    @task()
    def split_data_task(data_path):
        from src.model_training import split_data

        return split_data(data_path)

    @task()
    def train_model_task(data_path):
        from src.model_training import train_model

        return train_model(data_path)

    print("ML pipeline")

    load_op = load_data_task()
    preproc_op = preprocess_data_task(load_op)
    encode_op = encode_data_task(preproc_op)
    split_op = split_data_task(encode_op)
    train_op = train_model_task(split_op)

    print(train_op)  # this is just so the object is used and ruff stops complaining
    # load_op >> preproc_op >> encode_op >> split_op >> train_op


ml_pipeline = ml_pipeline_dag()
