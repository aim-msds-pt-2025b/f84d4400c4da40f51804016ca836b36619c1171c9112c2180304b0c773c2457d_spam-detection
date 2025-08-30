FROM apache/airflow:2.9.3-python3.12
USER airflow

COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PATH="/opt/airflow:$PATH"

RUN python3 -m nltk.downloader stopwords
