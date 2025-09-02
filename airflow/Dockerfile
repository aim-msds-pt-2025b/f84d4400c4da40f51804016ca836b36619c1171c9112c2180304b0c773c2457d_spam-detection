FROM apache/airflow:3.0.3-python3.12
USER airflow

COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PATH="/opt/airflow:$PATH"
ENV PYTHONPATH="/opt/airflow:$PYTHONPATH"

RUN python3 -m nltk.downloader stopwords
