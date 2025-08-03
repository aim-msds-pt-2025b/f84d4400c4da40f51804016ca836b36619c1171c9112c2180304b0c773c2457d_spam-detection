FROM python:alpine

WORKDIR /app

COPY docker_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && python3 -m nltk.downloader stopwords

COPY src/app.py /app/
COPY src/preprocessing.py /app/src
COPY models/model.pkl /app/models/
COPY outputs/tfidf.pkl /app/outputs/

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]