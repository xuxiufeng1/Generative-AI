FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["gunicorn", "--bind", ":8080", "--workers", "2", "main:app"]