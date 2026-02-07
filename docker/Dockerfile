FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "harness.run_experiments", "--config", "configs/default.yaml"]
