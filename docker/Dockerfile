FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY orchestrator /app/orchestrator
COPY workers /app/workers
COPY tools /app/tools
COPY evaluation /app/evaluation
COPY observability /app/observability
COPY api /app/api

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ".[crew]"


ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
