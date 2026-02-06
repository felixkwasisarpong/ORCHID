FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY orchestrator /app/orchestrator
COPY workers /app/workers
COPY tools /app/tools
COPY evaluation /app/evaluation
COPY observability /app/observability
COPY api /app/api

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
