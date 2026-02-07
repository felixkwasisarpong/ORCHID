FROM docker:27-cli AS docker_cli

FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker

CMD ["python", "-m", "harness.run_experiments", "--config", "configs/default.yaml"]
