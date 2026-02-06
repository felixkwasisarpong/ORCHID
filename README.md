# ORCHID
Research-grade scaffold for hybrid agent orchestration combining a graph control plane, CrewAI worker plane, and MCP tool plane.

**What this is**
- Modular architecture for experimentation, benchmarking, and instrumentation
- Minimal yet extensible implementation with strong typing and traceability

**What this is not**
- Production hardening, security, or full feature coverage

## Architecture

- **Control plane**: `orchestrator/` implements a minimal LangGraph-style sequential graph.
- **Worker plane**: `workers/crew/` wraps CrewAI teams with structured input/output.
- **Tool plane**: `tools/` provides an MCP client and a synthetic MCP server with fault injection.
- **Observability**: `observability/` offers JSON logging, trace export, and replay helpers.
- **Evaluation**: `evaluation/` runs scenarios with fault injection and metrics.
- **API**: `api/` exposes orchestration as a FastAPI service.

## Repository Layout

- `orchestrator/graph.py` Graph orchestrator + example workflow
- `orchestrator/nodes/` BaseNode, CrewNode, ToolNode
- `workers/crew/` CrewAI config and runner
- `tools/mcp_client.py` MCP client adapter
- `tools/synthetic_mcp_server/` Fault-injecting MCP server
- `evaluation/` Experiment harness + scenarios
- `observability/` Logging + trace utilities + replay
- `api/app.py` Orchestration API
- `docker/` Dockerfile and docker-compose

## Install

### Poetry
```bash
poetry install
```

Optional CrewAI support (includes OpenAI/Anthropic integration):
```bash
poetry install --extras crew
```

### uv
```bash
uv venv
uv pip install -e .
```

Optional CrewAI support (includes OpenAI/Anthropic integration):
```bash
uv pip install -e ".[crew]"
```

## Run the MCP Server

```bash
uvicorn tools.synthetic_mcp_server.app:app --port 9000
```

## Run the Orchestrator API

```bash
MCP_BASE_URL=http://localhost:9000 uvicorn api.app:app --port 8000
```

## Run an Experiment

```bash
python -m evaluation.harness --scenarios evaluation/scenarios --output evaluation/results --export-csv
```

Scenarios support modes: `hybrid`, `crew`, `langgraph`. See `evaluation/scenarios/example.json`.

## LLM Runtimes

Supported runtimes: `ollama` (default), `anthropic`, `openai`.

You can pass LLM config via API:
```json
{
  "user_input": "Summarize and call tool.",
  "llm": { "runtime": "ollama", "model": "llama3", "base_url": "http://localhost:11434" }
}
```

You can also set LLM config per scenario in `evaluation/scenarios/*.json`:
```json
{
  "llm": { "runtime": "openai", "model": "gpt-4o-mini" }
}
```

### Runtime Notes
- **Ollama**: set `base_url` to your Ollama host (e.g. `http://localhost:11434` or `http://host.docker.internal:11434` inside Docker).
- **OpenAI**: set `OPENAI_API_KEY` in the environment.
- **Anthropic**: set `ANTHROPIC_API_KEY` in the environment.

If you are running via Docker, rebuild after dependency changes:
```bash
cd docker
docker compose build --no-cache
docker compose up
```

## Example Workflow

`User request → planning node → CrewAI execution → tool call → validation node → final output`

The example workflow is built in `orchestrator/graph.py` and wired to the API in `api/app.py`.

## Fault Injection

Use the synthetic MCP server’s `fault` config to simulate:
- `success`
- `timeout`
- `malformed`
- `rate_limit`
- `delay`
- `random`

See `evaluation/scenarios/example.json` for an example.

## Docker

```bash
cd docker
docker compose up --build
```

Set provider keys in `docker/.env` (see `docker/.env.example`). For Ollama on your host, use:
```
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

If you want Ollama in Docker, the compose file includes an `ollama` service. Pull a model:
```bash
cd docker
docker compose up -d ollama
docker compose exec ollama ollama pull llama3
```

## Notes

- Logging uses JSON lines for easy ingestion.
- Trace events are stored in state and emitted as log entries for replay.
- CrewAI is optional; the scaffold runs with a fallback stub if CrewAI is not installed.
