# ORCHID Orchestrator Benchmarks

Research-grade Python prototype for comparing three in-process orchestrators over a shared tool-calling workload.

**Orchestrators**
- LangGraph (graph/state machine)
- CrewAI (role/task multi-agent)
- AutoGen (multi-agent conversation)

**LLM runtimes**
- Ollama (local)
- OpenAI (cloud)
- Anthropic (cloud)

**Tools**
- Docker MCP Gateway filesystem tools via stdio transport by default (streaming HTTP optional)

## Setup

1) Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ensure Docker MCP Gateway is available.

Stdio transport (default): the harness spawns the gateway automatically per run. If your gateway needs a filesystem root, set `gateway_cmd` in the YAML config with `{sandbox_root}` as a placeholder.

Streaming transport (optional): start a long-running gateway and point the client at it.

```bash
# Example streaming gateway startup (adjust filesystem root to your sandbox directory)
docker mcp gateway run --server filesystem --root $(pwd)/evaluation/sandboxes --listen 127.0.0.1:8080
```

3) Set API keys if using cloud runtimes.

```bash
export OPENAI_API_KEY=... 
export ANTHROPIC_API_KEY=...
```

## Run Experiments

Use the harness to run a full suite or targeted runs.

```bash
# Full default suite
python -m harness.run_experiments --config configs/default.yaml

# Single orchestrator + runtime + task
python -m harness.run_experiments --orchestrator langgraph --runtime ollama --task task_01_count_lines
```

## Smoke Test

Runs 1 task with each orchestrator using Ollama and filesystem tools.

```bash
python scripts/smoke_test.py
# or module form:
python -m scripts.smoke_test
```

## Run Harness in Docker

If you want the harness to run inside Docker while still spawning the MCP filesystem container, you must:
- mount the Docker socket
- mount the repo at the same absolute path
- pass `HOST_PWD` so the config uses host paths for bind mounts

Build and run:
```bash
docker build -f docker/Dockerfile -t orchid-harness .
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v \"$(pwd):$(pwd)\" -w \"$(pwd)\" \
  -e HOST_PWD=\"$(pwd)\" \
  orchid-harness \
  python -m harness.run_experiments --config configs/smoke.docker.yaml
```

If Ollama runs on the host machine, add:
```bash
-e OLLAMA_BASE_URL="http://host.docker.internal:11434"
```

Docker Compose (run from repo root):
```bash
docker compose -f docker/compose.harness.yml up --build
```

## Traces and Summary

- JSONL traces: `evaluation/results/traces/` (one file per run)
- Summary CSV: `evaluation/results/summary_<timestamp>.csv`

Each trace includes consistent counters (`llm_calls`, `tool_calls`, `retries`, `total_latency_ms`) and per-step timings.

## Configuration Notes

- Edit `configs/default.yaml` to change models, max steps, retries, or fault injection.
- Config values support environment variable expansion (e.g., `${HOST_PWD}`).
- For streaming MCP transport, set:
  - `transport: http`
  - `http_url: http://127.0.0.1:8080` (or your gateway address)
- For stdio transport with explicit filesystem root, set:
  - `gateway_cmd: ["docker", "mcp", "gateway", "run", "--server", "filesystem", "--root", "{sandbox_root}"]`
- For Docker MCP Tools filesystem server (direct container), use:
  - `gateway_cmd: ["docker", "run", "-i", "--rm", "-v", "{sandbox_root}:/local-directory", "mcp/filesystem", "/local-directory"]`
- If your filesystem tool names differ from the defaults, update `benchmarks/tasks.py`.

## Fault Injection

Supported in the harness via config or CLI:
- Permission fault: `--fault-permission path/inside/sandbox`
- Missing file fault: `--fault-missing path/inside/sandbox`
- Latency/jitter: `--fault-latency-ms 250 --fault-jitter-ms 50`
- Tool timeout: `--fault-timeout-s 5`

## MCP Stdio Client (Minimal Adapter)

Requirements:
- Docker installed and running
- MCP Gateway available (`docker mcp gateway run`)

Smoke test:
```bash
python example/smoke_test.py
```

The client writes JSONL call traces to `logs/mcp_calls.jsonl`.

## Repository Layout

- `orchestrators/` LangGraph, CrewAI, AutoGen engines
- `runtimes/` Ollama, OpenAI, Anthropic clients
- `tools/` MCP Gateway client
- `benchmarks/` tasks and validator
- `harness/` experiment runner
- `observability/` JSONL logger and Pydantic schemas
- `configs/` experiment configs
