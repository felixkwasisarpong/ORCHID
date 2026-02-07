# ORCHID
Research-grade prototype for comparing three orchestrators over the same tool-calling workload.

**Orchestrators**
- LangGraph-style (in-process state machine)
- CrewAI (role/task multi-agent, in-process)
- Temporal (durable workflows, replay-safe activities)

**LLM runtimes**
- Ollama (local)
- OpenAI (cloud)
- Anthropic (cloud)

**What this is**
- Minimal but extensible benchmark harness with shared schemas and traces
- Deterministic prompts, caps, and tool schemas across orchestrators

**What this is not**
- Production hardening or complete feature coverage

## Architecture
- `orchestrators/` Engines for LangGraph, CrewAI, Temporal
- `runtimes/` LLM runtime clients (Ollama/OpenAI/Anthropic)
- `tools/mcp_gateway_client.py` MCP Gateway client (stdio by default, Streamable HTTP optional)
- `benchmarks/` 12 filesystem tasks + validators
- `harness/` Experiment runner + smoke test
- `observability/` JSONL traces and schemas
- `docker/` Dockerfiles + Temporal compose

Legacy scaffolding for earlier experiments remains in `orchestrator/`, `evaluation/`, and `api/`.

## Install
### Poetry
```bash
poetry install
```
Optional CrewAI support:
```bash
poetry install --extras crew
```

### uv
```bash
uv venv
uv pip install -e .
```
Optional CrewAI support:
```bash
uv pip install -e ".[crew]"
```

## MCP Gateway (Filesystem Tools)
The default transport is **stdio**, spawning the Docker MCP Gateway:

```bash
export MCP_GATEWAY_ALLOWED_PATHS="/absolute/path/to/ORCHID/evaluation/sandboxes"
docker mcp gateway run
```

The harness defaults to `stdio` with `docker mcp gateway run`. To override, edit `configs/experiment.yaml` or set:
- `MCP_GATEWAY_COMMAND`
- `MCP_GATEWAY_ARGS`

For Streamable HTTP, set in config:
```yaml
mcp:
  transport: streamable_http
  base_url: http://localhost:8085
```

## Temporal (Local)
Start Temporal using Docker Compose:
```bash
cd docker
docker compose -f temporal-compose.yml up -d
```

## Run Experiments
LangGraph + Ollama:
```bash
python -m harness.run_experiments --config configs/experiment.yaml --orchestrator langgraph --runtime ollama --model llama3
```

CrewAI + OpenAI:
```bash
export OPENAI_API_KEY=YOUR_KEY
python -m harness.run_experiments --config configs/experiment.yaml --orchestrator crewai --runtime openai --model gpt-4o-mini
```

Temporal:
```bash
python -m workers.temporal_worker
python -m harness.run_experiments --config configs/experiment.yaml --orchestrator temporal
```

Or start the worker automatically:
```bash
python -m harness.run_temporal_experiments --config configs/experiment.yaml
```

## Smoke Test
Runs one task with each orchestrator using Ollama + filesystem tools:
```bash
python -m harness.smoke_test
```

## Results
- JSONL traces: `evaluation/results/traces.jsonl`
- Summary CSV: `evaluation/results/summary.csv`

## Runtime Notes
- **Ollama**: set `OLLAMA_BASE_URL` (or use `runtime.base_url` in config)
- **OpenAI**: set `OPENAI_API_KEY`
- **Anthropic**: set `ANTHROPIC_API_KEY`

## Configs
Edit `configs/experiment.yaml` for:
- orchestrator selection
- runtime/model
- episode caps (`max_steps`, `max_llm_retries`, `max_tool_retries`)
- fault injection (latency/jitter/timeouts, permission/missing paths)
- Temporal address/task queue
