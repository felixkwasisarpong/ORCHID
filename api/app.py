"""FastAPI entrypoint for orchestration API."""
from __future__ import annotations

import os
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator.graph import build_example_graph
from orchestrator.state import GlobalState
from tools.mcp_client import FaultConfig, MCPClient, MCPTransport, stdio_server_from_env
from workers.crew.agents import CrewRunner
from workers.crew.llm import LLMConfig
from workers.crew.tasks import default_crew_config


app = FastAPI(title="ORCHID Orchestrator", version="0.1.0")


class RunRequest(BaseModel):
    user_input: str
    fault: FaultConfig | None = None
    llm: LLMConfig | None = None


class RunResponse(BaseModel):
    request_id: str
    final_output: str | None
    errors: list[str]
    trace: list[dict[str, object]]


@app.on_event("startup")
async def startup() -> None:
    mcp_base_url = os.getenv("MCP_BASE_URL", "http://localhost:9000")
    transport = MCPTransport(os.getenv("MCP_TRANSPORT", MCPTransport.HTTP))
    if transport == MCPTransport.STDIO:
        command, args, env = stdio_server_from_env()
        app.state.mcp_client = MCPClient(
            transport=transport,
            command=command,
            args=args,
            env=env,
        )
    else:
        app.state.mcp_client = MCPClient(transport=transport, base_url=mcp_base_url)
    app.state.crew_runner = CrewRunner(default_crew_config())


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.mcp_client.close()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    request_id = str(uuid.uuid4())
    state = GlobalState(request_id=request_id, user_input=request.user_input)
    crew_runner = app.state.crew_runner
    if request.llm is not None:
        crew_runner = CrewRunner(default_crew_config(), llm_config=request.llm)
    graph = build_example_graph(
        app.state.mcp_client,
        crew_runner,
        fault_config=request.fault,
    )
    final_state = await graph.run(state)
    trace = [event.model_dump(mode="json") for event in final_state.trace]
    return RunResponse(
        request_id=request_id,
        final_output=final_state.final_output,
        errors=final_state.errors,
        trace=trace,
    )
