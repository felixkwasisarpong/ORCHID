"""Graph-based orchestrator implementation."""
from __future__ import annotations

import json
import os
from typing import Iterable

from observability.logger import get_logger, log_event
from orchestrator.nodes.base_node import BaseNode
from orchestrator.nodes.crew_node import CrewNode
from orchestrator.nodes.tool_node import ToolNode
from orchestrator.policies import FailureAction, FailurePolicy, RetryPolicy
from orchestrator.state import GlobalState
from tools.mcp_client import MCPClient, FaultConfig, MCPTransport
from workers.crew.agents import CrewRunner


class Graph:
    """Minimal sequential graph runner with tracing."""

    def __init__(self, nodes: Iterable[BaseNode]) -> None:
        self.nodes = list(nodes)
        self.logger = get_logger("orchid")

    async def run(self, state: GlobalState) -> GlobalState:
        log_event(self.logger, "graph_start", request_id=state.request_id)
        for node in self.nodes:
            before_errors = len(state.errors)
            state = await node.execute(state)
            after_errors = len(state.errors)
            if after_errors > before_errors and node.failure_policy.on_failure == FailureAction.HALT:
                log_event(
                    self.logger,
                    "graph_halt",
                    request_id=state.request_id,
                    node=node.name,
                )
                break
        log_event(self.logger, "graph_end", request_id=state.request_id)
        return state


class PlanningNode(BaseNode):
    """Create a lightweight plan from the user request."""

    async def run(self, state: GlobalState) -> GlobalState:
        plan = (
            "Plan: interpret request, delegate to crew, call tool, validate output. "
            f"Input='{state.user_input}'."
        )
        return state.model_copy(update={"plan": plan})


class ValidationNode(BaseNode):
    """Validate tool output and finalize response."""

    async def run(self, state: GlobalState) -> GlobalState:
        tool_result = state.tool_result or {}
        validation = "validated" if tool_result else "tool_result_missing"
        final_output = {
            "summary": state.crew_output or "",
            "tool_result": tool_result,
            "validation": validation,
        }
        return state.model_copy(
            update={
                "validation": validation,
                "final_output": str(final_output),
            }
        )


def resolve_tool_config(mcp_client: MCPClient) -> tuple[str, dict[str, object] | None]:
    """Resolve MCP tool name and payload overrides based on env + transport."""
    tool_name = os.getenv("MCP_TOOL_NAME")
    transport = getattr(mcp_client, "transport", MCPTransport.HTTP)
    if tool_name is None:
        tool_name = "synthetic_tool" if transport == MCPTransport.HTTP else "list_directory"
    payload_override = None
    tool_args_json = os.getenv("MCP_TOOL_ARGS_JSON")
    if tool_args_json:
        try:
            payload_override = json.loads(tool_args_json)
        except json.JSONDecodeError:
            payload_override = None
    elif tool_name in {"list_files", "list_directory"}:
        payload_override = {"path": os.getenv("MCP_TOOL_PATH", ".")}
    return tool_name, payload_override


def build_example_graph(
    mcp_client: MCPClient,
    crew_runner: CrewRunner,
    fault_config: FaultConfig | None = None,
) -> Graph:
    """Example workflow: plan -> crew -> tool -> validate."""
    tool_name, payload_override = resolve_tool_config(mcp_client)

    plan_node = PlanningNode(
        name="planner",
        retry_policy=RetryPolicy(max_attempts=1),
    )
    crew_node = CrewNode(
        name="crew_exec",
        crew_runner=crew_runner,
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=300.0,
    )
    tool_node = ToolNode(
        name="tool_call",
        mcp_client=mcp_client,
        tool_name=tool_name,
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=10.0,
        fault_config=fault_config,
        payload_override=payload_override,
    )
    validation_node = ValidationNode(
        name="validator",
        failure_policy=FailurePolicy(on_failure=FailureAction.CONTINUE),
    )
    return Graph([plan_node, crew_node, tool_node, validation_node])
