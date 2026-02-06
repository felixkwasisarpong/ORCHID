"""Node that calls an MCP tool via the MCP client."""
from __future__ import annotations

from typing import Any

from orchestrator.nodes.base_node import BaseNode
from orchestrator.state import GlobalState
from tools.mcp_client import MCPClient, MCPResponse, FaultConfig


class ToolNode(BaseNode):
    """Invoke a tool using the MCP client."""

    def __init__(
        self,
        name: str,
        mcp_client: MCPClient,
        tool_name: str,
        input_field: str = "crew_output",
        output_field: str = "tool_result",
        fault_config: FaultConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.input_field = input_field
        self.output_field = output_field
        self.fault_config = fault_config

    async def run(self, state: GlobalState) -> GlobalState:
        payload = getattr(state, self.input_field)
        if payload is None:
            return state.add_error(f"{self.name} missing input '{self.input_field}'")
        self.logger.info(
            "tool_request",
            extra={
                "request_id": state.request_id,
                "node": self.name,
                "tool": self.tool_name,
            },
        )
        response: MCPResponse = await self.mcp_client.call_tool(
            tool_name=self.tool_name,
            payload={"text": payload},
            request_id=state.request_id,
            fault=self.fault_config,
        )
        self.logger.info(
            "tool_response",
            extra={
                "request_id": state.request_id,
                "node": self.name,
                "status": response.status,
                "metadata": response.metadata,
            },
        )
        metadata = {**state.metadata, "tool": response.metadata}
        return state.model_copy(
            update={
                self.output_field: response.output,
                "metadata": metadata,
            }
        )
