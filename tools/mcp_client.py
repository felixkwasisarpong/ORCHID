"""MCP client adapter abstraction."""
from __future__ import annotations

from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field


class FaultMode(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    RATE_LIMIT = "rate_limit"
    DELAY = "delay"
    RANDOM = "random"


class FaultConfig(BaseModel):
    mode: FaultMode = FaultMode.SUCCESS
    delay_ms: int = 0
    random_weights: dict[str, float] = Field(default_factory=dict)


class MCPRequest(BaseModel):
    tool: str
    input: dict[str, Any]
    request_id: str
    fault: FaultConfig | None = None


class MCPResponse(BaseModel):
    status: str
    output: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPClientError(RuntimeError):
    pass


class MCPClient:
    """Async MCP client with a simple HTTP interface."""

    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def call_tool(
        self,
        tool_name: str,
        payload: dict[str, Any],
        request_id: str,
        fault: FaultConfig | None = None,
    ) -> MCPResponse:
        req = MCPRequest(tool=tool_name, input=payload, request_id=request_id, fault=fault)
        try:
            response = await self._client.post("/tool", json=req.model_dump())
        except httpx.TimeoutException as exc:
            raise MCPClientError(f"MCP timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise MCPClientError(f"MCP http error: {exc}") from exc

        if response.status_code != 200:
            raise MCPClientError(f"MCP error status={response.status_code}")

        try:
            data = response.json()
        except ValueError as exc:
            raise MCPClientError("MCP malformed response") from exc
        return MCPResponse(**data)

    async def close(self) -> None:
        await self._client.aclose()
