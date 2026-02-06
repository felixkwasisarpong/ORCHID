"""MCP client adapter abstraction."""
from __future__ import annotations

import os
import shlex
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


class MCPTransport(str, Enum):
    HTTP = "http"
    STDIO = "stdio"


class MCPClientError(RuntimeError):
    pass


def stdio_server_from_env() -> tuple[str, list[str], dict[str, str] | None]:
    command = os.getenv("MCP_SERVER_COMMAND", "npx")
    args_env = os.getenv("MCP_SERVER_ARGS", "")
    if args_env:
        args = shlex.split(args_env)
    else:
        paths_env = os.getenv("MCP_FS_PATHS", "/app")
        paths = [item.strip() for item in paths_env.split(",") if item.strip()]
        args = ["-y", "@modelcontextprotocol/server-filesystem", *paths]
    env: dict[str, str] = {}
    allowed_env = os.getenv("MCP_FS_ALLOWED_PATHS", "")
    if allowed_env:
        env["ALLOWED_PATHS"] = allowed_env
    return command, args, env or None


class MCPClient:
    """MCP client supporting HTTP (synthetic) and STDIO (filesystem) transports."""

    def __init__(
        self,
        transport: MCPTransport = MCPTransport.HTTP,
        base_url: str | None = None,
        timeout_s: float = 5.0,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.transport = transport
        self.base_url = base_url or "http://localhost:9000"
        self.timeout_s = timeout_s
        self.command = command
        self.args = args or []
        self.env = env
        self._client: httpx.AsyncClient | None = None
        self._stdio_cm = None
        self._session_cm = None
        self._session = None

        if self.transport == MCPTransport.HTTP:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s)

    async def _ensure_stdio_session(self) -> None:
        if self._session is not None:
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:  # noqa: BLE001
            raise MCPClientError(f"MCP stdio requires mcp package: {exc}") from exc

        if not self.command:
            raise MCPClientError("MCP stdio requires a command to launch the server")
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def call_tool(
        self,
        tool_name: str,
        payload: dict[str, Any],
        request_id: str,
        fault: FaultConfig | None = None,
    ) -> MCPResponse:
        if self.transport == MCPTransport.HTTP:
            if self._client is None:
                raise MCPClientError("HTTP client not initialized")
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

        await self._ensure_stdio_session()
        result = await self._session.call_tool(tool_name, arguments=payload)
        output = result.model_dump()
        return MCPResponse(
            status="ok",
            output=output,
            metadata={"transport": self.transport.value, "request_id": request_id},
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)
