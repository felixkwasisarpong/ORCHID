"""MCP Gateway client supporting stdio and Streamable HTTP transports."""
from __future__ import annotations

import asyncio
import os
import random
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MCPGatewayTransport(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class MCPGatewayError(RuntimeError):
    pass


class ToolSpec(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


@dataclass
class FaultSettings:
    latency_ms: int = 0
    jitter_ms: int = 0
    timeout_s: float | None = None


class MCPGatewayClient:
    def __init__(
        self,
        transport: MCPGatewayTransport = MCPGatewayTransport.STDIO,
        base_url: str | None = None,
        timeout_s: float = 10.0,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        reuse_session: bool | None = None,
        faults: FaultSettings | None = None,
    ) -> None:
        self.transport = transport
        self.base_url = base_url or os.getenv("MCP_GATEWAY_HTTP_URL", "http://localhost:8085")
        self.timeout_s = timeout_s
        self.command = command
        self.args = args or []
        self.env = env
        if reuse_session is None:
            reuse_session = True
        self.reuse_session = reuse_session
        self.faults = faults or FaultSettings()
        self._stdio_cm = None
        self._session_cm = None
        self._session = None
        self._http_cm = None
        self._http_session_cm = None
        self._http_session = None
        self._lock = None
        self._tool_registry: dict[str, ToolSpec] = {}

    async def _ensure_stdio_session(self) -> None:
        if self._session is not None:
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:  # noqa: BLE001
            raise MCPGatewayError(f"MCP stdio requires mcp package: {exc}") from exc

        if not self.command:
            raise MCPGatewayError("MCP stdio requires a command to launch the gateway")
        if self._lock is None:
            self._lock = asyncio.Lock()
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        async with self._lock:
            if self._session is not None:
                return
            self._stdio_cm = stdio_client(params)
            read, write = await self._stdio_cm.__aenter__()
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()

    async def _ensure_http_session(self) -> None:
        if self._http_session is not None:
            return
        try:
            from mcp.client.streamable_http import streamablehttp_client
            from mcp.client.session import ClientSession
        except Exception as exc:  # noqa: BLE001
            raise MCPGatewayError(f"MCP streamable HTTP requires mcp package: {exc}") from exc

        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._http_session is not None:
                return
            self._http_cm = streamablehttp_client(self.base_url)
            read, write, _ = await self._http_cm.__aenter__()
            self._http_session_cm = ClientSession(read, write)
            self._http_session = await self._http_session_cm.__aenter__()
            await self._http_session.initialize()

    async def list_tools(self) -> list[ToolSpec]:
        if self.transport == MCPGatewayTransport.STDIO:
            await self._ensure_stdio_session()
            result = await self._session.list_tools()
        else:
            await self._ensure_http_session()
            result = await self._http_session.list_tools()

        tools = [ToolSpec(name=tool.name, description=tool.description, input_schema=tool.inputSchema) for tool in result.tools]
        self._tool_registry = {tool.name: tool for tool in tools}
        return tools

    def get_tool(self, name: str) -> ToolSpec | None:
        return self._tool_registry.get(name)

    async def _delay_if_needed(self) -> None:
        if self.faults.latency_ms <= 0 and self.faults.jitter_ms <= 0:
            return
        jitter = random.randint(0, max(self.faults.jitter_ms, 0))
        delay_ms = max(self.faults.latency_ms, 0) + jitter
        await asyncio.sleep(delay_ms / 1000.0)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        await self._delay_if_needed()

        if not self._tool_registry:
            await self.list_tools()

        tool = self._tool_registry.get(name)
        if tool is None:
            raise MCPGatewayError(f"Unknown tool: {name}")
        _validate_args(tool.input_schema, arguments)

        async def _call() -> dict[str, Any]:
            if self.transport == MCPGatewayTransport.STDIO:
                await self._ensure_stdio_session()
                result = await self._session.call_tool(name, arguments=arguments)
            else:
                await self._ensure_http_session()
                result = await self._http_session.call_tool(name, arguments=arguments)
            return result.model_dump()

        if self.faults.timeout_s:
            try:
                return await asyncio.wait_for(_call(), timeout=self.faults.timeout_s)
            except asyncio.TimeoutError as exc:
                raise MCPGatewayError("Tool call timed out") from exc
        return await _call()

    async def close(self) -> None:
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except RuntimeError:
                pass
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except RuntimeError:
                pass
        if self._http_session_cm is not None:
            try:
                await self._http_session_cm.__aexit__(None, None, None)
            except RuntimeError:
                pass
        if self._http_cm is not None:
            try:
                await self._http_cm.__aexit__(None, None, None)
            except RuntimeError:
                pass


def stdio_gateway_from_env() -> tuple[str, list[str], dict[str, str] | None]:
    command = os.getenv("MCP_GATEWAY_COMMAND", "docker")
    args_env = os.getenv("MCP_GATEWAY_ARGS", "mcp gateway run")
    args = shlex.split(args_env) if args_env else ["mcp", "gateway", "run"]
    env: dict[str, str] = {}
    allowed_paths = os.getenv("MCP_GATEWAY_ALLOWED_PATHS") or os.getenv("MCP_ALLOWED_PATHS")
    if allowed_paths:
        env["ALLOWED_PATHS"] = allowed_paths
    return command, args, env or None


def _validate_args(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    try:
        import jsonschema
    except Exception as exc:  # noqa: BLE001
        raise MCPGatewayError(f"jsonschema is required for tool validation: {exc}") from exc
    try:
        jsonschema.validate(arguments, schema)
    except jsonschema.ValidationError as exc:
        raise MCPGatewayError(f"Tool arguments failed validation: {exc.message}") from exc
