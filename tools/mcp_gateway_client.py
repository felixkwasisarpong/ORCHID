"""MCP Gateway client supporting stdio (default) and HTTP streaming."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from observability.trace_schema import MCPToolSpec, ToolRegistry


@dataclass
class MCPClientConfig:
    transport: str = "stdio"  # "stdio" or "http"
    gateway_cmd: list[str] = None
    http_url: Optional[str] = None
    request_timeout_s: float = 10.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    path_rewrite_from: Optional[str] = None
    path_rewrite_to: Optional[str] = None

    def resolved_gateway_cmd(self) -> list[str]:
        if self.gateway_cmd:
            return self.gateway_cmd
        return ["docker", "mcp", "gateway", "run"]


class MCPGatewayClient:
    def __init__(self, config: MCPClientConfig) -> None:
        self.config = config
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._id_counter = 0
        self._tool_registry: Optional[ToolRegistry] = None

    async def __aenter__(self) -> "MCPGatewayClient":
        if self.config.transport == "stdio":
            await self._start_stdio()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None

    async def list_tools(self) -> ToolRegistry:
        if self._tool_registry is not None:
            return self._tool_registry
        try:
            result = await self._request("tools/list", {})
        except Exception:  # noqa: BLE001
            result = await self._request("tools.list", {})
        tools = []
        for item in result.get("tools", []):
            tools.append(
                MCPToolSpec(
                    name=item.get("name", ""),
                    description=item.get("description"),
                    input_schema=item.get("inputSchema", {}) or {},
                )
            )
        self._tool_registry = ToolRegistry(tools=tools)
        return self._tool_registry

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        await self._apply_latency()
        rewritten_arguments = self._rewrite_arguments(arguments)
        try:
            return await self._request("tools/call", {"name": name, "arguments": rewritten_arguments})
        except Exception:  # noqa: BLE001
            return await self._request("tools.call", {"name": name, "arguments": rewritten_arguments})

    async def _apply_latency(self) -> None:
        if self.config.latency_ms <= 0 and self.config.jitter_ms <= 0:
            return
        jitter = 0.0
        if self.config.jitter_ms > 0:
            jitter = random.uniform(-self.config.jitter_ms, self.config.jitter_ms)
        delay = max(0.0, (self.config.latency_ms + jitter) / 1000.0)
        if delay > 0:
            await asyncio.sleep(delay)

    async def _start_stdio(self) -> None:
        if self._proc:
            return
        cmd = self.config.resolved_gateway_cmd()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "orchid", "version": "0.1"},
                "capabilities": {},
            },
        )

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if "id" in payload:
                future = self._pending.pop(payload["id"], None)
                if future and not future.done():
                    future.set_result(payload)

    async def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.transport == "http":
            return await self._request_http(method, params)
        if not self._proc:
            await self._start_stdio()
        request_id = self._next_id()
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        data = json.dumps(payload).encode("utf-8") + b"\n"
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        try:
            response = await asyncio.wait_for(future, timeout=self.config.request_timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result", {})

    async def _request_http(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.http_url:
            raise RuntimeError("HTTP transport selected but http_url is not configured")
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        async with httpx.AsyncClient(timeout=self.config.request_timeout_s) as client:
            response = await client.post(self.config.http_url, json=payload)
            response.raise_for_status()
            data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data.get("result", {})

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _rewrite_arguments(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._rewrite_arguments(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [self._rewrite_arguments(item) for item in value]
        if not isinstance(value, str):
            return value

        source_root = self.config.path_rewrite_from
        target_root = self.config.path_rewrite_to
        if not source_root or not target_root:
            return value

        source_root = source_root.rstrip("/")
        if value == source_root:
            return target_root
        if value.startswith(f"{source_root}/"):
            return f"{target_root}{value[len(source_root):]}"
        return value
