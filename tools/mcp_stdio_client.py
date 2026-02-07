"""Minimal MCP stdio client adapter for Docker MCP Gateway."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class MCPToolCallTrace(BaseModel):
    timestamp: str
    tool_name: str
    request_id: int
    attempt: int


class MCPToolResultTrace(BaseModel):
    timestamp: str
    tool_name: str
    latency_ms: float
    success: bool
    error_type: Optional[str] = None
    request_id: int
    attempt: int


@dataclass
class MCPStdioConfig:
    request_timeout_s: float = 10.0
    max_retries: int = 1
    retry_backoff_s: float = 0.1
    restart_on_timeout: bool = True
    protocol_version: str = "2024-11-05"
    client_name: str = "orchid-mcp"
    client_version: str = "0.1"
    gateway_cmd: Optional[List[str]] = None

    def resolved_gateway_cmd(self) -> List[str]:
        if self.gateway_cmd:
            return self.gateway_cmd
        return ["docker", "mcp", "gateway", "run"]


class MCPStdioClient:
    """Async MCP stdio client with retries, timeouts, and auto-restart."""

    def __init__(self, config: MCPStdioConfig, log_path: str | Path = "logs/mcp_calls.jsonl") -> None:
        self.config = config
        self.log_path = Path(log_path)
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._read_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._id_counter = 0
        self._write_lock = asyncio.Lock()
        self._restart_lock = asyncio.Lock()
        self._tool_cache: Optional[Dict[str, Any]] = None
        self._closed = False

    async def start(self) -> None:
        self._closed = False
        await self._ensure_proc()

    async def stop(self) -> None:
        self._closed = True
        await self._shutdown_proc()

    async def warmup(self, count: int = 3) -> None:
        for _ in range(max(0, count)):
            try:
                await self.list_tools(force=True)
            except Exception:  # noqa: BLE001
                await asyncio.sleep(self.config.retry_backoff_s)

    async def list_tools(self, force: bool = False) -> Dict[str, Any]:
        if self._tool_cache is not None and not force:
            return self._tool_cache
        result = await self._request_with_retry(["tools/list", "tools.list"], {}, "tools.list")
        self._tool_cache = result
        return result

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request_with_retry(
            ["tools/call", "tools.call"],
            {"name": name, "arguments": arguments},
            name,
        )

    async def _request_with_retry(
        self,
        methods: List[str],
        params: Dict[str, Any],
        tool_name: str,
        allow_restart: bool = True,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 2):
            for method in methods:
                start = time.perf_counter()
                request_id: Optional[int] = None
                try:
                    result, request_id = await self._request(method, params, tool_name, attempt)
                    latency_ms = (time.perf_counter() - start) * 1000
                    self._log_result(tool_name, latency_ms, True, None, request_id, attempt)
                    return result
                except Exception as exc:  # noqa: BLE001
                    latency_ms = (time.perf_counter() - start) * 1000
                    self._log_result(tool_name, latency_ms, False, type(exc).__name__, request_id, attempt)
                    last_error = exc
                    if allow_restart and self._should_restart(exc):
                        await self._restart_proc()
                    if method != methods[-1]:
                        continue
                    if attempt <= self.config.max_retries:
                        await asyncio.sleep(self.config.retry_backoff_s)
        raise RuntimeError(f"MCP request failed after retries: {last_error}")

    async def _request(
        self, method: str, params: Dict[str, Any], tool_name: str, attempt: int
    ) -> Tuple[Dict[str, Any], int]:
        await self._ensure_proc()
        assert self._proc and self._proc.stdin

        async with self._write_lock:
            request_id = self._next_id()
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            future = asyncio.get_event_loop().create_future()
            self._pending[request_id] = future
            self._log_call(tool_name, request_id, attempt)
            try:
                self._proc.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
                await self._proc.stdin.drain()
            except Exception as exc:  # noqa: BLE001
                self._pending.pop(request_id, None)
                raise exc

        try:
            response = await asyncio.wait_for(future, timeout=self.config.request_timeout_s)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise exc

        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result", {}), request_id

    async def _ensure_proc(self) -> None:
        if self._proc and self._proc.returncode is None:
            return
        await self._start_proc()

    async def _start_proc(self) -> None:
        cmd = self.config.resolved_gateway_cmd()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._read_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._request_with_retry(
            ["initialize"],
            {"protocolVersion": self.config.protocol_version, "clientInfo": {"name": self.config.client_name, "version": self.config.client_version}},
            "initialize",
            allow_restart=False,
        )

    async def _shutdown_proc(self) -> None:
        if self._read_task:
            self._read_task.cancel()
            self._read_task = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None
        self._mark_pending_error("client stopped")

    async def _restart_proc(self) -> None:
        if self._closed:
            return
        async with self._restart_lock:
            await self._shutdown_proc()
            await self._start_proc()

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
            request_id = payload.get("id")
            if request_id is None:
                continue
            future = self._pending.pop(request_id, None)
            if future and not future.done():
                future.set_result(payload)

        self._mark_pending_error("subprocess terminated")
        self._proc = None

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break

    def _mark_pending_error(self, message: str) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(RuntimeError(message))
        self._pending.clear()

    def _should_restart(self, exc: Exception) -> bool:
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return True
        if isinstance(exc, asyncio.TimeoutError):
            return self.config.restart_on_timeout
        if self._proc is None:
            return True
        if self._proc.returncode is not None:
            return True
        return False

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _log_call(self, tool_name: str, request_id: int, attempt: int) -> None:
        trace = MCPToolCallTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            request_id=request_id,
            attempt=attempt,
        )
        self._write_log(trace.model_dump())

    def _log_result(
        self,
        tool_name: str,
        latency_ms: float,
        success: bool,
        error_type: Optional[str],
        request_id: Optional[int],
        attempt: int,
    ) -> None:
        trace = MCPToolResultTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            request_id=request_id or -1,
            attempt=attempt,
        )
        self._write_log(trace.model_dump())

    def _write_log(self, payload: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
