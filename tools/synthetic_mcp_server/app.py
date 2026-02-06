"""Synthetic MCP server with fault injection."""
from __future__ import annotations

import asyncio
import random
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response

from tools.mcp_client import FaultConfig, FaultMode, MCPRequest, MCPResponse


app = FastAPI(title="Synthetic MCP Server", version="0.1.0")


def choose_mode(config: FaultConfig) -> FaultMode:
    if config.mode != FaultMode.RANDOM:
        return config.mode
    weights = config.random_weights or {
        "success": 0.7,
        "delay": 0.15,
        "timeout": 0.05,
        "malformed": 0.05,
        "rate_limit": 0.05,
    }
    items = list(weights.items())
    modes = [FaultMode(key) for key, _ in items]
    probs = [value for _, value in items]
    return random.choices(modes, weights=probs, k=1)[0]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tool", response_model=None)
async def tool_call(request: MCPRequest) -> Response:
    fault = request.fault or FaultConfig()
    mode = choose_mode(fault)

    if mode == FaultMode.TIMEOUT:
        delay_s = max(fault.delay_ms / 1000.0, 6.0)
        await asyncio.sleep(delay_s)
        raise HTTPException(status_code=504, detail="timeout")

    if mode == FaultMode.RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate_limit")

    if mode == FaultMode.MALFORMED:
        return PlainTextResponse(content="<<malformed>>", status_code=200)

    if mode == FaultMode.DELAY:
        delay_s = max(fault.delay_ms / 1000.0, 1.0)
        await asyncio.sleep(delay_s)

    output: dict[str, Any] = {
        "tool": request.tool,
        "echo": request.input,
        "request_id": request.request_id,
    }
    return MCPResponse(status="ok", output=output, metadata={"mode": mode.value})
