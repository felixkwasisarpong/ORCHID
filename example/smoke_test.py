"""Smoke test for MCP stdio client."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mcp_stdio_client import MCPStdioClient, MCPStdioConfig


def main() -> None:
    config = MCPStdioConfig(
        gateway_cmd=[
            "docker",
            "mcp",
            "gateway",
            "run",
        ]
    )
    client = MCPStdioClient(config)

    async def run() -> None:
        await client.start()
        await client.warmup(2)
        tools = await client.list_tools()
        print("Tools:", tools)
        try:
            result = await client.call_tool("list_directory", {"path": "."})
            print("list_directory result:", result)
        except Exception as exc:  # noqa: BLE001
            print("list_directory failed:", exc)
        await client.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
