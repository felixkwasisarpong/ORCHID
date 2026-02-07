"""LangGraph-style orchestrator implementation."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from observability.trace_schema import RunCounters, RunTrace, StepResult
from orchestrators.episode import (
    SYSTEM_PROMPT,
    build_messages,
    build_repair_messages,
    parse_step_action,
)
from orchestrators.types import EpisodeConfig
from runtimes import build_client
from runtimes.base import RuntimeConfig
from tools.mcp_gateway_client import MCPGatewayClient


class LangGraphEngine:
    def __init__(
        self,
        runtime: RuntimeConfig,
        mcp_client: MCPGatewayClient,
        episode: EpisodeConfig,
    ) -> None:
        self.runtime = runtime
        self.mcp_client = mcp_client
        self.episode = episode

    async def run(
        self,
        run_id: str,
        task_id: str,
        description: str,
        sandbox: Path,
        validate: Callable[[Path], dict[str, Any]],
        allowed_tool_names: list[str],
        seed: int | None = None,
    ) -> RunTrace:
        client = build_client(self.runtime)
        counters = RunCounters()
        steps: list[StepResult] = []
        errors: list[str] = []
        success = False
        start = time.perf_counter()
        tools = await self.mcp_client.list_tools()
        allowed_tools = [tool for tool in tools if tool.name in allowed_tool_names] if allowed_tool_names else tools
        if allowed_tool_names:
            missing = sorted(set(allowed_tool_names) - {tool.name for tool in allowed_tools})
            if missing:
                errors.append(f"Missing required tools: {missing}")

        for step_index in range(self.episode.max_steps):
            step_start = time.perf_counter()
            messages = build_messages(
                description,
                str(sandbox),
                allowed_tools,
                steps,
                self.episode.max_steps,
            )
            llm_retries = 0
            action = None
            llm_latency = None
            while llm_retries <= self.episode.max_llm_retries:
                llm_start = time.perf_counter()
                response = await client.complete(messages, system_prompt=SYSTEM_PROMPT)
                llm_latency = (time.perf_counter() - llm_start) * 1000.0
                counters.llm_calls += 1
                counters.total_latency_ms += llm_latency
                try:
                    action = parse_step_action(response.content)
                    break
                except Exception as exc:  # noqa: BLE001
                    llm_retries += 1
                    counters.retries += 1
                    if llm_retries > self.episode.max_llm_retries:
                        errors.append(f"LLM output invalid: {exc}")
                        action = None
                        break
                    messages = build_repair_messages(messages, response.content, str(exc))

            if action is None:
                break

            step_result = StepResult(step_index=step_index, action=action, llm_latency_ms=llm_latency)

            if action.type == "tool_call":
                tool_latency = None
                tool_errors: list[str] = []
                for attempt in range(self.episode.max_tool_retries + 1):
                    try:
                        tool_start = time.perf_counter()
                        tool_output = await self.mcp_client.call_tool(
                            action.tool_call.name,
                            action.tool_call.arguments,
                        )
                        tool_latency = (time.perf_counter() - tool_start) * 1000.0
                        counters.tool_calls += 1
                        step_result.tool_result = tool_output
                        break
                    except Exception as exc:  # noqa: BLE001
                        tool_errors.append(str(exc))
                        counters.tool_calls += 1
                        counters.retries += 1
                        if attempt >= self.episode.max_tool_retries:
                            errors.append(f"Tool call failed: {exc}")
                if tool_errors:
                    step_result.errors.extend(tool_errors)
                step_result.tool_latency_ms = tool_latency

            validation_latency = None
            try:
                validation_start = time.perf_counter()
                validation = validate(sandbox)
                validation_latency = (time.perf_counter() - validation_start) * 1000.0
                step_result.validation = validation
                if validation.get("success") is True:
                    success = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Validation failed: {exc}")

            step_result.validation_latency_ms = validation_latency
            step_result.total_latency_ms = (time.perf_counter() - step_start) * 1000.0
            steps.append(step_result)

            if success or action.type == "finalize":
                break

        total_latency = (time.perf_counter() - start) * 1000.0
        counters.total_latency_ms = total_latency

        trace = RunTrace(
            run_id=run_id,
            orchestrator="langgraph",
            runtime=self.runtime.runtime.value,
            model=self.runtime.model,
            task_id=task_id,
            seed=seed,
            steps=steps,
            errors=errors,
            success=success,
            counters=counters,
            metadata={"sandbox": str(sandbox)},
        )
        trace.end_time = datetime.now(tz=timezone.utc).isoformat()
        await client.close()
        return trace
