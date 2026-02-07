"""Shared prompt, parsing, and tool-call logic for orchestrators."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from observability.trace_schema import StepAction, StepResult, TaskSpec, ToolRegistry
from runtimes.base import RuntimeClient
from tools.mcp_gateway_client import MCPGatewayClient


@dataclass
class EpisodeConfig:
    max_steps: int
    max_llm_retries: int
    max_tool_retries: int
    timeout_s: float
    tool_timeout_s: float


def build_messages(
    task: TaskSpec,
    tool_registry: ToolRegistry,
    history: List[StepResult],
    sandbox_root: Path,
) -> List[Dict[str, str]]:
    tool_lines = []
    for tool in tool_registry.tools:
        tool_lines.append(
            f"- {tool.name}: {tool.description or ''} schema={json.dumps(tool.input_schema)}"
        )
    tools_block = "\n".join(tool_lines)

    history_lines = []
    for step in history[-5:]:
        summary = {
            "step": step.step_index,
            "action": step.action.model_dump(),
            "validated": step.validated,
            "error": step.error,
        }
        if step.tool_result is not None:
            summary["tool_result"] = step.tool_result
        history_lines.append(json.dumps(summary))
    history_block = "\n".join(history_lines) if history_lines else "(no prior steps)"

    system_prompt = (
        "You are a research agent executing filesystem tasks via tools. "
        "Return ONLY valid JSON matching the StepAction schema."
    )

    user_prompt = f"""
Task ID: {task.id}
Task: {task.description}
Sandbox root: {sandbox_root}
Allowed tools:
{tools_block}

Step history (most recent last):
{history_block}

StepAction schema:
{{
  "action_type": "tool_call" | "finalize",
  "tool_call": {{"name": "tool_name", "arguments": {{...}}}},
  "final_answer": "..."
}}

Rules:
- If more tool work is needed, choose action_type=tool_call.
- If you believe the goal is satisfied, choose action_type=finalize and explain briefly in final_answer.
- Use only allowed tool names.
- Before write operations, ensure parent directories exist.
- Unless explicitly requested, write plain text outputs (not JSON wrappers).
- Output JSON only, no extra text.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def call_llm_for_action(
    runtime: RuntimeClient,
    messages: List[Dict[str, str]],
    max_retries: int,
    seed: Optional[int] = None,
) -> Tuple[StepAction, int, float, int]:
    llm_calls = 0
    retries = 0
    total_latency_ms = 0.0
    last_error: Optional[Exception] = None
    current_messages = list(messages)

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        raw = await runtime.chat(current_messages, seed=seed)
        t1 = time.perf_counter()
        llm_calls += 1
        total_latency_ms += (t1 - t0) * 1000
        try:
            payload = json.loads(raw)
            action = StepAction.model_validate(payload)
            return action, llm_calls, total_latency_ms, retries
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            retries += 1
            repair_prompt = (
                "The JSON was invalid or did not match schema. "
                f"Error: {exc}. Return ONLY corrected JSON. Original output: {raw}"
            )
            current_messages = current_messages + [{"role": "user", "content": repair_prompt}]

    raise RuntimeError(f"LLM failed to produce valid StepAction JSON: {last_error}")


async def call_tool_with_retries(
    client: MCPGatewayClient,
    name: str,
    arguments: Dict[str, Any],
    max_retries: int,
    timeout_s: float,
) -> Tuple[Dict[str, Any], int, float, int]:
    tool_calls = 0
    retries = 0
    total_latency_ms = 0.0
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(client.call_tool(name, arguments), timeout=timeout_s)
            t1 = time.perf_counter()
            tool_calls += 1
            total_latency_ms += (t1 - t0) * 1000
            tool_error = extract_tool_error(result)
            if tool_error:
                raise RuntimeError(tool_error)
            return result, tool_calls, total_latency_ms, retries
        except Exception as exc:  # noqa: BLE001
            t1 = time.perf_counter()
            tool_calls += 1
            total_latency_ms += (t1 - t0) * 1000
            last_error = exc
            if attempt >= max_retries:
                break
            retries += 1

    raise RuntimeError(f"Tool call failed after retries: {last_error}")


def extract_tool_error(result: Dict[str, Any]) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    if not result.get("isError"):
        return None
    content = result.get("content")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        if text_parts:
            return " | ".join(text_parts)
    return "Tool returned isError=true"


async def run_step_loop(
    task: TaskSpec,
    sandbox_root: Path,
    runtime: RuntimeClient,
    tool_client: MCPGatewayClient,
    validator: Callable[[Path], Tuple[bool, str]],
    episode_config: EpisodeConfig,
    seed: int,
) -> Tuple[List[StepResult], int, int, int]:
    history: List[StepResult] = []
    llm_calls = 0
    tool_calls = 0
    retries = 0

    tools = await tool_client.list_tools()
    allowed_tools = [tool for tool in tools.tools if tool.name in task.allowed_tools]
    tool_registry = ToolRegistry(tools=allowed_tools)

    for step_index in range(episode_config.max_steps):
        step_start = time.perf_counter()
        messages = build_messages(task, tool_registry, history, sandbox_root)
        action, llm_inc, llm_latency_ms, llm_retries = await call_llm_for_action(
            runtime,
            messages,
            episode_config.max_llm_retries,
            seed=seed,
        )
        llm_calls += llm_inc
        retries += llm_retries

        tool_latency_ms = 0.0
        tool_result = None
        error = None
        tool_retries = 0

        if action.action_type == "tool_call":
            if action.tool_call is None:
                error = "tool_call missing"
            elif action.tool_call.name not in task.allowed_tools:
                error = f"Tool {action.tool_call.name} not allowed"
            else:
                try:
                    result, tool_inc, tool_latency_ms, tool_retries = await call_tool_with_retries(
                        tool_client,
                        action.tool_call.name,
                        action.tool_call.arguments,
                        episode_config.max_tool_retries,
                        episode_config.tool_timeout_s,
                    )
                    tool_calls += tool_inc
                    retries += tool_retries
                    tool_result = result
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)

        validated, validation_error = validator(sandbox_root)
        step_end = time.perf_counter()

        step_result = StepResult(
            step_index=step_index,
            action=action,
            tool_result=tool_result,
            validated=validated,
            validation_error=validation_error if not validated else None,
            llm_latency_ms=llm_latency_ms,
            tool_latency_ms=tool_latency_ms,
            step_latency_ms=(step_end - step_start) * 1000,
            error=error,
            retries=llm_retries + tool_retries,
        )
        history.append(step_result)

        if validated:
            break
        if action.action_type == "finalize":
            break
        if error:
            break

    return history, llm_calls, tool_calls, retries
