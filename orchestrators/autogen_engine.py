"""AutoGen-based orchestrator (multi-agent conversation)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Tuple

from observability.trace_schema import RunTrace, StepAction, StepResult, TaskSpec, ToolRegistry
from orchestrators.common import (
    EpisodeConfig,
    build_messages,
    call_llm_for_action,
    call_tool_with_retries,
)
from runtimes.base import RuntimeClient
from tools.mcp_gateway_client import MCPGatewayClient

try:
    from autogen_agentchat.messages import TextMessage
except Exception:  # noqa: BLE001
    TextMessage = None

try:
    import autogen as autogen_legacy
except Exception:  # noqa: BLE001
    autogen_legacy = None


class AutoGenMetrics:
    def __init__(self) -> None:
        self.llm_calls = 0
        self.total_latency_ms = 0.0


class RuntimeConversableAgent:  # lightweight wrapper around AutoGen if available
    def __init__(self, name: str, runtime: RuntimeClient, seed: int, metrics: AutoGenMetrics) -> None:
        self.name = name
        self.runtime = runtime
        self.seed = seed
        self.metrics = metrics
        self.transcript: list[object] = []

    async def generate(self, prompt: str) -> str:
        if TextMessage is not None:
            self.transcript.append(TextMessage(content=prompt, source="user"))
        start = time.perf_counter()
        result = await self.runtime.chat([{"role": "user", "content": prompt}], seed=self.seed)
        end = time.perf_counter()
        self.metrics.llm_calls += 1
        self.metrics.total_latency_ms += (end - start) * 1000
        if TextMessage is not None:
            self.transcript.append(TextMessage(content=result, source=self.name))
        return result


@dataclass
class AutoGenEngine:
    runtime: RuntimeClient
    tool_client: MCPGatewayClient
    validator: Callable[[Path], Tuple[bool, str]]
    episode_config: EpisodeConfig

    async def run(
        self, task: TaskSpec, sandbox_root: Path, seed: int, run_id: str, runtime_name: str
    ) -> RunTrace:
        if TextMessage is None and autogen_legacy is None:
            raise RuntimeError(
                "AutoGen is not installed. Install pyautogen/autogen-agentchat to use this orchestrator."
            )

        started_at = datetime.now(timezone.utc).isoformat()
        start_ts = time.perf_counter()
        tools = await self.tool_client.list_tools()
        allowed_tools = [tool for tool in tools.tools if tool.name in task.allowed_tools]
        tool_registry = ToolRegistry(tools=allowed_tools)

        history: List[StepResult] = []
        llm_calls = 0
        tool_calls = 0
        retries = 0

        for step_index in range(self.episode_config.max_steps):
            step_start = time.perf_counter()
            messages = build_messages(task, tool_registry, history, sandbox_root)
            prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

            metrics = AutoGenMetrics()
            planner = RuntimeConversableAgent("planner", self.runtime, seed, metrics)
            critic = RuntimeConversableAgent("critic", self.runtime, seed, metrics)

            planner_output = await planner.generate(prompt_text)
            critic_prompt = f"{prompt_text}\n\nPlanner output:\n{planner_output}\n\nReturn corrected StepAction JSON only."
            critic_output = await critic.generate(critic_prompt)

            llm_calls += metrics.llm_calls
            llm_latency_ms = metrics.total_latency_ms
            llm_retries = 0

            tool_latency_ms = 0.0
            tool_result = None
            error = None
            tool_retries = 0

            try:
                try:
                    action = StepAction.model_validate(json.loads(critic_output))
                except Exception:  # noqa: BLE001
                    action, llm_inc, llm_latency_ms2, llm_retries = await call_llm_for_action(
                        self.runtime,
                        messages,
                        self.episode_config.max_llm_retries,
                        seed=seed,
                    )
                    llm_calls += llm_inc
                    llm_latency_ms += llm_latency_ms2
                    retries += llm_retries
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                break

            if action.action_type == "tool_call":
                if action.tool_call is None:
                    error = "tool_call missing"
                elif action.tool_call.name not in task.allowed_tools:
                    error = f"Tool {action.tool_call.name} not allowed"
                else:
                    try:
                        result, tool_inc, tool_latency_ms, tool_retries = await call_tool_with_retries(
                            self.tool_client,
                            action.tool_call.name,
                            action.tool_call.arguments,
                            self.episode_config.max_tool_retries,
                            self.episode_config.tool_timeout_s,
                        )
                        tool_calls += tool_inc
                        retries += tool_retries
                        tool_result = result
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc)

            validated, validation_error = self.validator(sandbox_root)
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

            if validated or error or action.action_type == "finalize":
                break

        end_ts = time.perf_counter()
        ended_at = datetime.now(timezone.utc).isoformat()
        success = False
        error_msg = None
        if history:
            success = history[-1].validated
            if history[-1].error:
                error_msg = history[-1].error

        return RunTrace(
            run_id=run_id,
            orchestrator="autogen",
            runtime=runtime_name,
            task_id=task.id,
            seed=seed,
            started_at=started_at,
            ended_at=ended_at,
            total_latency_ms=(end_ts - start_ts) * 1000,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            retries=retries,
            steps=history,
            success=success,
            error=error_msg,
            fault_config={},
        )
