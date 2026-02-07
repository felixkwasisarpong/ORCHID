"""CrewAI-based orchestrator (multi-agent roles)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Tuple

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
    from crewai import Agent, Crew, Process, Task
except Exception:  # noqa: BLE001
    Agent = None
    Crew = None
    Process = None
    Task = None


def _crewai_llm_model(runtime_name: str, runtime: RuntimeClient) -> str:
    model = runtime.config.model
    if runtime_name == "ollama":
        return f"ollama/{model}"
    if runtime_name == "anthropic":
        return f"anthropic/{model}"
    return model


def _extract_crew_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    for attr in ("raw", "output", "result"):
        value = getattr(output, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return str(output)


@dataclass
class CrewAIEngine:
    runtime: RuntimeClient
    tool_client: MCPGatewayClient
    validator: Callable[[Path], Tuple[bool, str]]
    episode_config: EpisodeConfig

    async def run(
        self, task: TaskSpec, sandbox_root: Path, seed: int, run_id: str, runtime_name: str
    ) -> RunTrace:
        if Agent is None:
            raise RuntimeError("crewai is not installed. Install crewai to use this orchestrator.")
        os.environ["CREWAI_TRACING_ENABLED"] = "false"
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["CREWAI_TESTING"] = "true"
        os.environ["LITELLM_LOG"] = "CRITICAL"
        if runtime_name == "ollama":
            os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # CrewAI routes through LiteLLM; silence proxy-related logger noise in harness runs.
        try:
            import litellm
            from litellm.litellm_core_utils import litellm_logging as litellm_logging_mod

            litellm.suppress_debug_info = True
            litellm.verbose_logger.disabled = True
            litellm.verbose_logger.setLevel(logging.CRITICAL)
            litellm_logging_mod.verbose_logger.disabled = True
            litellm_logging_mod.verbose_logger.setLevel(logging.CRITICAL)
        except Exception:  # noqa: BLE001
            pass

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

            llm_model = _crewai_llm_model(runtime_name, self.runtime)
            planner = Agent(
                role="Planner",
                goal="Propose the next StepAction JSON.",
                backstory="Expert at planning tool usage.",
                llm=llm_model,
                verbose=False,
                max_iter=1,
                max_retry_limit=0,
                allow_delegation=False,
            )
            critic = Agent(
                role="Critic",
                goal="Validate and correct the StepAction JSON.",
                backstory="Ensures strict schema compliance.",
                llm=llm_model,
                verbose=False,
                max_iter=1,
                max_retry_limit=0,
                allow_delegation=False,
            )

            plan_task = Task(
                description=f"Use the context to propose a StepAction JSON.\n\n{prompt_text}",
                expected_output="Valid StepAction JSON only.",
                agent=planner,
            )
            critique_task = Task(
                description=(
                    "Review the planner's output and return a corrected StepAction JSON only. "
                    f"\n\nContext:\n{prompt_text}"
                ),
                expected_output="Valid StepAction JSON only.",
                agent=critic,
            )

            crew = Crew(
                agents=[planner, critic],
                tasks=[plan_task, critique_task],
                process=Process.sequential,
                verbose=False,
            )

            llm_start = time.perf_counter()
            crew_output = await asyncio.to_thread(crew.kickoff)
            llm_end = time.perf_counter()
            llm_calls += 2
            llm_latency_ms = (llm_end - llm_start) * 1000
            llm_retries = 0

            tool_latency_ms = 0.0
            tool_result = None
            error = None
            tool_retries = 0

            try:
                # Attempt to parse crew output first; fallback to repair loop if invalid.
                try:
                    action = StepAction.model_validate(json.loads(_extract_crew_output_text(crew_output)))
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
                action = None

            if action and action.action_type == "tool_call":
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

            if action is None:
                break

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
            orchestrator="crewai",
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
