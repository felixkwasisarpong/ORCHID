"""CrewAI-based orchestrator implementation."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from observability.trace_schema import RunCounters, RunTrace, StepResult
from orchestrators.episode import SYSTEM_PROMPT, build_messages, parse_step_action
from orchestrators.types import EpisodeConfig
from runtimes.base import RuntimeConfig
from tools.mcp_gateway_client import MCPGatewayClient
from workers.crew.agents import CrewRunner
from workers.crew.crew_config import AgentConfig, CrewConfig, TaskConfig
from workers.crew.llm import LLMConfig, LLMRuntime


def _to_llm_config(runtime: RuntimeConfig) -> LLMConfig:
    runtime_map = {
        "ollama": LLMRuntime.OLLAMA,
        "openai": LLMRuntime.OPENAI,
        "anthropic": LLMRuntime.ANTHROPIC,
    }
    return LLMConfig(
        runtime=runtime_map[runtime.runtime.value],
        model=runtime.model,
        base_url=runtime.base_url,
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        seed=runtime.seed,
    )


def _crew_config() -> CrewConfig:
    agents = [
        AgentConfig(
            name="Planner",
            role="Planner",
            goal="Select the next tool call or finalize based on the task context.",
            backstory="Expert at structured tool planning.",
        ),
        AgentConfig(
            name="Verifier",
            role="Verifier",
            goal="Ensure the output is valid JSON that matches the schema.",
            backstory="Focused on producing valid structured responses.",
        ),
    ]
    tasks = [
        TaskConfig(
            name="Decide",
            description="Decide the next step and output ONLY a JSON object matching the StepAction schema.",
            expected_output="Valid StepAction JSON.",
        ),
        TaskConfig(
            name="Verify",
            description="Check the JSON for schema correctness and re-output ONLY valid StepAction JSON.",
            expected_output="Valid StepAction JSON.",
        ),
    ]
    return CrewConfig(agents=agents, tasks=tasks, process="sequential")


class CrewAIEngine:
    def __init__(
        self,
        runtime: RuntimeConfig,
        mcp_client: MCPGatewayClient,
        episode: EpisodeConfig,
    ) -> None:
        self.runtime = runtime
        self.mcp_client = mcp_client
        self.episode = episode
        self.crew_runner = CrewRunner(_crew_config(), llm_config=_to_llm_config(runtime))

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
            prompt = SYSTEM_PROMPT + "\n" + messages[0]["content"]

            llm_retries = 0
            action = None
            llm_latency = None
            last_output = ""
            while llm_retries <= self.episode.max_llm_retries:
                llm_start = time.perf_counter()
                crew_result = await self.crew_runner.run(prompt)
                llm_latency = (time.perf_counter() - llm_start) * 1000.0
                counters.llm_calls += 1
                last_output = crew_result.output
                try:
                    action = parse_step_action(crew_result.output)
                    break
                except Exception as exc:  # noqa: BLE001
                    llm_retries += 1
                    counters.retries += 1
                    if llm_retries > self.episode.max_llm_retries:
                        errors.append(f"LLM output invalid: {exc}")
                        action = None
                        break
                    prompt = (
                        SYSTEM_PROMPT
                        + "\n"
                        + messages[0]["content"]
                        + "\nPrevious output:\n"
                        + last_output
                        + f"\nError: {exc}\nReturn ONLY valid JSON."
                    )

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
            orchestrator="crewai",
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
        return trace
