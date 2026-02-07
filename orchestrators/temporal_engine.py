"""Temporal-based orchestrator implementation."""
from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from temporalio import activity, client, workflow

from benchmarks.tasks import all_tasks
from observability.trace_schema import RunCounters, RunTrace, StepResult
from orchestrators.episode import SYSTEM_PROMPT, build_messages, build_repair_messages, parse_step_action
from orchestrators.types import EpisodeConfig
from runtimes import build_client
from runtimes.base import RuntimeConfig
from tools.mcp_gateway_client import (
    MCPGatewayClient,
    MCPGatewayTransport,
    FaultSettings,
    ToolSpec,
    stdio_gateway_from_env,
)


class TemporalSettings(BaseModel):
    address: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "orchid-task-queue"
    workflow_timeout_s: int = 600
    activity_timeout_s: int = 120


class MCPSettings(BaseModel):
    transport: MCPGatewayTransport = MCPGatewayTransport.STDIO
    base_url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    reuse_session: bool = True


class FaultSettingsModel(BaseModel):
    latency_ms: int = 0
    jitter_ms: int = 0
    timeout_s: float | None = None


class WorkflowInput(BaseModel):
    run_id: str
    task_id: str
    description: str
    sandbox_path: str
    allowed_tool_names: list[str]
    runtime: RuntimeConfig
    episode: EpisodeConfig
    mcp: MCPSettings
    faults: FaultSettingsModel
    seed: int | None = None
    activity_timeout_s: float = 120.0


class LLMActivityInput(BaseModel):
    runtime: RuntimeConfig
    messages: list[dict[str, str]]
    system_prompt: str


class ToolCallInput(BaseModel):
    mcp: MCPSettings
    faults: FaultSettingsModel
    name: str
    arguments: dict[str, Any]


class ValidationInput(BaseModel):
    task_id: str
    sandbox_path: str


class ToolListInput(BaseModel):
    mcp: MCPSettings
    faults: FaultSettingsModel


@activity.defn
async def list_tools_activity(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = ToolListInput.model_validate(payload)
    command = data.mcp.command
    args = data.mcp.args
    env = None
    if data.mcp.transport == MCPGatewayTransport.STDIO and not command:
        command, args, env = stdio_gateway_from_env()
    client = MCPGatewayClient(
        transport=data.mcp.transport,
        base_url=data.mcp.base_url,
        command=command,
        args=args,
        env=env,
        reuse_session=data.mcp.reuse_session,
        faults=FaultSettings(
            latency_ms=data.faults.latency_ms,
            jitter_ms=data.faults.jitter_ms,
            timeout_s=data.faults.timeout_s,
        ),
    )
    tools = await client.list_tools()
    await client.close()
    return [tool.model_dump(mode="json") for tool in tools]


@activity.defn
async def llm_decide_activity(payload: dict[str, Any]) -> dict[str, Any]:
    data = LLMActivityInput.model_validate(payload)
    client = build_client(data.runtime)
    start = time.perf_counter()
    response = await client.complete(data.messages, system_prompt=data.system_prompt)
    latency_ms = (time.perf_counter() - start) * 1000.0
    await client.close()
    return {"content": response.content, "usage": response.usage, "model": response.model, "latency_ms": latency_ms}


@activity.defn
async def tool_call_activity(payload: dict[str, Any]) -> dict[str, Any]:
    data = ToolCallInput.model_validate(payload)
    command = data.mcp.command
    args = data.mcp.args
    env = None
    if data.mcp.transport == MCPGatewayTransport.STDIO and not command:
        command, args, env = stdio_gateway_from_env()
    client = MCPGatewayClient(
        transport=data.mcp.transport,
        base_url=data.mcp.base_url,
        command=command,
        args=args,
        env=env,
        reuse_session=data.mcp.reuse_session,
        faults=FaultSettings(
            latency_ms=data.faults.latency_ms,
            jitter_ms=data.faults.jitter_ms,
            timeout_s=data.faults.timeout_s,
        ),
    )
    start = time.perf_counter()
    output = await client.call_tool(data.name, data.arguments)
    latency_ms = (time.perf_counter() - start) * 1000.0
    await client.close()
    return {"output": output, "latency_ms": latency_ms}


@activity.defn
async def validate_activity(payload: dict[str, Any]) -> dict[str, Any]:
    data = ValidationInput.model_validate(payload)
    tasks = {task.spec.task_id: task for task in all_tasks()}
    task = tasks.get(data.task_id)
    if not task:
        return {"success": False, "message": "Unknown task"}
    start = time.perf_counter()
    result = task.validate(Path(data.sandbox_path))
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"result": result, "latency_ms": latency_ms}


@workflow.defn
class OrchidWorkflow:
    @workflow.run
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = WorkflowInput.model_validate(payload)
        counters = RunCounters()
        steps: list[StepResult] = []
        errors: list[str] = []
        success = False
        start_time = workflow.now().isoformat()

        tools_payload = ToolListInput(mcp=data.mcp, faults=data.faults).model_dump(mode="json")
        tools_raw = await workflow.execute_activity(
            list_tools_activity,
            tools_payload,
            start_to_close_timeout=timedelta(seconds=data.activity_timeout_s),
        )
        tools = [ToolSpec.model_validate(tool) for tool in tools_raw]
        allowed_tools = (
            tools if not data.allowed_tool_names else [tool for tool in tools if tool.name in data.allowed_tool_names]
        )
        if data.allowed_tool_names:
            missing = sorted(set(data.allowed_tool_names) - {tool.name for tool in allowed_tools})
            if missing:
                errors.append(f"Missing required tools: {missing}")

        for step_index in range(data.episode.max_steps):
            messages = build_messages(
                data.description,
                data.sandbox_path,
                allowed_tools,
                steps,
                data.episode.max_steps,
            )
            llm_retries = 0
            action = None
            llm_latency_total = 0.0
            while llm_retries <= data.episode.max_llm_retries:
                llm_payload = LLMActivityInput(
                    runtime=data.runtime,
                    messages=messages,
                    system_prompt=SYSTEM_PROMPT,
                ).model_dump(mode="json")
                llm_result = await workflow.execute_activity(
                    llm_decide_activity,
                    llm_payload,
                    start_to_close_timeout=timedelta(seconds=data.activity_timeout_s),
                )
                llm_latency_total += float(llm_result.get("latency_ms", 0.0))
                counters.llm_calls += 1
                try:
                    action = parse_step_action(llm_result["content"])
                    break
                except Exception as exc:  # noqa: BLE001
                    llm_retries += 1
                    counters.retries += 1
                    if llm_retries > data.episode.max_llm_retries:
                        errors.append(f"LLM output invalid: {exc}")
                        action = None
                        break
                    messages = build_repair_messages(messages, llm_result["content"], str(exc))

            if action is None:
                break

            step_result = StepResult(step_index=step_index, action=action, llm_latency_ms=llm_latency_total)

            if action.type == "tool_call":
                tool_latency_total = 0.0
                tool_errors: list[str] = []
                for attempt in range(data.episode.max_tool_retries + 1):
                    try:
                        tool_payload = ToolCallInput(
                            mcp=data.mcp,
                            faults=data.faults,
                            name=action.tool_call.name,
                            arguments=action.tool_call.arguments,
                        ).model_dump(mode="json")
                        tool_result = await workflow.execute_activity(
                            tool_call_activity,
                            tool_payload,
                            start_to_close_timeout=timedelta(seconds=data.activity_timeout_s),
                        )
                        tool_latency_total += float(tool_result.get("latency_ms", 0.0))
                        counters.tool_calls += 1
                        step_result.tool_result = tool_result.get("output")
                        break
                    except Exception as exc:  # noqa: BLE001
                        tool_errors.append(str(exc))
                        counters.tool_calls += 1
                        counters.retries += 1
                        if attempt >= data.episode.max_tool_retries:
                            errors.append(f"Tool call failed: {exc}")
                if tool_errors:
                    step_result.errors.extend(tool_errors)
                step_result.tool_latency_ms = tool_latency_total

            validation_payload = ValidationInput(
                task_id=data.task_id,
                sandbox_path=data.sandbox_path,
            ).model_dump(mode="json")
            validation_result = await workflow.execute_activity(
                validate_activity,
                validation_payload,
                start_to_close_timeout=timedelta(seconds=data.activity_timeout_s),
            )
            validation = validation_result.get("result")
            validation_latency = float(validation_result.get("latency_ms", 0.0))
            step_result.validation = validation
            step_result.validation_latency_ms = validation_latency
            if validation and validation.get("success") is True:
                success = True

            step_result.total_latency_ms = (
                (step_result.llm_latency_ms or 0.0)
                + (step_result.tool_latency_ms or 0.0)
                + (step_result.validation_latency_ms or 0.0)
            )
            steps.append(step_result)

            if success or action.type == "finalize":
                break

        counters.total_latency_ms = sum(step.total_latency_ms or 0.0 for step in steps)

        trace = RunTrace(
            run_id=data.run_id,
            orchestrator="temporal",
            runtime=data.runtime.runtime.value,
            model=data.runtime.model,
            task_id=data.task_id,
            seed=data.seed,
            steps=steps,
            errors=errors,
            success=success,
            counters=counters,
            metadata={"sandbox": data.sandbox_path},
            start_time=start_time,
        )
        trace.end_time = workflow.now().isoformat()
        return trace.model_dump(mode="json")


class TemporalEngine:
    def __init__(
        self,
        runtime: RuntimeConfig,
        episode: EpisodeConfig,
        settings: TemporalSettings,
        mcp: MCPSettings,
        faults: FaultSettingsModel,
    ) -> None:
        self.runtime = runtime
        self.episode = episode
        self.settings = settings
        self.mcp = mcp
        self.faults = faults

    async def run(
        self,
        run_id: str,
        task_id: str,
        description: str,
        sandbox: Path,
        allowed_tool_names: list[str],
        seed: int | None = None,
    ) -> RunTrace:
        client_handle = await client.Client.connect(self.settings.address, namespace=self.settings.namespace)
        payload = WorkflowInput(
            run_id=run_id,
            task_id=task_id,
            description=description,
            sandbox_path=str(sandbox),
            allowed_tool_names=allowed_tool_names,
            runtime=self.runtime,
            episode=self.episode,
            mcp=self.mcp,
            faults=self.faults,
            seed=seed,
            activity_timeout_s=self.settings.activity_timeout_s,
        ).model_dump(mode="json")

        result = await client_handle.execute_workflow(
            OrchidWorkflow.run,
            payload,
            id=run_id,
            task_queue=self.settings.task_queue,
            execution_timeout=timedelta(seconds=self.settings.workflow_timeout_s),
        )
        return RunTrace.model_validate(result)
