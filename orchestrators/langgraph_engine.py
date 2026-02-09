"""LangGraph-based orchestrator (state machine)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Tuple, TypedDict

from observability.trace_schema import RunTrace, StepResult, TaskSpec, ToolRegistry
from orchestrators.common import (
    EpisodeConfig,
    build_messages,
    call_llm_for_action,
    call_tool_with_retries,
)
from runtimes.base import RuntimeClient
from tools.mcp_gateway_client import MCPGatewayClient

try:
    from langgraph.graph import END, StateGraph
except Exception:  # noqa: BLE001
    StateGraph = None
    END = None


class LGState(TypedDict):
    history: List[StepResult]
    step_index: int
    llm_calls: int
    tool_calls: int
    retries: int
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_total_tokens: int
    llm_cost_usd: float
    done: bool


@dataclass
class LangGraphEngine:
    runtime: RuntimeClient
    tool_client: MCPGatewayClient
    validator: Callable[[Path], Tuple[bool, str]]
    episode_config: EpisodeConfig

    async def run(
        self, task: TaskSpec, sandbox_root: Path, seed: int, run_id: str, runtime_name: str
    ) -> RunTrace:
        if StateGraph is None:
            raise RuntimeError("langgraph is not installed. Install langgraph to use this orchestrator.")

        started_at = datetime.now(timezone.utc).isoformat()
        start_ts = time.perf_counter()
        tools = await self.tool_client.list_tools()
        allowed_tools = [tool for tool in tools.tools if tool.name in task.allowed_tools]
        tool_registry = ToolRegistry(tools=allowed_tools)

        async def step_node(state: LGState) -> LGState:
            step_index = state["step_index"]
            step_start = time.perf_counter()
            messages = build_messages(task, tool_registry, state["history"], sandbox_root)
            action, llm_metrics = await call_llm_for_action(
                self.runtime,
                messages,
                self.episode_config.max_llm_retries,
                seed=seed,
            )
            state["llm_calls"] += llm_metrics.llm_calls
            state["retries"] += llm_metrics.retries
            state["llm_prompt_tokens"] += llm_metrics.usage.prompt_tokens
            state["llm_completion_tokens"] += llm_metrics.usage.completion_tokens
            state["llm_total_tokens"] += llm_metrics.usage.total_tokens
            state["llm_cost_usd"] += llm_metrics.cost_usd

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
                            self.tool_client,
                            action.tool_call.name,
                            action.tool_call.arguments,
                            self.episode_config.max_tool_retries,
                            self.episode_config.tool_timeout_s,
                        )
                        state["tool_calls"] += tool_inc
                        state["retries"] += tool_retries
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
                llm_latency_ms=llm_metrics.latency_ms,
                llm_prompt_tokens=llm_metrics.usage.prompt_tokens,
                llm_completion_tokens=llm_metrics.usage.completion_tokens,
                llm_total_tokens=llm_metrics.usage.total_tokens,
                llm_cost_usd=llm_metrics.cost_usd,
                tool_latency_ms=tool_latency_ms,
                step_latency_ms=(step_end - step_start) * 1000,
                error=error,
                retries=llm_metrics.retries + tool_retries,
            )
            state["history"].append(step_result)
            state["step_index"] += 1

            if validated or error or action.action_type == "finalize":
                state["done"] = True
            if state["step_index"] >= self.episode_config.max_steps:
                state["done"] = True
            return state

        def route(state: LGState) -> str:
            return END if state["done"] else "step"

        graph = StateGraph(LGState)
        graph.add_node("step", step_node)
        graph.set_entry_point("step")
        graph.add_conditional_edges("step", route, {"step": "step", END: END})

        app = graph.compile()
        init_state: LGState = {
            "history": [],
            "step_index": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "retries": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "llm_cost_usd": 0.0,
            "done": False,
        }
        final_state = await app.ainvoke(init_state)
        end_ts = time.perf_counter()
        ended_at = datetime.now(timezone.utc).isoformat()

        success = False
        error_msg = None
        if final_state["history"]:
            success = final_state["history"][-1].validated
            if final_state["history"][-1].error:
                error_msg = final_state["history"][-1].error

        return RunTrace(
            run_id=run_id,
            orchestrator="langgraph",
            runtime=runtime_name,
            task_id=task.id,
            seed=seed,
            started_at=started_at,
            ended_at=ended_at,
            total_latency_ms=(end_ts - start_ts) * 1000,
            llm_calls=final_state["llm_calls"],
            tool_calls=final_state["tool_calls"],
            retries=final_state["retries"],
            llm_prompt_tokens=final_state["llm_prompt_tokens"],
            llm_completion_tokens=final_state["llm_completion_tokens"],
            llm_total_tokens=final_state["llm_total_tokens"],
            llm_cost_usd=final_state["llm_cost_usd"],
            steps=final_state["history"],
            success=success,
            error=error_msg,
            fault_config={},
        )
