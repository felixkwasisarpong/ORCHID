"""Pydantic trace schemas for experiment runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Annotated

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StepActionToolCall(BaseModel):
    type: Literal["tool_call"]
    tool_call: ToolCall


class StepActionFinalize(BaseModel):
    type: Literal["finalize"]
    answer: str


StepAction = Annotated[
    StepActionToolCall | StepActionFinalize,
    Field(discriminator="type"),
]


class StepResult(BaseModel):
    step_index: int
    action: StepAction
    llm_latency_ms: float | None = None
    tool_latency_ms: float | None = None
    validation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    tool_result: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class RunCounters(BaseModel):
    llm_calls: int = 0
    tool_calls: int = 0
    retries: int = 0
    total_latency_ms: float = 0.0


class TaskSpec(BaseModel):
    task_id: str
    description: str
    max_steps: int = 6
    allowed_tools: list[str] = Field(default_factory=list)
    success_criteria: str | None = None


class RunTrace(BaseModel):
    run_id: str
    orchestrator: str
    runtime: str
    model: str | None = None
    task_id: str
    seed: int | None = None
    start_time: str = Field(default_factory=utc_now_iso)
    end_time: str | None = None
    success: bool = False
    errors: list[str] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    counters: RunCounters = Field(default_factory=RunCounters)
    metadata: dict[str, Any] = Field(default_factory=dict)
