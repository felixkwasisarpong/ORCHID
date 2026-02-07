"""Pydantic models for structured traces and action schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class StepAction(BaseModel):
    action_type: Literal["tool_call", "finalize"]
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None

    @model_validator(mode="after")
    def _validate_action(self) -> "StepAction":
        if self.action_type == "tool_call":
            if self.tool_call is None:
                raise ValueError("tool_call must be provided when action_type=tool_call")
        if self.action_type == "finalize":
            if not self.final_answer:
                raise ValueError("final_answer must be provided when action_type=finalize")
        return self


class StepResult(BaseModel):
    step_index: int
    action: StepAction
    tool_result: Optional[Any] = None
    validated: bool = False
    validation_error: Optional[str] = None
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    step_latency_ms: float = 0.0
    error: Optional[str] = None
    retries: int = 0


class TaskSpec(BaseModel):
    id: str
    name: str
    description: str
    allowed_tools: List[str]
    max_steps: int = 8
    max_llm_retries: int = 2
    max_tool_retries: int = 1
    timeout_s: float = 20.0


class RunTrace(BaseModel):
    run_id: str
    orchestrator: str
    runtime: str
    task_id: str
    seed: int
    started_at: str
    ended_at: str
    total_latency_ms: float
    llm_calls: int
    tool_calls: int
    retries: int
    steps: List[StepResult]
    success: bool
    error: Optional[str] = None
    fault_config: Dict[str, Any] = Field(default_factory=dict)


class MCPToolSpec(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ToolRegistry(BaseModel):
    tools: List[MCPToolSpec]
