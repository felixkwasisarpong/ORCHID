"""Typed global state for orchestration runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """Single node execution event for tracing/replay."""

    node: str
    status: Literal["success", "error"]
    attempt: int
    start_time: datetime
    end_time: datetime
    duration_ms: float
    error: str | None = None


class GlobalState(BaseModel):
    """Global orchestration state passed between nodes."""

    request_id: str
    user_input: str

    plan: str | None = None
    crew_output: str | None = None
    tool_result: dict[str, Any] | None = None
    validation: str | None = None
    final_output: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    trace: list[TraceEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def add_trace(self, event: TraceEvent) -> "GlobalState":
        return self.model_copy(update={"trace": [*self.trace, event]})

    def add_error(self, error: str) -> "GlobalState":
        return self.model_copy(update={"errors": [*self.errors, error]})


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
