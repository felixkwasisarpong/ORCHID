"""Base class for graph nodes."""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from observability.logger import get_logger, log_event
from orchestrator.policies import FailurePolicy, RetryPolicy
from orchestrator.state import GlobalState, TraceEvent, now_utc


class BaseNode(ABC):
    """Base node providing retries, timeouts, and trace emission."""

    def __init__(
        self,
        name: str,
        retry_policy: RetryPolicy | None = None,
        failure_policy: FailurePolicy | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.name = name
        self.retry_policy = retry_policy or RetryPolicy()
        self.failure_policy = failure_policy or FailurePolicy()
        self.timeout_s = timeout_s
        self.logger = get_logger("orchid")

    @abstractmethod
    async def run(self, state: GlobalState) -> GlobalState:
        """Implement node logic: input state -> output state."""

    async def execute(self, state: GlobalState) -> GlobalState:
        """Execute with retry/backoff and tracing."""
        attempts = self.retry_policy.max_attempts
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            start_wall = now_utc()
            start = time.perf_counter()
            try:
                log_event(
                    self.logger,
                    "node_start",
                    request_id=state.request_id,
                    node=self.name,
                    attempt=attempt,
                )
                if self.timeout_s is None:
                    result = await self.run(state)
                else:
                    result = await asyncio.wait_for(self.run(state), timeout=self.timeout_s)
                end_wall = now_utc()
                duration_ms = (time.perf_counter() - start) * 1000.0
                trace = TraceEvent(
                    node=self.name,
                    status="success",
                    attempt=attempt,
                    start_time=start_wall,
                    end_time=end_wall,
                    duration_ms=duration_ms,
                )
                result = result.add_trace(trace)
                log_event(
                    self.logger,
                    "trace",
                    request_id=result.request_id,
                    trace_event=trace.model_dump(mode="json"),
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if isinstance(exc, asyncio.TimeoutError):
                    error_text = "timeout"
                else:
                    error_text = str(exc)
                end_wall = now_utc()
                duration_ms = (time.perf_counter() - start) * 1000.0
                trace = TraceEvent(
                    node=self.name,
                    status="error",
                    attempt=attempt,
                    start_time=start_wall,
                    end_time=end_wall,
                    duration_ms=duration_ms,
                    error=error_text,
                )
                state = state.add_trace(trace)
                log_event(
                    self.logger,
                    "trace",
                    request_id=state.request_id,
                    trace_event=trace.model_dump(mode="json"),
                )
                log_event(
                    self.logger,
                    "node_error",
                    request_id=state.request_id,
                    node=self.name,
                    attempt=attempt,
                    error=error_text,
                )
                if not self.retry_policy.should_retry(exc) or attempt >= attempts:
                    break
                await asyncio.sleep(self.retry_policy.backoff_s(attempt))
        error_message = f"{self.name} failed: {last_error}" if last_error else f"{self.name} failed"
        state = state.add_error(error_message)
        return state
