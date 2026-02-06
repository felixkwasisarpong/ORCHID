"""Tracing helpers for timing and trace export."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from orchestrator.state import TraceEvent


@dataclass
class TraceTimer:
    """Context manager for wall-clock timings."""

    start: float | None = None
    end: float | None = None

    def __enter__(self) -> "TraceTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return (self.end - self.start) * 1000.0


def export_trace_json(trace: Iterable[TraceEvent]) -> list[dict[str, object]]:
    """Convert trace events to JSON-serializable dicts."""
    return [event.model_dump() for event in trace]
