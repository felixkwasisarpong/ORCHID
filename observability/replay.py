"""Replay orchestration traces from JSON log files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from orchestrator.state import TraceEvent


def load_trace_from_logs(path: str | Path, request_id: str) -> list[TraceEvent]:
    """Load trace events for a request ID from a JSONL log file."""
    events: list[TraceEvent] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event_type") != "trace" or payload.get("request_id") != request_id:
                continue
            data = payload.get("trace_event")
            if isinstance(data, dict):
                events.append(TraceEvent(**data))
    return events


def replay_summary(trace: Iterable[TraceEvent]) -> dict[str, object]:
    """Compute a simple summary for visualization hooks."""
    durations = [event.duration_ms for event in trace]
    return {
        "nodes": [event.node for event in trace],
        "total_duration_ms": sum(durations),
        "events": len(durations),
    }
